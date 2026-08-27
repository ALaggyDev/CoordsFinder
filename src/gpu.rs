//! `wgpu` compute backend for coordinate searches.
//!
//! The shader is specialized to the selected algorithm, filter length, Y
//! range, and mismatch tolerance. Each plan tile is dispatched separately so
//! result capacity and progress reporting remain bounded.

use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::config::ScanConfig;
use crate::scan::{ScanPlan, candidate_count, directional_filter};
use crate::types::{Match, RotationInfo};

const RESULT_CAPACITY: u32 = 262_144;
const WORKGROUP_XZ: u32 = 16;
const CANDIDATES_PER_THREAD_Y: u32 = 32;

#[derive(Clone, Copy, Eq, PartialEq)]
struct ShaderSpecialization {
    algorithm: crate::types::TextureAlgorithm,
    error_tolerance: i32,
    filter_count: usize,
    y_start: i32,
    y_end: i32,
}

impl From<&ScanConfig> for ShaderSpecialization {
    fn from(config: &ScanConfig) -> Self {
        Self {
            algorithm: config.algorithm,
            error_tolerance: config.error_tolerance,
            filter_count: config.filter.len(),
            y_start: config.y_range.start,
            y_end: config.y_range.end,
        }
    }
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct GpuFilter {
    // One 16-byte uniform record: xyz offsets, then rotation | visible_mask << 8.
    values: [i32; 4],
}

impl From<RotationInfo> for GpuFilter {
    fn from(value: RotationInfo) -> Self {
        Self {
            values: [
                i32::from(value.x),
                i32::from(value.y),
                i32::from(value.z),
                i32::from(value.rotation) | (i32::from(value.visible_mask) << 8),
            ],
        }
    }
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct GpuResult {
    x: i32,
    y: i32,
    z: i32,
    mismatches: i32,
    direction: i32,
}

impl From<GpuResult> for Match {
    fn from(value: GpuResult) -> Self {
        Self {
            x: value.x,
            y: value.y,
            z: value.z,
            mismatches: value.mismatches,
            direction: value.direction,
        }
    }
}

/// GPU scanner whose pipeline is specialized for the config passed to [`Self::new`].
pub struct GpuScanner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: Box<wgpu::ComputePipeline>,
    bind_group: wgpu::BindGroup,
    params: wgpu::Buffer,
    filters: wgpu::Buffer,
    results: wgpu::Buffer,
    counters: wgpu::Buffer,
    specialization: ShaderSpecialization,
    adapter_name: String,
    adapter_backend: wgpu::Backend,
    max_workgroups_per_dimension: u32,
}

impl GpuScanner {
    /// Initializes a compute device and compiles a pipeline for `config`.
    pub fn new(config: &ScanConfig) -> Result<Self, String> {
        pollster::block_on(Self::new_async(config))
    }

    async fn new_async(config: &ScanConfig) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|error| format!("could not find a wgpu adapter: {error}"))?;
        let adapter_info = adapter.get_info();
        if !adapter.features().contains(wgpu::Features::SHADER_INT64) {
            return Err(format!(
                "wgpu adapter '{}' does not support 64-bit shader integers",
                adapter_info.name
            ));
        }
        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("CoordsFinder device"),
                required_features: wgpu::Features::SHADER_INT64,
                required_limits: wgpu::Limits::default().using_resolution(limits.clone()),
                ..Default::default()
            })
            .await
            .map_err(|error| format!("could not create wgpu device: {error}"))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("CoordsFinder search shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("search.wgsl").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("CoordsFinder search bind group layout"),
            entries: &[
                uniform_layout_entry(0),
                uniform_layout_entry(1),
                storage_layout_entry(2, false),
                storage_layout_entry(3, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("CoordsFinder search pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        // A process scans one config, so compile only the exact kernel it needs.
        let specialization = ShaderSpecialization::from(config);
        let y_span = i64::from(specialization.y_end) - i64::from(specialization.y_start);
        let constants = [
            ("TEXTURE_ALGORITHM", specialization.algorithm as u32 as f64),
            ("ERROR_TOLERANCE", specialization.error_tolerance as f64),
            ("FILTER_COUNT", specialization.filter_count as f64),
            ("Y_START", specialization.y_start as f64),
            ("Y_SPAN", y_span as f64),
        ];
        let pipeline = Box::new(
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("CoordsFinder config-specialized search pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("search"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
                cache: None,
            }),
        );

        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("search parameters"),
            size: 5 * 4,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let filters = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("texture filters"),
            size: 256 * size_of::<GpuFilter>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let result_bytes = u64::from(RESULT_CAPACITY) * size_of::<GpuResult>() as u64;
        let results = storage_buffer(&device, "search results", result_bytes, true);
        let counters = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("search counters"),
            contents: bytemuck::bytes_of(&[0_u32; 2]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CoordsFinder search bindings"),
            layout: &bind_group_layout,
            entries: &[
                binding(0, &params),
                binding(1, &filters),
                binding(2, &results),
                binding(3, &counters),
            ],
        });
        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group,
            params,
            filters,
            results,
            counters,
            specialization,
            adapter_name: adapter_info.name,
            adapter_backend: adapter_info.backend,
            max_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
        })
    }

    /// Returns the human-readable name reported by the selected adapter.
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// Returns the graphics API used by the selected adapter.
    pub fn adapter_backend(&self) -> wgpu::Backend {
        self.adapter_backend
    }

    /// Executes a plan one tile at a time and reports matches and progress.
    ///
    /// The config must have the specialization-sensitive values used to create
    /// this scanner; changing those values requires constructing a new scanner.
    pub fn scan(
        &self,
        config: &ScanConfig,
        plan: &ScanPlan<'_>,
        mut sink: impl FnMut(&[Match]),
        mut progress: impl FnMut(u64, usize),
        cancelled: impl Fn() -> bool,
    ) -> Result<(), String> {
        if self.specialization != ShaderSpecialization::from(config) {
            return Err("GPU scanner was used with a different shader configuration".to_owned());
        }
        let filters: Vec<Vec<GpuFilter>> = config
            .directions
            .iter()
            .map(|&direction| {
                directional_filter(&config.filter, direction)
                    .into_iter()
                    .map(GpuFilter::from)
                    .collect()
            })
            .collect();

        let mut candidates = 0_u64;
        for (index, item) in plan.iter().enumerate() {
            if cancelled() {
                break;
            }
            let x_span = (i64::from(item.end.x) - i64::from(item.start.x)) as u32;
            let y_span = (i64::from(item.end.y) - i64::from(item.start.y)) as u32;
            let z_span = (i64::from(item.end.z) - i64::from(item.start.z)) as u32;
            let workgroups = [
                x_span.div_ceil(WORKGROUP_XZ),
                y_span.div_ceil(CANDIDATES_PER_THREAD_Y),
                z_span.div_ceil(WORKGROUP_XZ),
            ];
            if workgroups
                .iter()
                .any(|&count| count > self.max_workgroups_per_dimension)
            {
                return Err(
                    "gpuTileSize or Y range exceeds this adapter's dispatch limits".to_owned(),
                );
            }
            let params = [
                item.start.x as u32,
                item.start.z as u32,
                x_span,
                z_span,
                item.direction as u32,
            ];
            self.queue
                .write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
            self.queue.write_buffer(
                &self.filters,
                0,
                bytemuck::cast_slice(&filters[item.direction_index]),
            );
            self.queue
                .write_buffer(&self.counters, 0, bytemuck::bytes_of(&[0_u32; 2]));

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("CoordsFinder search commands"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("CoordsFinder search pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
            }
            self.queue.submit([encoder.finish()]);

            // Reading the counters also waits for this tile's dispatch. Results
            // are downloaded only when the shader reports at least one match.
            let counter_bytes = read_buffer(&self.device, &self.queue, &self.counters, 8)?;
            let counters: &[u32] = bytemuck::cast_slice(&counter_bytes);
            if counters[1] != 0 {
                return Err(format!(
                    "a GPU tile produced more than {RESULT_CAPACITY} matches; reduce gpuTileSize"
                ));
            }
            if counters[0] > 0 {
                let result_bytes = u64::from(counters[0]) * size_of::<GpuResult>() as u64;
                let bytes = read_buffer(&self.device, &self.queue, &self.results, result_bytes)?;
                let matches: Vec<Match> = bytemuck::cast_slice::<u8, GpuResult>(&bytes)
                    .iter()
                    .copied()
                    .map(Match::from)
                    .collect();
                sink(&matches);
            }
            candidates = candidates.saturating_add(candidate_count(&item).0);
            progress(candidates, index + 1);
        }
        Ok(())
    }
}

fn storage_buffer(
    device: &wgpu::Device,
    label: &'static str,
    size: u64,
    writable: bool,
) -> wgpu::Buffer {
    let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    if writable {
        usage |= wgpu::BufferUsages::COPY_SRC;
    }
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

fn binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn storage_layout_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn read_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    size: u64,
) -> Result<Vec<u8>, String> {
    let (sender, receiver) = mpsc::sync_channel(1);
    wgpu::util::DownloadBuffer::read_buffer(device, queue, &buffer.slice(..size), move |result| {
        let result = result
            .map(|download| download.to_vec())
            .map_err(|error| format!("could not read GPU results: {error}"));
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|error| format!("GPU wait failed: {error}"))?;
    receiver
        .recv()
        .map_err(|_| "GPU result callback did not complete".to_owned())?
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::config::{IntRange, ScanOrder, TileSize};
    use crate::scan::make_plan;
    use crate::texture::get_texture;
    use crate::types::TextureAlgorithm;

    #[test]
    fn search_shader_is_valid_wgsl() {
        let module = wgpu::naga::front::wgsl::parse_str(include_str!("search.wgsl")).unwrap();
        let mut validator = wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::SHADER_INT64,
        );
        validator.validate(&module).unwrap();
    }

    #[test]
    fn gpu_matches_all_reference_algorithms_when_available() {
        for algorithm in [
            TextureAlgorithm::Vanilla1,
            TextureAlgorithm::Vanilla2,
            TextureAlgorithm::Vanilla3,
            TextureAlgorithm::Sodium1,
            TextureAlgorithm::Sodium2,
        ] {
            let mut config = ScanConfig {
                algorithm,
                scan_order: ScanOrder::Linear,
                directions: vec![0],
                x_range: IntRange { start: 0, end: 1 },
                y_range: IntRange { start: 0, end: 1 },
                z_range: IntRange { start: 0, end: 1 },
                gpu_tile_size: TileSize { x: 1, z: 1 },
                filter: vec![RotationInfo::new(0, 0, 0, 0, false)],
                ..ScanConfig::default()
            };
            let Ok(scanner) = GpuScanner::new(&config) else {
                eprintln!("skipping GPU test: no compatible adapter");
                return;
            };
            for coordinate in [(0, 0, 0), (4096, 0, 4096), (-1, 0, -3)] {
                config.x_range = IntRange {
                    start: coordinate.0,
                    end: coordinate.0 + 1,
                };
                config.z_range = IntRange {
                    start: coordinate.2,
                    end: coordinate.2 + 1,
                };
                config.filter[0] = RotationInfo::new(
                    0,
                    0,
                    0,
                    get_texture(algorithm, coordinate.0, coordinate.1, coordinate.2, 4),
                    false,
                );
                let plan = make_plan(&config, config.gpu_tile_size).unwrap();
                let mut matches = Vec::new();
                scanner
                    .scan(
                        &config,
                        &plan,
                        |batch| matches.extend_from_slice(batch),
                        |_, _| {},
                        || false,
                    )
                    .unwrap();
                assert_eq!(
                    matches,
                    vec![Match {
                        x: coordinate.0,
                        y: coordinate.1,
                        z: coordinate.2,
                        mismatches: 0,
                        direction: 0
                    }],
                    "{algorithm}"
                );
            }
        }

        // A tolerance equal to the one-filter length makes every coordinate a
        // match, exposing any skipped or duplicated candidates in Y batching.
        let config = ScanConfig {
            algorithm: TextureAlgorithm::Vanilla3,
            scan_order: ScanOrder::Linear,
            directions: vec![0],
            x_range: IntRange { start: -4, end: 5 },
            y_range: IntRange {
                start: -17,
                end: 18,
            },
            z_range: IntRange { start: -3, end: 4 },
            error_tolerance: 1,
            gpu_tile_size: TileSize { x: 9, z: 7 },
            filter: vec![RotationInfo::new(0, 0, 0, 0, false)],
            ..ScanConfig::default()
        };
        let Ok(scanner) = GpuScanner::new(&config) else {
            eprintln!("skipping GPU test: no compatible adapter");
            return;
        };
        let plan = make_plan(&config, config.gpu_tile_size).unwrap();
        let mut matches = Vec::new();
        scanner
            .scan(
                &config,
                &plan,
                |batch| matches.extend_from_slice(batch),
                |_, _| {},
                || false,
            )
            .unwrap();
        let coordinates: HashSet<_> = matches
            .iter()
            .map(|found| (found.x, found.y, found.z))
            .collect();
        assert_eq!(matches.len(), 9 * 35 * 7);
        assert_eq!(coordinates.len(), matches.len());
        for x in -4..5 {
            for y in -17..18 {
                for z in -3..4 {
                    assert!(coordinates.contains(&(x, y, z)));
                }
            }
        }
    }
}
