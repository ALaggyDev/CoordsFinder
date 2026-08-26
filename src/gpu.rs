use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::config::ScanConfig;
use crate::scan::{ScanPlan, candidate_count, directional_filter};
use crate::types::{Match, RotationInfo};

const RESULT_CAPACITY: u32 = 262_144;
const WORKGROUP_XZ: u32 = 16;
const CANDIDATES_PER_THREAD_Y: u32 = 16;
const TEXTURE_ALGORITHM_COUNT: usize = 5;
const SEARCH_PIPELINE_COUNT: usize = TEXTURE_ALGORITHM_COUNT * 2;

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

pub struct GpuScanner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Box<[wgpu::ComputePipeline; SEARCH_PIPELINE_COUNT]>,
    bind_group: wgpu::BindGroup,
    params: wgpu::Buffer,
    filters: wgpu::Buffer,
    results: wgpu::Buffer,
    counters: wgpu::Buffer,
    adapter_name: String,
    max_workgroups_per_dimension: u32,
}

impl GpuScanner {
    pub fn new() -> Result<Self, String> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|error| format!("could not find a wgpu adapter: {error}"))?;
        if !adapter.features().contains(wgpu::Features::SHADER_INT64) {
            return Err(format!(
                "wgpu adapter '{}' does not support 64-bit shader integers",
                adapter.get_info().name
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
        // Specialization removes unused RNG paths and mismatch bookkeeping when
        // the configured tolerance is zero.
        let pipelines = Box::new(std::array::from_fn(|index| {
            let algorithm = index / 2;
            let zero_error_tolerance = index % 2;
            let constants = [
                ("TEXTURE_ALGORITHM", algorithm as f64),
                ("ZERO_ERROR_TOLERANCE", zero_error_tolerance as f64),
            ];
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("CoordsFinder specialized search pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("search"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
                cache: None,
            })
        }));

        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("search parameters"),
            size: 10 * 4,
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
            pipelines,
            bind_group,
            params,
            filters,
            results,
            counters,
            adapter_name: adapter.get_info().name,
            max_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
        })
    }

    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    pub fn scan(
        &self,
        config: &ScanConfig,
        plan: &ScanPlan,
        mut sink: impl FnMut(&[Match]),
        mut progress: impl FnMut(u64, usize),
        cancelled: impl Fn() -> bool,
    ) -> Result<(), String> {
        let pipeline_index =
            config.algorithm as usize * 2 + usize::from(config.error_tolerance == 0);
        let pipeline = &self.pipelines[pipeline_index];
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
        for (index, item) in plan.items.iter().enumerate() {
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
                item.start.y as u32,
                item.start.z as u32,
                x_span,
                y_span,
                z_span,
                config.error_tolerance as u32,
                config.filter.len() as u32,
                item.direction as u32,
                RESULT_CAPACITY,
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
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
            }
            self.queue.submit([encoder.finish()]);

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
            candidates = candidates.saturating_add(candidate_count(item).0);
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
        let Ok(scanner) = GpuScanner::new() else {
            eprintln!("skipping GPU test: no compatible adapter");
            return;
        };
        for (coordinate, algorithm) in [(17, -4, -31), (-1, -2, -3), (-29_999_984, -64, 29_999_983)]
            .into_iter()
            .flat_map(|coordinate| {
                [
                    TextureAlgorithm::Vanilla1,
                    TextureAlgorithm::Vanilla2,
                    TextureAlgorithm::Vanilla3,
                    TextureAlgorithm::Sodium1,
                    TextureAlgorithm::Sodium2,
                ]
                .map(|algorithm| (coordinate, algorithm))
            })
        {
            let config = ScanConfig {
                algorithm,
                scan_order: ScanOrder::Linear,
                directions: vec![0],
                x_range: IntRange {
                    start: coordinate.0,
                    end: coordinate.0 + 1,
                },
                y_range: IntRange {
                    start: coordinate.1,
                    end: coordinate.1 + 1,
                },
                z_range: IntRange {
                    start: coordinate.2,
                    end: coordinate.2 + 1,
                },
                gpu_tile_size: TileSize { x: 1, z: 1 },
                filter: vec![RotationInfo::new(
                    0,
                    0,
                    0,
                    get_texture(algorithm, coordinate.0, coordinate.1, coordinate.2, 4),
                    false,
                )],
                ..ScanConfig::default()
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
