//! `wgpu` compute backend for coordinate searches.
//!
//! The shader is specialized to the selected algorithm, filter length, Y
//! range, and mismatch tolerance. Each plan tile is dispatched separately so
//! result capacity and progress reporting remain bounded.

use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::config::ScanConfig;
use crate::filter::prepare_filters;
use crate::sample_plan::{
    MAX_SAMPLE_SIZE, SamplePlan, TriePlacement, anchor_grid, select_sample_plan,
};
use crate::scan::{ScanPlan, candidate_count};
use crate::types::{CompiledRotation, MAX_FILTER_COUNT, Match};

const RESULT_CAPACITY: u32 = 262_144;
const WORKGROUP_XZ: u32 = 16;
const SAMPLE_WORKGROUP_XZ: u32 = 8;
const SAMPLE_WORKGROUP_Y: u32 = 4;
const NAIVE_CANDIDATES_PER_THREAD_Y: u32 = 32;
const MAX_TRIE_NODES: usize = 1 + MAX_FILTER_COUNT * MAX_SAMPLE_SIZE;

#[derive(Clone, Copy, Eq, PartialEq)]
struct ShaderSpecialization {
    algorithm: crate::types::TextureAlgorithm,
    error_tolerance: i32,
    y_start: i32,
    y_end: i32,
}

impl ShaderSpecialization {
    fn new(config: &ScanConfig) -> Result<Self, String> {
        prepare_filters(
            &config.filter,
            config.algorithm,
            &config.directions,
            config.error_tolerance,
        )?;
        Ok(Self {
            algorithm: config.algorithm,
            error_tolerance: config.error_tolerance,
            y_start: config.y_range.start,
            y_end: config.y_range.end,
        })
    }
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct GpuFilter {
    // One 16-byte uniform record: xyz offsets, then the 16-way acceptance mask.
    values: [i32; 4],
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct GpuTrieNode {
    children: [u32; 4],
    // Output start and count, followed by alignment padding.
    metadata: [u32; 4],
}

impl From<crate::sample_plan::FlatTrieNode> for GpuTrieNode {
    fn from(value: crate::sample_plan::FlatTrieNode) -> Self {
        Self {
            children: value.children,
            metadata: [value.output_start, value.output_count, 0, 0],
        }
    }
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct GpuTriePlacement {
    // XYZ sample origin, then four packed u8 witness-filter indices.
    values: [i32; 4],
}

impl From<TriePlacement> for GpuTriePlacement {
    fn from(value: TriePlacement) -> Self {
        let packed_witnesses = value
            .witness_indices
            .into_iter()
            .enumerate()
            .fold(0_u32, |packed, (index, witness)| {
                packed | (u32::from(witness) << (index * 8))
            });
        Self {
            values: [
                value.origin.x,
                value.origin.y,
                value.origin.z,
                packed_witnesses as i32,
            ],
        }
    }
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct GpuSampleParams {
    anchor_start: [i32; 4],
    anchor_steps: [u32; 4],
    anchor_counts: [u32; 4],
    tile_start: [i32; 4],
    tile_end: [i32; 4],
    sample_offsets: [[i32; 4]; MAX_SAMPLE_SIZE],
    metadata: [u32; 4],
}

struct GpuSamplePlan {
    plan: SamplePlan,
    nodes: Vec<GpuTrieNode>,
    placements: Vec<GpuTriePlacement>,
}

impl GpuSamplePlan {
    fn new(plan: SamplePlan) -> Self {
        debug_assert!(plan.nodes.len() <= MAX_TRIE_NODES);
        debug_assert!(plan.placements.len() <= MAX_FILTER_COUNT);
        let nodes = plan.nodes.iter().copied().map(GpuTrieNode::from).collect();
        let placements = plan
            .placements
            .iter()
            .copied()
            .map(GpuTriePlacement::from)
            .collect();
        Self {
            plan,
            nodes,
            placements,
        }
    }
}

impl From<CompiledRotation> for GpuFilter {
    fn from(value: CompiledRotation) -> Self {
        Self {
            values: [
                i32::from(value.x),
                i32::from(value.y),
                i32::from(value.z),
                i32::from(value.accepted_indices),
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
    naive_pipeline: Box<wgpu::ComputePipeline>,
    sample_pipeline: Box<wgpu::ComputePipeline>,
    bind_group: wgpu::BindGroup,
    params: wgpu::Buffer,
    filters: wgpu::Buffer,
    results: wgpu::Buffer,
    counters: wgpu::Buffer,
    sample_params: wgpu::Buffer,
    trie_nodes: wgpu::Buffer,
    trie_placements: wgpu::Buffer,
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
                uniform_layout_entry(4),
                storage_layout_entry(5, true),
                storage_layout_entry(6, true),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("CoordsFinder search pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        // A process scans one config, so compile only the exact kernel it needs.
        let specialization = ShaderSpecialization::new(config)?;
        let y_span = i64::from(specialization.y_end) - i64::from(specialization.y_start);
        let constants = [
            ("TEXTURE_ALGORITHM", specialization.algorithm as u32 as f64),
            ("ERROR_TOLERANCE", specialization.error_tolerance as f64),
            ("Y_START", specialization.y_start as f64),
            ("Y_SPAN", y_span as f64),
        ];
        let naive_pipeline = Box::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("CoordsFinder config-specialized search pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("search"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
                cache: None,
            },
        ));
        let sample_pipeline = Box::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("CoordsFinder compact-template trie pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("search_sample_trie"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
                cache: None,
            },
        ));

        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("search parameters"),
            size: 7 * 4,
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
        let sample_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sample-trie parameters"),
            size: size_of::<GpuSampleParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let trie_nodes = storage_buffer(
            &device,
            "sample-trie nodes",
            (MAX_TRIE_NODES * size_of::<GpuTrieNode>()) as u64,
            false,
        );
        let trie_placements = storage_buffer(
            &device,
            "sample-trie placements",
            (MAX_FILTER_COUNT * size_of::<GpuTriePlacement>()) as u64,
            false,
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CoordsFinder search bindings"),
            layout: &bind_group_layout,
            entries: &[
                binding(0, &params),
                binding(1, &filters),
                binding(2, &results),
                binding(3, &counters),
                binding(4, &sample_params),
                binding(5, &trie_nodes),
                binding(6, &trie_placements),
            ],
        });
        Ok(Self {
            device,
            queue,
            naive_pipeline,
            sample_pipeline,
            bind_group,
            params,
            filters,
            results,
            counters,
            sample_params,
            trie_nodes,
            trie_placements,
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
        if self.specialization != ShaderSpecialization::new(config)? {
            return Err("GPU scanner was used with a different shader configuration".to_owned());
        }
        let prepared = prepare_filters(
            &config.filter,
            config.algorithm,
            &config.directions,
            config.error_tolerance,
        )?;
        let filters: Vec<Vec<GpuFilter>> = prepared
            .directions
            .iter()
            .map(|direction| {
                direction
                    .constraints
                    .iter()
                    .copied()
                    .map(GpuFilter::from)
                    .collect()
            })
            .collect();
        let sample_plans = prepared
            .directions
            .iter()
            .map(|direction| {
                select_sample_plan(
                    &direction.constraints,
                    config.algorithm,
                    direction.forced_errors,
                    config.error_tolerance,
                    config.search_mode,
                )
                .map(|plan| plan.map(GpuSamplePlan::new))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut candidates = 0_u64;
        for (index, item) in plan.iter().enumerate() {
            if cancelled() {
                break;
            }
            let direction_filter = &prepared.directions[item.direction_index];
            if direction_filter.forced_errors > config.error_tolerance {
                candidates = candidates.saturating_add(candidate_count(&item).0);
                progress(candidates, index + 1);
                continue;
            }
            let x_span = (i64::from(item.end.x) - i64::from(item.start.x)) as u32;
            let y_span = (i64::from(item.end.y) - i64::from(item.start.y)) as u32;
            let z_span = (i64::from(item.end.z) - i64::from(item.start.z)) as u32;
            let sample_dispatch = sample_plans[item.direction_index]
                .as_ref()
                .and_then(|sample| anchor_grid(&sample.plan, &item).map(|grid| (sample, grid)));
            let workgroups = sample_dispatch.as_ref().map_or_else(
                || {
                    [
                        x_span.div_ceil(WORKGROUP_XZ),
                        y_span.div_ceil(NAIVE_CANDIDATES_PER_THREAD_Y),
                        z_span.div_ceil(WORKGROUP_XZ),
                    ]
                },
                |(_, grid)| {
                    [
                        grid.counts[0].div_ceil(SAMPLE_WORKGROUP_XZ),
                        grid.counts[1].div_ceil(SAMPLE_WORKGROUP_Y),
                        grid.counts[2].div_ceil(SAMPLE_WORKGROUP_XZ),
                    ]
                },
            );
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
                direction_filter.forced_errors as u32,
                direction_filter.constraints.len() as u32,
            ];
            self.queue
                .write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
            self.queue.write_buffer(
                &self.filters,
                0,
                bytemuck::cast_slice(&filters[item.direction_index]),
            );
            if let Some((sample, grid)) = &sample_dispatch {
                let sample_params = gpu_sample_params(sample, *grid, &item);
                self.queue
                    .write_buffer(&self.sample_params, 0, bytemuck::bytes_of(&sample_params));
                self.queue
                    .write_buffer(&self.trie_nodes, 0, bytemuck::cast_slice(&sample.nodes));
                self.queue.write_buffer(
                    &self.trie_placements,
                    0,
                    bytemuck::cast_slice(&sample.placements),
                );
            }
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
                pass.set_pipeline(if sample_dispatch.is_some() {
                    &self.sample_pipeline
                } else {
                    &self.naive_pipeline
                });
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
                    "a GPU tile produced more than {RESULT_CAPACITY} matches; add more filters"
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

fn gpu_sample_params(
    sample: &GpuSamplePlan,
    grid: crate::sample_plan::AnchorGrid,
    item: &crate::scan::WorkItem,
) -> GpuSampleParams {
    let mut sample_offsets = [[0_i32; 4]; MAX_SAMPLE_SIZE];
    for (target, offset) in sample_offsets.iter_mut().zip(&sample.plan.offsets) {
        *target = [offset.x, offset.y, offset.z, 0];
    }
    GpuSampleParams {
        anchor_start: [grid.start.x, grid.start.y, grid.start.z, 0],
        anchor_steps: [grid.steps[0], grid.steps[1], grid.steps[2], 0],
        anchor_counts: [grid.counts[0], grid.counts[1], grid.counts[2], 0],
        tile_start: [item.start.x, item.start.y, item.start.z, 0],
        tile_end: [item.end.x, item.end.y, item.end.z, 0],
        sample_offsets,
        metadata: [sample.plan.sample_size() as u32, 0, 0, 0],
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
    use crate::types::{RotationInfo, SearchMode, TextureAlgorithm};

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

        let coordinate = (-32..32)
            .find(|&x| get_texture(TextureAlgorithm::Vanilla3, x, 0, 0, 16) == 5)
            .unwrap();
        let config = ScanConfig {
            algorithm: TextureAlgorithm::Vanilla3,
            scan_order: ScanOrder::Linear,
            directions: vec![0],
            x_range: IntRange {
                start: coordinate,
                end: coordinate + 1,
            },
            y_range: IntRange { start: 0, end: 1 },
            z_range: IntRange { start: 0, end: 1 },
            gpu_tile_size: TileSize { x: 1, z: 1 },
            filter: vec![
                RotationInfo::netherrack(0, 0, 0, 1, crate::types::Face::Up),
                RotationInfo::netherrack(0, 0, 0, 3, crate::types::Face::North),
                RotationInfo::netherrack(0, 0, 0, 2, crate::types::Face::East),
            ],
            ..ScanConfig::default()
        };
        let scanner = GpuScanner::new(&config).unwrap();
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
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].x, coordinate);

        for algorithm in [TextureAlgorithm::Vanilla1, TextureAlgorithm::Vanilla3] {
            let target = (1_i32, 0_i32, -2_i32);
            let mut filter = Vec::new();
            for x in 0..4_i8 {
                for z in 0..4_i8 {
                    filter.push(RotationInfo::new(
                        x,
                        0,
                        z,
                        get_texture(
                            algorithm,
                            target.0 + i32::from(x),
                            target.1,
                            target.2 + i32::from(z),
                            4,
                        ),
                        false,
                    ));
                }
            }
            let mut config = ScanConfig {
                algorithm,
                search_mode: SearchMode::SampleTrie,
                scan_order: ScanOrder::Linear,
                directions: vec![0],
                x_range: IntRange { start: -4, end: 5 },
                y_range: IntRange { start: -1, end: 2 },
                z_range: IntRange { start: -5, end: 4 },
                gpu_tile_size: TileSize { x: 5, z: 4 },
                filter,
                ..ScanConfig::default()
            };
            let scanner = GpuScanner::new(&config).unwrap();
            let run = |config: &ScanConfig| {
                let plan = make_plan(config, config.gpu_tile_size).unwrap();
                let mut found = Vec::new();
                scanner
                    .scan(
                        config,
                        &plan,
                        |batch| found.extend_from_slice(batch),
                        |_, _| {},
                        || false,
                    )
                    .unwrap();
                found.sort_by_key(|item| (item.x, item.y, item.z, item.direction));
                found
            };
            let sampled = run(&config);
            config.search_mode = SearchMode::Naive;
            let naive = run(&config);
            assert_eq!(sampled, naive, "{algorithm}");
            assert!(
                sampled
                    .iter()
                    .any(|found| { (found.x, found.y, found.z) == (target.0, target.1, target.2) })
            );
        }
    }
}
