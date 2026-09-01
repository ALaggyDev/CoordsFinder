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
use crate::scan::{ScanPlan, WorkItem, candidate_count};
use crate::types::{CompiledRotation, Match, TextureAlgorithm};

const RESULT_CAPACITY: u32 = 262_144;
const WORKGROUP_XZ: u32 = 16;
const CANDIDATES_PER_THREAD_Y: u32 = 32;
const PACKED_WORKGROUP_X: u32 = 256;
const PACKED_Z_BAND: i32 = 1024;
const PACKED_Y_MIN: i32 = -60;
const PACKED_Y_END: i32 = 1;
const PACKED_MIN_MASKS_PER_ROW: usize = 5;

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

impl GpuFilter {
    /// Packs both representations needed by the two-stage shader into `w`.
    /// The low 16 bits retain the exact acceptance mask; bits 16..19 contain
    /// the four visible rotations used by the cheap packed-Y mask pass.
    fn packed(value: CompiledRotation, visible_rotations: u8) -> Self {
        Self {
            values: [
                i32::from(value.x),
                i32::from(value.y),
                i32::from(value.z),
                i32::from(value.accepted_indices) | (i32::from(visible_rotations) << 16),
            ],
        }
    }
}

/// Direction-specific data for the deliberately narrow packed GPU fast path.
struct PackedDirection {
    filters: Vec<GpuFilter>,
    buckets: Vec<GpuBucket>,
    dense_stride: i32,
    source_z_residue: i32,
    minimum_dx: i32,
    maximum_dx: i32,
    minimum_dz: i32,
    maximum_dz: i32,
    initial_live: u64,
}

/// Start/count pair padded to the 16-byte stride required by a uniform array.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct GpuBucket {
    values: [u32; 4],
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
    packed_generate_pipeline: Box<wgpu::ComputePipeline>,
    packed_filter_pipeline: Box<wgpu::ComputePipeline>,
    packed_bind_group_layout: wgpu::BindGroupLayout,
    packed_params: wgpu::Buffer,
    packed_buckets: wgpu::Buffer,
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
        let specialization = ShaderSpecialization::new(config)?;
        let y_span = i64::from(specialization.y_end) - i64::from(specialization.y_start);
        let constants = [
            ("TEXTURE_ALGORITHM", specialization.algorithm as u32 as f64),
            ("ERROR_TOLERANCE", specialization.error_tolerance as f64),
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

        let packed_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("CoordsFinder packed-Y search shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("packed_search.wgsl").into()),
        });
        let packed_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("CoordsFinder packed-Y bind group layout"),
                entries: &[
                    uniform_layout_entry(0),
                    uniform_layout_entry(1),
                    storage_layout_entry(2, false),
                    storage_layout_entry(3, false),
                    storage_layout_entry(4, false),
                    uniform_layout_entry(5),
                ],
            });
        let packed_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("CoordsFinder packed-Y pipeline layout"),
                bind_group_layouts: &[Some(&packed_bind_group_layout)],
                immediate_size: 0,
            });
        let packed_generate_pipeline = Box::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("CoordsFinder packed-Y signature pipeline"),
                layout: Some(&packed_pipeline_layout),
                module: &packed_shader,
                entry_point: Some("generate_source_signatures"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));
        let packed_filter_pipeline = Box::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("CoordsFinder packed-Y candidate pipeline"),
                layout: Some(&packed_pipeline_layout),
                module: &packed_shader,
                entry_point: Some("filter_candidates"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("search parameters"),
            size: 7 * 4,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let packed_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("packed-Y search parameters"),
            size: 16 * 4,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let packed_buckets = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("packed-Y observation buckets"),
            size: 256 * size_of::<GpuBucket>() as u64,
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
            packed_generate_pipeline,
            packed_filter_pipeline,
            packed_bind_group_layout,
            packed_params,
            packed_buckets,
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
        let packed_directions = prepared
            .directions
            .iter()
            .map(|direction| {
                prepare_packed_direction(config, &direction.constraints, direction.forced_errors)
            })
            .collect::<Vec<_>>();

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
            self.queue
                .write_buffer(&self.counters, 0, bytemuck::bytes_of(&[0_u32; 2]));

            let used_packed = if let Some(packed) = &packed_directions[item.direction_index] {
                self.dispatch_packed_item(&item, packed, x_span, z_span)?
            } else {
                false
            };
            if !used_packed {
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

                let mut encoder =
                    self.device
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
            }

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

    /// Dispatches source generation and candidate-owned filtering for one tile.
    /// Returns `false` when coordinate halos or buffer limits require the
    /// ordinary shader for this particular item.
    fn dispatch_packed_item(
        &self,
        item: &WorkItem,
        packed: &PackedDirection,
        x_span: u32,
        z_span: u32,
    ) -> Result<bool, String> {
        let Some(source_x_start) = item.start.x.checked_add(packed.minimum_dx) else {
            return Ok(false);
        };
        let Some(source_x_end) = item.end.x.checked_add(packed.maximum_dx) else {
            return Ok(false);
        };
        let source_width = u32::try_from(i64::from(source_x_end) - i64::from(source_x_start))
            .map_err(|_| "packed GPU source-X halo is too wide".to_owned())?;

        // One source buffer is reused for every candidate-Z band in the tile.
        // Size it for the widest band including the observation dz halo.
        let mut maximum_source_rows = 0_u32;
        for band_start in (item.start.z..item.end.z).step_by(PACKED_Z_BAND as usize) {
            let band_end = item.end.z.min(band_start.saturating_add(PACKED_Z_BAND));
            let Some((_, rows)) = packed_source_rows(band_start, band_end, packed) else {
                return Ok(false);
            };
            maximum_source_rows = maximum_source_rows.max(rows);
        }
        if maximum_source_rows == 0 {
            return Ok(false);
        }

        let source_bytes = u64::from(source_width)
            .checked_mul(u64::from(maximum_source_rows))
            .and_then(|size| size.checked_mul(4 * size_of::<u64>() as u64))
            .ok_or_else(|| "packed GPU source buffer size overflowed".to_owned())?;
        let limits = self.device.limits();
        if source_bytes > limits.max_buffer_size
            || source_bytes > limits.max_storage_buffer_binding_size
        {
            return Ok(false);
        }

        let source_masks =
            storage_buffer(&self.device, "packed-Y source masks", source_bytes, false);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CoordsFinder packed-Y bindings"),
            layout: &self.packed_bind_group_layout,
            entries: &[
                binding(0, &self.packed_params),
                binding(1, &self.filters),
                binding(2, &source_masks),
                binding(3, &self.results),
                binding(4, &self.counters),
                binding(5, &self.packed_buckets),
            ],
        });
        self.queue
            .write_buffer(&self.filters, 0, bytemuck::cast_slice(&packed.filters));
        self.queue.write_buffer(
            &self.packed_buckets,
            0,
            bytemuck::cast_slice(&packed.buckets),
        );

        for band_start in (item.start.z..item.end.z).step_by(PACKED_Z_BAND as usize) {
            let band_end = item.end.z.min(band_start.saturating_add(PACKED_Z_BAND));
            let (source_z_start, source_rows) =
                packed_source_rows(band_start, band_end, packed).unwrap();
            let candidate_rows = (band_end - band_start) as u32;
            let params = [
                source_x_start as u32,
                source_z_start as u32,
                source_width,
                source_rows,
                item.start.x as u32,
                band_start as u32,
                x_span,
                candidate_rows,
                item.direction as u32,
                packed.filters.len() as u32,
                packed.dense_stride as u32,
                packed.source_z_residue as u32,
                packed.initial_live as u32,
                (packed.initial_live >> 32) as u32,
                packed.minimum_dx as u32,
                0,
            ];
            self.queue
                .write_buffer(&self.packed_params, 0, bytemuck::bytes_of(&params));

            let signature_workgroups = [source_width.div_ceil(PACKED_WORKGROUP_X), source_rows, 1];
            let candidate_workgroups = [x_span.div_ceil(PACKED_WORKGROUP_X), candidate_rows, 1];
            if signature_workgroups
                .iter()
                .chain(candidate_workgroups.iter())
                .any(|&count| count > self.max_workgroups_per_dimension)
            {
                return Err("packed GPU tile exceeds this adapter's dispatch limits".to_owned());
            }

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("CoordsFinder packed-Y commands"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("packed-Y source signature pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.packed_generate_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(
                    signature_workgroups[0],
                    signature_workgroups[1],
                    signature_workgroups[2],
                );
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("packed-Y candidate filter pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.packed_filter_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(
                    candidate_workgroups[0],
                    candidate_workgroups[1],
                    candidate_workgroups[2],
                );
            }
            self.queue.submit([encoder.finish()]);
        }

        // z_span is passed separately so an accidental mismatch between plan
        // bounds and dispatch arithmetic is caught in debug/test builds.
        debug_assert_eq!(z_span, (item.end.z - item.start.z) as u32);
        Ok(true)
    }
}

/// Builds the packed shader's direction data when every cheap-pass invariant
/// holds and a single dense source-Z modulus is available.
fn prepare_packed_direction(
    config: &ScanConfig,
    constraints: &[CompiledRotation],
    forced_errors: i32,
) -> Option<PackedDirection> {
    if config.algorithm != TextureAlgorithm::Vanilla3
        || std::env::var_os("COORDSFINDER_DISABLE_PACKED_GPU").is_some()
        || config.error_tolerance != 0
        || forced_errors != 0
        || config.y_range.start < PACKED_Y_MIN
        || config.y_range.end > PACKED_Y_END
        || constraints.is_empty()
    {
        return None;
    }

    let mut representable = Vec::with_capacity(constraints.len());
    for &constraint in constraints {
        if !(0..=1).contains(&constraint.y) {
            return None;
        }
        representable.push((
            constraint,
            visible_rotation_mask(constraint.accepted_indices)?,
        ));
    }

    let dense_stride = select_dense_stride(constraints);
    if dense_stride == 0 {
        // The first prototype intentionally leaves the general six-of-32
        // source cover to the existing shader. Dense indexing is direct and
        // is the relevant path for the supplied performance benchmark.
        return None;
    }
    let source_z_residue = if dense_stride % 2 == 0 {
        dense_stride / 2
    } else {
        0
    };
    let initial_live = packed_origin_y_mask(config.y_range.start, config.y_range.end);

    // Each observation belongs to exactly one candidate-Z residue because
    // candidate_z + dz must equal the selected source residue. Reordering into
    // these buckets removes a coverage test and roughly three quarters of the
    // observation-loop iterations from the benchmark's stride-four kernel.
    let mut filters = Vec::with_capacity(constraints.len());
    let mut buckets = Vec::with_capacity(dense_stride as usize);
    for candidate_residue in 0..dense_stride {
        let start = filters.len() as u32;
        let mut bucket = representable
            .iter()
            .copied()
            .filter(|(constraint, _)| {
                (candidate_residue + i32::from(constraint.z)).rem_euclid(dense_stride)
                    == source_z_residue
            })
            .collect::<Vec<_>>();
        // Narrower visible-rotation unions are normally more selective, so
        // place them before side masks to make `live == 0` happen sooner.
        bucket.sort_by_key(|(_, visible)| visible.count_ones());
        filters.extend(
            bucket
                .into_iter()
                .map(|(constraint, visible)| GpuFilter::packed(constraint, visible)),
        );
        buckets.push(GpuBucket {
            values: [start, filters.len() as u32 - start, 0, 0],
        });
    }
    debug_assert_eq!(filters.len(), constraints.len());

    // Append a verifier list for each candidate residue containing only the
    // observations omitted from that residue's mask pass. Rechecking the mask
    // bucket would be exact but redundant: every surviving bit already passed
    // those observations.
    for candidate_residue in 0..dense_stride {
        let start = filters.len() as u32;
        filters.extend(
            representable
                .iter()
                .copied()
                .filter(|(constraint, _)| {
                    (candidate_residue + i32::from(constraint.z)).rem_euclid(dense_stride)
                        != source_z_residue
                })
                .map(|(constraint, visible)| GpuFilter::packed(constraint, visible)),
        );
        let bucket = &mut buckets[candidate_residue as usize];
        bucket.values[2] = start;
        bucket.values[3] = filters.len() as u32 - start;
    }
    if filters.len() > 256 {
        return None;
    }

    Some(PackedDirection {
        filters,
        buckets,
        dense_stride,
        source_z_residue,
        minimum_dx: constraints
            .iter()
            .map(|constraint| i32::from(constraint.x))
            .min()?,
        maximum_dx: constraints
            .iter()
            .map(|constraint| i32::from(constraint.x))
            .max()?,
        minimum_dz: constraints
            .iter()
            .map(|constraint| i32::from(constraint.z))
            .min()?,
        maximum_dz: constraints
            .iter()
            .map(|constraint| i32::from(constraint.z))
            .max()?,
        initial_live,
    })
}

/// Converts a complete 16-way model mask into four visible Vanilla-3 groups.
/// Partial groups (notably some netherrack constraints) cannot be represented
/// by a four-mask source signature and therefore reject the packed fast path.
fn visible_rotation_mask(accepted_indices: u16) -> Option<u8> {
    let mut visible = 0_u8;
    let mut represented = 0_u16;
    for rotation in 0..4 {
        let group = 0xf_u16 << (rotation * 4);
        if accepted_indices & group == group {
            visible |= 1 << rotation;
            represented |= group;
        }
    }
    (visible != 0 && represented == accepted_indices).then_some(visible)
}

fn select_dense_stride(constraints: &[CompiledRotation]) -> i32 {
    let mut stride = constraints.len() / PACKED_MIN_MASKS_PER_ROW;
    while stride >= 2 {
        let mut counts = vec![0_usize; stride];
        for constraint in constraints {
            counts[i32::from(constraint.z).rem_euclid(stride as i32) as usize] += 1;
        }
        if counts
            .into_iter()
            .all(|count| count >= PACKED_MIN_MASKS_PER_ROW)
        {
            return stride as i32;
        }
        stride -= 1;
    }
    0
}

fn packed_origin_y_mask(start: i32, end: i32) -> u64 {
    let count = (end - start) as u32;
    ((1_u64 << count) - 1) << (start - PACKED_Y_MIN)
}

/// Returns the first selected source Z and row count covering a candidate band
/// plus its observation halo. `None` means checked coordinate arithmetic found
/// an i32 edge case, for which the general shader is used instead.
fn packed_source_rows(
    band_start: i32,
    band_end: i32,
    packed: &PackedDirection,
) -> Option<(i32, u32)> {
    let minimum_source_z = band_start.checked_add(packed.minimum_dz)?;
    let maximum_source_z_exclusive = band_end.checked_add(packed.maximum_dz)?;
    let delta = (packed.source_z_residue - minimum_source_z.rem_euclid(packed.dense_stride))
        .rem_euclid(packed.dense_stride);
    let first = minimum_source_z.checked_add(delta)?;
    if first >= maximum_source_z_exclusive {
        return Some((first, 0));
    }
    let rows = (i64::from(maximum_source_z_exclusive) - 1 - i64::from(first))
        / i64::from(packed.dense_stride)
        + 1;
    Some((first, u32::try_from(rows).ok()?))
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
    use crate::config::{IntRange, ScanOrder, TileSize, load};
    use crate::scan::make_plan;
    use crate::texture::{TextureSampler, Vanilla3, get_texture};
    use crate::types::{RotationInfo, TextureAlgorithm};

    #[test]
    fn search_shader_is_valid_wgsl() {
        for source in [
            include_str!("search.wgsl"),
            include_str!("packed_search.wgsl"),
        ] {
            let module = wgpu::naga::front::wgsl::parse_str(source).unwrap();
            let mut validator = wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::SHADER_INT64,
            );
            validator.validate(&module).unwrap();
        }
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
    }

    #[test]
    fn packed_gpu_matches_brute_force_when_available() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut config = load(root.join("examples/packed-cpu-benchmark.conf")).unwrap();
        config.scan_order = ScanOrder::Linear;
        config.x_range = IntRange {
            start: -136,
            end: -134,
        };
        config.y_range = IntRange { start: -60, end: 1 };
        config.z_range = IntRange {
            start: -1200,
            end: 0,
        };
        config.gpu_tile_size = TileSize { x: 2, z: 1200 };

        let prepared = prepare_filters(
            &config.filter,
            config.algorithm,
            &config.directions,
            config.error_tolerance,
        )
        .unwrap();
        let direction = &prepared.directions[0];
        assert!(
            prepare_packed_direction(&config, &direction.constraints, direction.forced_errors)
                .is_some()
        );

        let Ok(scanner) = GpuScanner::new(&config) else {
            eprintln!("skipping packed GPU test: no compatible adapter");
            return;
        };
        let plan = make_plan(&config, config.gpu_tile_size).unwrap();
        let mut packed = Vec::new();
        scanner
            .scan(
                &config,
                &plan,
                |batch| packed.extend_from_slice(batch),
                |_, _| {},
                || false,
            )
            .unwrap();

        let mut brute = Vec::new();
        for x in config.x_range.start..config.x_range.end {
            for z in config.z_range.start..config.z_range.end {
                for y in config.y_range.start..config.y_range.end {
                    let exact = direction.constraints.iter().all(|observation| {
                        let variant = Vanilla3::sample(
                            x.wrapping_add(i32::from(observation.x)),
                            y.wrapping_add(i32::from(observation.y)),
                            z.wrapping_add(i32::from(observation.z)),
                            16,
                        );
                        observation.accepted_indices & (1 << variant) != 0
                    });
                    if exact {
                        brute.push(Match {
                            x,
                            y,
                            z,
                            mismatches: 0,
                            direction: 0,
                        });
                    }
                }
            }
        }
        packed.sort_by_key(|found| (found.x, found.y, found.z));
        brute.sort_by_key(|found| (found.x, found.y, found.z));
        assert_eq!(packed, brute);
    }
}
