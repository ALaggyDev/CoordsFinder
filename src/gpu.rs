use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::config::ScanConfig;
use crate::scan::{ScanPlan, candidate_count, directional_filter};
use crate::types::{Match, RotationInfo};

const RESULT_CAPACITY: u32 = 262_144;
const WORKGROUP_SIZE: u32 = 8;

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct GpuFilter {
    x: u32,
    y: u32,
    z: u32,
    rotation: u32,
    visible_mask: u32,
}

impl From<RotationInfo> for GpuFilter {
    fn from(value: RotationInfo) -> Self {
        Self {
            x: i32::from(value.x) as u32,
            y: i32::from(value.y) as u32,
            z: i32::from(value.z) as u32,
            rotation: u32::from(value.rotation),
            visible_mask: u32::from(value.visible_mask),
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
    pipeline: wgpu::ComputePipeline,
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
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("CoordsFinder search pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("search"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params = storage_buffer(&device, "search parameters", 11 * 4, false);
        let filters = storage_buffer(&device, "texture filters", 256 * 20, false);
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
            layout: &pipeline.get_bind_group_layout(0),
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
    ) -> Result<(), String> {
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
            let x_span = (i64::from(item.end.x) - i64::from(item.start.x)) as u32;
            let y_span = (i64::from(item.end.y) - i64::from(item.start.y)) as u32;
            let z_span = (i64::from(item.end.z) - i64::from(item.start.z)) as u32;
            let workgroups = [
                x_span.div_ceil(WORKGROUP_SIZE),
                z_span.div_ceil(WORKGROUP_SIZE),
                y_span,
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
                config.algorithm as u32,
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
                pass.set_pipeline(&self.pipeline);
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
    use super::*;
    use crate::config::{IntRange, ScanOrder, TileSize};
    use crate::scan::make_plan;
    use crate::texture::get_texture;
    use crate::types::TextureAlgorithm;

    #[test]
    fn gpu_matches_all_reference_algorithms_when_available() {
        let Ok(scanner) = GpuScanner::new() else {
            eprintln!("skipping GPU test: no compatible adapter");
            return;
        };
        let coordinate = (17, -4, -31);
        for algorithm in [
            TextureAlgorithm::Vanilla1,
            TextureAlgorithm::Vanilla2,
            TextureAlgorithm::Vanilla3,
            TextureAlgorithm::Sodium1,
            TextureAlgorithm::Sodium2,
        ] {
            let config = ScanConfig {
                algorithm,
                scan_order: ScanOrder::Linear,
                directions: vec![0],
                x_range: IntRange { start: 17, end: 18 },
                y_range: IntRange { start: -4, end: -3 },
                z_range: IntRange {
                    start: -31,
                    end: -30,
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
                )
                .unwrap();
            assert_eq!(
                matches,
                vec![Match {
                    x: 17,
                    y: -4,
                    z: -31,
                    mismatches: 0,
                    direction: 0
                }],
                "{algorithm}"
            );
        }
    }
}
