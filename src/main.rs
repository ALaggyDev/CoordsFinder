//! Command-line entry point and backend selection for CoordsFinder.

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use clap::{Parser, ValueEnum};
use coordsfinder::VERSION;
use coordsfinder::config::{ScanOrder, load};
use coordsfinder::cpu::CpuScanner;
use coordsfinder::gpu::GpuScanner;
use coordsfinder::scan::{ScanPlan, make_plan};

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum Backend {
    /// Prefer GPU and fall back to CPU when no compatible adapter is available.
    Auto,
    /// Use the portable multithreaded CPU implementation.
    Cpu,
    /// Use the wgpu compute implementation.
    Gpu,
}

#[derive(Debug, Parser)]
#[command(
    name = "CoordsFinder",
    version = VERSION,
    about = "Crack Minecraft coordinates from texture rotations!",
    disable_version_flag = true,
    arg_required_else_help = true
)]
struct Options {
    /// Search configuration file.
    config: PathBuf,

    /// Execution backend.
    #[arg(short, long, value_enum, default_value_t = Backend::Auto)]
    backend: Backend,

    /// CPU worker count (defaults to the available hardware parallelism).
    #[arg(short, long)]
    threads: Option<NonZeroUsize>,

    /// Validate and summarize the config without scanning.
    #[arg(short = 'e', long)]
    validate: bool,

    /// Also append matches to this file.
    #[arg(short, long, value_name = "FILE")]
    output: Option<PathBuf>,

    /// Print version information.
    #[arg(short = 'v', long, action = clap::ArgAction::Version)]
    version: (),
}

/// Runtime scanner selection after auto-detection and fallback.
enum Scanner {
    Cpu(CpuScanner),
    Gpu(Box<GpuScanner>),
}

/// Shared reporting state works with both sequential GPU callbacks and
/// callbacks arriving from multiple CPU worker threads.
struct Reporter {
    started: Instant,
    last_progress: Mutex<Instant>,
    matches: AtomicU64,
    total_items: usize,
    verbose: bool,
    output: Option<Mutex<File>>,
}

impl Reporter {
    fn new(
        total_items: usize,
        verbose: bool,
        output_path: Option<&PathBuf>,
    ) -> Result<Self, String> {
        let started = Instant::now();
        let output = output_path
            .map(|path| {
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .map(Mutex::new)
                    .map_err(|error| {
                        format!("could not open output file {}: {error}", path.display())
                    })
            })
            .transpose()?;
        Ok(Self {
            started,
            last_progress: Mutex::new(started),
            matches: AtomicU64::new(0),
            total_items,
            verbose,
            output,
        })
    }

    fn report_matches(&self, matches: &[coordsfinder::types::Match]) {
        self.matches
            .fetch_add(matches.len() as u64, Ordering::Relaxed);
        let stdout = io::stdout();
        let mut stdout = stdout.lock();
        for found in matches {
            writeln!(
                stdout,
                "Found with {} mismatch(es)! ({}, {}, {}), direction {}",
                found.mismatches, found.x, found.y, found.z, found.direction
            )
            .expect("could not write match to stdout");
            stdout.flush().expect("could not flush match to stdout");
            if let Some(output) = &self.output {
                let mut output = output.lock().unwrap();
                writeln!(
                    output,
                    "Found with {} mismatch(es)! ({}, {}, {}), direction {}",
                    found.mismatches, found.x, found.y, found.z, found.direction
                )
                .expect("could not write match to output file");
                output
                    .flush()
                    .expect("could not flush match to output file");
            }
        }
    }

    fn report_progress(&self, candidates: u64, completed: usize) {
        let now = Instant::now();
        let mut last = self.last_progress.lock().unwrap();
        if !self.verbose && now.duration_since(*last) < Duration::from_secs(1) {
            return;
        }
        let elapsed = self.started.elapsed().as_secs_f64();
        eprintln!(
            "Progress: {completed}/{} work items, {:.3} M candidates/s, {} match(es).",
            self.total_items,
            candidates as f64 / elapsed / 1_000_000.0,
            self.matches.load(Ordering::Relaxed)
        );
        *last = now;
    }
}

fn print_plan(label: &str, plan: &ScanPlan<'_>, stdout: bool) {
    let candidates = if plan.total_candidates_saturated {
        format!(">= {} (display saturated)", plan.total_candidates)
    } else {
        plan.total_candidates.to_string()
    };
    let summary = format!(
        "{label} plan: {} work items; candidates: {candidates}.",
        plan.total_items()
    );
    if stdout {
        println!("{summary}");
    } else {
        eprintln!("{summary}");
    }
}

fn automatic_threads() -> usize {
    std::thread::available_parallelism()
        .map(NonZeroUsize::get)
        .unwrap_or(1)
}

fn select_scanner(
    options: &Options,
    config: &coordsfinder::config::ScanConfig,
) -> Result<Scanner, String> {
    let threads = options
        .threads
        .map(NonZeroUsize::get)
        .unwrap_or_else(automatic_threads);
    match options.backend {
        Backend::Cpu => CpuScanner::new(threads).map(Scanner::Cpu),
        Backend::Gpu => GpuScanner::new(config).map(|scanner| Scanner::Gpu(Box::new(scanner))),
        Backend::Auto => match GpuScanner::new(config) {
            Ok(scanner) => Ok(Scanner::Gpu(Box::new(scanner))),
            Err(reason) => {
                eprintln!("GPU unavailable ({reason}); falling back to CPU.");
                CpuScanner::new(threads).map(Scanner::Cpu)
            }
        },
    }
}

fn run(options: Options) -> Result<ExitCode, String> {
    let config = load(&options.config)?;
    let order = match config.scan_order {
        ScanOrder::Linear => "linear",
        ScanOrder::Spiral => "spiral",
    };
    let loaded = format!(
        "Loaded {} with {} filters and {} direction(s).",
        config.source_path.display(),
        config.filter.len(),
        config.directions.len()
    );
    if options.validate {
        println!("{loaded}");
        println!("Algorithm: {}; order: {order}.", config.algorithm);
    } else {
        eprintln!("{loaded}");
        eprintln!("Algorithm: {}; order: {order}.", config.algorithm);
    }

    // Validation covers both tile configurations without requiring a GPU.
    if options.validate {
        let cpu_plan = make_plan(&config, config.cpu_tile_size)?;
        let gpu_plan = make_plan(&config, config.gpu_tile_size)?;
        print_plan("CPU", &cpu_plan, true);
        print_plan("GPU", &gpu_plan, true);
        println!("Config and backend plans are valid.");
        return Ok(ExitCode::SUCCESS);
    }

    if options.output.is_none() {
        eprintln!(
            "Warning: no --output file was specified; matches will only be written to stdout."
        );
    }

    let scanner = select_scanner(&options, &config)?;
    let tile_size = match &scanner {
        Scanner::Cpu(scanner) => {
            eprintln!("Backend: CPU ({} threads).", scanner.threads());
            config.cpu_tile_size
        }
        Scanner::Gpu(scanner) => {
            eprintln!(
                "Backend: wgpu/{:?} ({}).",
                scanner.adapter_backend(),
                scanner.adapter_name()
            );
            config.gpu_tile_size
        }
    };
    let plan = make_plan(&config, tile_size)?;
    let plan_label = match &scanner {
        Scanner::Cpu(_) => "CPU",
        Scanner::Gpu(_) => "GPU",
    };
    print_plan(plan_label, &plan, false);

    let cancelled = Arc::new(AtomicBool::new(false));
    let signal = Arc::clone(&cancelled);
    ctrlc::set_handler(move || signal.store(true, Ordering::Relaxed))
        .map_err(|error| format!("could not install interrupt handler: {error}"))?;

    let reporter = Reporter::new(plan.total_items(), config.verbose, options.output.as_ref())?;
    match scanner {
        Scanner::Cpu(scanner) => scanner.scan(
            &config,
            &plan,
            |matches| reporter.report_matches(matches),
            |candidates, completed| reporter.report_progress(candidates, completed),
            || cancelled.load(Ordering::Relaxed),
        )?,
        Scanner::Gpu(scanner) => scanner.scan(
            &config,
            &plan,
            |matches| reporter.report_matches(matches),
            |candidates, completed| reporter.report_progress(candidates, completed),
            || cancelled.load(Ordering::Relaxed),
        )?,
    }

    let elapsed = reporter.started.elapsed().as_secs_f64();
    if cancelled.load(Ordering::Relaxed) {
        eprintln!("Scan cancelled after {elapsed:.2} seconds.");
        return Ok(ExitCode::from(130));
    }
    eprintln!(
        "All done in {elapsed:.2} seconds ({} match(es)).",
        reporter.matches.load(Ordering::Relaxed)
    );
    Ok(ExitCode::SUCCESS)
}

fn main() -> ExitCode {
    match run(Options::parse()) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("Error: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clap_parses_backend_and_threads() {
        let options = Options::try_parse_from([
            "coordsfinder",
            "--backend",
            "cpu",
            "--threads",
            "4",
            "--output",
            "matches.txt",
            "example.conf",
        ])
        .unwrap();
        assert_eq!(options.backend, Backend::Cpu);
        assert_eq!(options.threads.map(NonZeroUsize::get), Some(4));
        assert_eq!(options.output, Some(PathBuf::from("matches.txt")));
        assert_eq!(options.config, PathBuf::from("example.conf"));
    }

    #[test]
    fn clap_rejects_zero_threads() {
        assert!(
            Options::try_parse_from(["coordsfinder", "--threads", "0", "example.conf"]).is_err()
        );
    }
}
