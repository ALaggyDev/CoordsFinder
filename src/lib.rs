//! Core types and algorithms for CoordsFinder.

pub mod config;
pub mod cpu;
pub mod filter;
pub mod gpu;
pub mod sample_plan;
pub mod scan;
pub mod texture;
pub mod types;

/// Version of the CoordsFinder crate and command-line application.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
