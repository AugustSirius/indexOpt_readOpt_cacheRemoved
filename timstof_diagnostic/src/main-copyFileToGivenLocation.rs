use clap::Parser;
use colored::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use sysinfo::System;
use timsrust::{
    converters::ConvertableDomain,
    readers::{FrameReader, MetadataReader},
    MSLevel,
};

#[derive(Parser, Debug)]
#[command(author, version, about = "TimsTOF Read Performance Diagnostic Tool", long_about = None)]
struct Args {
    /// Path to the Bruker .d folder
    d_folder: String,
    
    /// Optional: number of frames to process (default: all)
    #[arg(short = 'n', long)]
    max_frames: Option<usize>,
    
    /// Optional: number of threads (default: auto)
    #[arg(short = 't', long)]
    threads: Option<usize>,
    
    /// Optional: enable verbose output
    #[arg(short = 'v', long)]
    verbose: bool,
    
    /// Optional: force use of local storage (auto-detect by default)
    #[arg(short = 'l', long)]
    force_local: bool,
    
    /// Optional: disable local storage optimization
    #[arg(long)]
    no_local: bool,
}

// Add this struct for local storage management
struct LocalStorageManager {
    original_path: PathBuf,
    local_path: Option<PathBuf>,
    cleanup_on_drop: bool,
}

impl LocalStorageManager {
    fn new(original_path: &Path) -> Self {
        Self {
            original_path: original_path.to_path_buf(),
            local_path: None,
            cleanup_on_drop: true,
        }
    }
    
    fn should_use_local(&self, force_local: bool, no_local: bool) -> bool {
        if no_local {
            return false;
        }
        
        if force_local {
            return true;
        }
        
        // Auto-detect: Check if path is on network storage
        self.is_network_storage(&self.original_path)
    }
    
    fn is_network_storage(&self, path: &Path) -> bool {
        let fs_type = get_filesystem_type(path);
        matches!(fs_type.as_str(), "lustre" | "nfs" | "gpfs" | "beegfs" | "cifs")
    }
    
    fn copy_to_local(&mut self) -> Result<PathBuf, Box<dyn std::error::Error>> {
        println!("\n{}", "Local Storage Optimization".bold().cyan());
        println!("{}", "─".repeat(40).cyan());
        
        // Test performance first
        let avg_read_time = self.test_read_performance(&self.original_path)?;
        println!("  Source read performance: {:.2} ms/frame", avg_read_time);
        
        if avg_read_time < 1.0 {
            println!("  {} Source is already fast, skipping local copy", "→".blue());
            return Ok(self.original_path.clone());
        }
        
        // Find best local storage
        let local_base = self.find_best_local_storage()?;
        let folder_name = self.original_path.file_name()
            .ok_or("Invalid folder name")?
            .to_string_lossy();
        
        // Create unique local path with PID
        let pid = std::process::id();
        let local_path = local_base.join(format!("{}_{}", folder_name, pid));
        
        // Check space
        let data_size = self.get_folder_size(&self.original_path)?;
        let available_space = self.get_available_space(&local_base)?;
        
        println!("  Data size: {:.2} GB", data_size as f64 / 1_073_741_824.0);
        println!("  Available space: {:.2} GB", available_space as f64 / 1_073_741_824.0);
        
        if available_space < data_size + 1_073_741_824 { // Need 1GB buffer
            return Err("Insufficient local storage space".into());
        }
        
        // Copy data
        println!("  {} Copying to: {}", "→".blue(), local_path.display());
        let copy_start = Instant::now();
        
        self.copy_dir_recursive(&self.original_path, &local_path)?;
        
        let copy_time = copy_start.elapsed();
        println!("  {} Copy completed in {:.2}s", "✓".green(), copy_time.as_secs_f64());
        
        // Verify copy
        let local_size = self.get_folder_size(&local_path)?;
        if (local_size as i64 - data_size as i64).abs() > 1_048_576 { // 1MB tolerance
            return Err("Copy verification failed: size mismatch".into());
        }
        
        self.local_path = Some(local_path.clone());
        Ok(local_path)
    }
    
    fn test_read_performance(&self, path: &Path) -> Result<f64, Box<dyn std::error::Error>> {
        // Quick test: read a few frames
        let frames = FrameReader::new(path)?;
        let mut total_time = 0.0;
        let test_count = 10.min(frames.len());
        
        for i in 0..test_count {
            let start = Instant::now();
            let _ = frames.get(i)?;
            total_time += start.elapsed().as_secs_f64() * 1000.0;
        }
        
        Ok(total_time / test_count as f64)
    }
    
    fn find_best_local_storage(&self) -> Result<PathBuf, Box<dyn std::error::Error>> {
        // Priority order: /tmp, /scratch, /local, /dev/shm
        let candidates = vec![
            PathBuf::from("/tmp"),
            PathBuf::from("/scratch"),
            PathBuf::from("/local"),
            PathBuf::from("/var/tmp"),
        ];
        
        for candidate in candidates {
            if candidate.exists() && candidate.is_dir() {
                // Check if writable
                let test_file = candidate.join(format!("test_write_{}", std::process::id()));
                if fs::write(&test_file, b"test").is_ok() {
                    let _ = fs::remove_file(&test_file);
                    
                    // Check available space (need at least 20GB)
                    if let Ok(space) = self.get_available_space(&candidate) {
                        if space > 20 * 1_073_741_824 {
                            println!("  Selected local storage: {}", candidate.display());
                            return Ok(candidate);
                        }
                    }
                }
            }
        }
        
        Err("No suitable local storage found".into())
    }
    
    fn get_folder_size(&self, path: &Path) -> Result<u64, Box<dyn std::error::Error>> {
        let output = std::process::Command::new("du")
            .arg("-sb")
            .arg(path)
            .output()?;
        
        let size_str = String::from_utf8_lossy(&output.stdout);
        let size = size_str
            .split_whitespace()
            .next()
            .and_then(|s| s.parse::<u64>().ok())
            .ok_or("Failed to parse folder size")?;
        
        Ok(size)
    }
    
    fn get_available_space(&self, path: &Path) -> Result<u64, Box<dyn std::error::Error>> {
        let output = std::process::Command::new("df")
            .arg("-B1")
            .arg(path)
            .output()?;
        
        let output_str = String::from_utf8_lossy(&output.stdout);
        let lines: Vec<&str> = output_str.lines().collect();
        
        if lines.len() > 1 {
            let fields: Vec<&str> = lines[1].split_whitespace().collect();
            if fields.len() > 3 {
                return fields[3].parse::<u64>()
                    .map_err(|e| e.into());
            }
        }
        
        Err("Failed to get available space".into())
    }
    
    fn copy_dir_recursive(&self, from: &Path, to: &Path) -> Result<(), Box<dyn std::error::Error>> {
        fs::create_dir_all(to)?;
        
        for entry in fs::read_dir(from)? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            let from_path = entry.path();
            let to_path = to.join(entry.file_name());
            
            if file_type.is_dir() {
                self.copy_dir_recursive(&from_path, &to_path)?;
            } else {
                fs::copy(&from_path, &to_path)?;
            }
        }
        
        Ok(())
    }
    
    fn get_working_path(&self) -> &Path {
        self.local_path.as_deref().unwrap_or(&self.original_path)
    }
}

impl Drop for LocalStorageManager {
    fn drop(&mut self) {
        if self.cleanup_on_drop {
            if let Some(local_path) = &self.local_path {
                println!("\n{} Cleaning up local storage...", "→".blue());
                if let Err(e) = fs::remove_dir_all(local_path) {
                    eprintln!("  {} Failed to cleanup {}: {}", "⚠".yellow(), local_path.display(), e);
                } else {
                    println!("  {} Cleaned up: {}", "✓".green(), local_path.display());
                }
            }
        }
    }
}

// Keep all existing structs and functions...
// [Previous SystemInfo, PerformanceMetrics, FrameStats, DiagnosticLogger definitions remain the same]

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SystemInfo {
    hostname: String,
    cpu_count: usize,
    cpu_physical: usize,
    rayon_threads: usize,
    total_memory_gb: f64,
    available_memory_gb: f64,
    filesystem_type: String,
    rust_version: String,
    timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PerformanceMetrics {
    stage: String,
    duration_ms: f64,
    frames_processed: Option<usize>,
    throughput_fps: Option<f64>,
    memory_used_mb: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FrameStats {
    index: usize,
    ms_level: String,
    rt_seconds: f64,
    num_peaks: usize,
    num_scans: usize,
    read_time_ms: f64,
    process_time_ms: f64,
}

struct DiagnosticLogger {
    metrics: Arc<Mutex<Vec<PerformanceMetrics>>>,
    frame_stats: Arc<Mutex<Vec<FrameStats>>>,
    verbose: bool,
}

// [Keep all existing DiagnosticLogger methods...]
impl DiagnosticLogger {
    fn new(verbose: bool) -> Self {
        DiagnosticLogger {
            metrics: Arc::new(Mutex::new(Vec::new())),
            frame_stats: Arc::new(Mutex::new(Vec::new())),
            verbose,
        }
    }
    
    fn log_metric(&self, metric: PerformanceMetrics) {
        if self.verbose {
            println!("  {} {} took {:.2} ms", 
                     "→".blue(), 
                     metric.stage, 
                     metric.duration_ms);
        }
        self.metrics.lock().unwrap().push(metric);
    }
    
    fn log_frame(&self, stats: FrameStats) {
        self.frame_stats.lock().unwrap().push(stats);
    }
    
    fn save_report(&self, output_path: &str) -> std::io::Result<()> {
        let metrics = self.metrics.lock().unwrap();
        let frame_stats = self.frame_stats.lock().unwrap();
        
        let report = serde_json::json!({
            "metrics": metrics.clone(),
            "frame_stats": frame_stats.clone(),
        });
        
        let mut file = File::create(output_path)?;
        file.write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
        Ok(())
    }
}

// Update get_filesystem_type to be accessible
fn get_filesystem_type(path: &Path) -> String {
    #[cfg(target_os = "linux")]
    {
        if let Ok(output) = std::process::Command::new("df")
            .arg("-T")
            .arg(path)
            .output()
        {
            let output_str = String::from_utf8_lossy(&output.stdout);
            let lines: Vec<&str> = output_str.lines().collect();
            if lines.len() > 1 {
                let fields: Vec<&str> = lines[1].split_whitespace().collect();
                if fields.len() > 1 {
                    return fields[1].to_string();
                }
            }
        }
    }
    "unknown".to_string()
}

// [Keep all other existing functions: get_system_info, print_header, print_system_info, test_io_performance, test_parallel_overhead, read_timstof_diagnostic, print_summary]

fn get_system_info() -> SystemInfo {
    let mut sys = System::new_all();
    sys.refresh_all();
    
    let filesystem_type = get_filesystem_type(&std::env::current_dir().unwrap());
    
    SystemInfo {
        hostname: System::host_name().unwrap_or_else(|| "unknown".to_string()),
        cpu_count: num_cpus::get(),
        cpu_physical: num_cpus::get_physical(),
        rayon_threads: rayon::current_num_threads(),
        total_memory_gb: sys.total_memory() as f64 / 1_073_741_824.0,
        available_memory_gb: sys.available_memory() as f64 / 1_073_741_824.0,
        filesystem_type,
        rust_version: "1.70+".to_string(),
        timestamp: chrono::Local::now().to_rfc3339(),
    }
}

fn print_header(d_folder: &Path) {
    println!("\n{}", "═".repeat(80).blue());
    println!("{}", "    TimsTOF Read Performance Diagnostic Tool".bold().blue());
    println!("{}", "═".repeat(80).blue());
    println!("\n{} {}", "Data folder:".bold(), d_folder.display());
    println!("{} {}", "Start time:".bold(), chrono::Local::now().format("%Y-%m-%d %H:%M:%S"));
}

fn print_system_info(info: &SystemInfo) {
    println!("\n{}", "System Information".bold().green());
    println!("{}", "─".repeat(40).green());
    println!("  Hostname:        {}", info.hostname);
    println!("  CPUs:            {} logical, {} physical", info.cpu_count, info.cpu_physical);
    println!("  Rayon threads:   {}", info.rayon_threads);
    println!("  Memory:          {:.1} GB total, {:.1} GB available", 
             info.total_memory_gb, info.available_memory_gb);
    println!("  Filesystem:      {}", info.filesystem_type);
}

// [Keep existing test functions but they need the Path parameter]
fn test_io_performance(d_folder: &Path, logger: &DiagnosticLogger) -> std::io::Result<()> {
    println!("\n{}", "I/O Performance Test".bold().yellow());
    println!("{}", "─".repeat(40).yellow());
    
    let tdf_path = d_folder.join("analysis.tdf");
    let file_size = std::fs::metadata(&tdf_path)?.len();
    println!("  TDF file size: {:.2} MB", file_size as f64 / 1_048_576.0);
    
    // Test sequential read
    let mut buffer = vec![0u8; 1024 * 1024]; // 1MB buffer
    let start = Instant::now();
    let mut file = File::open(&tdf_path)?;
    use std::io::Read;
    let mut total_read = 0;
    for _ in 0..10 {
        if let Ok(n) = file.read(&mut buffer) {
            total_read += n;
            if n == 0 { break; }
        }
    }
    let seq_time = start.elapsed();
    let seq_throughput = (total_read as f64 / 1_048_576.0) / seq_time.as_secs_f64();
    println!("  Sequential read: {:.2} MB/s", seq_throughput);
    
    logger.log_metric(PerformanceMetrics {
        stage: "io_sequential_read".to_string(),
        duration_ms: seq_time.as_secs_f64() * 1000.0,
        frames_processed: None,
        throughput_fps: Some(seq_throughput),
        memory_used_mb: None,
    });
    
    // Test random access
    let start = Instant::now();
    for i in 0..10 {
        let offset = (i * 1024 * 1024) % file_size;
        use std::io::{Seek, SeekFrom};
        let mut file = File::open(&tdf_path)?;
        file.seek(SeekFrom::Start(offset))?;
        let _ = file.read(&mut buffer[..1024]);
    }
    let random_time = start.elapsed();
    println!("  Random access (10 seeks): {:.2} ms", random_time.as_secs_f64() * 1000.0);
    
    logger.log_metric(PerformanceMetrics {
        stage: "io_random_access".to_string(),
        duration_ms: random_time.as_secs_f64() * 1000.0,
        frames_processed: None,
        throughput_fps: None,
        memory_used_mb: None,
    });
    
    Ok(())
}

fn test_parallel_overhead(logger: &DiagnosticLogger) {
    println!("\n{}", "Parallel Processing Overhead Test".bold().cyan());
    println!("{}", "─".repeat(40).cyan());
    
    // Test thread pool creation
    let start = Instant::now();
    let _sum: usize = (0..1000).into_par_iter().map(|x| x).sum();
    let pool_time = start.elapsed();
    println!("  Thread pool (1000 tasks): {:.2} ms", pool_time.as_secs_f64() * 1000.0);
    
    // Test different chunk sizes
    let test_sizes = vec![1, 10, 100, 1000];
    for chunk_size in test_sizes {
        let start = Instant::now();
        let _: Vec<_> = (0..10000)
            .into_par_iter()
            .with_min_len(chunk_size)
            .map(|x| x * 2)
            .collect();
        let duration = start.elapsed();
        println!("  Chunk size {:4}: {:.2} ms", chunk_size, duration.as_secs_f64() * 1000.0);
        
        logger.log_metric(PerformanceMetrics {
            stage: format!("parallel_chunk_{}", chunk_size),
            duration_ms: duration.as_secs_f64() * 1000.0,
            frames_processed: Some(10000),
            throughput_fps: Some(10000.0 / duration.as_secs_f64()),
            memory_used_mb: None,
        });
    }
    
    // Test memory allocation in parallel
    let start = Instant::now();
    let _: Vec<Vec<u8>> = (0..100)
        .into_par_iter()
        .map(|_| vec![0u8; 1024 * 1024]) // 1MB each
        .collect();
    let alloc_time = start.elapsed();
    println!("  Parallel memory allocation (100MB): {:.2} ms", 
             alloc_time.as_secs_f64() * 1000.0);
}

fn read_timstof_diagnostic(
    d_folder: &Path, 
    max_frames: Option<usize>,
    logger: &DiagnosticLogger
) -> Result<(), Box<dyn std::error::Error>> {
    
    println!("\n{}", "Reading TimsTOF Data".bold().magenta());
    println!("{}", "─".repeat(40).magenta());
    
    // Step 1: Read metadata
    let metadata_start = Instant::now();
    let tdf_path = d_folder.join("analysis.tdf");
    let meta = MetadataReader::new(&tdf_path)?;
    let mz_cv = Arc::new(meta.mz_converter);
    let im_cv = Arc::new(meta.im_converter);
    let metadata_time = metadata_start.elapsed();
    
    println!("  {} Metadata read: {:.2} ms", "✓".green(), metadata_time.as_secs_f64() * 1000.0);
    logger.log_metric(PerformanceMetrics {
        stage: "metadata_read".to_string(),
        duration_ms: metadata_time.as_secs_f64() * 1000.0,
        frames_processed: None,
        throughput_fps: None,
        memory_used_mb: None,
    });
    
    // Step 2: Initialize frame reader
    let frame_init_start = Instant::now();
    let frames = FrameReader::new(d_folder)?;
    let total_frames = frames.len();
    let frame_init_time = frame_init_start.elapsed();
    
    println!("  {} Frame reader init: {:.2} ms ({} frames)", 
             "✓".green(), 
             frame_init_time.as_secs_f64() * 1000.0,
             total_frames);
    
    logger.log_metric(PerformanceMetrics {
        stage: "frame_reader_init".to_string(),
        duration_ms: frame_init_time.as_secs_f64() * 1000.0,
        frames_processed: Some(total_frames),
        throughput_fps: None,
        memory_used_mb: None,
    });
    
    // Step 3: Test single frame read
    println!("\n  Testing individual frame reads...");
    let test_indices = vec![0, 1, total_frames/2, total_frames-1];
    
    for &idx in &test_indices {
        if idx >= total_frames { continue; }
        
        let start = Instant::now();
        match frames.get(idx) {
            Ok(frame) => {
                let duration = start.elapsed();
                println!("    Frame {:5}: {:.2} ms (MS{}, {} peaks)", 
                         idx, 
                         duration.as_secs_f64() * 1000.0,
                         match frame.ms_level {
                             MSLevel::MS1 => "1",
                             MSLevel::MS2 => "2",
                             _ => "?",
                         },
                         frame.tof_indices.len());
                
                logger.log_frame(FrameStats {
                    index: idx,
                    ms_level: format!("{:?}", frame.ms_level),
                    rt_seconds: frame.rt_in_seconds,
                    num_peaks: frame.tof_indices.len(),
                    num_scans: frame.scan_offsets.len() - 1,
                    read_time_ms: duration.as_secs_f64() * 1000.0,
                    process_time_ms: 0.0,
                });
            },
            Err(e) => {
                println!("    Frame {:5}: {} {}", idx, "FAILED".red(), e);
            }
        }
    }
    
    // Step 4: Process frames (limited by max_frames if specified)
    let frames_to_process = max_frames.unwrap_or(total_frames).min(total_frames);
    let start_frame = if frames_to_process < total_frames { 0 } else {
        // Check if first frame is readable
        match frames.get(0) {
            Ok(_) => 0,
            Err(_) => {
                println!("  {} Skipping first frame (unreadable)", "⚠".yellow());
                1
            }
        }
    };
    
    println!("\n  Processing {} frames (starting from index {})...", 
             frames_to_process - start_frame, start_frame);
    
    // Sequential processing test
    let seq_start = Instant::now();
    let mut seq_ms1_count = 0;
    let mut seq_ms2_count = 0;
    let mut seq_total_peaks = 0;
    
    for idx in start_frame..(start_frame + 10).min(frames_to_process) {
        if let Ok(frame) = frames.get(idx) {
            seq_total_peaks += frame.tof_indices.len();
            match frame.ms_level {
                MSLevel::MS1 => seq_ms1_count += 1,
                MSLevel::MS2 => seq_ms2_count += 1,
                _ => {}
            }
        }
    }
    let seq_time = seq_start.elapsed();
    println!("    Sequential (10 frames): {:.2} ms ({:.1} fps)", 
             seq_time.as_secs_f64() * 1000.0,
             10.0 / seq_time.as_secs_f64());
    
    // Parallel processing test with progress tracking
    let par_start = Instant::now();
    let processed_count = Arc::new(AtomicUsize::new(0));
    let ms1_count = Arc::new(AtomicUsize::new(0));
    let ms2_count = Arc::new(AtomicUsize::new(0));
    let total_peaks = Arc::new(AtomicUsize::new(0));
    let failed_count = Arc::new(AtomicUsize::new(0));
    
    let chunk_size = (frames_to_process / rayon::current_num_threads()).max(10);
    
    let _frame_results: Vec<Result<(usize, usize, String), String>> = (start_frame..frames_to_process)
        .into_par_iter()
        .with_min_len(chunk_size)
        .map(|idx| {
            let frame_start = Instant::now();
            
            match frames.get(idx) {
                Ok(frame) => {
                    let read_time = frame_start.elapsed();
                    let process_start = Instant::now();
                    
                    let _rt_min = frame.rt_in_seconds as f32 / 60.0;
                    let peak_count = frame.tof_indices.len();
                    let scan_count = frame.scan_offsets.len() - 1;
                    
                    // Simulate processing
                    let mut processed_peaks = 0;
                    match frame.ms_level {
                        MSLevel::MS1 => {
                            ms1_count.fetch_add(1, Ordering::Relaxed);
                            // Simulate MS1 processing
                            for &tof in frame.tof_indices.iter() {
                                let _mz = mz_cv.convert(tof as f64) as f32;
                                processed_peaks += 1;
                            }
                        },
                        MSLevel::MS2 => {
                            ms2_count.fetch_add(1, Ordering::Relaxed);
                            // Simulate MS2 processing
                            let qs = &frame.quadrupole_settings;
                            for _win in 0..qs.isolation_mz.len() {
                                processed_peaks += frame.tof_indices.len() / qs.isolation_mz.len().max(1);
                            }
                        },
                        _ => {}
                    }
                    
                    total_peaks.fetch_add(peak_count, Ordering::Relaxed);
                    let count = processed_count.fetch_add(1, Ordering::Relaxed) + 1;
                    
                    let process_time = process_start.elapsed();
                    
                    // Log detailed frame stats
                    logger.log_frame(FrameStats {
                        index: idx,
                        ms_level: format!("{:?}", frame.ms_level),
                        rt_seconds: frame.rt_in_seconds,
                        num_peaks: peak_count,
                        num_scans: scan_count,
                        read_time_ms: read_time.as_secs_f64() * 1000.0,
                        process_time_ms: process_time.as_secs_f64() * 1000.0,
                    });
                    
                    // Print progress every 100 frames
                    if count % 100 == 0 {
                        println!("    Processed {} / {} frames", count, frames_to_process - start_frame);
                    }
                    
                    Ok((peak_count, scan_count, format!("{:?}", frame.ms_level)))
                },
                Err(e) => {
                    failed_count.fetch_add(1, Ordering::Relaxed);
                    Err(format!("Frame {} error: {}", idx, e))
                }
            }
        })
        .collect();
    
    let par_time = par_start.elapsed();
    let successful = processed_count.load(Ordering::Relaxed);
    let failed = failed_count.load(Ordering::Relaxed);
    
    println!("\n  {} Parallel processing complete", "✓".green());
    println!("    Time:          {:.2} s", par_time.as_secs_f64());
    println!("    Throughput:    {:.1} frames/sec", successful as f64 / par_time.as_secs_f64());
    println!("    MS1 frames:    {}", ms1_count.load(Ordering::Relaxed));
    println!("    MS2 frames:    {}", ms2_count.load(Ordering::Relaxed));
    println!("    Total peaks:   {}", total_peaks.load(Ordering::Relaxed));
    println!("    Failed frames: {}", failed);
    
    logger.log_metric(PerformanceMetrics {
        stage: "parallel_frame_processing".to_string(),
        duration_ms: par_time.as_secs_f64() * 1000.0,
        frames_processed: Some(successful),
        throughput_fps: Some(successful as f64 / par_time.as_secs_f64()),
        memory_used_mb: None,
    });
    
    // Compare sequential vs parallel
    if seq_time.as_secs_f64() > 0.0 {
        let speedup = (seq_time.as_secs_f64() * (successful as f64 / 10.0)) / par_time.as_secs_f64();
        println!("\n  Parallel speedup: {:.2}x", speedup);
    }
    
    Ok(())
}

fn print_summary(logger: &DiagnosticLogger) {
    println!("\n{}", "Performance Summary".bold().green());
    println!("{}", "═".repeat(80).green());
    
    let metrics = logger.metrics.lock().unwrap();
    let frame_stats = logger.frame_stats.lock().unwrap();
    
    // Group metrics by stage
    let mut stage_times: HashMap<String, Vec<f64>> = HashMap::new();
    for metric in metrics.iter() {
        stage_times.entry(metric.stage.clone())
            .or_insert_with(Vec::new)
            .push(metric.duration_ms);
    }
    
    println!("\n{}", "Stage Timings:".bold());
    for (stage, times) in stage_times.iter() {
        let avg = times.iter().sum::<f64>() / times.len() as f64;
        let min = times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        println!("  {:30} avg: {:8.2} ms, min: {:8.2} ms, max: {:8.2} ms", 
                 stage, avg, min, max);
    }
    
    // Frame statistics
    if !frame_stats.is_empty() {
        println!("\n{}", "Frame Statistics:".bold());
        
        let avg_read = frame_stats.iter().map(|s| s.read_time_ms).sum::<f64>() / frame_stats.len() as f64;
        let avg_process = frame_stats.iter().map(|s| s.process_time_ms).sum::<f64>() / frame_stats.len() as f64;
        let avg_peaks = frame_stats.iter().map(|s| s.num_peaks).sum::<usize>() / frame_stats.len();
        
        println!("  Average read time:    {:.2} ms", avg_read);
        println!("  Average process time: {:.2} ms", avg_process);
        println!("  Average peaks/frame:  {}", avg_peaks);
    }
}

// Updated main function with local storage optimization
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    
    // Set thread pool if specified
    if let Some(threads) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()?;
    }
    
    let d_path = Path::new(&args.d_folder);
    if !d_path.exists() {
        eprintln!("{} Folder {:?} not found", "Error:".red().bold(), d_path);
        std::process::exit(1);
    }
    
    // Initialize local storage manager
    let mut storage_manager = LocalStorageManager::new(d_path);
    
    // Determine if we should use local storage
    let working_path = if storage_manager.should_use_local(args.force_local, args.no_local) {
        println!("{} Detected network storage, optimizing with local copy", "→".blue());
        match storage_manager.copy_to_local() {
            Ok(path) => path,
            Err(e) => {
                eprintln!("{} Failed to copy to local storage: {}", "⚠".yellow(), e);
                eprintln!("  Falling back to original path");
                d_path.to_path_buf()
            }
        }
    } else {
        println!("{} Using original path (local storage or fast network)", "→".blue());
        d_path.to_path_buf()
    };
    
    print_header(&working_path);
    
    let system_info = get_system_info();
    print_system_info(&system_info);
    
    let logger = DiagnosticLogger::new(args.verbose);
    
    // Save system info
    let report_name = format!("diagnostic_{}_{}.json", 
                              system_info.hostname,
                              chrono::Local::now().format("%Y%m%d_%H%M%S"));
    
    // Run diagnostics with the working path
    test_io_performance(&working_path, &logger)?;
    test_parallel_overhead(&logger);
    read_timstof_diagnostic(&working_path, args.max_frames, &logger)?;
    
    print_summary(&logger);
    
    // Save detailed report
    logger.save_report(&report_name)?;
    println!("\n{} Detailed report saved to: {}", "✓".green().bold(), report_name);
    
    println!("\n{}", "═".repeat(80).blue());
    println!("{}", "Diagnostic Complete".bold().blue());
    println!("{}", "═".repeat(80).blue());
    
    // Cleanup happens automatically when storage_manager is dropped
    
    Ok(())
}