mod utils;
mod processing;

use clap::Parser;
use std::sync::{Arc, Mutex};
use std::io::BufWriter;

use utils::{
    read_timstof_data, build_indexed_data, read_parquet_with_polars,
    library_records_to_dataframe, merge_library_and_report, get_unique_precursor_ids, 
    process_library_fast, create_rt_im_dicts, build_lib_matrix, build_precursors_matrix_step1, 
    build_precursors_matrix_step2, build_range_matrix_step3, build_precursors_matrix_step3, 
    build_frag_info, LibCols, PrecursorLibData, prepare_precursor_lib_data,
    extract_unique_rt_im_values, save_unique_values_to_files, UniqueValues
};
use processing::{
    FastChunkFinder, build_rt_intensity_matrix_optimized, prepare_precursor_features,
    calculate_mz_range, extract_ms2_data, build_mask_matrices, extract_aligned_rt_values,
    reshape_and_combine_matrices, process_single_precursor_compressed
};

use std::io::Write;
use rayon::prelude::*;
use std::{error::Error, path::Path, time::Instant, env, fs::File};
use ndarray::{Array1, Array2, Array3, Array4, s, Axis};
use polars::prelude::*;
use ndarray_npy::{NpzWriter, write_npy};

/// A Rust program for processing TimsTOF data
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to TimsTOF .d folder
    #[arg(short = 'd', long)]
    d_folder: String,

    /// Path to fitted IM report parquet file
    #[arg(short = 'r', long)]
    report_file_path: String,

    /// Path to library file (e.g., .tsv)
    #[arg(short = 'l', long)]
    lib_file_path: String,
    
    /// Output directory name
    #[arg(short = 'o', long, default_value = "output")]
    output_dir: String,

    /// Number of parallel threads
    #[arg(short = 't', long, default_value_t = 32)]
    parallel_threads: usize,
    
    /// Maximum number of precursors
    #[arg(short = 'n', long, default_value_t = 50000)]
    max_precursors: usize,
}

#[derive(Debug)]
pub struct CompressedPrecursorResults {
    pub index: usize,
    pub precursor_id: String,
    pub rt_counts: Array1<f32>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let max_precursors = args.max_precursors;
    let parallel_threads = args.parallel_threads;
    let output_dir = &args.output_dir;
    let d_folder = &args.d_folder;
    let report_file_path = &args.report_file_path;
    let lib_file_path = &args.lib_file_path;
    
    std::fs::create_dir_all(output_dir)?;
    
    // OPTIMIZED: Configure thread pool with better work stealing
    rayon::ThreadPoolBuilder::new()
        .num_threads(parallel_threads)
        .thread_name(|i| format!("worker-{}", i))
        .build_global()
        .unwrap();
    
    println!("Initialized parallel processing with {} threads", parallel_threads);
    
    let d_path = Path::new(&d_folder);
    if !d_path.exists() {
        return Err(format!("folder {:?} not found", d_path).into());
    }
    
    // ================================ DATA LOADING AND INDEXING ================================
    println!("\n========== DATA PREPARATION PHASE ==========");
    let total_start = Instant::now();
    
    println!("Reading TimsTOF data from raw files...");
    let raw_data_start = Instant::now();
    let raw_data = read_timstof_data(d_path)?;
    println!("Raw data reading time: {:.5} seconds", raw_data_start.elapsed().as_secs_f32());
    println!("  - MS1 data points: {}", raw_data.ms1_data.mz_values.len());
    println!("  - MS2 windows: {}", raw_data.ms2_windows.len());

    println!("\nExtracting unique RT values...");
    let extract_start = Instant::now();
    let unique_values = extract_unique_rt_im_values(&raw_data);
    println!("Extraction time: {:.5} seconds", extract_start.elapsed().as_secs_f32());
    
    println!("\n========== UNIQUE VALUE STATISTICS ==========");
    println!("Unique MS2 RT values: {}", unique_values.ms2_rt_values.len());
    
    println!("\nSaving unique values to '{}'...", output_dir);
    save_unique_values_to_files(&unique_values, output_dir)?;
    println!("Unique values saved successfully.");
    
    let ms2_rt_values = unique_values.ms2_rt_values;
    
    // OPTIMIZED: Parallel index building
    println!("\nBuilding indexed data structures...");
    let index_start = Instant::now();
    let (ms1_indexed, ms2_indexed_pairs) = build_indexed_data(raw_data)?;
    println!("Index building time: {:.5} seconds", index_start.elapsed().as_secs_f32());
    
    println!("Total data preparation time: {:.5} seconds", total_start.elapsed().as_secs_f32());
    
    // OPTIMIZED: Use parallel chunk finder construction
    let finder = FastChunkFinder::new_parallel(ms2_indexed_pairs)?;
    
    // ================================ LIBRARY AND REPORT LOADING ================================
    println!("\n========== LIBRARY AND REPORT PROCESSING ==========");
    let lib_processing_start = Instant::now();

    let library_records = process_library_fast(lib_file_path)?;
    let library_df = library_records_to_dataframe(library_records.clone())?;
    let report_df = read_parquet_with_polars(report_file_path)?;
    let diann_result = merge_library_and_report(library_df, report_df)?;
    
    let diann_precursor_id_all = get_unique_precursor_ids(&diann_result)?;
    let (assay_rt_kept_dict, assay_im_kept_dict) = create_rt_im_dicts(&diann_precursor_id_all)?;
    
    println!("Library and report processing time: {:.5} seconds", lib_processing_start.elapsed().as_secs_f32());
    
    let device = "cpu";
    let frag_repeat_num = 5;
    
    // ================================ BATCH PRECURSOR PROCESSING ================================
    println!("\n========== BATCH PRECURSOR PROCESSING ==========");
    
    println!("\n[Step 1] Preparing library data for batch processing");
    let prep_start = Instant::now();
    
    let unique_precursor_ids: Vec<String> = diann_precursor_id_all
        .column("transition_group_id")?
        .str()?
        .into_iter()
        .filter_map(|opt| opt.map(|s| s.to_string()))
        .collect();

    let total_unique_precursors = unique_precursor_ids.len();
    println!("\n========== LIBRARY STATISTICS ==========");
    println!("Total unique precursor IDs in library: {}", total_unique_precursors);
    
    let lib_cols = LibCols::default();
    
    let precursor_lib_data_list = prepare_precursor_lib_data(
        &library_records,
        &unique_precursor_ids,
        &assay_rt_kept_dict,
        &assay_im_kept_dict,
        &lib_cols,
        max_precursors,
    )?;
    
    println!("  - Prepared data for {} precursors", precursor_lib_data_list.len());
    println!("  - Preparation time: {:.5} seconds", prep_start.elapsed().as_secs_f32());
    
    drop(library_records);
    println!("  - Released library_records from memory");
    
    println!("\n[Step 2] Processing individual precursors");
    
    let batch_start = Instant::now();
    
    use std::sync::atomic::{AtomicUsize, Ordering};
    let processed_count = Arc::new(AtomicUsize::new(0));
    let total_count = precursor_lib_data_list.len();
    
    // OPTIMIZED: Use crossbeam channel for better result collection
    let (tx, rx) = std::sync::mpsc::channel();
    
    // OPTIMIZED: Process with work-stealing and chunking
    let chunk_size = (total_count / (parallel_threads * 4)).max(1);
    
    precursor_lib_data_list
        .par_iter()
        .enumerate()
        .with_min_len(chunk_size)
        .for_each_with(tx.clone(), |tx, (original_index, precursor_data)| {
            let result = process_single_precursor_compressed(
                precursor_data,
                &ms1_indexed,
                &finder,
                frag_repeat_num,
                device,
                &ms2_rt_values,
            );
            
            let current = processed_count.fetch_add(1, Ordering::Relaxed) + 1;
            if current % 1000 == 0 || current == total_count {
                let percentage = (current as f64 / total_count as f64) * 100.0;
                println!(
                    "[Progress] Processed {}/{} precursors ({:.1}%)",
                    current,
                    total_count,
                    percentage
                );
            }
            
            match result {
                Ok((precursor_id, rt_counts)) => {
                    let compressed_result = CompressedPrecursorResults {
                        index: original_index,
                        precursor_id: precursor_id.clone(),
                        rt_counts,
                    };
                    let _ = tx.send(compressed_result);
                },
                Err(e) => {
                    eprintln!("[{}/{}] ✗ Error processing {} (index: {}): {}", 
                              current, total_count, precursor_data.precursor_id, original_index, e);
                }
            }
        });
    
    drop(tx);
    
    // Collect results
    let mut results: Vec<CompressedPrecursorResults> = rx.into_iter().collect();
    
    let batch_elapsed = batch_start.elapsed();
    
    println!("\n========== SAVING RESULTS TO SEPARATE FILES ==========");
    let save_start = Instant::now();
    
    // OPTIMIZED: Parallel sort
    results.par_sort_unstable_by_key(|r| r.index);
    
    // Save results
    save_results_as_separate_files(&results, &precursor_lib_data_list, output_dir)?;
    
    println!("Save time: {:.5} seconds", save_start.elapsed().as_secs_f32());
    
    println!("\n========== PROCESSING SUMMARY ==========");
    println!("Total unique precursor IDs in library: {}", total_unique_precursors);
    println!("Successfully processed: {} precursors", results.len());
    println!("Processing mode: Parallel ({} threads)", parallel_threads);
    println!("Total batch processing time: {:.5} seconds", batch_elapsed.as_secs_f32());
    println!("Average time per precursor: {:.5} seconds", 
             batch_elapsed.as_secs_f32() / precursor_lib_data_list.len() as f32);
    
    Ok(())
}

// The function should look like this (without the duplicate import):
fn save_results_as_separate_files(
    results: &[CompressedPrecursorResults], 
    original_precursor_list: &[PrecursorLibData],
    output_dir: &str,
) -> Result<(), Box<dyn Error>> {
    
    if results.is_empty() {
        return Err("No results to save".into());
    }
    
    let rt_len = results[0].rt_counts.len();
    
    // Verify consistency
    for result in results {
        if result.rt_counts.len() != rt_len {
            return Err("Inconsistent RT count lengths across precursors".into());
        }
    }
    
    // Create result map
    let result_map: std::collections::HashMap<usize, &CompressedPrecursorResults> = 
        results.iter().map(|r| (r.index, r)).collect();
    
    let n_original = original_precursor_list.len();
    let mut all_rt_matrix = Array2::<f32>::zeros((n_original, rt_len));
    let mut precursor_ids = Vec::with_capacity(n_original);
    let mut status_list = Vec::with_capacity(n_original);
    
    // OPTIMIZED: Parallel matrix filling for large datasets
    if n_original > 1000 {
        let chunk_size = (n_original / rayon::current_num_threads()).max(100);
        
        all_rt_matrix.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .with_min_len(chunk_size)
            .for_each(|(i, mut row)| {
                if let Some(result) = result_map.get(&i) {
                    row.assign(&result.rt_counts);
                }
            });
        
        for (i, precursor_data) in original_precursor_list.iter().enumerate() {
            precursor_ids.push(precursor_data.precursor_id.clone());
            status_list.push(if result_map.contains_key(&i) { "SUCCESS" } else { "FAILED" });
        }
    } else {
        // Sequential for small datasets
        for (i, precursor_data) in original_precursor_list.iter().enumerate() {
            precursor_ids.push(precursor_data.precursor_id.clone());
            
            if let Some(result) = result_map.get(&i) {
                all_rt_matrix.slice_mut(s![i, ..]).assign(&result.rt_counts);
                status_list.push("SUCCESS");
            } else {
                status_list.push("FAILED");
            }
        }
    }
    
    // Save RT matrix
    let npy_file_path = Path::new(output_dir).join("all_rt_matrix.npy");
    println!("Saving RT matrix to: {:?}", npy_file_path);
    write_npy(npy_file_path, &all_rt_matrix)?;

    // Save precursor IDs with status
    let id_file_path = Path::new(output_dir).join("precursor_ids_RT.txt");
    println!("Saving precursor IDs to: {:?}", id_file_path);
    
    let mut id_file = BufWriter::new(File::create(id_file_path)?);
    
    writeln!(id_file, "# Precursor IDs corresponding to rows in all_rt_matrix.npy")?;
    writeln!(id_file, "# Total precursors: {}", n_original)?;
    writeln!(id_file, "# Successfully processed: {}", results.len())?;
    writeln!(id_file, "# Failed: {}", n_original - results.len())?;
    writeln!(id_file, "# RT matrix shape: ({}, {})", n_original, rt_len)?;
    writeln!(id_file, "# Row_Index\tPrecursor_ID\tStatus")?;
    
    for (i, (id, status)) in precursor_ids.iter().zip(status_list.iter()).enumerate() {
        writeln!(id_file, "{}\t{}\t{}", i, id, status)?;
    }
    
    println!("\nSuccessfully saved:");
    println!("  - RT matrix: all_rt_matrix.npy (shape: {} x {})", n_original, rt_len);
    println!("  - Precursor IDs: precursor_ids_RT.txt ({} entries)", n_original);
    println!("  - Successfully processed: {} precursors", results.len());
    println!("  - Failed: {} precursors", n_original - results.len());
    
    Ok(())
}

// Add optimized FastChunkFinder implementation
impl FastChunkFinder {
    pub fn new_parallel(mut pairs: Vec<((f32, f32), utils::IndexedTimsTOFData)>) -> Result<Self, Box<dyn Error>> {
        if pairs.is_empty() { 
            return Err("no MS2 windows collected".into()); 
        }
        
        // OPTIMIZED: Parallel sort for large datasets
        if pairs.len() > 100 {
            pairs.par_sort_unstable_by(|a, b| {
                a.0.0.partial_cmp(&b.0.0).unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            pairs.sort_by(|a, b| a.0.0.partial_cmp(&b.0.0).unwrap());
        }
        
        let n = pairs.len();
        
        // OPTIMIZED: Parallel extraction
        let (low_bounds, high_bounds): (Vec<f32>, Vec<f32>) = pairs
            .par_iter()
            .map(|((l, h), _)| (*l, *h))
            .unzip();
        
        let chunks: Vec<utils::IndexedTimsTOFData> = pairs
            .into_par_iter()
            .map(|(_, data)| data)
            .collect();
        
        Ok(Self { low_bounds, high_bounds, chunks })
    }
}