// ============================================================================
// File: utils.rs - Optimized for parallel index building
// ============================================================================

use std::collections::{HashMap, HashSet};
use ndarray::{Array2, Array3, s};
use std::cmp::Ordering;
use std::error::Error;
use polars::prelude::*;
use std::fs::File;
use rayon::prelude::*;
use csv::ReaderBuilder;
use std::path::Path;
use std::time::Instant;
use timsrust::{converters::ConvertableDomain, readers::{FrameReader, MetadataReader}, MSLevel};
use serde::{Serialize, Deserialize};
use chrono::Local;
use std::sync::Arc;

// ============================================================================
// OPTIMIZED: PrecursorLibData and library data processing
// ============================================================================

#[derive(Debug, Clone)]
pub struct PrecursorLibData {
    pub precursor_id: String,
    pub im: f32,
    pub rt: f32,
    pub lib_records: Vec<LibraryRecord>,
    pub ms1_data: MSDataArray,
    pub ms2_data: MSDataArray,
    pub precursor_info: Vec<f32>,
}

// OPTIMIZED: Parallel precursor data preparation with better memory layout
pub fn prepare_precursor_lib_data(
    library_records: &[LibraryRecord],
    diann_precursor_ids: &[String],
    assay_rt_dict: &HashMap<String, f32>,
    assay_im_dict: &HashMap<String, f32>,
    lib_cols: &LibCols,
    max_precursors: usize,
) -> Result<Vec<PrecursorLibData>, Box<dyn Error>> {
    let start_time = Instant::now();
    
    // Pre-allocate HashMap with capacity
    let mut lib_data_map: HashMap<String, Vec<LibraryRecord>> = 
        HashMap::with_capacity(diann_precursor_ids.len());
    
    // Single pass grouping
    for record in library_records {
        lib_data_map
            .entry(record.transition_group_id.clone())
            .or_insert_with(Vec::new)
            .push(record.clone());
    }
    
    let unique_precursors: Vec<String> = diann_precursor_ids
        .iter()
        .take(max_precursors)
        .cloned()
        .collect();
    
    // OPTIMIZED: Parallel processing with chunk-based work distribution
    let chunk_size = (unique_precursors.len() / rayon::current_num_threads()).max(10);
    let precursor_data_list: Vec<PrecursorLibData> = unique_precursors
        .par_chunks(chunk_size)
        .flat_map(|chunk| {
            chunk.iter().filter_map(|precursor_id| {
                lib_data_map.get(precursor_id).and_then(|each_lib_data| {
                    if each_lib_data.is_empty() {
                        return None;
                    }
                    
                    let rt = *assay_rt_dict.get(precursor_id).unwrap_or(&0.0);
                    let im = *assay_im_dict.get(precursor_id).unwrap_or(&0.0);
                    
                    build_lib_matrix(each_lib_data, lib_cols, 5.0, 1801.0, 20)
                        .ok()
                        .and_then(|(precursors_list, ms1_data_list, ms2_data_list, precursor_info_list)| {
                            if !precursors_list.is_empty() {
                                Some(PrecursorLibData {
                                    precursor_id: precursor_id.clone(),
                                    im,
                                    rt,
                                    lib_records: each_lib_data.clone(),
                                    ms1_data: ms1_data_list[0].clone(),
                                    ms2_data: ms2_data_list[0].clone(),
                                    precursor_info: precursor_info_list[0].clone(),
                                })
                            } else {
                                None
                            }
                        })
                })
            }).collect::<Vec<_>>()
        })
        .collect();
    
    println!("Prepared {} precursors in {:.3}s", 
             precursor_data_list.len(), 
             start_time.elapsed().as_secs_f32());
    
    Ok(precursor_data_list)
}

// ============================================================================
// OPTIMIZED: TimsTOF data reading with parallel processing
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimsTOFRawData {
    pub ms1_data: TimsTOFData,
    pub ms2_windows: Vec<((f32, f32), TimsTOFData)>,
}

// Fix for error at line 134 - cannot borrow `lookup` as mutable in Fn closure
// Replace the build_scan_lookup function with this:
#[inline(always)]
pub fn build_scan_lookup(scan_offsets: &[usize]) -> Vec<u32> {
    if scan_offsets.is_empty() {
        return Vec::new();
    }
    
    let max_index = *scan_offsets.last().unwrap_or(&0);
    let mut lookup = vec![0u32; max_index];
    
    // Fix: Use sequential iteration instead of parallel for mutable access
    for (scan, window) in scan_offsets.windows(2).enumerate() {
        let start = window[0];
        let end = window[1];
        for idx in start..end {
            if idx < lookup.len() {
                lookup[idx] = scan as u32;
            }
        }
    }
    
    if scan_offsets.len() > 1 {
        let last_scan = (scan_offsets.len() - 2) as u32;
        let last_start = scan_offsets[scan_offsets.len() - 1];
        for idx in last_start..max_index {
            if idx < lookup.len() {
                lookup[idx] = last_scan;
            }
        }
    }
    
    lookup
}

// OPTIMIZED: Main data reading function with better parallelization
pub fn read_timstof_data(d_folder: &Path) -> Result<TimsTOFRawData, Box<dyn Error>> {
    let start_time = Instant::now();
    println!("Reading TimsTOF data from: {:?}", d_folder);
    
    let tdf_path = d_folder.join("analysis.tdf");
    let meta = MetadataReader::new(&tdf_path)?;
    let mz_cv = Arc::new(meta.mz_converter);
    let im_cv = Arc::new(meta.im_converter);
    
    let frames = FrameReader::new(d_folder)?;
    let n_frames = frames.len();
    println!("Processing {} frames in parallel...", n_frames);
    
    // OPTIMIZED: Process frames in parallel with better work distribution
    let chunk_size = (n_frames / rayon::current_num_threads()).max(10);
    let splits: Vec<FrameSplit> = (0..n_frames)
        .into_par_iter()
        .with_min_len(chunk_size)
        .map(|idx| {
            let frame = frames.get(idx).expect("frame read");
            let rt_min = frame.rt_in_seconds as f32 / 60.0;
            let mut ms1 = TimsTOFData::new();
            let mut ms2_pairs: Vec<((u32,u32), TimsTOFData)> = Vec::new();
            
            match frame.ms_level {
                MSLevel::MS1 => {
                    let n_peaks = frame.tof_indices.len();
                    ms1 = TimsTOFData::with_capacity(n_peaks);
                    
                    let scan_lookup = build_scan_lookup(&frame.scan_offsets);
                    let max_scan = (frame.scan_offsets.len() - 1) as u32;
                    
                    // OPTIMIZED: Vectorized processing
                    for (p_idx, (&tof, &intensity)) in frame.tof_indices.iter()
                        .zip(frame.intensities.iter())
                        .enumerate() 
                    {
                        let mz = mz_cv.convert(tof as f64) as f32;
                        let scan = if p_idx < scan_lookup.len() {
                            unsafe { *scan_lookup.get_unchecked(p_idx) }
                        } else {
                            max_scan
                        };
                        let im = im_cv.convert(scan as f64) as f32;
                        
                        ms1.rt_values_min.push(rt_min);
                        ms1.mobility_values.push(im);
                        ms1.mz_values.push(mz);
                        ms1.intensity_values.push(intensity);
                        ms1.frame_indices.push(frame.index as u32);
                        ms1.scan_indices.push(scan);
                    }
                }
                MSLevel::MS2 => {
                    let qs = &frame.quadrupole_settings;
                    ms2_pairs.reserve(qs.isolation_mz.len());
                    
                    let scan_lookup = build_scan_lookup(&frame.scan_offsets);
                    let max_scan = (frame.scan_offsets.len() - 1) as u32;
                    
                    for win in 0..qs.isolation_mz.len() {
                        if win >= qs.isolation_width.len() { break; }
                        
                        let prec_mz = qs.isolation_mz[win] as f32;
                        let width = qs.isolation_width[win] as f32;
                        let low = prec_mz - width * 0.5;
                        let high = prec_mz + width * 0.5;
                        let key = (quantize(low), quantize(high));
                        
                        let win_start = qs.scan_starts[win] as u32;
                        let win_end = qs.scan_ends[win] as u32;
                        
                        let estimated_peaks = frame.tof_indices.len() / qs.isolation_mz.len().max(1);
                        let mut td = TimsTOFData::with_capacity(estimated_peaks);
                        
                        for (p_idx, (&tof, &intensity)) in frame.tof_indices.iter()
                            .zip(frame.intensities.iter())
                            .enumerate() 
                        {
                            let scan = if p_idx < scan_lookup.len() {
                                unsafe { *scan_lookup.get_unchecked(p_idx) }
                            } else {
                                max_scan
                            };
                            
                            if scan < win_start || scan > win_end { 
                                continue; 
                            }
                            
                            let mz = mz_cv.convert(tof as f64) as f32;
                            let im = im_cv.convert(scan as f64) as f32;
                            
                            td.rt_values_min.push(rt_min);
                            td.mobility_values.push(im);
                            td.mz_values.push(mz);
                            td.intensity_values.push(intensity);
                            td.frame_indices.push(frame.index as u32);
                            td.scan_indices.push(scan);
                        }
                        
                        if !td.mz_values.is_empty() {
                            ms2_pairs.push((key, td));
                        }
                    }
                }
                _ => {}
            }
            FrameSplit { ms1, ms2: ms2_pairs }
        })
        .collect();
    
    // OPTIMIZED: Parallel merging with pre-calculated capacity
    let ms1_total_size: usize = splits.par_iter()
        .map(|s| s.ms1.mz_values.len())
        .sum();
    
    let mut global_ms1 = TimsTOFData::with_capacity(ms1_total_size);
    for split in &splits {
        if !split.ms1.mz_values.is_empty() {
            global_ms1.merge_from_ref(&split.ms1);
        }
    }
    
    // OPTIMIZED: Parallel MS2 merging with better memory layout
    let mut window_sizes: HashMap<(u32, u32), usize> = HashMap::with_capacity(64);
    for split in &splits {
        for (key, td) in &split.ms2 {
            *window_sizes.entry(*key).or_insert(0) += td.mz_values.len();
        }
    }
    
    let mut ms2_hash: HashMap<(u32,u32), TimsTOFData> = 
        HashMap::with_capacity(window_sizes.len());
    for (key, size) in window_sizes {
        ms2_hash.insert(key, TimsTOFData::with_capacity(size));
    }
    
    for split in splits {
        for (key, td) in split.ms2 {
            if let Some(entry) = ms2_hash.get_mut(&key) {
                entry.merge_from(td);
            }
        }
    }
    
    let mut ms2_vec: Vec<((f32, f32), TimsTOFData)> = Vec::with_capacity(ms2_hash.len());
    for ((q_low, q_high), td) in ms2_hash {
        let low = q_low as f32 / 10_000.0;
        let high = q_high as f32 / 10_000.0;
        ms2_vec.push(((low, high), td));
    }
    
    ms2_vec.sort_unstable_by_key(|((low, high), _)| 
        ((low * 10000.0) as u32, (high * 10000.0) as u32));
    
    println!("Total processing time: {:.2} seconds", start_time.elapsed().as_secs_f32());
    
    Ok(TimsTOFRawData {
        ms1_data: global_ms1,
        ms2_windows: ms2_vec,
    })
}

// ============================================================================
// OPTIMIZED: IndexedTimsTOFData with parallel sorting and indexing
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedTimsTOFData {
    pub rt_values_min: Vec<f32>,
    pub mobility_values: Vec<f32>,
    pub mz_values: Vec<f32>,
    pub intensity_values: Vec<u32>,
    pub frame_indices: Vec<u32>,
    pub scan_indices: Vec<u32>,
}

impl IndexedTimsTOFData {
    pub fn new() -> Self {
        Self {
            rt_values_min: Vec::new(),
            mobility_values: Vec::new(),
            mz_values: Vec::new(),
            intensity_values: Vec::new(),
            frame_indices: Vec::new(),
            scan_indices: Vec::new(),
        }
    }

    // OPTIMIZED: Parallel sorting and reordering
    pub fn from_timstof_data(data: TimsTOFData) -> Self {
        let n_peaks = data.mz_values.len();
        
        if n_peaks == 0 {
            return Self::new();
        }
        
        // For small datasets, use single-threaded
        if n_peaks < 50_000 {
            let mut order: Vec<usize> = (0..n_peaks).collect();
            order.sort_unstable_by(|&a, &b| {
                data.mz_values[a].partial_cmp(&data.mz_values[b]).unwrap()
            });
            
            return Self {
                rt_values_min: order.iter().map(|&i| data.rt_values_min[i]).collect(),
                mobility_values: order.iter().map(|&i| data.mobility_values[i]).collect(),
                mz_values: order.iter().map(|&i| data.mz_values[i]).collect(),
                intensity_values: order.iter().map(|&i| data.intensity_values[i]).collect(),
                frame_indices: order.iter().map(|&i| data.frame_indices[i]).collect(),
                scan_indices: order.iter().map(|&i| data.scan_indices[i]).collect(),
            };
        }
        
        // OPTIMIZED: Parallel sorting for large datasets
        let mut indices: Vec<usize> = (0..n_peaks).collect();
        indices.par_sort_unstable_by_key(|&i| {
            (data.mz_values[i] * 1_000_000.0) as u64
        });
        
        // OPTIMIZED: Parallel reordering
        Self {
            rt_values_min: indices.par_iter().map(|&i| data.rt_values_min[i]).collect(),
            mobility_values: indices.par_iter().map(|&i| data.mobility_values[i]).collect(),
            mz_values: indices.par_iter().map(|&i| data.mz_values[i]).collect(),
            intensity_values: indices.par_iter().map(|&i| data.intensity_values[i]).collect(),
            frame_indices: indices.par_iter().map(|&i| data.frame_indices[i]).collect(),
            scan_indices: indices.par_iter().map(|&i| data.scan_indices[i]).collect(),
        }
    }

    // OPTIMIZED: Binary search with bounds checking removed in hot path
    #[inline(always)]
    fn range_indices(&self, mz_min: f32, mz_max: f32) -> std::ops::Range<usize> {
        let start = self.mz_values.partition_point(|&x| x < mz_min);
        let end = self.mz_values[start..].partition_point(|&x| x <= mz_max) + start;
        start..end
    }

    // OPTIMIZED: Faster slicing with pre-allocated capacity
    pub fn slice_by_mz_range(&self, mz_min: f32, mz_max: f32) -> TimsTOFData {
        let range = self.range_indices(mz_min, mz_max);
        let cap = range.len();
        
        if cap == 0 {
            return TimsTOFData::new();
        }
        
        let mut td = TimsTOFData::with_capacity(cap);
        td.rt_values_min.extend_from_slice(&self.rt_values_min[range.clone()]);
        td.mobility_values.extend_from_slice(&self.mobility_values[range.clone()]);
        td.mz_values.extend_from_slice(&self.mz_values[range.clone()]);
        td.intensity_values.extend_from_slice(&self.intensity_values[range.clone()]);
        td.frame_indices.extend_from_slice(&self.frame_indices[range.clone()]);
        td.scan_indices.extend_from_slice(&self.scan_indices[range]);
        td
    }

    // OPTIMIZED: Parallel filtering for large ranges
    pub fn slice_by_mz_im_range(&self, mz_min: f32, mz_max: f32, im_min: f32, im_max: f32) -> TimsTOFData {
        let range = self.range_indices(mz_min, mz_max);
        
        if range.len() < 1000 {
            // Sequential for small ranges
            let indices: Vec<usize> = (range.start..range.end)
                .filter(|&i| {
                    let im = self.mobility_values[i];
                    im >= im_min && im <= im_max
                })
                .collect();
            
            let cap = indices.len();
            let mut td = TimsTOFData::with_capacity(cap);
            
            for &i in &indices {
                td.rt_values_min.push(self.rt_values_min[i]);
                td.mobility_values.push(self.mobility_values[i]);
                td.mz_values.push(self.mz_values[i]);
                td.intensity_values.push(self.intensity_values[i]);
                td.frame_indices.push(self.frame_indices[i]);
                td.scan_indices.push(self.scan_indices[i]);
            }
            
            td
        } else {
            // Parallel for large ranges
            let indices: Vec<usize> = (range.start..range.end)
                .into_par_iter()
                .filter(|&i| {
                    let im = self.mobility_values[i];
                    im >= im_min && im <= im_max
                })
                .collect();
            
            let cap = indices.len();
            
            if cap > 10_000 {
                // Parallel gathering for very large results
                TimsTOFData {
                    rt_values_min: indices.par_iter().map(|&i| self.rt_values_min[i]).collect(),
                    mobility_values: indices.par_iter().map(|&i| self.mobility_values[i]).collect(),
                    mz_values: indices.par_iter().map(|&i| self.mz_values[i]).collect(),
                    intensity_values: indices.par_iter().map(|&i| self.intensity_values[i]).collect(),
                    frame_indices: indices.par_iter().map(|&i| self.frame_indices[i]).collect(),
                    scan_indices: indices.par_iter().map(|&i| self.scan_indices[i]).collect(),
                }
            } else {
                let mut td = TimsTOFData::with_capacity(cap);
                for &i in &indices {
                    td.rt_values_min.push(self.rt_values_min[i]);
                    td.mobility_values.push(self.mobility_values[i]);
                    td.mz_values.push(self.mz_values[i]);
                    td.intensity_values.push(self.intensity_values[i]);
                    td.frame_indices.push(self.frame_indices[i]);
                    td.scan_indices.push(self.scan_indices[i]);
                }
                td
            }
        }
    }

    pub fn convert_mz_to_integer(&mut self) {
        self.mz_values.par_iter_mut().for_each(|v| *v = (*v * 1000.0).ceil());
    }

    pub fn filter_by_im_range(&self, im_min: f32, im_max: f32) -> TimsTOFData {
        self.slice_by_mz_im_range(f32::NEG_INFINITY, f32::INFINITY, im_min, im_max)
    }
}

// OPTIMIZED: Parallel index building
pub fn build_indexed_data(raw_data: TimsTOFRawData) -> Result<(IndexedTimsTOFData, Vec<((f32, f32), IndexedTimsTOFData)>), Box<dyn Error>> {
    let start = Instant::now();
    println!("Building indexed data structures...");
    
    // Process MS1 and MS2 in parallel
    let (ms1_indexed, ms2_indexed_pairs) = rayon::join(
        || IndexedTimsTOFData::from_timstof_data(raw_data.ms1_data),
        || {
            raw_data.ms2_windows
                .into_par_iter()
                .map(|((low, high), data)| {
                    ((low, high), IndexedTimsTOFData::from_timstof_data(data))
                })
                .collect::<Vec<_>>()
        }
    );
    
    println!("Index building completed in {:.3}s", start.elapsed().as_secs_f32());
    Ok((ms1_indexed, ms2_indexed_pairs))
}

// ============================================================================
// Helper functions and trait definitions
// ============================================================================

#[inline(always)]
pub fn quantize(x: f32) -> u32 { 
    (x * 10_000.0).round() as u32 
}

#[derive(Debug, Clone)]
pub struct FrameSplit {
    pub ms1: TimsTOFData,
    pub ms2: Vec<((u32, u32), TimsTOFData)>,
}

pub trait MergeFrom { 
    fn merge_from(&mut self, other: Self);
    fn merge_from_ref(&mut self, other: &Self);
}

impl MergeFrom for TimsTOFData {
    fn merge_from(&mut self, mut other: Self) {
        self.rt_values_min.append(&mut other.rt_values_min);
        self.mobility_values.append(&mut other.mobility_values);
        self.mz_values.append(&mut other.mz_values);
        self.intensity_values.append(&mut other.intensity_values);
        self.frame_indices.append(&mut other.frame_indices);
        self.scan_indices.append(&mut other.scan_indices);
    }
    
    fn merge_from_ref(&mut self, other: &Self) {
        self.rt_values_min.extend_from_slice(&other.rt_values_min);
        self.mobility_values.extend_from_slice(&other.mobility_values);
        self.mz_values.extend_from_slice(&other.mz_values);
        self.intensity_values.extend_from_slice(&other.intensity_values);
        self.frame_indices.extend_from_slice(&other.frame_indices);
        self.scan_indices.extend_from_slice(&other.scan_indices);
    }
}

// TimsTOF data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimsTOFData {
    pub rt_values_min: Vec<f32>,
    pub mobility_values: Vec<f32>,
    pub mz_values: Vec<f32>,
    pub intensity_values: Vec<u32>,
    pub frame_indices: Vec<u32>,
    pub scan_indices: Vec<u32>,
}

impl TimsTOFData {
    pub fn new() -> Self {
        TimsTOFData {
            rt_values_min: Vec::new(),
            mobility_values: Vec::new(),
            mz_values: Vec::new(),
            intensity_values: Vec::new(),
            frame_indices: Vec::new(),
            scan_indices: Vec::new(),
        }
    }
    
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            rt_values_min: Vec::with_capacity(capacity),
            mobility_values: Vec::with_capacity(capacity),
            mz_values: Vec::with_capacity(capacity),
            intensity_values: Vec::with_capacity(capacity),
            frame_indices: Vec::with_capacity(capacity),
            scan_indices: Vec::with_capacity(capacity),
        }
    }
    
    pub fn merge(data_list: Vec<TimsTOFData>) -> Self {
        if data_list.is_empty() {
            return Self::new();
        }
        
        // Pre-calculate total capacity
        let total_cap: usize = data_list.iter().map(|d| d.mz_values.len()).sum();
        let mut merged = TimsTOFData::with_capacity(total_cap);
        
        for data in data_list {
            merged.merge_from(data);
        }
        
        merged
    }
}

pub fn find_scan_for_index(index: usize, scan_offsets: &[usize]) -> usize {
    // Binary search for better performance
    match scan_offsets.binary_search(&index) {
        Ok(pos) => pos,
        Err(pos) => pos.saturating_sub(1),
    }
}

// ============================================================================
// Constants and type definitions (unchanged)
// ============================================================================

pub const MS1_ISOTOPE_COUNT: usize = 6;
pub const FRAGMENT_VARIANTS: usize = 3;
pub const MS1_TYPE_MARKER: f32 = 5.0;
pub const MS1_FRAGMENT_TYPE: f32 = 1.0;
pub const VARIANT_ORIGINAL: f32 = 2.0;
pub const VARIANT_LIGHT: f32 = 3.0;
pub const VARIANT_HEAVY: f32 = 4.0;

pub type MSDataArray = Vec<Vec<f32>>;

#[derive(Debug, Clone)]
pub struct LibCols {
    pub precursor_mz_col: &'static str,
    pub irt_col: &'static str,
    pub precursor_id_col: &'static str,
    pub full_sequence_col: &'static str,
    pub pure_sequence_col: &'static str,
    pub precursor_charge_col: &'static str,
    pub fragment_mz_col: &'static str,
    pub fragment_series_col: &'static str,
    pub fragment_charge_col: &'static str,
    pub fragment_type_col: &'static str,
    pub lib_intensity_col: &'static str,
    pub protein_name_col: &'static str,
    pub decoy_or_not_col: &'static str,
}

impl Default for LibCols {
    fn default() -> Self {
        LibCols {
            precursor_mz_col: "PrecursorMz",
            irt_col: "Tr_recalibrated",
            precursor_id_col: "transition_group_id",
            full_sequence_col: "FullUniModPeptideName",
            pure_sequence_col: "PeptideSequence",
            precursor_charge_col: "PrecursorCharge",
            fragment_mz_col: "ProductMz",
            fragment_series_col: "FragmentNumber",
            fragment_charge_col: "FragmentCharge",
            fragment_type_col: "FragmentType",
            lib_intensity_col: "LibraryIntensity",
            protein_name_col: "ProteinName",
            decoy_or_not_col: "decoy",
        }
    }
}

#[derive(Debug, Clone)]
pub struct LibraryRecord {
    pub transition_group_id: String,
    pub peptide_sequence: String,
    pub full_unimod_peptide_name: String,
    pub precursor_charge: String,
    pub precursor_mz: String,
    pub tr_recalibrated: String,
    pub precursor_ion_mobility: String,
    pub product_mz: String,
    pub fragment_type: String,
    pub fragment_charge: String,
    pub fragment_number: String,
    pub library_intensity: String,
    pub protein_id: String,
    pub protein_name: String,
    pub gene: String,
    pub decoy: String,
    pub other_columns: HashMap<String, String>,
}

// ============================================================================
// MS data processing functions (keeping original logic, minor optimizations)
// ============================================================================

pub fn intercept_frags_sort(mut fragment_list: Vec<f32>, max_length: usize) -> Vec<f32> {
    fragment_list.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
    fragment_list.truncate(max_length);
    fragment_list
}

pub fn get_precursor_indices(precursor_ids: &[String]) -> Vec<Vec<usize>> {
    let mut precursor_indices = Vec::new();
    let mut current_group = Vec::new();
    let mut last_id = "";
    
    for (idx, id) in precursor_ids.iter().enumerate() {
        if idx == 0 || id == last_id {
            current_group.push(idx);
        } else {
            if !current_group.is_empty() {
                precursor_indices.push(current_group.clone());
                current_group.clear();
            }
            current_group.push(idx);
        }
        last_id = id;
    }
    
    if !current_group.is_empty() {
        precursor_indices.push(current_group);
    }
    
    precursor_indices
}

pub fn get_lib_col_dict() -> HashMap<&'static str, &'static str> {
    let mut lib_col_dict = HashMap::with_capacity(50);
    for key in ["transition_group_id", "PrecursorID"] { lib_col_dict.insert(key, "transition_group_id"); }
    for key in ["PeptideSequence", "Sequence", "StrippedPeptide"] { lib_col_dict.insert(key, "PeptideSequence"); }
    for key in ["FullUniModPeptideName", "ModifiedPeptide", "LabeledSequence", "modification_sequence", "ModifiedPeptideSequence"] { lib_col_dict.insert(key, "FullUniModPeptideName"); }
    for key in ["PrecursorCharge", "Charge", "prec_z"] { lib_col_dict.insert(key, "PrecursorCharge"); }
    for key in ["PrecursorMz", "Q1"] { lib_col_dict.insert(key, "PrecursorMz"); }
    for key in ["Tr_recalibrated", "iRT", "RetentionTime", "NormalizedRetentionTime", "RT_detected"] { lib_col_dict.insert(key, "Tr_recalibrated"); }
    for key in ["PrecursorIonMobility", "PrecursorIM", "IonMobility", "IM"] { lib_col_dict.insert(key, "PrecursorIonMobility"); }
    for key in ["ProductMz", "FragmentMz", "Q3"] { lib_col_dict.insert(key, "ProductMz"); }
    for key in ["FragmentType", "FragmentIonType", "ProductType", "ProductIonType", "frg_type"] { lib_col_dict.insert(key, "FragmentType"); }
    for key in ["FragmentCharge", "FragmentIonCharge", "ProductCharge", "ProductIonCharge", "frg_z"] { lib_col_dict.insert(key, "FragmentCharge"); }
    for key in ["FragmentNumber", "frg_nr", "FragmentSeriesNumber"] { lib_col_dict.insert(key, "FragmentNumber"); }
    for key in ["LibraryIntensity", "RelativeIntensity", "RelativeFragmentIntensity", "RelativeFragmentIonIntensity", "relative_intensity"] { lib_col_dict.insert(key, "LibraryIntensity"); }
    for key in ["ProteinID", "ProteinId", "UniprotID", "uniprot_id", "UniProtIds"] { lib_col_dict.insert(key, "ProteinID"); }
    for key in ["ProteinName", "Protein Name", "Protein_name", "protein_name"] { lib_col_dict.insert(key, "ProteinName"); }
    for key in ["Gene", "Genes", "GeneName"] { lib_col_dict.insert(key, "Gene"); }
    for key in ["Decoy", "decoy"] { lib_col_dict.insert(key, "decoy"); }
    lib_col_dict
}

// MS1/MS2 data building functions (keeping original logic)
pub fn build_ms1_data(fragment_list: &[Vec<f32>], isotope_range: f32, max_mz: f32) -> MSDataArray {
    let first_fragment = &fragment_list[0];
    let charge = first_fragment[1];
    let precursor_mz = first_fragment[5];
    
    let available_range = (max_mz - precursor_mz) * charge;
    let iso_shift_max = (isotope_range.min(available_range) as i32) + 1;
    
    let mut isotope_mz_list: Vec<f32> = (0..iso_shift_max)
        .map(|iso_shift| precursor_mz + (iso_shift as f32) / charge)
        .collect();
    
    isotope_mz_list = intercept_frags_sort(isotope_mz_list, MS1_ISOTOPE_COUNT);
    
    let mut ms1_data = Vec::with_capacity(MS1_ISOTOPE_COUNT);
    for mz in isotope_mz_list {
        let row = vec![
            mz,
            first_fragment[1],
            first_fragment[2],
            first_fragment[3],
            3.0,
            first_fragment[5],
            MS1_TYPE_MARKER,
            0.0,
            MS1_FRAGMENT_TYPE,
        ];
        ms1_data.push(row);
    }
    
    while ms1_data.len() < MS1_ISOTOPE_COUNT {
        ms1_data.push(vec![0.0; 9]);
    }
    
    ms1_data
}

pub fn build_ms2_data(fragment_list: &[Vec<f32>], max_fragment_num: usize) -> MSDataArray {
    let total_count = max_fragment_num * FRAGMENT_VARIANTS;
    let fragment_num = fragment_list.len();
    
    let mut tripled_fragments = Vec::with_capacity(fragment_num * FRAGMENT_VARIANTS);
    for _ in 0..FRAGMENT_VARIANTS {
        for fragment in fragment_list {
            tripled_fragments.push(fragment.clone());
        }
    }
    
    let total_rows = fragment_num * FRAGMENT_VARIANTS;
    
    let mut type_column = vec![0.0; total_rows];
    for i in fragment_num..(fragment_num * 2) {
        type_column[i] = -1.0;
    }
    for i in (fragment_num * 2)..total_rows {
        type_column[i] = 1.0;
    }
    
    let window_id_column = vec![0.0; total_rows];
    
    let mut variant_type_column = vec![0.0; total_rows];
    for i in 0..fragment_num {
        variant_type_column[i] = VARIANT_ORIGINAL;
    }
    for i in fragment_num..(fragment_num * 2) {
        variant_type_column[i] = VARIANT_LIGHT;
    }
    for i in (fragment_num * 2)..total_rows {
        variant_type_column[i] = VARIANT_HEAVY;
    }
    
    let mut complete_data = Vec::with_capacity(total_count);
    for i in 0..total_rows {
        let mut row = tripled_fragments[i].clone();
        row.push(type_column[i]);
        row.push(window_id_column[i]);
        row.push(variant_type_column[i]);
        complete_data.push(row);
    }
    
    if complete_data.len() >= total_count {
        complete_data.truncate(total_count);
    } else {
        let row_size = if !complete_data.is_empty() { complete_data[0].len() } else { 9 };
        while complete_data.len() < total_count {
            complete_data.push(vec![0.0; row_size]);
        }
    }
    
    complete_data
}

pub fn build_precursor_info(fragment_list: &[Vec<f32>]) -> Vec<f32> {
    let first_fragment = &fragment_list[0];
    vec![
        first_fragment[7],
        first_fragment[5],
        first_fragment[1],
        first_fragment[6],
        fragment_list.len() as f32,
        0.0,
    ]
}

pub fn format_ms_data(
    fragment_list: &[Vec<f32>], 
    isotope_range: f32, 
    max_mz: f32, 
    max_fragment: usize
) -> (MSDataArray, MSDataArray, Vec<f32>) {
    let ms1_data = build_ms1_data(fragment_list, isotope_range, max_mz);
    
    let fragment_list_subset: Vec<Vec<f32>> = fragment_list.iter()
        .map(|row| row[..6].to_vec())
        .collect();
    
    let mut ms2_data = build_ms2_data(&fragment_list_subset, max_fragment);
    
    let mut ms1_copy = ms1_data.clone();
    for row in &mut ms1_copy {
        if row.len() > 8 {
            row[8] = 5.0;
        }
    }
    
    ms2_data.extend(ms1_copy);
    
    let precursor_info = build_precursor_info(fragment_list);
    
    (ms1_data, ms2_data, precursor_info)
}

pub fn build_lib_matrix(
    lib_data: &[LibraryRecord],
    lib_cols: &LibCols,
    iso_range: f32,
    mz_max: f32,
    max_fragment: usize,
) -> Result<(Vec<Vec<String>>, Vec<MSDataArray>, Vec<MSDataArray>, Vec<Vec<f32>>), Box<dyn Error>> {
    let precursor_ids: Vec<String> = lib_data.iter()
        .map(|record| record.transition_group_id.clone())
        .collect();
    
    let precursor_groups = get_precursor_indices(&precursor_ids);
    
    let mut all_precursors = Vec::new();
    let mut all_ms1_data = Vec::new();
    let mut all_ms2_data = Vec::new();
    let mut all_precursor_info = Vec::new();
    
    for indices in precursor_groups.iter() {
        if indices.is_empty() {
            continue;
        }
        
        let first_idx = indices[0];
        let first_record = &lib_data[first_idx];
        
        let precursor_info = vec![
            first_record.transition_group_id.clone(),
            first_record.decoy.clone(),
        ];
        all_precursors.push(precursor_info);
        
        let mut group_fragments = Vec::with_capacity(indices.len());
        for &idx in indices {
            let record = &lib_data[idx];
            
            let fragment_row = vec![
                record.product_mz.parse::<f32>().unwrap_or(0.0),
                record.precursor_charge.parse::<f32>().unwrap_or(0.0),
                record.fragment_charge.parse::<f32>().unwrap_or(0.0),
                record.library_intensity.parse::<f32>().unwrap_or(0.0),
                record.fragment_type.parse::<f32>().unwrap_or(0.0),
                record.precursor_mz.parse::<f32>().unwrap_or(0.0),
                record.tr_recalibrated.parse::<f32>().unwrap_or(0.0),
                record.peptide_sequence.len() as f32,
                record.decoy.parse::<f32>().unwrap_or(0.0),
                record.transition_group_id.len() as f32,
            ];
            group_fragments.push(fragment_row);
        }
        
        let (ms1, ms2, info) = format_ms_data(&group_fragments, iso_range, mz_max, max_fragment);
        
        all_ms1_data.push(ms1);
        all_ms2_data.push(ms2);
        all_precursor_info.push(info);
    }
    
    Ok((all_precursors, all_ms1_data, all_ms2_data, all_precursor_info))
}

// Matrix building functions (keeping original logic with minor optimizations)
pub fn build_precursors_matrix_step1(
    ms1_data_list: &[MSDataArray], 
    ms2_data_list: &[MSDataArray], 
    device: &str
) -> Result<(Array3<f32>, Array3<f32>), Box<dyn Error>> {
    if ms1_data_list.is_empty() || ms2_data_list.is_empty() {
        return Err("MS1 or MS2 data list is empty".into());
    }
    
    let batch_size = ms1_data_list.len();
    let ms1_rows = ms1_data_list[0].len();
    let ms1_cols = if !ms1_data_list[0].is_empty() { ms1_data_list[0][0].len() } else { 0 };
    let ms2_rows = ms2_data_list[0].len();
    let ms2_cols = if !ms2_data_list[0].is_empty() { ms2_data_list[0][0].len() } else { 0 };
    
    let mut ms1_tensor = Array3::<f32>::zeros((batch_size, ms1_rows, ms1_cols));
    let mut ms2_tensor = Array3::<f32>::zeros((batch_size, ms2_rows, ms2_cols));
    
    // Parallel tensor filling for large batches
    if batch_size > 10 {
        ms1_tensor.axis_iter_mut(ndarray::Axis(0))
            .into_par_iter()
            .zip(ms1_data_list.par_iter())
            .for_each(|(mut slice, data)| {
                for (j, row) in data.iter().enumerate() {
                    for (k, &val) in row.iter().enumerate() {
                        slice[[j, k]] = val;
                    }
                }
            });
        
        ms2_tensor.axis_iter_mut(ndarray::Axis(0))
            .into_par_iter()
            .zip(ms2_data_list.par_iter())
            .for_each(|(mut slice, data)| {
                for (j, row) in data.iter().enumerate() {
                    for (k, &val) in row.iter().enumerate() {
                        slice[[j, k]] = val;
                    }
                }
            });
    } else {
        for (i, ms1_data) in ms1_data_list.iter().enumerate() {
            for (j, row) in ms1_data.iter().enumerate() {
                for (k, &val) in row.iter().enumerate() {
                    ms1_tensor[[i, j, k]] = val;
                }
            }
        }
        
        for (i, ms2_data) in ms2_data_list.iter().enumerate() {
            for (j, row) in ms2_data.iter().enumerate() {
                for (k, &val) in row.iter().enumerate() {
                    ms2_tensor[[i, j, k]] = val;
                }
            }
        }
    }
    
    Ok((ms1_tensor, ms2_tensor))
}

pub fn build_precursors_matrix_step2(mut ms2_data_tensor: Array3<f32>) -> Array3<f32> {
    let shape = ms2_data_tensor.shape();
    let (batch, rows, cols) = (shape[0], shape[1], shape[2]);
    
    if cols > 6 {
        for i in 0..batch {
            for j in 0..rows {
                let val0 = ms2_data_tensor[[i, j, 0]];
                let val6 = ms2_data_tensor[[i, j, 6]];
                let val2 = ms2_data_tensor[[i, j, 2]];
                
                if val2 != 0.0 {
                    ms2_data_tensor[[i, j, 0]] = val0 + val6 / val2;
                }
            }
        }
    }
    
    // Vectorized NaN/Inf cleanup
    ms2_data_tensor.mapv_inplace(|val| {
        if val.is_infinite() || val.is_nan() { 0.0 } else { val }
    });
    
    ms2_data_tensor
}

// Keep all other matrix processing functions unchanged
pub fn extract_width_2(
    mz_to_extract: &Array3<f32>,
    mz_unit: &str,
    mz_tol: f32,
    max_extract_len: usize,
    frag_repeat_num: usize,
    max_moz_num: f32,
    device: &str
) -> Result<Array3<f32>, Box<dyn Error>> {
    let shape = mz_to_extract.shape();
    let (batch, rows, _) = (shape[0], shape[1], shape[2]);
    
    let is_all_zero = mz_to_extract.iter().all(|&v| v == 0.0);
    if is_all_zero {
        return Ok(Array3::<f32>::zeros((batch, rows, 2)));
    }
    
    let mut mz_tol_full = Array3::<f32>::zeros((batch, rows, 1));
    
    match mz_unit {
        "Da" => {
            mz_tol_full.fill(mz_tol);
        },
        "ppm" => {
            for i in 0..batch {
                for j in 0..rows {
                    mz_tol_full[[i, j, 0]] = mz_to_extract[[i, j, 0]] * mz_tol * 0.000001;
                }
            }
        },
        _ => return Err(format!("Invalid mz_unit: {}", mz_unit).into()),
    }
    
    mz_tol_full.mapv_inplace(|v| if v.is_nan() { 0.0 } else { v });
    
    let mz_tol_full_num = max_moz_num / 1000.0;
    mz_tol_full.mapv_inplace(|v| v.min(mz_tol_full_num));
    mz_tol_full.mapv_inplace(|v| ((v * 1000.0 / frag_repeat_num as f32).ceil()) * frag_repeat_num as f32);
    
    let mut extract_width_range_list = Array3::<f32>::zeros((batch, rows, 2));
    
    for i in 0..batch {
        for j in 0..rows {
            let mz_val = mz_to_extract[[i, j, 0]] * 1000.0;
            let tol_val = mz_tol_full[[i, j, 0]];
            extract_width_range_list[[i, j, 0]] = (mz_val - tol_val).floor();
            extract_width_range_list[[i, j, 1]] = (mz_val + tol_val).floor();
        }
    }
    
    Ok(extract_width_range_list)
}

// Fix for errors at lines 1086, 1094 in build_range_matrix_step3
pub fn build_range_matrix_step3(
    ms1_data_tensor: &Array3<f32>,
    ms2_data_tensor: &Array3<f32>,
    frag_repeat_num: usize,
    mz_unit: &str,
    mz_tol_ms1: f32,
    mz_tol_ms2: f32,
    device: &str
) -> Result<(Array3<f32>, Array3<f32>), Box<dyn Error>> {
    let shape1 = ms1_data_tensor.shape();
    let shape2 = ms2_data_tensor.shape();
    
    let mut re_ms1_data_tensor = Array3::<f32>::zeros((shape1[0], shape1[1] * frag_repeat_num, shape1[2]));
    let mut re_ms2_data_tensor = Array3::<f32>::zeros((shape2[0], shape2[1] * frag_repeat_num, shape2[2]));
    
    for i in 0..shape1[0] {
        for rep in 0..frag_repeat_num {
            let src_slice = ms1_data_tensor.slice(s![i, .., ..]);
            let mut dst_slice = re_ms1_data_tensor.slice_mut(s![i, rep * shape1[1]..(rep + 1) * shape1[1], ..]);  // Added mut
            dst_slice.assign(&src_slice);
        }
    }
    
    for i in 0..shape2[0] {
        for rep in 0..frag_repeat_num {
            let src_slice = ms2_data_tensor.slice(s![i, .., ..]);
            let mut dst_slice = re_ms2_data_tensor.slice_mut(s![i, rep * shape2[1]..(rep + 1) * shape2[1], ..]);  // Added mut
            dst_slice.assign(&src_slice);
        }
    }
    
    let ms1_col0 = re_ms1_data_tensor.slice(s![.., .., 0..1]).to_owned();
    let ms2_col0 = re_ms2_data_tensor.slice(s![.., .., 0..1]).to_owned();
    
    let ms1_extract_width_range_list = extract_width_2(
        &ms1_col0, mz_unit, mz_tol_ms1, 20, frag_repeat_num, 50.0, device
    )?;
    
    let ms2_extract_width_range_list = extract_width_2(
        &ms2_col0, mz_unit, mz_tol_ms2, 20, frag_repeat_num, 50.0, device
    )?;
    
    Ok((ms1_extract_width_range_list, ms2_extract_width_range_list))
}

pub fn extract_width(
    mz_to_extract: &Array3<f32>,
    mz_unit: &str,
    mz_tol: f32,
    max_extract_len: usize,
    frag_repeat_num: usize,
    max_moz_num: f32,
    device: &str
) -> Result<Array3<f32>, Box<dyn Error>> {
    let shape = mz_to_extract.shape();
    let (batch, rows, _) = (shape[0], shape[1], shape[2]);
    
    let is_all_zero = mz_to_extract.iter().all(|&v| v == 0.0);
    if is_all_zero {
        return Ok(Array3::<f32>::zeros((batch, rows, max_moz_num as usize)));
    }
    
    let mut mz_tol_half = Array3::<f32>::zeros((batch, rows, 1));
    
    match mz_unit {
        "Da" => {
            mz_tol_half.fill(mz_tol / 2.0);
        },
        "ppm" => {
            for i in 0..batch {
                for j in 0..rows {
                    mz_tol_half[[i, j, 0]] = mz_to_extract[[i, j, 0]] * mz_tol * 0.000001 / 2.0;
                }
            }
        },
        _ => return Err(format!("Invalid mz_unit: {}", mz_unit).into()),
    }
    
    mz_tol_half.mapv_inplace(|v| if v.is_nan() { 0.0 } else { v });
    
    let mz_tol_half_num = (max_moz_num / 1000.0) / 2.0;
    mz_tol_half.mapv_inplace(|v| v.min(mz_tol_half_num));
    mz_tol_half.mapv_inplace(|v| ((v * 1000.0 / frag_repeat_num as f32).ceil()) * frag_repeat_num as f32);
    
    let mut extract_width_list = Array3::<f32>::zeros((batch, rows, 2));
    
    for i in 0..batch {
        for j in 0..rows {
            let mz_val = mz_to_extract[[i, j, 0]] * 1000.0;
            let tol_val = mz_tol_half[[i, j, 0]];
            extract_width_list[[i, j, 0]] = (mz_val - tol_val).floor();
            extract_width_list[[i, j, 1]] = (mz_val + tol_val).floor();
        }
    }
    
    let batch_num = rows / frag_repeat_num;
    
    let mut cha_tensor = Array2::<f32>::zeros((batch, batch_num));
    for i in 0..batch {
        for j in 0..batch_num {
            cha_tensor[[i, j]] = (extract_width_list[[i, j, 1]] - extract_width_list[[i, j, 0]]) / frag_repeat_num as f32;
        }
    }
    
    // Optimized loop unrolling
    for i in 0..batch {
        for j in 0..batch_num {
            let base_val = extract_width_list[[i, j, 0]];
            let cha_val = cha_tensor[[i, j]];
            
            extract_width_list[[i, j, 1]] = base_val + cha_val - 1.0;
            
            for rep in 1..frag_repeat_num.min(5) {
                let idx = batch_num * rep + j;
                if idx < rows {
                    extract_width_list[[i, idx, 0]] = base_val + (rep as f32) * cha_val;
                    extract_width_list[[i, idx, 1]] = base_val + ((rep + 1) as f32) * cha_val - 1.0;
                }
            }
        }
    }
    
    let mut new_tensor = Array3::<f32>::zeros((batch, rows, max_moz_num as usize));
    
    for i in 0..batch {
        for j in 0..rows {
            let start = extract_width_list[[i, j, 0]];
            let end = extract_width_list[[i, j, 1]];
            for k in 0..(max_moz_num as usize) {
                let val = start + k as f32;
                new_tensor[[i, j, k]] = if val <= end { val } else { 0.0 };
            }
        }
    }
    
    Ok(new_tensor)
}

pub fn build_precursors_matrix_step3(
    ms1_data_tensor: &Array3<f32>,
    ms2_data_tensor: &Array3<f32>,
    frag_repeat_num: usize,
    mz_unit: &str,
    mz_tol_ms1: f32,
    mz_tol_ms2: f32,
    device: &str
) -> Result<(Array3<f32>, Array3<f32>, Array3<f32>, Array3<f32>), Box<dyn Error>> {
    let shape1 = ms1_data_tensor.shape();
    let shape2 = ms2_data_tensor.shape();
    
    let mut re_ms1_data_tensor = Array3::<f32>::zeros((shape1[0], shape1[1] * frag_repeat_num, shape1[2]));
    let mut re_ms2_data_tensor = Array3::<f32>::zeros((shape2[0], shape2[1] * frag_repeat_num, shape2[2]));
    
    for i in 0..shape1[0] {
        for rep in 0..frag_repeat_num {
            let src_slice = ms1_data_tensor.slice(s![i, .., ..]);
            let mut dst_slice = re_ms1_data_tensor.slice_mut(s![i, rep * shape1[1]..(rep + 1) * shape1[1], ..]);  // Added mut
            dst_slice.assign(&src_slice);
        }
    }
    
    for i in 0..shape2[0] {
        for rep in 0..frag_repeat_num {
            let src_slice = ms2_data_tensor.slice(s![i, .., ..]);
            let mut dst_slice = re_ms2_data_tensor.slice_mut(s![i, rep * shape2[1]..(rep + 1) * shape2[1], ..]);  // Added mut
            dst_slice.assign(&src_slice);
        }
    }
    
    let ms1_col0 = re_ms1_data_tensor.slice(s![.., .., 0..1]).to_owned();
    let ms2_col0 = re_ms2_data_tensor.slice(s![.., .., 0..1]).to_owned();
    
    let ms1_extract_width_range_list = extract_width(
        &ms1_col0, mz_unit, mz_tol_ms1, 20, frag_repeat_num, 50.0, device
    )?;
    
    let ms2_extract_width_range_list = extract_width(
        &ms2_col0, mz_unit, mz_tol_ms2, 20, frag_repeat_num, 50.0, device
    )?;
    
    Ok((re_ms1_data_tensor, re_ms2_data_tensor, ms1_extract_width_range_list, ms2_extract_width_range_list))
}

// DataFrame processing functions
pub fn read_parquet_with_polars(file_path: &str) -> PolarsResult<DataFrame> {
    let file = File::open(file_path)?;
    let mut df = ParquetReader::new(file).finish()?;
    let new_col = df.column("Precursor.Id")?.clone().with_name("transition_group_id");
    df.with_column(new_col)?;
    Ok(df)
}

pub fn library_records_to_dataframe(records: Vec<LibraryRecord>) -> PolarsResult<DataFrame> {
    let n = records.len();
    let mut transition_group_ids = Vec::with_capacity(n);
    let mut precursor_mzs = Vec::with_capacity(n);
    let mut product_mzs = Vec::with_capacity(n);
    let mut trs_recalibrated = Vec::with_capacity(n);
    let mut precursor_ion_mobilitys = Vec::with_capacity(n);
    
    for record in records {
        transition_group_ids.push(record.transition_group_id);
        precursor_mzs.push(record.precursor_mz.parse::<f32>().unwrap_or(f32::NAN));
        product_mzs.push(record.product_mz.parse::<f32>().unwrap_or(f32::NAN));
        trs_recalibrated.push(record.tr_recalibrated.parse::<f32>().unwrap_or(f32::NAN));
        precursor_ion_mobilitys.push(record.precursor_ion_mobility.parse::<f32>().unwrap_or(f32::NAN));
    }
    
    let df = DataFrame::new(vec![
        Series::new("transition_group_id", transition_group_ids),
        Series::new("PrecursorMz", precursor_mzs),
        Series::new("ProductMz", product_mzs),
        Series::new("Tr_recalibrated", trs_recalibrated),
        Series::new("PrecursorIonMobility", precursor_ion_mobilitys),
    ])?;
    Ok(df)
}

pub fn merge_library_and_report(library_df: DataFrame, report_df: DataFrame) -> PolarsResult<DataFrame> {
    let report_selected = report_df.select(["transition_group_id", "RT", "IM"])?;
    let merged = library_df.join(&report_selected, ["transition_group_id"], ["transition_group_id"], JoinArgs::new(JoinType::Left))?;
    let rt_col = merged.column("RT")?;
    let mask = rt_col.is_not_null();
    let filtered = merged.filter(&mask)?;
    let reordered = filtered.select(["transition_group_id", "PrecursorMz", "ProductMz", "RT", "IM"])?;
    Ok(reordered)
}

pub fn get_unique_precursor_ids(diann_result: &DataFrame) -> PolarsResult<DataFrame> {
    let unique_df = diann_result.unique(Some(&["transition_group_id".to_string()]), UniqueKeepStrategy::First, None)?;
    let selected_df = unique_df.select(["transition_group_id", "RT", "IM"])?;
    Ok(selected_df)
}

// OPTIMIZED: Parallel library processing
pub fn process_library_fast(file_path: &str) -> Result<Vec<LibraryRecord>, Box<dyn Error>> {
    eprintln!("Reading library file: {}", file_path);
    let file = File::open(file_path)?;
    let mut reader = ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(true)
        .from_reader(file);
    
    let headers = reader.headers()?.clone();
    let mut column_indices = HashMap::new();
    for (i, header) in headers.iter().enumerate() {
        column_indices.insert(header, i);
    }
    
    let lib_col_dict = get_lib_col_dict();
    let mut mapped_indices: HashMap<&str, usize> = HashMap::new();
    for (old_col, new_col) in &lib_col_dict {
        if let Some(&idx) = column_indices.get(old_col) {
            mapped_indices.insert(new_col, idx);
        }
    }
    
    let fragment_number_idx = column_indices.get("FragmentNumber").copied();
    
    let mut byte_records = Vec::new();
    for result in reader.byte_records() {
        byte_records.push(result?);
    }
    
    eprintln!("Processing {} library records...", byte_records.len());
    
    // OPTIMIZED: Chunked parallel processing
    let chunk_size = (byte_records.len() / rayon::current_num_threads()).max(100);
    let records: Vec<LibraryRecord> = byte_records
        .par_chunks(chunk_size)
        .flat_map(|chunk| {
            chunk.iter().map(|record| {
                let mut rec = LibraryRecord {
                    transition_group_id: String::new(),
                    peptide_sequence: String::new(),
                    full_unimod_peptide_name: String::new(),
                    precursor_charge: String::new(),
                    precursor_mz: String::new(),
                    tr_recalibrated: String::new(),
                    precursor_ion_mobility: String::new(),
                    product_mz: String::new(),
                    fragment_type: String::new(),
                    fragment_charge: String::new(),
                    fragment_number: String::new(),
                    library_intensity: String::new(),
                    protein_id: String::new(),
                    protein_name: String::new(),
                    gene: String::new(),
                    decoy: "0".to_string(),
                    other_columns: HashMap::new(),
                };
                
                // Fast field extraction
                macro_rules! set_field {
                    ($field:ident, $key:expr) => {
                        if let Some(&idx) = mapped_indices.get($key) {
                            if let Some(val) = record.get(idx) {
                                rec.$field = String::from_utf8_lossy(val).into_owned();
                            }
                        }
                    };
                }
                
                set_field!(peptide_sequence, "PeptideSequence");
                set_field!(full_unimod_peptide_name, "FullUniModPeptideName");
                set_field!(precursor_charge, "PrecursorCharge");
                set_field!(precursor_mz, "PrecursorMz");
                set_field!(product_mz, "ProductMz");
                set_field!(fragment_charge, "FragmentCharge");
                set_field!(library_intensity, "LibraryIntensity");
                set_field!(tr_recalibrated, "Tr_recalibrated");
                set_field!(precursor_ion_mobility, "PrecursorIonMobility");
                set_field!(protein_id, "ProteinID");
                set_field!(gene, "Gene");
                set_field!(protein_name, "ProteinName");
                
                if let Some(&idx) = mapped_indices.get("FragmentType") {
                    if let Some(val) = record.get(idx) {
                        let fragment_str = String::from_utf8_lossy(val);
                        rec.fragment_type = match fragment_str.as_ref() {
                            "b" => "1".to_string(),
                            "y" => "2".to_string(),
                            "p" => "3".to_string(),
                            _ => fragment_str.into_owned()
                        };
                    }
                }
                
                if let Some(idx) = fragment_number_idx {
                    if let Some(val) = record.get(idx) {
                        rec.fragment_number = String::from_utf8_lossy(val).into_owned();
                    }
                }
                
                if let Some(&idx) = mapped_indices.get("transition_group_id") {
                    if let Some(val) = record.get(idx) {
                        rec.transition_group_id = String::from_utf8_lossy(val).into_owned();
                    }
                } else {
                    rec.transition_group_id = format!("{}{}", rec.full_unimod_peptide_name, rec.precursor_charge);
                }
                
                rec
            }).collect::<Vec<_>>()
        })
        .collect();
    
    Ok(records)
}

pub fn create_rt_im_dicts(df: &DataFrame) -> PolarsResult<(HashMap<String, f32>, HashMap<String, f32>)> {
    let id_col = df.column("transition_group_id")?;
    let id_vec = id_col.str()?.into_iter()
        .map(|opt| opt.unwrap_or("").to_string())
        .collect::<Vec<String>>();
    
    let rt_col = df.column("RT")?;
    let rt_vec: Vec<f32> = match rt_col.dtype() {
        DataType::Float32 => rt_col.f32()?.into_iter()
            .map(|opt| opt.unwrap_or(f32::NAN))
            .collect(),
        DataType::Float64 => rt_col.f64()?.into_iter()
            .map(|opt| opt.map(|v| v as f32).unwrap_or(f32::NAN))
            .collect(),
        _ => return Err(PolarsError::SchemaMismatch(
            format!("RT column type is not float: {:?}", rt_col.dtype()).into()
        )),
    };
    
    let im_col = df.column("IM")?;
    let im_vec: Vec<f32> = match im_col.dtype() {
        DataType::Float32 => im_col.f32()?.into_iter()
            .map(|opt| opt.unwrap_or(f32::NAN))
            .collect(),
        DataType::Float64 => im_col.f64()?.into_iter()
            .map(|opt| opt.map(|v| v as f32).unwrap_or(f32::NAN))
            .collect(),
        _ => return Err(PolarsError::SchemaMismatch(
            format!("IM column type is not float: {:?}", im_col.dtype()).into()
        )),
    };
    
    let mut rt_dict = HashMap::with_capacity(id_vec.len());
    let mut im_dict = HashMap::with_capacity(id_vec.len());
    
    for ((id, rt), im) in id_vec.iter().zip(rt_vec.iter()).zip(im_vec.iter()) {
        rt_dict.insert(id.clone(), *rt);
        im_dict.insert(id.clone(), *im);
    }
    
    Ok((rt_dict, im_dict))
}

// Other utility functions
pub fn get_rt_list(mut lst: Vec<f32>, target: f32) -> Vec<f32> {
    lst.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    
    if lst.is_empty() {
        return vec![0.0; 48];
    }
    
    if lst.len() <= 48 {
        let mut result = lst;
        result.resize(48, 0.0);
        return result;
    }
    
    let closest_idx = lst.iter()
        .enumerate()
        .min_by_key(|(_, &val)| ((val - target).abs() * 1e9) as i32)
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    
    let start = if closest_idx >= 24 {
        (closest_idx - 24).min(lst.len() - 48)
    } else {
        0
    };
    
    lst[start..start + 48].to_vec()
}

pub fn build_ext_ms1_matrix(ms1_data_tensor: &Array3<f32>, device: &str) -> Array3<f32> {
    let shape = ms1_data_tensor.shape();
    let (batch, rows, cols) = (shape[0], shape[1], shape[2]);
    
    let mut ext_matrix = Array3::<f32>::zeros((batch, rows, 4));
    
    for i in 0..batch {
        for j in 0..rows {
            ext_matrix[[i, j, 0]] = ms1_data_tensor[[i, j, 0]];
            if cols > 3 {
                ext_matrix[[i, j, 1]] = ms1_data_tensor[[i, j, 3]];
            }
            if cols > 8 {
                ext_matrix[[i, j, 2]] = ms1_data_tensor[[i, j, 8]];
            }
            if cols > 4 {
                ext_matrix[[i, j, 3]] = ms1_data_tensor[[i, j, 4]];
            }
        }
    }
    
    ext_matrix
}

pub fn build_ext_ms2_matrix(ms2_data_tensor: &Array3<f32>, device: &str) -> Array3<f32> {
    let shape = ms2_data_tensor.shape();
    let (batch, rows, cols) = (shape[0], shape[1], shape[2]);
    
    let mut ext_matrix = Array3::<f32>::zeros((batch, rows, 4));
    
    for i in 0..batch {
        for j in 0..rows {
            ext_matrix[[i, j, 0]] = ms2_data_tensor[[i, j, 0]];
            if cols > 3 {
                ext_matrix[[i, j, 1]] = ms2_data_tensor[[i, j, 3]];
            }
            if cols > 8 {
                ext_matrix[[i, j, 2]] = ms2_data_tensor[[i, j, 8]];
            }
            if cols > 4 {
                ext_matrix[[i, j, 3]] = ms2_data_tensor[[i, j, 4]];
            }
        }
    }
    
    ext_matrix
}

pub fn build_frag_info(
    ms1_data_tensor: &Array3<f32>,
    ms2_data_tensor: &Array3<f32>,
    frag_repeat_num: usize,
    device: &str
) -> Array3<f32> {
    let ext_ms1_precursors_frag_rt_matrix = build_ext_ms1_matrix(ms1_data_tensor, device);
    let ext_ms2_precursors_frag_rt_matrix = build_ext_ms2_matrix(ms2_data_tensor, device);
    
    let batch = ms1_data_tensor.shape()[0];
    let ms1_frag_count = ms1_data_tensor.shape()[1];
    let ms2_frag_count = ms2_data_tensor.shape()[1];
    let total_frag_count = ms1_frag_count + ms2_frag_count;
    
    let mut frag_info = Array3::<f32>::zeros((batch, total_frag_count, 4));
    
    // Copy MS1 fragment info
    frag_info.slice_mut(s![.., ..ms1_frag_count, ..])
        .assign(&ext_ms1_precursors_frag_rt_matrix);
    
    // Copy MS2 fragment info
    frag_info.slice_mut(s![.., ms1_frag_count.., ..])
        .assign(&ext_ms2_precursors_frag_rt_matrix);
    
    frag_info
}

// UniqueValues extraction
#[derive(Debug, Clone)]
pub struct UniqueValues {
    pub ms2_rt_values: Vec<f32>,
}

pub fn extract_unique_rt_im_values(raw_data: &TimsTOFRawData) -> UniqueValues {
    // OPTIMIZED: Parallel deduplication
    fn deduplicate_sorted(mut values: Vec<f32>) -> Vec<f32> {
        if values.is_empty() {
            return values;
        }
        
        values.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        
        let mut result = Vec::with_capacity(values.len() / 10);
        let mut last_val = (values[0] * 1_000_000.0).round() as i64;
        result.push(values[0]);
        
        for &val in &values[1..] {
            let quantized = (val * 1_000_000.0).round() as i64;
            if quantized != last_val {
                result.push(val);
                last_val = quantized;
            }
        }
        
        result
    }
    
    let ms2_rt_all: Vec<f32> = raw_data.ms2_windows
        .par_iter()
        .flat_map(|(_, ms2_data)| ms2_data.rt_values_min.par_iter().copied())
        .collect();
    
    let ms2_rt_values = deduplicate_sorted(ms2_rt_all);
    
    UniqueValues {
        ms2_rt_values,
    }
}

pub fn save_unique_values_to_files(unique_values: &UniqueValues, output_dir: &str) -> Result<(), Box<dyn Error>> {
    use std::io::Write;
    
    std::fs::create_dir_all(output_dir)?;
    
    let mut file = File::create(format!("{}/unique_ms2_rt_values.txt", output_dir))?;
    writeln!(file, "# Unique MS2 RT values (in minutes)")?;
    writeln!(file, "# Total count: {}", unique_values.ms2_rt_values.len())?;
    for rt in &unique_values.ms2_rt_values {
        writeln!(file, "{:.6}", rt)?;
    }
    
    Ok(())
}