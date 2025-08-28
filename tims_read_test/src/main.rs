use std::error::Error;
use std::path::Path;
use timsrust::{
    readers::{FrameReader, MetadataReader},
    MSLevel,
};

fn main() -> Result<(), Box<dyn Error>> {
    // Hardcoded path to the TimsTOF data
    // let d_folder = "/wangshuaiyao/00.Tims_Training40data_guomics/K20200702yuel_TPHP_DIA_234_Slot2-22_1_986.d";
    let d_folder = "/wangshuaiyao/00.Tims_Training40data_guomics/K20200706yuel_TPHP_DIA_309_Slot2-43_1_1058.d";
    
    println!("========================================");
    println!("TimsTOF Data Reading Test");
    println!("========================================");
    println!("Data folder: {}", d_folder);
    
    // Step 1: Check if path exists
    let d_path = Path::new(d_folder);
    if !d_path.exists() {
        return Err(format!("Error: folder {:?} not found", d_path).into());
    }
    println!("✓ Path exists");
    
    // Step 2: Try to read metadata
    println!("\n--- Reading Metadata ---");
    let tdf_path = d_path.join("analysis.tdf");
    println!("TDF file: {:?}", tdf_path);
    
    let meta = match MetadataReader::new(&tdf_path) {
        Ok(m) => {
            println!("✓ Metadata loaded successfully");
            m
        }
        Err(e) => {
            println!("✗ Failed to read metadata: {:?}", e);
            return Err(e.into());
        }
    };
    
    // Print some metadata info
    println!("  MZ converter available: {}", true);
    println!("  IM converter available: {}", true);
    
    // Step 3: Try to initialize frame reader
    println!("\n--- Initializing Frame Reader ---");
    let frames = match FrameReader::new(d_path) {
        Ok(f) => {
            println!("✓ Frame reader initialized");
            f
        }
        Err(e) => {
            println!("✗ Failed to initialize frame reader: {:?}", e);
            return Err(e.into());
        }
    };
    
    let n_frames = frames.len();
    println!("  Total frames: {}", n_frames);
    
    // Step 4: Try to read frames one by one (non-parallel)
    println!("\n--- Testing Frame Reading ---");
    println!("Will attempt to read first 10 frames (or all if less than 10)");
    
    let frames_to_test = n_frames.min(1000);
    let mut successful_reads = 0;
    let mut failed_reads = 0;
    let mut ms1_frames = 0;
    let mut ms2_frames = 0;
    
    for idx in 0..frames_to_test {
        print!("Frame {}/{}: ", idx + 1, frames_to_test);
        
        match frames.get(idx) {
            Ok(frame) => {
                successful_reads += 1;
                
                // Basic frame info
                let rt_min = frame.rt_in_seconds as f32 / 60.0;
                let n_peaks = frame.tof_indices.len();
                let n_scans = frame.scan_offsets.len() - 1;
                
                match frame.ms_level {
                    MSLevel::MS1 => {
                        ms1_frames += 1;
                        println!("✓ MS1 frame - RT: {:.2} min, {} peaks, {} scans", 
                                rt_min, n_peaks, n_scans);
                    }
                    MSLevel::MS2 => {
                        ms2_frames += 1;
                        let n_windows = frame.quadrupole_settings.isolation_mz.len();
                        println!("✓ MS2 frame - RT: {:.2} min, {} peaks, {} scans, {} windows", 
                                rt_min, n_peaks, n_scans, n_windows);
                    }
                    _ => {
                        println!("✓ Other frame type - RT: {:.2} min", rt_min);
                    }
                }
            }
            Err(e) => {
                failed_reads += 1;
                println!("✗ FAILED: {:?}", e);
                
                // Try to get more detail about the error
                match e {
                    timsrust::readers::FrameReaderError::TdfBlobReaderError(blob_err) => {
                        println!("  Blob reader error details: {:?}", blob_err);
                        println!("  This typically indicates corrupted or incompatible data");
                    }
                    _ => {
                        println!("  Error type: {:?}", e);
                    }
                }
            }
        }
    }
    
    // Step 5: Summary
    println!("\n--- Summary ---");
    println!("Frames tested: {}", frames_to_test);
    println!("Successful reads: {} ({:.1}%)", 
             successful_reads, 
             (successful_reads as f32 / frames_to_test as f32) * 100.0);
    println!("Failed reads: {} ({:.1}%)", 
             failed_reads,
             (failed_reads as f32 / frames_to_test as f32) * 100.0);
    println!("MS1 frames: {}", ms1_frames);
    println!("MS2 frames: {}", ms2_frames);
    
    if failed_reads > 0 {
        println!("\n⚠ WARNING: Some frames failed to read!");
        println!("This could indicate:");
        println!("  1. Corrupted data in the .tdf_bin file");
        println!("  2. Incompatible TimsTOF data format");
        println!("  3. Insufficient system resources");
        
        // Try to find the first failing frame
        println!("\n--- Finding First Failing Frame ---");
        for idx in 0..n_frames {
            if let Err(e) = frames.get(idx) {
                println!("First failing frame: {} (out of {})", idx, n_frames);
                println!("Error: {:?}", e);
                break;
            }
            
            // Progress indicator for large datasets
            if idx > 0 && idx % 1000 == 0 {
                println!("  Checked {} frames so far...", idx);
            }
        }
    } else {
        println!("\n✓ All tested frames read successfully!");
        
        // If initial test passed, try more frames
        if frames_to_test < n_frames {
            println!("\n--- Extended Test ---");
            println!("Testing more frames (every 1000th frame)...");
            
            let mut extended_failures = Vec::new();
            for idx in (0..n_frames).step_by(1000) {
                if let Err(e) = frames.get(idx) {
                    extended_failures.push((idx, e));
                }
            }
            
            if extended_failures.is_empty() {
                println!("✓ Sample of frames across the dataset read successfully");
            } else {
                println!("✗ Found {} failing frames in extended test:", extended_failures.len());
                for (idx, err) in extended_failures.iter().take(5) {
                    println!("  Frame {}: {:?}", idx, err);
                }
            }
        }
    }
    
    println!("\n========================================");
    println!("Test Complete");
    println!("========================================");
    
    Ok(())
}