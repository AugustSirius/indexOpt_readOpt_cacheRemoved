对于RSM处理应该用 02.rust_for_rsm_optimizedRead-modified-skipfirstErrorFrame-cancelSliceIntoParIter 版本
因为在优化 build index data 和 indexed_timstof_data 过程中不小心重新对 slice_by_mz_im_range 引入了 into_par_iter() 
这样会导致 threads 更高时因为资源问题 contention 处理速度更慢

