/** 
	Copyright Dr. Xiang Fu

	Author: Dr. Xiang Fu
	All Rights Reserved.
	Created: 05/10/2022
	Revised: 08/02/2022 -> added coset fft and ifft
	Distributed FFT algorithm
*/
extern crate ark_ff;
extern crate ark_poly;
extern crate ark_std;
extern crate mpi;
extern crate ark_serialize;
use self::ark_ff::{PrimeField};
//use self::ark_poly::{EvaluationDomain};
use self::ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
use self::ark_std::log2;
use profiler::config::*;
use tools::*;
//use profiler::config::*;
//use crate::tools::*;
//use crate::profiler::config::*;

#[cfg(feature = "parallel")]
use self::ark_std::cmp::max;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

//use mpi::point_to_point as p2p;
//use mpi::topology::Rank;
use self::mpi::traits::*;
use self::mpi::environment::*;
use crate::poly::dis_vec::DisVec;
//use crate::poly::dis_vec::*;
use crate::poly::common::*;
use crate::poly::serial::*;


/** serial version (logical) of DIZK 
See DIZK paper at USENIX'18.
	Row/column method. Size required to be 2^{k}
*/
pub fn serial_dizk_fft<F: PrimeField>(vec: &mut Vec<F>){
	serial_dizk_fft_worker(vec, true);
}

/** serial version (logical) of DIZK 
See DIZK paper at USENIX'18.
	Row/column method. Size required to be 2^{k}
*/
pub fn serial_dizk_ifft<F: PrimeField+CanonicalSerialize+CanonicalDeserialize+Clone>(vec: &mut Vec<F>){
	serial_dizk_fft_worker(vec, false);
}

/** Re-implementation of the DIZK's nlog(n) FFT algorithm.
See DIZK paper at USENIX'18.
	Row/column method. Size required to be 2^{k}
*/
pub fn serial_dizk_fft_worker<F: PrimeField>(vec: &mut Vec<F>, bfft: bool){
	//1. split into square of rows/cols
	let n = vec.len();
	assert!(n.is_power_of_two(), "n is not power of 2!");
	let k = log2(n);
	let rows:usize = 1<<(k/2);
	let cols:usize = n/rows as usize;
	let mut vec2 = vec![F::zero(); n]; //needit to save temporary data

	//2. pass1: group by columns
	let mut col_vec = vec![F::zero(); rows]; //rows: column size (num of ele)
	let omega_shift = if bfft {F::get_root_of_unity(n as u64).unwrap()} 
		else{ F::get_root_of_unity(n as u64).unwrap().inverse().unwrap() };
	let mut arr_omega= vec![F::zero(); cols];
	arr_omega[0] = F::one();
	for i in 1..cols{
		arr_omega[i] = arr_omega[i-1]*omega_shift;
	}
	for group in 0..cols{
		//2.1 build up the col vector
		for index in 0..rows{
			//col_vec[index] = vec[index*rows+ group];
			col_vec[index] = vec[index*cols+ group];
		}
		//2.2 run FFT 
		//dump_vec("DEBUG 201: before COL fft", &col_vec);
		if bfft{
			serial_fft(&mut col_vec);
		}else{
			serial_ifft(&mut col_vec);
		}
		//dump_vec("DEBUG 202: after COL fft", &col_vec);

		//2.3 copy back
		let mut nth_root = F::one();
		let update = arr_omega[group];
		for index in 0..rows{
			vec2[index*cols+ group] = col_vec[index] * nth_root;
			nth_root *= update;
		}
	
	}

	//2. pass2: group of rows and FFT.
	let mut row_vec= vec![F::zero(); cols]; 
	for group in 0..rows{
		//2.1 build up the row vector
		for index in 0..cols{
			row_vec[index] = vec2[group*cols+ index];
		}
		//2.2 run FFT 
		//dump_vec("DEBUG USE 201: BEFORE ROW FFT: ", &row_vec);
		if bfft{
			serial_fft(&mut row_vec);
		}else{
			serial_ifft(&mut row_vec);
		}
		//dump_vec("DEBUG USE 202: AFTER ROW FFT: ", &row_vec);
		
		//2.3 copy back (NOTE: switch diagnose row->col and col to row
		//in the new matrix: new_rows: cols, new_cols: rows
		// new_row_size = rows, new_col_size = cols
		// old_row_id: group, old_col_id: index
		// new_row_id: index, new_col_id: group
		// new_id = new_row_id*new_row_size + new_col_id
		//		  = index*rows+group
		for index in 0..cols{
			vec[group + index*rows] = row_vec[index];
		}
	}
	
	//dump_vec("DEBUG USE 299: SERIAL_DIZK_COMPLETE", &vec);
}

// ----------------------------------------------------
// Section Related to Distributed Dizk FFT
// ----------------------------------------------------
/** generate the jobs of processing col group for i'th processor
of np processors. num_jobs should be a power of two. 
	Output: an array of column ID/row ID (depending on col/row mode)
 */
fn gen_col_jobs(i:u64, np: u64, num_jobs: u64) -> Vec<u64>{
	let size = num_jobs/np;
	assert!(size*np==num_jobs, "num_jobs (cols): {} % np: {} !=0", num_jobs, np);
	let mut vec = vec![0; (num_jobs/np) as usize];
	for idx in 0..size{
		vec[idx as usize] = np * idx + i;
	}	
	return vec;
}

/** The function is designed such that all jobs are located
within the partition for processor i */
fn gen_row_jobs<F:PrimeField+CanonicalSerialize+CanonicalDeserialize+Clone>(i:u64, np: u64, row_size: u64, num_jobs: u64, 
	dv: &DisVec<F>) -> Vec<u64>{
	let my_jobs = num_jobs/np; //number of rows to process
	assert!(my_jobs*np==num_jobs, "my_jobs*num_jobs!=np");
	let (start_idx, part_size) = dv.get_partition_info(i);
	assert!(start_idx%row_size==0, "start_idx%row_size!=0");
	assert!(part_size%row_size==0, "part_size%row_size!=0");
	let mut vec = vec![0; my_jobs as usize];
	for idx in 0..my_jobs{
		vec[idx as usize] = start_idx/row_size + idx;
	}	
	return vec;
}

/** given a slice [start_idx, start_idx+partsize), find the
first valid element in column id: col_id. If there is NO such available,
return start_idx+partsize (which is OUT OF THE SCOPE */
fn get_first_col_idx(col_id: u64, start_idx: u64, part_size: u64, cols: u64) -> u64{
	let row_size = cols;
	let row_id = start_idx/row_size;
	let mut first_idx = row_id * row_size + col_id;
	if first_idx>start_idx+part_size{
		first_idx = start_idx+part_size;
	}	
	return first_idx;
}


/** Generate the copy plan. (two modes: column mode and row mode)
	Input:
		part_size: the partition data size
		start_idx: the index of the first element in the DisVec of the partition
		target_executor: must be from 0..np
		np: number of partitions (processors)
		rows, cols: rows and columns for decision making
	Output: the copy plan for the target executor, for each
		col_job it is assigned (in increasing order)
		(idx, stepwise, num) -> the idx to start copy, stepwise increase, and
			number of elements to copy (the IDX is relative from the start
			of its partition, could be starting from 0, i.e., NOT the real
			index in the entire distributed vector).
		total_num: sum of num
*/
fn get_copy_col_plan(part_size: u64, start_idx: u64, 
	target_executor: u64, np: u64, cols: u64) 
	-> (u64, Vec<(u64, u64, u64)>){
	//println!("DEBUG USE 101: ENTERING get_copy_plan: start_idx: {}, target_executor: {}, np: {}, rows: {}, cols: {}, bcols: {}", start_idx, target_executor, np, rows, cols, b_column);
	let mut total_sum = 0;
	let njobs = cols;
	let vjobs = gen_col_jobs(target_executor, np, njobs);
	//println!("DEBUG USE 102: njobs: {}, vjobs: {:?}, part_size: {}", njobs, &vjobs, part_size);
	let vjob_len = vjobs.len();
	let mut vec = vec![(0u64,0u64,0u64); vjob_len as usize];
	for i in 0..vjob_len{
		let job_id = vjobs[i as usize];	
		let end_idx = start_idx + part_size-1; //included
		let idx = get_first_col_idx(job_id, start_idx, part_size, cols);
		let step_size = cols;
		let num = (end_idx-idx)/step_size+1;
		total_sum += num;
		vec[i as usize] = (idx-start_idx, step_size, num); //num is 0 means invalid entry
		//println!("DEBUG USE 103: col id: {}: job: {:?}", i, &vec[i as usize]);
	}
	//println!("DEBUG USE 104: total_sum: {}, vec: {:?}", total_sum, vec);
	return (total_sum, vec);
}

/** execute the copy plan and construct a vector of total_size */
fn exec_copy_plan<F: PrimeField>(vsrc: &Vec<F>, total_size: u64, plan: &Vec<(u64,u64,u64)> ) -> Vec<F>{
	let mut vec_res = vec![F::zero(); total_size as usize];
	let mut idx = 0;
	for i in 0..plan.len(){
		let (start_idx, step, count) = plan[i];
		for j in 0..count{
			vec_res[idx] = vsrc[(start_idx+step*j) as usize];
			idx+=1;
		}
	}
	return vec_res;
}


/** prepares and send the column group data to all executors, by
taking the data from its own partition for dvec.
	my_rank: its own rank
	rows, cols: should be power of 2
	dvec: the distributed vector
	univ: the MPI handler
Return: the data sent over from other processors.
*/
fn send_col_data<F:PrimeField+CanonicalSerialize+CanonicalDeserialize+Clone>(my_rank: u64, np: u64, cols: u64, dvec: &DisVec<F>, univ: &Universe) -> Vec<Vec<F>>{
	//1. get my partition
	let (start_idx, part_size) = dvec.get_partition_info(my_rank);	
	//println!("DEBUG USE 777: part_size: {}, cols: {}", part_size, cols);
	assert!(part_size%cols==0, "partition size: {} NOT DIVISBLE by row size: {}! This is needed for saving communication cost in last row FFT step", part_size, cols);
	let vsrc = &dvec.partition;

	//2. prepare the data for each rank
	let mut vec_data = vec![vec![]; np as usize];
	for i in 0..np{
		let (total_size, vplan) = get_copy_col_plan(part_size, start_idx, i, np,  cols);
		//println!("DEBUG USE 106.1: my_rank: {}, target: {}, vplan: {:?}", my_rank, i, vplan);
		vec_data[i as usize]  = exec_copy_plan(&vsrc, total_size, &vplan);
	}

	let vdata = nonblock_broadcast_new(&vec_data, np, univ, F::zero());
	return vdata;
}

/** prepare an array of omega^0 ... omega^size*/
fn get_arr_omega<F:PrimeField>(rows: usize, cols: usize, bfft: bool) -> Vec<F>{
	let n:u64 = (rows*cols) as u64;
	let omega_shift = if bfft {F::get_root_of_unity(n).unwrap()} else
			{F::get_root_of_unity(n).unwrap().inverse().unwrap()};
	let mut arr_omega= vec![F::zero(); cols];
	arr_omega[0] = F::one();
	for i in 1..cols{
		arr_omega[i] = arr_omega[i-1]*omega_shift;
	}
	return arr_omega;
}


/** Receive the col data and process all jobs. After all jobs
	are processed, the partition has a number of columns.
	Dispatch the data to all executors for row jobs.
	my_rank: the current processor rank,
	np: number of processors
	Return: the row_data from all processors
*/
fn process_col_data<F:PrimeField+CanonicalSerialize+CanonicalDeserialize+Clone>(my_rank: u64, np: u64, rows: u64, cols: u64, dvec: &mut DisVec<F>, univ: &Universe, vdata: &Vec<Vec<F>>, bfft: bool) -> Vec<Vec<F>>{
	//1. receive the data from all processes
	let b_perf = false;
	let b_perf2 = false;
	let mut t1 = Timer::new();
	t1.start();
	let mut t2 = Timer::new();
	t2.start();
	let jobs = gen_col_jobs(my_rank, np, cols); 
	let mut vcol_jobs:Vec<Vec<F>> = vec![vec![F::zero(); rows as usize]; jobs.len()];
	//println!("DEBUG USE 500: myrank: {}, job: {:?}", my_rank, jobs);
	if b_perf2{log_perf(LOG1, &format!("------ ---- ProcColData Step1: jobs: {}", jobs.len()), &mut t1);}

	//2. assemble the data for all col jobs (sync as we need full data)
	//println!("DEBUG 501: entering receive_and process_col_data");
	for proc in 0..np{
		let (p_start, part_size) = dvec.get_partition_info(proc);
		//println!("DEBUG 501: peer: {}, p_start: {}, part_size: {}", proc, p_start, part_size);
		let (total_len, plan) = get_copy_col_plan(part_size, p_start, my_rank, np, cols);
		//println!("DEBUG 502: plan {} -> {}: {:?}", proc, my_rank, &plan);
		let mut idx = 0;
		for j in 0..plan.len(){
			let (start_idx, _, count) = plan[j as usize];
			let real_idx = p_start + start_idx;
			//println!("DEBUG USE 503: vcol_jobs: {:?}", &vcol_jobs);
			//println!("DEBUG USE 503.5: start_idx: {}, step: {}, count: {}, real_idx: {}", start_idx, step, count, real_idx);
			for k in 0..count{
				let idx_in_col = real_idx/ cols + k;
				//println!("DEBUG USE 504: === MYRANK: {} ==== rows: {}, cols: {}, idx_in_col: {}, count: {}, setting cvols_jobs[{}][{}] to {}", my_rank, rows, cols, idx_in_col, count, j, idx_in_col, &data[idx as usize]);
				vcol_jobs[j as usize][idx_in_col as usize] = vdata[proc as usize][idx as usize];
				idx+=1;
			} 
		}
		assert!(idx==total_len, "NOT all elements are copied into col job!"); 
		//println!("DEBUG USE 505: assembled col data: {:?}", vcol_jobs);
	}
	//println!("DEBUG USE 301: assembled col data for rank: {}", my_rank);
	//for i in 0..vcol_jobs.len(){
	//	dump_vec("job: ", &vcol_jobs[i as usize]);
	//}
	if b_perf2{log_perf(LOG1, &format!("------ ---- ProcColData Step2: Assemble Col Data"), &mut t1);}
	

	//3. FFT on all col jobs
	let arr_omega = get_arr_omega::<F>(rows as usize, cols as usize, bfft);
	for i in 0..jobs.len(){
		let col_id = jobs[i as usize];
		let vec = &mut vcol_jobs[i as usize];
		//dump_vec("DEBUG USE 301 before COL fft", &vec);
		if bfft{
			serial_fft(vec);
		}else{
			serial_ifft(vec);
		}
		//dump_vec("DEBUG USE 302 AFTER COL fft", &vec);

		let mut nth_root = F::one();
		let update = arr_omega[col_id as usize];
		for index in 0..rows{
			vec[index as usize] *=  nth_root;
			nth_root *= update;
		}
	}
	if b_perf2{log_perf(LOG1, &format!("------ ---- ProcColData Step3: FFT: jobs: {}", jobs.len()), &mut t1);}

	//4. assemble data for each executor
	let mut vec_to_send:Vec<Vec<F>> = vec![];
	for proc in 0..np{
		let proc_job = gen_row_jobs(proc, np, cols, rows, dvec);
		let mut res = vec![F::zero(); proc_job.len()*vcol_jobs.len()];
		let mut idx = 0usize;
		for row_id in proc_job{
			for col_data in &vcol_jobs{
				res[idx] = col_data[row_id as usize];	
				idx+=1;
			}
		}
		vec_to_send.push(res);
	}
	if b_perf2{log_perf(LOG1, &format!("------ ---- ProcColData Step4: Assemble Col Data"), &mut t1);}


	//5. send them out
	let vrow_data = nonblock_broadcast_new(&vec_to_send, np, univ, F::zero());
	//println!("DEBUG USE 801: broadcast DONE:");
	if b_perf2{log_perf(LOG1, &format!("------ ---- ProcColData Step5: Broadcast"), &mut t1);}
	if b_perf{log_perf(LOG1, &format!("------ ---- ProcColData TOTAL"), &mut t2);}
	return vrow_data;
}

/** Receive the row data and process all jobs (row FFT). 
	Still need to SWAP row/colmn idx thus re-dispatching data.
	ASSUMPTION: partition size is a multiple of ROW size and COL size
	vdata: the data from all other processors
*/
fn process_row_data<F:PrimeField+CanonicalSerialize+CanonicalDeserialize+Clone>(my_rank: u64, np: u64, rows: u64, cols: u64, dvec: &mut DisVec<F>, univ: &Universe, vdata: &Vec<Vec<F>>, bfft: bool){
	//1. receive all the data
	let (_, part_size) = dvec.get_partition_info(my_rank);
	let njobs = rows/np;
	let mut vrow_jobs: Vec<Vec<F>> = vec![vec![F::zero(); cols as usize]; njobs as usize];
	let row_jobs = gen_row_jobs(my_rank, np, cols, rows, dvec);
	for proc in 0..np{
		let col_jobs = gen_col_jobs(proc, np, cols);
		let mut idx:usize = 0;
		for j in 0..row_jobs.len(){
			for col_id in &col_jobs{
				vrow_jobs[j as usize][*col_id as usize] = vdata[proc as usize][idx];
				idx+=1;
			} 
		}
	}

	//2. FFT all
	for i in 0..njobs{
		//dump_vec("DEBUG USE 901: BEFORE ROW FFT", &vrow_jobs[i as usize]);
		if bfft{
			serial_fft(&mut vrow_jobs[i as usize]);
		}else{
			serial_ifft(&mut vrow_jobs[i as usize]);
		}
		//dump_vec("DEBUG USE 902: AFTER ROW FFT", &(vrow_jobs[i as usize]))
	}
	//println!("---- DEBUG USE 888 ----, processor: {} RECEIVED:", my_rank);
	//for row_j in vrow_jobs{ dump_vec("ROW:", &row_j); }

	//3. construct the data to send for column-row swap
	let unit_size = part_size/np;
	let mut vec_to_send = vec![vec![F::zero();unit_size as usize]; np as usize];
	let mut vec_counter:Vec<usize> = vec![0usize; np as usize];
	let my_start_row = my_rank * (part_size/cols);
	//println!("DEBUG USE 101: my_rank: {}, my_start_now: {}, part_size: {}, cols: {}", my_rank, my_start_row, part_size, cols);
	for i in 0..njobs{
		let vsrc = &vrow_jobs[i as usize];
		for j in 0..cols{
			//vec[idx] =  vsrc[j as usize];
			let new_row_id = j;
			let new_col_id = my_start_row + i;
			let new_idx = new_row_id * rows  + new_col_id;
			let new_proc = (((new_idx / cols))/njobs) as usize;
			//println!("myrank: {}, row: {}, col: {}, new_row_id: {}, new_col_id: {}, new_idx: {}, new_proc: {}", my_rank, i, j, new_row_id, new_col_id, new_idx, new_proc);
			vec_to_send[new_proc][vec_counter[new_proc]] = vsrc[j as usize];
			vec_counter[new_proc] += 1;
		}
	}
	//println!("DEBUG USE 900: prodessor: {}. After Copy\n", my_rank);
	//for i in 0..np {dump_vec("SEND", &vec_to_send[i as usize]);}
	let vpart_data  = nonblock_broadcast_new(&vec_to_send, np, univ, F::zero());

	//4. receive and store the data back into partition data
	let mut vpart = vec![F::zero(); part_size as usize];
	//println!("DEBUG USE 871: part_size: {}", part_size);
	assert!(rows==cols || 2*rows==cols, "assumption rows==cols || 2rows==cols not saitsifed!");
	let (v1, v2) = ((rows/np) * my_rank, (rows/np)*(my_rank+1));
	let (b_rowidx, e_rowidx) = if 2*rows==cols {(2*v1, 2*v2)} else {(v1, v2)};
	//println!("DEBUG USE 901: my_rank: {}, b_rowidx: {}, e_rowidx: {}", my_rank, b_rowidx, e_rowidx);
	for proc in 0..np{
		let data = &vpart_data[proc as usize];
		let mut idx = 0;
		for rid in 0..rows/np{
			for cid in b_rowidx..e_rowidx{
				let real_idx = (proc*rows/np+rid)*cols+ cid;
				//swap col and row id
				let (new_cid, new_rid) = (real_idx/cols, real_idx%cols); 
				let offset = new_rid * rows + new_cid - my_rank*part_size;
				//println!("DEBUG USE 902.5: new_rid: {}, rows: {}, np: {}, my_rank: {}, new_cid: {}, cols: {}, offset: {}", new_rid, rows, np, my_rank, new_cid, cols, offset);
				//println!("DEBUG USE 902: my_rank: {}, src proc: {}, rid: {}, cid: {}, real_idx: {}, new_rid: {}, new_cid: {}, offset: {}, idx: {}", my_rank, proc, rid, cid, real_idx, new_rid, new_cid, offset, idx);
				vpart[offset as usize] = data[idx];
				//println!("DEBUG USE 903: my_rank{}, saving data: {} to offset {}", my_rank, &data[idx], offset);
				idx+=1;
			}
		}
	}
	dvec.partition = vpart;
	//println!("DEBUG USE 999: AFTER assembling: my_rank: {} ---", my_rank);
}

/** the REAL distributed version. 
 Re-implementation of the DIZK's nlog(n) FFT algorithm.
See DIZK paper at USENIX'18.  
Row/column method. Size required to be 2^{k}.
*/
pub fn distributed_dizk_fft<F: PrimeField+CanonicalSerialize+CanonicalDeserialize+Clone>(dvec: &mut DisVec<F>, univ:&Universe){
	distributed_dizk_fft_worker(dvec, univ, true);
}

/** the ifft distributed */
pub fn distributed_dizk_ifft<F: PrimeField+CanonicalSerialize+CanonicalDeserialize+Clone>(dvec: &mut DisVec<F>, univ:&Universe){
	distributed_dizk_fft_worker(dvec, univ, false);
}
/** Re-implementation of the DIZK's nlog(n) FFT algorithm.
See DIZK paper at USENIX'18.  
Row/column method. Size required to be 2^{k}.
*/
pub fn distributed_dizk_fft_worker<F: PrimeField+CanonicalSerialize+CanonicalDeserialize+Clone>(dvec: &mut DisVec<F>, univ:&Universe, bfft: bool){
	//0. get the n processors and decide main thread
	let b_perf = false;
	let b_perf2 = false;
	let mut t1 = Timer::new();
	t1.start();
	let mut t2 = Timer::new();
	t2.start();

	let n:u64 = dvec.len as u64;
	//let np = RUN_CONFIG.n_proc;
	let k = log2(n as usize);
	assert!(n.is_power_of_two(), "n is not power of 2!");
	let rows:u64= 1<<(k/2) as u64;
	let cols:u64= n/rows as u64;

	let world = univ.world();
	let np = world.size() as u64;
	let rank = world.rank() as u64;
	//println!("===========================\n");
	//println!("DISTRIBUTED dizk_fft, rank: {}\n", rank);
	//println!("===========================\n");

	//1. prepare the data to send for col_group execution
	let vdata = send_col_data(rank, np, cols, dvec, univ);
	if b_perf2{log_perf(LOG1, &format!("------ --FFT Step1 send_col_data: size: {}", dvec.len), &mut t1);}

	//2. receive the data for col_group execution (FFT) and send data 
	//for row group jobs.
	let vrow_data = process_col_data(rank, np, rows, cols, dvec,  univ, &vdata, bfft);
	if b_perf2{log_perf(LOG1, &format!("------ --FFT Step2 process col_data"), &mut t1);}

	//3. receive and process the row FFT data
	process_row_data(rank, np, rows, cols, dvec,  univ, &vrow_data, bfft);
	if b_perf2{log_perf(LOG1, &format!("------ --FFT Step3 process row_data"), &mut t1);}

	RUN_CONFIG.better_barrier("dis_fft_worker");
	if b_perf2{log_perf(LOG1, &format!("------ --FFT Step4 sync"), &mut t1);}
	if b_perf{log_perf(LOG1, &format!("------ --FFT TOTAL"), &mut t2);}
	//println!("****** DEBUG USE 1000: node {} reached barrier. ****\n\n", rank);
}

/// distributed version of coset_fft: applying fft on shifted base
/// (omega*t). Assumption: t is NOT any of omega^i 
/// Assumption: dvec len has to be power of two
/// NEEDS to be called at all nodes!
pub fn distributed_dizk_fft_coset<F: PrimeField+CanonicalSerialize+CanonicalDeserialize+Clone>(dvec: &mut DisVec<F>, univ:&Universe, t: F){
	//1. modify the input set
	let b_perf = false;
	let mut t1 = Timer::new();
	t1.start();
	let mut t2 = Timer::new();
	t2.start();

	if dvec.main_processor!=0 {panic!("dvec.main_processor!=0");}
	if !dvec.b_in_cluster {panic!("call to_partitions first!");}
	let me = RUN_CONFIG.my_rank as usize;
	let me64 = me as u64;
	let unit_size = dvec.len / (RUN_CONFIG.n_proc as usize);
	let mut coset = t.pow(&[me64 * (unit_size as u64)]);
	for i in 0..dvec.partition.len(){
		dvec.partition[i] *= coset;
		coset *= t;
	}

	if b_perf{log_perf(LOG1, &format!("------ --FFT Coset Step1"), &mut t1);}
	//2. call fft
	distributed_dizk_fft_worker(dvec, univ, true);
	if b_perf{log_perf(LOG1, &format!("------ --FFT Coset Step2"), &mut t1);}
	if b_perf{log_perf(LOG1, &format!("------ --FFT Coset TOTAL: size: {}", dvec.len ), &mut t2);}
}

/// coset ifft (see fft_coset doc)
pub fn distributed_dizk_ifft_coset<F: PrimeField+CanonicalSerialize+CanonicalDeserialize+Clone>(dvec: &mut DisVec<F>, univ:&Universe, t: F){
	//1. call fft
	if dvec.main_processor!=0 {panic!("dvec.main_processor!=0");}
	if !dvec.b_in_cluster {panic!("call to_partitions first!");}
	let me = RUN_CONFIG.my_rank as usize;
	let me64 = me as u64;
	let unit_size = dvec.len / (RUN_CONFIG.n_proc as usize);
	distributed_dizk_fft_worker(dvec, univ, false); //ifft

	//w. modify the input set
	let t = t.inverse().unwrap();
	let mut coset = t.pow(&[me64 * (unit_size as u64)]);
	for i in 0..dvec.partition.len(){
		dvec.partition[i] *= coset;
		coset *= t;
	}

}

#[cfg(test)]
mod tests {
    //use ark_bls12_381::Fr;
	extern crate once_cell;
	extern crate ark_ff;
	use crate::tools::*;
	use crate::profiler::config::*;
	use crate::poly::disfft::*;
	use self::ark_ff::UniformRand;
	type Fr381=ark_bls12_381::Fr;

	fn get_min_test_size()->usize{
		let np = RUN_CONFIG.n_proc as usize;
		return closest_pow2(np*np);	
	}
	#[test]
	fn test_serial_fft_ifft(){
		let size = 2usize;
		let mut vec = rand_arr_field_ele::<Fr381>(size, get_time());
		let vec2 = vec.clone();
		serial_fft(&mut vec);
		serial_ifft(&mut vec);
		assert!(vec_equals(&vec, &vec2), "serial_fft_ifft fails");
	}

	#[test]
	fn test_serial_dizk_fft(){
		let size:usize = 1<<11;
		let mut vec = rand_arr_field_ele::<Fr381>(size, get_time());
		let mut vec2 = vec.clone();
		serial_fft(&mut vec);
		serial_dizk_fft(&mut vec2);
		assert!(vec_equals(&vec, &vec2), "test_serial_dizk_fft fails");
	}

	#[test]
	fn test_serial_dizk_ifft(){
		let size:usize = 1<<11;
		let mut vec = rand_arr_field_ele::<Fr381>(size, get_time());
		let mut vec2 = vec.clone();
		serial_ifft(&mut vec);
		serial_dizk_ifft(&mut vec2);
		assert!(vec_equals(&vec, &vec2), "test_serial_dizk_ifft fails");
	}

	fn test_dist_dizk_fft_worker(log_size: usize, bfft: bool){
		let size:usize = 1<<log_size;
		let my_rank = RUN_CONFIG.univ.world().rank() as u64;
		let main_rank = 0u64; //ONLY PROCESS the dv1 generated for main rank
		let vecid = 101u64;
		//let univ = &RUN_CONFIG.univ;
		if my_rank==main_rank{ 
 			let vec = rand_arr_field_ele::<Fr381>(size, get_time());
			let mut vec2 = vec.clone();
			if bfft{
				serial_dizk_fft(&mut vec2);
			}else{
				serial_dizk_ifft(&mut vec2);
			}
			let mut dv1 = DisVec::new_dis_vec_with_id(vecid, main_rank,vec.len(), vec);
			dv1.to_partitions(&RUN_CONFIG.univ);
			if bfft{
				distributed_dizk_fft(&mut dv1, &RUN_CONFIG.univ);
			}else{
				distributed_dizk_ifft(&mut dv1, &RUN_CONFIG.univ);
			}
			dv1.from_partitions(&RUN_CONFIG.univ);
			assert!(vec_equals(&vec2, &dv1.vec), "failed test_dis_dizk_fft");
			RUN_CONFIG.better_barrier("test_dis_fft_worker1");
		}else{
			let mut dv1:DisVec<Fr381> = DisVec::new_dis_vec_with_id(vecid, main_rank, size, vec![]); //just pass the vecid, content is fake
			dv1.to_partitions(&RUN_CONFIG.univ);
			if bfft{
				distributed_dizk_fft(&mut dv1, &RUN_CONFIG.univ);
			}else{
				distributed_dizk_ifft(&mut dv1, &RUN_CONFIG.univ);
			}
			dv1.from_partitions(&RUN_CONFIG.univ);
			RUN_CONFIG.better_barrier("test_dis_fft_worker2");
		}
	}

	#[test]
	fn test_distributed_dizk_fft(){
		test_dist_dizk_fft_worker(4, true);
	}

	#[test]
	fn test_distributed_dizk_ifft(){
		test_dist_dizk_fft_worker(4, false);
	}

	fn test_distributed_fft_coset_worker(bfft: bool){
		let size = get_min_test_size()*2;
		let mut rng = gen_rng_from_seed(17328123u128);
		let mut v1 = rand_arr_field_ele::<Fr381>(size, 127239234234u128);
		let mut dv1 = DisVec::<Fr381>::new_dis_vec_with_id(0, 0, size, v1.clone());
		dv1.to_partitions(&RUN_CONFIG.univ);
		let t = Fr381::rand(&mut rng);
		if bfft{
			v1 = fft_coset(&v1, t);
		}else{
			v1 = ifft_coset(&v1, t);
		}
		if bfft{	
			distributed_dizk_fft_coset(&mut dv1, &RUN_CONFIG.univ, t);
		}else{
			distributed_dizk_ifft_coset(&mut dv1, &RUN_CONFIG.univ, t);
		}
		let v2 = dv1.collect_from_partitions(&RUN_CONFIG.univ);
		if RUN_CONFIG.my_rank==0{
			if bfft{
				assert!(v1==v2, "failed test_dist_fft_coset");
			}else{
				assert!(v1==v2, "failed test_dist_ifft_coset");
			}
		}
		
	}
	#[test]
	fn test_distributed_fft_ifft_coset(){
		test_distributed_fft_coset_worker(true);
		test_distributed_fft_coset_worker(false);
	}
}
