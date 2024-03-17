/** 
	Copyright Dr. Xiang Fu

	Author: Dr. Xiang Fu
	All Rights Reserved.
	Created: 05/13/2022
	Modified 08/31/2022 -> add real_len and functions for computing real_len

	Distributed Vector Object
	Given n processors, distributed to n processors.
*/
extern crate ark_ff;
extern crate ark_poly;
extern crate ark_std;
extern crate ark_serialize;
extern crate mpi;
use self::ark_ff::{PrimeField};
use self::ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
//use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
//use ark_std::log2;
use crate::tools::*;
use super::common::*;
//use std::convert::TryInto;
//use std::collections::HashMap;
//use self::ark_std::rand::RngCore;

//use mpi::point_to_point as p2p;
//use mpi::topology::Rank;
use self::mpi::traits::*;
use self::mpi::environment::*;
//use std::time::{SystemTime};
//use ark_ff::{PrimeField};
//use ark_std::rand::rngs::StdRng;
//use ark_std::rand::SeedableRng;
//use std::time::{UNIX_EPOCH};
//use ark_bls12_381::Bls12_381;
//use ark_ec::{PairingEngine};

#[cfg(feature = "parallel")]
use self::ark_std::cmp::max;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::profiler::config::*;


/** Distributed Vector of Finite Foeld Elements.
	Each node maintains a separate DistributedVector object.
	They share the SAME value of id and main_processor.
 */
#[derive(Clone)]
pub struct DisVec<F: PrimeField+CanonicalSerialize+CanonicalDeserialize+Clone>{
	/** a random ID */
	pub id: u64,  //NOTE: REQUIRED TO be equal to be main_processor. fix later
	/** initial starting vector (non-distributed) */
	pub vec: Vec<F>,
	/** whether the vec is in cluster */
	pub b_in_cluster: bool,
	/** length */
	pub len: usize,
	/** number of processors, will be set when to_partition is called */
	pub np: usize,
	/** main processor */
	pub main_processor: usize, 
	/** partition, will be set in to_partitions */
	pub partition: Vec<F>,
	/** real length (like real_degree+1 of a polymomial */
	pub real_len: usize,
	/** if the real_len is set */
	pub real_len_set: bool,
	/** if the disvec is destroyed (mem free-ed) */
	pub b_destroyed: bool
}


impl <F:PrimeField+CanonicalSerialize+CanonicalDeserialize+Clone> DisVec<F>{
	/** release all mem */
	pub fn destroy(&mut self){
		self.b_destroyed = true;
		self.partition.clear();
		self.partition.shrink_to_fit();
		self.vec.clear();
		self.vec.shrink_to_fit();
		self.len = 0;
	}
	/** reset the vector */
	pub fn reset_vec(&mut self, v: Vec<F>){
		self.vec = v;
		self.len = self.vec.len();
	}

	/** Print info. Dump up to limit_num elements */
	pub fn dump(&self, limit_num: usize){
		println!("MYRANK: {}: distributed: {}, len: {}, main_processor: {}", RUN_CONFIG.my_rank, self.b_in_cluster, self.len, self.main_processor);
		let limit = if self.len>limit_num {limit_num} else {self.len};
		if self.b_in_cluster{
			dump_slice("Partition: ", &self.partition[0..limit]); 
		}else{
			if self.vec.len()<self.len{
				println!("NO DATA before to_partition(). len: {}", 
					self.vec.len());
			}else{
				dump_slice("Vector: ", &self.vec[0..limit]); 
			}
		}
	}

	/// generate power t series t^0, t^1, ... t^degree
	/// NOTE: actually creates size:  degree+1
	pub fn power_ts(degree: usize, t: F) -> DisVec<F>{
		let me = RUN_CONFIG.my_rank;
		let np = RUN_CONFIG.n_proc;
		let n = degree + 1;
		let unit = n/np;
		let me_start = unit * me;
		let mysize = if me==np-1 {n/np + n%np} else {n/np};
		let mut mypart = vec![F::zero(); mysize];
		let mut cur_ele = t.pow(&[me_start as u64]);
		for i in 0..mysize{
			mypart[i] = cur_ele;
			cur_ele = cur_ele * t;
		}
		let res = Self::new_from_each_node(0, 0, n, mypart);
		return res;
	}

	/// GENERATE a random instance
	pub fn rand_inst(seed: u128, size: usize) -> DisVec<F>{
		return Self::rand_inst_worker(seed, size, false);
	}
	pub fn rand_inst_worker(seed: u128, size: usize, blast_zero: bool) -> DisVec<F>{
		let me = RUN_CONFIG.my_rank;
		let np = RUN_CONFIG.n_proc;
		let mysize = if me==np-1 {size/np + size%np} else {size/np};	
		let myseed = seed *(13+ (np as u128)) + (me as u128);
		let mut mypart = rand_arr_field_ele::<F>(mysize, myseed);	
		if me==np-1 && blast_zero{
			mypart[mysize-1] = F::zero();
		}
		let res = Self::new_from_each_node(0, 0, size, mypart);
		return res;
	}

	/** STATIC factory method with a given ID, NOTE: when it's not the
		main worker, vec may be an EMPTY worker. if it's the main processor,
		size must match the vec. */
	pub fn new_dis_vec_with_id(id: u64, main_processor: u64, size: usize, vec: Vec<F>) -> DisVec<F>{
		let dv = DisVec{id: id, vec: vec, b_in_cluster: false, len: size, np: RUN_CONFIG.n_proc as usize, main_processor: main_processor as usize, partition: vec![], real_len: size, real_len_set: false, b_destroyed: false};
		return dv;
	}

	///compute the dotprod: sum a[i]*b[i]
	///requiring two having the same length
	///ONLY the main node has the result 
	pub fn dot_prod(&self, other: &Self)->F{
		let me = RUN_CONFIG.my_rank;
		let np = RUN_CONFIG.n_proc;
		assert!(self.len==other.len, "self.len != other.len");
		assert!(self.main_processor==other.main_processor, "self.len != other.len");
		assert!(self.b_in_cluster, "self not in cluster!");
		assert!(other.b_in_cluster, "self not in cluster!");

		//1. each local one computes
		let mut part_sum = F::zero();	
		for i in 0..self.partition.len(){
			part_sum += self.partition[i] * other.partition[i];
		}

		//2. all report result to main node
		let vec2d = all_to_one_vec(me, self.main_processor, &vec![part_sum]);
		let mut sum = F::zero();
		if me==self.main_processor{
			assert!(vec2d.len()==np, "vec2d.len != np");
			for i in 0..vec2d.len(){
				sum += vec2d[i][0]; 
			}
		}
		
		return sum;
	}
	/// take a sub-list of [0..n] (n not included)
	/// and broadcast to each node so that EACH NODE has
	/// the SAME COPY of the sub-list
	/// assumption n is < 1024 (only used for public inputs)
	pub fn sublist_at_each(&self, n: usize)->Vec<F>{
		let me = RUN_CONFIG.my_rank;
		let world = RUN_CONFIG.univ.world();
		assert!(self.b_in_cluster, "sublist_at_each: call partition first!");
		if me==self.main_processor{
			assert!(n<self.partition.len(), "n: {}>partition.len: {}", n, self.partition.len());
		}
		
		//2. broadcast to everybody
		let send_size = F::zero().serialized_size() * n;
		let mut vec_to_send = vec![0u8; send_size];
		let root_process = world.process_at_rank(self.main_processor as i32);
		if me==self.main_processor{
			vec_to_send = to_vecu8(&self.partition[0..n].to_vec());
		}
		root_process.broadcast_into(&mut vec_to_send);
		assert!(vec_to_send.len()==send_size, "me: {}, vec_to_send.size(): {} != send_size: {}", me, vec_to_send.len(), send_size);

		let vec_ret = from_vecu8::<F>(&vec_to_send, F::zero());
		return vec_ret;
	}

	/// ASSUMPTION: the file exists at all servers
	/// read the file at ALL nodes (by taking chunks)
	pub fn new_from_each_node_from_file(id: u64, main_processor: u64, fpath: &str)->Self{
		//1. read the file and get the number
		let me = RUN_CONFIG.my_rank;
		let np = RUN_CONFIG.n_proc;
		let filepath = fpath.to_string();
		let total_len = read_1st_line_as_u64(&filepath) as usize;
		let unit = total_len/np;
		let me_start = unit * me;
		let me_end = if me==np-1 {total_len} else {unit*(me+1)}; 

		//2. read the vec
		let partition = read_slice_fe_from::<F>(&filepath, me_start, me_end);
		let res = Self::new_from_each_node(id, main_processor, total_len, partition);
		return res;
}

	/// ASSUMPTION: the file exists at all servers
	/// read the file at ALL nodes (by taking chunks)
	/// total_len is the target total_len to scale to
	pub fn new_from_each_node_from_file_and_repartition_to(id: u64, main_processor: u64, fpath: &str, new_total_len: usize)->Self{
		//1. read the file and get the number
		let me = RUN_CONFIG.my_rank;
		let np = RUN_CONFIG.n_proc;
		let filepath = fpath.to_string();
		let total_len = read_1st_line_as_u64(&filepath) as usize;
		// if too small, load separately and then rescale
        //let new_total_len = if total_len<new_total_len/4 {total_len}
        //  else {new_total_len};

		let unit = new_total_len/np;
		let mut me_start = unit * me;
		let mut me_end = if me==np-1 {new_total_len} else {unit*(me+1)}; 
		me_start = if me_start >= total_len {total_len} else {me_start};
		me_end = if me_end >= total_len {total_len} else {me_end};
		let my_part_len = if me==np-1 {unit + new_total_len%np} else {unit};

		//2. read the vec
		let mut partition= read_slice_fe_from::<F>(&filepath, me_start, me_end);
		assert!(my_part_len>=partition.len(), "my_part_len: {} < partition.len(): {}", my_part_len, partition.len());
		let part2_len = my_part_len - partition.len();
		let mut part2 = vec![F::zero(); part2_len];
		partition.append(&mut part2);
		let res = Self::new_from_each_node(id, main_processor, new_total_len, partition);
		return res;
}

	/// constructor. All nodes jointly creates the distributed vector.
	/// they should be passing the same information of id, main_processor, size
	/// the total_len IS the TOTAL SIZE globally. Partition size must match the share 
	pub fn new_from_each_node(id: u64, main_processor: u64, total_len: usize, partition: Vec<F>) -> DisVec<F>{
		//1. validity check
		let (begin, end) = Self::get_share_bounds(RUN_CONFIG.my_rank as u64,
			RUN_CONFIG.n_proc as u64, total_len as u64);
		let share_len = end - begin; 
		if share_len!=partition.len(){panic!("share_len: {} !=partition.len(): {} at node: {}", share_len, partition.len(), id);}
		//2. create it
		let res = DisVec{
			id: id,
			vec: vec![],
			b_in_cluster: true,
			len: total_len,
			np: RUN_CONFIG.n_proc as usize,
			main_processor: main_processor as usize,
			partition: partition,
			real_len: 0,
			real_len_set: false,
			b_destroyed: false,
		};
		return res;
	}

	/** convert a serial to disVec */
	pub fn from_serial(vec: &Vec<F>) -> DisVec<F>{
		let v2 = vec.clone();
		return Self::new_dis_vec(v2);
	}
	
	/** STATIC factory method. generates a random ID */
	pub fn new_dis_vec(vec: Vec<F>) -> DisVec<F>{
		//let mut rng = gen_rng();
		//let id = rng.next_u64();
		let id = 0u64;
		let vlen = (&vec).len();
		let dv = DisVec{id: id, vec: vec, b_in_cluster: false, len: vlen, np: RUN_CONFIG.n_proc, main_processor: 0, partition: vec![], real_len: 0, real_len_set: false, b_destroyed: false};
		return dv;
	}


	/** ALL nodes will get the value located at idx */	
	pub fn get_for_each_node(&self, idx: usize)->F{
		//1. the node who go tthe right range submit the answer
		assert!(self.b_in_cluster, 
			"call to_partition() before get_for_each_node()");
		let me = RUN_CONFIG.my_rank;
		let main = self.main_processor;
		let np = RUN_CONFIG.n_proc;
		let total = self.len;
		assert!(idx<total, "ERROR: idx: {} >=self.len: {}!", idx, total);	
		let (begin, end) = Self::get_share_bounds(me as u64, np as u64, 
			total as u64);
		let world = RUN_CONFIG.univ.world();
		let mut b_skip_send = false;
		let mut value = F::zero();
		if idx>=begin && idx<end{
			let offset = idx-begin;
			value = self.partition[offset];
			let vec_bytes = to_vecu8(&vec![value]);
			if me!=main{
				world.process_at_rank(main as i32).
					send_with_tag(&vec_bytes, me as i32);
			}else{
				b_skip_send = true;
			}
		}
		
		//2. main node broadcast
		if me==main && !b_skip_send{
			let r1 =  world.any_process().receive_vec::<u8>();
			let v = from_vecu8::<F>(&r1.0, F::zero());
			value = v[0];
		}
		let vres1d = broadcast_small_arr(&vec![value], main);
		let value = vres1d[0];		
		return value;
	}

	/** return the i'th share of start and end. Static version.
		NOTE: end is actually NOT included
	*/
	pub fn get_share_bounds(i: u64, n: u64, total_len: u64) -> (usize, usize){
		return get_share_start_end(i, n, total_len);
	}

	/// return [start, end) of share i
	pub fn get_share_bounds_usize(&self, i: usize) -> (usize, usize){
		let (start, end) = get_share_start_end(i as u64, RUN_CONFIG.n_proc as u64, self.len as u64);
		return (start as usize, end as usize);
	}

	/** return the i'th share of start and end. 
		NOTE: end is actually not included.
	*/
	pub fn get_share_boundary(&self, i: u64, n: u64) -> (usize, usize){
		let total_len = self.len as u64;
		return DisVec::<F>::get_share_bounds(i, n, total_len);
	}

	/** return the i'th share of the vec. */
	fn get_share(&self, i: u64, n: u64) -> Vec<F>{
		let (start,end) = self.get_share_boundary(i, n);
		let vec_share = &self.vec[start..end];
		return vec_share.to_vec();
	}

	/** return the process who's responsible for dispatching
	and merge partitions */
	pub fn get_main_processor(&self) -> u64{
		return self.main_processor as u64;
	}

	/** return the partition information: start_idx and size
		for partition located at rank */
	pub fn get_partition_info(&self, rank: u64) -> (u64, u64){
		let (start_idx, end_idx) = self.get_share_boundary(rank, self.np as u64);
		let size = end_idx - start_idx;
		return (start_idx as u64, size as u64);
	}

	/** save the data to cluster. main_rank: id%world.size().
		Assuming that the processor at main_rank will 
		perform the job.
	 */
	pub fn to_partitions(&mut self, univ: &Universe){
		//1. get the n processors and decide main thread
		if self.b_in_cluster{ return; }
		let world = univ.world();
		let np = RUN_CONFIG.n_proc;
		let n = world.size() as u64;
		//let rank = world.rank() as u64;
		//let main_rank = self.get_main_processor();
		self.np = n as usize;

/* RECOVER IF NOT WORKING
		//2. main thread: distribute and send copies
		if rank==main_rank{
			for i in 0..n{
				let vshare = self.get_share(i, n);
				//println!("TO processor {}", i);
				if i==rank{
					self.partition = vshare;
				}else{
					let vec_bytes= to_vecu8(&vshare);
					world.process_at_rank(i as i32).
						send_with_tag(&vec_bytes, self.id as i32);
				}
			}
		}else{//as the receiver, sav ethe vec
			let r1 = world.any_process().receive_vec::<u8>();
			let v = from_vecu8::<F>(&r1.0, F::zero());
			self.partition = v;
		}
*/
		let mut sizes = vec![self.len/np; np];
		sizes[np-1] += self.len%np;
		let sample = F::zero();
		self.partition = one_to_all(self.main_processor as usize, &self.vec, &sizes, sample);

		self.b_in_cluster = true;
		self.vec.clear();
		self.vec = Vec::new();
		RUN_CONFIG.better_barrier("to_partitions");
	}

	/** used for DEBUG. all reports their knowledge of the main processor
		to node 0, and node 0 check if consistent (only one main node)
		If not, node 0 will panic */
	pub fn check_main(&self){
		log(LOG1, &format!("DEBUG USE 88888: check_main: SLOW!"));
		let me = RUN_CONFIG.my_rank;
		let vres = all_to_one(me, 0, self.main_processor as u64);
		let main1 = vres[0];
		if me==0{//check
			for i in 0..vres.len(){
				let main_i = vres[i];
				assert!(main_i==main1, 
					"node {} has {} != {} at 0", i, main_i, main1);
			}
		}
		RUN_CONFIG.better_barrier("wait for check");
	}

	/** From the main_node broadcast the len attribute to all other nodes.
		This function should be called at ALL NODES.
	 */
	pub fn synch_size(&mut self){
		//1. get the n processors and decide main thread
		let me = RUN_CONFIG.my_rank;
		let mut total_len = self.len;
		if self.b_in_cluster {
			total_len = 0;
			let vres = all_to_one(me, self.main_processor as usize, self.partition.len() as u64);
			for x in vres{ total_len += x as usize;}	
		}

		//2. main thread: distribute and send copies
		let vres = slow_broadcast_small_arr::<usize>(&vec![total_len],self.main_processor);
		self.len = vres[0];	
		RUN_CONFIG.better_barrier("sync_size");
	}
	/** merge the data to main processor. 
		After merge, the main processor's vec has the entire data.
		ONLY PRODUCES
		the data at main_processor; other nodes cooperate
		but did NOT get the right answer.
	The node performs the assembly at the main_rank. All other nodes
	send in the partition (and do NOT assemble).		
	This function is ONLY mainly used for testing purpose.
	Assume the settingo of the world is the same as 
	when the partition is generated.
	 */
	pub fn from_partitions(&mut self, univ: &Universe){
		//1. get the n processors and decide main thread
		let world = univ.world();
		let rank = world.rank() as u64;
		let main_rank = self.get_main_processor();

		//1. EVERYONE (EXCLUDING the main_rank) sends out the partition 
		if !self.b_in_cluster{ return; }
		else{
			let vec = self.collect_from_partitions(univ);
			if rank==main_rank{self.vec = vec;}
		}
	}
	
	/// ALL NODES build up the same copy of chunk by collection
	/// from all shares [NOTE: all nodes put the same result into
	/// vec_res] - this is different from collect_partition_data
	/// which only returns valid data at main processor
	pub fn all_node_collect_chunk(&self, 
		rng: &(usize, usize), vec_res: &mut Vec<F>){
		//1. prepare tosend for each node
		if !self.b_in_cluster { panic!("call to_partition first!"); }
		let np = RUN_CONFIG.n_proc as usize;
		let me = RUN_CONFIG.my_rank as usize;
		let mut vsend: Vec<Vec<F>> = vec![vec![]; np];
		let my_share = self.get_share_bounds_usize(me);

		//2. send and receive by asynch broadcast
		for i in 0..np{
			let iset = share_intersect(&my_share, &rng);
			if iset.1-iset.0>0{
				vsend[i] = self.partition[
					iset.0-my_share.0..iset.1-my_share.0].to_vec();
			}
		}
		let vrecv = nonblock_broadcast(&vsend, np as u64, &RUN_CONFIG.univ, F::zero());

		//3. build up the copy from each node
		let mut idx = 0;
		for i in 0..np{
			let row = &vrecv[i];
			for j in 0..row.len(){
				vec_res[idx] = row[j];
				idx+=1;
			}
		}
		RUN_CONFIG.better_barrier("all_node_collect_chunk");
	}

	/** convert to serial. assume memory is ok 
		NOTE: only return correct result at main processor 
	*/
	pub fn to_serial(&self) -> Vec<F>{
		return self.collect_from_partitions(&RUN_CONFIG.univ);
	}

	/** collect the partition data to cluster. ONLY PRODUCES
		the data at main_processor; other nodes cooperate
		but did NOT get the right answer.
		If not in cluster already -> just return the vec component.
		NEEDS TO BE CALLED AT ALL NODES!
	 */
	pub fn collect_from_partitions(&self, univ: &Universe)->Vec<F>{
		//0. check if in cluster
		let world = univ.world();
		if !self.b_in_cluster{
			//DO NOT CALL barrier() as this is a SINGLE mode call
			//RUN_CONFIG.better_barrier("collect_from_partitions_1");
			//if RUN_CONFIG.my_rank!=self.main_processor{
			//	panic!("Called collect_partitions at non main-processor node when not in cluster mode!");
			//}
			return self.vec.clone();
		}
		//1. get the n processors and decide main thread
		let n = world.size() as u64;
		let rank = world.rank() as u64;
		let main_rank = self.get_main_processor();
		let mut vec_ret = if rank==main_rank {vec![F::zero(); self.len]}
			else {vec![F::zero(); 1]};

		//1. EVERYONE (EXCLUDING the main_rank) sends out the partition 
		if rank != main_rank{
			let v = &self.partition;
			let vbytes = to_vecu8(&v);	
			//println!("DEBUG USE 301: send {} --> {}, vec: {}, self.len(): {}, self.partitionlen: {}", rank, main_rank, vbytes.len(), self.len, self.partition.len());
			world.process_at_rank(main_rank as i32).send_with_tag(&vbytes, 
				rank as i32);
		}

		//2. main thread ONLY: receive it
		if rank==main_rank{
			for _i in 0..n-1{
				let r1 = world.any_process().receive_vec::<u8>();
				let src_rank = r1.1.tag() as i32;
				let (start, end) = self.get_share_boundary(src_rank as u64, n);
				let v = from_vecu8::<F>(&r1.0, F::zero());
				for j in start..end{
					vec_ret[j] = v[j-start];
				}
			}
			//COPY over my own stuff
			let (start, _end) = self.get_share_boundary(rank as u64, n);
			for k in 0..self.partition.len(){
				vec_ret[k+start] = self.partition[k];
			}		
		}
		RUN_CONFIG.better_barrier("collect_from_partitions_2");
		return vec_ret;
	}

	/** return the chucnk of data storing on my vec or partition, based
	on b_in_cluster.
	When end_off < start_off or start_off greater than vec.len() or
	partition.len() return an empty vec, pad zeros
	if there are not sufficient elements */
	pub fn take_data(&self, start_off: usize, end_off: usize) -> Vec<F>{
		return self.take_data_worker(start_off, end_off, false);
	}

	/** return the chucnk of data storing on my vec or partition, based
	on b_in_cluster.
	When end_off < start_off or start_off greater than vec.len() or
	partition.len() return an empty vec, pad zeros
	if there are not sufficient elements,
	b_reverse: if it's reverse mode */
	pub fn take_data_worker(&self, start_off: usize, end_off: usize, b_reverse: bool) -> Vec<F>{
		//1. sanitize data input
		if end_off<start_off {return vec![];}
		let len = if self.b_in_cluster {self.partition.len()} else {self.vec.len()};
		let end_off = if end_off<len {end_off} else {len};

		//2. calculate the idx to chop off data
		let mut part1: Vec<F>;	
		part1 = if self.b_in_cluster {self.partition[start_off..end_off].to_vec()} else {self.vec[start_off..end_off].to_vec()};
		if b_reverse {part1.reverse();}

		//3. padd data
		let cur_size = part1.len();
		let exp_size = end_off-start_off;
		if cur_size==exp_size{
			return part1;
		}else{
			//println!("DEBUG USE 888: exp_size-cur_size: {}, start_off: {}, end_off: {}, this.partition.len: {}, this.vec.len:{}, bin_cluster: {}", exp_size-cur_size, start_off, end_off, self.partition.len(), self.vec.len(), self.b_in_cluster);
			let mut vzero = vec![F::zero(); exp_size-cur_size];
			if !b_reverse{
				part1.append(&mut vzero);
				return part1;
			}else{
				vzero.append(&mut part1); //padded zero on the left
				return vzero;
			}
		}
	}
	/** calculate the size for re_partition from node src to node j.
	return a vector of 4 numbers
	[start_off, end_off, start, end]
	the start_off and end_off are the relative OFFSET of the data
	INSIDE the partition of src node, the (start,end) are
	the corresponding ABSOLUTE location in the entire distributed
	vector. The END is actulaly NOT included. i.e.
	len = end- start;
	If NO DATA to send, set all to 0  */
	pub fn gen_repart_plan(&self, src: usize, dst: usize, target_len: usize)
		->(usize, usize, usize, usize){
		return self.gen_repart_plan_worker(src, dst, target_len, false, false);
	}

	/** calculate the size for re_partition from node src to node j.
	return a vector of 4 numbers
	[start_off, end_off, start, end]
	the start_off and end_off are the relative OFFSET of the data
	INSIDE the partition of src node, the (start,end) are
	the corresponding ABSOLUTE location in the entire (DESTINATION) distributed
	vector. The END is actulaly NOT included. i.e.
	len = end- start;
	If NO DATA to send, set all to 0.
	b_reverse: if it's reverse mode.
	b_use_real_len_as_src_len: whether to use the real_len as source len
	so to chop off leading zeros (used for poly division only)
	  */
	pub fn gen_repart_plan_worker(&self, src: usize, dst: usize, 
target_len: usize, b_reverse: bool, b_use_real_len_as_src_len: bool)
		->(usize, usize, usize, usize){
		let np = RUN_CONFIG.n_proc;
		if !self.b_in_cluster && self.main_processor!=src{
			return (0,0,0,0); //nothing to send
		}
		if b_use_real_len_as_src_len && !self.real_len_set{
			panic!("gen_report_plan_worker ERR: set real_len first!");
		}

		//assuming has valid start, end (i.e. either in cluster or src is the
		//main processor for local (non-incluster) mode
		let src_len = if b_use_real_len_as_src_len {self.real_len} else {self.len};	
		let (ss, se) = if self.b_in_cluster{self.get_share_boundary(src as u64, np as u64)} else {(0, self.len)};
		let mut src_start = if !b_reverse {
			if ss >= src_len {src_len} else {ss}
		} else { if se >= src_len {0} else {src_len - se} };
		let mut src_end = if !b_reverse {
			if se >= src_len {src_len} else {se}
		} else { if ss >= src_len {0} else {src_len-ss}};
		//now the src_start and src_end are the ABSOLUTE LOCATION
		//in the SOURCE partition. If reverse direction, needs to shift
		//to right
		if b_reverse && src_len<target_len{
			let diff = target_len - src_len;
			src_start += diff;
			src_end += diff;
		}
		// src_start and src_end reflects the LOGICAL SECTION
		// of the current node in the DESTINATATION.

		let (dest_start, dest_end) = DisVec::<F>::get_share_bounds(dst as u64, np as u64, target_len as u64);

		let start = if src_start<dest_start {dest_start} else {src_start};
		let end = if src_end<dest_end {src_end} else {dest_end};
		if end>start{
			let mut start_off = start-src_start;
			let mut end_off = end-src_start;
			//map it back to real position
			if b_reverse{
				if src_len<target_len{
					let diff = target_len - src_len;
					(start_off, end_off) = (src_len + diff- end - ss,
							src_len+diff-start - ss);
				}else{
					(start_off, end_off) = (src_len-end -ss, src_len-start-ss);
				}
			}
			return (start_off, end_off, start, end);
		}else{
			return (0,0,0,0); //nothing to send
		}
	}  
	/** target len: is the target total len of all partition vectors.
	It's the total of (degree + 1) from all nodes. We don't check if
	target_len is LESS THAN current dvec.len. In that case, it might
	chop off non-zero items. */
	pub fn repartition(&mut self, target_len: usize){
		self.repartition_worker(target_len, false, false);
	}

	/** target len: is the target total len of all partition vectors.
	It's the total of (degree + 1) from all nodes.
	b_rev: whether if it's reverse mode,
	b_use_real_len_as_src_len: whether use real_len as the source len
		so that to chop off leading zeros (used for division only)
	 */
	pub fn repartition_worker(&mut self, target_len: usize, b_rev: bool, b_use_real_len_as_src_len: bool){
		//1. build up the data to send
		let np = RUN_CONFIG.n_proc;
		let my_rank = RUN_CONFIG.my_rank;
		let mut vsend: Vec<Vec<F>> = vec![vec![]; np];
		//println!("*****\n DEBUG USE 700: DisPoly: {}, main_processor: {},  repartition: my_rank: {}, target_len: {}, cur_len: {}, in_cluster: {}", self.id, self.main_processor, my_rank, target_len, self.len, self.b_in_cluster);
		for i in 0..np{
			let (start_offset, end_offset, _dest_start, _dest_end) = self.
				gen_repart_plan_worker(my_rank, i, target_len, b_rev, b_use_real_len_as_src_len);
			//println!(" !!!! DEBUG USE 702: DisPoly: {} REPARTITION, my_rank: {} -> {}, start: {}, end: {}. dest_start: {}, dest_end: {}, target_len: {},  Main_processor: {}", self.id, my_rank, i, start_offset, end_offset, _dest_start, _dest_end, target_len, self.main_processor);
			let row = self.take_data_worker(start_offset, end_offset, b_rev); 
			vsend[i] = row;
		}

		//2. broadcast and receive
		let vrecv = nonblock_broadcast(&vsend, np as u64, &RUN_CONFIG.univ, F::zero());

		//3. assemble data
		let (my_start, my_end) = DisVec::<F>::get_share_bounds(my_rank as u64, np as u64, target_len as u64); 
		let my_part_len = my_end - my_start;
		self.partition = vec![F::zero(); my_part_len];
		for i in 0..np{
			let (_, _, s, e) = self.gen_repart_plan_worker(
				i, my_rank, target_len, b_rev, b_use_real_len_as_src_len);
			if e>s{//valid entry
				let my_s = s - my_start;
				let my_e = e - my_start;
				for k in 0..my_e-my_s{
					if k>=vrecv[i].len(){
						println!("reparition_worker: me: {}, k: {}, my_e: {}, my_s: {}, i: {}, self.partition.len: {}, vrecv[i].len: {}, vrecv.len: {}", RUN_CONFIG.my_rank, k, my_e, my_s, i, self.partition.len(), vrecv[i].len(), vrecv.len());
					}
					self.partition[my_s+k] = vrecv[i][k]; 
				}
			}
		}
		
		//4. set the new target_len
		self.len = target_len;
		self.b_in_cluster = true;
		self.real_len_set = false;
		RUN_CONFIG.better_barrier("repartition");
	}

	/** write each nodes data to file */
	pub fn write_to_each_node(&self, fname: &String){
		let my_rank = RUN_CONFIG.my_rank;
		let newfname = format!("{}.node_{}", fname, my_rank);
		assert!(self.b_in_cluster, "make disploy distributed first!");
		let v8 = to_vecu8(&self.partition);
		write_vecu8(&v8, &newfname);
		RUN_CONFIG.better_barrier("write_coefs_to_file");
	}

	/** read partition from file. Note: each processor read its own
	piece. NOTE: need to provide total_size in advance! (because all nodes
	need to know the correct number and be consistent. 
	*/
	pub fn read_each_node_from_file(fname: &String, total_size: usize)->DisVec<F>{
		let zero = F::zero();
		let mut ds = DisVec::new_dis_vec_with_id(0,0,0,vec![]);
		ds.b_in_cluster = true;
		ds.len = total_size;
		let my_rank = RUN_CONFIG.my_rank;
		let newfname = format!("{}.node_{}", fname, my_rank);
		let vec_serialized = read_vecu8(&newfname);
		ds.partition = from_vecu8(&vec_serialized, zero);
		RUN_CONFIG.better_barrier("read_node_from_file");
		return ds;
	}

	/** Let each partition be [a0, ..., an]. Evaluate the result
	of (r+a0)...(r+an). Send back results to main node.
	ONLY main node returns the correct vec. But the function
	needs to be called at ALL NODES.
	Return: a vector of size np.
	res[0]: the of evaluating first partition
	res[1]: the result of evaluating first + second partition
	...
	res[np-1]: the actual result of evaluating the entire vector
	*/
	pub fn eval_chunk_binacc(&self, r: &F)->Vec<F>{
		let np = RUN_CONFIG.n_proc; 
		let mut vec = vec![F::zero(); np];	
		let rank = RUN_CONFIG.my_rank;
		let main_rank = self.main_processor;
		let world = RUN_CONFIG.univ.world();
		if !self.b_in_cluster {
			panic!("ERR eval_chunk_binacc: call to_partitions() first!");
		}

		//1. evaluate my own partition
		let mut res = F::one();
		for i in 0..self.partition.len(){
			res *= self.partition[i] + r;
		}
		vec[rank] = res;
		//2. send and receive results
		if rank!=main_rank{
			let vec_bytes= to_vecu8(&vec![res]);
			world.process_at_rank(main_rank as i32).
						send_with_tag(&vec_bytes, rank as i32);
		}else{//as the receiver, sav ethe vec
			for _i in 0..np-1{
				let r1 = world.any_process().receive_vec::<u8>();
				let id = r1.1.tag();
				let v = from_vecu8::<F>(&r1.0, F::zero())[0];
				vec[id as usize] = v;
			}
			//further processing: vec[np-1] will be the binacc result of the entire.
			for i in 1..np{
				vec[i] = vec[i-1]*vec[i];
			}
		}
		RUN_CONFIG.better_barrier("eval_binacc");
		return vec;
	}

	/** works both in cluster and non cluster mode.
		Needs to be called at ALL NODES.
		AFTER processing ALL nodes get the right number
	*/
	pub fn get_real_len(&self) -> usize{
		if self.real_len_set {return self.real_len;}

		//1. each node computes the highest non_zero_idx
		let np = RUN_CONFIG.n_proc;
		let me = RUN_CONFIG.my_rank;
		let main = self.get_main_processor() as usize;
		let vec = if self.b_in_cluster {&self.partition} else {&self.vec};
		let part_len = vec.len();
		let mut idx_nzero = if part_len<1 {0} else {part_len-1};
		let mut b_found_it= false;
		for i in 0..part_len{
			idx_nzero = part_len - i - 1;
			if !vec[idx_nzero].is_zero() {
				b_found_it = true;
				break;
			}
		}
		let mut computed_len = if b_found_it {idx_nzero + 1} else {0};

		if self.b_in_cluster{
			let vres= all_to_one(me, main, computed_len as u64);
			if me==main{
				let part_len = self.len/np;
				for i in 0..np{
					let idx = np - i - 1;
					if vres[idx]>0{
						computed_len = part_len * idx + (vres[idx] as usize);
						break;
					}
				}
			}
		}
		let mut timer = Timer::new();
		timer.start();
		//let vres = broadcast_small_arr(&vec![computed_len],self.main_processor);
		let vres = slow_broadcast_small_arr(&vec![computed_len],self.main_processor);
		let real_len = vres[0];

		//let (_, real_len)= Self::send_tuple(self.main_processor, computed_len, computed_len);
		//println!("DEBUG USE 3021: dis_vec_id: {}, main_processor: {}, me: {}, real_len: {}", self.id, self.main_processor, me, real_len);	
		//RUN_CONFIG.better_barrier("REAL_LEN_TO REMOVE");

		return real_len;
	}

	pub fn set_real_len(&mut self){
		self.id = self.main_processor as u64; //make it consistent
		if self.real_len_set {
			return;
		}
		let rlen = self.get_real_len();
		self.real_len_set = true;
		self.real_len = rlen;
	}

	//all nodes update the len from main
	pub fn broadcast_len_from_main(&mut self){
		let send_size = 8*2;
		let world = RUN_CONFIG.univ.world();
		let me = RUN_CONFIG.my_rank;
		let mut vec_to_send = vec![0u8; send_size];
		let root_process = world.process_at_rank(self.main_processor as i32);
		if me==self.main_processor{
			vec_to_send = to_vecu8::<usize>(&vec![self.len, self.real_len]);
		}
		root_process.broadcast_into(&mut vec_to_send);
		assert!(vec_to_send.len()==send_size, "me: {}, vec_to_send.size(): {} != send_size: {}", me, vec_to_send.len(), send_size);

		let vec_ret = from_vecu8::<usize>(&vec_to_send, 0usize);
		self.len = vec_ret[0];
		self.real_len = vec_ret[1];
	}

	//all nodes update the len from main
	//return (len, real_len);
	pub fn send_tuple(sender: usize, num1: usize, num2: usize)->(usize, usize){
		let send_size = 8*2;
		let world = RUN_CONFIG.univ.world();
		let me = RUN_CONFIG.my_rank;
		let mut vec_to_send = vec![0u8; send_size];
		let root_process = world.process_at_rank(sender as i32);
		if me==sender{
			vec_to_send = to_vecu8::<usize>(&vec![num1, num2]);
		}
		root_process.broadcast_into(&mut vec_to_send);
		let vec_ret = from_vecu8::<usize>(&vec_to_send, 0usize);
		return (vec_ret[0], vec_ret[1]);
	}

	/** generate a sublist starting [start, end) */
	pub fn subvec(&self, start: usize, end: usize)->Self{
		assert!(self.b_in_cluster, "subsec: call partition first!");
		let me = RUN_CONFIG.my_rank;
		let np = RUN_CONFIG.n_proc;
		let new_part = subvec(&self.partition, self.len, me, np, start, end);
		let nd = Self{
			id: self.id,
			vec: vec![],
			b_in_cluster: true,
			len: end-start,
			np: self.np,
			main_processor: self.main_processor,
			partition: new_part,
			real_len_set: true ,
			real_len: end-start,
			b_destroyed: false,
		};
		return nd;
	}
	
}


#[cfg(test)]
mod tests {
	use crate::tools::*;
	use crate::poly::dis_vec::*;
	use crate::poly::dis_poly::*;
	use self::ark_poly::Polynomial;
	//use mpi::point_to_point as p2p;
	//use mpi::topology::Rank;
	//use mpi::traits::*;
	type Fr381=ark_bls12_381::Fr;
	use self::ark_ff::Zero;
	//use crate::profiler::config::*;

/*
	#[test]
	fn test_serial_v8(){
		let v1 = rand_arr_field_ele::<Fr381>(1024, get_time());
		let v2 = to_vecu8(&v1);
		let v3 = from_vecu8::<Fr381>(&v2, Fr381::zero());
		assert!(vec_equals(&v1, &v3), "failed serialization to and from v8");
	}

	#[test]
	fn test_to_partitions1(){
		let v1 = rand_arr_field_ele::<Fr381>(4, get_time());
		let v2 = v1.clone();
		let mut dv1 = DisVec::new_dis_vec_with_id(0, 0, v1.len(), v1);
		dv1.to_partitions(&RUN_CONFIG.univ);
		dv1.from_partitions(&RUN_CONFIG.univ);
		if RUN_CONFIG.my_rank==dv1.main_processor{
			assert!(vec_equals(&v2, &dv1.vec), "failed test_to_partitions");
		}
	}

	#[test]
	fn test_eval_binacc(){
		let v1 = rand_arr_field_ele::<Fr381>(1025, get_time());
		let r = Fr381::from(123456u64); //HAS TO BE SAME ON ALL NODES
		let p = DisPoly::<Fr381>::binacc_poly(&v1);
		let expected = p.evaluate(&r);
		let mut dv1 = DisVec::new_dis_vec_with_id(0, 0, v1.len(), v1);
		dv1.to_partitions(&RUN_CONFIG.univ);
		let vec = dv1.eval_chunk_binacc(&r);
		let res = vec[vec.len()-1]; //the last is the expected for entire disvec.
		if RUN_CONFIG.my_rank==dv1.main_processor{
			assert!(res==expected, "FAILED eval_chunk_binacc. expected: {}, actual: {}", expected, res);
		}
	}

	#[test]
	fn test_real_len(){
		let np = RUN_CONFIG.n_proc;
		let mut v1 = vec![Fr381::zero(); np*128];
		let mut v2 = vec![Fr381::zero(); np*128];
		let idx = (np-2)*128 + 12;
		let idx2 = (np-2)*128 + 1;
		v1[idx] = Fr381::from(3u64);
		v2[idx2] = Fr381::from(3u64);
		let mut dv1 = DisVec::new_dis_vec_with_id(0, 0, v1.len(), v1);
//		let mut dv2 = DisVec::new_dis_vec_with_id(0, 0, v2.len(), v2);
		dv1.to_partitions(&RUN_CONFIG.univ);
//		dv2.to_partitions(&RUN_CONFIG.univ);
		dv1.set_real_len();
//		dv2.set_real_len();
		assert!(dv1.real_len_set==true, "real_len_set is not true");
		assert!(dv1.real_len==idx+1,
			"real_len: {} !=(np-2)*128+12+1: {}!", dv1.real_len, idx+1);
//		assert!(dv2.real_len_set==true, "real_len_set is not true");
//		assert!(dv2.real_len==idx2+1,
//			"real_len: {} !=(np-2)*128+12+1: {}!", dv2.real_len, idx2+1);
	}
*/

	
	#[test]
	fn test_sublist(){
		let me = RUN_CONFIG.my_rank;
		let mut svc1 = vec![];
		let mylen = 2048;
		for i in 1..mylen+1{ svc1.push(Fr381::from(i as u64)); }
		let svc = svc1.clone();
		let mut dvc = DisVec::<Fr381>::new_dis_vec_with_id(0, 0, mylen, svc1);
		dvc.to_partitions(&RUN_CONFIG.univ);
		let test_cases = vec![
			(1033, 2044),
			(20, 700), 
			(1, 5), 
			(0, 3), 
			(2044, 2045),
			(733, 1566),
			(2, 721),
			(1, 1591),
			(2041, 2047)
		];
		for tc in test_cases{
			let (start, end) = (tc.0, tc.1);
			let ds1 = dvc.subvec(start, end);
			let s1 = ds1.to_serial();
			if me==0{ 
				let s2 = svc[start..end].to_vec();
				if s1!=s2{
					dump_vec("s1: ", &s1);
					dump_vec("s2: ", &s2);
				}
				assert!(s1==s2, "s1!=s2 for [{}, {})", start, end); 
			} 
		}
	}
}
