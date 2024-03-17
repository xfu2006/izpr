/** 
	Copyright Dr. Xiang Fu

	Author: Dr. Xiang Fu
	All Rights Reserved.
	Created: 05/24/2022
	Revised: 07/13/2022 (added dummy version of mul and div)
	Revised: 09/01/2022 (re-implemened add, sub, mul)
	Revised: 09/08/2022 (added mod_by and mul_by_xk)
	Revised: 01/07/2023 (improved binacc - from_mainnode)
	Distributed Polynomial Object
	Given n processors, distributed to n processors.
*/
#[cfg(feature = "parallel")]
use ark_std::cmp::max;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

extern crate ark_ff;
extern crate ark_poly;
extern crate ark_std;
extern crate mpi;
extern crate ark_serialize;

use self::ark_ff::{PrimeField,Zero};
use self::ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
use self::ark_poly::{Polynomial,DenseUVPolynomial, univariate::DensePolynomial};
//use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use self::ark_std::log2;
use crate::tools::*;
use self::ark_std::rand::Rng;
use poly::common::*;
use super::serial::*;
use std::convert::TryInto;
//use std::collections::HashMap;
//use ark_std::rand::RngCore;

//use mpi::point_to_point as p2p;
//use mpi::topology::Rank;
use self::mpi::traits::*;
//use mpi::environment::*;
//use std::time::{SystemTime};
//use ark_ff::{PrimeField};
//use ark_std::rand::rngs::StdRng;
//use ark_std::rand::SeedableRng;
//use std::time::{UNIX_EPOCH};
//use ark_bls12_381::Bls12_381;
//use ark_ec::{PairingEngine};


use crate::profiler::config::*;
use super::dis_vec::*;
use super::disfft::*;

/** Distributed Polynomial */
#[derive(Clone)]
pub struct DisPoly<F:PrimeField+CanonicalSerialize+CanonicalDeserialize+Clone> {
	/** a random ID */
	pub id: u64, 
	/** distributed vector of co-efficients */
	pub dvec: DisVec<F>,
	/** whether it is zero */
	pub is_zero: bool,
	/** whether it is one */
	pub is_one: bool,
}

/// Data structure for feea_new
/// When level (counting from bottom) < feea_new_dis_bar
/// all DisPoly have not beein distributed to nodes yet,
/// just stored at main node
struct FeeaData<F:PrimeField>{
	/** the bin_acc constructed for each node at each level,
	level 0 has 2^levels Dispoly elements, level 1 has half size, ... 
	last level has one DisPoly*/ 
	pub vec_p1: Vec<Vec<DisPoly<F>>>,
	/** the second polynomial for each node at each level*/
	pub vec_p2: Vec<Vec<DisPoly<F>>>,
	/** the Bizout coeffts, s.t. s*p1 + t*p2 = 1 */
	pub vec_s: Vec<Vec<DisPoly<F>>>,
	pub vec_t: Vec<Vec<DisPoly<F>>>,
	/** u*p1 + p2 = p2 at parent level */
	pub vec_u: Vec<Vec<DisPoly<F>>>,
}

fn create_vec_dispoly<F:PrimeField>(size: usize)->Vec<DisPoly<F>>{
	let me = RUN_CONFIG.my_rank;
	let mut vec = vec![];
	for _i in 0..size{
		let v = vec![F::from(1u64)];
		let dv = DisVec::<F>::new_dis_vec_with_id(0, me as u64, 1, v); 
		let dp = DisPoly::<F>{id: 0, dvec: dv, is_one: false, is_zero: false};
		vec.push(dp);
	}
	return vec;
	
}

fn create_2dvec_dispoly<F:PrimeField>(levels: usize)->Vec<Vec<DisPoly<F>>>{
	let mut vec = vec![];
	let mut size = 1<<(levels-1); 
	for _i in 0..size{
		let vpoly = create_vec_dispoly::<F>(size);
		vec.push(vpoly);
		size /= 2;
	}
	return vec;
	
}
impl <F:PrimeField> FeeaData<F>{
	pub fn new(levels: usize)->FeeaData<F>{
		let res = FeeaData::<F>{
			vec_p1: create_2dvec_dispoly::<F>(levels),
			vec_p2: create_2dvec_dispoly::<F>(levels),
			vec_s: create_2dvec_dispoly::<F>(levels),
			vec_t: create_2dvec_dispoly::<F>(levels),
			vec_u: create_2dvec_dispoly::<F>(levels),
		};
		return res;
	}
}

impl <F:PrimeField+CanonicalSerialize+CanonicalDeserialize+Clone> DisPoly <F>{
	/** release memory */
	pub fn destroy(&mut self){
		self.dvec.destroy();	
	}

	/** reset flags */
	pub fn reset_flags(&mut self){
		self.is_zero = false;
		self.is_one= false;
	}
	/** return the degree */
	pub fn degree(&mut self)->usize{
		self.dvec.set_real_len();
		let real_len = self.dvec.real_len;
		let res = if real_len>0 {real_len-1} else {0};	
		return res;
	}
	/** repartition to new size (by padding zeros) and re-distributing load */
	pub fn repartition(&mut self, new_size: usize){
		//if self.dvec.len>new_size{
		//	panic!("repartition ERR: new_size:{} < dvec.len:{}!", new_size, self.dvec.len);
		//}
		self.dvec.repartition(new_size);
	}

	/** force to shrink to real_len */
	pub fn repartition_to_real_len(&mut self){
		let b_perf = false;
		let old_n = self.dvec.len;
		self.dvec.set_real_len();
		let new_n = self.dvec.real_len;	
		self.dvec.repartition(new_n);
		self.dvec.real_len_set = true;
		if b_perf {log(LOG1, &format!("-- Repartition_to_real_len: me: {}, old_len: {}, new_len: {}, real_len: {}, real_len_set: {}", RUN_CONFIG.my_rank, old_n, new_n, self.dvec.real_len, self.dvec.real_len_set));}
	}
	/** Make sure the node has sufficient memory.
		Note it ONLY returns valid response at MAIN PROCESSOR.
		For other nodes: return vec![] */
	pub fn to_serial(&self) -> DensePolynomial<F>{
		let mut d2 = self.clone();
		let mycoefs = d2.coefs();
		let p = DensePolynomial::<F>::from_coefficients_vec(mycoefs);
		return p;
	}

	/** ALL NODES will get the value answer */
	pub fn to_serial_at_each_node(&self) -> DensePolynomial<F>{
		let mut d2 = self.clone();
		let degree = d2.dvec.get_real_len() - 1; 
		let n = degree + 1;
		let mycoefs = d2.coefs();
		let main_id = self.dvec.main_processor;
		let me = RUN_CONFIG.my_rank;
		if main_id==me{
			assert!(n==mycoefs.len(), "n: {} != mycoefs.len(): {}", n, mycoefs.len());
		}
		//1. main node broadcast
		let send_size = F::zero().serialized_size() * n;
		let mut vec_to_send = vec![0u8; send_size];
		let world = RUN_CONFIG.univ.world();
		let root_process = world.process_at_rank(self.dvec.main_processor as i32);
		if me==self.dvec.main_processor{
			vec_to_send = to_vecu8(&mycoefs);
		}
		root_process.broadcast_into(&mut vec_to_send);
		assert!(vec_to_send.len()==send_size, "me: {}, vec_to_send.size(): {} != send_size: {}", me, vec_to_send.len(), send_size);

		let coefs= from_vecu8::<F>(&vec_to_send, F::zero());

		//2. all other nodes get it the broadcast
		let p = DensePolynomial::<F>::from_coefficients_vec(coefs);
		return p;
	}

	/// enforce to partitions
	pub fn to_partitions(&mut self){
		if !self.dvec.b_in_cluster{
			self.dvec.to_partitions(&RUN_CONFIG.univ);
		}
	}

	/** Construct a new DisPoly from a serial. At this moment not 
		partitioned yet. ONLY the main processor builds it. Other
		nodes has the length info but an empty dvec. NOTE: plen is the
		degree+1 (or a nubmer higher than it).
	 */
	pub fn from_serial(id: u64, p: &DensePolynomial<F>, plen: usize) 
		-> DisPoly<F>{
		let np = RUN_CONFIG.n_proc;
		let main_processor = id % (np as u64);
		let my_rank = RUN_CONFIG.my_rank as u64;
		let mut vec = if my_rank==main_processor {p.coeffs.clone()} else {vec![]};
		if vec.len()>plen{ 
			panic!("ERR from_serial: NODE {}: vec.len: {} > plen: {}", my_rank ,vec.len(), plen); 
		}
		if vec.len()<plen{
			vec.resize(plen, F::zero());
		} 
		assert!(vec.len()==plen, "vec.len(): {} != plen: {}", vec.len(), plen);
		let mut dv = DisVec::<F>::new_dis_vec_with_id(id, main_processor, plen, vec);
		dv.broadcast_len_from_main();
		let ret = DisPoly{id: id, dvec: dv, is_one: false, is_zero: false};
		return ret;	
	}

	pub fn check_same_main_processor(p1: &mut DisPoly<F>, p2: &mut DisPoly<F>){
		if p1.dvec.main_processor!=p2.dvec.main_processor{
			p2.reset_main_processor(p1.dvec.main_processor);
		}
	}

	/** reset the main processor. ASSUMPTION: all nodes have
	agreement on the main_processor.
	 */
	pub fn reset_main_processor(&mut self,  new_val: usize){
		//ENALBE IT for debugging
		//self.dvec.check_main();
		let me = RUN_CONFIG.my_rank;
		let old_val = self.dvec.main_processor;
		if old_val==new_val{
			self.dvec.synch_size();
			return;
		}
		if self.dvec.b_in_cluster{
			self.dvec.synch_size();
			self.dvec.main_processor = new_val;
		}else{
			//old --> new
			let world = RUN_CONFIG.univ.world();
			if me== self.dvec.main_processor{
				let vbytes = to_vecu8(&self.dvec.vec);
				world.process_at_rank(new_val as i32).
					send_with_tag(&vbytes, me as i32);
			}
			if me== new_val{//wait
				let r1= world.any_process().receive_vec::<u8>();
				self.dvec.vec = from_vecu8::<F>(&r1.0, F::zero());
			}
			self.dvec.main_processor = new_val;
			self.dvec.synch_size();
		}
	}

	/** check Serial: fn rev(). used for division. 
		Reverse the co-ef list. n may be higher than the 
		target len
	*/
	pub fn rev(&self, new_deg: usize)-> DisPoly::<F>{
		let mut cp = self.clone();
		cp.to_partitions();
		cp.dvec.set_real_len();
		let fdeg = self.dvec.real_len - 1;
		assert!(fdeg<=new_deg, "ERROR: f.degree():{} > deg: {}!", fdeg, new_deg);
		cp.dvec.repartition_worker(new_deg+1, true, true);	
		return cp;
	}

	/**
		generate a vector (start_off, end_off, start, end)
		for the data from src node to dst node.
		both end_off and end are actually not included
		k: the mul_by_xk factor.
		start_off and end_off relative position
	*/
	fn gen_rescale_plan(&self, src: usize, dst: usize, k: usize)->
		(usize, usize, usize, usize){
		if !self.dvec.b_in_cluster {panic!("gen_rescale: partition first!");}
		let np = RUN_CONFIG.n_proc;
		let (mut src_start, mut src_end) = self.
			dvec.get_share_boundary(src as u64, np as u64); 
		src_start += k;
		src_end += k; //because shifting k to the right
		let target_len = k + self.dvec.len; //after shift righting k positions
		let (dest_start, dest_end) = DisVec::<F>::get_share_bounds(dst as u64, np as u64, target_len as u64);
		let start = if src_start > dest_start {src_start} else {dest_start};
		let end = if src_end < dest_end {src_end} else {dest_end};
		if end<=start{
			return (0,0,0,0);
		}
		let s_off = start - src_start;
		let e_off = end - src_start;
		return (s_off,e_off,start,end)
	} 

	/** shrink by k */
	fn gen_shrink_plan(&self, src: usize, dst: usize, k: usize)->
		(usize, usize, usize, usize){
		if !self.dvec.b_in_cluster {panic!("gen_rescale: partition first!");}
		if k>self.dvec.len{
			panic!("gen_shrink_plan ERR: k: {}> self.dvec.len: {}", k, self.dvec.len);
		}
		let np = RUN_CONFIG.n_proc;
		let (mut src_start, mut src_end) = self.
			dvec.get_share_boundary(src as u64, np as u64); 
		let old_src_start = src_start;
		//let old_src_end = src_end;
		src_start  = if src_start>k {src_start -k } else {0};
		src_end = if src_end>k {src_end-k} else {0};
		let target_len = self.dvec.len-k; //after shift righting k positions
		let (dest_start, dest_end) = DisVec::<F>::get_share_bounds(dst as u64, np as u64, target_len as u64);
		let start = if src_start > dest_start {src_start} else {dest_start};
		let end = if src_end < dest_end {src_end} else {dest_end};
		if end<=start{
			return (0,0,0,0);
		}
		let s_off = start + k - old_src_start;
		let e_off = end +k - old_src_start;
		//println!("DEBUG USE 301: s_off: {}, e_off: {}, start: {}, end: {}, old_src_start: {}, old_src_end: {}, dest_start: {}, dest_end: {}, k: {}", s_off, e_off, start, end, old_src_start, old_src_end, dest_start, dest_end, k);
		return (s_off,e_off,start,end)
	} 

	/* return the polynomoial by dividing self with x^k */
	pub fn div_by_xk(&self, k: usize) -> DisPoly<F>{
		if k==0 {
			return self.clone();
		}

		
		if k>self.dvec.len{//return zero
			let sp = DensePolynomial::<F>::zero();
			let dp = DisPoly::from_serial(self.id, &sp, 1);
			return dp;
		}

		if !self.dvec.b_in_cluster{
			let sp = self.to_serial();
			let sp2 = div_by_xk(&sp, k);
			let dp = DisPoly::from_serial(self.id, &sp2, sp2.degree()+1);
			return dp;
		}


		//basically a re-implementation of re_partition
		let target_len = self.dvec.len - k;
		let target_real_len = if self.dvec.real_len_set {self.dvec.real_len - k}
			else {self.dvec.real_len};
		let mut dp = self.clone();
		let np = RUN_CONFIG.n_proc;
		let my_rank = RUN_CONFIG.my_rank;
		let mut vsend: Vec<Vec<F>> = vec![vec![]; np];
		for i in 0..np{
			let (start_offset, end_offset, _dest_start, _dest_end) = dp.
				gen_shrink_plan(my_rank, i, k);
			let row = self.dvec.take_data(start_offset, end_offset);
			vsend[i] = row;
		}

		//2. broadcast and receive
		let vrecv = nonblock_broadcast(&vsend, np as u64, &RUN_CONFIG.univ, F::zero());

		//3. assemble data
		let (my_start, my_end) = DisVec::<F>::get_share_bounds(my_rank as u64, np as u64, target_len as u64); 
		let my_part_len = my_end - my_start;
		dp.dvec.partition = vec![F::zero(); my_part_len];
		for i in 0..np{
			let (_srcs, _srce, s, e) = dp.gen_shrink_plan(i, my_rank, k);
			if e>s{//valid entry
				let my_s = s - my_start;
				let my_e = e - my_start;
				for k in 0..my_e-my_s{
					dp.dvec.partition[my_s+k] = vrecv[i][k]; 
				}
			}
		}
		
		//4. set the new target_len
		dp.dvec.len = target_len;
		dp.dvec.real_len = target_real_len;
		dp.dvec.b_in_cluster = true;
		RUN_CONFIG.better_barrier("dp.div_by_xk");
		return dp;
	}

	/* UPDATE the entire polynomial by multiplying all coefs by v.
		Make the updates to itself */
	pub fn div_by_const(&mut self, inp_v: F) {
		let v = inp_v.inverse().unwrap();
		self.mul_by_const(v);
	}

	pub fn mul_by_const(&mut self, v: F) {
		let mut timer = Timer::new();
		timer.start();

		if !self.dvec.b_in_cluster{
			let sp = self.to_serial();
			let sp2 = get_poly(vec![v]);
			let sp3 = &sp * &sp2;
			let coefs = sp3.coeffs();
			self.dvec.vec = coefs.to_vec();
			self.reset_flags();
			RUN_CONFIG.better_barrier("dp.div_by_const");

			log_perf(LOG2, "PERF_USE_MULCONST", &mut timer);
			return;
		}

		let part_len = self.dvec.partition.len();
		for i in 0..part_len{
			self.dvec.partition[i] = self.dvec.partition[i] * v;
		}

		//2. broadcast and receive
		self.reset_flags();

		log_perf(LOG2, "PERF_USE_MULCONST", &mut timer);
		RUN_CONFIG.better_barrier("dp.div_by_const");
	}


	/* return the polynomoial by multiplying self with k */
	pub fn mul_by_xk(&self, k: usize) -> DisPoly<F>{
		let mut timer = Timer::new();
		timer.start();

		if k==0 {
			return self.clone();
		}

		if !self.dvec.b_in_cluster{
			let sp = self.to_serial();
			let sp2 = mul_by_xk(&sp, k);
			let dp = DisPoly::from_serial(self.id, &sp2, self.dvec.len + k);
//println!("DEBUG USE 699: me: {}, set dp_len to : {}, k: {}", RUN_CONFIG.my_rank, dp.dvec.len, k);

			log_perf(LOG2, "PERF_USE_MULBYXK", &mut timer);
			return dp;
		}

		//basically a re-implementation of re_partition
		let target_len = self.dvec.len + k;
		let target_real_len = if self.dvec.real_len_set {self.dvec.real_len + k}
			else {self.dvec.real_len};
		let mut dp = self.clone();
		let np = RUN_CONFIG.n_proc;
		let my_rank = RUN_CONFIG.my_rank;
		let mut vsend: Vec<Vec<F>> = vec![vec![]; np];
		for i in 0..np{
			let (start_offset, end_offset, _dest_start, _dest_end) = dp.
				gen_rescale_plan(my_rank, i, k);
			let row = self.dvec.take_data(start_offset, end_offset);
			vsend[i] = row;
		}

		//2. broadcast and receive
		let vrecv = nonblock_broadcast(&vsend, np as u64, &RUN_CONFIG.univ, F::zero());

		//3. assemble data
		let (my_start, my_end) = DisVec::<F>::get_share_bounds(my_rank as u64, np as u64, target_len as u64); 
		let my_part_len = my_end - my_start;
		dp.dvec.partition = vec![F::zero(); my_part_len];
		for i in 0..np{
			let (_, _, s, e) = dp.gen_rescale_plan(i, my_rank, k);
			if e>s{//valid entry
				let my_s = s - my_start;
				let my_e = e - my_start;
				for k in 0..my_e-my_s{
					dp.dvec.partition[my_s+k] = vrecv[i][k]; 
				}
			}
		}
		
		//4. set the new target_len
		dp.dvec.len = target_len;
		dp.dvec.real_len = target_real_len;
		dp.dvec.b_in_cluster = true;
		RUN_CONFIG.better_barrier("dp.mul_by_xk");

		log_perf(LOG2, "PERF_USE_MULBYXK", &mut timer);
		return dp;
	}

	/** return the polynomial lower than k */
	pub fn mod_by_xk(&self, k: usize) -> DisPoly::<F>{
		let mut timer = Timer::new();
		timer.start();

		if !self.dvec.b_in_cluster{ //serial version
			let p = self.to_serial();
			let p2 = mod_by_xk(&p, k);
			let new_len = if self.dvec.len>=k {k} else {self.dvec.len};
			let res = Self::from_serial(0, &p2, new_len);

			log_perf(LOG2, "PERF_USE_MODBYXK", &mut timer);
			return res;
		}else{//distributed version	
			let new_n = k;
			let cur_n = if self.dvec.real_len_set {self.dvec.real_len} else {self.dvec.len};
			if k==0{
				let pz = DensePolynomial::<F>::zero();
				let dp = DisPoly::<F>::from_serial(self.id, &pz, 1);

				log_perf(LOG2, "PERF_USE_MODBYXK", &mut timer);
				return dp;
			}else if new_n>cur_n{
				let res = self.clone();
				log_perf(LOG2, "PERF_USE_MODBYXK", &mut timer);
				return res;

			}else{//repartition
				let mut dp = self.clone();
				dp.repartition(new_n);
				log_perf(LOG2, "PERF_USE_MODBYXK", &mut timer);
				return dp;
			}
		}
	}

	/** compute the inverse of f mod x^(2^k).
		NOTE: k is used as the exponent in 2^k.
		Ref: (1) http://people.seas.harvard.edu/~madhusudan/MIT/ST15/scribe/lect06.pdf
	*/
	pub fn inv(&mut self, k: usize) -> DisPoly<F>{
		return self.inv_v2(k);
	}

	/** compute the inverse of f mod x^(2^k).
		NOTE: k is used as the exponent in 2^k.
		Ref: (1) http://people.seas.harvard.edu/~madhusudan/MIT/ST15/scribe/lect06.pdf
		Version 1: pure distributed version
	*/
	pub fn inv_v1(&mut self, k: usize) -> DisPoly<F>{
		let mut timer = Timer::new();
		timer.start();
		let b_perf = false;

		let me = RUN_CONFIG.my_rank;
		let np = RUN_CONFIG.n_proc;
		if !self.dvec.b_in_cluster || self.dvec.len<2*np{//serial mode
			//SOMETIMES, when it's too small, it's causing 
			//computing of c0 crash (no partition when degree is too low
			//at main node. thus requiring dvec.len>=2*np
			let sp = self.to_serial();
			let p2;
			if me==self.dvec.main_processor {
				p2 = inv(&sp, k);
			}else{
				p2 = DensePolynomial::<F>::zero()
			};
			let n = 2_usize.pow(k as u32);
			assert!(p2.degree()+1<=n, "INV ERR: p2.degree+1: {} > n: {}", p2.degree()+1, n);
			let dp = Self::from_serial(self.id, &p2, n);
			RUN_CONFIG.better_barrier("inv");
			return dp;
		}else{//distributed mode
			if me==self.dvec.main_processor{
				if self.dvec.partition.len()==0 || self.dvec.partition[0].is_zero(){
					dump_vec("DUMP 101 partition:", &self.dvec.partition);
					panic!("INV err: coef0 can't be zero. me: {}, dvec.partition.len: {}, dvec.len: {}, in_cluster: {}", me, self.dvec.partition.len(), self.dvec.len, self.dvec.b_in_cluster);
				}
			}
			let c0 = if me==0 {self.dvec.partition[0].inverse().unwrap()} else {F::from(1u64)};
			let mut a = DisPoly::<F>::from_serial(self.id, &DensePolynomial::<F>::from_coefficients_vec(vec![c0]),1);
			let mut t = 1;
			let mut b: DisPoly::<F>;

			if b_perf {log_perf(LOG1, "++++ inv part 1", &mut timer);}

			//2. iterative case
			for _u in 0..k{
				//1. compute b
				let mut m_a = a.mod_by_xk(2*t);
				let mut m_g = self.mod_by_xk(2*t);
				let mut ag_1 = Self::mul(&mut m_a , &mut m_g);
				//let mut ag_1 = Self::sub(&mut ag,&mut one);
				ag_1.dec();
				b = ag_1.div_by_xk(t);
				b = b.mod_by_xk(t);

				//2. compute a	
				let mut neg_ba = Self::mul(&mut b, &mut a);
				neg_ba.neg();
				let a1 = neg_ba.mod_by_xk(t);
				let mut xt_a1 = a1.mul_by_xk(t);
				a = Self::add(&mut a, &mut xt_a1);
				a = a.mod_by_xk(2*t);
				t = 2*t;
				if b_perf {log_perf(LOG1, &format!("inv round: {}", _u), &mut timer);}
			}//end for
			RUN_CONFIG.better_barrier("inv");
			return a;
		}
	}
 
	/** compute the inverse of f mod x^(2^k).
		NOTE: k is used as the exponent in 2^k.
		Ref: (1) http://people.seas.harvard.edu/~madhusudan/MIT/ST15/scribe/lect06.pdf
		Version 1: pure distributed version
	*/
	pub fn inv_v2(&mut self, k: usize) -> DisPoly<F>{
		let b_perf = false;
		let b_mem = false;
		let mut timer = Timer::new();
		let mut timer2 = Timer::new();
		timer.start();
		timer2.start();

		let me = RUN_CONFIG.my_rank;
		let np = RUN_CONFIG.n_proc;

		if !self.dvec.b_in_cluster || self.dvec.len<2*np{//serial mode
			//SOMETIMES, when it's too small, it's causing 
			//computing of c0 crash (no partition when degree is too low
			//at main node. thus requiring dvec.len>=2*np
			let sp = self.to_serial();
			let p2;
			if me==self.dvec.main_processor {
				p2 = inv(&sp, k);
			}else{
				p2 = DensePolynomial::<F>::zero()
			};
			let n = 2_usize.pow(k as u32);
			assert!(p2.degree()+1<=n, "INV ERR: p2.degree+1: {} > n: {}", p2.degree()+1, n);
			let dp = Self::from_serial(self.id, &p2, n);
			RUN_CONFIG.better_barrier("inv");
			return dp;
		}else{//distributed mode
			if me==self.dvec.main_processor{
				if self.dvec.partition.len()==0 || self.dvec.partition[0].is_zero(){
					dump_vec("DUMP 101 partition:", &self.dvec.partition);
					panic!("INV err: coef0 can't be zero. me: {}, dvec.partition.len: {}, dvec.len: {}, in_cluster: {}", me, self.dvec.partition.len(), self.dvec.len, self.dvec.b_in_cluster);
				}
			}
			if b_perf {log_perf(LOG1, "++++ inv part 1", &mut timer);}

			//1. compute the SMALLer component first
			let log_bar = if k>RUN_CONFIG.log_inv_bar 
				{RUN_CONFIG.log_inv_bar} else {k};


			let dsp = self.mod_by_xk(1<<log_bar);
			let sp = dsp.to_serial();
			//let sp = self.to_serial(); //only main node gets the result


			let sa = if me==self.dvec.main_processor {inv(&sp, log_bar)}
				else {get_poly(vec![F::from(1u64)])};

			let mut a = Self::from_serial(self.id, &sa, sa.degree()+1); 
			a.dvec.main_processor = self.dvec.main_processor;
			a.dvec.synch_size(); //needed as other nodes need to know size
			a.to_partitions();
			if b_perf {log_perf(LOG1, "++++ inv serial part", &mut timer);}


			//2. prep for the second part of the loop
			let mut t = 1<<log_bar;
			let mut b: DisPoly::<F>;

			//2. iterative case
			for _u in (log_bar)..k{
				//1. compute b
				let mut m_a = a.mod_by_xk(2*t);
				let mut m_g = self.mod_by_xk(2*t);
				let mut ag_1 = Self::mul(&mut m_a , &mut m_g);
				//let mut ag_1 = Self::sub(&mut ag,&mut one);
				ag_1.dec();
				b = ag_1.div_by_xk(t);
				b = b.mod_by_xk(t);

				//2. compute a	
				let mut neg_ba = Self::mul(&mut b, &mut a);
				neg_ba.neg();
				let a1 = neg_ba.mod_by_xk(t);
				let mut xt_a1 = a1.mul_by_xk(t);
				a = Self::add(&mut a, &mut xt_a1);
				a = a.mod_by_xk(2*t);
				t = 2*t;
				if b_perf {log_perf(LOG1, &format!("-- inv round: {}", _u), &mut timer);}
				if b_mem {dump_mem_usage(&format!("INV: iteration: {}", _u));}
			}//end for
			if b_perf {log_perf(LOG1, "inv TOTAL: ", &mut timer2);}
			if b_mem {dump_mem_usage(&format!("INV: degree {}", 1<<k));}

			//3. pack and return
			RUN_CONFIG.better_barrier("inv");
			return a;
		}
	}
 


	/*** align their length */
	fn align_len(p1: &mut DisPoly<F>, p2: &mut DisPoly<F>){
		let d1 = p1.dvec.len;
		let d2 = p2.dvec.len;
		if d1>d2{
			p2.repartition(d1);
		}
		if d2>d1{
			p1.repartition(d2);
		}
		RUN_CONFIG.better_barrier("repartition");
	}

	/*** align their length */
	fn align_len_mul(p1: &mut DisPoly<F>, p2: &mut DisPoly<F>){
		//let me = RUN_CONFIG.my_rank;
		p1.dvec.synch_size();
		p2.dvec.synch_size();
		p1.dvec.set_real_len();
		p2.dvec.set_real_len();
		let d = closest_pow2(p1.dvec.real_len + p2.dvec.real_len);
		if d!=p2.dvec.len{
			p2.repartition(d);
		}
		if d!=p1.dvec.len{
			p1.repartition(d);
		}
	}

	/** decrease itself by one */
	pub fn dec(&mut self){
		let one = F::one();
		let zero= F::zero();
		if !self.dvec.b_in_cluster{
			if self.dvec.vec.len()>0{
				self.dvec.vec[0] = self.dvec.vec[0] - one;
			}else{
				self.dvec.vec = vec![zero - one];
			}
		}else{
			if RUN_CONFIG.my_rank==self.dvec.main_processor{
				if self.dvec.partition.len()>0{
					self.dvec.partition[0] = self.dvec.partition[0] - one;
				}else{
					self.dvec.partition = vec![zero - one];
				}
			}
		}
		self.reset_flags();
		RUN_CONFIG.better_barrier("DisPoly::dec");
	}

	/** negate itself */
	pub fn neg(&mut self){
		let zero= F::zero();
		if !self.dvec.b_in_cluster{
			let len = self.dvec.vec.len();
			for i in 0..len{
				self.dvec.vec[i] = zero - self.dvec.vec[i];
			}
		}else{
			let len = self.dvec.partition.len();
			for i in 0..len{
				self.dvec.partition[i] = zero - self.dvec.partition[i];
			}
		}
		self.reset_flags();
		RUN_CONFIG.better_barrier("DisPoly::neg");
	}

	/*** polynomial sub. NEED TO CALL AT ALL NODES */
	pub fn sub_worker(p1: &mut DisPoly<F>, p2: &mut DisPoly<F>, bsub:bool) -> DisPoly<F>{
		//0. serial case
		if !p1.dvec.b_in_cluster &&  !p2.dvec.b_in_cluster{
			let sp1 = p1.to_serial();
			let sp2 = p2.to_serial();
			let sp3 = if bsub {&sp1-&sp2} else {&sp1 + &sp2};
			let len1 = p1.dvec.len;
			let len2 = p2.dvec.len;
			//println!("DEBUG USE 888: len1: {}, len2: {}, in1: {}, in2: {}", len1, len2, p1.dvec.b_in_cluster, p2.dvec.b_in_cluster);
			let len = if len1>len2 {len1} else {len2};
			let mut dp3 = DisPoly::<F>::
				from_serial(p1.dvec.main_processor as u64, &sp3, len);
			dp3.dvec.main_processor = p1.dvec.main_processor;
			return dp3;
		}

		//1. align main processor
		p1.to_partitions();
		p2.to_partitions();
		Self::check_same_main_processor(p1, p2);
	
		//2. align for len/degree
		Self::align_len(p1, p2);

		//3. at each node: do the subtraction
		let part_len = p1.dvec.partition.len();
		let mut my_part = vec![F::zero(); part_len];
		for i in 0..part_len{
			if bsub{
				my_part[i] = p1.dvec.partition[i] - p2.dvec.partition[i];
			}else{
				my_part[i] = p1.dvec.partition[i] + p2.dvec.partition[i];
			}
		}
		//use the same id with p1 to ensure the same partition scheme
		let mut dv= DisVec::<F>::new_from_each_node(p1.id, p1.dvec.main_processor as u64, p1.dvec.len, my_part);
		dv.set_real_len();
		let dp_res = DisPoly{id: p1.id, dvec: dv, is_one: false, is_zero: false};
		return dp_res;
	}

	/*** return p1 - p2 */
	pub fn sub(p1: &mut DisPoly<F>, p2: &mut DisPoly<F>) -> DisPoly<F>{
		return Self::sub_worker(p1, p2, true);
	}

	/*** return p1 + p2 */
	pub fn add(p1: &mut DisPoly<F>, p2: &mut DisPoly<F>) -> DisPoly<F>{
		return Self::sub_worker(p1, p2, false);
	}

	/// get the derivative
	pub fn logical_get_derivative(&self) -> DisPoly<F>{
		let mut d2 = self.clone();
		let c1 = d2.coefs();
		let p = DensePolynomial::<F>::from_coefficients_vec(c1);
		let dp = get_derivative(&p);
		let dres = DisPoly::<F>::from_serial(0, &dp, self.dvec.len);
		return dres;
	}

	/// get the derivative
	pub fn get_derivative(&self) -> DisPoly<F>{
		if !self.dvec.b_in_cluster{ return self.logical_get_derivative(); }
		//1. everyone sends the LOWEST coef to prev node and expects one
		//from next proc
		let me = RUN_CONFIG.my_rank;
		let np = RUN_CONFIG.n_proc;
		let vdata = to_vecu8(&vec![self.dvec.partition[0]]);
		let world = RUN_CONFIG.univ.world();
		let prev_id = (me + np -1) %np;
		let next_id = (me + 1)%np;
		world.process_at_rank(prev_id as i32).send_with_tag(&vdata, me as i32);
		let vrec = world.process_at_rank(next_id as i32).receive_vec::<u8>().0;
		let sample = F::zero();
		let higher_coef = from_vecu8(&vrec, sample)[0];

		//2. everyone computes the new coef
		let (me_base, _end) = self.dvec.get_share_bounds_usize(me);
		let (nxt_base, _) = self.dvec.get_share_bounds_usize(next_id);
		let mut new_partition = self.dvec.partition.clone();
		let new_len = new_partition.len();
		for i in 0..new_len-1{
			new_partition[i] = F::from((i+me_base+1) as u64)*new_partition[i+1];
		}
		new_partition[new_len-1] = higher_coef * F::from(nxt_base as u64);

		//3. builds a dvec from each node and set up DisPoly
		//4. set up the DisPoly
		let new_dv = DisVec::<F>::new_from_each_node(self.dvec.id as u64, self.dvec.main_processor as u64, self.dvec.len, new_partition);
		let dres = DisPoly::<F>{
			id: self.id,
			dvec: new_dv,
			is_zero: false,
			is_one: false
		};
		return dres;
	}

	/// compute the chunked derivative to match the function 
	/// get_derivative_shifted() in ZaModularTrace Verifier
	/// The difference is that it ACCUMULATES the results
	/// of get_derivative_shifted()
	/// NEEDS TO BE CALLED AT ALL NODES, BUT RETURN CORRECT SOLUTION
	/// AT MAIN NODES
	pub fn logical_compute_chunked_derivative(&self, r: &F) -> Vec<F>{
		let sp1 = self.to_serial();
		let me = RUN_CONFIG.my_rank;
		let np = RUN_CONFIG.n_proc;
		let total_len = self.dvec.len; 	
		let mut res = vec![F::zero(); np+1];
		if me!=self.dvec.get_main_processor() as usize {return res;}

		//1. process each chunk 
		let mut coefs = sp1.coeffs().to_vec();
		coefs.resize(total_len, F::zero());
		let unit = total_len/np;
		for i in 0..np{
			let base = if i==0 {r.inverse().unwrap()} else 
				{r.pow(&[(i*unit-1) as u64])};
			let n = if i<np-1 {total_len/np} else {total_len/np + total_len%np};
			let slice = coefs[i*unit..i*unit+n].to_vec();
			let mut new_coefs = vec![F::zero(); slice.len()];
			for j in 0..slice.len(){
				new_coefs[j] = F::from((i*unit+j) as u64) * slice[j]; 
			}
			let pd = get_poly(new_coefs);
			let v = pd.evaluate(&r);
			res[i+1] = res[i] + base * v;
		}

		return res;
	}

	/// compute the chunked derivative to match the function 
	/// get_derivative_shifted() in ZaModularTrace Verifier
	/// The difference is that it ACCUMULATES the results
	/// of get_derivative_shifted()
	/// NEEDS TO BE CALLED AT ALL NODES, BUT RETURN CORRECT SOLUTION
	/// AT MAIN NODES
	pub fn compute_chunked_derivative(&self, r: &F) -> Vec<F>{
		if !self.dvec.b_in_cluster{ 
			return self.logical_compute_chunked_derivative(r); 
		}
		//0. main node needs to broadcast r to be consistent at all nodes
		let me = RUN_CONFIG.my_rank;
		let np = RUN_CONFIG.n_proc;
		let vr = broadcast_small_arr(&vec![r.clone()], 
			self.dvec.main_processor);
		let r = vr[0];

		//1. everyone computes the new coef
		let mut new_partition = self.dvec.partition.clone();
		let total_len = self.dvec.len; 	
		let unit = total_len/np;
		let mut res = vec![F::zero(); np+1];
		let base = if me==0 {r.inverse().unwrap()} else 
			{r.pow(&[(me*unit-1) as u64])};
		for i in 0..new_partition.len(){
			new_partition[i] = F::from((me*unit+i) as u64) * new_partition[i];
		}
		let sp = get_poly(new_partition);
		let v = sp.evaluate(&r);
		let me_res = base * v;
		let main_id = self.dvec.main_processor;

		//2. all report to the main node
		let arr_res = gather_small_arr(&vec![me_res], self.dvec.main_processor);
		if me==main_id{
			for i in 0..arr_res.len(){
				res[i+1] = res[i] + arr_res[i][0];
			}
		}

		return res;
	}


	/// node 0 broadcast its lowest item to ALL
	pub fn get_coef0_at_each_node(&self)->F{
		let send_size = F::zero().serialized_size();
		let world = RUN_CONFIG.univ.world();
		let me = RUN_CONFIG.my_rank;
		let np = RUN_CONFIG.n_proc;
		let mut vec_to_send = vec![0u8; send_size];
		let sender = if self.dvec.b_in_cluster {np-1} else {self.dvec.main_processor};
		let root_process = world.process_at_rank(sender as i32);
		if me==sender{
			let val = if self.dvec.b_in_cluster {self.dvec.partition[0]} else {self.dvec.vec[0]};
			vec_to_send = to_vecu8::<F>(&vec![val]);
		}
		root_process.broadcast_into(&mut vec_to_send);
		assert!(vec_to_send.len()==send_size, "me: {}, vec_to_send.size(): {} != send_size: {}", me, vec_to_send.len(), send_size);

		let vec_ret = from_vecu8::<F>(&vec_to_send, F::zero()); 
		let v = vec_ret[0];
		return v;
	}

	/// polynomial mul 
	pub fn mul(p1: &mut DisPoly<F>, p2: &mut DisPoly<F>) -> DisPoly<F>{
		//1. sequential if it's TOO SMALL.
		let np = RUN_CONFIG.n_proc;
		let min_size = np * np;
		let p1_len = p1.dvec.len;
		let p2_len = p2.dvec.len;
		let mut timer = Timer::new();
		timer.start();
	
		//1.5 special handling (zero, 1, or len1 case)
		if p1_len==1{
			if p1.is_zero{ return Self::zero(0);}
			else if p1.is_one {return p2.clone();} 
/*
			else{
				let v = p1.get_coef0_at_each_node();
				let mut dpret = p2.clone();
				dpret.mul_by_const(v);	
				return dpret;
			}
*/
		}
		if p2_len==1{
			if p2.is_zero{ return Self::zero(0);}
			else if p2.is_one {return p1.clone();} 
			else{
				let v = p2.get_coef0_at_each_node();
				let mut dpret = p1.clone();
				dpret.mul_by_const(v);	
				return dpret;
			}
		}

		let target_size = closest_pow2(p1.dvec.len+p2.dvec.len);

		//let me = RUN_CONFIG.my_rank;
		//1. align main processor
		p1.to_partitions();
		p2.to_partitions();
		Self::check_same_main_processor(p1, p2);


		//2. align for len/degree
		Self::align_len_mul(p1, p2);

		let size = if p1.dvec.len< p2.dvec.len {p1.dvec.len} else {p2.dvec.len};
		if size<min_size{
			let sp1 = p1.to_serial();
			let sp2 = p2.to_serial();
			let res = &sp1 * &sp2;
			let mut dp = DisPoly::from_serial(p1.id, &res, target_size);
			dp.dvec.main_processor = p1.dvec.main_processor;
			dp.dvec.synch_size();
			log_perf(LOG2, "PERF_USE_SERIALMUL", &mut timer);
			return dp;
		}

		//3. mul by fft 
		//println!("DEBUG USE 500: me: {}, p1.len: {}, p2.len: {}", RUN_CONFIG.my_rank, p1.dvec.len, p2.dvec.len);
		let univ = &RUN_CONFIG.univ;
		let vec1 = p1.dvec.partition.clone();
		let vec2 = p2.dvec.partition.clone();
		let len = p1.dvec.len;
		let part_len = p1.dvec.partition.len();
		let mut dvec1 = DisVec::<F>::new_from_each_node(p1.dvec.id, p1.dvec.main_processor as u64, len, vec1);
		let mut dvec2 = DisVec::<F>::new_from_each_node(p1.dvec.id, p1.dvec.main_processor as u64, len, vec2);

		distributed_dizk_fft(&mut dvec1, univ);
		distributed_dizk_fft(&mut dvec2, univ);

		for i in 0..part_len{
			dvec1.partition[i] = dvec1.partition[i] * dvec2.partition[i];
		}

		distributed_dizk_ifft(&mut dvec1, univ);
		let res = DisPoly{id: p1.id, dvec: dvec1, is_one: false, is_zero: false};
		//no need to resynch size as it's already synched for fft
		return res;

	}

	/** return zero polynomial */
	pub fn zero(id: u64) -> DisPoly::<F>{
		let vec = vec![F::zero()];
		let p = get_poly(vec);
		let mut dp = Self::from_serial(id, &p, 1);
		dp.is_zero = true;
		return dp;
	}

	/** return one polynomial */
	pub fn one(id: u64) -> DisPoly::<F>{
		let vec = vec![F::from(1u64)];
		let p = get_poly(vec);
		let mut dp = Self::from_serial(id, &p, 1);
		dp.is_one= true;
		return dp;
	}

	/** faster divide_with_q_and_r using the Hensel lift
	(1) http://people.seas.harvard.edu/~madhusudan/MIT/ST15/scribe/lect06.pdf
	(2) http://people.csail.mit.edu/madhu/ST12/scribe/lect06.pdf
	*/
	pub fn divide_with_q_and_r(p1: &mut DisPoly<F>, p2: &mut DisPoly<F>) -> 
		(DisPoly<F>, DisPoly<F>){
		let b_perf = false;
		let b_perf2 = false;
		let b_mem = false;

		let mut timer = Timer::new();
		let mut t2 = Timer::new();
		let p1_main = p1.dvec.main_processor;
		timer.start();
		t2.start();
		p1.dvec.set_real_len();
		p2.dvec.set_real_len();

		if p1.dvec.real_len<p2.dvec.real_len {
			//log_perf(LOG1, &format!("PERF_USE_DIV case 1: p1: {}, p2: {}", p1.dvec.real_len, p2.dvec.real_len), &mut timer);
			return (Self::zero(p1_main as u64), p2.clone());
		}

		let np = RUN_CONFIG.n_proc;
		let min_size = np * np;
		let size = if p1.dvec.real_len > p2.dvec.real_len {p1.dvec.real_len} else {p2.dvec.real_len};
		let bar = 1<<RUN_CONFIG.log_div_bar;
		if size < min_size || size<bar{
		//if !p1.dvec.b_in_cluster || !p2.dvec.b_in_cluster || size < min_size{
			let sp1 = p1.to_serial();
			let mut sp2 = p2.to_serial();
			if RUN_CONFIG.my_rank!=p1.dvec.main_processor{ sp2 = get_poly(vec![F::from(1u64)]);}
			let (q,r) = adapt_divide_with_q_and_r(&sp1, &sp2);
			let mut dq= Self::from_serial(p1_main as u64, &q, q.degree()+1);
			let mut dr= Self::from_serial(p1_main as u64, &r, r.degree()+1);
			dq.dvec.synch_size();
			dr.dvec.synch_size();

			//log_perf(LOG1, &format!("PERF_USE_DIV: case 2. p1.real: {}, p2. real: {}", p1.dvec.real_len, p2.dvec.real_len), &mut timer);
			return (dq,dr);
		}
		if b_perf {log_perf(LOG1, "\n\n****part1 of div", &mut t2);}


		//DISTRIBUTED CASE
		p1.to_partitions();
		p2.to_partitions();
		let f = p1; //just to be compatible with serial version
		let g = p2;
		f.dvec.real_len_set = false; //enforace broadcast real_len
		g.dvec.real_len_set = false;
		f.dvec.set_real_len();
		g.dvec.set_real_len();
		let flen = f.dvec.real_len;
		let glen = g.dvec.real_len;
		assert!(flen>=1, "flen:{}<1", flen);
		assert!(glen>=1, "glen:{}<1", glen);

		let mut rev_f = f.rev(flen-1);
		let mut rev_g = g.rev(glen-1);
		let diff_deg = flen - glen;
		//RECOVER LATER IF THERE ARE BUGS!
		//let log_diff_deg = ceil_log2(diff_deg);
		//let mut inv_rev_g = rev_g.inv(log_diff_deg+1);

		let log_diff_deg = ceil_log2(diff_deg+1);
		let mut inv_rev_g = rev_g.inv(log_diff_deg);
		if b_perf {log_perf(LOG1, "inv of g", &mut t2);}


		let mut timer3 = Timer::new();
		timer3.start();
		if b_perf2 {
			log(LOG1, &format!("inv_rev_g: len: {}, real_len: {}, rev_f len: {}, real_len: {}, log_diff_deg+1: {}", inv_rev_g.dvec.len, inv_rev_g.dvec.real_len, rev_f.dvec.len, rev_f.dvec.real_len, log_diff_deg+1));
		}
		let prod_inv_rev_g_f = Self::mul(&mut inv_rev_g, &mut rev_f);
		if b_perf2 {
			log_perf(LOG1, "part3-step1", &mut timer3);
		}
		let mut rev_q = prod_inv_rev_g_f.mod_by_xk(diff_deg+1);
		rev_q.dvec.set_real_len();
		let rev_q_degree = rev_q.dvec.real_len - 1;
		let mut q = rev_q.rev(rev_q_degree);
		q.dvec.set_real_len();
		let q_degree = q.dvec.real_len - 1;
		let degree_diff = diff_deg - q_degree;
		if b_perf2 {log_perf(LOG1, "part3-step2", &mut timer3);}

		let mut q = Self::mul_by_xk(&q, degree_diff);	
		q.dvec.real_len_set = false;
		q.dvec.set_real_len();
		let real_len = q.dvec.real_len;
		q.dvec.repartition(real_len);
		q.id = p1_main as u64;
		q.reset_main_processor(p1_main);

		if b_perf2 {
			log(LOG1, &format!("q.len: {}, real_len: {}, g.len: {}, real_len: {}", q.dvec.len, q.dvec.real_len, g.dvec.len, g.dvec.real_len));
		}
		if b_perf2 {log_perf(LOG1, "part3-step3", &mut timer3);}

		let mut qg = Self::mul(&mut q, g);
		if b_perf2 {log_perf(LOG1, "part3-step4", &mut timer3);}
		let r = Self::sub(f, &mut qg);
		q.dvec.real_len_set = false;
		q.dvec.set_real_len();
		let real_len = q.dvec.real_len;
		q.dvec.repartition(real_len);
		if b_perf2 {
			log_perf(LOG1, "part3-step5", &mut timer3);
			log(LOG1, &format!("q.len: {}, real_len: {}, r.len: {}, real_len: {}", q.dvec.len, q.dvec.real_len, r.dvec.len, r.dvec.real_len));
		}


		if b_perf {log_perf(LOG1, "rest of div", &mut t2);}
		if b_mem {dump_mem_usage(&format!("DIV: degree: {}, {}", flen, glen));}
		return (q,r);
	}

	/// compute the FAST gcd (using hgcd) and return the bizout's coef
	pub fn feea_unused_old(p1: &DisPoly<F>, p2: &DisPoly<F>) -> 
		(DisPoly<F>, DisPoly<F>, DisPoly<F>){
		let n = if p1.dvec.len>p2.dvec.len {p1.dvec.len} else {p2.dvec.len};
		let sp1 = p1.to_serial();
		let sp2 = p2.to_serial();
		let pone = get_poly(vec![F::from(1u64)]);
		let (gcd, s, t) = if RUN_CONFIG.my_rank==p1.dvec.main_processor {
			feea(&sp1, &sp2)
		} else {
			(pone.clone(), pone.clone(), pone.clone())
		};
		let d_gcd = DisPoly::<F>::from_serial(0, &gcd, n);
		let d_s = DisPoly::<F>::from_serial(0, &s, n);
		let d_t = DisPoly::<F>::from_serial(0, &t, n); //degree is over aprox
		RUN_CONFIG.better_barrier("feea");
		if 201-191>5 {panic!("THIS FUNCTION IS OUTDATED. DO NOT CALL IT!");}
		return (d_gcd, d_s, d_t);
	}

	/// polynomial evaluation. NOTE only returns valid result at the
	/// the MAIN processor!
	pub fn logical_eval(&self, r: &F) -> F{
		// dummy implementaion below
		let sp1 = self.to_serial();
		let res = sp1.evaluate(&r);
		return res;
	}

	pub fn eval(&self, r: &F) -> F{
		//0. serial case
		if !self.dvec.b_in_cluster{
			let sp = DensePolynomial::<F>::from_coefficients_vec(self.dvec.vec.clone());
			return sp.evaluate(r);
		}
		//1. broadcast r
		let me = RUN_CONFIG.my_rank;
		let vr = broadcast_small_arr(&vec![r.clone()], 
			self.dvec.main_processor);
		let r = vr[0];

		//2. everybody computes eval and multiply with position
		let (me_base, _end) = self.dvec.get_share_bounds_usize(me);
		let factor_base = r.pow(&[me_base as u64]);
		let sp = DensePolynomial::<F>::from_coefficients_vec(self.dvec.partition.clone());
		let res = sp.evaluate(&r) * factor_base;

		//3. all_to_1 data and merge result
		let arr_res = gather_small_arr(&vec![res], self.dvec.main_processor);
		let mut res = F::from(0u64);
		if me==self.dvec.main_processor{
			for vec in arr_res{
				res += vec[0];
			}
		}
		return res;
	}

	/// local polynomial evaluation. 
	/// ONLY the main processor returns the correct result
	/// OTHER Other nodes returns an empty vector. SHOULD be called
	/// by all nodes. Return a vector of size: num_processors.
	/// res[0] is the polynomial OUTPUT for chunk 0
	/// res[1] is the polynomial OUTPUT for chunk 1 
	/// res[np-1] is the output of the entire polynomial. 
	/// Assumption: coefs are distributed evenly (with slighly larger
	/// last chunk)
	pub fn eval_chunks(&self, r: &F) -> Vec<F>{
		let np = RUN_CONFIG.n_proc;
		let mut local_res = vec![F::zero();np];
		if !self.dvec.b_in_cluster{panic!("call to_partitions() first!");}

		//1. compute the local partition as poly
		let sp = DensePolynomial::<F>::from_coefficients_vec(self.dvec.partition.clone());
		let v = sp.evaluate(&r);

		//2. send over and return
		let me = RUN_CONFIG.my_rank;
		local_res[me] = v;
		let main_rank = self.dvec.get_main_processor() as usize;
		let world = RUN_CONFIG.univ.world();
		if me!=main_rank{
			let vbytes = to_vecu8(&vec![v]);
			world.process_at_rank(main_rank as i32).send_with_tag(&vbytes, me as i32);
		}else{//collect
			for _i in 0..np-1{
				let r1 = world.any_process().receive_vec::<u8>();
				let vec = from_vecu8::<F>(&r1.0, F::zero());
				let src = r1.1.tag() as usize;
				local_res[src] = vec[0];		
			}
		}

		//3. process the local result and accumulate the output
		let mut res = vec![F::zero(); np];
		let mut base = F::one();
		let unit_size = self.dvec.len/np;
		let factor = r.pow(&[unit_size as u64]);
		let mut cur_val = F::zero();
		for i in 0..np{//process each chunk
			cur_val += local_res[i] * base;
			res[i] = cur_val;
			base *= factor;
		}

		//2. return the local_result
		RUN_CONFIG.better_barrier("eval_chunks");
		return res; 
	}

	/// return if the polynomial is zero
	/// return correct value at EACH node!
	pub fn is_zero(&self) -> bool{
		let rlen = self.dvec.get_real_len();
		return rlen==0;
	}	

	/** Print info, dump up to limit_num elements */
	pub fn dump(&self, limit_num: usize){
		let np = RUN_CONFIG.n_proc;
		let my_rank = RUN_CONFIG.my_rank;
		println!("--------  DisPoly {}: Processor: {}. Total: {} processors-----------", 
			self.id, my_rank, np);
		self.dvec.dump(limit_num);
	}

	/** NEED TO BE CALLED AT ALL NODES.
		It collects all coefficients. Make sure caller has sufficient memory.
		This function is inefficient (returning a copy of all coefs).
		Mainly used for testing.
		returns the VALID solution ONLY
			at the main processor (but needs cooperation from partitions);
		SLOW! should be used for testing only!
	*/
	pub fn coefs(&mut self) -> Vec<F>{
		return self.dvec.collect_from_partitions(&RUN_CONFIG.univ);
/*
		if self.dvec.b_in_cluster{
			let my_rank = RUN_CONFIG.my_rank;	
			let main_proc = self.dvec.main_processor;
			//self.vec is set for main_processor
			self.dvec.from_partitions(&RUN_CONFIG.univ);  
			if my_rank==main_proc{
				return self.dvec.vec.clone();
			}else{
				return vec![];
			}
		}else{
			let my_rank = RUN_CONFIG.my_rank;
			if self.dvec.main_processor!=my_rank{
				return vec![];
			}else{
				return self.dvec.vec.clone();
			}
		}
*/
	}

	/** Construct a dpoly object, read data from src_file based
	on the its own processor_id (each reads its own fragement).
		Workflow: each processor reachs a segment of roots and build
	a polynomial and then multiply these local polynomials and
	build up a DisPoly object. When local polynomials are small,
	they are multiplied together. When they are big, they'll be
	treated as distributed.
		ASSUMPTION: the data file is available on ALL NODES!
		OUT DATED: use the display_from_roots_in_file_from_mainnode instead.
	pub fn dispoly_from_roots_in_file(id: u64, main_processor: u64, src_file: &String) -> DisPoly<F>{
		let log_2_group_size = 1;
		let distbar = 1<<20;
		return DisPoly::<F>::dispoly_from_roots_in_file_worker(id, main_processor, src_file, log_2_group_size, distbar);
	}
	*/

	/** Construct a dpoly object, read data from src_file based
	on the its own processor_id (each reads its own fragement).
		Workflow: the MAIN NODE reads the entire data and passes
	on the share to data node. Then each node builds an actual DisPoly
	but also a collection of fake DisPolys. Then they are merged into ONE.
	When local polynomials are small,
	they are multiplied together. When they are big, they'll be
	treated as distributed.
		ASSUMPTION: only need the data file to be present at main node.
		This subsumes the "dispoly_from_roots_in_file"
		node_file: node architecture file like /tmp/nodelist.txt
	*/
	pub fn dispoly_from_roots_in_file_from_mainnode(id: u64, main_processor: u64, src_file: &String, netarch: &(Vec<u64>,Vec<usize>,Vec<Vec<usize>>)) -> DisPoly<F>{
		let log_2_group_size = 1;
		let distbar = 1<<20;
		return DisPoly::<F>::dispoly_from_roots_in_file_from_mainnode_worker(id, main_processor, src_file, log_2_group_size, distbar, netarch);
	}

	/** the worker function. The extra log_2_group_size and dist_bar
		are used for generating mul plan 
		NOTE: DEPRECATED!
	*/
	pub fn dispoly_from_roots_in_file_worker(id: u64, main_processor: u64, src_file: &String, log_2_group_size: usize, dist_bar: usize) -> DisPoly<F>{
		//1. decide the range to read
		let np = RUN_CONFIG.n_proc;
		let my_rank = RUN_CONFIG.my_rank;
		let total = read_1st_line_as_u64(src_file);
		let b_perf = false;
		let mut timer = Timer::new();
		timer.start();
		assert!(total>=(np as u64), "Number of roots: {} shoud > np: {}!", total, np);

		//2. build up n local distributed DisPoly, with one real, others fake
		//re-arrange the order so that the elements of DisPoly in
		//arr_poly's main_processor is its idx%main_processor
		//i.e., arr_poly[0]'s main processor is main_processor
		let mut arr_poly:Vec<DisPoly<F>> = vec![DisPoly::
			fake_single(0, 0, 0); np];
		for i in 0..np{
			let newid = id*(np as u64) + (i as u64); 
			let nmp = (main_processor + (i as u64))%(np as u64); 
			if nmp==my_rank as u64{
				let (start, end) = DisVec::<F>::get_share_bounds(nmp as u64, np as u64, total as u64);
				//idx start+1: coz first line is total
				let v: Vec<F> = read_arr_from(src_file, start+1, end); 
				assert!(v.len()>=1, "the size of input: {} is too small for share at {}",v.len(), my_rank);
				let dp = DisPoly::single_from_vec(newid, nmp, &v);
				arr_poly[i]  = dp;
			}else{
				let (start, end) = DisVec::<F>::get_share_bounds(nmp as u64, np as u64, total as u64);
				let part_len = end-start+1; //because len is degree + 1
				let dp = DisPoly::fake_single(newid, nmp, part_len);
				arr_poly[i] = dp;
			}
		}
		if b_perf {log_perf(LOG1, &format!("disploly::fromeachnode step 1:"), &mut timer);}

		//println!("*** DEBUG USE 1001: my_rank: {} ***, len: {}, {}, {}, {}", my_rank, arr_poly[0].dvec.len, arr_poly[1].dvec.len, arr_poly[2].dvec.len, arr_poly[3].dvec.len);
		//3. multiply all
		let (vplan, dist_round_id) = DisPoly::<F>::gen_chain_mul_plan_worker(
			&arr_poly, log_2_group_size, dist_bar);	
		//println!("DEBUG USE 1002: plan: {:?}, dist_bar: {}", &vplan, dist_round_id);
		DisPoly::<F>::chain_mul_worker(&mut arr_poly,vplan, dist_round_id); 
		//println!("DEBUG USE 1002 DONE: dist_bar: {}",  dist_round_id);
		RUN_CONFIG.better_barrier("dis_poly_from_roots");
		if b_perf {log_perf(LOG1, &format!("disploly::fromeachnode step 2:"), &mut timer);}
		let mut res = arr_poly[0].clone();
		res.dvec.set_real_len();
		return res;
	}

	/** check poly mem consumption */
	pub fn test_mem_leak(size: usize){
		dump_mem_usage("BEFORE p1");
		let _p1 =  rand_poly::<F>(size, 101);
		dump_mem_usage("BEFORE p2");
		let _p2 =  rand_poly::<F>(size, 101);
	}

	/** the worker function. The extra log_2_group_size and dist_bar
		are used for generating mul plan
		NOTE: the main node reads the ENTIRE FILE and then
		pass the information to all other nodes. ASSUMPTION:
		the main node has the access of the file.  
		node_file: the location of nodes list such as /tmp/nodeslist.txt
	*/
	pub fn dispoly_from_roots_in_file_from_mainnode_worker(id: u64, main_processor: u64, src_file: &String, log_2_group_size: usize, dist_bar: usize, netarch: &(Vec<u64>,Vec<usize>,Vec<Vec<usize>>)) -> DisPoly<F>{

		//1. create the "fake" DisPoly array
		let b_perf = false;
		let b_mem = false;
		if b_mem {dump_mem_usage("-- BEFORE binacc starts --");}
		let mut timer = Timer::new();
		timer.start();
		let np = RUN_CONFIG.n_proc;
		let me = RUN_CONFIG.my_rank;
		//let world = RUN_CONFIG.univ.world();
		let mut total = if me==main_processor as usize
			{read_1st_line_as_u64(src_file) as usize} else {0usize};
		let vt = broadcast_small_arr(&vec![total], main_processor as usize);
		total = vt[0];

		let mut arr_poly = vec![DisPoly::fake_single(0, 0, 0);np];
		for i in 0..np{
			let newid = id*(np as u64) + (i as u64); 
			let nmp = (main_processor + (i as u64))%(np as u64); 
			let (start, end) = DisVec::<F>::get_share_bounds(nmp as u64, np as u64, total as u64);
			let part_len = end - start + 1;
			arr_poly[i] = DisPoly::fake_single(newid, nmp, part_len);
			//for each node: ONE will be reset later
		}
		if b_perf {log_perf(LOG1, &format!("-- DisPolyFromMain: build arr_poly"), &mut timer);}
		if b_mem {dump_mem_usage("-- build arr_poly");}

		/* RECOVER LATER IF APPROACH2  NOT WORKING!
		//2. the main node sends the data to all receiver node
		//vdata will be used for setting the REAL DisPoly at each node
		let mut vdata = vec![F::zero(); 0]; 
		let mut t1= Timer::new();
		let mut sel_size = 0usize;
		let mut total_bytes = 0usize;
		t1.start();
		if me==main_processor as usize{//send data
			for i in 0..np{//send to everyone except myself
				let nmp = (main_processor + (i as u64))%(np as u64); 
				let (start, end) = DisVec::<F>::get_share_bounds(nmp as u64, np as u64, total as u64);
				let v = read_arr_from(src_file, start+1, end); 
				if me==nmp as usize{//no need to send
					vdata = v;
				}else{//send
					let vec_bytes= to_vecu8(&v);
					sel_size = vec_bytes.len();
					world.process_at_rank(nmp as i32).
						send_with_tag(&vec_bytes, me as i32);
				}
				total_bytes += sel_size;
			}
		}else{//receive data
			let r1 = world.any_process().receive_vec::<u8>();
			//sel_size = r1.0.len();
			vdata = from_vecu8::<F>(&r1.0, F::zero());
		}
		RUN_CONFIG.better_barrier("DisPoly::from_mainonde send/receive");


		if b_perf {log_perf(LOG1, &format!("-- DisPolyFromMain: distribute vecs: ele: {}, bytes: {}", total, total_bytes), &mut timer);}
		if b_mem {dump_mem_usage("-- distribute vecs");}
		RECOVER LATER ABOVE              */ 

		//3. NEW version
		broadcast_file_to_all_nodes(src_file, netarch);
		if b_perf {log_perf(LOG1, &format!("-- DisPolyFromMain: transfer src file"), &mut timer);}
		let nmp = (main_processor + (me as u64))%(np as u64); 
		let (start, end) = DisVec::<F>::get_share_bounds(nmp as u64, np as u64, total as u64);
		let vdata = read_arr_from(src_file, start+1, end); 
		if b_perf {log_perf(LOG1, &format!("-- DisPolyFromMain: local read"), &mut timer);}



		for i in 0..np{
			let newid = id*(np as u64) + (i as u64); 
			let nmp = (main_processor + (i as u64))%(np as u64); 
			if me == nmp as usize{
				arr_poly[i] = DisPoly::single_from_vec(newid, nmp, &vdata);
			}
		}
		if b_perf {log_perf(LOG1, &format!("-- DisPolyFromMain: local binacc"),&mut timer);}
		if b_mem {dump_mem_usage("-- local binacc ");}

/*
		//3. each node builds up its own local copy
		let newid = id*(np as u64) + (me as u64); 
		let nmp = (main_processor + (me as u64))%(np as u64); 
		t1.clear(); t1.start();
		arr_poly[me as usize] = DisPoly::single_from_vec(newid, nmp, &vdata);
		t1.stop();
		RUN_CONFIG.better_barrier("DisPoly::build local binacc.");
		log(LOG2, &format!("DisPoly::from_mainnode build local DisPoly: {} elements, Cost: {} ms", vdata.len(), t1.time_us/1000));
		if b_perf {log_perf(LOG1, &format!("-- local binacc"),&mut timer);}
		if b_mem {dump_mem_usage("-- local binacc ");}
*/

		//4. chain_mul all dispoly in the array to get the final dispoly
		let (vplan, dist_round_id) = DisPoly::<F>::gen_chain_mul_plan_worker(
			&arr_poly, log_2_group_size, dist_bar);	
		//log(LOG1, &format!("DEBUG USE 1002: plan: {:?}, dist_bar: {}", &vplan, dist_round_id));
		DisPoly::<F>::chain_mul_worker(&mut arr_poly,vplan, dist_round_id); 
		//println!("DEBUG USE 1002 DONE: dist_bar: {}",  dist_round_id);
		RUN_CONFIG.better_barrier("dis_poly_from_roots_from_mainnode");
		if b_perf {log_perf(LOG1, &format!("-- DisPolyFromMain: chainmul"),&mut timer);}
		if b_mem {dump_mem_usage("-- chainmul");}

		//5. return
		//let mut res = arr_poly[0].clone();
		//res.dvec.synch_size();
		arr_poly[0].dvec.synch_size();
		let mut res = arr_poly[0].clone();
		res.dvec.set_real_len();
		if b_perf {log_perf(LOG1, &format!("-- DisPolyFromMain: build binacc res"), &mut timer);}
		if b_mem {dump_mem_usage("-- build binacc res");}
		return res;

	}

	/** the worker function. The extra log_2_group_size and dist_bar
		are used for generating mul plan
		NOTE: the main node reads the ENTIRE FILE and then
		pass the information to all other nodes. ASSUMPTION:
		the main node has the access of the file.  */
	pub fn dispoly_from_roots_in_file_from_mainnode_worker_old(id: u64, main_processor: u64, src_file: &String, log_2_group_size: usize, dist_bar: usize) -> DisPoly<F>{
		//1. main node read data
		let np = RUN_CONFIG.n_proc;
		let my_rank = RUN_CONFIG.my_rank;
		let mut total = if my_rank==main_processor as usize
			{read_1st_line_as_u64(src_file) as usize} else {0usize};

		//2. build a dis_vec object jointly
		let vdata = if my_rank==main_processor as usize 
			{read_arr_from(src_file, 1, total)} else{vec![F::zero(); total]};
		let mut dvec = DisVec::<F>::new_dis_vec_with_id(id,main_processor,total,vdata);
		dvec.synch_size();
		total = dvec.len; //this is the right value now
		assert!(total>=np, "Number of roots: {} shoud > np: {}!", total, np);
		dvec.to_partitions(&RUN_CONFIG.univ);
			
		//3. build up n local distributed DisPoly, with one real, others fake
		//re-arrange the order so that the elements of DisPoly in
		//arr_poly's main_processor is its idx%main_processor
		//i.e., arr_poly[0]'s main processor is main_processor
		let mut arr_poly = vec![DisPoly::fake_single(0, 0, 0);np];
		for i in 0..np{
			let newid = id*(np as u64) + (i as u64); 
			let nmp = (main_processor + (i as u64))%(np as u64); 
			if nmp==my_rank as u64{
				let v = &dvec.partition; 
				assert!(v.len()>=1, "partition.len: {} small at {}",
					v.len(), my_rank);
				let dp = DisPoly::single_from_vec(newid, nmp, v);
				arr_poly[i]  = dp;
			}else{
				let (start, end) = DisVec::<F>::get_share_bounds(nmp as u64, np as u64, total as u64);
				let part_len = end-start+1; //because len is degree + 1
				let dp = DisPoly::fake_single(newid, nmp, part_len);
				arr_poly[i] = dp;
			}
		}

		//println!("*** DEBUG USE 1001: my_rank: {} ***, len: {}, {}, {}, {}", my_rank, arr_poly[0].dvec.len, arr_poly[1].dvec.len, arr_poly[2].dvec.len, arr_poly[3].dvec.len);
		//3. multiply all
		let (vplan, dist_round_id) = DisPoly::<F>::gen_chain_mul_plan_worker(
			&arr_poly, log_2_group_size, dist_bar);	
		//println!("DEBUG USE 1002: plan: {:?}, dist_bar: {}", &vplan, dist_round_id);
		DisPoly::<F>::chain_mul_worker(&mut arr_poly,vplan, dist_round_id); 
		//println!("DEBUG USE 1002 DONE: dist_bar: {}",  dist_round_id);
		RUN_CONFIG.better_barrier("dis_poly_from_roots_from_mainnode");
		return arr_poly[0].clone();
	}


	/** from a vector of roots generate the corresponding DensePolynomial:
		(x+v[0])(x+v[1]) ... (x+v[n]) 
	*/ 
	pub fn binacc_poly(vec: &Vec<F>) -> DensePolynomial<F>{
		return Self::binacc_poly_worker(vec, 0, vec.len(), 0);
	}
	/** from a vector of roots generate the corresponding DensePolynomial:
		(x+v[0])(x+v[1]) ... (x+v[n]) 
		start and end indicates the starting and end position (not included)
	*/ 
	pub fn binacc_poly_worker(vec: &Vec<F>, start: usize, end: usize, depth: usize) -> DensePolynomial<F>{
		let b_mem = false;
			
	    let n = end-start;
  		if n==0{
            let v1 = vec![F::one()];
            return get_poly(v1);
        }else if n==1{
	        let one = F::from(1u64);
	        let p = DensePolynomial::<F>::from_coefficients_vec(
				vec![vec[start], one]);
	        return p;
	    }else{
			if b_mem {dump_mem_usage(&format!("before_binacc: depth: {}", depth));}
	        let n2 = n/2;
			if b_mem {dump_mem_usage(&format!("before_calling p1: depth: {}", depth));}
	        let p1 = DisPoly::binacc_poly_worker(vec, start, start+n2, depth+1);
			if b_mem {dump_mem_usage(&format!("after_calling p1: depth: {}", depth));}
	        let p2 = DisPoly::binacc_poly_worker(vec, start+n2, start+n, depth+1);
			if b_mem {dump_mem_usage(&format!("after_calling p2: depth: {}", depth));}
	        let p:DensePolynomial<F> = &p1 * &p2;
			if b_mem {dump_mem_usage(&format!("after prod. depth: {}", depth));}
	        return p;
	    }
	}

	/** construct a SINGLE (not partitioned yet) DisPoly object */
	pub fn single_from_vec(id: u64, main_processor: u64, v: &Vec<F>)-> DisPoly<F>{
		let b_mem = false;

		if b_mem {dump_mem_usage("---- before single_from_vec ");}
		let dp = DisPoly::binacc_poly(v);
		if b_mem {dump_mem_usage(&format!("---- after binacc_poly: size: {} ", dp.degree()));}
		let coef = dp.coeffs;
		let dv = DisVec::<F>::new_dis_vec_with_id(id, main_processor, coef.len(), coef);
		if b_mem {dump_mem_usage("---- after disvec");}
		let dp = DisPoly{id: id, dvec: dv, is_zero: false, is_one: false};
		if b_mem {dump_mem_usage("---- after new_dp");}
		return dp;
	}


	/** Generate a NON-main processor object. Will be populated with
	contents by executing to_partition later */
	pub fn fake_single(id: u64, main_processor: u64, size: usize)->DisPoly<F>{
		let dv = DisVec::<F>::new_dis_vec_with_id(id, 
			main_processor, size, vec![]);
		let dp = DisPoly{id: id, dvec: dv, is_zero: false, is_one: false};
		return dp;
	}

	/** Generate the chain multiplication plan. 
	Returns 3d array, each is a group of distributed polynomials.
	Two stages: in stage 1, each processor will be involved in ONE and ONLY
		ONE group_mul operation, either as main_processor or siblings.
		Thus, each row contains just ONE element.
	In stage 2, sequentially execute a sequence of group muls, each
		node will participate in every group mul.
	Returns also a usize element which indicates the round number where
		group multiplication should be done distributedly.
	*/
	pub fn gen_chain_mul_plan(v: &mut Vec<DisPoly<F>>) 
	-> (Vec<Vec<Vec<usize>>>, usize){
		return DisPoly::<F>::gen_chain_mul_plan_worker(
			v, 1, 1usize<<30);
	}

	/** generate multiplication plan.
		log_2_round_size is the log2(round_group_size)
		dist_bar: is the bar for doing distributed 
	*/
	fn gen_chain_mul_plan_worker(v: &Vec<DisPoly<F>>, 
		log_2_round_size: usize, dist_bar: usize)
	->(Vec<Vec<Vec<usize>>>, usize){
		let np = RUN_CONFIG.n_proc;
		assert!(np==v.len(), "np!=v.len()");
	
		let mut rounds = (log2(np) as usize)/(log_2_round_size as usize);
		if (log2(np) as usize)%log_2_round_size!=0 {rounds+=1;}
		let group_size = 1<<log_2_round_size;
		let mut res: Vec<Vec<Vec<usize>>> = vec![];
		let mut groups = np;
		let mut poly_degree = v[0].dvec.len-1;
		let mut dist_round_id = rounds+1;
		for i in 0..rounds{
			//log!(LOG1, &format("DEBUG USE 101: plan mul round: {}, poly_degree: {}, disbar: {}, dist_round_id: {}", i, poly_degree, dist_bar, dist_round_id));
			if poly_degree>dist_bar && dist_round_id==rounds+1{
				dist_round_id = i;
			}
			let mut row: Vec<Vec<usize>> = vec![];
			groups = if groups%group_size==0 {groups/group_size} else
				{groups/group_size+1};
			for j in 0..groups{
				let mut rec = vec![];
				for k in 0..group_size{
					let id = (j*group_size + k)<<(i*log_2_round_size);
					if id<np{
						rec.push(id);
					}
				}
				row.push(rec);
			}
			res.push(row);
			poly_degree = poly_degree * group_size;
		}
		return (res, dist_round_id);
	}

	/** multiply all DisPoly object and save the result into
		the 0'th DisPoly.
		Do all ops locally: all nodes send over the data to the main processor.
		Return 1 if the node has done anywork (it can be 0 ONLY WHEN
			in stage 1 of gen_chain_mul_plan).	
	*/ 
	pub fn group_mul_local(v: &mut Vec<DisPoly<F>>, vids: &Vec<usize>) -> usize{
		let my_rank= RUN_CONFIG.my_rank;
		let mut i_use = 0usize;
		let receiver_id = v[vids[0]].dvec.main_processor;
		let world = RUN_CONFIG.univ.world();
		let group_size = vids.len();
		let mut total_len:usize = 1;
		for u in 0..vids.len(){
			let id = vids[u];
			let dp = &mut v[id];
			total_len += dp.dvec.len-1;
			if dp.dvec.main_processor==my_rank{//do something
				if u==0{//receive
					//1. receive the arrays
					let mut vdata = vec![];
					for _i in 0..group_size-1{
						let r1 = world.any_process().receive_vec::<u8>();
						let data = from_vecu8::<F>(&r1.0, F::zero());
						//let proc = r1.1.source_rank();
						//does not cost a lot, as it's taking ownership
						let pi = DensePolynomial::<F>::
							from_coefficients_vec(data);
						vdata.push(pi);
					}

					//2. multiply them up
					let mut pret = DensePolynomial::<F>::
						from_coefficients_vec(dp.dvec.vec.clone());
					for i in 0..group_size-1{
						pret = &pret * &(vdata[i]);
					}
					dp.dvec.reset_vec(pret.coeffs);
					
				}else{//send synchronously as just 1 to SEND
					let vec_bytes= to_vecu8(&dp.dvec.vec);
					world.process_at_rank(receiver_id as i32).
						send_with_tag(&vec_bytes, my_rank as i32);
				}
				i_use += 1;
			}else{//EVEN IF not involved. calculate the target size
			}
		}
		v[vids[0]].dvec.len = total_len;
		//println!("DEBUG USE 101 DONE!!!: node: {}, vids: {:?}, receiver_id: {}", my_rank, &vids, receiver_id);
		return i_use;
	}

	/** compute the closest 2^k that >= sum of degrees */
	fn sum_dvec_lens(v: &Vec<DisPoly<F>>, vids: &Vec<usize>) -> usize{
		let mut sum = 0;
		for i in 0..vids.len(){
			let dp = &v[vids[i]];
			sum += dp.dvec.len;
		}
		let log2m = log2(sum);
		let mut res = 1<<log2m;
		if res<sum {res = 2*res;}
		return res;
	}


	/** multiply all and save result into 0'th. Perform distribuedly.
		All nodes need to participate.
		NOTE: except v[0], for saving cost, ALL others are destroyed,
		we did not perform IFFT to convert back to co-efs */	
	pub fn group_mul_dist(v: &mut Vec<DisPoly<F>>, vids: &Vec<usize>) {
		//0. init
		let b_perf = false;
		let b_mem = false;
		let mut timer = Timer::new();
		timer.start();

		let _my_rank = RUN_CONFIG.my_rank;
		let total_dvec_len = DisPoly::<F>::sum_dvec_lens(v, vids);
		if b_perf {log_perf(LOG1, &format!("group_mul_dist: total_dvec_len: {}, vids.len: {}", total_dvec_len, vids.len()), &mut timer);}
		if b_mem {dump_mem_usage("BEFORE group_mul_dist");}
		
		for i in 0..vids.len(){ 
			let dp = &mut v[vids[i]];
			dp.dvec.repartition(total_dvec_len); 
		}
		RUN_CONFIG.better_barrier("group_mul_dist1");
		if b_perf {log_perf(LOG1, &format!("group_mul_dist: repartition all dvec"), &mut timer);}
		if b_mem {dump_mem_usage("group_mul_dist: repartition");}

		//2. fft
		let univ = &RUN_CONFIG.univ;
		for i in 0..vids.len(){ 
			let dp = &mut v[vids[i]];
			distributed_dizk_fft(&mut dp.dvec, univ);
		}
		if b_perf {log_perf(LOG1, &format!("group_mul_dist: fft"), &mut timer);}
		if b_mem {dump_mem_usage("group_mul_dist: fft");}

		//3. multiply points (JUST work on this partition)
		let _np = RUN_CONFIG.n_proc;
		let size = v[vids[0]].dvec.partition.len();
		for i in 0..vids.len(){
			let vi_part_size = (&v[vids[i]]).dvec.partition.len();
			assert!(vi_part_size==size, "UNMATCHED partition size at {}", i);
		}
		let target_id = vids[0];
		let (v1, v2) = v.split_at_mut(target_id);
		let (v3, v4) = v2.split_at_mut(1);
		let dp_res = &mut v3[0];
		for j in 0..size{
			for i in 1..vids.len(){
				let id = vids[i];
				assert!(id!=target_id, "id shouldbe < or > target_id");
				let dp_inp = if id<target_id {&v1[id]} else {&v4[id-target_id-1]};
				dp_res.dvec.partition[j] *= dp_inp.dvec.partition[j];
			}
		}
		RUN_CONFIG.better_barrier("group_mul_dist2");
		if b_perf {log_perf(LOG1, &format!("group_mul_dist: mul points"), &mut timer);}
		if b_mem {dump_mem_usage("group_mul_dist: mul points");}

		
		//4. ifft
		/* RECOVER IF LATER NOT WORKING 
		for i in 0..vids.len(){ 
			let dp = &mut v[vids[i]];
			distributed_dizk_ifft(&mut dp.dvec, univ);
		}
		*/
		
		for i in 1..vids.len(){ 
			let dp = &mut v[vids[i]];
			dp.destroy();
		}
		distributed_dizk_ifft(&mut v[vids[0]].dvec, univ);
		if b_perf {log_perf(LOG1, &format!("group_mul_dist: ifft"), 
			&mut timer);}
		if b_mem {dump_mem_usage("group_mul_dist: ifft");}
	}

	/** Joint collaborative function multiply all DisPoly Object
		and the result is stored in the FIRST DisPoly
		Note: the first DisPoly in v has the main_processor
	*/
	pub fn chain_mul(v: &mut Vec<DisPoly<F>>){
		let (vplan, dist_round_id) = DisPoly::<F>::gen_chain_mul_plan(v);	
		DisPoly::<F>::chain_mul_worker(v, vplan, dist_round_id);
	}

	/** free up all other elements EXCLUDING the first one */
	fn free_up(v: &mut Vec<DisPoly<F>>, vecids: &Vec<usize>){
		let b_mem = false;

		if b_mem {dump_mem_usage("BEFORE freeup");}
		for i in 1..vecids.len(){
			let id = vecids[i];
			if b_mem {log(LOG1, &format!("FREE up dpoly: {}", id));}
			let dp = &mut v[id];
			dp.destroy();
		}
		if b_mem {dump_mem_usage("AFTER freeup");}
	}
	pub fn chain_mul_worker(v: &mut Vec<DisPoly<F>>, vplan: Vec<Vec<Vec<usize>>>, dist_round_id: usize){
		let b_mem = false;
		let b_perf = false;
		let _my_rank = RUN_CONFIG.my_rank;
		let mut timer = Timer::new();
		timer.start();

		let mut i:usize = 0;
		//if b_perf {println!("*** DEBUG USE 2001 *** chain_mul_worker STARTED for node: {}, for vplan: {:?}, dist_id: {}", _my_rank, &vplan, dist_round_id);}
		for arr_vecids in &vplan{
			//1. do the multiplication locally or globally
			if i<dist_round_id{
				let mut total = 0usize;
				for vecids in arr_vecids{ 
					total += DisPoly::<F>::group_mul_local(v, &vecids);
					Self::free_up(v, vecids);
				}
				let msg = format!("total contrib of rank: {} to local_mul!=1, it is {}", RUN_CONFIG.my_rank, total);
				assert!(total<=1, "{}", &msg);
				if b_perf {log_perf(LOG1, &format!("-- --localmul round: {}, vecids: {:?}:", i, arr_vecids), &mut timer);}
				if b_mem {dump_mem_usage(&format!("-- --localmul round: {}, vecids: {:?}:", i, &arr_vecids));}
				
			}else{
				for vecids in arr_vecids{ 
					DisPoly::<F>::group_mul_dist(v, &vecids);
					Self::free_up(v, vecids);
				}
				if b_perf {log_perf(LOG1, &format!("-- --GroupMul round: {}, vecids: {:?}:", i, arr_vecids), &mut timer);}
				if b_mem {dump_mem_usage(&format!("-- --GroupMul round: {}, vecids: {:?}:", i, &arr_vecids));}
			}
			i+=1;
			RUN_CONFIG.better_barrier("chain_mul_worker");

		}
		//println!("*** DEBUG USE 2004 *** chain_mul_worker completed for node: {}, for vplan: {:?}, dist_id: {}", _my_rank, &vplan, dist_round_id);
	} 


	/** write the coeffcients to file. Note: each processor write its own
	piece. */
	pub fn write_coefs_to_file(&self, fname: &String){
		let my_rank = RUN_CONFIG.my_rank;
		let newfname = format!("{}.node_{}", fname, my_rank);
		assert!(self.dvec.b_in_cluster, "ERR in DisPoly::write_coefs_to_file: {}: make disploy distributed first!", fname);
		let v8 = to_vecu8(&self.dvec.partition);
		write_vecu8(&v8, &newfname);
		//synchronize  (to avoid early termnation)
		RUN_CONFIG.better_barrier("write_coefs_to_file");
	}
	/** read coefficients from file. Note: each processor read its own
	piece. NOTE: need to provide n (deg+1) in advance! (because all nodes
	need to know the correct number and be consistent. 
	*/
	pub fn read_coefs_from_file(&mut self, fname: &String, n: usize){
		let zero = F::zero();
		self.dvec.b_in_cluster = true;
		self.dvec.len = n;
		let my_rank = RUN_CONFIG.my_rank;
		let newfname = format!("{}.node_{}", fname, my_rank);
		let vec_serialized = read_vecu8(&newfname);
		self.dvec.partition = from_vecu8(&vec_serialized, zero);
		self.reset_flags();
		RUN_CONFIG.better_barrier("read_coefs_from_file");
	}
	/** no need to set up n, will synch size later */
	pub fn read_coefs_from_file_new(&mut self, fname: &String){
		let zero = F::zero();
		self.dvec.b_in_cluster = true;
		let my_rank = RUN_CONFIG.my_rank;
		let newfname = format!("{}.node_{}", fname, my_rank);
		let vec_serialized = read_vecu8(&newfname);
		self.dvec.partition = from_vecu8(&vec_serialized, zero);
		self.dvec.real_len_set = false;

		//3. collect the total len
		let main = self.dvec.get_main_processor() as usize;
		let world = RUN_CONFIG.univ.world();
		let np = RUN_CONFIG.n_proc as usize;
		let mut total_len = 0;
		let me = RUN_CONFIG.my_rank;
		let mut vres = vec![0usize; np];
		if me==main{//wait 
				vres[me as usize] = self.dvec.partition.len();
				for _i in 0..np-1{
					let r1 = world.any_process().receive_vec::<u8>();
					let data = r1.0;
					let sender = r1.1.tag();
					let v = u64::from_be_bytes(data[0..8].try_into().unwrap());
					vres[sender as usize] = v as usize;
				}
				for i in 0..np{
					total_len += vres[i];
				}
		}else{
				let mylen = self.dvec.partition.len() as u64;
				let vec_bytes= mylen.to_be_bytes();
				world.process_at_rank(main as i32).
					send_with_tag(&vec_bytes, me as i32);
		}
		//4. main broadcast it
		let mut val = total_len as u64;
		world.process_at_rank(main as i32).broadcast_into(&mut val);
		self.dvec.len = val as usize;
		self.reset_flags();
		RUN_CONFIG.better_barrier("newread_coefs_from_file step 2");
	}

	pub fn gen_dp(size: usize)->DisPoly<F>{
		let dp = Self::gen_dp_from_seed(size, get_time());
		return dp;
	}
	pub fn gen_dp_from_seed(size: usize, seed: u128)->DisPoly<F>{
		let np = RUN_CONFIG.n_proc;
		let me = RUN_CONFIG.my_rank;
		let share = size/np;
		let last_share = size/np + size%np;
		let my_size = if me<np-1 {share} else {last_share}; 
		let arr_fe = rand_arr_field_ele::<F>(my_size, seed + (me as u128) * 17171);
		let dv= DisVec::<F>::new_from_each_node(0, 0, size, arr_fe);
		let mut dp= DisPoly{id: 0, dvec: dv, is_one: false, is_zero: false};
		dp.to_partitions();
		return dp;
	}
	pub fn gen_binacc(size: usize) -> DisPoly<F>{
		let n = size;
  		let mut rng = gen_rng();
		let mut vec = vec![];
		for _i in 0..n{
			let v:u64 = rng.gen::<u64>() + 1;
			vec.push(F::from(v));
		}
    	let p = Self::binacc_poly(&vec);
		return DisPoly::from_serial(0, &p, p.degree()+1);
	}
	pub fn gen_binacc_from_seed(size: usize, seed: u128, netarch: &(Vec<u64>,Vec<usize>,Vec<Vec<usize>>)) -> DisPoly<F>{
		let b_perf = false;
		let mut timer = Timer::new();
		timer.start();
		let me = RUN_CONFIG.my_rank;
		let np = RUN_CONFIG.n_proc;
		let tmp_file = format!("/tmp/arr_{}.dat", seed);
		if me==0{
			//make size-np for avoiding overflow power of 2 
			//due to (coefs +1)
			let arr = rand_arr_unique_u64(size-np, 62, seed as u128);
			write_arr_with_size(&arr, &tmp_file);
		}
		RUN_CONFIG.better_barrier("wait writing is done");
		if b_perf {log_perf(LOG1, &format!(" -- Creating file: {}", size), &mut timer);}
		let p = DisPoly::<F>::dispoly_from_roots_in_file_from_mainnode(0, 0, &tmp_file, netarch);
		if me==0{
			remove_file(&tmp_file);
		}
		RUN_CONFIG.better_barrier("wait remove is done");
		if b_perf {log_perf(LOG1, &format!(" -- Creating DPoly: {}", size), &mut timer);}
		return p;
	}

	// ---- The following are for FEEA (fast GCD distributed)
	/// 2x2 matrix multiply with 1x1
	/** 
		Return the coef at the highest degree, i.e.,
		given f = a_0 + .... a_i x^i, returning a_i
		ALL NODES will get the same value (main node broadcast)
	*/
	pub fn lead_coef(f: &mut DisPoly<F>) -> F{
		if f.is_zero() {return F::zero();}
		let np = RUN_CONFIG.n_proc as usize;
		let me = RUN_CONFIG.my_rank as usize;
		//1. identify the segment containing the last
		f.dvec.set_real_len();	
		let mut node_id = 0;
		let degree = f.dvec.real_len - 1;
		let share_size = f.dvec.len/np;
		if f.dvec.b_in_cluster{
			node_id = if degree>=share_size*np {np-1} else {degree/share_size};
		}
		//2. broadcast 
		let send_size = F::zero().serialized_size();
		let mut vec_send= vec![0u8; send_size]; 
		let world = RUN_CONFIG.univ.world();
		let root_process = world.process_at_rank(node_id as i32);
		if me==node_id{//if last one search for 
			if f.dvec.b_in_cluster{
				let local_degree = degree - share_size * node_id;
				let res = f.dvec.partition[local_degree]; 
				vec_send = to_vecu8(&vec![res]);
			}else{
				let res = f.dvec.vec[degree]; 
				vec_send = to_vecu8(&vec![res]);
			}
		}
		root_process.broadcast_into(&mut vec_send);
		let res =  from_vecu8::<F>(&vec_send, F::zero())[0];
		return res;
	}
	pub fn m2xm1(m1: &mut [DisPoly<F>; 4], m2: &mut [DisPoly<F>; 2]) -> [DisPoly<F>; 2]{
			let mut v1 = Self::mul(&mut m1[0],&mut m2[0]);
			let mut v2 = Self::mul(&mut m1[1] , &mut m2[1]);
			let mut v3 = Self::mul(&mut m1[2],&mut m2[0]);
			let mut v4 = Self::mul(&mut m1[3] , &mut m2[1]);
		let res = [
			Self::add(&mut v1, &mut v2),
			Self::add(&mut v3, &mut v4)
		];
		return res;
	}

	/// m2 x m2 -> m2
	/// m2 is represented as a 4 elements 1-d array layed all elements
	/// row by row
	pub fn m2xm2(m1: &mut [DisPoly<F>; 4], m2: &mut [DisPoly<F>; 4]) -> [DisPoly<F>; 4]{
		let mut timer = Timer::new();
		timer.start();
		let res = [
			Self::add(&mut Self::mul(&mut m1[0] , &mut m2[0]) , &mut Self::mul(&mut m1[1] , &mut m2[2])),
			Self::add(&mut Self::mul(&mut m1[0] , &mut m2[1]) , &mut Self::mul(&mut m1[1] , &mut m2[3])),
			Self::add(&mut Self::mul(&mut m1[2] , &mut m2[0]) , &mut Self::mul(&mut m1[3] , &mut m2[2])),
			Self::add(&mut Self::mul(&mut m1[2] , &mut m2[1]) , &mut Self::mul(&mut m1[3] , &mut m2[3])),
		];
		log_perf(LOG2, "PERF_USE_MX", &mut timer);
		return res;
	}

	/// return the half gcd matrix accoring to ref
	/// Performance: 10k -> 12 seconds, 100k->124 seconds, 1 million: 1287 seconds
	/// *** NOTE *** 
	/// For a and b have factors, occasinally returned polynomials
	/// DO NOT STRADDLE over a.degree()/2, strangely (need the sequence of
	/// egcd-matrix to be nomal!
	///
	/// Performance: almost the same as serial when 8 nodes per computer
	pub fn hgcd_worker(a: &mut DisPoly<F>, b: &mut DisPoly<F>, level: usize) -> [DisPoly<F>;4]{
		a.dvec.set_real_len();
		b.dvec.set_real_len();


		let max_size = if a.dvec.real_len > b.dvec.real_len {a.dvec.real_len} else {b.dvec.real_len};
		if max_size<RUN_CONFIG.minbar_dis_feea{//do serial version
			let sa = a.to_serial();
			let sb = b.to_serial();
			let r1 = hgcd_worker(&sa, &sb);
			let r2 = [
				Self::from_serial(a.id, &r1[0], max_size), 
				Self::from_serial(a.id, &r1[1], max_size), 
				Self::from_serial(a.id, &r1[2], max_size), 
				Self::from_serial(a.id, &r1[3], max_size) 
			];		
			return r2;
		}

		let d = a.degree();
		let m = half(d);
		let one = Self::one(0);
		let zero= Self::zero(0);
		let b_degree = b.degree();
		if b_degree<m || b_degree==0{
			return [one.clone(), zero.clone(), zero.clone(), one.clone()];	
		}
	
		let mut a1 = Self::div_by_xk(&a, m);
		let mut b1 = Self::div_by_xk(&b, m);
		let mut mat1 = Self::hgcd(&mut a1, &mut b1, level+1);
		let cp_a = a.clone();
		let cp_b = b.clone();
		let mut res= Self::m2xm1(&mut mat1, &mut [cp_a, cp_b]);

		let (lres, rres) = res.split_at_mut(1);
		let t = &mut (lres[0]);
		let s = &mut (rres[0]);
		s.dvec.set_real_len(); //get the right degree
		if s.degree()<m{ 
			return mat1;
		}


		let (mut q,mut r) = Self::divide_with_q_and_r(t, s);
		if r.is_zero(){
			let mut zerop = zero.clone();
			let negq = Self::sub(&mut zerop,  &mut q);
			let mut mat2 = [zero.clone(), one.clone(), one.clone(), negq.clone()]; 
			let res = Self::m2xm2(&mut mat2, &mut mat1);
			return res;
		}
	
		let v = Self::lead_coef(&mut r);
		let spv = get_poly(vec![v]);
		let pv = Self::from_serial(a.id, &spv, 1);
		let rbar = &mut r;
		rbar.mul_by_const(v);
		let mut vq = &mut q;
		vq.mul_by_const(v);

		let negvq = Self::sub(&mut zero.clone(), &mut vq);
		let mut mat2 = [zero.clone(), one.clone(), pv, negvq];
		let l = 2*m - s.degree();
		let mut s1 = Self::div_by_xk(&s, l);
		let mut r1 = Self::div_by_xk(&rbar, l);
		let mut mat3 = Self::hgcd(&mut s1, &mut r1, level+1);
		let mut res1 = Self::m2xm2(&mut mat3, &mut mat2);
		let res = Self::m2xm2(&mut res1, &mut mat1);
		return res;
	}

	/** call xgcd_special if degree is low */
	//#[inline(always)]
	pub fn hgcd(a: &mut DisPoly<F>, b: &mut DisPoly<F>, level: usize) -> [DisPoly<F>;4]{
		let a_degree = a.degree();
		if a_degree<=256{
			//let (_,_,s1,t1,s2,t2) = xgcd_special(&a.to_serial_at_each_node(), &b.to_serial_at_each_node(), half(a_degree));
			let mut timer = Timer::new();
			timer.start();
			let (_,_,s1,t1,s2,t2) = xgcd_special(&a.to_serial(), &b.to_serial(), half(a_degree));

			let mat = [
				Self::from_serial(a.id, &s1, s1.degree()+1),
				Self::from_serial(a.id, &t1, t1.degree()+1),
				Self::from_serial(a.id, &s2, s2.degree()+1),
				Self::from_serial(a.id, &t2, t2.degree()+1)];
			log_perf(LOG2, "PERF_USE_XGCD", &mut timer);
			return mat;
		}else{
			return Self::hgcd_worker(a, b, level+1);
		}	
	}

	pub fn mgcd(a: &mut DisPoly<F>, b: &mut DisPoly<F>)
		-> [DisPoly<F>;4]{
			return Self::mgcd_worker(a, b, 1);
	}

	/// requires that degree of a>=b. Output 2x2 M s.t. M(a b) = (gcd(a,b) 0)
	pub fn mgcd_worker(a: &mut DisPoly<F>, b: &mut DisPoly<F>, level: usize) 
		-> [DisPoly<F>;4]{
		let b_perf = false;
		let mut timer = Timer::new();
		timer.start();

		let zerop = DisPoly::<F>::zero(0);
		let one = DisPoly::<F>::one(0);
		let zero = DisPoly::<F>::zero(0);
		let mut mat1 = Self::hgcd(a, b, level+1);

		let cp_a = a.clone();
		let cp_b = b.clone();


		let mut res = Self::m2xm1(&mut mat1, &mut [cp_a, cp_b]);
		let (lres, rres) = res.split_at_mut(1);
		let t = &mut (lres[0]);
		let s = &mut (rres[0]);
		if s.is_zero(){ 
			if b_perf {log_perf(LOG1, &format!("++++ mgcd_workerCase1: level: {}, a: {}, b: {}, s: {}, t: {}", level, a.dvec.len, b.dvec.len, s.dvec.len, t.dvec.len), &mut timer);}
			return mat1; 
		}


		s.dvec.set_real_len();
		if s.dvec.real_len*2 < s.dvec.len{//time to repack
			let rlen = s.dvec.real_len;
			s.repartition(rlen);
		}

		t.dvec.set_real_len();
		if t.dvec.real_len*2 < t.dvec.len{//time to repack
			let rlen = t.dvec.real_len;
			t.repartition(rlen);
		}
	

		let (mut q,mut r) = Self::divide_with_q_and_r(t, s);
		if r.is_zero(){
			let negq = Self::sub(&mut zerop.clone(), &mut q);
			let mut mat2=[zero.clone(),one.clone(),one.clone(), negq.clone()]; 
			let res = Self::m2xm2(&mut mat2, &mut mat1);
			if b_perf {log_perf(LOG1, &format!("++++ mgcd_workerCase2: level: {}, a: {}, b: {}, q: {}, r: {}", level, a.dvec.len, b.dvec.len, q.dvec.len, r.dvec.len), &mut timer);}
			return res;
		}

		let v = Self::lead_coef(&mut r);
		let spv = get_poly(vec![v]);
		let pv = Self::from_serial(a.id, &spv, 1);
		let mut rbar = &mut r;
		rbar.mul_by_const(v);
		let mut vq = &mut q;
		vq.mul_by_const(v);


		let negvq = Self::sub(&mut zerop.clone(),  &mut vq);
		let mut mat2 = [zero.clone(), one.clone(), pv.clone(), negvq.clone()];
		let mut mat3 = Self::mgcd_worker(s, &mut rbar, level+1);
		let res = Self::m2xm2(&mut Self::m2xm2(&mut mat3, &mut mat2), &mut mat1);
		if b_perf {log_perf(LOG1, &format!("++++ mgcd_workerCase3: level: {}, a: {}, b: {}", level, a.dvec.len, b.dvec.len), &mut timer);}

		return res;
	}


	/// returns gcd(a,b), s and t s.t. sa + tb = gcd(a,b)
	pub fn feea_worker(a: &mut DisPoly<F>, b: &mut DisPoly<F>, level: usize) -> (DisPoly<F>, DisPoly<F>, DisPoly<F>){
		let b_perf = false;
		let mut timer = Timer::new();
		timer.start();

		if a.degree()>b.degree(){
			let mut mat= Self::mgcd_worker(a, b, level+1);
			let (lres, rres) = mat.split_at_mut(1);
			let s = &mut (lres[0]);
			let t = &mut (rres[0]);
			let g = Self::add(&mut Self::mul(s,a) ,  &mut Self::mul(t,b));
			if b_perf {log_perf(LOG1, 
				&format!("++++ FEEAWorkerCase1: level: {}, size a: {}, b: {}, s: {}, t: {}", level, a.dvec.len, b.dvec.len, s.dvec.len, t.dvec.len), 
				&mut timer);}
			return (g, s.clone(), t.clone());
		}else if a.degree()==b.degree(){
			//THIS PART of the algorithm should be fixed
			//Let q = a/b (a constant as degree is the same)
			//Let sb + t(a - qb) = g  // here s,t is the return of feea(b,&modres)
			//Let s'b + ta = g
			// This leads to s' = s-tq
			let (mut q, mut r) = Self::divide_with_q_and_r(a, b);
			let (g, mut s, mut t) = Self::feea_worker(b, &mut r, level+1); 
			let sprime = Self::sub(&mut s , &mut Self::mul(&mut t , &mut q)); 
			if b_perf {log_perf(LOG1, 
				&format!("++++ FEEAWorkerCase2: level: {}, size a: {}, b: {}, s: {}, t: {}", level, a.dvec.len, b.dvec.len, s.dvec.len, t.dvec.len), 
				&mut timer);}
			return (g, t, sprime);
		}else{
			//THIS PART of the algorithm of the original program should be fixed
			let (g, s, t) = Self::feea_worker(b, a, level+1);
			if b_perf {log_perf(LOG1, 
				&format!("++++ FEEAWorkerCase3: level: {}, size a: {}, b: {}, s: {}, t: {}", level, a.dvec.len, b.dvec.len, s.dvec.len, t.dvec.len), 
				&mut timer);}
			return (g,t.clone(),s.clone());
		}
	}

	//THIS IS THE ONE TO CALL: fast Eucledian algorithm
	/// returns gcd(a,b), s and t s.t. sa + tb = gcd(a,b)
	/// Performance: SERIAL version 
	///            1M entries - 500 seconds sec (100 times of mul)
	///	Distributed (this one):
	/// 1k: 2sec
	/// 16k: 6sec 
	/// 128k: 63sec
	/// 1M: 680 sec -> AFTER improving DIV -> 80 sec 8 nodes
	pub fn feea(a: &mut DisPoly<F>, b: &mut DisPoly<F>) -> 
		(DisPoly<F>, DisPoly<F>, DisPoly<F>){
		let b_perf = false;
		Self::check_same_main_processor(a, b);
		a.dvec.set_real_len();
		b.dvec.set_real_len();
		let mut timer = Timer::new();
		timer.start();
		let max_size = if a.dvec.real_len > b.dvec.real_len {a.dvec.real_len} else {b.dvec.real_len};
		if max_size<RUN_CONFIG.minbar_dis_feea{//do serial version
			let sa = a.to_serial();
			let sb = b.to_serial();
			let mut r1 = get_poly(vec![F::from(1u64)]);
			let mut r2 = get_poly(vec![F::from(1u64)]);
			let mut r3 = get_poly(vec![F::from(1u64)]);
			if RUN_CONFIG.my_rank==a.dvec.main_processor{
				(r1, r2, r3) = feea(&sa, &sb);
				if b_perf {log_perf(LOG1, 
					&format!("++++ Sequentail FEEA size: {}", sa.degree()), 
					&mut timer);}
			}
			RUN_CONFIG.better_barrier("wait for feea");
			let mut dr1 = Self::from_serial(a.id, &r1, r1.degree()+1);
			let mut dr2 = Self::from_serial(a.id, &r2, r2.degree()+1);
			let mut dr3 = Self::from_serial(a.id, &r3, r3.degree()+1);
			dr1.dvec.synch_size();
			dr2.dvec.synch_size();
			dr3.dvec.synch_size();
			return (dr1, dr2, dr3);
			
		}

		let (mut g,mut s,mut t) = Self::feea_worker(a, b, 1);
		let v = Self::lead_coef(&mut g);
		if v.is_zero() { return (g,s,t); }

		g.div_by_const(v);	
		s.div_by_const(v);	
		t.div_by_const(v);	
		if b_perf {log_perf(LOG1, 
			&format!("++++ Distributed FEEA: size a: {}, b: {}, s: {}, t: {}, gcd: {}", a.dvec.len, b.dvec.len, s.dvec.len, t.dvec.len, g.dvec.len), 
			&mut timer);}
		return (g, s, t);
	}

	// --------------------------------------------------------
	// New FEEA (gcd + Bizout's coeffs)
	// --------------------------------------------------------

	/// get my logical index in round. If not in round, return
	/// np (the max size possible + 1)
	/// round starts from 1 and ends at log(np) - 1
	fn get_my_idx_in_round(me: usize, round: usize)->usize{
		let factor = 1<<round;
		return if me%factor==0 {me/factor} else {RUN_CONFIG.n_proc};
	}

	/// get the real idx located in round
	fn get_real_idx_in_round(idx: usize, round: usize) -> usize{
		let factor = 1<<round;
		return idx*factor;
	} 


	/// compute distributed binacc at each nodes and propagate up
	/// populate the data for fd
	/// dis_level decides whether to do it serial or distributed
	fn compute_binacc(a: &mut DisVec<F>, rounds: usize, fd: &mut FeeaData<F>, dis_level: usize){
		//1. compute the binacc at each node (partitioning a)
		let mut timer = Timer::new();
		timer.start();
		let me = RUN_CONFIG.my_rank;
		let np = RUN_CONFIG.n_proc;
		assert!(a.b_in_cluster, "run a.to_partitions!");
		let vdata = &a.partition;
		let poly = if vdata.len()==0 {get_poly(vec![F::from(1u64)])}
			else{ DisPoly::<F>::binacc_poly(vdata)};
		let da = DisPoly::from_serial(me as u64, &poly, poly.degree()+1);
		fd.vec_p1[0][me] = da;	
		let world = RUN_CONFIG.univ.world();
		RUN_CONFIG.better_barrier("wait for all serial binacc");
		for i in 0..np{
			fd.vec_p1[0][i].dvec.main_processor = i;
		}

		//2. for each round
		let mut unit = 1;
		for round in 1..rounds{
			unit *= 2;
			if round<dis_level{
				//2.1 serial case child-> parent
				let idx = Self::get_my_idx_in_round(me, round-1);
				if idx!=np && idx%2==1{//sender
					let receiver= Self::get_real_idx_in_round(idx-1, round-1);
					assert!(!fd.vec_p1[round-1][idx].dvec.b_in_cluster, "vec_p1[{}] should not be in cluster!", round-1);
					let vec_bytes= to_vecu8(&fd.vec_p1[round-1][idx].dvec.vec);
					world.process_at_rank(receiver as i32).send_with_tag(&vec_bytes, me as i32);
				}
				let idx_new = Self::get_my_idx_in_round(me, round);
				if idx!=np && idx%2==0{//receiver
					let r1 = world.any_process().receive_vec::<u8>();
					let v = from_vecu8::<F>(&r1.0, F::zero());
					let p1 = get_poly(v);
					let p2 = fd.vec_p1[round-1][idx].to_serial();
					let p = &p1 * &p2;
					let mut newdp = Self::
						from_serial(me as u64, &p, p.degree()+1);
					newdp.dvec.main_processor = idx_new * unit;
					fd.vec_p1[round][idx_new] = newdp;
				}
			}else{
				//2.2 distributed case
				//sequentially process the multiplication of all nodes
				let num_nodes = 1<<(rounds-round-1);
				for i in 0..num_nodes{
					let receiver = Self::get_real_idx_in_round(i*2, round-1);
					let sender = Self::get_real_idx_in_round(i*2+1, round-1);
					let mut dp1 = fd.vec_p1[round-1][2*i].clone();
					let mut dp2 = fd.vec_p1[round-1][2*i+1].clone();
					dp1.reset_main_processor(receiver);
					dp2.reset_main_processor(sender);
					if round==dis_level{
						dp1.to_partitions();
						dp2.to_partitions();
					}

					let mut dp = Self::mul(&mut dp1, &mut dp2);
					dp.reset_main_processor(receiver);
					fd.vec_p1[round][i] = dp;
				}
			}
			//3. set up the main processor correctly for all vectors
			for i in 0..fd.vec_p1[round].len(){
				fd.vec_p1[round][i].dvec.main_processor = i*unit;
			}
			RUN_CONFIG.better_barrier("round");
		}
		log_perf(LOG1, "compute_bin", &mut timer); 
	}

	/// compute p2 (propagate from top level to bottom level)
	fn compute_p2(b: &mut DisPoly<F>, rounds: usize, fd: &mut FeeaData<F>, dis_level: usize){
		//1. compute the binacc at each node
		let mut timer = Timer::new();
		let mut timer2 = Timer::new();
		timer.start();
		timer2.start();
		let me = RUN_CONFIG.my_rank;
		let me64 = me as u64;
		let np = RUN_CONFIG.n_proc;
		if !b.dvec.b_in_cluster {b.to_partitions();}
		let world = RUN_CONFIG.univ.world();
		fd.vec_p2[rounds-1][0] = b.clone();

		//2. for each round
		let mut unit = np;
		for round in (1..rounds).rev(){
			unit = unit / 2;
			let num_nodes = 1<<(rounds-round-1);
			if round<=dis_level{
				//1.1 convert to serial at border
				if round==dis_level{//convert all p2 to serial
					for i in 0..num_nodes{
						let owner = Self::get_real_idx_in_round(i, round);
						let dp = &mut fd.vec_p2[round][i];
						dp.reset_main_processor(owner);
						let sp = dp.to_serial(); //only owner gets it
						let mut dp2 = Self::from_serial(dp.dvec.main_processor as u64, &sp, sp.degree() + 1);
						dp2.dvec.synch_size();
						fd.vec_p2[round][i] = dp2;	
					}
				}

				//1.2 pass p2 from parent -> right child
				let idx = Self::get_my_idx_in_round(me, round);
				if idx!=np{//sender
					let receiver= Self::get_real_idx_in_round(idx*2+1, round-1);
					assert!(!fd.vec_p2[round][idx].dvec.b_in_cluster, "vec_p2[{}] [{}] should not be in cluster!", round, idx);
					let vec_bytes= to_vecu8(&fd.vec_p2[round][idx].dvec.vec);
					world.process_at_rank(receiver as i32).send_with_tag(&vec_bytes, me as i32);
				}

				//1.3 right child receives from parent
				let idx_new = Self::get_my_idx_in_round(me, round-1);
				let p2; //serial poly
				if idx_new!=np{
					if idx_new%2==1{//right child
						let r1 = world.any_process().receive_vec::<u8>();
						let v = from_vecu8::<F>(&r1.0, F::zero());
						p2 = get_poly(v);
					}else{//left child
						let dp2 = &mut fd.vec_p2[round][idx];
						assert!(dp2.dvec.main_processor==me, "dp2 main is not me: {}", me);
						p2 = get_poly(dp2.dvec.vec.clone());
					}
					//1.4 every valid node perform calculation
					let dp1 = & fd.vec_p1[round-1][idx_new];
					assert!(!dp1.dvec.b_in_cluster, "dp1 should be single node mode!");
					assert!(dp1.dvec.main_processor==me, "dp1.main processor incorrect");
					let p1 = get_poly(fd.vec_p1[round-1][idx_new].dvec.vec.clone());
					let (du, myp2) = adapt_divide_with_q_and_r(&p2, &p1);
					fd.vec_p2[round-1][idx_new] = Self::from_serial(me64, &myp2, myp2.degree()+1);
					fd.vec_u[round-1][idx_new] = Self::from_serial(me64, &du, du.degree()+1);
				}
				//1.5 sets the main processor of all calculated nodes
				let num_nodes_target = 1<<(rounds-round);
				for i in 0..num_nodes_target{
					let idx = Self::get_real_idx_in_round(i, round-1);
					fd.vec_p2[round-1][i].dvec.main_processor = idx;
					fd.vec_p2[round-1][i].dvec.id = idx as u64;
					fd.vec_u[round-1][i].dvec.main_processor = idx;
					fd.vec_u[round-1][i].dvec.id = idx as u64;
				}
			}else{
				//2.2 distributed case
				//sequentially process the multiplication of all nodes
				for i in 0..num_nodes{
				  {//block 1 for receiver
let mut t1 = Timer::new();
t1.start();
					let dp = &mut fd.vec_p2[round][i];
					let receiver = Self::get_real_idx_in_round(i*2, round-1);
					let dp1 = &mut fd.vec_p1[round-1][i*2];
					assert!(dp1.dvec.main_processor==receiver, "dp1.main_processor != receiver: {}", receiver);
					dp.reset_main_processor(receiver);
					let (mut du1, mut dp2_1) = Self::divide_with_q_and_r(dp, dp1);
					dp2_1.reset_main_processor(receiver);
					du1.reset_main_processor(receiver);
					fd.vec_p2[round-1][i*2] = dp2_1;
					fd.vec_u[round-1][i*2] = du1;
				  }
				  {//block 2
let mut t1 = Timer::new();
t1.start();
					let dp = &mut fd.vec_p2[round][i];
					let sender= Self::get_real_idx_in_round(i*2+1, round-1);
					let dp1 = &mut fd.vec_p1[round-1][i*2+1];
					dp1.reset_main_processor(sender);
					dp.reset_main_processor(sender);
					let (mut du1, mut dp2_1) = Self::divide_with_q_and_r(dp, dp1);
					dp2_1.reset_main_processor(sender);
					du1.reset_main_processor(sender);
					let receiver = Self::get_real_idx_in_round(i*2, round-1);
					dp.reset_main_processor(receiver);
					fd.vec_p2[round-1][i*2+1] = dp2_1;
					fd.vec_u[round-1][i*2+1] = du1;
				  }
				}//end for
			}
			RUN_CONFIG.better_barrier("round");
			log_perf(LOG1, &format!("compute_p2 round: {}", round), &mut timer); 
		}
		log_perf(LOG1, "compute_p2", &mut timer); 
	}

	/// check poly locally defined correctly
	fn verify_local(p: &DisPoly<F>, name: &str, main: usize){
		assert!(!p.dvec.b_in_cluster, "{} should be local poly", name);
		assert!(p.dvec.main_processor==main, "{}.main != {}", name, main);
	}

	/// only works when the main proc
	fn extract_poly(dp:&DisPoly<F>)-> DensePolynomial<F>{
		Self::verify_local(dp, "poly", RUN_CONFIG.my_rank);
		let p = get_poly(dp.dvec.vec.clone());
		return p;
	}

	//compute the s and t (Bizout's coefs from bottom-up)
	fn compute_st(rounds: usize, fd: &mut FeeaData<F>, dis_level: usize){
		//1. init set up
		let mut timer = Timer::new();
		let mut timer2 = Timer::new();
		timer.start();
		timer2.start();
		let me = RUN_CONFIG.my_rank;
		let me64 = me as u64;
		let np = RUN_CONFIG.n_proc;
		let world = RUN_CONFIG.univ.world();

		//2. for each round
		for round in 0..rounds{
			if round==0{//compute local s, t (everyone computes it)
				assert!(0<dis_level, "dis_level must be > 0!");
				Self::verify_local(&fd.vec_p1[0][me], "p1", me);
				Self::verify_local(&fd.vec_p2[0][me], "p2", me);
				Self::verify_local(&fd.vec_u[0][me], "u", me);
				let p1 = Self::extract_poly(&fd.vec_p1[0][me]);	
				let p2 = Self::extract_poly(&fd.vec_p2[0][me]);	
				let u = Self::extract_poly(&fd.vec_u[0][me]);	
				let (_, mut s, t) = feea(&p1, &p2);
				if round!=rounds-1{
					s = &s - &(&u*&t);
				}
				fd.vec_s[0][me] = Self::from_serial(me64, &s, s.degree()+1); 
				fd.vec_t[0][me] = Self::from_serial(me64, &t, t.degree()+1); 
			}else if round<dis_level{
				//2.1 serial case. right child -> parent: s2, t2
				let idx = Self::get_my_idx_in_round(me, round-1);
				if idx!=np && idx%2==1{//sender (right child)
					let receiver= Self::get_real_idx_in_round(idx-1, round-1);
					Self::verify_local(&fd.vec_s[round-1][idx], "s", me);
					let vec_bytes= to_vecu8(&fd.vec_s[round-1][idx].dvec.vec);
					world.process_at_rank(receiver as i32).send_with_tag(&vec_bytes, 0);
					Self::verify_local(&fd.vec_t[round-1][idx], "t", me);
					let vec_bytes= to_vecu8(&fd.vec_t[round-1][idx].dvec.vec);
					world.process_at_rank(receiver as i32).send_with_tag(&vec_bytes, 1);
				}

				let idx_new = Self::get_my_idx_in_round(me, round);
				if idx_new!=np{//receiver
					//2.2 parent receive the vector s2 and t2
					let r1 = world.any_process().receive_vec::<u8>();
					let v1 = from_vecu8::<F>(&r1.0, F::zero());
					let tag1 = r1.1.tag() as usize;

					let r2 = world.any_process().receive_vec::<u8>();
					let v2 = from_vecu8::<F>(&r2.0, F::zero());
					let tag2 = r2.1.tag() as usize;
					assert!(tag1+tag2==1, "invalid tags!");
					
					let (s2,t2) = if tag1==0 {(get_poly(v1), get_poly(v2))}
						else {(get_poly(v2), get_poly(v1))};

					//2.2 parent get from left child s1, t1, p1_child
					// also p1 and p2 from itself
					Self::verify_local(&fd.vec_s[round-1][idx_new*2],"s", me);
					Self::verify_local(&fd.vec_t[round-1][idx_new*2],"t", me);
					Self::verify_local(&fd.vec_p1[round-1][idx_new*2],"p1", me);
					Self::verify_local(&fd.vec_p1[round][idx_new],"p1", me);
					Self::verify_local(&fd.vec_p2[round][idx_new],"p2", me);
					Self::verify_local(&fd.vec_u[round][idx_new],"u", me);
					let s1 = Self::extract_poly(&fd.vec_s[round-1][idx_new*2]);
					let t1 = Self::extract_poly(&fd.vec_t[round-1][idx_new*2]);
					let p1_child = Self::extract_poly(&fd.vec_p1[round-1][idx_new*2]);
					let p1= Self::extract_poly(&fd.vec_p1[round][idx_new]);
					let p2= Self::extract_poly(&fd.vec_p2[round][idx_new]);
					let u = Self::extract_poly(&fd.vec_u[round][idx_new]);	

					//2.3 perform calculation
					let s = &s1 * &s2;
					let t = &t1 + &(&t2*&(&s1*&p1_child));
					let (u2, t) = adapt_divide_with_q_and_r(&t, &p1);
					let mut s = &s + &(&u2 * &p2);

					if round!=rounds-1{
						s = &s - &(&u*&t);
					}
					fd.vec_s[round][idx_new] = Self::from_serial(me64, &s, s.degree()+1);
					fd.vec_t[round][idx_new] = Self::from_serial(me64, &t, t.degree()+1);
					
				}
			}else{
				let num_nodes = 1<<(rounds-round-1);
				//2.2 distributed case
				//sequentially process the multiplication of all nodes
				for i in 0..num_nodes{
					let lc = Self::get_real_idx_in_round(i*2, round-1);
					let rc = Self::get_real_idx_in_round(i*2+1, round-1);
					let mut ds1 = &mut fd.vec_s[round-1][2*i].clone();
					let mut dt1 = &mut fd.vec_t[round-1][2*i].clone();
					let mut dp1_child = &mut fd.vec_p1[round-1][2*i].clone();
					let mut dp1 = &mut fd.vec_p1[round][i];
					let mut ds2 = &mut fd.vec_s[round-1][2*i+1].clone();
					let mut dt2 = &mut fd.vec_t[round-1][2*i+1].clone();
					let mut dp2 = &mut fd.vec_p2[round][i];
					let mut du = &mut fd.vec_u[round][i];

					//Unfortunately, cannot use a for loop of a vec due to move
					//forbidden, this looks silly
					if round==dis_level{
						ds1.dvec.main_processor=lc;
						ds1.dvec.synch_size();
						ds1.to_partitions();
						dt1.dvec.main_processor=lc;
						dt1.dvec.synch_size();
						dt1.to_partitions();
						dp1_child.dvec.main_processor=lc;
						dp1_child.dvec.synch_size();
						dp1_child.to_partitions();
						dp1.dvec.main_processor=lc;
						dp1.dvec.synch_size();
						dp1.to_partitions();
						dp2.dvec.main_processor=lc;
						dp2.dvec.synch_size();
						dp2.to_partitions();
						du.dvec.main_processor=lc;
						du.dvec.synch_size();
						du.to_partitions();
	
						ds2.dvec.main_processor=rc;
						ds2.dvec.synch_size();
						ds2.to_partitions();
						dt2.dvec.main_processor=rc;
						dt2.dvec.synch_size();
						dt2.to_partitions();
					}

					let mut s = Self::mul(ds1, ds2);
					let mut tmp1 = Self::mul(ds1, dp1_child); 
					let mut tmp2 = Self::mul(dt2, &mut tmp1); 
					let mut t = Self::add(dt1, &mut tmp2);

					let (mut u2, mut t)= Self::divide_with_q_and_r(&mut t, dp1);
					let mut s = Self::add(&mut s,
						&mut Self::mul(&mut u2, dp2) );
					if round!=rounds-1{
						s = Self::sub(&mut s, 
							&mut Self::mul(du, &mut t)
						);
					}
					fd.vec_s[round][i] = s;
					fd.vec_t[round][i] = t;
					
				}
			}
			RUN_CONFIG.better_barrier("round");
			log_perf(LOG1, &format!(" --compute_st round: {}", round), &mut timer2);
		}
		log_perf(LOG1, &format!("compute_st"), &mut timer2);
	}


	/// decide at which level (counting 0, from bottom)
	/// to perform the distributed operations (lower than this level)
	/// should do individual operations only 
	fn decide_dis_level(total_size: usize, rounds: usize)-> usize{
		let unit_size = total_size/(1<<rounds);
		let mut size = unit_size;
		let mut log_nodes = rounds;
		let points = RUN_CONFIG.log_speed_up_points.clone();

		let mut res = rounds - 1;	
		for i in 0..rounds{
			if points.len() > log_nodes &&
				points[log_nodes] <= size{
				res = i;
				break;
			}
			size = size * 2;
			log_nodes = log_nodes/2;
		}

		return if res>0 && res<rounds {res} else {rounds-1};
	}

	/** new distributed version take advantage of the fact
		that one polynomial can be factored as
		co-primed list of factor polynomials.
		Assumption: a is a list of (not necessary) group elements. 
		dis_level is the level ID for performing distributed operations.
		THIS FUNCTION IS ABANDONED, in practice on small servers
it's 2-3 times slower than feea, it achieves on 128 nodes server the
same speed on gcp with feea() function. Not used in this project 
any more.
	*/
	pub fn feea_new(a: &mut DisVec<F>, b: &mut DisPoly<F>) -> 
		(DisPoly<F>, DisPoly<F>, DisPoly<F>){
		//1. set up
		let mut timer = Timer::new();
		timer.start();
		let np = RUN_CONFIG.n_proc;
		let k = (log2(np)+1) as usize;
		assert!(1<<(k-1) == np, "np is now a power of 2!");
		a.to_partitions(&RUN_CONFIG.univ);
		b.dvec.set_real_len();
		a.set_real_len();
		let max_size = if a.real_len>b.dvec.real_len {a.real_len}
			else {b.dvec.real_len};
		let dis_level = Self::decide_dis_level(max_size, k); 

		//2. call worker
		let res = Self::feea_new_worker(a, b, dis_level);
		RUN_CONFIG.better_barrier("wait for all");
		log_perf(LOG1, "feea_new", &mut timer);
		return res;
	}

	/** assumption: a is already into partitions, and real len has been
		set */
	pub fn feea_new_worker(a: &mut DisVec<F>, b: &mut DisPoly<F>, dis_level: usize) -> 
		(DisPoly<F>, DisPoly<F>, DisPoly<F>){
		//1. compute settings
		let np = RUN_CONFIG.n_proc;
		let k = (log2(np)+1) as usize;
		assert!(1<<(k-1) == np, "np is now a power of 2!");
		let mut fd = FeeaData::<F>::new(k);
		assert!(dis_level<k, "disbar: {} should be < k: {}",dis_level,k);
		log(LOG1, &format!("dis_level: {}", dis_level));

		//2. propagate bottom-up for accumulator (characteristic poly)
		//set up the vec_p1 in FeeaData
		Self::compute_binacc(a, k, &mut fd, dis_level);

		//3. propagate top-down for p2
		//set up vec_p2
		Self::compute_p2(b, k, &mut fd, dis_level);

		//4. bottom-up compute s and t again
		Self::compute_st(k, &mut fd, dis_level);

		//5. return 
		let s = &mut fd.vec_s[k-1][0];
		let t = &mut fd.vec_t[k-1][0];
		let p1 = &mut fd.vec_p1[k-1][0];
		let p2 = &mut fd.vec_p2[k-1][0];
		s.reset_main_processor(0);
		t.reset_main_processor(0);
		p1.reset_main_processor(0);
		p2.reset_main_processor(0);
		let gcd = Self::add(
			&mut Self::mul(s, p1),
			&mut Self::mul(t, p2)
		);
		return (gcd, s.clone(), t.clone());
	}	

}

#[cfg(test)]
mod tests {
	extern crate ark_ff;
	use crate::tools::*;
	//use crate::poly::dis_vec::*;
	use crate::poly::dis_poly::*;
	use self::ark_ff::UniformRand;
	//use ark_poly::{univariate::DensePolynomial};
	//use mpi::point_to_point as p2p;
	//use mpi::topology::Rank;
	//use mpi::traits::*;
	type Fr381=ark_bls12_381::Fr;
	//use crate::profiler::config::*;
	fn gen_dp(size: usize)->DisPoly<Fr381>{
		return DisPoly::<Fr381>::gen_dp(size);
	}

	fn test_from_roots_for_size(n: usize, main_processor: u64, log_2_round_size: usize, dist_bar: usize){
		test_from_roots_for_size_worker(n, main_processor, log_2_round_size, dist_bar, false);
	}

	fn test_from_roots_for_size_from_mainnode(n: usize, main_processor: u64, 
		log_2_round_size: usize, 
		dist_bar: usize){
		test_from_roots_for_size_worker(n, main_processor, log_2_round_size, dist_bar, true);
	}

	//call function for specific size
	fn test_from_roots_for_size_worker(n: usize, main_processor: u64, 
		log_2_round_size: usize, 
		dist_bar: usize, bmainnode: bool){
		let my_rank = RUN_CONFIG.my_rank;
		let fname = format!("/tmp/t_{}.dat", n);

		if my_rank==main_processor as usize{
			//1. generate a random u64 array and write to tmp file
			let mut arru64 = rand_arr_unique_u64(n, 20, get_time());
			let mut a1 = vec![arru64.len() as u64];
			let arr_f = arru64_to_arrft::<Fr381>(&arru64);
			a1.append(&mut arru64);
			write_arr(&a1, &fname);
			RUN_CONFIG.better_barrier("test_from_roots_for_size1");
			//println!("STEP1 : rank: {}, log2_round_size: {}", my_rank, log_2_round_size);
	
			//2. construct a serial from the roots
			let serial_p = DisPoly::<Fr381>::binacc_poly(&arr_f);
			let serial_vec = serial_p.coeffs;
			//println!("STEP2 : rank: {}", my_rank);

			//3. construct discrete one 
			let nodes_file = "/tmp/tmp_nodelist.txt";
			let netarch =  get_network_arch(&nodes_file.to_string());
			let mut d_p = if !bmainnode {DisPoly::<Fr381>::dispoly_from_roots_in_file_worker(101, main_processor, &fname, log_2_round_size, dist_bar)}
				else {DisPoly::<Fr381>::dispoly_from_roots_in_file_from_mainnode_worker(101, main_processor, &fname, log_2_round_size, dist_bar, &netarch)};
			//println!("STEP3 BEFORE coefs: rank: {}", my_rank);
			let d_vec = d_p.coefs();
			//println!("STEP3 AFTER: rank: {}", my_rank);

			//4. check
			let svec2 = serial_vec.clone();
			let dvec2 = d_vec.clone();
	        let ps = DensePolynomial::<Fr381>::from_coefficients_vec(serial_vec);
	        let pd = DensePolynomial::<Fr381>::from_coefficients_vec(d_vec);
			let b_eq = ps==pd;
			if !b_eq{
				dump_vec("-----serial_vec:----\n", &svec2);
				dump_vec("\n\n======dist_vec:=====\n", &dvec2);
			}
			//println!("STEP4 : rank: {}", my_rank);
			assert!(b_eq, "sequental results not equal to distributed!");
		}else{
			//3. CO-ordinate to build the distriuted vec. Check will be done
			//by main processor
			let nodes_file = "/tmp/tmp_nodelist.txt";
			let netarch =  get_network_arch(&nodes_file.to_string());
			RUN_CONFIG.better_barrier("test_dispoly_from_roots2");
			let mut _d_p = if !bmainnode {DisPoly::<Fr381>::dispoly_from_roots_in_file_worker(101, main_processor, &fname, log_2_round_size, dist_bar)}
				else {DisPoly::<Fr381>::dispoly_from_roots_in_file_from_mainnode_worker(101, main_processor, &fname, log_2_round_size, dist_bar, &netarch)};
			let _d_vec = _d_p.coefs();
		}
		RUN_CONFIG.better_barrier("test_dispoly_from_roots3");
		//println!("DONE!!! : rank: {}, log2_round_size: {}", my_rank, log_2_round_size);
	}

	// ==== UNIT TEST CASES BELOW =================
	#[test]
	fn slow_test_from_roots_for_all_stage1_1k_diff_groups(){
		let np = RUN_CONFIG.n_proc;
		let minsize = np*np*np*4;
		let bar = log2(minsize)-2;
		test_from_roots_for_size(minsize+1025, 1, 1, 1<<bar); 
		test_from_roots_for_size(minsize+2048, 0, 1, 1<<bar); 
		test_from_roots_for_size(minsize+2048, 0, 2, 1<<bar); 
		test_from_roots_for_size(minsize+1011, 0, 4, 1<<bar); 
	}

	#[test]
	fn quick_test_from_roots_from_main(){//in testscript: 8 nodes
		//8 elements, main_processor: 1, round_group_size: 2^1, bar:2^20
		let np = RUN_CONFIG.n_proc;
		let minsize = np*np *np*4;
		test_from_roots_for_size_from_mainnode(minsize, 1, 1, 1<<20); 
		//same, but switch main processor to 0
		test_from_roots_for_size_from_mainnode(minsize*2, 0, 1, 1<<20); 
		test_from_roots_for_size_from_mainnode(minsize*4, 0, 1, 1<<20); 
		test_from_roots_for_size_from_mainnode(minsize*8, 1, 1, 1<<20); 
	}

	#[test]
	fn quick_test_from_roots(){//in testscript: 8 nodes
		//8 elements, main_processor: 1, round_group_size: 2^1, bar:2^20
		let np = RUN_CONFIG.n_proc;
		let minsize = np*np *np*4;
		test_from_roots_for_size(minsize*1, 1, 1, 1<<20); 
		//same, but switch main processor to 0
/*
		test_from_roots_for_size(minsize*2, 0, 1, 1<<20); 
		test_from_roots_for_size(minsize*4, 0, 1, 1<<20); 
		test_from_roots_for_size(minsize*8, 1, 1, 1<<20); 
*/
	}

	#[test]
	fn quick_test_sizes(){
		let np = RUN_CONFIG.n_proc;
		let minsize = np*np*np*4;
		let bar = log2(minsize)-2;
		for size in 16..20{
			test_from_roots_for_size(minsize+size, 1, 1, 1<<20); 
		}
	}


	#[test]
	fn quick_test_dist3stage(){//8 nodes: eval has 3 stages, bar is LOW
		//1. all dist	
		let np = RUN_CONFIG.n_proc;
		let minsize = np*np*np*4;
		let bar = log2(minsize)-2;
		test_from_roots_for_size(minsize, 1, 1, 1<<0); 
	}

	#[test]
	fn test_from_roots_dist_simple(){//8 nodes: eval has 3 stages
		//1. all dist	
		let np = RUN_CONFIG.n_proc;
		let minsize = np*np*np*4;
		let bar = log2(minsize)-2;
		test_from_roots_for_size(minsize+128, 0, 1, 1<<0); 
		//2. 2 local 1 dist
		test_from_roots_for_size(minsize+128, 0, 1, 1<<5); 
		//3. 1 local 1 dist and another 1 dist
		test_from_roots_for_size(minsize+128, 0, 1, 1<<4); 
		//4. test number ranges
		for _i in 253..258{
			test_from_roots_for_size(minsize+128, 0, 1, 1<<4); 
		}
	}


	#[test]
	fn test_eval_chunks(){
		let dp = gen_dp(1010);
		let r = Fr381::from(3712312u64);
		let vec = dp.eval_chunks(&r);
		let res = dp.eval(&r);
		if RUN_CONFIG.my_rank==dp.dvec.main_processor{
			assert!(res==vec[vec.len()-1], "FAILED eval_chunks. expected: {}, actual: {}", res, vec[vec.len()-1]);
		}
	}

	#[test]
	fn test_eval_chunked_derivative(){
		let np = RUN_CONFIG.n_proc;
		let dp = gen_dp(1010);
		let sp1 = dp.to_serial();
		let r = Fr381::from(3712312u64);
		let res= dp.compute_chunked_derivative(&r);
		if RUN_CONFIG.my_rank==dp.dvec.main_processor{
			let dp1 = get_derivative(&sp1);
			let res2 = dp1.evaluate(&r);
			assert!(res2==res[np], "ERROR: compute_chunked_derivative: res2: {} != res[np]: {}", res2, res[np]);
		}
	}

	#[test]
	fn test_ops(){
		let np = RUN_CONFIG.n_proc;
		let d2 = np * 128 + 13;
		let d1 = (np-2) * 128 + 17;

		let target_degree = d1+109; //real degree
		let mut dp1= gen_dp(d1);
		let mut dp2= gen_dp(d2);		
		let sp1 = dp1.to_serial();
		let sp2 = dp2.to_serial();
		let k = d1/2;


		let dp3 = DisPoly::<Fr381>::sub(&mut dp1, &mut dp2);
		let dp3_a = DisPoly::<Fr381>::add(&mut dp1, &mut dp2);
		let dp3_m = DisPoly::<Fr381>::mul(&mut dp1, &mut dp2);
		let dp3_r = DisPoly::<Fr381>::rev(&dp1, target_degree);
		let dp3_mod = DisPoly::<Fr381>::mod_by_xk(&dp1, k);


		let sp3 = dp3.to_serial();
		let sp3_a = dp3_a.to_serial();
		let sp3_m = dp3_m.to_serial();
		let sp3_r = dp3_r.to_serial();
		let sp3_mod = dp3_mod.to_serial();

		let sp32 = &sp1 - &sp2;
		let sp32_a = &sp1 + &sp2;
		let sp32_m = &sp1 * &sp2;
		let sp32_r = rev(&sp1, target_degree);
		let sp32_mod = mod_by_xk(&sp1, k);


		if RUN_CONFIG.my_rank==dp3.dvec.main_processor{
			assert!(sp3==sp32, "ERROR: test_sub: res2: {:?} != expected {:?}", sp3, sp32);
			assert!(sp3_a==sp32_a, "ERROR: test_add: res2: {:?} != expected {:?}", sp3_a, sp32_m);
			assert!(sp3_m==sp32_m, "ERROR: test_mul: res2: {:?} != expected {:?}", sp3_m, sp32_m);
			assert!(sp3_r==sp32_r, "ERROR: test_rev: res2: {:?} != expected {:?}", sp3_r, sp32_r);
			assert!(sp3_mod==sp32_mod, "ERROR: test_mod: res2: {:?} != expected {:?}", sp3_mod, sp32_mod);
		}
	}

	#[test]
	fn test_mod_mul_by_xk(){
		let np = RUN_CONFIG.n_proc;
		let d1 = (np-2) * 128 + 17;
		let mut dp1 = gen_dp(d1);
		let arr_k = vec![0, 1, d1/2-1, d1/2, d1/2+1, d1-2, d1-1, d1, d1+1, d1+10, d1+101];
		for k in arr_k{
			let sp1 = dp1.to_serial();
			let dp3_mod = DisPoly::<Fr381>::mod_by_xk(&dp1, k);
			let dp3_mul= DisPoly::<Fr381>::mul_by_xk(&dp1, k);
			let dp3_div= DisPoly::<Fr381>::div_by_xk(&dp1, k);

			let sp3_mod = dp3_mod.to_serial();
			let sp3_mul= dp3_mul.to_serial();
			let sp3_div= dp3_div.to_serial();

			let sp32_mod = mod_by_xk(&sp1, k);
			let sp32_mul= mul_by_xk(&sp1, k);
			let sp32_div= div_by_xk(&sp1, k);

			if RUN_CONFIG.my_rank==dp3_mod.dvec.main_processor{
				assert!(sp3_mod==sp32_mod, "ERROR: test_mod: for case k: {}, res2: {:?} != expected {:?}", k, sp3_mod, sp32_mod);
				assert!(sp3_mul==sp32_mul, "ERROR: test_mul: for case k: {}, res2: {:?} != expected {:?}", k, sp3_mul, sp32_mul);
				assert!(sp3_div==sp32_div, "ERROR: test_mul: for case k: {}, res2: {:?} != expected {:?}", k, sp3_div, sp32_div);
			}
		}
	}

	#[test]
	fn test_m2xm1_m2xm2(){
		let np = RUN_CONFIG.n_proc;
		let d = np * 16;
		let seed = 192834u128;
		let zero = DisPoly::<Fr381>::one(0);
		
		let mut dm1 = [
			DisPoly::<Fr381>::gen_dp_from_seed(d, seed),
			DisPoly::<Fr381>::gen_dp_from_seed(d, seed+1),
			DisPoly::<Fr381>::gen_dp_from_seed(d, seed+2),
			DisPoly::<Fr381>::gen_dp_from_seed(d, seed+3)
		];
		let mut dm2 = [
			DisPoly::<Fr381>::gen_dp_from_seed(d, seed+5),
			DisPoly::<Fr381>::gen_dp_from_seed(d, seed+6)
		];
		let mut dm3 = [
			DisPoly::<Fr381>::gen_dp_from_seed(d, seed+7),
			DisPoly::<Fr381>::gen_dp_from_seed(d, seed+8),
			DisPoly::<Fr381>::gen_dp_from_seed(d, seed+9),
			DisPoly::<Fr381>::gen_dp_from_seed(d, seed+10)
		];
		let mut sm1 = [
			dm1[0].to_serial(),
			dm1[1].to_serial(),
			dm1[2].to_serial(),
			dm1[3].to_serial(),
		];
		let mut sm2 = [
			dm2[0].to_serial(),
			dm2[1].to_serial()
		];
		let mut sm3 = [
			dm3[0].to_serial(),
			dm3[1].to_serial(),
			dm3[2].to_serial(),
			dm3[3].to_serial(),
		];

		//test m2xm1
		let dres = DisPoly::<Fr381>::m2xm1(&mut dm1, &mut dm2);
		let sres2 = [dres[0].to_serial(), dres[1].to_serial()];
		let sres = m2xm1(&sm1, &sm2);
		if RUN_CONFIG.my_rank==dres[0].dvec.main_processor{
			assert!(sres2[0]==sres[0], "m2xm1 test failed on res0");
			assert!(sres2[1]==sres[1], "m2xm1 test failed on res1");
		}

		//test m2xm2
		let dres = DisPoly::<Fr381>::m2xm2(&mut dm1, &mut dm3);
		let sres2 = [dres[0].to_serial(), dres[1].to_serial(), dres[2].to_serial(), dres[3].to_serial()];
		let sres = m2xm2(&sm1, &sm3);
		if RUN_CONFIG.my_rank==dres[0].dvec.main_processor{
			assert!(sres2[0]==sres[0], "m2xm2 test failed on res0");
			assert!(sres2[1]==sres[1], "m2xm2 test failed on res1");
			assert!(sres2[2]==sres[2], "m2xm2 test failed on res2");
			assert!(sres2[3]==sres[3], "m2xm2 test failed on res3");
		}
	}

	#[test]
	fn test_hgcd_worker(){
		let np = RUN_CONFIG.n_proc;
		let d = np * 16;
		let seed = 192834u128;
		let zero = DisPoly::<Fr381>::one(0);
		let mut a = DisPoly::<Fr381>::gen_dp_from_seed(d, seed);
		let mut b = DisPoly::<Fr381>::gen_dp_from_seed(d, seed+10123);
		let mut sa = a.to_serial();
		let mut sb = b.to_serial();

		//check htcd_worker
		let res1 = DisPoly::<Fr381>::hgcd_worker(&mut a, &mut b, 0);
		let res2 = hgcd_worker(&sa, &sb);
		for i in 0..4{
			let res1_i = res1[i].to_serial();
			if RUN_CONFIG.my_rank==res1[0].dvec.main_processor{
				assert!(res1_i==res2[i], "hgcd_worker failed on res[{}]", i);
			}
		}

	}

	#[test]
	fn test_dis_feea(){
		let np = RUN_CONFIG.n_proc;
		//let d = np * 1024;
		let d = np * 32;
		let seed = 192834u128;
		let zero = DisPoly::<Fr381>::one(0);
		let mut a = DisPoly::<Fr381>::gen_dp_from_seed(d, seed);
		let mut b = DisPoly::<Fr381>::gen_dp_from_seed(d, seed+10123);
		let mut sa = a.to_serial_at_each_node(); //to avoid 0 degree
		let mut sb = b.to_serial_at_each_node();

		//check feea 
		let (s1, t1, g1) = DisPoly::<Fr381>::feea(&mut a, &mut b);
		let (s2, t2, g2) = feea(&sa, &sb);
		let ss1 = s1.to_serial();
		let st1 = t1.to_serial();
		let sg1 = g1.to_serial();
		if RUN_CONFIG.my_rank==a.dvec.main_processor{
			assert!(s2==ss1, "feea failed on s");
			assert!(t2==st1, "feea failed on t");
			assert!(g2==sg1, "feea failed on g");
			println!("Dis FEEA (gcd) passed!");
		}
	
		RUN_CONFIG.better_barrier("feea test");
	}
	#[test]
	fn test_dis_eval(){
		let np = RUN_CONFIG.n_proc;
		//let d = np * 1024;
		let d = np * 32;
		let seed = 192834u128;
  		let mut rng = gen_rng();
		let r = Fr381::rand(&mut rng); //tihs might be different for each node
		//the one uses the r at main node
		let mut dp= DisPoly::<Fr381>::gen_dp(d);
		let mut sp = dp.to_serial();
		let dres = dp.eval(&r);
		let sres = sp.evaluate(&r);
		if RUN_CONFIG.my_rank==dp.dvec.main_processor{
			assert!(dres==sres, "eval failed: dres: {}, sres: {}", dres, sres);
		}
		RUN_CONFIG.better_barrier("eval test");
	}
	#[test]
	fn test_derivative(){
		let np = RUN_CONFIG.n_proc;
		//let d = np * 1024;
		let d = np * 32;
		let seed = 192834u128;
  		let mut rng = gen_rng();
		//the one uses the r at main node
		let mut dp= DisPoly::<Fr381>::gen_dp(d);
		let dp1 = dp.get_derivative();
		let dp2 = dp.logical_get_derivative();
		let sp1 = dp1.to_serial();
		let sp2 = dp2.to_serial();
		if RUN_CONFIG.my_rank==dp1.dvec.main_processor{
			assert!(sp1==sp2, "get_derivative()");
		}
		RUN_CONFIG.better_barrier("derivative test");
	}

	#[test]
	fn test_chunked_derivative(){
		let np = RUN_CONFIG.n_proc;
		//let d = np * 1024;
		let d = np * 32;
		let seed = 192834u128;
  		let mut rng = gen_rng();
		let r = Fr381::rand(&mut rng);
		//the one uses the r at main node
		let mut dp= DisPoly::<Fr381>::gen_dp(d);
		let res1= dp.compute_chunked_derivative(&r);
		let res2= dp.logical_compute_chunked_derivative(&r);
		if RUN_CONFIG.my_rank==dp.dvec.main_processor{
			assert!(res1==res2, "get_chunked_derivative failed");
		}
		RUN_CONFIG.better_barrier("chunked_derivative");
	}
	#[test]
	fn test_inv(){
		let np = RUN_CONFIG.n_proc;
		let mut d1 = (np-2) * 128 + 17;
		let mut dp1 = gen_dp(d1);
		let arr_k = vec![0, 1, 2, 3, 4, 10, 12];
		//let arr_k = vec![10];
		for k in arr_k{
			let sp1 = dp1.to_serial();
			let dp3_inv= DisPoly::<Fr381>::inv(&mut dp1, k);

			let sp3_inv= dp3_inv.to_serial();

			let sp32_inv= if RUN_CONFIG.my_rank==dp1.dvec.main_processor {inv(&sp1, k)} else {DensePolynomial::<Fr381>::zero()}; //as sp1==0 will cause error in inv.

			if RUN_CONFIG.my_rank==dp1.dvec.main_processor{
				assert!(sp3_inv==sp32_inv, "ERROR: test_inv: for case k: {}, res2: {:?} != expected {:?}", k, sp3_inv, sp32_inv);
			}
		}
	}
	#[test]
	fn test_reset_main_proessor(){
		let np = RUN_CONFIG.n_proc;
		let d1 = np*np;
		let mut dp1= DisPoly::<Fr381>::gen_dp(d1);
		dp1.to_partitions();
		let r = Fr381::from(37u64);
		let s1 = dp1.to_serial();
		let v1 = s1.evaluate(&r);
		let new_main = np-1;
		dp1.reset_main_processor(new_main);
		let s2 = dp1.to_serial();	
		let v2 = s2.evaluate(&r);
		let vres1 = broadcast_small_arr(&vec![v1], 0);
		let vres2 = broadcast_small_arr(&vec![v2],  new_main);
		assert!(vres1[0]==vres2[0], "failed reset_main_prcessor");
	}
	//assuming dis_bar is in range
	fn test_feea_new_worker(size: usize, dis_bar: usize){
		//1. set up
		let np = RUN_CONFIG.n_proc;
		let k = (log2(np)+1) as usize;
		assert!(1<<(k-1) == np, "np is now a power of 2!");
		let seed = 12098321u128;
		let arr = rand_arr_field_ele::<Fr381>(size, seed);
		let mut a = DisVec::<Fr381>::from_serial(&arr); 
		let p = DisPoly::<Fr381>::binacc_poly(&arr);
		let mut dp1 = DisPoly::from_serial(0, &p, size+1);
		let mut b= DisPoly::<Fr381>::gen_dp(size);
		a.to_partitions(&RUN_CONFIG.univ);
		b.dvec.set_real_len();
		a.set_real_len();

		//2. call worker
		let (_gcd, mut s, mut t)= DisPoly::<Fr381>::feea_new_worker(&mut a, &mut b, dis_bar);
		let one = get_poly::<Fr381>(vec![Fr381::from(1u64)]);
		let res = DisPoly::<Fr381>::add(
			&mut DisPoly::<Fr381>::mul(&mut s, &mut dp1),
			&mut DisPoly::<Fr381>::mul(&mut t, &mut b)
		);
		let sres = res.to_serial();
		if RUN_CONFIG.my_rank==res.dvec.main_processor{
			assert!(sres==one, "failed DisGCD test for size: {}, level: {}", size, dis_bar);
		}
		RUN_CONFIG.better_barrier("wait check");
	}

	fn test_feea_new_wrapper(size: usize){
		let np = RUN_CONFIG.n_proc;
		let k = (log2(np)+1) as usize;
		for i in 1..k{
			test_feea_new_worker(size, i);
		}
	}

	fn test_feea_new(){ //THIS FUNCTION IS NOT USED 
		//small case
		let np = RUN_CONFIG.n_proc;
		let small_size = np * 4;
		test_feea_new_wrapper(small_size);

		//large case
		let large_size = np * 1024;
		test_feea_new_wrapper(large_size);
	}
	#[test]
	fn test_div(){
		let np = RUN_CONFIG.n_proc;
		//let d1 = np * 128 + 13;
		//let d2 = (np-2) * 128 + 17;
	//	let d1 = np*np*16;
	//	let d2 = np*np*3;
		let d1 = (1<<(RUN_CONFIG.log_div_bar+2));
		let d2 = d1/2;

		let mut dp1= DisPoly::<Fr381>::gen_dp(d1);
		let mut dp2= DisPoly::<Fr381>::gen_dp(d2);		
		let sp1 = dp1.to_serial();
		let sp2 = dp2.to_serial();
		let sp2_2 = sp2.clone();
		let sp1_2 = sp1.clone();

		let (sq, sr) = if RUN_CONFIG.my_rank==dp1.dvec.main_processor {new_divide_with_q_and_r::<Fr381>(&sp1, &sp2)} else {(sp1, sp2)};
		let sq_2 = sq.clone();
		let sr_2 = sr.clone();
		let (dq, dr) = DisPoly::<Fr381>::divide_with_q_and_r(&mut dp1, &mut dp2);
		let s_dq = dq.to_serial();
		let s_dr = dr.to_serial();
		let sp1_3 = &sq_2*&sp2_2 + sr_2;
		if RUN_CONFIG.my_rank==dp1.dvec.main_processor{
//			print_poly("sq:", &sq);
//			print_poly("sr:", &sr);
//			print_poly("s_dq:", &s_dq);
//			print_poly("s_dr:", &s_dr);
			assert!(sq==s_dq, "ERROR: sq != dq");
			assert!(sr==s_dr, "ERROR: sr != dr");
			assert!(sp1_3==sp1_2, "ERROR: sp1!= sp1_2");
		}
	}

}
