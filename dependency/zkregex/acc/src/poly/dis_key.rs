/** 
	Copyright Dr. Xiang Fu

	Author: Dr. Xiang Fu
	All Rights Reserved.
	Created: 06/02/2022
	Revised: 07/15/2022 (added G2 group)
	Revised: 07/22/2022 (added beta -scaling series)
	Distributed Power Series Object
	(seret alpha and power series g^{alpha^0}, g^{alpha^1}, ..., g^{alpha^n})
	also g^{beta alpha_0}, ..., g^{beta alpha^n})
	It has the same series for G2 group.
	
*/
extern crate ark_ff;
extern crate ark_ec;
extern crate ark_poly;
extern crate ark_std;
extern crate mpi;
extern crate ark_serialize;

extern crate ark_bls12_381;
use self::ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
use self::ark_ff::PrimeField;
use profiler::config::*;
use super::common::*;
use self::ark_ff::{Field, One};
use crate::tools::*;
use super::dis_vec::*;
use super::dis_poly::*;
use self::ark_poly::{DenseUVPolynomial,univariate::DensePolynomial};
use self::ark_ec::msm::{FixedBase, VariableBaseMSM};
use self::ark_ec::{AffineCurve, PairingEngine, ProjectiveCurve};
//use self::ark_ec::{AffineCurve, PairingEngine, ProjectiveCurve};
//use ark_ec::{group::Group, AffineCurve, PairingEngine, ProjectiveCurve};
//use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
//use ark_std::log2;
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

#[cfg(feature = "parallel")]
use self::ark_std::cmp::max;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

//use crate::profiler::config::*;
//use super::dis_vec::*;
//use super::disfft::*;

/** Distribted Key (a power series) of a secret group element
	Profiling: 1M -> 400MB for Bls12-381, peak mem usage when gen: 650MB.
*/
#[derive(Clone)]
pub struct DisKey<E: PairingEngine>{
	/** a random ID */
	pub id: u64, 
	/** the secret toxic alpha */
	pub _alpha: E::Fr,
	/** another secret toxic, for scaling (knowledge extract prf for KZG) */
	pub _beta: E::Fr,
	/** another toxic, mainly for v-sql scheme */
	pub _theta: E::Fr,
	/** another toxic, mainly for v_sql scheme */
	pub _s2: E::Fr,
	/** public. the base g */
	pub g: E::G1Projective,
	/** public. the other base h*/
	pub h: E::G1Affine,
	/** public. [log(g) * log(h)]_1 */
	pub gh: E::G1Affine,
	/*** same base g in G2 */
	pub g_g2: E::G2Affine,
	/*** same base h in G2 */
	pub h_g2: E::G2Affine,
	/*** g2^alpha */
	pub g_alpha_g2: E::G2Affine,
	/** h^beta */
	pub h_beta: E::G1Affine, 
	/** h^alpha */
	pub h_alpha: E::G1Affine, 
	/** g^theta over G2*/
	pub g_theta_g2: E::G2Affine, 
	/** g^beta over G2 */
	pub g_beta_g2: E::G2Affine, 
	/** g^theta */
	pub g_theta: E::G1Affine, 
	/** h^theta */
	pub h_theta: E::G1Affine, 
	/** h^beta over g2*/
	pub h_beta_g2: E::G2Affine,
	/** the size of the ENTIRE DisKey Object*/
	pub n: usize,
	/** the PARTITION: it's depending on the node ID and total processors */
	pub powers_g: Vec<E::G1Affine>,
	/** partitioned. series of g^{beta alpha^k} */
	pub powers_g_beta: Vec<E::G1Affine>,
	/** the same sequence over group G2 */
	pub powers_g2: Vec<E::G2Affine>,
	/** powers_g1[1] */
	pub g_alpha: E::G1Affine,
}

/** dump a collection of points */
pub fn dump_vec_pts<C: AffineCurve>(s: &str, v: &Vec<C>){
	println!("=========== {} ==============", s);
	for i in 0..v.len(){
		println!("v[{}]: {}", i, v[i]);
	}
	println!("\n");
}

/*** Generate a vector of powerw of the following form. 
	g^{beta * (alpha)^{0 + base}
	It is used to generate LOCAL segment of power series at each node.

	the use of base here is for partitioned reprsentation of KZG key power
series. For examle, let's say each partition of MPI stores 1024 items
in the key series, then for the first partition it stores
	g^{beta * (alpha) ^{0+base:0} ..... g^{beta * (alpha)^{1023+base:0}} 
The second partition would store:
	g^{(alpha) ^{0+base:1024} ..... g^{(alpha)^{1023+base:1024}} 
	where the beta can be set to alpha^{1024}
	then it is represented as 
	g^{beta * (alpha ^{0})}, .... beta * g^{alpha^1023}
	It return a Vec of E::G1Affine points of size n.
*/
pub fn gen_powers<E:PairingEngine>(g: E::G1Projective, alpha: E::Fr, beta: E::Fr, base: u64, n: usize) -> Vec<E::G1Affine>{
	let b_perf = false;
	let mut timer = Timer::new();
	timer.start();

 
	let alpha_base = alpha.pow(&[base]) * beta;
	let mut arr_alpha= vec![E::Fr::one(); n];
	arr_alpha[0] = alpha_base;
	let mut cur = alpha.clone();
	for i in 1..n{
		arr_alpha[i] = cur * alpha_base;
		cur *= &alpha;
	}
	if b_perf {log_perf(LOG1, "gen_powers step1: gen arr_alpha", &mut timer);}
	let window_size = FixedBase::get_mul_window_size(n+1);
	let scalar_bits = E::Fr::MODULUS_BIT_SIZE as usize;
	let g_table = FixedBase::get_window_table(scalar_bits, window_size, g);
	let powers_proj= FixedBase::msm::<E::G1Projective>(
            scalar_bits,
            window_size,
            &g_table,
            &arr_alpha,
	);
	if b_perf {log_perf(LOG1, "gen_powers step2: fixed msm", &mut timer);}

	let res = E::G1Projective::batch_normalization_into_affine(&powers_proj);
	if b_perf {log_perf(LOG1, "gen_powers step3: batch norm", &mut timer);}

	return res;
}

/// G2 version. This could be better refactored, but let it stay as it is.
pub fn gen_powers_g2<E:PairingEngine>(g: E::G2Projective, alpha: E::Fr, beta: E::Fr, base: u64, n: usize) -> Vec<E::G2Affine>{
	let alpha_base = alpha.pow(&[base]) * beta;
	let mut arr_alpha= vec![E::Fr::one(); n];
	arr_alpha[0] = alpha_base;
	let mut cur = alpha.clone();
	for i in 1..n{
		arr_alpha[i] = cur * alpha_base;
		cur *= &alpha;
	}
	let window_size = FixedBase::get_mul_window_size(n+1);
	let scalar_bits = E::Fr::MODULUS_BIT_SIZE as usize;
	let g_table = FixedBase::get_window_table(scalar_bits, window_size, g);
	let powers_proj= FixedBase::msm::<E::G2Projective>(
            scalar_bits,
            window_size,
            &g_table,
            &arr_alpha,
	);
	let res = E::G2Projective::batch_normalization_into_affine(&powers_proj);
	return res;
}

/** compute vbase[0]^vexp[0] * ... vbase[n]^vexp[n] */
pub fn multi_scalar_mul<G>(vbase: &Vec<G>, vexp: &Vec<G::ScalarField>) -> G where  G: AffineCurve, G::Projective: VariableBaseMSM<MSMBase = G, Scalar = G::ScalarField> {
		//from ark_kzg10 code
		let f = DensePolynomial::from_coefficients_vec(vexp.to_vec());
  		let (num_leading_zeros, plain_coeffs) =
            skip_leading_zeros_and_convert_to_bigints(&f);
        let res: _ = <G::Projective as VariableBaseMSM>::msm(
            &vbase[num_leading_zeros..],
            plain_coeffs.as_slice(),
        );
		let res = res.into_affine();
		return res;
}


/** for testing purpuprse. The same as gen_powers */
pub fn logical_gen_powers<E:PairingEngine>(g: E::G1Projective, alpha: E::Fr, beta: E::Fr, base: u64, n: usize) -> Vec<E::G1Affine>{
	let mut vres = vec![];
	for i in 0..n{
		let sumexp = (i as u64) + base;
		let exp_v = beta * alpha.pow(&[sumexp]);
		let item = g.into_affine().mul(exp_v);
		vres.push(item);
	}
	return E::G1Projective::batch_normalization_into_affine(&vres);
}

/** for testing purpose. All factors are u64. When making calls make sure
that all sequence generated are u64 in range.
base should be i32 actually.
*/
pub fn logical_gen_powers_u64<E:PairingEngine>(g: E::G1Projective, u_alpha: u64, u_beta: u64, base: u64, n: usize) -> Vec<E::G1Affine>{
	let mut vres = vec![];
	assert!(base< (1<<31), "base too large!>= 2^32");
	for i in 0..n{
		let sumexp = (i as u32) + (base as u32);
		let (exp1, b1)  = u_alpha.overflowing_pow(sumexp);
		assert!(!b1, "pow overflowed!");
		let (exp_v, b2) = u_beta.overflowing_mul(exp1);
		assert!(!b2, "mul exp overflowed!");
		let item = g.into_affine().mul(exp_v);
		vres.push(item);
	}
	return E::G1Projective::batch_normalization_into_affine(&vres);
}


impl<E> DisKey<E> where E: PairingEngine{

	/** generate a generator */
	pub fn gen_g(seed: u64) -> E::G1Projective{
		let gen = E::G1Affine::prime_subgroup_generator();
		let r_factor = E::Fr::from(seed);
		let g = gen.mul(r_factor);
		return g;
	}
	pub fn gen_g2(seed: u64) -> E::G2Affine{
		let gen = E::G2Affine::prime_subgroup_generator();
		let r_factor = E::Fr::from(seed);
		let g = gen.mul(r_factor);
		return g.into_affine();
	}

	/** generate a key, alternative constructor */
	pub fn gen_key(id: u64, n: usize, g_seed: u64, alpha_seed: u64)->DisKey<E>{
		//THIS FUNCTION can be improved later using a hash function
		let g = DisKey::<E>::gen_g(g_seed); 
		let u_s2 = g_seed + 17827;
		let theta = E::Fr::from(g_seed + 772342);
		let s2 = E::Fr::from(u_s2);
		let h = g.into_affine().mul(s2);
		let g_g2 = DisKey::<E>::gen_g2(g_seed); 
		let h_g2 = g_g2.mul(s2).into_affine();
		let alpha = E::Fr::from(alpha_seed);
		let beta = E::Fr::from(alpha_seed+29234823421u64);
		let key = DisKey::<E>::gen(g, id, n, alpha, g_g2, beta, h, h_g2, theta, s2);
		return key;
	}

	/** convenience function. Generate a fixed key serie */
	pub fn gen_key1(n: usize) -> DisKey<E>{
		return DisKey::<E>::gen_key(0, n, 371239u64, 22319234u64);
	}

	/** convenience function */
	pub fn gen_key2(n: usize) -> DisKey<E>{
		return DisKey::<E>::gen_key(0, n, 7127u64, 11992843u64);
	}


	/// CONSTRUCTOR. 
	/// generate a DisKey Object. Need to be called at
	/// each node. Each node generate its own segment of data separately.
	/// NOTE: g and g2 requires to satisfy e(g, g2) = e(g2, g)
	/// same for h, h2, i.e., their logairithm regarding the default generator
	/// is the same for G1 and G2.
	pub fn gen(g: E::G1Projective, id: u64, n: usize, alpha: E::Fr, g2: E::G2Affine, beta: E::Fr, h: E::G1Projective, h_g2: E::G2Affine, theta: E::Fr, s_2: E::Fr) -> DisKey<E>{
		//0. set up
		let me = RUN_CONFIG.my_rank;
		let np = RUN_CONFIG.n_proc;
		let b_perf = false;
		let b_mem = false; 
		let mut timer = Timer::new();
		timer.start();

		//1. generate my own partition
		if b_mem {dump_mem_usage("BEFORE_genkey");}
		let (start, end) = get_share_start_end(me as u64, np as u64, n as u64); 
		let beta_one = E::Fr::from(1u64);
		let base = start;
		let part_len = end-start;	

		//log(LOG1, &format!("Diskey::gen size: {}, part_len: {}", n, part_len));

		let arr_gs_g2 = gen_powers_g2::<E>(g2.into_projective(), alpha, beta_one, base as u64, part_len);
		if b_perf {log_perf(LOG1, "PERF_USE_gen_gs_g2", &mut timer);}
		if b_mem {dump_mem_usage("AFTER gen_gs_g2");}

		let arr_gs = gen_powers::<E>(g, alpha,beta_one,base as u64, part_len);
		if b_perf {log_perf(LOG1, "PERF_USE_gen_gs", &mut timer);}
		if b_mem {dump_mem_usage("AFTER gen_gs");}

		let arr_gs_beta = gen_powers::<E>(g,alpha,beta,base as u64, part_len);
		if b_perf {log_perf(LOG1, "PERF_USE_gen_gs_beta", &mut timer);}
		if b_mem {dump_mem_usage("AFTER gen_gs_beta");}

		let h_beta = h.into_affine().mul(beta).into_affine();
		let h_alpha= h.into_affine().mul(alpha).into_affine();
		let g_theta_g2= g2.mul(theta).into_affine();
		let g_beta_g2= g2.mul(beta).into_affine();
		let g_theta= g.into_affine().mul(theta).into_affine();
		let h_theta= h.into_affine().mul(theta).into_affine();
		let h_beta_g2 = h_g2.mul(beta).into_affine();
		if b_perf {log_perf(LOG1, "PERF_USE_gen_others", &mut timer);}
		if b_mem {dump_mem_usage("AFTER gen_others");}

		//2. return the object
		return DisKey{
			id: id,
			g: g,
			g_alpha: g.into_affine().mul(alpha).into_affine(),
			h: h.into_affine(),
			gh: h.into_affine().mul(alpha).into_affine(),
			h_g2: h_g2,
			g_g2: g2,
			g_alpha_g2: g2.mul(alpha).into_affine(),
			_alpha: alpha,
			_beta: beta,
			_theta: theta,
			_s2: s_2,
			g_theta_g2: g_theta_g2,
			g_beta_g2: g_beta_g2,
			g_theta: g_theta,
			h_theta: h_theta,
			h_beta: h_beta,
			h_alpha: h_alpha,
			h_beta_g2: h_beta_g2,
			n: n,
			powers_g: arr_gs,
			powers_g_beta: arr_gs_beta,
			powers_g2: arr_gs_g2,
		};
	}

	/** dump itself */
	pub fn dump(&self){
		let s = format!("Node: {} of {}", 
			RUN_CONFIG.my_rank, RUN_CONFIG.n_proc);
		dump_vec_pts(&s, &self.powers_g);
	}

	/// For all nodes>0: return empty (and send its own copy) to
	/// node 0; node 0 will return the full copy of the entire key series.
	///	SLOW! For testing purpose ONLY! 
	pub fn from_partitions(&self) -> Vec<E::G1Affine>   where <E as PairingEngine>::G1Affine: CanonicalSerialize, <E as PairingEngine>::G1Affine: CanonicalDeserialize{
		let my_rank = RUN_CONFIG.my_rank;
		let np = RUN_CONFIG.n_proc;
		let world = RUN_CONFIG.univ.world();
		let mut vec:Vec<E::G1Affine> = vec![];
		if my_rank>0{//send
			let vbytes = to_vecu8(&self.powers_g);	
			world.process_at_rank(0).send_with_tag(&vbytes, 
				my_rank as i32);
		}else{//receive
			let g_affine = self.g.into_affine();
			vec = vec![g_affine; self.n]; 
			for _i in 0..np-1{
				let r1 = world.any_process().receive_vec::<u8>();
				let src_rank = r1.1.tag() as i32;
				let (start,end)=get_share_start_end(
					src_rank as u64, np as u64, self.n as u64);
				let v = from_vecu8(&r1.0, g_affine);
				for j in start..end{
					vec[j] = v[j-start];
				}
			}
			//COPY over my own stuff
			let (start, _end) = get_share_start_end(
				my_rank as u64, np as u64, self.n as u64);
			for k in 0..self.powers_g.len(){
				vec[k+start] = self.powers_g[k];
			}		
		}
		RUN_CONFIG.better_barrier("from_partitions");
		return vec;
	}

	pub fn sub_key(&self, new_n: usize, newid: u64) -> DisKey<E>
where <E as PairingEngine>::Fr: CanonicalSerialize + CanonicalDeserialize + Clone, <E as PairingEngine>::G1Affine: CanonicalSerialize, <E as PairingEngine>::G1Affine: CanonicalDeserialize{
		return self.sub_key_new(new_n, newid);
	}

	/// basically to simulate dis_vec repartition
	fn repartition_vec<G: CanonicalSerialize+CanonicalDeserialize+Clone+Copy>(old_vec: &Vec<G>, cur_len: usize, new_n: usize) -> Vec<G>{
		//0. set up
		let my_rank = RUN_CONFIG.my_rank;
		let np = RUN_CONFIG.n_proc;
		let b_perf = false;
		let b_mem = false;
		let mut timer = Timer::new();
		timer.start();

		if b_mem {dump_mem_usage("repartition_vec start");}
		//2. build the vetor to send
		let vrecv;
		{//block to release vsend
			let mut vsend: Vec<Vec<G>> = vec![vec![]; np];
			for i in 0..np{
				let (start_offset, end_offset, _, _) = 
					gen_rescale_plan(my_rank, i, np, cur_len, new_n);
				let row = old_vec[start_offset..end_offset].to_vec();
				vsend[i] = row;
			}
			if b_mem {dump_mem_usage("AFTER build row data");}
			if b_perf {log_perf(LOG1, &format!("AFTER build row: n: {}, new_n: {}", cur_len, new_n), &mut timer);}
	
			//3. broadcast
			let sample = old_vec[0].clone();
			vrecv = nonblock_broadcast(&vsend, np as u64, &RUN_CONFIG.univ, 
				sample);
			if b_mem {dump_mem_usage("=== AFTER non-braodcast");}
			if b_perf {log_perf(LOG1, &format!("AFTER nonbroadcast n: {}, new_n: {}", cur_len, new_n), &mut timer);}
		}
		if b_mem {dump_mem_usage("=== OUT of block");}

		//3. reassemble
		let (my_start, my_end) = get_share_start_end(my_rank as u64, np as u64, new_n as u64); 
		let my_part_len = my_end - my_start;
		let g_affine = old_vec[0].clone();
		let mut vecret = vec![g_affine; my_part_len];
		for i in 0..np{
			let (_, _, s, e) = gen_rescale_plan(i,my_rank,np,cur_len,new_n);
			if e>s{//valid entry
				let my_s = s - my_start;
				let my_e = e - my_start;
				for k in 0..my_e-my_s{
					vecret[my_s+k] = vrecv[i][k]; 
				}
			}
		}
		return vecret;
	}
	
	//NEW version: smaller emmory consumption
	pub fn sub_key_new(&self, new_n: usize, newid: u64) -> DisKey<E>
where <E as PairingEngine>::Fr: CanonicalSerialize + CanonicalDeserialize + Clone, <E as PairingEngine>::G1Affine: CanonicalSerialize, <E as PairingEngine>::G1Affine: CanonicalDeserialize{
		let cur_len = self.n;
		let b_perf = false;
		let b_mem = false;
		let mut timer = Timer::new();
		timer.start();
		//let world = &(RUN_CONFIG.univ.world());
		if b_mem {dump_mem_usage("sub_key_NEW starts");}

		let g2_new = Self::repartition_vec(&self.powers_g2, cur_len, new_n);
		if b_mem {dump_mem_usage("AFTER send-receive g_2");}
		if b_perf {log_perf(LOG1, &format!("AFTER send-receive g_2: {}, new_n: {}", self.n, new_n), &mut timer);}
		

		let g_new = Self::repartition_vec(&self.powers_g, cur_len, new_n); 
		if b_mem {dump_mem_usage("AFTER send-receive g");}
		if b_perf {log_perf(LOG1, &format!("AFTER send-receive g n: {}, new_n: {}", self.n, new_n), &mut timer);}

		let gbeta_new=Self::repartition_vec(&self.powers_g_beta,cur_len,new_n);
		if b_mem {dump_mem_usage("AFTER send-receive g_beta");}
		if b_perf {log_perf(LOG1, &format!("AFTER send-receive g_beta n: {}, new_n: {}", self.n, new_n), &mut timer);}

		//6. set the new target_len
		let newkey = DisKey{
			id: newid,
			g: self.g.clone(),
			g_alpha: self.g_alpha.clone(),
			h: self.h.clone(),
			gh: self.gh.clone(),
			h_g2: self.h_g2.clone(),
			g_g2: self.g_g2.clone(),
			_s2: self._s2.clone(),
			_theta: self._theta.clone(),
			g_theta_g2: self.g_theta_g2.clone(),
			g_theta: self.g_theta.clone(),
			h_theta: self.h_theta.clone(),
			g_alpha_g2: self.g_alpha_g2.clone(),
			g_beta_g2: self.g_beta_g2.clone(),
			_alpha: self._alpha.clone(),
			_beta: self._beta.clone(),
			h_beta: self.h_beta.clone(),
			h_alpha: self.h_alpha.clone(),
			h_beta_g2: self.h_beta_g2.clone(),
			n: new_n,
			powers_g: g_new,
			powers_g_beta: gbeta_new,
			powers_g2: g2_new 
		};
		RUN_CONFIG.better_barrier("subkey2");
		return newkey;
	}

	/// generate a subkey with smaller size. 
	/// Basically it's to repartition  (DEPRECATED - too much mem usage)
	pub fn sub_key_old(&self, new_n: usize, newid: u64) -> DisKey<E>
where <E as PairingEngine>::Fr: CanonicalSerialize + CanonicalDeserialize + Clone, <E as PairingEngine>::G1Affine: CanonicalSerialize, <E as PairingEngine>::G1Affine: CanonicalDeserialize{
		let my_rank = RUN_CONFIG.my_rank;
		let np = RUN_CONFIG.n_proc;
		let cur_len = self.n;
		let b_perf = true;
		let b_mem = true;
		let mut timer = Timer::new();
		timer.start();
		//let world = &(RUN_CONFIG.univ.world());

		//1. send over data
		if b_mem {dump_mem_usage("BEFORE build row data");}
		let mut vsend: Vec<Vec<E::G1Affine>> = vec![vec![]; np];
		let mut vsend_beta: Vec<Vec<E::G1Affine>> = vec![vec![]; np];
		let mut vsend_g2: Vec<Vec<E::G2Affine>> = vec![vec![]; np];
		for i in 0..np{
			let (start_offset, end_offset, _, _) = 
				gen_rescale_plan(my_rank, i, np, cur_len, new_n);
			let row = self.powers_g[start_offset..end_offset].to_vec();
			let row_g2 = self.powers_g2[start_offset..end_offset].to_vec();
			let row_beta = self.powers_g_beta[start_offset..end_offset].to_vec();
			vsend[i] = row;
			vsend_beta[i] = row_beta;
			vsend_g2[i] = row_g2;
		}

		//2. broadcast and receive g series
		if b_mem {dump_mem_usage("AFTER build row data");}
		if b_perf {log_perf(LOG1, &format!("AFTER build row: n: {}, new_n: {}", self.n, new_n), &mut timer);}
		let sample = self.g.into_affine();
		let vrecv = nonblock_broadcast(&vsend, np as u64, &RUN_CONFIG.univ, sample);
		if b_mem {dump_mem_usage("=== AFTER non-braodcast");}
		if b_perf {log_perf(LOG1, &format!("AFTER nonbroadcast n: {}, new_n: {}", self.n, new_n), &mut timer);}
		let (my_start, my_end) = get_share_start_end(my_rank as u64, np as u64, new_n as u64); 
		let my_part_len = my_end - my_start;
		let g_affine = self.g.into_affine();
		let mut vecret = vec![g_affine; my_part_len];
		for i in 0..np{
			let (_, _, s, e) = gen_rescale_plan(
				i, my_rank, np, cur_len, new_n);
			if e>s{//valid entry
				let my_s = s - my_start;
				let my_e = e - my_start;
				for k in 0..my_e-my_s{
					vecret[my_s+k] = vrecv[i][k]; 
				}
			}
		}
		RUN_CONFIG.better_barrier("sub_key");
		if b_mem {dump_mem_usage("AFTER send-receive g");}
		if b_perf {log_perf(LOG1, &format!("AFTER send-receive g n: {}, new_n: {}", self.n, new_n), &mut timer);}

		//3. broadcast and receive g_beta series
		let sample = self.g.into_affine();
		let vrecv = nonblock_broadcast(&vsend_beta, np as u64, &RUN_CONFIG.univ, sample);
		let (my_start, my_end) = get_share_start_end(my_rank as u64, np as u64, new_n as u64); 
		let my_part_len = my_end - my_start;
		let g_affine = self.g.into_affine();
		let mut vecret_beta = vec![g_affine; my_part_len];
		for i in 0..np{
			let (_, _, s, e) = gen_rescale_plan(
				i, my_rank, np, cur_len, new_n);
			if e>s{//valid entry
				let my_s = s - my_start;
				let my_e = e - my_start;
				for k in 0..my_e-my_s{
					vecret_beta[my_s+k] = vrecv[i][k]; 
				}
			}
		}
		RUN_CONFIG.better_barrier("sub_key");
		if b_mem {dump_mem_usage("AFTER send-receive g_beta");}
		if b_perf {log_perf(LOG1, &format!("AFTER send-receive g_beta n: {}, new_n: {}", self.n, new_n), &mut timer);}

		//4. broadcast and receive G2 elements
		// this could be better rafactored - we'll stay with code at this moment
		let sample = self.g_g2;
		let vrecv = nonblock_broadcast(&vsend_g2, np as u64, &RUN_CONFIG.univ, sample);

		//5. assemble data for G2 elements
		let (my_start, my_end) = get_share_start_end(my_rank as u64, np as u64, new_n as u64); 
		let my_part_len = my_end - my_start;
		let g_affine = self.g_g2;
		let mut vecret_g2 = vec![g_affine; my_part_len];
		for i in 0..np{
			let (_, _, s, e) = gen_rescale_plan(
				i, my_rank, np, cur_len, new_n);
			if e>s{//valid entry
				let my_s = s - my_start;
				let my_e = e - my_start;
				for k in 0..my_e-my_s{
					vecret_g2[my_s+k] = vrecv[i][k]; 
				}
			}
		}
		if b_mem {dump_mem_usage("AFTER send-receive g_2");}
		if b_perf {log_perf(LOG1, &format!("AFTER send-receive g_2n: {}, new_n: {}", self.n, new_n), &mut timer);}
		
		//6. set the new target_len
		let newkey = DisKey{
			id: newid,
			g: self.g.clone(),
			g_alpha: self.g_alpha.clone(),
			h: self.h.clone(),
			gh: self.gh.clone(),
			h_g2: self.h_g2.clone(),
			g_g2: self.g_g2.clone(),
			_s2: self._s2.clone(),
			_theta: self._theta.clone(),
			g_theta_g2: self.g_theta_g2.clone(),
			g_theta: self.g_theta.clone(),
			h_theta: self.h_theta.clone(),
			g_alpha_g2: self.g_alpha_g2.clone(),
			g_beta_g2: self.g_beta_g2.clone(),
			_alpha: self._alpha.clone(),
			_beta: self._beta.clone(),
			h_beta: self.h_beta.clone(),
			h_alpha: self.h_alpha.clone(),
			h_beta_g2: self.h_beta_g2.clone(),
			n: new_n,
			powers_g: vecret,
			powers_g_beta: vecret_beta,
			powers_g2: vecret_g2
		};
		RUN_CONFIG.better_barrier("subkey2");
		return newkey;
	}

	/** evaluate a DisPoly. the DisPoly should be lower than its length
		generate: g^P(\alpha)
	 */
	pub fn eval_poly(&self, p: &mut DisVec<E::Fr>, b_use_beta: bool)-> E::G1Affine 
 where <E as PairingEngine>::Fr: CanonicalDeserialize+CanonicalSerialize,
<E as PairingEngine>::G1Affine: CanonicalSerialize+CanonicalDeserialize,
    <<E as PairingEngine>::G1Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=E::G1Affine, Scalar=<<E  as PairingEngine>::G1Affine as AffineCurve>::ScalarField>{
		let b_perf = false;
		let b_mem = false;

		let mut timer = Timer::new();
		timer.start();
		let old_len = p.len;
		p.set_real_len();
		let newn = p.real_len;
		if newn<old_len{
			//log(LOG1, &format!("DEBUG USE 888: eval_poly: resize poly from {} -> {}", old_len, newn));
			p.repartition(newn);
		}
		assert!(newn<=self.n, "polynomial degree: {} > key.len: {}!", newn, self.n);
		if newn==self.n{//perfect match
			let res = self.eval_poly_worker(p, b_use_beta);
			if b_perf {log_perf(LOG1,"--  after eval",&mut timer);}
			if b_mem {dump_mem_usage("--  after eval");}
			return res;
		}else if newn<self.n/2 { //worthwile to sub-key as it's expensive
			let key = self.sub_key(newn, self.id+1);
			if b_perf {log_perf(LOG1, "--  subkey", &mut timer);}
			if b_mem {dump_mem_usage("subkey");}
			let res = key.eval_poly_worker(p, b_use_beta);
			if b_perf {log_perf(LOG1,"--  after eval",&mut timer);}
			if b_mem {dump_mem_usage("--  after eval");}
			return res;
		}else{//rescale p instead
			let old_pn = p.len;
			p.repartition(self.n);
			if b_perf {log_perf(LOG1,"--  after repart p",&mut timer);}
			if b_mem {dump_mem_usage("--  after repart p");}
			let res = self.eval_poly_worker(p, b_use_beta);
			if b_perf {log_perf(LOG1,"--  after eval",&mut timer);}
			if b_mem {dump_mem_usage("--  after eval");}
			p.repartition(old_pn);
			if b_perf {log_perf(LOG1,"--  after repar_back p",&mut timer);}
			if b_mem {dump_mem_usage("--  after repar_back p");}
			return res;
		}
	}

	//eval the polynomial over group g2. This could be refactored better later.
	pub fn eval_poly_g2(&self, p: &mut DisVec<E::Fr>)-> E::G2Affine 
 where <E as PairingEngine>::Fr: CanonicalDeserialize+CanonicalSerialize,
<E as PairingEngine>::G2Affine: CanonicalSerialize+CanonicalDeserialize,
    <<E as PairingEngine>::G2Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=E::G2Affine, Scalar=<<E  as PairingEngine>::G2Affine as AffineCurve>::ScalarField>{
		let b_perf = false;
		let b_mem = false;
		let mut timer = Timer::new();
		timer.start();
		let newn = p.len;
		assert!(newn<=self.n, "polynomial degree:{} > key.len: {}!", newn, self.n);
		if newn==self.n{//perfect match
			let res = self.eval_poly_worker_g2(p);
			if b_perf {log_perf(LOG1,"--  after eval",&mut timer);}
			if b_mem {dump_mem_usage("--  after eval");}
			return res;
		}else if newn<self.n/2 { //worthwile to sub-key as it's expensive
			let key = self.sub_key(newn, self.id+1);
			if b_perf {log_perf(LOG1, "--  subkey", &mut timer);}
			if b_mem {dump_mem_usage("subkey");}
			let res = key.eval_poly_worker_g2(p);
			if b_perf {log_perf(LOG1,"--  after eval",&mut timer);}
			if b_mem {dump_mem_usage("--  after eval");}
			return res;
		}else{//rescale p instead
			let old_pn = p.len;
			p.repartition(self.n);
			if b_perf {log_perf(LOG1,"--  after repart p",&mut timer);}
			if b_mem {dump_mem_usage("--  after repart p");}
			let res = self.eval_poly_worker_g2(p);
			if b_perf {log_perf(LOG1,"--  after eval",&mut timer);}
			if b_mem {dump_mem_usage("--  after eval");}
			p.repartition(old_pn);
			if b_perf {log_perf(LOG1,"--  after repar_back p",&mut timer);}
			if b_mem {dump_mem_usage("--  after repar_back p");}
			return res;
		}
	}


	/** generate the KZG commitment of polynomial. Needs the polynomial
		mut access as if not in cluster, will run to_partitions of it.
		Return a Vec of 1 element
	*/
	pub fn gen_kzg(&self, p: &mut DisPoly<E::Fr>)-> Vec<E::G1Affine>  
 where <E as PairingEngine>::Fr: CanonicalDeserialize+CanonicalSerialize,
<E as PairingEngine>::G1Affine: CanonicalSerialize+CanonicalDeserialize,
    <<E as PairingEngine>::G1Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=E::G1Affine, Scalar=<<E  as PairingEngine>::G1Affine as AffineCurve>::ScalarField>{
		if !p.dvec.b_in_cluster{
			let me = RUN_CONFIG.my_rank;
			panic!("me: {}: poly needs to be partitioned first before gen_kzg!", me);
		}

		let ret = vec![self.eval_poly(&mut p.dvec, false)];
		return ret;
	}

	/* use the beta series (scaled)*/
	pub fn gen_kzg_beta(&self, p: &mut DisPoly<E::Fr>)-> Vec<E::G1Affine>  
 where <E as PairingEngine>::Fr: CanonicalDeserialize+CanonicalSerialize,
<E as PairingEngine>::G1Affine: CanonicalSerialize+CanonicalDeserialize,
    <<E as PairingEngine>::G1Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=E::G1Affine, Scalar=<<E  as PairingEngine>::G1Affine as AffineCurve>::ScalarField>{
		if !p.dvec.b_in_cluster{
			panic!("poly needs to be partitioned first before gen_kzg!");
		}
		let ret = vec![self.eval_poly(&mut p.dvec, true)];
		return ret;
	}

	//version on g2. This could be refactored better
	pub fn gen_kzg_g2(&self, p: &mut DisPoly<E::Fr>)-> Vec<E::G2Affine>  
 where <E as PairingEngine>::Fr: CanonicalDeserialize+CanonicalSerialize,
<E as PairingEngine>::G2Affine: CanonicalSerialize+CanonicalDeserialize,
    <<E as PairingEngine>::G2Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=E::G2Affine, Scalar=<<E  as PairingEngine>::G2Affine as AffineCurve>::ScalarField>{
		if !p.dvec.b_in_cluster{
			panic!("poly needs to be partitioned first before gen_kzg!");
		}
		let ret = vec![self.eval_poly_g2(&mut p.dvec)];
		return ret;
	}



	/** assume degree are the same, we might partition p to distributed
	if it's not incluster. NOTE: only node 0 returns the valid result.
	b_use_beta indicates if to use the powers_g_beta serie */
	pub fn eval_poly_worker(&self, p: &DisVec<E::Fr>, b_use_beta: bool)-> E::G1Affine 
		where <E as PairingEngine>::Fr: CanonicalDeserialize+CanonicalSerialize,
<E as PairingEngine>::G1Affine: CanonicalSerialize+CanonicalDeserialize,
	<<E as PairingEngine>::G1Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=E::G1Affine, Scalar=<<E  as PairingEngine>::G1Affine as AffineCurve>::ScalarField>
	{
		let b_perf = false;
		let mut timer = Timer::new();
		timer.start();
		let n= p.len;
		assert!(n==self.n, "eval_poly_worker expects p has the same degree!");
		assert!(p.b_in_cluster, "p must be distributed!");
		let my_rank = RUN_CONFIG.my_rank;
		let np = RUN_CONFIG.n_proc;
		let world = RUN_CONFIG.univ.world();
		//if !p.dvec.b_in_cluster{ p.dvec.to_partitions(&RUN_CONFIG.univ);}

		//1. perform the following at all nodes
		assert!(self.powers_g.len()==p.partition.len(), 
			"pwoers_g.len != dvec.parition.len");
		if b_perf {log_perf(LOG1, &format!("DisKey::gen_kzg setup n: {}", n), &mut timer);}
		
		let local_res:E::G1Affine = if !b_use_beta {multi_scalar_mul(&self.powers_g, &p.partition)} else {multi_scalar_mul(&self.powers_g_beta, &p.partition)};
		//let msg = format!("DEBUG USE 100. my_rank: {}, partition:", my_rank);
		//dump_vec(&msg, &p.partition);
		//println!("DEBUG USE 101: eval_poly my_rank: {} step 2 --- result: {}", my_rank, &local_res);
		if b_perf {log_perf(LOG1, &format!("DisKey::gen_kzg msm n: {}, part_len: {}", n, &p.partition.len()), &mut timer);}

		//2. send to node 0
		let vdata = vec![local_res];
		let vbytes = to_vecu8(&vdata);
		world.process_at_rank(0).send_with_tag(&vbytes, my_rank as i32);

		//3. node 0 collect
		let g1_affine = self.g.into_affine();
		let mut res: E::G1Affine = g1_affine.clone();
		if my_rank==0{
			for i in 0..np{
				let r1 = world.any_process().receive_vec::<u8>();
				let v = from_vecu8(&r1.0, g1_affine);
				let _src= r1.1.source_rank();
				let pt:E::G1Affine = v[0];
				if i==0{ res = pt;}
				else{ res = res + pt;}
				//println!("DEBUG USE 200: received from {}: pt: {}, afer res+pt -> {}", _src, &pt, &res);
			}
		}
		RUN_CONFIG.better_barrier("eval_poly_worker");
		if b_perf {log_perf(LOG1, &format!("DisKey::gen_kzg assemble result: np: {}", np), &mut timer);}
		return res;
	}

	//eval_poly_worker on G2. This could be refactored better
	pub fn eval_poly_worker_g2(&self, p: &DisVec<E::Fr>)-> E::G2Affine 
		where <E as PairingEngine>::Fr: CanonicalDeserialize+CanonicalSerialize,
<E as PairingEngine>::G2Affine: CanonicalSerialize+CanonicalDeserialize,
	<<E as PairingEngine>::G2Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=E::G2Affine, Scalar=<<E  as PairingEngine>::G2Affine as AffineCurve>::ScalarField>
	{
		let b_perf = false;
		let mut timer = Timer::new();
		timer.start();
		let n= p.len;
		assert!(n==self.n, "eval_poly_worker expects p has the same degree!");
		assert!(p.b_in_cluster, "p must be distributed!");
		let my_rank = RUN_CONFIG.my_rank;
		let np = RUN_CONFIG.n_proc;
		let world = RUN_CONFIG.univ.world();
		//if !p.dvec.b_in_cluster{ p.dvec.to_partitions(&RUN_CONFIG.univ);}
		if b_perf {log_perf(LOG1, &format!("DisKey::gen_kzg_g2 setup n: {}", n), &mut timer);}

		//1. perform the following at all nodes
		assert!(self.powers_g.len()==p.partition.len(), 
			"pwoers_g.len != dvec.parition.len");
		let local_res:E::G2Affine = multi_scalar_mul(&self.powers_g2, &p.partition);
		//let msg = format!("DEBUG USE 100. my_rank: {}, partition:", my_rank);
		//dump_vec(&msg, &p.partition);
		//println!("DEBUG USE 101: eval_poly my_rank: {} step 2 --- result: {}", my_rank, &local_res);
		if b_perf {log_perf(LOG1, &format!("DisKey::gen_kzg_g2 msm n: {}, part_len: {}", n, &p.partition.len()), &mut timer);}

		//2. send to node 0
		let vdata = vec![local_res];
		let vbytes = to_vecu8(&vdata);
		world.process_at_rank(0).send_with_tag(&vbytes, my_rank as i32);

		//3. node 0 collect
		let g1_affine = self.g_g2;
		let mut res: E::G2Affine = g1_affine.clone();
		if my_rank==0{
			for i in 0..np{
				let r1 = world.any_process().receive_vec::<u8>();
				let v = from_vecu8(&r1.0, g1_affine);
				let _src= r1.1.source_rank();
				let pt:E::G2Affine = v[0];
				if i==0{ res = pt;}
				else{ res = res + pt;}
				//println!("DEBUG USE 200: received from {}: pt: {}, afer res+pt -> {}", _src, &pt, &res);
			}
		}
		RUN_CONFIG.better_barrier("eval_poly_worker_g2");
		if b_perf {log_perf(LOG1, &format!("DisKey::gen_kzg_g2 assumeble result: np: {}", np), &mut timer);}
		return res;
	}

	/** LOGICALLY evaluate the polynomial. return g^p(alpha).
		SLOW. for testing only.
	 */
	pub fn logical_eval_poly(&self, p: &Vec<E::Fr>) -> E:: G1Affine{
		//1. evalute the polynomial itself
		let mut exp = E::Fr::from(0u64);
		let mut item = E::Fr::from(1u64);
		for i in 0..p.len(){
			let monomial = item * p[i]; 
			exp = exp + monomial;
			item = item * self._alpha;
		}

		//2. raise it to power 
		let g_affine = self.g.into_affine();
		let res = g_affine.mul(exp).into_affine();
		return res;
	} 
}

#[cfg(test)]
mod tests {
	use crate::poly::dis_key::*;
	//use crate::poly::dis_vec::*;
	//use crate::profiler::config::*;
	use self::ark_bls12_381::Bls12_381;
	use self::ark_ec::{AffineCurve, PairingEngine};
	use self::ark_ff::UniformRand;
	//use crate::tools::*;
	//use ark_poly::{univariate::DensePolynomial};
	//use mpi::point_to_point as p2p;
	//use mpi::topology::Rank;
	//use mpi::traits::*;
	type Fr=ark_bls12_381::Fr;
	type PE=Bls12_381;
	type G1Affine=<Bls12_381 as PairingEngine>::G1Affine; 
	type G1Projective=<Bls12_381 as PairingEngine>::G1Projective; 
	type G2Affine=<Bls12_381 as PairingEngine>::G2Affine; 
	type G2Projective=<Bls12_381 as PairingEngine>::G2Affine; 

	pub fn gen_g(seed: u64) -> G1Projective{
		let gen = G1Affine::prime_subgroup_generator();
		let r_factor = Fr::from(seed);
		let g = gen.mul(r_factor);
		return g;
	}

	pub fn gen_g2(seed: u64) -> G2Affine{
		let gen = G2Affine::prime_subgroup_generator();
		let r_factor = Fr::from(seed);
		let g = gen.mul(r_factor);
		return g.into_affine();
	}


	/** assert if two vecs are the same */
	fn assert_eq_vec(s: &str, v1: &Vec<G1Affine>, v2: &Vec<G1Affine>){
		for i in 0..v1.len(){
			if v1[i] != v2[i]{
				println!("========== DUMP assert_eq_vec at {} =====", i);
				println!("v1[{}]: {}", i, &v1[i]);
				println!("v2[{}]: {}", i, &v2[i]);
				assert!(false, "FAILED on {}", s);
			}
		}
	}


	/** assert SMALL test cases for gen_powers */	
	fn test_gen_powers_u64(g: G1Projective, u_alpha: u64, u_beta: u64, base: u64, n: usize){
		if RUN_CONFIG.my_rank!=0 {return;}
		let alpha = Fr::from(u_alpha);
		let beta = Fr::from(u_beta);
		let v1 = gen_powers::<PE>(g, alpha, beta, base, n);
		let v2 = logical_gen_powers::<PE>(g, alpha, beta, base, n);
		let v3 = logical_gen_powers_u64::<PE>(g, u_alpha, u_beta, base, n);
		assert_eq_vec("quick_test_gen_pows: failed v2==v3", &v2, &v3);  
		assert_eq_vec("quick_test_gen_pows: failed v1==v2", &v1, &v2);  
	}

/* RECOVER LATER
	#[test]
	fn quick_test_gen_power_u64(){
		let g = gen_g(13572); 
		//alpha: 2, beta: 1, base: 0, n: 4
		test_gen_powers_u64(g, 2, 1, 0, 4);
		//alpha: 2, beta: 3, base 4, n: 4
		test_gen_powers_u64(g, 2, 3, 4, 4);
		//alpha: 3, beta: 2, base 5, n: 4
		test_gen_powers_u64(g, 3, 2, 5, 8);
	}

	#[test]
	fn rand_test_gen_power(){
		if RUN_CONFIG.my_rank!=0 {return;}
		let g = gen_g(13572); 
		let mut rng = gen_rng();
		let times = 1;
		let alpha = Fr::rand(&mut rng);	
		let beta = Fr::rand(&mut rng);	
		let mut base = 1024;
		let n = 128;
		for _i in 0..times{
			let v1 = gen_powers::<PE>(g, alpha, beta, base, n);
			let v2 = logical_gen_powers::<PE>(g, alpha, beta, base, n);
			base += 1;
			assert_eq_vec("random_test_gen_pow: failed v1==v2", &v1, &v2);  
		}
	}

	#[test]
	fn quick_debug_gen_g(){
		let g = gen_g(13572); 
		let h = gen_g(77123); 
		let h_g2 = gen_g2(77123); 
		let g2 = gen_g2(13572); 
		let alpha = Fr::from(2u64);
		let beta = Fr::from(372342123u64);
		let theta = Fr::from(223112389u64);
		let s2 = Fr::from(234209823u64);
		
		let key = DisKey::<PE>::gen(g, 0, 9, alpha, g2, beta, h, h_g2, theta, s2);
		//key.dump();
		let v1 = key.from_partitions();
		
		//manually build
		if RUN_CONFIG.my_rank==0{
			let exps:Vec<u64> = vec![1, 2, 4, 8, 16, 32, 64, 128, 256];
			let mut v2:Vec<G1Affine> = vec![];	
			for i in 0..9{
				let exp = exps[i];
				let item = g.into_affine().mul(exp).into_affine();
				v2.push(item);
			}
			//dump_vec_pts("EXPECTED", &v2);
			assert_eq_vec("quick_dump_gen_g", &v1, &v2);
			//println!("DEBUG USE 1000!!!: ");
		}
	}

	#[test]
	fn quick_test_subkey(){
		let g = gen_g(13572); 
		let g2 = gen_g2(13572); 
		let alpha = Fr::from(2u64);
		let beta = Fr::from(32234u64);
		let h = gen_g(332123);
		let h_g2 = gen_g2(332123);
		let id = 11;
		let len = 3502;
		let theta = Fr::from(223112389u64);
		let s2 = Fr::from(234209823u64);
		let key = DisKey::<PE>::gen(g, id, len, alpha, g2, beta, h, h_g2, theta, s2);
		let v1 = key.from_partitions();
		let my_rank = RUN_CONFIG.my_rank;
	
		let arrn = vec![len, len-2, len-3, len/2, len/2-1, len/4, 5];	
		for new_n in arrn{
			let key2 = key.sub_key(new_n, id+1);
			let v2 = key2.from_partitions();
			//check on main thread
			if my_rank==0{
				let v1_1 = v1[0..new_n].to_vec();
				assert_eq_vec("quick_test_subkey", &v1_1, &v2);
			}
		}
	}
*/
	fn test_eval_for_size(k: &DisKey::<PE>, size: usize){
		let vec_f = rand_arr_field_ele::<Fr>(size, 17234098234209384u128);
		let vf2 = vec_f.clone();
		let mut p = DisVec::new_dis_vec_with_id(100, 1, size, vec_f);
		p.to_partitions(&RUN_CONFIG.univ);
		let res1 = k.eval_poly(&mut p, false);	
		if RUN_CONFIG.my_rank==0{
			let res2 = k.logical_eval_poly(&vf2);
			assert!(res1==res2, "fail eval_for_size");
		}
	}

	#[test]
	fn quick_test_eval(){
		let np = RUN_CONFIG.n_proc;
		let g = gen_g(13572); 
		let h = gen_g(228272); 
		let h_g2 = gen_g2(228272); 
		let g2 = gen_g2(13572); 
		let alpha = Fr::from(2u64);
		let beta = Fr::from(271u64);
		let id = 11;
		let len = 121*np + 7;
		let theta = Fr::from(223112389u64);
		let s2 = Fr::from(234209823u64);
		let key = DisKey::<PE>::gen(g, id, len, alpha, g2, beta, h, h_g2, theta, s2);

		test_eval_for_size(&key, 5);
		test_eval_for_size(&key, len/2-1);
		test_eval_for_size(&key, len/2+1);
		test_eval_for_size(&key, len/2+10);
		test_eval_for_size(&key, len-2);
		test_eval_for_size(&key, len-1);
		
	}
}
