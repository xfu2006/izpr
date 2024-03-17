/*
    Copyright Dr. Xiang Fu

    Author: Xiang Fu
    All Rights Reserved.
    Created: 08/10/2023
	Completed (without fold prove): 08/16/2023
	Modified: 08/21/2023 (to add fold prove).
	Completed: 08/22/2023

	This files encodes the zero knowledge version of the cq protocol.
	See appendix H.4 of our paper IZPR.
*/
extern crate ark_ff;
extern crate ark_ec;
extern crate ark_poly;
extern crate ark_serialize;
extern crate ark_std;
//extern crate mpi;
extern crate ark_bls12_381;
extern crate acc;
extern crate rayon;
extern crate itertools;

use std::sync::Arc;
use std::collections::HashMap;
use self::itertools::Itertools;
use self::rayon::prelude::*; 
use self::ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Read, SerializationError, Write};
use self::ark_ec::{AffineCurve, PairingEngine,ProjectiveCurve};
use self::ark_ec::msm::{VariableBaseMSM};
use self::ark_ff::{UniformRand,FftField,Field,One,Zero};
use self::acc::tools::{gen_rng_from_seed, to_vecu8, hash};
use self::acc::tools::{log_perf,Timer};
use self::acc::poly::serial::{get_poly, serial_ifft, inplace_fft_coset, inplace_ifft_coset, old_divide_with_q_and_r};
use self::acc::profiler::config::{LOG1};
use self::ark_poly::{Polynomial};
use izpr::serial_poly_utils::{KzgProverKey,KzgVerifierKey, precompute_quotient_poly, vmsm,compute_powers};
use izpr::serial_pn_lookup::{MyArc};
use izpr::serial_logical_poly_utils::*;

#[derive(Clone)]
pub struct  MyHashMap<PE>{
	pub map: HashMap<PE, usize>
}


/// Aux Information for CQ
//#[derive(Clone)]
#[derive(Clone,CanonicalSerialize,CanonicalDeserialize)]
pub struct CqAux<PE:PairingEngine>{
	/// must be power of 2
	pub n: usize,
	/// commitment to t: Pedersen commitment to T(X) = \sum t_i L_i(X), over G2
	pub commit_t2: PE::G2Affine,
	/// r_t: the random nonce used in commit_t2
	pub r_t: PE::Fr,
	/// t_s_1: [T(s)]_1
	pub t_s_1: PE::G1Affine,
	/// array of quotient polynomials
	pub arr_q: Vec<PE::G1Affine>,
	/// loc_idx which maps each element to its idx
	pub map_idx: MyHashMap<PE::Fr>
}

// Implement Serialization for `HashMap<T>`
impl<T: CanonicalSerialize> CanonicalSerialize for MyHashMap<T> {
    #[inline]
    fn serialize<W: Write>(&self, mut writer: W) -> Result<(), SerializationError> {
		let entries = self.map.keys().len();
		entries.serialize(&mut writer).unwrap();
		for (key, value) in &self.map{
			key.serialize(&mut writer).unwrap();
			value.serialize(&mut writer).unwrap();
		}
		Ok(())
    }

    #[inline]
    fn serialized_size(&self) -> usize {
		let mut size = 0;
		for (key, value) in &self.map{
			size += key.serialized_size();
			size += value.serialized_size();
		}
		size
    }

    #[inline]
    fn serialize_uncompressed<W: Write>(&self, mut writer: W) -> Result<(), SerializationError> {
        self.serialize(&mut writer)
    }

    #[inline]
    fn uncompressed_size(&self) -> usize {
		self.serialized_size()
    }

    #[inline]
    fn serialize_unchecked<W: Write>(&self, mut writer: W) -> Result<(), SerializationError> {
        self.serialize(&mut writer)
    }
}

impl<T: FftField> CanonicalDeserialize for MyHashMap<T> {
    #[inline]
    fn deserialize<R: Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_unchecked(reader)
    }

    #[inline]
    fn deserialize_uncompressed<R: Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_unchecked(reader)
    }

    #[inline]
    fn deserialize_unchecked<R: Read>(mut reader: R) -> Result<Self, SerializationError> {
		let entries = usize::deserialize(&mut reader).unwrap();
		let mut map = HashMap::<T,usize>::new();
		for _i in 0..entries{
			let key = T::deserialize(&mut reader).unwrap();
			let value = usize::deserialize(&mut reader).unwrap();	
			map.insert(key, value);
		}
		let mymap = MyHashMap{map:map};
		Ok(mymap)
    }
}

/// Proof for CQ
#[derive(Clone,CanonicalSerialize,CanonicalDeserialize)]
pub struct CqProof<PE:PairingEngine>{
	/// Pedersen vector commitment to vector m
	pub commit_m: PE::G1Affine,
	/// Pedersen vector commitment to vector A
	pub commit_a: PE::G1Affine,
	/// Pedersen vector commitment to Q_A
	pub commit_qa: PE::G1Affine,
	/// Pedersen vector commit to B
	pub commit_b: PE::G1Affine,
	/// Pedersen vector commitment to P(X) = B(X) X^{N-n}
	pub commit_p: PE::G1Affine,
	/// Pedersen vector commitment to Q_B
	pub commit_q_b: PE::G1Affine,
	/// eval of b(X) at gamma
	pub b_gamma: PE::Fr,
	/// eval of t(X) at gamma
	pub t_gamma: PE::Fr,
	/// eval of q_b(X) at gamma
	pub q_b_gamma: PE::Fr,
	/// [h(s)]_1
	pub prf_gamma: PE::G1Affine,
	/// [A(0)] + r_a_0 [zv(s)]_1
	pub commit_a0: PE::G1Affine,
	/// [q_a0(s)]_1 + r_q_a0[zv(s)]_1
	pub commit_q_a0: PE::G1Affine,
	/// balance term for a0
	pub bal_a0: PE::G1Affine,
	/// DDLOG random 1st msag for qa_0
	pub msg1_q_a0: PE::G1Affine,
	/// response 1 for qa_0
	pub res1_q_a0: PE::Fr,
	/// response 2 for qa_0
	pub res2_q_a0: PE::Fr,
	/// [B(0)] + r_b_0 [zh(s)]_1
	pub commit_b0: PE::G1Affine,
	/// [q_b0(s)]_1 + r_q_b0[zh(s)]_1
	pub commit_q_b0: PE::G1Affine,
	/// balance term for b0
	pub bal_b0: PE::G1Affine,
	/// DDLOG random 1st msag for qb_0
	pub msg1_q_b0: PE::G1Affine,
	/// response 1 for qb_0
	pub res1_q_b0: PE::Fr,
	/// response 2 for qb_0
	pub res2_q_b0: PE::Fr,
	/// msg1 for the DDLOG proof for diff_a0_b0
	pub msg1_diff_a0_b0: PE::G1Affine,
	/// response 1 for DDLOG proof for commit_a0/commit_b0^{N/n}
	pub res1_diff_a0_b0: PE::Fr,
	/// response 2 for DDLOG proof for commit_a0/commit_b0^{N/n}
	pub res2_diff_a0_b0: PE::Fr,
}

/// return a seed based on current time
pub fn gen_seed() -> u128{
	return 123098123123123u128; //to be improved later
}

/// compute the Pedersen commitment for t 
/// Define T(X) = \sum_{i=1}^n t_i L_i(X)
/// Commit_t = [T(s)] + r [z_V(s)]
/// vec_lag are the vector of [L_i(s)]
/// zvs is [z_V(s)] where z_V(X) is the vanishing polys of all roots of unity
/// n is a power of 2
/// the function returns (random_opening r, commit_t, [t(s)])
pub fn ped_commit<G:AffineCurve>(
	t: &Vec<G::ScalarField>, 
	n: usize, 
	vec_lag: &Vec<G>,
	zvs: G)->(G::ScalarField, G, G)
where <G as AffineCurve>::Projective: VariableBaseMSM<MSMBase=G, Scalar=<G as AffineCurve>::ScalarField>{
	//1. generate random opening r
	assert!(n.is_power_of_two(), "n is not power of 2");
	assert!(n==vec_lag.len(), "n!=vec_lag.len()");
	assert!(n>=t.len(), "n<t.len()");
	let mut rng = gen_rng_from_seed(gen_seed());
	let r = G::ScalarField::rand(&mut rng);
	let comp2 = zvs.mul(r).into_affine();

	//2. compute the variable multiscalar mul
	let comp1 = vmsm(vec_lag, t); 
	(r, comp1 + comp2, comp1)
}

/// supply the random opening
/// return (commitment, commitment_without_random)
pub fn ped_commit_with_random<G:AffineCurve>(
	t: &Vec<G::ScalarField>, 
	r: G::ScalarField,
	n: usize, 
	vec_lag: &Vec<G>,
	zvs: G)->(G, G)
where <G as AffineCurve>::Projective: VariableBaseMSM<MSMBase=G, Scalar=<G as AffineCurve>::ScalarField>{
	//1. generate random opening r
	assert!(n.is_power_of_two(), "n is not power of 2");
	assert!(n==vec_lag.len(), "n!=vec_lag.len()");
	assert!(n>=t.len(), "n<t.len()");
	let comp2 = zvs.mul(r).into_affine();

	//2. compute the variable multiscalar mul
	let comp1 = vmsm(vec_lag, t); 
	(comp1 + comp2, comp1)
}

/// this deals with that the case is sparse
/// idx_arr indicates which elements of vec_lag to take	
/// this is to compute \sum_{i=1}^{idx_arr.len()} vec_lag[idx_arr[i]]^t[i]
/// Return (random_nonce, commit_t = [t(s)] + r*zvs, [t(s)])
pub fn sparse_ped_commit<G:AffineCurve>(
	t: &Vec<G::ScalarField>, 
	idx_arr: &Vec<usize>,
	vec_lag: &Vec<G>,
	zvs: G)->(G::ScalarField, G, G)
where <G as AffineCurve>::Projective: VariableBaseMSM<MSMBase=G, Scalar=<G as AffineCurve>::ScalarField>{
	//1. generate random opening r
	let n = t.len();
	assert!(n==idx_arr.len(), "idx_arr.len != t.len");
	assert!(n<=vec_lag.len(), "n>vec_lag.len()");
	let mut rng = gen_rng_from_seed(gen_seed());
	let r = G::ScalarField::rand(&mut rng);
	let comp2 = zvs.mul(r).into_affine();

	//2. collect the vector of group elements and exponents
	let arr_g = idx_arr.par_iter().map(|idx| vec_lag[*idx]).collect::<Vec<_>>();
	assert!(arr_g.len()==n, "arr_g.len()!=n");

	//3. compute the variable multiscalar mul
	let comp1 = vmsm(&arr_g, t); 
	(r, comp1 + comp2, comp1)
}

/// return sum_{i=0}^n vec[i]*bases[idx_arr[i]]
pub fn sparse_inner_prod<F:FftField>(
	idx_arr: &Vec<usize>,
	vec: &Vec<F>,
	bases: &Vec<F>) -> F{
	let n = idx_arr.len();
	assert!(n==vec.len(), "n!=vec.len");
	assert!(n<=bases.len(), "n>bases.len");
	let arr_bases = idx_arr.par_iter().map(|idx| bases[*idx]).
		collect::<Vec<_>>();

	let res1 = vec.par_iter().zip(arr_bases.par_iter())
		.map(|(v1, v2)| (*v1) * (*v2)).sum();

	res1
}

/// generate the preprocessed information
/// given the lookup table, preprocess it, generate the aux information
/// to speed up proof, and the commitment to table.
/// NOTE THAT: compared with the original scheme, the
/// [L_i(s)] and [(L_i(s) - L_i(0)/s] are moved to KzgProverKey
/// because they are orthoganal to T.
pub fn preprocess_cq<PE:PairingEngine>(
	pk: &KzgProverKey<PE>, 
	lookup_table: &Vec<PE::Fr>)-> CqAux<PE> where
<<PE as PairingEngine>::G1Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G1Affine, Scalar=<<PE as PairingEngine>::G1Affine as AffineCurve>::ScalarField>,
 <<PE as PairingEngine>::G2Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G2Affine, Scalar=<<PE as PairingEngine>::G2Affine as AffineCurve>::ScalarField> {
	//1. compute the quotient polynomials
	let b_perf = false;
	let mut timer = Timer::new();
	if b_perf{println!("\n** preprocess cq");}
	let n = pk.n;
	let arr_q = precompute_quotient_poly(lookup_table, n, &pk.vec_g1);
	if b_perf{log_perf(LOG1, "-- quotient poly", &mut timer);}
	let map_idx = lookup_table.par_iter().enumerate()
		.map(|(idx, v)| (*v, idx))
		.collect::<HashMap<PE::Fr,usize>>();

	//2. compute the Commit_T = [T(s)]_1 + [r*z_V(s)]_1
	let (r_t, commit_t2, _) = ped_commit(lookup_table, n, 
		&pk.lag_all_2, pk.zv_s2);
	let (_, _, t_s1) = ped_commit(lookup_table, n, 
		&pk.lag_all, pk.zv_s1);
	CqAux{n: n, commit_t2: commit_t2, r_t: r_t, arr_q: arr_q,
		map_idx: MyHashMap{map:map_idx}, t_s_1: t_s1}
}

/// generate a proof that query_table is contained in lookup table
/// returns the proof that query_table \subset lookup_table
/// and the random opening for generarating and commit_query_table
/// i.e., (proof, r_query_table, commit_query_table)
pub fn prove_cq<PE:PairingEngine>(
	pk: &KzgProverKey<PE>, 
	aux: &CqAux<PE>, 
	lookup_table: &Vec<PE::Fr>,
	query_table: &Vec<PE::Fr>) 
	-> (CqProof<PE>, PE::Fr, PE::G1Affine) where
<<PE as PairingEngine>::G1Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G1Affine, Scalar=<<PE as PairingEngine>::G1Affine as AffineCurve>::ScalarField>,
 <<PE as PairingEngine>::G2Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G2Affine, Scalar=<<PE as PairingEngine>::G2Affine as AffineCurve>::ScalarField>{
	//1. check input
	let b_perf = false;
	let mut timer = Timer::new();
	let n2 = query_table.len();
	assert!(n2.is_power_of_two(), "query table size is not power of 2");

	//2. compute commit_query, commit_m
	let (r_query, commit_query, _) = ped_commit::<PE::G1Affine>(
		&query_table, n2, &pk.lag_all_n2, pk.zv_s1_n2);
	if b_perf {log_perf(LOG1, "-- commit_query_table ", &mut timer);}
	let (idx_m, vec_m) = gen_m(&aux.map_idx.map, &query_table);

	if b_perf {log_perf(LOG1, "-- gen_m", &mut timer);}
	let (r_m, commit_m, _) = sparse_ped_commit(&vec_m, &idx_m, 
		&pk.lag_all, pk.zv_s1);
	if b_perf {log_perf(LOG1, "-- commit_m v2", &mut timer);}


	//3. compute beta
	let beta = hash::<PE::Fr>(&to_vecu8(&vec![commit_query, commit_m]));


	//4. compute A_i and commit_a
	let idx_a = idx_m.clone();
	let one = PE::Fr::one();
	let vec_a = vec_m.par_iter().zip(idx_m.par_iter())
		.map(|(m,idx)| *m/(lookup_table[*idx] + beta)).
		collect::<Vec<PE::Fr>>();
	let (r_a, commit_a, a_s) = sparse_ped_commit(&vec_a, &idx_a, 
		&pk.lag_all, pk.zv_s1);
	if b_perf {log_perf(LOG1, "-- commit_a", &mut timer);}


	//5. compute commit_qa
	let g1 = PE::G1Affine::prime_subgroup_generator();
	let (_, _, qa_s) = sparse_ped_commit(&vec_a, &idx_a, &aux.arr_q, pk.zv_s1);
	let commit_qa = qa_s + aux.t_s_1.mul(r_a).into_affine() 
		+ a_s.mul(aux.r_t).into_affine() 
		+ pk.zv_s1.mul(aux.r_t * r_a).into_affine()
		+ g1.mul(r_a * beta - r_m).into_affine();
	if b_perf {log_perf(LOG1, "-- commit_qa", &mut timer);}

	//6. comptue b, p and qb
	let vec_b = query_table.par_iter().map(|t_i| one/(*t_i + beta)).
		collect::<Vec<PE::Fr>>();
	let (r_b, commit_b, _) = ped_commit(&vec_b, n2, &pk.lag_all_n2, pk.zv_s1_n2); 
	if b_perf {log_perf(LOG1, "-- commit_b", &mut timer);}

	let (coef_q_b, coef_b, coef_t) = compute_coeff_q_b(&vec_b,r_b,query_table,
		r_query,beta,n2);
	
	if b_perf {log_perf(LOG1, "-- compute q_b", &mut timer);}

	let commit_q_b = vmsm(&pk.vec_g1, &coef_q_b);
	let mut coef_b2 =  coef_b.clone()[0..pk.n2+1].to_vec(); //degree n, can cap
	let mut coef_p = vec![PE::Fr::zero(); pk.n-pk.n2-1];
	coef_p.append(&mut coef_b2);
	let commit_p = vmsm(&pk.vec_g1, &coef_p);
	if b_perf {log_perf(LOG1, "-- commit_q_b and commit_p", &mut timer);}

	//7. compute gamma
	let mut v1 = to_vecu8(&vec![commit_a, commit_qa, commit_b, 
		commit_q_b, commit_p]);
	let mut v2 = to_vecu8(&vec![beta]);
	v1.append(&mut v2);
	let gamma = hash::<PE::Fr>(&v1);

	//8. compute b_gamma, t_gamma, q_b_gamma
	let b_gamma = eval_coefs_at(&coef_b, gamma); 
	let t_gamma = eval_coefs_at(&coef_t, gamma);
	let z_h_gamma = gamma.pow(&[n2 as u64]) - one;
	let q_b_gamma = (b_gamma*(t_gamma + beta) - one)/z_h_gamma;

	//9 compute eta
	let eta = hash::<PE::Fr>(&to_vecu8(
		&vec![gamma, b_gamma, t_gamma, q_b_gamma]));
	let v = b_gamma + eta * t_gamma  + eta * eta * q_b_gamma;


	//10. compute h(X)
	let b_x = get_poly(coef_b);
	let t_x = get_poly(coef_t);
	let q_b_x = get_poly(coef_q_b);
	let eta_x = get_poly(vec![eta]);
	let eta_x2 = get_poly(vec![eta*eta]);
	let v_x = get_poly(vec![PE::Fr::zero() - v]);
	let part1 = b_x + (&t_x * &eta_x) + (&eta_x2 * &q_b_x) + v_x;
	let part2 = get_poly(vec![PE::Fr::zero() - gamma, PE::Fr::one()]);
	let (h_x, remainder_x) = old_divide_with_q_and_r(&part1, &part2);
	assert!(remainder_x.is_zero(), "remainder is not zero");
	if b_perf {log_perf(LOG1, "-- compute h(X)", &mut timer);}

	//11. compute prf gamma
	let coefs_h = h_x.coeffs;
	let prf_gamma = vmsm(&pk.vec_g1, &coefs_h);
	if b_perf {log_perf(LOG1, "-- compute prf_gamma: [h(s)]_1", &mut timer);}

	//12. compute prf_q_a0 (including a DDLOG proof)
	let mut rng = gen_rng_from_seed(gen_seed());
	let (r_q_a0, commit_q_a0, _) = sparse_ped_commit(&vec_a, &idx_a, 
		&pk.lag_diff, pk.zv_s1);
	let r_a0 = PE::Fr::rand(&mut rng);
	let a0 = sparse_inner_prod(&idx_a, &vec_a, &pk.lag0_fe);

	let commit_a0 = g1.mul(a0).into_affine()+pk.zv_s1.mul(r_a0).into_affine();	
	let bal_a0 = pk.zv_s1.mul(r_a - r_a0).into_affine() +
		pk.zv_s_s1.mul(PE::Fr::zero()-r_q_a0).into_affine(); //bal term for a0

	let ddlog_rand_a0 = PE::Fr::rand(&mut rng);
	let ddlog_rand_a1 = PE::Fr::rand(&mut rng);
	let msg1_q_a0 = pk.zv_s1.mul(ddlog_rand_a0).into_affine() + 
		pk.zv_s_s1.mul(ddlog_rand_a1).into_affine();
	//verifier challenge
	let ver_ch_a0 = hash::<PE::Fr>(&to_vecu8(&vec![bal_a0, msg1_q_a0])); 
	let res1_q_a0 = ddlog_rand_a0 - ver_ch_a0*(r_a-r_a0);
	let res2_q_a0 = ddlog_rand_a1 - ver_ch_a0*(PE::Fr::zero() - r_q_a0);

	//13. compute prf_q_b0 (including a DDLOG proof)
	let coef_b = evals_to_coefs(&vec_b, PE::Fr::one());
	let b0 = coef_b[0];
	let coef_q_b0 = &coef_b[1..coef_b.len()].to_vec();
	let r_b0 = PE::Fr::rand(&mut rng);
	let commit_b0 = g1.mul(b0).into_affine()+
		pk.zv_s1_n2.mul(r_b0).into_affine();
	let r_q_b0 = PE::Fr::rand(&mut rng);
	let commit_q_b0 = vmsm(&pk.vec_g1, &coef_q_b0) + 
		pk.zv_s1_n2.mul(r_q_b0).into_affine();

	let bal_b0 = pk.zv_s1_n2.mul(r_b - r_b0).into_affine() +
		pk.zv_s_s1_n2.mul(PE::Fr::zero()-r_q_b0).into_affine(); 
	let ddlog_rand_b0 = PE::Fr::rand(&mut rng);
	let ddlog_rand_b1 = PE::Fr::rand(&mut rng);
	let msg1_q_b0 = pk.zv_s1_n2.mul(ddlog_rand_b0).into_affine() + 
		pk.zv_s_s1_n2.mul(ddlog_rand_b1).into_affine();
	//verifier challenge
	let ver_ch_b0 = hash::<PE::Fr>(&to_vecu8(&vec![bal_b0, msg1_q_b0])); 
	let res1_q_b0 = ddlog_rand_b0 - ver_ch_b0*(r_a-r_b0);
	let res2_q_b0 = ddlog_rand_b1 - ver_ch_b0*(PE::Fr::zero() - r_q_b0);

	//14. produce the DDLOG proof for Commit_a0/Commit_b0^{N/n}
	let fr_n = PE::Fr::from((pk.n/pk.n2) as u64);
	let ddlog_rand_diff0 = PE::Fr::rand(&mut rng);
	let ddlog_rand_diff1 = PE::Fr::rand(&mut rng);
	let commit_diff_a0_b0 = (commit_b0.into_projective() - 
		commit_a0.mul(fr_n)).into_affine();
	let msg1_diff_a0_b0 = pk.zv_s1.mul(ddlog_rand_diff0).into_affine() + 
		pk.zv_s1_n2.mul(ddlog_rand_diff1).into_affine();
	let ver_ch_diff = hash::<PE::Fr>(&to_vecu8(&vec![commit_diff_a0_b0, 
		msg1_diff_a0_b0])); 
	let res1_diff_a0_b0 = ddlog_rand_diff0 
		- ver_ch_diff * (fr_n*(PE::Fr::zero()-r_a0));
	let res2_diff_a0_b0 = ddlog_rand_diff1 - ver_ch_diff * r_b0;


	(CqProof{commit_m: commit_m, commit_a: commit_a,
		commit_qa: commit_qa, commit_b: commit_b, commit_p: commit_p,
		commit_q_b: commit_q_b,
		b_gamma: b_gamma, t_gamma: t_gamma, q_b_gamma: q_b_gamma,
		prf_gamma: prf_gamma,
		commit_a0: commit_a0, commit_q_a0: commit_q_a0, msg1_q_a0: msg1_q_a0,
		res1_q_a0: res1_q_a0, res2_q_a0: res2_q_a0, bal_a0: bal_a0,
		commit_b0: commit_b0, commit_q_b0: commit_q_b0, bal_b0: bal_b0,
		res1_q_b0: res1_q_b0, res2_q_b0: res2_q_b0, msg1_q_b0: msg1_q_b0,
		msg1_diff_a0_b0: msg1_diff_a0_b0,		 
		res1_diff_a0_b0: res1_diff_a0_b0, res2_diff_a0_b0: res2_diff_a0_b0
		}, 
	r_query, commit_query)
} 

/// SLOWER (logical) version: verify the proof, for debugging
pub fn slow_verify_cq<PE:PairingEngine>(
	vk: &KzgVerifierKey<PE>,
	commit_lookup_table: PE::G2Affine, 
	commit_query_table: PE::G1Affine,
	prf: &CqProof<PE>)-> bool where
<<PE as PairingEngine>::G1Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G1Affine, Scalar=<<PE as PairingEngine>::G1Affine as AffineCurve>::ScalarField>,
 <<PE as PairingEngine>::G2Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G2Affine, Scalar=<<PE as PairingEngine>::G2Affine as AffineCurve>::ScalarField>{
	//1. compute beta
	let beta = hash::<PE::Fr>(&to_vecu8(&vec![commit_query_table, prf.commit_m]));	

	//2. verify equation e(commit_a, commit_t) = 
	// e(commit_qa, [zv_s]) * e(commit_m - beta commit_a, [1])
	let lhs = PE::pairing(prf.commit_a, commit_lookup_table);	
	let diff_g = prf.commit_m.into_projective() - prf.commit_a.mul(beta);
	let rhs = PE::pairing(prf.commit_qa, vk.zv_s2) * PE::pairing(diff_g,vk.g2);	 	let b1 = lhs==rhs;

	//3. check e(c_b, [s^(N-n-1)]) = e(p, [1])
	let b2 = PE::pairing(prf.commit_b, vk.s2_n_diff) ==
			PE::pairing(prf.commit_p, vk.g2);

	//4. compute hash gamma
	let mut v1 = to_vecu8(&vec![prf.commit_a, prf.commit_qa, prf.commit_b, 
		prf.commit_q_b, prf.commit_p]);
	let mut v2 = to_vecu8(&vec![beta]);
	v1.append(&mut v2);
	let gamma = hash::<PE::Fr>(&v1);

	//5 compute eta and verify prf_gamma
	let eta = hash::<PE::Fr>(&to_vecu8(
		&vec![gamma, prf.b_gamma, prf.t_gamma, prf.q_b_gamma]));
	let v = prf.b_gamma + eta * prf.t_gamma  + eta * eta * prf.q_b_gamma;
	let c = prf.commit_b + commit_query_table.mul(eta).into_affine() + 
		prf.commit_q_b.mul(eta * eta).into_affine();
	let lhs_c = c.into_projective() - vk.g1.mul(v) + prf.prf_gamma.mul(gamma);
	let b3 = PE::pairing(lhs_c, vk.g2) == PE::pairing(prf.prf_gamma, vk.s2);

	//6. verify the q_a0
	let lhs_a0 = prf.commit_a.into_projective() - 
		prf.commit_a0.into_projective() - prf.bal_a0.into_projective();	
	let b4 = PE::pairing(lhs_a0, vk.g2) ==
			PE::pairing(prf.commit_q_a0, vk.s2);

	let ver_ch_a0=hash::<PE::Fr>(&to_vecu8(&vec![prf.bal_a0, prf.msg1_q_a0])); 
	let b5 = prf.msg1_q_a0 ==
		(vk.zv_s1.mul(prf.res1_q_a0).into_affine() 
			+ vk.zv_s_s1.mul(prf.res2_q_a0).into_affine())
		+ prf.bal_a0.mul(ver_ch_a0).into_affine();
	
	//7. verify the q_b0
	let lhs_b0 = prf.commit_b.into_projective() - 
		prf.commit_b0.into_projective() - prf.bal_b0.into_projective();	
	let b6 = PE::pairing(lhs_b0, vk.g2) ==
			PE::pairing(prf.commit_q_b0, vk.s2);
	let ver_ch_b0=hash::<PE::Fr>(&to_vecu8(&vec![prf.bal_b0, prf.msg1_q_b0])); 
	let b7 = prf.msg1_q_b0 ==
		(vk.zv_s1_n2.mul(prf.res1_q_b0).into_affine() 
			+ vk.zv_s_s1_n2.mul(prf.res2_q_b0).into_affine())
		+ prf.bal_b0.mul(ver_ch_b0).into_affine();

	//8 verify the commit_diff_a0_b0
	let fr_n = PE::Fr::from((vk.n/vk.n2) as u64);
	let commit_diff_a0_b0 = (prf.commit_b0.into_projective() - 
		prf.commit_a0.mul(fr_n)).into_affine();
	let ver_ch_diff = hash::<PE::Fr>(&to_vecu8(&vec![commit_diff_a0_b0, 
		prf.msg1_diff_a0_b0])); 
	let b8 = prf.msg1_diff_a0_b0 ==
		(vk.zv_s1.mul(prf.res1_diff_a0_b0).into_affine() + 
			vk.zv_s1_n2.mul(prf.res2_diff_a0_b0).into_affine() )
		+ commit_diff_a0_b0.mul(ver_ch_diff).into_affine();

	return b1 && b2 && b3 && b4 && b5 && b6 && b7 && b8;
}

/// verify the proof
pub fn verify_cq<PE:PairingEngine>(
	vk: &KzgVerifierKey<PE>,
	commit_lookup_table: PE::G2Affine, 
	commit_query_table: PE::G1Affine,
	prf: &CqProof<PE>)-> bool where
<<PE as PairingEngine>::G1Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G1Affine, Scalar=<<PE as PairingEngine>::G1Affine as AffineCurve>::ScalarField>,
 <<PE as PairingEngine>::G2Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G2Affine, Scalar=<<PE as PairingEngine>::G2Affine as AffineCurve>::ScalarField>{
	//1. compute beta
	let b_perf = false;
	let mut timer = Timer::new();
	let mut rng = gen_rng_from_seed(gen_seed());
	let rc = PE::Fr::rand(&mut rng); //random challenge for acc equations
	let mut cur_rc = PE::Fr::one();
	let g1_0 = vk.g1.mul(PE::Fr::zero());
	let beta = hash::<PE::Fr>(&to_vecu8(&vec![commit_query_table, 
		prf.commit_m]));	

	//2. verify equation e(commit_a, commit_t) = 
	// e(commit_qa, [zv_s]) * e(commit_m - beta commit_a, [1])
	let gt_commit_lookup = PE::pairing(prf.commit_a, commit_lookup_table);	
	let diff_g = prf.commit_m.into_projective() - prf.commit_a.mul(beta);
	let acc_zv_s2 = (g1_0 - prf.commit_qa.into_projective()).into_affine();
	let mut acc_g2 = (g1_0 - diff_g).into_affine();
	cur_rc = cur_rc * rc;

	//3. check e(c_b, [s^(N-n-1)]) = e(p, [1])
	let acc_s2_n_diff =  prf.commit_b.mul(cur_rc);
	acc_g2 = acc_g2 +  (g1_0 - prf.commit_p.into_projective())
		.into_affine().mul(cur_rc).into_affine();	
	cur_rc = cur_rc * rc;
	

	//4. compute hash gamma
	let mut v1 = to_vecu8(&vec![prf.commit_a, prf.commit_qa, prf.commit_b, 
		prf.commit_q_b, prf.commit_p]);
	let mut v2 = to_vecu8(&vec![beta]);
	v1.append(&mut v2);
	let gamma = hash::<PE::Fr>(&v1);

	//5 compute eta and verify prf_gamma
	let eta = hash::<PE::Fr>(&to_vecu8(
		&vec![gamma, prf.b_gamma, prf.t_gamma, prf.q_b_gamma]));
	let v = prf.b_gamma + eta * prf.t_gamma  + eta * eta * prf.q_b_gamma;
	let c = prf.commit_b + commit_query_table.mul(eta).into_affine() + 
		prf.commit_q_b.mul(eta * eta).into_affine();
	let lhs_c = c.into_projective() - vk.g1.mul(v) + prf.prf_gamma.mul(gamma);
	acc_g2 = acc_g2 + lhs_c.into_affine().mul(cur_rc).into_affine();
	let mut acc_s2 = (g1_0 - prf.prf_gamma.into_projective())
		.into_affine().mul(cur_rc).into_affine();
	cur_rc = cur_rc * rc;
	

	//6. verify the q_a0
	let lhs_a0 = prf.commit_a.into_projective() - 
		prf.commit_a0.into_projective() - prf.bal_a0.into_projective();	
	acc_g2 = acc_g2 + lhs_a0.into_affine().mul(cur_rc).into_affine();
	acc_s2 = acc_s2 + (g1_0 - prf.commit_q_a0.into_projective())
		.into_affine().mul(cur_rc).into_affine();
	cur_rc = cur_rc * rc;

	let ver_ch_a0=hash::<PE::Fr>(&to_vecu8(&vec![prf.bal_a0, prf.msg1_q_a0])); 
	let b5 = prf.msg1_q_a0 ==
		(vk.zv_s1.mul(prf.res1_q_a0).into_affine() 
			+ vk.zv_s_s1.mul(prf.res2_q_a0).into_affine())
		+ prf.bal_a0.mul(ver_ch_a0).into_affine();
	
	//7. verify the q_b0
	let lhs_b0 = prf.commit_b.into_projective() - 
		prf.commit_b0.into_projective() - prf.bal_b0.into_projective();	
	let b6 = PE::pairing(lhs_b0, vk.g2) ==
			PE::pairing(prf.commit_q_b0, vk.s2);
	acc_g2 = acc_g2 + lhs_b0.into_affine().mul(cur_rc).into_affine();
	acc_s2 = acc_s2 + (g1_0 - prf.commit_q_b0.into_projective()).into_affine().mul(cur_rc).into_affine();
 
	let ver_ch_b0=hash::<PE::Fr>(&to_vecu8(&vec![prf.bal_b0, prf.msg1_q_b0])); 
	let b7 = prf.msg1_q_b0 ==
		(vk.zv_s1_n2.mul(prf.res1_q_b0).into_affine() 
			+ vk.zv_s_s1_n2.mul(prf.res2_q_b0).into_affine())
		+ prf.bal_b0.mul(ver_ch_b0).into_affine();

	//8 verify the commit_diff_a0_b0
	let fr_n = PE::Fr::from((vk.n/vk.n2) as u64);
	let commit_diff_a0_b0 = (prf.commit_b0.into_projective() - 
		prf.commit_a0.mul(fr_n)).into_affine();
	let ver_ch_diff = hash::<PE::Fr>(&to_vecu8(&vec![commit_diff_a0_b0, 
		prf.msg1_diff_a0_b0])); 
	let b8 = prf.msg1_diff_a0_b0 ==
		(vk.zv_s1.mul(prf.res1_diff_a0_b0).into_affine() + 
			vk.zv_s1_n2.mul(prf.res2_diff_a0_b0).into_affine() )
		+ commit_diff_a0_b0.mul(ver_ch_diff).into_affine();
	if b_perf{log_perf(LOG1, "-- G1 ops", &mut timer);}

	//9. accmulate all pairings
	let gt_zv_s2 = PE::pairing(acc_zv_s2, vk.zv_s2);
	let gt_g2 = PE::pairing(acc_g2, vk.g2);
	let gt_s2_n_diff = PE::pairing(acc_s2_n_diff, vk.s2_n_diff);
	let gt_s2 = PE::pairing(acc_s2, vk.s2);
	let gt_prod = gt_zv_s2 * gt_g2 * gt_s2_n_diff * gt_s2 * gt_commit_lookup;
	let b9 = gt_prod.is_one();
	if b_perf{log_perf(LOG1, "-- pairings", &mut timer);}

	return b5 && b6 && b7 && b8 && b9;
}

/// generate a proof for 
/// \sum_i=0^k \alpha^i*query_table[i]
/// in \sum_i=0^k alpha^i*lookup_table[i]
/// return (proof, array of nonces used, array of commitments of query table)
pub fn fold_prove_cq<PE:PairingEngine>(
	pk: &KzgProverKey<PE>, 
	arr_aux: &Vec<MyArc<CqAux<PE>>>, 
	arr_lookup_table: &Vec<MyArc<Vec<PE::Fr>>>,
	arr_query_table: &Vec<MyArc<Vec<PE::Fr>>>,
	alpha: PE::Fr) 
	-> (CqProof<PE>, Vec<PE::Fr>, Vec<PE::G1Affine>) where
<<PE as PairingEngine>::G1Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G1Affine, Scalar=<<PE as PairingEngine>::G1Affine as AffineCurve>::ScalarField>,
 <<PE as PairingEngine>::G2Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G2Affine, Scalar=<<PE as PairingEngine>::G2Affine as AffineCurve>::ScalarField>{
	//1. check input
	let (b_perf, b_test) = (true, false);
	let mut timer = Timer::new();
	let k = arr_query_table.len(); //number of instances of folding
	let n2 = arr_query_table[0].arc.len();
	assert!(n2.is_power_of_two(), "query table size is not power of 2");
	for i in 0..arr_query_table.len() {
		assert!(arr_query_table[i].arc.len()==n2, "len of query_table[{}]!=n2", i);
		assert!(arr_lookup_table[i].arc.len()==arr_lookup_table[0].arc.len(),
			"lookup table {} len != 1st lookup", i);
	}
	if b_perf {println!("-------- fold prove --------");}

	//2. compute commit_query, commit_m
	let vec_3m = (0..k).into_par_iter().map(|i|
		ped_commit::<PE::G1Affine>(&arr_query_table[i].arc, n2, 
			&pk.lag_all_n2, pk.zv_s1_n2)).collect::<Vec<(_,_,_)>>();
	let r_querys = vec_3m.clone().into_par_iter().map(|(r,_,_)| r).collect::
		<Vec<PE::Fr>>();
	let commit_query: PE::G1Affine= 
		vec_3m.clone().into_par_iter().zip((0..k).into_par_iter())
		.map(|((_,c,_),i)| c.mul(alpha.pow(&[i as u64])).into_affine()).sum();
	let commit_querys = 
		vec_3m.clone().into_par_iter().zip((0..k).into_par_iter())
		.map(|((_,c,_),_i)| c).collect::<Vec<PE::G1Affine>>();
	let (idx_m, vec_m) = gen_m(&arr_aux[0].arc.map_idx.map, &arr_query_table[0].arc);
	if b_test{
		for i in 0..k{
			let (im,vm) = gen_m(&arr_aux[i].arc.map_idx.map,&arr_query_table[i].arc);
			assert!(im==idx_m && vm==vec_m, "inconsisnt m generation!");
		}
	}
	let (r_m, commit_m, _) = sparse_ped_commit(&vec_m, &idx_m, 
		&pk.lag_all, pk.zv_s1);
	if b_perf {log_perf(LOG1, "-------- commit_m", &mut timer);}

	//3. compute beta
	let beta = hash::<PE::Fr>(&to_vecu8(&vec![commit_query, commit_m]));


	//4. compute A_i and commit_a
	let idx_a = idx_m.clone();
	let one = PE::Fr::one();
	let vec_a = vec_m.par_iter().zip(idx_m.par_iter())
		.map(|(m,idx)| *m/(beta + 
			(0..k).into_iter().map(|x| arr_lookup_table[x].arc[*idx] * alpha.pow(&[x as u64])).sum::<PE::Fr>()
		)).
		collect::<Vec<PE::Fr>>();
	let (r_a, commit_a, a_s) = sparse_ped_commit(&vec_a, &idx_a, 
		&pk.lag_all, pk.zv_s1);
	if b_perf {log_perf(LOG1, "-------- commit_a", &mut timer);}

	let g1 = PE::G1Affine::prime_subgroup_generator();
	let arr_alpha = compute_powers(k, alpha);
	/*
	let arr_q: Vec<PE::G1Affine> = 
		(0..arr_aux[0].arr_q.len()).into_par_iter().map(|i|
			(0..k).into_iter().map(|idx|
				arr_aux[idx].arr_q[i].mul(alpha.pow(&[idx as u64]))
					.into_affine()).sum()
			).collect();
	let arr_q: Vec<PE::G1Affine> = 
		(0..arr_aux[0].arr_q.len()).into_par_iter().map(|i|
			(0..k).into_iter().map(|idx|
				arr_aux[idx].arr_q[i].mul(arr_alpha[idx])
					.into_affine()).sum()
			).collect();
	*/
	let mut arr_q: Vec<PE::G1Affine> = 
		(0..idx_a.len()).into_par_iter().map(|i|
			(0..k).into_iter().map(|idx|
				arr_aux[idx].arc.arr_q[idx_a[i]].mul(arr_alpha[idx])
					.into_affine()).sum()
			).collect();
	assert!(arr_q.len()==idx_a.len(), "arr_q.len!=idx_a.len");
	if n2>arr_q.len(){
		let g1 = PE::G1Affine::prime_subgroup_generator();
		arr_q.append(&mut vec![g1; n2-arr_q.len()]);
	}

	//let (_, _, qa_s) = sparse_ped_commit(&vec_a, &idx_a, &arr_q, pk.zv_s1);
	let (_, _, qa_s) = ped_commit(&vec_a, n2, &arr_q, pk.zv_s1);
	let t_s_1:PE::G1Affine = (0..k).into_par_iter().map(|idx|
		arr_aux[idx].arc.t_s_1.mul(alpha.pow(&[idx as u64])).into_affine()).sum();
	let r_t: PE::Fr = (0..k).into_par_iter().map(|idx|
		arr_aux[idx].arc.r_t * alpha.pow(&[idx as u64])).sum();	
	let commit_qa = qa_s + t_s_1.mul(r_a).into_affine() 
		+ a_s.mul(r_t).into_affine() 
		+ pk.zv_s1.mul(r_t * r_a).into_affine()
		+ g1.mul(r_a * beta - r_m).into_affine();
	if b_perf {log_perf(LOG1, "-------- commit_qa", &mut timer);}


	//6. comptue b, p and qb
	let query_table = (0..n2).into_par_iter().map(|idx|
		(0..k).into_iter().map(|x|
			arr_query_table[x].arc[idx]*alpha.pow(&[x as u64])).sum()).
			collect::<Vec<PE::Fr>>();
	let vec_b = (0..n2).into_par_iter().map(|idx|
		one/(query_table[idx] + beta)).
		collect::<Vec<PE::Fr>>();
	let r_query = (0..k).into_iter().map(|x|
		r_querys[x] * alpha.pow(&[x as u64])).sum();
	let (r_b, commit_b, _)=ped_commit(&vec_b, n2, &pk.lag_all_n2, pk.zv_s1_n2); 
	if b_perf {log_perf(LOG1, "---------- commit_b", &mut timer);}

	let (coef_q_b, coef_b, coef_t) = compute_coeff_q_b(&vec_b,r_b,&query_table,
		r_query,beta,n2);
	if b_perf {log_perf(LOG1, "---------- compute q_b", &mut timer);}

	let commit_q_b = vmsm(&pk.vec_g1, &coef_q_b);
	let mut coef_b2 =  coef_b.clone()[0..pk.n2+1].to_vec(); //degree n, can cap
	let mut coef_p = vec![PE::Fr::zero(); pk.n-pk.n2-1];
	coef_p.append(&mut coef_b2);
	let commit_p = vmsm(&pk.vec_g1, &coef_p);
	if b_perf {log_perf(LOG1, "---------- commit_q_b and commit_p", &mut timer);}

	//7. compute gamma
	let mut v1 = to_vecu8(&vec![commit_a, commit_qa, commit_b, 
		commit_q_b, commit_p]);
	let mut v2 = to_vecu8(&vec![beta]);
	v1.append(&mut v2);
	let gamma = hash::<PE::Fr>(&v1);

	//8. compute b_gamma, t_gamma, q_b_gamma
	let b_gamma = eval_coefs_at(&coef_b, gamma); 
	let t_gamma = eval_coefs_at(&coef_t, gamma);
	let z_h_gamma = gamma.pow(&[n2 as u64]) - one;
	let q_b_gamma = (b_gamma*(t_gamma + beta) - one)/z_h_gamma;

	//9 compute eta
	let eta = hash::<PE::Fr>(&to_vecu8(
		&vec![gamma, b_gamma, t_gamma, q_b_gamma]));
	let v = b_gamma + eta * t_gamma  + eta * eta * q_b_gamma;
	if b_perf {log_perf(LOG1, "---------- compute b_gamma, t_gamma...", &mut timer);}

	//10. compute h(X)
	let b_x = get_poly(coef_b);
	let t_x = get_poly(coef_t);
	let q_b_x = get_poly(coef_q_b);
	let eta_x = get_poly(vec![eta]);
	let eta_x2 = get_poly(vec![eta*eta]);
	let v_x = get_poly(vec![PE::Fr::zero() - v]);
	let part1 = b_x + (&t_x * &eta_x) + (&eta_x2 * &q_b_x) + v_x;
	let part2 = get_poly(vec![PE::Fr::zero() - gamma, PE::Fr::one()]);
	let (h_x, remainder_x) = old_divide_with_q_and_r(&part1, &part2);
	assert!(remainder_x.is_zero(), "remainder is not zero");
	if b_perf {log_perf(LOG1, "---------- compute h(X)", &mut timer);}

	//11. compute prf gamma
	let coefs_h = h_x.coeffs;
	let prf_gamma = vmsm(&pk.vec_g1, &coefs_h);
	if b_perf {log_perf(LOG1, "---------- compute prf_gamma: [h(s)]_1", &mut timer);}

	//12. compute prf_q_a0 (including a DDLOG proof)
	let mut rng = gen_rng_from_seed(gen_seed());
	let (r_q_a0, commit_q_a0, _) = sparse_ped_commit(&vec_a, &idx_a, 
		&pk.lag_diff, pk.zv_s1);
	let r_a0 = PE::Fr::rand(&mut rng);
	let a0 = sparse_inner_prod(&idx_a, &vec_a, &pk.lag0_fe);

	let commit_a0 = g1.mul(a0).into_affine()+pk.zv_s1.mul(r_a0).into_affine();	
	let bal_a0 = pk.zv_s1.mul(r_a - r_a0).into_affine() +
		pk.zv_s_s1.mul(PE::Fr::zero()-r_q_a0).into_affine(); //bal term for a0

	let ddlog_rand_a0 = PE::Fr::rand(&mut rng);
	let ddlog_rand_a1 = PE::Fr::rand(&mut rng);
	let msg1_q_a0 = pk.zv_s1.mul(ddlog_rand_a0).into_affine() + 
		pk.zv_s_s1.mul(ddlog_rand_a1).into_affine();
	//verifier challenge
	let ver_ch_a0 = hash::<PE::Fr>(&to_vecu8(&vec![bal_a0, msg1_q_a0])); 
	let res1_q_a0 = ddlog_rand_a0 - ver_ch_a0*(r_a-r_a0);
	let res2_q_a0 = ddlog_rand_a1 - ver_ch_a0*(PE::Fr::zero() - r_q_a0);

	//13. compute prf_q_b0 (including a DDLOG proof)
	let coef_b = evals_to_coefs(&vec_b, PE::Fr::one());
	let b0 = coef_b[0];
	let coef_q_b0 = &coef_b[1..coef_b.len()].to_vec();
	let r_b0 = PE::Fr::rand(&mut rng);
	let commit_b0 = g1.mul(b0).into_affine()+
		pk.zv_s1_n2.mul(r_b0).into_affine();
	let r_q_b0 = PE::Fr::rand(&mut rng);
	let commit_q_b0 = vmsm(&pk.vec_g1, &coef_q_b0) + 
		pk.zv_s1_n2.mul(r_q_b0).into_affine();

	let bal_b0 = pk.zv_s1_n2.mul(r_b - r_b0).into_affine() +
		pk.zv_s_s1_n2.mul(PE::Fr::zero()-r_q_b0).into_affine(); 
	let ddlog_rand_b0 = PE::Fr::rand(&mut rng);
	let ddlog_rand_b1 = PE::Fr::rand(&mut rng);
	let msg1_q_b0 = pk.zv_s1_n2.mul(ddlog_rand_b0).into_affine() + 
		pk.zv_s_s1_n2.mul(ddlog_rand_b1).into_affine();
	//verifier challenge
	let ver_ch_b0 = hash::<PE::Fr>(&to_vecu8(&vec![bal_b0, msg1_q_b0])); 
	let res1_q_b0 = ddlog_rand_b0 - ver_ch_b0*(r_a-r_b0);
	let res2_q_b0 = ddlog_rand_b1 - ver_ch_b0*(PE::Fr::zero() - r_q_b0);

	//14. produce the DDLOG proof for Commit_a0/Commit_b0^{N/n}
	let fr_n = PE::Fr::from((pk.n/pk.n2) as u64);
	let ddlog_rand_diff0 = PE::Fr::rand(&mut rng);
	let ddlog_rand_diff1 = PE::Fr::rand(&mut rng);
	let commit_diff_a0_b0 = (commit_b0.into_projective() - 
		commit_a0.mul(fr_n)).into_affine();
	let msg1_diff_a0_b0 = pk.zv_s1.mul(ddlog_rand_diff0).into_affine() + 
		pk.zv_s1_n2.mul(ddlog_rand_diff1).into_affine();
	let ver_ch_diff = hash::<PE::Fr>(&to_vecu8(&vec![commit_diff_a0_b0, 
		msg1_diff_a0_b0])); 
	let res1_diff_a0_b0 = ddlog_rand_diff0 
		- ver_ch_diff * (fr_n*(PE::Fr::zero()-r_a0));
	let res2_diff_a0_b0 = ddlog_rand_diff1 - ver_ch_diff * r_b0;
	if b_perf {log_perf(LOG1, "---------- compute zk_qa0 DLOG", &mut timer);}

	(CqProof{commit_m: commit_m, commit_a: commit_a,
		commit_qa: commit_qa, 
		commit_b: commit_b, commit_p: commit_p,
		commit_q_b: commit_q_b,
		b_gamma: b_gamma, t_gamma: t_gamma, q_b_gamma: q_b_gamma,
		prf_gamma: prf_gamma,
		commit_a0: commit_a0, commit_q_a0: commit_q_a0, msg1_q_a0: msg1_q_a0,
		res1_q_a0: res1_q_a0, res2_q_a0: res2_q_a0, bal_a0: bal_a0 ,
		commit_b0: commit_b0, commit_q_b0: commit_q_b0, bal_b0: bal_b0,
		res1_q_b0: res1_q_b0, res2_q_b0: res2_q_b0, msg1_q_b0: msg1_q_b0,
		msg1_diff_a0_b0: msg1_diff_a0_b0,		 
		res1_diff_a0_b0: res1_diff_a0_b0, res2_diff_a0_b0: res2_diff_a0_b0
		}, 
	r_querys, commit_querys)
} 

/* ----------- Utility functions below --------------- */

/// generate vector m such that m_i is the number of
/// It returns the compact form (idx_arr, compact_m)
/// where compact_m[i] is m[idx_arr[i]], i.e., it contains
/// all non-zero elements of m_i.
/// the lookup_loc_info maps a field element in lookup_table
/// to its location in lookup table
fn gen_m<F:Field>(lookup_loc_info: &HashMap<F,usize>, 
	query_table: &Vec<F>)->(Vec<usize>,Vec<F>){
	//1. build occurence hashmap
	let mut hashmap = HashMap::new(); //map loc_idx -> occurence
	for v in query_table{ 
		let loc_in_lookup = lookup_loc_info.get(&v).unwrap(); //crash if not 
		*hashmap.entry(loc_in_lookup).or_insert(0usize) += 1; 
	}

	//2. build the result
	let mut idx_arr = vec![];
	let mut occ_arr = vec![];
	for key in hashmap.keys().sorted(){
		idx_arr.push(**key);
		let occ = hashmap.get(key).unwrap();
		let focc = F::from(*occ as u64);
		occ_arr.push(focc);
	}
	return (idx_arr, occ_arr);
}

/// expand the sparse representation (idx, values) into a full vector
/// of size target_n so that res[i] = 0 if i not in idx, but
/// result[idx[i]] = values[i] for each i in [0, len(values) 
/// for debug use only (slow)
fn expand_sparse_vec<F:Field>(target_n: usize, idx: &Vec<usize>, 
	values: &Vec<F>)->Vec<F>{
	let mut res = vec![F::zero(); target_n];
	for i in 0..idx.len(){
		res[idx[i]] = values[i];
	}	
	res
}

/// Given vector t
/// and the random point r, compute all co-efficients
/// of T(X) = \sum_{i=0}^{n-1} t_i L_i(X) + r * z_V{X}
/// where V = {\omega^0, ..., \omega^{n-1} for n'th root of unity \omega
pub fn compute_coefs_of_lag_points<F:FftField>(t:&Vec<F>, r: F)->Vec<F>{
	let n = t.len();
	assert!(n.is_power_of_two(), "n is not power of 2!");
	let b_test = false;


	//1. fft to convert the first part
	let mut coefs = t.clone();
	serial_ifft(&mut coefs);

	//2. add the coefs for z_V(X)
	coefs.push(r); //the highest degree
	coefs[0] = coefs[0] - r; //to apply plus poly r(X^n -1)
	assert!(coefs.len()==n+1, "coefs degree is not n!");

	//3. self check
	if b_test{
		let s = F::from(12312312098123u128);
		let v1 = logical_eval_as_poly_ext(&t, r, s, n);
		let v2 = eval_coefs_at(&coefs, s);
		assert!(v1==v2, "compute_coefs_of_lag_points failed");
	}

	coefs
}


/// evaluate z_V(X) over {\mu^0, ..., \mu^{2n-1}} where \mu^2n = 1
/// z_V(X) = X^n - 1
pub fn eval_z_h_over_2n<F:FftField>(n: usize, coset_shift: F)->Vec<F>{
	assert!(n.is_power_of_two(), "n is not power of 2");
	let mut vc = vec![F::zero(); 2*n];
	vc[0] = F::zero() - F::one();
	vc[n] = F::one(); //thus encoding coefs array for x^n - 1
	inplace_fft_coset(&mut vc, coset_shift);

	vc
}

/// compute omega^n-1 for {omega^i} where omega^n2 = 1
pub fn eval_z_h_over_custom_n<F:FftField>(n: usize, n2: usize, coset_shift: F)->Vec<F>{
	assert!(n.is_power_of_two(), "n is not power of 2");
	assert!(n2.is_power_of_two(), "n2 is not power of 2");
	assert!(n2>=2*n, "n2<=2*n");
	let mut vc = vec![F::zero(); n2];
	vc[0] = F::zero() - F::one();
	vc[n] = F::one(); //thus encoding coefs array for x^n - 1
	inplace_fft_coset(&mut vc, coset_shift);

	vc
}


/// convert poly from coefs form to array of points evaluated
/// at each lagrange poly, i.e., using FFT.
/// Given p(X) = \sum c_i X^n, return
/// {p(\omega^0), ..., p(\omega^{n-1})
pub fn coefs_to_evals<F:FftField>(c: &Vec<F>, coset_shift: F) ->Vec<F>{
	let n = c.len();
	assert!(n.is_power_of_two(), "n is not power of 2");
	let mut c2 = c.clone();
	inplace_fft_coset(&mut c2, coset_shift);
	
	c2
}

/// Given {p(\omega^0), ..., p(\omega^{n-1})
/// compute the coefficients array of p(X)
pub fn evals_to_coefs<F:FftField>(v: &Vec<F>, coset_shift: F) -> Vec<F>{
    let n = v.len();
    assert!(n.is_power_of_two(), "n is not power of 2");
    let mut v2 = v.clone();
    inplace_ifft_coset(&mut v2, coset_shift);


    v2
}


/// let c be the vector of co-effs of a polynomial C(X)
/// evaluate C(x) = \sum_{i=0}^n c_i x^i
pub fn eval_coefs_at<F:FftField>(c: &Vec<F>, x:F)->F{
	let c2 = c.clone();
	let p = get_poly(c2);
	p.evaluate(&x)	
}

/// extend the vector to the desired size with 0's elements
pub fn extend_vec_to_size<F:FftField>(vec: &mut Vec<F>, target_size: usize){
	let n = vec.len();
	if n>=target_size {return;}
	let mut sec2 = vec![F::zero(); target_size - n];
	vec.append(&mut sec2);
}

/// compute the co-efficients of the polynomial q_b so that
/// cap_b(X) (cap_t(X) + beta) - 1 = q_b(X) z_H(x)
/// where H is the domain of roots of unity of n
/// See step (4)
/// Analsyis: cap_b(X) (with r_b) is degree n (i.e., n+1 co-efficients).
/// samme are cap_t(X) with r_t. LHS is degree 2n. z_H(X) is degree n
/// we have q_b(X) degree n
/// Retirn (coefs_q_b, coefs_b, coefs_t)
fn compute_coeff_q_b<F:FftField>(b: &Vec<F>, r_b: F, t: &Vec<F>, r_t: F, beta: F, n: usize)->(Vec<F>,Vec<F>, Vec<F>){
	assert!(n.is_power_of_two(), "n is not power of 2!");
	let b_test = false;
	let b_perf = false;
	let mut timer = Timer::new();

	//1. compute values of b(X), t(X), z_H(X) over domain [0, 2n)
	// call to_coefs to convert (b, r_rb) and etc first
	let mut coef_b = compute_coefs_of_lag_points(&b, r_b);
	let mut coef_t = compute_coefs_of_lag_points(&t, r_t);
	extend_vec_to_size(&mut coef_b, 2*n);
	extend_vec_to_size(&mut coef_t, 2*n); 
	assert!(coef_b.len()==coef_t.len() && coef_b.len()==2*n, "coefs len nok!");
	if b_perf {log_perf(LOG1, "---- compute coefs b(X), t(X)", &mut timer);}

	//2. compute the corresponding values of q_b
	let mut rng = gen_rng_from_seed(gen_seed());
	let coset_shift = F::rand(&mut rng);
	let v_b = coefs_to_evals(&coef_b, coset_shift);
	let v_t = coefs_to_evals(&coef_t, coset_shift);
	let v_z_h = eval_z_h_over_2n::<F>(n, coset_shift);
	assert!(v_b.len()==n*2, "v_b.len() != 2n");
	assert!(v_t.len()==n*2, "v_t.len() != 2n");
	assert!(v_z_h.len()==n*2, "v_z_h.len() != 2n");
	let mut v_q_b = vec![F::zero(); v_b.len()];
	let one = F::one();
	for i in 0..v_b.len(){
		v_q_b[i] = (v_b[i]*(v_t[i] + beta) - one)/v_z_h[i];
	}
	let coefs_q_b = evals_to_coefs(&v_q_b, coset_shift); 
	let coefs_q_b = coefs_q_b[0..n+1].to_vec();
	if b_perf {log_perf(LOG1, "---- compute coefs q_b(X)", &mut timer);}
	
	//4. check q(b) degree <=n
	if b_test{
		for i in coefs_q_b.len()-1..0{
			if coefs_q_b[i]!=F::zero(){
				assert!(i<=n, "coefs_q_b.degree(): {} > n: {}", i, n);
				break;
			}
		}
	}

	//5. UNIT testing by random sample
	if b_test{
		//let gamma = F::from(327324234u128);
		let gamma = F::get_root_of_unity(2*n as u64).unwrap() * coset_shift;
		let v_b = eval_coefs_at(&coef_b, gamma);
		let v_t = eval_coefs_at(&coef_t, gamma);
		let v_z_h = gamma.pow(&[n as u64]) - one;
		let v_q_b = eval_coefs_at(&coefs_q_b, gamma);

		assert!(v_b * (v_t + beta) - one == v_q_b * v_z_h, 
			"compute_q_b validation failed!");
	}

	(coefs_q_b, coef_b, coef_t)
}

#[cfg(test)]
mod tests{
	extern crate ark_bls12_381;
	extern crate acc;
	extern crate rayon;
	extern crate ark_ff;
	extern crate ark_ec;

	use self::ark_bls12_381::Bls12_381;
	type Fr381 = ark_bls12_381::Fr;
	type PE381 = Bls12_381;
	type G1_381= ark_bls12_381::G1Affine;

	use std::sync::Arc;
	use self::rayon::prelude::*; 
	use izpr::serial_poly_utils::{default_trapdoor, setup_kzg};
	use self::ark_ff::{Field,FftField, PrimeField, One, Zero};
	use self::ark_ec::{AffineCurve, PairingEngine,ProjectiveCurve};
	use self::acc::tools::{rand_arr_field_ele};
	use izpr::serial_poly_utils::GLOBAL_TRAPDOOR_S;
	use izpr::serial_cq::{preprocess_cq, prove_cq, slow_verify_cq, verify_cq,gen_seed,fold_prove_cq,CqAux};
	use izpr::serial_pn_lookup::{MyArc};

	#[test]
	pub fn test_cq(){
		let n = 8usize;
		let seed = gen_seed();
		let trapdoor = default_trapdoor::<PE381>();
		let lookup_table = rand_arr_field_ele::<Fr381>(n, seed); 
		let query_table = vec![lookup_table[1], lookup_table[2], lookup_table[3], lookup_table[1]];
		let (pkey, vkey) = setup_kzg::<PE381>(n, query_table.len(), &trapdoor);
		let cq_aux = preprocess_cq::<PE381>(&pkey, &lookup_table);
		let (prf, r_query_table, commit_query_table) = prove_cq(&pkey, 
			&cq_aux, &lookup_table, &query_table);
		let bres2 = slow_verify_cq(&vkey, cq_aux.commit_t2, commit_query_table, &prf);
		let bres = verify_cq(&vkey, cq_aux.commit_t2, commit_query_table, &prf);
		assert!(bres && bres2, "cq failed"); 
	}

	#[test]
	pub fn test_fold_cq(){
		let k =4;
		let n = 8usize;
		let seed = gen_seed();
		let trapdoor = default_trapdoor::<PE381>();
		let alpha = rand_arr_field_ele::<Fr381>(1, seed+5)[0];
		let lookup_tables = (0..k).into_par_iter().map(|x|
			MyArc::new(rand_arr_field_ele::<Fr381>(n, seed + x as u128)) 
		).collect::<Vec<MyArc<Vec<Fr381>>>>();
		let query_tables = (0..k).into_par_iter().map(|x|
			MyArc::new(vec![lookup_tables[x].arc[1], lookup_tables[x].arc[2], 
				lookup_tables[x].arc[3], lookup_tables[x].arc[1]])
		).collect::<Vec<MyArc<Vec<Fr381>>>>();
		let (pkey, vkey) = setup_kzg::<PE381>(n, query_tables[0].arc.len(), 
			&trapdoor);
		let cq_auxs = (0..k).into_par_iter().map(|x|
			MyArc::new(preprocess_cq::<PE381>(&pkey, &lookup_tables[x].arc))
		).collect::<Vec<MyArc<CqAux<PE381>>>>();
		let (prf, r_query_tables, commit_query_tables) 
			= fold_prove_cq::<PE381>(&pkey, &cq_auxs, 
				&lookup_tables, &query_tables, alpha);
		let commit_query_table = (0..k).into_par_iter().map(|x|
			commit_query_tables[x].mul(alpha.pow(&[x as u64])).into_affine()
		).sum();
		let commit_t2 = (0..k).into_par_iter().map(|x|
			cq_auxs[x].arc.commit_t2.mul(alpha.pow(&[x as u64])).into_affine()
		).sum();

		let bres2 = slow_verify_cq(&vkey, commit_t2, commit_query_table, &prf);
		let bres = verify_cq(&vkey, commit_t2, commit_query_table, &prf);
		assert!(bres && bres2, "fold_cq failed"); 
	}
	
}
