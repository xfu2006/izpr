/*
    Copyright Dr. Xiang Fu

    Author: Xiang Fu
    All Rights Reserved.
    Created: 08/23/2023
	Completed: 08/27/2023

	This files implements the zk-pn-lookup (see section 3 of paper)
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

use std::sync::Arc;
use self::rayon::prelude::*; 
use self::ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Read, SerializationError, Write};
use self::ark_ec::{AffineCurve, PairingEngine, ProjectiveCurve,msm::VariableBaseMSM};
use self::acc::tools::{rand_arr_field_ele, gen_rng_from_seed, to_vecu8, hash, log_perf, Timer};
//use self::ark_poly::{Polynomial};
use self::acc::poly::serial::{get_poly, old_divide_with_q_and_r};
use self::acc::profiler::config::{LOG1};
use self::ark_ff::{One,Zero,PrimeField,BigInteger,Field,UniformRand};
use izpr::serial_rng::{RngProverKey, RngVerifierKey, setup_rng, prove_rng, verify_rng,RngProof};

use izpr::serial_cq::{ped_commit,ped_commit_with_random,gen_seed,preprocess_cq, fold_prove_cq, verify_cq, CqAux, CqProof, coefs_to_evals, evals_to_coefs, eval_z_h_over_2n, eval_coefs_at, extend_vec_to_size, compute_coefs_of_lag_points, eval_z_h_over_custom_n};
use izpr::serial_poly_utils::{KzgProverKey,KzgVerifierKey,vmsm,default_trapdoor,precompute_lag_group,setup_kzg};

// -- the following Arc seerialization code is adapted from ark_work
#[derive(Clone)]
pub struct MyArc<T>{
	pub arc: Arc<T>,
}
 
impl <T> MyArc<T>{
	pub fn new(arc_inp: T) -> Self{
		return Self{arc: Arc::new(arc_inp)};
	}
	pub fn as_ref(&self){
		return self.as_ref();
	}
}

impl<T: CanonicalSerialize> CanonicalSerialize for MyArc<T> {
    #[inline]
    fn serialize<W: Write>(&self, mut writer: W) -> Result<(), SerializationError> {
        self.as_ref().serialize(&mut writer)
    }

    #[inline]
    fn serialized_size(&self) -> usize {
        self.as_ref().serialized_size()
    }

    #[inline]
    fn serialize_uncompressed<W: Write>(&self, mut writer: W) -> Result<(), SerializationError> {
        self.as_ref().serialize_uncompressed(&mut writer)
    }

    #[inline]
    fn uncompressed_size(&self) -> usize {
        self.as_ref().uncompressed_size()
    }

    #[inline]
    fn serialize_unchecked<W: Write>(&self, mut writer: W) -> Result<(), SerializationError> {
        self.as_ref().serialize_unchecked(&mut writer)
    }
}

impl<T: CanonicalDeserialize> CanonicalDeserialize for MyArc<T> {
    #[inline]
    fn deserialize<R: Read>(mut reader: R) -> Result<Self, SerializationError> {
        Ok(MyArc::new(T::deserialize(&mut reader)?))
    }

    #[inline]
    fn deserialize_uncompressed<R: Read>(mut reader: R) -> Result<Self, SerializationError> {
        Ok(MyArc::new(T::deserialize_uncompressed(&mut reader)?))
    }

    #[inline]
    fn deserialize_unchecked<R: Read>(mut reader: R) -> Result<Self, SerializationError> {
        Ok(MyArc::new(T::deserialize_unchecked(&mut reader)?))
    }
}

//-- the above code is the port of Rc code from ark_work

/// The Prover Key of Positive-Negative Lookup
//#[derive(Clone)]
#[derive(Clone,CanonicalSerialize,CanonicalDeserialize)]
pub struct PnLookupProverKey<PE:PairingEngine>{
	/// each element in [0,2^log_bound)
	pub log_bound: usize,
	/// for range proof, e.g., 20 bit unit of 160 bit element
	pub log_unit_bound: usize,
	/// number of elements
	pub n: usize,
	/// t1 table [0, s_0, ..., s_{n-1}
	pub t1: MyArc<Vec<PE::Fr>>,
	/// t' table [s_0, ..., s_{n-1}, 2^{logbound}-1]
	pub t2: MyArc<Vec<PE::Fr>>,
	/// random nonce for commit_t1
	pub r_t1: PE::Fr,
	/// random nonce for commit_t2
	pub r_t2: PE::Fr,
	/// prover key for range proof
	pub pk_rng: RngProverKey::<PE>,
	/// pk_kzg for cq proof
	pub pk_kzg: KzgProverKey::<PE>,
	/// aux info for t1
	pub cq_aux_t1: MyArc<CqAux::<PE>>,
	/// aux info for t2
	pub cq_aux_t2: MyArc<CqAux::<PE>>,
}

/// The Verifier Key
//#[derive(Clone)]
#[derive(Clone,CanonicalSerialize,CanonicalDeserialize)]
pub struct PnLookupVerifierKey<PE:PairingEngine>{
	/// each element in [0,2^log_bound)
	pub log_bound: usize,
	/// for range proof, e.g., 20 bit unit of 160 bit element
	pub log_unit_bound: usize,
	/// number of elements in lookup table
	pub n: usize,
	/// number of elements in query table
	pub n2: usize,
	/// commitment to t1
	pub commit_t1: PE::G2Affine,
	/// commitment to t2
	pub commit_t2: PE::G2Affine,
	/// verifier key for range proof
	pub vk_rng: RngVerifierKey::<PE>,
	/// vk_kzg for cq proof
	pub vk_kzg: KzgVerifierKey::<PE>,
}

/// PN-Lookup Proof
#[derive(Clone,CanonicalSerialize,CanonicalDeserialize)]
pub struct PnLookupProof<PE:PairingEngine>{
	/// commit to array u
	pub commit_u: PE::G1Affine,
	/// commit to array v
	pub commit_v: PE::G1Affine,
	/// proof for the cq fold proof
	pub prf_fold: CqProof<PE>,
	/// commit to query table 1
	pub commit_query_table1: PE::G1Affine,
	/// commit to query table 2
	pub commit_query_table2: PE::G1Affine,
	/// range proof
	pub prf_rng: RngProof<PE>,
	/// commitment to polynomial q_o
	pub commit_qo: PE::G1Affine,
	/// commitment to polynomial d
	pub commit_d: PE::G1Affine,
	/// commitment to polynomial v (note diffrom commit_v)
	pub commit_poly_v: PE::G1Affine,
	/// commitment to polynomial q
	pub commit_q: PE::G1Affine,
	/// eval of o(X) at gamma
	pub gamma_o: PE::Fr,
	/// eval of d(X) at gamma
	pub gamma_d: PE::Fr,
	/// eval of poly_v(X) at gamma
	pub gamma_v: PE::Fr,
	/// eval of q(X) at gamma
	pub gamma_q: PE::Fr,
	/// eval of q_o(X) at gamma
	pub gamma_qo: PE::Fr,
	/// proof for evaluation at gamma
	pub prf_gamma: PE::G1Affine
}


/// generate the prover and verifier keys
/// it is required that lookup_table should be sorted in ascending order
/// log_bound: each element in sorted_eles in range [0, 2^log_bound]
/// log_unit_bound: e.g., 20 bits unit of 160 numbers, used for range prf
/// n: must be |sorted_eles| + 1 and a power of 2
/// n2: query table size
pub fn setup_pn_lookup<PE:PairingEngine>(
	log_bound: usize,
	log_unit_bound: usize,
	n: usize,
	n2: usize,
	sorted_eles: &Vec<PE::Fr>) 
	-> (PnLookupProverKey<PE>, PnLookupVerifierKey<PE>) where
<<PE as PairingEngine>::G1Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G1Affine, Scalar=<<PE as PairingEngine>::G1Affine as AffineCurve>::ScalarField>,
 <<PE as PairingEngine>::G2Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G2Affine, Scalar=<<PE as PairingEngine>::G2Affine as AffineCurve>::ScalarField> {
	//1. data check
	let btest = true;
	assert!(n.is_power_of_two(), "n is not power of 2");
	assert!(n==sorted_eles.len()+1, "n!=sorted_eles.len + 1");
	if btest{
		for i in 0..sorted_eles.len()-1{
			assert!(sorted_eles[i]<sorted_eles[i+1], "not sorted!");
		}
	}

	//2. produce t and t' -> t1 and t2
	let mut t1 = vec![PE::Fr::zero()];
	t1.append(&mut sorted_eles.clone());	
	let mut t2 = sorted_eles.clone();
	t2.append(&mut vec![gen_pow2::<PE::Fr>(log_bound)-PE::Fr::one()]);

	//3. produce commit_t and commit_t' 
	let trapdoor = default_trapdoor::<PE>();
	let s = trapdoor.s;
	let lags = precompute_lag_group::<PE::G2Affine>(n, s);
	let g2 = PE::G2Affine::prime_subgroup_generator();
	let zvs = g2.mul(s.pow(&[n as u64])-PE::Fr::one()).into_affine();
	let (r_t1, commit_t1, _) = ped_commit(&t1, n, &lags, zvs);
	let (r_t2, commit_t2, _) = ped_commit(&t2, n, &lags, zvs);

	//4. set up the range proof
	let (pk_rng, vk_rng) = setup_rng(log_bound, log_unit_bound, n2);

	//5. set up the cq  
	let (pk_kzg, vk_kzg) = setup_kzg::<PE>(n, n2, &trapdoor);
	let cq_aux_t1 = MyArc::new(preprocess_cq::<PE>(&pk_kzg, &t1));
	let cq_aux_t2 = MyArc::new(preprocess_cq::<PE>(&pk_kzg, &t2));

	(PnLookupProverKey{log_bound: log_bound, log_unit_bound: log_unit_bound,
		n: n, t1: MyArc::new(t1), t2: MyArc::new(t2), 
		r_t1: r_t1, r_t2: r_t2, pk_rng: pk_rng,
		pk_kzg: pk_kzg, cq_aux_t1: cq_aux_t1, cq_aux_t2: cq_aux_t2},
	 PnLookupVerifierKey{log_bound: log_bound, log_unit_bound: log_unit_bound,
		n: n, n2: n2, 
		commit_t1: commit_t1, commit_t2: commit_t2, vk_rng: vk_rng,
		vk_kzg: vk_kzg}
	)

}

/// generate a Commit_o where o is a boolean array
/// that encodes whether each element of query_table t belongs to
/// the sorted elements in set_up (encoded as t1 and t2 in prover key)
/// return (Commit_o, proof)
/// The r_query and r_o are the random nonces in generating commitments
/// to query_table  and array o.
/// return (commit_query_table, commit_o, proof, vector_o)
pub fn prove_pn_lookup<PE:PairingEngine>(
	pk: &PnLookupProverKey<PE>, 
	query_table: &Vec<PE::Fr>, 
	r_query: PE::Fr, 
	r_o: PE::Fr) 
	-> (PE::G1Affine, PE::G1Affine, PnLookupProof<PE>, Vec<PE::Fr>) where
<<PE as PairingEngine>::G1Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G1Affine, Scalar=<<PE as PairingEngine>::G1Affine as AffineCurve>::ScalarField>,
 <<PE as PairingEngine>::G2Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G2Affine, Scalar=<<PE as PairingEngine>::G2Affine as AffineCurve>::ScalarField> {
	//1. get location array [s_i] and boolean array [o_i]
	let (b_test, b_perf) = (true, true);
	let n2 = query_table.len();
	let (s, o) = gen_output(&pk.t1.arc,&pk.t2.arc,query_table); 
	let mut timer = Timer::new();

	//2. compute commit_s and commit_o, commit_u, commit_v
	if b_perf {println!("------ pn_lookup prove -----");}
	let vec_u = s.clone().into_par_iter().map(|x| pk.t1.arc[x]).collect::<Vec<_>>();
	let vec_v = s.into_par_iter().map(|x| pk.t2.arc[x]).collect::<Vec<_>>();
	assert!(n2==pk.pk_kzg.n2, "n2!=pk.pk_pkg.n2");
	let (commit_o, _) = ped_commit_with_random(&o, r_o, n2, &pk.pk_kzg.lag_all_n2, pk.pk_kzg.zv_s1_n2);
	let (commit_query, _) = ped_commit_with_random(&query_table, r_query, n2, &pk.pk_kzg.lag_all_n2, pk.pk_kzg.zv_s1_n2);
	let (r_u, commit_u, _) = ped_commit(&vec_u, n2, &pk.pk_kzg.lag_all_n2, pk.pk_kzg.zv_s1_n2);
	let (r_v, commit_v, _) = ped_commit(&vec_v, n2, &pk.pk_kzg.lag_all_n2, pk.pk_kzg.zv_s1_n2);
	if b_perf {log_perf(LOG1, "------ compute com_o, com_u", &mut timer);}

	//3. fold prove
	let alpha = hash::<PE::Fr>(&to_vecu8(
		&vec![commit_query, commit_o, commit_u, commit_v]
	));
	let (prf_fold, _vec_r, vec_commit_query) = fold_prove_cq(&pk.pk_kzg,
		&vec![pk.cq_aux_t1.clone(), pk.cq_aux_t2.clone()],  //note clone of Arc
		&vec![pk.t1.clone(), pk.t2.clone()], //clone of Arc not expensive 
		&vec![MyArc::new(vec_u.clone()), MyArc::new(vec_v.clone())], alpha);
	if b_perf {log_perf(LOG1, "------ fold prove cq", &mut timer);}

	//4. range proof, for vec of query-vec_u and vec_v-query
	let query_u = vec_u.into_par_iter().zip(query_table.into_par_iter()).map(|(x,y)|
		*y-x).collect::<Vec<PE::Fr>>();
	let v_query = vec_v.into_par_iter().zip(query_table.into_par_iter()).map(|(x,y)|
		x-*y).collect::<Vec<PE::Fr>>();
	let r_q_u = r_query  - r_u ;
	let r_v_q = r_v - r_query;
	//let commit_q_u = q_u_s + pk.pk_kzg.zv_s1_n2.mul(r_q_u).into_affine();
	//let commit_v_q = v_q_s + pk.pk_kzg.zv_s1_n2.mul(r_v_q).into_affine(); 
	let prf_rng = prove_rng(&pk.pk_rng, &query_u, r_q_u, &v_query, r_v_q); 
	if b_perf {log_perf(LOG1, "------ range proof ", &mut timer);}

	//5. compute poly q_o to prove o is boolean
	let mut rng = gen_rng_from_seed(gen_seed());
	let coset = PE::Fr::rand(&mut rng);
	let mut coef_o = compute_coefs_of_lag_points(&o, r_o);
	extend_vec_to_size(&mut coef_o, 2*n2);
	let v_o = coefs_to_evals(&coef_o, coset);
	let v_z_h = eval_z_h_over_2n(n2, coset);
	let one = PE::Fr::one();
	assert!(v_o.len()==v_z_h.len(), "v_o.len != v_z_h.len");
	let v_qo = v_o.into_par_iter().zip(v_z_h.into_par_iter()).map(|(vo,vz)|
		vo * (vo - one) / vz).collect::<Vec<PE::Fr>>(); 
	let coef_qo = evals_to_coefs(&v_qo, coset);
	if b_test {assert!(get_degree(&coef_qo)<=n2+1, "qo highest degree>n2+1");}
	if b_perf {log_perf(LOG1, "------ compute q_o", &mut timer);}

	//6. compute different polynomial d and its inverse v, and quotient poly q
	let d = query_u; 
	let v: Vec<PE::Fr> = d.clone().into_par_iter().map(|x|
		if x.is_zero() {one} else {x.inverse().unwrap()}).collect::<_>();

	//7. compute commits of o, d, v, q
	let coset = PE::Fr::rand(&mut rng);
	let rs = rand_arr_field_ele::<PE::Fr>(2, gen_seed());
	let (r_d, r_v) = (rs[0], rs[1]);
	let n3 = 4 * n2;
	let mut coef_d = compute_coefs_of_lag_points(&d, r_d); 
	let mut coef_v = compute_coefs_of_lag_points(&v, r_v); 
	extend_vec_to_size(&mut coef_d, n3);
	extend_vec_to_size(&mut coef_v, n3);
	extend_vec_to_size(&mut coef_o, n3);
	let v_o = coefs_to_evals(&coef_o, coset);
	let v_d = coefs_to_evals(&coef_d, coset);
	let v_v = coefs_to_evals(&coef_v, coset);
	let v_z_h = eval_z_h_over_custom_n(n2, n3, coset);
	assert!(v_o.len()==n3 && v_d.len()==n3 && v_v.len()==n3 &&
		v_z_h.len()==n3, "ERROR in len");

	let v_q = (0..n3).into_par_iter().map(|i|
		(v_o[i]*v_d[i] + (one - v_o[i])*(v_v[i]*v_d[i] - one))/v_z_h[i]).
		collect::<Vec<PE::Fr>>();
	let coef_q = evals_to_coefs(&v_q, coset); 

	if b_test{
		assert!(get_degree(&coef_o)<=n2, "coef_o degree:{}>n2+1: {} ERR", get_degree(&coef_o), n2+1);
		assert!(get_degree(&coef_d)<=n2, "coef_o degree:{}>n2+1: {} ERR", get_degree(&coef_d), n2+1);
		assert!(get_degree(&coef_v)<=n2, "coef_o degree:{}>n2+1: {} ERR", get_degree(&coef_v), n2+1);
		assert!(get_degree(&coef_q)<=3*n2, "coef_o degree:{}>3*n2: {} ERR", get_degree(&coef_v), 3*n2);
		let s = PE::Fr::from(123123123u64);
		let s_o = eval_coefs_at(&coef_o, s);
		let s_d = eval_coefs_at(&coef_d, s);
		let s_v = eval_coefs_at(&coef_v, s);
		let s_q = eval_coefs_at(&coef_q, s);
		let s_z = s.pow(&[n2 as u64]) - one;
		let lhs = s_o*s_d + (one-s_o)*(s_v * s_d - one);
		let rhs = s_z * s_q;
		assert!(lhs==rhs, "FAILED self-test on coef_q");
	}

	//8. send over o_gamma, d_gamma, v_gamma, q_gamm
	let commit_qo= vmsm(&pk.pk_kzg.vec_g1, &coef_qo[0..n2+1].to_vec());
	let commit_d= vmsm(&pk.pk_kzg.vec_g1, &coef_d[0..n2+1].to_vec());
	let commit_poly_v = vmsm(&pk.pk_kzg.vec_g1, &coef_v[0..n2+1].to_vec());
	let commit_q = vmsm(&pk.pk_kzg.vec_g1, &coef_q[0..3*n2+1].to_vec());
	let g1 = PE::G1Affine::prime_subgroup_generator();
	let mut bytes_prf_rng:Vec<u8> = vec![];
	RngProof::<PE>::serialize(&prf_rng, &mut bytes_prf_rng).unwrap();
	let mut bytes_prf_fold:Vec<u8> = vec![];
	CqProof::<PE>::serialize(&prf_fold, &mut bytes_prf_rng).unwrap();
	let mut vbs = to_vecu8(&vec![g1.mul(alpha).into_affine(), 
			vec_commit_query[0],
			vec_commit_query[1], commit_d, commit_poly_v, commit_q, commit_qo]);
	vbs.append(&mut bytes_prf_rng);
	vbs.append(&mut bytes_prf_fold);
	let gamma = hash::<PE::Fr>(&vbs);
	let gamma_o = eval_coefs_at(&coef_o, gamma);
	let gamma_d = eval_coefs_at(&coef_d, gamma);
	let gamma_v = eval_coefs_at(&coef_v, gamma);
	let gamma_q = eval_coefs_at(&coef_q, gamma);
	let gamma_qo = eval_coefs_at(&coef_qo, gamma);
	let gamma_z = gamma.pow(&[n2 as u64]) - one;
	if b_test{
		let lhs = gamma_o*gamma_d + (one-gamma_o)*(gamma_v * gamma_d - one);
		let rhs = gamma_z * gamma_q;
		assert!(lhs==rhs, "failed gamma values!");
	}

	//9. prf_gamma (kzg open proof for gamma_o, .....) 
	let eta = hash::<PE::Fr>(&to_vecu8(&vec![gamma, gamma_o, gamma_qo,
		gamma_d, gamma_v, gamma_q])); 		
	let v = gamma_o + gamma_qo * eta
		 + gamma_d * eta * eta + 
		gamma_v * eta * eta * eta + gamma_q * eta * eta * eta * eta;
	let zero = PE::Fr::zero();
	let lhs_h = get_poly(coef_o) 
		+ &get_poly(vec![eta]) * &get_poly(coef_qo)
		+ &get_poly(vec![eta*eta]) * &get_poly(coef_d)
		+ &get_poly(vec![eta*eta*eta]) * &get_poly(coef_v)
		+ &get_poly(vec![eta*eta*eta*eta]) * &get_poly(coef_q)
		+ get_poly(vec![zero - v]);
	let dividor = get_poly(vec![zero-gamma, one]); //s-t
	let (h, remainder) = old_divide_with_q_and_r(&lhs_h, &dividor);
	if b_test {assert!(remainder.is_zero(), "remainder is not zero");}
	let coefs_h = h.coeffs;
	let prf_gamma = vmsm(&pk.pk_kzg.vec_g1, &coefs_h);
	if b_perf {log_perf(LOG1, "------ weighted kzg proof", &mut timer);}

	//10. return
	(
		commit_query, commit_o,
		PnLookupProof{commit_u: commit_u, commit_v: commit_v,
			prf_fold: prf_fold, 
			commit_query_table1: vec_commit_query[0],
			commit_query_table2: vec_commit_query[1],
			prf_rng: prf_rng, 
			commit_d: commit_d, commit_poly_v:commit_poly_v,commit_q: commit_q,
			commit_qo: commit_qo,
			gamma_o: gamma_o, gamma_d: gamma_d, gamma_v: gamma_v, 
			gamma_q: gamma_q, gamma_qo: gamma_qo,
			prf_gamma: prf_gamma
		}, o
	)
} 

/// verify the proof
pub fn verify_pn_lookup<PE:PairingEngine>(
	vk: &PnLookupVerifierKey<PE>,
	commit_query: PE::G1Affine, 
	commit_o: PE::G1Affine, 
	prf: &PnLookupProof<PE>)-> bool where
<<PE as PairingEngine>::G1Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G1Affine, Scalar=<<PE as PairingEngine>::G1Affine as AffineCurve>::ScalarField>,
 <<PE as PairingEngine>::G2Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G2Affine, Scalar=<<PE as PairingEngine>::G2Affine as AffineCurve>::ScalarField> {
	//1. compute alpha
	let b_perf = false;
	let mut timer = Timer::new();
	if b_perf {println!("verify_pn_lookup:");}
	let alpha = hash::<PE::Fr>(&to_vecu8(
		&vec![commit_query, commit_o, prf.commit_u, prf.commit_v]
	));
	if b_perf {log_perf(LOG1, "-- hash alpha", &mut timer);}

	//2. check folded proof
	let commit_lookup_table = vk.commit_t1 
		+ vk.commit_t2.mul(alpha).into_affine(); 
	let commit_query_table = prf.commit_query_table1 + prf.commit_query_table2.mul(alpha).into_affine();
	let b1 = verify_cq(&vk.vk_kzg, commit_lookup_table, commit_query_table, &prf.prf_fold);
	if b_perf {log_perf(LOG1, "-- verify cq proof", &mut timer);}

	//3. check the range proof
	let commit_t_u = (commit_query.into_projective() - prf.commit_u.into_projective()).into_affine();
	let commit_v_t = (prf.commit_v.into_projective() - &commit_query.into_projective()).into_affine();
	let b2 = verify_rng(&vk.vk_rng, commit_t_u, commit_v_t, &prf.prf_rng);
	if b_perf {log_perf(LOG1, "-- verify range proof", &mut timer);}

	//4. compute gamma
	let g1 = PE::G1Affine::prime_subgroup_generator();
	let mut bytes_prf_rng:Vec<u8> = vec![];
	RngProof::<PE>::serialize(&prf.prf_rng, &mut bytes_prf_rng).unwrap();
	let mut bytes_prf_fold:Vec<u8> = vec![];
	CqProof::<PE>::serialize(&prf.prf_fold, &mut bytes_prf_rng).unwrap();
	let mut vbs = to_vecu8(&vec![g1.mul(alpha).into_affine(), 
			prf.commit_query_table1, prf.commit_query_table2,
			prf.commit_d, prf.commit_poly_v, prf.commit_q, prf.commit_qo]);
	vbs.append(&mut bytes_prf_rng);
	vbs.append(&mut bytes_prf_fold);
	let gamma = hash::<PE::Fr>(&vbs);
	if b_perf {log_perf(LOG1, "-- compute gamma", &mut timer);}

	//5. verify array o is boolean and validity of q
	let one = PE::Fr::one();
	let gamma_z = gamma.pow(&[vk.n2 as u64]) - one;
	let b3 = prf.gamma_o*(prf.gamma_o-one) == prf.gamma_qo * gamma_z;
	let b4 = prf.gamma_o * prf.gamma_d + (one-prf.gamma_o)*(prf.gamma_v * prf.gamma_d - one) == gamma_z * prf.gamma_q;

	//6. batch verify kzg 
	let eta = hash::<PE::Fr>(&to_vecu8(&vec![gamma, prf.gamma_o, 
		prf.gamma_qo, prf.gamma_d, prf.gamma_v, prf.gamma_q])); 		
	let v = prf.gamma_o + prf.gamma_qo * eta + prf.gamma_d * eta * eta + 
		prf.gamma_v * eta * eta * eta + prf.gamma_q * eta * eta * eta * eta;
	let lhs1 = commit_o + 
			prf.commit_qo.mul(eta).into_affine() +
			prf.commit_d.mul(eta*eta).into_affine() + 
			prf.commit_poly_v.mul(eta*eta*eta).into_affine() + 
			prf.commit_q.mul(eta*eta*eta*eta).into_affine();
	let lhs = lhs1.into_projective()  - g1.mul(v) 
		+ prf.prf_gamma.mul(gamma);
	let g2 = PE::G2Affine::prime_subgroup_generator();
	let b5 = PE::pairing(lhs, g2) == PE::pairing(prf.prf_gamma, vk.vk_kzg.s2);
	if b_perf {log_perf(LOG1, "-- verify weighted kzg proof", &mut timer);}

	//3. return
	println!("b1: {}, b2: {}, b3: {}, b4: {}, b5: {}", b1, b2, b3, b4, b5);
	b1 && b2 && b3 && b4 && b5
}

// ------------ Utility Functions ----------------
/// return the highest degree that is not zero
fn get_degree<F:Field>(v: &Vec<F>)->usize{
	let mut deg = v.len()-1;
	loop{
		if deg==0 {break;}
		if !v[deg].is_zero() {break;}
		deg -= 1;
	}

	deg
}

/// generate 2^bits (assuming bits in range)
fn gen_pow2<F:PrimeField>(bits: usize)->F{
	let mut bit_arr = vec![false; bits + 1];
	let n = bit_arr.len()-1;
	bit_arr[n] = true;
	F::from_bigint(F::BigInt::from_bits_le(&bit_arr)).unwrap()
}

/// produce location array s, s.t. t1_{s_i}<=t<=t2_{s_i}, 
/// and array o s.t. o_i = t==t1_{s_i}
/// assumption is that (t1, t2) is the sorted tables generated
/// for some sorted vector of elements and all elements of t in range.
fn gen_output<F:PrimeField>(t1: &Vec<F>, t2: &Vec<F>, t: &Vec<F>)
-> (Vec<usize>, Vec<F>){
	//1. assert data
	let b_test = false;
	let n = t1.len();
	assert!(n==t2.len(), "t2.len()!=n");
	let n2 = t.len();
	assert!(n.is_power_of_two() && n2.is_power_of_two(), "not pow of 2");

	//2. collect index first
	let arr_s:Vec<usize> = (0..n2).into_par_iter().map(|i|
		bin_search(t1, t[i])).collect();
	let arr_o: Vec<F> = (0..n2).into_par_iter().map(|i|
		if t1[arr_s[i]] == t[i] {F::one()} else {F::zero()}
	).collect();
	if b_test{
		for i in 0..n2{
			assert!(t1[arr_s[i]]<=t[i], "t1[arr_s[i]]>t[i]!");
			assert!(t2[arr_s[i]]>t[i], "t2[arr_s[i]]<=t[i]!");
		}
	}

	//3. return
	(arr_s, arr_o)
}

/// return the position of v (let it be k) in t1 where 
/// t1_k <=v <=t1_{k+1}, assuming t1 is sorted in ascending order
/// and starts with 0
fn bin_search<F:PrimeField>(t1: &Vec<F>, v: F) -> usize{
	let b_test = true;
	let s_res = t1.binary_search(&v);
	let res = match s_res{
		Ok(idx) => idx,
		Err(idx) => idx-1
	};

	if b_test{
		assert!(t1[res]<=v, "ERR: v<t1[idx]!");
		if res<t1.len()-1{
			assert!(t1[res+1]>v, "ERR: v>t1[idx+1]!");
		}else{
			assert!(v>t1[res-1], "ERR: v out of bound");
		}
	}	
	
	res	
}

#[cfg(test)]
mod tests{
	extern crate ark_bls12_381;
	extern crate acc;
	extern crate ark_ec;
	extern crate ark_ff;
	use self::ark_bls12_381::Bls12_381;
	type Fr381 = ark_bls12_381::Fr;
	type PE381 = Bls12_381;
	type G1_381= ark_bls12_381::G1Affine;

	//use izpr::serial_poly_utils::{default_trapdoor, setup_kzg};
	use self::acc::tools::{rand_arr_field_ele};
	use izpr::serial_cq::{ped_commit,gen_seed};
	use izpr::serial_rng::{rand_arr_fe,rand_arr_fe_with_seed};
	use izpr::serial_pn_lookup::{setup_pn_lookup, prove_pn_lookup, verify_pn_lookup};
	use izpr::serial_poly_utils::{vmsm,setup_kzg, default_trapdoor,precompute_lag_group};
	use self::ark_ec::{AffineCurve,ProjectiveCurve};
	use self::ark_ff::{Field};

	#[test]
	pub fn test_pn_lookup(){
		let bits = 32; 
		let unit_bits = 8;
		let num_ele = 16;
		let n2 = 4;
		let trap = default_trapdoor::<PE381>();
		let seed = gen_seed();
		let mut t = rand_arr_fe::<Fr381>(num_ele-1, bits);
		t.sort();
		let mut query_table = rand_arr_fe_with_seed::<Fr381>(n2, 
			bits, gen_seed()+1);
		query_table[1] = t[3];
		let (pk, vk) = setup_pn_lookup::<PE381>(bits,unit_bits, num_ele, n2, &t);
		let arr_r = rand_arr_fe::<Fr381>(3, bits);
		let (com_t, com_o, prf, _) = prove_pn_lookup(&pk, &query_table, 
			arr_r[0], arr_r[1]);
		let bres = verify_pn_lookup(&vk, com_t, com_o, &prf);
		assert!(bres, "pn_lookup failed");	
	}
}
