/*
    Copyright Dr. Xiang Fu

    Author: Xiang Fu
    All Rights Reserved.
    Created: 08/16/2023

	This files encodes the zero knowledge batched range proof
	based on cq. It includes an implementation of
	quasi-linear NIZK (QA-NIZK) of appendix D of LegoSnark.
	Note that the prove_function proves TWO instances of
	range proofs simultaneously. The main motivation is to save
	one copy of the lookup argument proof.

	NOTE that: the zk_range proof is pairing based and needs
	trusted setup.
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
extern crate rand;

use self::rayon::prelude::*; 
use self::ark_std::path::{Path};
use self::acc::profiler::config::{LOG1};
use self::acc::tools::{log_perf,Timer};
use self::ark_ec::{AffineCurve, PairingEngine,ProjectiveCurve};
use self::ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Read, SerializationError, Write};
use self::ark_ec::msm::{VariableBaseMSM};
use self::ark_ff::{One,Zero,PrimeField,BigInteger,Field};
use izpr::serial_poly_utils::{KzgProverKey,KzgVerifierKey,vmsm,default_trapdoor,precompute_lag_group,setup_kzg};
use izpr::serial_cq::{gen_seed,preprocess_cq,CqAux,prove_cq,CqProof,verify_cq};
use self::acc::tools::{rand_arr_field_ele,gen_rng_from_seed,from_vecu8,to_vecu8, write_vecu8, read_vecu8};


/// The Prover Key of QA-NIZK
//#[derive(Clone)]
#[derive(Clone,CanonicalSerialize,CanonicalDeserialize)]
pub struct QANizkProverKey<PE:PairingEngine>{
	/// number of rows
	pub rows: usize,
	/// number of columsn
	pub cols: usize,
	/// prover key p
	pub vec_p: Vec<PE::G1Affine>,
}

/// The Verifier Key of QA-NIZK
//#[derive(Clone)]
#[derive(Clone,CanonicalSerialize,CanonicalDeserialize)]
pub struct QANizkVerifierKey<PE:PairingEngine>{
	/// number of rows
	pub rows: usize,
	/// number of columsn
	pub cols: usize,
	/// verifier key c
	pub vec_c: Vec<PE::G2Affine>,
	/// verifier key a in g2
	pub a_g2: PE::G2Affine
}

/// QANizkProof 
#[derive(Clone,CanonicalSerialize,CanonicalDeserialize)]
pub struct QANizkProof<PE:PairingEngine>{
	pub prf: PE::G1Affine,
}

/// The Prover Key of Batched Range Proof
//#[derive(Clone)]
#[derive(Clone,CanonicalSerialize,CanonicalDeserialize)]
pub struct RngProverKey<PE:PairingEngine>{
	/// B in the paper. The range proof is for [0, 2^log_bound)
	pub log_bound: usize,
	/// the chunk size (e.g., 20 bits) out of 160 bits of B
	pub log_unit_bound: usize,
	/// prover key of qa_nizk
	pub pk_qa_nizk:	QANizkProverKey<PE>,
	/// prover key of KZG for cq
	pub cq_kzg_pk: KzgProverKey<PE>,
	/// aux infor for cq (lookup table for [0, 2^log_unit_bound))
	pub cq_aux: CqAux<PE>,
	/// qa_nizk matrix
	pub mat: Vec<Vec<PE::G1Affine>>,
	/// lookup table
	pub lookup_table: Vec<PE::Fr>,
}

/// The Verifier Key
#[derive(Clone,CanonicalSerialize,CanonicalDeserialize)]
pub struct RngVerifierKey<PE:PairingEngine>{
	/// verifier key of qa_nizk
	pub vk_qa_nizk:	QANizkVerifierKey<PE>,
	/// kzg verifier key for cq
	pub cq_kzg_vk: KzgVerifierKey<PE>,
	/// commit_lookup_table
	pub commit_lookup_table: PE::G2Affine
}

/// Range Proof
#[derive(Clone,CanonicalSerialize,CanonicalDeserialize)]
pub struct RngProof<PE:PairingEngine>{
	/// proof of qa_nizk
	pub prf_qa_nizk: QANizkProof<PE>,
	/// commitment to witness
	pub commit_w: PE::G1Affine,
	/// cq proof
	pub prf_cq: CqProof<PE>,
}

/// generate the prover and verifier keys for QANizk
pub fn setup_qa_nizk<PE:PairingEngine>(matrix: &Vec<Vec<PE::G1Affine>>)
	-> (QANizkProverKey<PE>, QANizkVerifierKey<PE>) where
<<PE as PairingEngine>::G1Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G1Affine, Scalar=<<PE as PairingEngine>::G1Affine as AffineCurve>::ScalarField>,
 <<PE as PairingEngine>::G2Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G2Affine, Scalar=<<PE as PairingEngine>::G2Affine as AffineCurve>::ScalarField> {
	//1. generate random vector k
	let rows = matrix.len();
	let cols = matrix[0].len();
	let vec_k = rand_arr_field_ele::<PE::Fr>(rows, gen_seed());
	let a = rand_arr_field_ele::<PE::Fr>(1, gen_seed()+2)[0];
	let rot_m = rotate_matrix(&matrix);
	assert!(rot_m.len()==cols, "rot_m.len()!=cols");
	let g2 = PE::G2Affine::prime_subgroup_generator();
	let vec_c = vec_k.par_iter().map(|x| g2.mul(*x * a).into_affine())
		.collect::<Vec<PE::G2Affine>>();
	assert!(vec_c.len()==rows, "vec_c.len()!=rows");
	let a_g2 = g2.mul(a).into_affine();
	let vec_p = rot_m.par_iter().map(|row| vmsm(row, &vec_k)).
		collect::<Vec<PE::G1Affine>>();
	assert!(vec_p.len()==cols, "vec_p.len()!=cols");

	(QANizkProverKey{rows: rows, cols: cols, vec_p: vec_p},
	 QANizkVerifierKey{rows: rows, cols: cols, vec_c: vec_c, a_g2: a_g2})
}

/// generate a proof of QZNIZK for M w = x
pub fn prove_qa_nizk<PE:PairingEngine>(
	pk: &QANizkProverKey<PE>, 
	w: &Vec<PE::Fr>) -> QANizkProof<PE> where
<<PE as PairingEngine>::G1Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G1Affine, Scalar=<<PE as PairingEngine>::G1Affine as AffineCurve>::ScalarField>,
 <<PE as PairingEngine>::G2Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G2Affine, Scalar=<<PE as PairingEngine>::G2Affine as AffineCurve>::ScalarField> {
	assert!(w.len()==pk.cols, "w.len()!=pk.cols");
	let prf = vmsm(&pk.vec_p, &w);
	QANizkProof{prf: prf}
} 

/// verify the QANIZK proof
pub fn verify_qa_nizk<PE:PairingEngine>(
	vk: &QANizkVerifierKey<PE>,
	x: &Vec<PE::G1Affine>,
	prf: &QANizkProof<PE>)-> bool where
<<PE as PairingEngine>::G1Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G1Affine, Scalar=<<PE as PairingEngine>::G1Affine as AffineCurve>::ScalarField>,
 <<PE as PairingEngine>::G2Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G2Affine, Scalar=<<PE as PairingEngine>::G2Affine as AffineCurve>::ScalarField> {
	assert!(x.len()==vk.rows, "x.len()!=vk.rows");
	let lhs: PE::Fqk= x.par_iter().zip(vk.vec_c.par_iter()).
		map(|(a,b)| PE::pairing(*a,*b)).product();
	let rhs = PE::pairing(prf.prf, vk.a_g2);
	lhs == rhs
}

/// generate the prover and verifier keys
/// we will try to prove each element in range [0, 2^log_bound)
/// a lookup table is generated for [0, 2^log_unit_bound)
/// e.g., log_bound = 160 (bit), and unit_bound = 20 (bit)
/// requires that log_bound %log_unit_bound==0
/// n: the number of elements in the batched range proof
/// NOTE: there are two batched range proof to prove, also
/// it should be run by a trusted setup (trapdoor.s is used)
pub fn setup_rng<PE:PairingEngine>(
	log_bound: usize,
	log_unit_bound: usize,
	n: usize
	) -> (RngProverKey<PE>, RngVerifierKey<PE>) where
<<PE as PairingEngine>::G1Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G1Affine, Scalar=<<PE as PairingEngine>::G1Affine as AffineCurve>::ScalarField>,
 <<PE as PairingEngine>::G2Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G2Affine, Scalar=<<PE as PairingEngine>::G2Affine as AffineCurve>::ScalarField> {
	//1. check inputs
	let n1 = log_bound;
	let n2 = log_unit_bound;
	let b_perf = true;
	assert!(n1%n2==0, "log_bound%log_unit_bound!=0");
	assert!(n.is_power_of_two(), "n is not power of 2");
	//assert!(n1.is_power_of_two(), "n1 is not power of 2");
	//assert!(n2.is_power_of_two(), "n2 is not power of 2");
	let k = n1/n2; //k units comprise one element (e.g, 8 20-bits -> 160bit)
	let one = PE::Fr::one();
	let mut timer = Timer::new();
	let mut timer2= Timer::new();
	let b_perf = true;

	//1.5 load
	let prv_key_file = format!("cache/rng_prover_key_{}_{}_{}.dat", log_unit_bound, log_bound, n);
	let ver_key_file = format!("cache/rng_verifier_key_{}_{}_{}.dat", log_unit_bound, log_bound, n);
	let b_cache = Path::new(&prv_key_file).exists();
	if b_cache{
		let mut bytes_prover_key = read_vecu8(&prv_key_file);
		let mut b1 = &bytes_prover_key[..];
		let pkey = RngProverKey::<PE>::deserialize(&mut b1).unwrap();
		let mut bytes_vierifer_key = read_vecu8(&ver_key_file);
		let mut b2 = &bytes_vierifer_key[..];
		let vkey = RngVerifierKey::<PE>::deserialize(&mut b2).unwrap();
		if b_perf {log_perf(LOG1, &format!("setup_rng: (QUICK) {}:", n1), &mut timer2);}
		return (pkey, vkey); 
	}

	//2. create M: 
	// row1 -> for commit_a1
	// row2 -> for commit_a2
	// row3 -> for all_commit of all witness
	let s = default_trapdoor::<PE>().s;
	let g1 = PE::G1Affine::prime_subgroup_generator();
	let g1_0 = g1.mul(PE::Fr::zero()).into_affine();
	let z_v_s = g1.mul(s.pow(&[n as u64]) - one).into_affine();
	let z_x_s = g1.mul(s.pow(&[(2*k*n) as u64]) - one).into_affine();
	let arr_k2pow = build_pow2s::<PE::Fr>(k, n2);
	let lag_bases = precompute_lag_group::<PE::G1Affine>(n,s); 
	let mut k2_lag = lag_bases.par_iter().map(|lag| 
		arr_k2pow.par_iter().map(|pow2| 
			lag.mul(*pow2).into_affine()).collect::<Vec<PE::G1Affine>>()).
			flatten().
			collect::<Vec<PE::G1Affine>>();
	let vec0 = vec![g1_0; k*n];
	let mut row1 = k2_lag.clone();
	row1.append(&mut vec0.clone());
	row1.append(&mut vec![z_v_s, g1_0, g1_0]);

	let mut row2 = vec0;
	row2.append(&mut k2_lag);
	row2.append(&mut vec![g1_0, z_v_s, g1_0]);
	if b_perf{log_perf(LOG1, "--** row1 and row 2", &mut timer);}

	let mut row3 = precompute_lag_group::<PE::G1Affine>(2*k*n, s);
	if b_perf{log_perf(LOG1, "--** row3", &mut timer);}
	row3.append(&mut vec![g1_0, g1_0, z_x_s]); 
	let mat = vec![row1, row2, row3];
	let (pk_qa_nizk, vk_qa_nizk) = setup_qa_nizk::<PE>(&mat);
	if b_perf{log_perf(LOG1, "--** set up qanizk", &mut timer);}

	//2. build the lookup table (cq protocol) keys
	let t_n = 1<<n2;
	let lookup_table = (0..t_n).into_par_iter()
		.map(|x| PE::Fr::from(x as u64))
		.collect::<Vec<PE::Fr>>();
	let trap = default_trapdoor::<PE>();
	assert!(t_n>2*k*n,"lookup_table: {} <= 2*k*n: {} too small!. k: {}, n: {}", t_n, 2*k*n, k, n);
	let (cq_kzg_pk, cq_kzg_vk) = setup_kzg::<PE>(t_n, 2*k*n, &trap);
	if b_perf{log_perf(LOG1, "--** setup_kzg", &mut timer);}
	let cq_aux = preprocess_cq::<PE>(&cq_kzg_pk, &lookup_table);
	let commit_t2 = cq_aux.commit_t2.clone();
	if b_perf{log_perf(LOG1, "--** cq_aux qanizk", &mut timer);}


	//3. return
	
	let pkey = 	RngProverKey{log_bound: log_bound, log_unit_bound: log_unit_bound,
			pk_qa_nizk: pk_qa_nizk, cq_kzg_pk: cq_kzg_pk, cq_aux: cq_aux,
			mat: mat, lookup_table: lookup_table};
	let vkey =	RngVerifierKey{vk_qa_nizk: vk_qa_nizk, cq_kzg_vk: cq_kzg_vk,
			commit_lookup_table: commit_t2};

	let mut b1:Vec<u8> = vec![];
	pkey.serialize(&mut b1);
	write_vecu8(&b1, &prv_key_file);
	let mut b2:Vec<u8> = vec![];
	vkey.serialize(&mut b2);
	write_vecu8(&b2, &ver_key_file);
	if b_perf {log_perf(LOG1, &format!("setup_rng (FULL): {}:", n1), &mut timer2);}

	(pkey, vkey)

}

/// generate a proof that each element of table two tables
/// t1, t2 belongs to
/// range [0, 2^log_bound), where log_bound is defined in prover key
/// commit_t is defined as \sum_{i=1}^n t_i [L_i(s)]_1 + r*[z_[n](s)]_1
/// We use parameter lag_key to represent array [ ... [L_i(s)]_1 ... [z_[n]](s)]
/// lag_key is contained in the setup information.
/// r is the random nonce for gennerating commit_t
/// return the range proof
pub fn prove_rng<PE:PairingEngine>(
	pk: &RngProverKey<PE>, 
	t1: &Vec<PE::Fr>, r_t1: PE::Fr,
	t2: &Vec<PE::Fr>, r_t2: PE::Fr)
	-> RngProof<PE> where
<<PE as PairingEngine>::G1Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G1Affine, Scalar=<<PE as PairingEngine>::G1Affine as AffineCurve>::ScalarField>,
 <<PE as PairingEngine>::G2Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G2Affine, Scalar=<<PE as PairingEngine>::G2Affine as AffineCurve>::ScalarField> {
	//1. produce the lookup argument proof
	let b_perf = true;
	let mut timer = Timer::new();
	let query_table = build_query_table(&pk, t1, t2);
	if b_perf {log_perf(LOG1, "-- range proof step 1: build query table", &mut timer); }
	let (prf_cq, r_w, commit_w) = prove_cq(&pk.cq_kzg_pk, &pk.cq_aux, 
		&pk.lookup_table, &query_table);
	if b_perf {log_perf(LOG1, "-- range proof step 2: prove cq", &mut timer);}


	//1. produce the witness
	let w = build_witness(pk, t1, r_t1, t2, r_t2, r_w);
	if b_perf {log_perf(LOG1, "-- range proof step 3: build wit", &mut timer); }

	//2. produce the qa_nizk proof
	let prf_qa_nizk = prove_qa_nizk(&pk.pk_qa_nizk, &w); 
	if b_perf {log_perf(LOG1, "-- range proof step 4: prove_qa_nizk", &mut timer);} 

	RngProof{ prf_qa_nizk: prf_qa_nizk, commit_w: commit_w, prf_cq: prf_cq }
} 

/// for debugging purpose: produce the commit_t1 and commit_t2
/// using the matrix M in prover key
pub fn produce_rng_commits<PE:PairingEngine>(
	pk: &RngProverKey<PE>, 
	t1: &Vec<PE::Fr>, r_t1: PE::Fr,
	t2: &Vec<PE::Fr>, r_t2: PE::Fr, r_w: PE::Fr)
-> (PE::G1Affine,PE::G1Affine) where
<<PE as PairingEngine>::G1Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G1Affine, Scalar=<<PE as PairingEngine>::G1Affine as AffineCurve>::ScalarField>,
 <<PE as PairingEngine>::G2Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G2Affine, Scalar=<<PE as PairingEngine>::G2Affine as AffineCurve>::ScalarField> {
	let w = build_witness(pk, t1, r_t1, t2, r_t2, r_w);
	let res = (0..2).into_par_iter().map(|x|
		vmsm(&pk.mat[x], &w)).collect::<Vec<PE::G1Affine>>();
	
	( res[0], res[1] )
}

/// verify the proof
pub fn verify_rng<PE:PairingEngine>(
	vk: &RngVerifierKey<PE>,
	commit_t1: PE::G1Affine, 
	commit_t2: PE::G1Affine, 
	prf: &RngProof<PE>)-> bool where
<<PE as PairingEngine>::G1Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G1Affine, Scalar=<<PE as PairingEngine>::G1Affine as AffineCurve>::ScalarField>,
 <<PE as PairingEngine>::G2Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G2Affine, Scalar=<<PE as PairingEngine>::G2Affine as AffineCurve>::ScalarField> {
	//1. qa-nizk check
	let arr_x = vec![commit_t1, commit_t2, prf.commit_w];
	let b1 = verify_qa_nizk(&vk.vk_qa_nizk, &arr_x, &prf.prf_qa_nizk);
	
	//2. check cq
	let b2 = verify_cq(&vk.cq_kzg_vk, vk.commit_lookup_table, 
		prf.commit_w, &prf.prf_cq);

	b1 && b2
}

// ------------- Utility Functions Below ---------------
/// rotate a matrix
fn rotate_matrix<G: AffineCurve>(m: &Vec<Vec<G>>) -> Vec<Vec<G>>{
    (0..m[0].len())
        .map(|i| m.par_iter().map(|inner| inner[i].clone()).collect::<Vec<G>>())
        .collect()
}

/// assuming v is in bound [0, 2^b), and b = k * b2
/// devide v into k chunks of b2 bits numbers
fn fe_to_chunks<F:PrimeField>(v: F, b: usize, b2: usize)->Vec<F>{
	assert!(b%b2==0, "b%b2!=0");
	let mut vbits = v.into_bigint().to_bits_le();
	let mut lead_bit = vbits.len()-1;
	while !vbits[lead_bit]{
		if lead_bit==0 {break};
		lead_bit-=1;
	}	
	assert!(lead_bit<=b, "vbits.len: {} > b: {}!", lead_bit, b);
	if vbits.len()<b {vbits.append(&mut vec![false; b-vbits.len()]);}
	let k = b/b2;	
	let chunks = (0..k).into_par_iter().map(|i|
		F::from_bigint(F::BigInt::from_bits_le(
			&vbits[i*b2..(i+1)*b2])).unwrap()).
		collect::<Vec<F>>();

	chunks
}

/// build {2^{0u}, ..., 2{ku} in field
fn build_pow2s<F:PrimeField>(k: usize, u: usize)->Vec<F>{
	let f_unit = F::from((1<<u) as u64);
	let mut cur_f = F::one();
	let mut v_units = vec![];
	for _i in 0..k{
		v_units.push(cur_f);	
		cur_f = cur_f * f_unit;
	}

	v_units
}

/// chunks to a single element
/// assuption: each chunk is in range [0, 2^b2]
/// total numbers b/b2, both are multiples of 8
/// pow2s are the power of 2s
fn chunks_to_fe<F:PrimeField>(vec: &Vec<F>, b: usize, b2: usize,
	pow2s: &Vec<F>)->F{
	let k = b/b2;
	assert!(b%b2==0, "b%b2!=0");
	assert!(vec.len()==k, "vec.len!=k");
	assert!(pow2s.len()==k, "pow2s.len!=k");
	assert!(b2<64, "cannot handle more than 64 bits");
	vec.par_iter().zip(pow2s.par_iter())
		.map(|(x,y)| *x * *y).sum()
}

/// returning the fe mod the bits
fn fe_mod<F:PrimeField>(v: F, bits: usize)->F{
	let bits = &v.into_bigint().to_bits_le()[0..bits];
	let res = F::from_bigint(F::BigInt::from_bits_le(&bits)).unwrap();
	
	res
}

/// produce random array of n elements each of bits
pub fn rand_arr_fe<F:PrimeField>(n: usize, bits: usize)->Vec<F>{
	let res = (0..n).into_par_iter().
		map(|x| fe_mod(F::rand(&mut gen_rng_from_seed((x*3717) as u128)),bits) )
		.collect::<Vec<F>>();

	res
}

pub fn rand_arr_fe_with_seed<F:PrimeField>(n: usize, bits: usize, seed: u128)->Vec<F>{
	let res = (0..n).into_par_iter().
		map(|x| fe_mod(F::rand(&mut gen_rng_from_seed((seed*3717 + (x as u128)*57) as u128)),bits) )
		.collect::<Vec<F>>();

	res
}

/// build the witness
fn build_witness<PE:PairingEngine>(
	pk: &RngProverKey<PE>, 
	t1: &Vec<PE::Fr>, r_t1: PE::Fr,
	t2: &Vec<PE::Fr>, r_t2: PE::Fr, r_w: PE::Fr)-> Vec<PE::Fr> where
<<PE as PairingEngine>::G1Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G1Affine, Scalar=<<PE as PairingEngine>::G1Affine as AffineCurve>::ScalarField>,
 <<PE as PairingEngine>::G2Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G2Affine, Scalar=<<PE as PairingEngine>::G2Affine as AffineCurve>::ScalarField> {
	//1. split all elements into array
	let (n1,n2) = (pk.log_bound, pk.log_unit_bound);
	let chunks1 = t1.par_iter().map(|v| fe_to_chunks(*v, n1, n2)).flatten().
		collect::<Vec<PE::Fr>>();
	let mut chunks2 = t2.par_iter().map(|v| fe_to_chunks(*v, n1, n2)).flatten().
		collect::<Vec<PE::Fr>>();

	//2. construct witness w
	let mut w = chunks1;
	w.append(&mut chunks2);
	w.append(&mut vec![r_t1, r_t2, r_w]);
	assert!(w.len()==pk.mat[0].len(), "w.len(): {} !=matrix cols: {}",
		w.len(), pk.mat[0].len());

	w
}

/// build the query table 
fn build_query_table<PE:PairingEngine>(
	pk: &RngProverKey<PE>, 
	t1: &Vec<PE::Fr>, t2: &Vec<PE::Fr>) -> Vec<PE::Fr> where
<<PE as PairingEngine>::G1Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G1Affine, Scalar=<<PE as PairingEngine>::G1Affine as AffineCurve>::ScalarField>,
 <<PE as PairingEngine>::G2Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G2Affine, Scalar=<<PE as PairingEngine>::G2Affine as AffineCurve>::ScalarField> {
	//1. split all elements into array
	let (n1,n2) = (pk.log_bound, pk.log_unit_bound);
	let chunks1 = t1.par_iter().map(|v| fe_to_chunks(*v, n1, n2)).flatten().
		collect::<Vec<PE::Fr>>();
	let mut chunks2 = t2.par_iter().map(|v| fe_to_chunks(*v, n1, n2)).flatten().
		collect::<Vec<PE::Fr>>();

	//2. construct query table 
	let mut w = chunks1;
	w.append(&mut chunks2);

	w
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
	use izpr::serial_cq::{gen_seed,ped_commit};
	use izpr::serial_rng::{setup_qa_nizk, prove_qa_nizk, verify_qa_nizk,setup_rng, fe_to_chunks, chunks_to_fe,build_pow2s,prove_rng, verify_rng, produce_rng_commits, rand_arr_fe,build_witness};
	use izpr::serial_poly_utils::{vmsm,setup_kzg, default_trapdoor,precompute_lag_group};
	use self::ark_ec::{AffineCurve,ProjectiveCurve};
	use self::ark_ff::{Field};

	#[test]
	pub fn test_chunks(){
		let chunks = vec![ //20 bits each
			Fr381::from(0x0FF124 as u64),
			Fr381::from(0x0C1124 as u64),
			Fr381::from(0x013124 as u64),
			Fr381::from(0x020124 as u64),
			Fr381::from(0x012124 as u64),
			Fr381::from(0x0FDCBA as u64),
			Fr381::from(0x0FDCBA as u64),
			Fr381::from(0x0FDCBA as u64)];
		let pow2s = build_pow2s(8, 20); 
		let v = chunks_to_fe(&chunks, 160, 20, &pow2s);
		let chunks2 = fe_to_chunks(v, 160, 20);
		assert!(chunks==chunks2, "fails test chunks ops");
	}
	pub fn test_qa_nizk(){
		let n = 64usize;
		let seed = gen_seed();
		let fe_m = vec![rand_arr_field_ele::<Fr381>(n, seed), 
			rand_arr_field_ele::<Fr381>(n,seed+1)];
		let mut m = vec![];
		let g1 = G1_381::prime_subgroup_generator();
		for i in 0..fe_m.len(){
			m.push( fe_m[i].iter().map(|x| g1.mul(*x).into_affine())
				.collect::<Vec<_>>());
		}
		let w = rand_arr_field_ele::<Fr381>(n, seed+3);
		let x = m.iter().map(|row| vmsm(row, &w)).collect::<Vec<_>>();
		let (pk, vk) = setup_qa_nizk::<PE381>(&m);
		let prf = prove_qa_nizk(&pk, &w);
		let bres = verify_qa_nizk(&vk, &x, &prf);
		assert!(bres, "qa_nizk test failed");
	}

	#[test]
	pub fn test_range_proof(){
		let bits = 32; 
		let unit_bits = 8;
		let num_ele = 4;
		let trap = default_trapdoor::<PE381>();
		let (pk, vk) = setup_rng::<PE381>(bits, unit_bits, num_ele);
		let t1 = rand_arr_fe::<Fr381>(num_ele, bits);
		let t2 = rand_arr_fe::<Fr381>(num_ele, bits);
		let arr_r = rand_arr_fe::<Fr381>(3, bits);
		let (com_t1, com_t2) = produce_rng_commits::<PE381>(&pk, 
			&t1, arr_r[0], &t2, arr_r[1], arr_r[2]); 

		let arr_lag = precompute_lag_group(num_ele, trap.s);  
		let g1 = G1_381::prime_subgroup_generator();
		let zv_s1 = g1.mul(trap.s.pow(&[num_ele as u64])-Fr381::from(1u32))
			.into_affine();
		let (_,_,t1_s) = ped_commit::<G1_381>(&t1, num_ele, &arr_lag, zv_s1);
		let (_,_,t2_s) = ped_commit::<G1_381>(&t2, num_ele, &arr_lag, zv_s1);

		assert!(com_t1==t1_s + zv_s1.mul(arr_r[0]).into_affine(), "com_t1 not right!");
		let prf = prove_rng(&pk, &t1, arr_r[0], &t2, arr_r[1]);
		let bres = verify_rng(&vk, com_t1, com_t2, &prf);
		assert!(bres, "range proof failed");	
	}
}
