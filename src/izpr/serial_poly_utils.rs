/*
    Copyright Dr. Xiang Fu

    Author: Trevor Conley, Xiang Fu
    All Rights Reserved.
    Created: 07/28/2023
	Completed: 08/31/2023
*/
extern crate ark_ff;
extern crate ark_ec;
extern crate ark_poly;
extern crate ark_serialize;
extern crate ark_std;
extern crate ark_bls12_381;
extern crate acc;
extern crate rayon;

use self::rayon::prelude::*; 
use self::acc::poly::serial::{serial_fft};
use self::ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Read, SerializationError, Write};

use self::ark_ff::{FftField};
use self::ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, Polynomial, };
use izpr::serial_poly_utils::acc::poly::common::closest_pow2;
use self::ark_ec::{AffineCurve, PairingEngine, ProjectiveCurve};
use self::ark_ff::{Field, PrimeField, One, Zero};
use self::ark_ec::msm::{FixedBase, VariableBaseMSM};
use self::acc::profiler::config::{LOG1};
use self::acc::tools::{log_perf,Timer};
use izpr::serial_cq::{evals_to_coefs};
use izpr::serial_group_fft2::{serial_group_fft, serial_group_ifft};

/// global trapdoor for debugging
pub const GLOBAL_TRAPDOOR_S: u64 = 2820923481723487u64;




/** 
	Precompute the kzg proofs for T(X) evaluates to a T(\omega^i) for each i.
	This is the algorithm presented in "Fast Amortized KZG Proof" 
 */
pub fn precompute_kzg_proof<G:AffineCurve>(
	table: &Vec<G::ScalarField>, 
	n: usize, 
	kzg_key: &Vec<G>
	)->Vec<G>
	where <G as AffineCurve>::Projective: VariableBaseMSM<MSMBase=G, Scalar=<G as AffineCurve>::ScalarField>{
	assert!(n==table.len(), "n! = table.len()");
	assert!(n.is_power_of_two(), "n not pow of 2");
	let f= evals_to_coefs(&table, G::ScalarField::one());
	let zero = G::ScalarField::zero();

	assert!(kzg_key.len()>=n, "kzg.len()<t.len()");
	let g1_0 = G::prime_subgroup_generator().mul(G::ScalarField::zero()).into_affine();
	let v1= kzg_key[0..n].to_vec();
	let mut v2 = v1.clone();
	v2.reverse();
	let mut vec_s = v2.clone();
	vec_s.append(&mut vec![g1_0; n]);
	let vec_y = serial_group_fft(&vec_s);

	let mut vec_c = vec![zero; n];
	vec_c.append(&mut f[1..n].to_vec());
	vec_c.append(&mut vec![zero]);
	assert!(vec_c.len()==2*n, "vec_v.len !=2*n");
	let mut vec_v = vec_c.clone();
	serial_fft(&mut vec_v);


	let nu = G::ScalarField::get_root_of_unity((2*n) as u64).unwrap();
	let arr_nu = compute_powers::<G::ScalarField>(2*n, nu);
	let vec_u = vec_y.clone().into_par_iter().zip(vec_v.clone().into_par_iter())
		.zip(arr_nu.into_par_iter()).map(|((y,v),nu)|
			y.mul(v * nu).into_affine() ).collect::<Vec<G>>();

	let vec_h = serial_group_ifft(&vec_u);

	//5. build up the KZG proofs from vec_h
	let vec_h = vec_h[0..n].to_vec();
	let kzg_prfs = serial_group_fft(&vec_h);

	kzg_prfs
}

/// compute all L_i(s) for each i. 
pub fn precompute_lags<F:FftField>(
	n: usize, 
	s: F)->Vec<F>{
	assert!(n.is_power_of_two(), "n is not pow of 2");
	//1. compute Z_n(s)
	let one = F::one();
	let z_n_s = s.pow(&[n as u64]) - one;
	let omega = F::get_root_of_unity(n as u64).unwrap();
	let arr_omega = compute_powers::<F>(n, omega);

	//2. compute Z_n'(omega_i)
	// n*(omega^i)^{n-1} = n/omega^i
	let z_n_prime = compute_derive_vanish::<F>(n);

	//3. L_i(s) = Z_n(s)/((s-omega^i) Z_n'(omega^i))
	let arr_l_i = z_n_prime.into_par_iter().zip(arr_omega.into_par_iter()).
		map(|(x, y)|
		z_n_s*((s-y) * x).inverse().unwrap()).collect::<Vec<F>>();

	arr_l_i	
}


/// compute all [L_i(s)] for each i. 
pub fn precompute_lag_group<G:AffineCurve>(
	n: usize, 
	s: G::ScalarField)->Vec<G> 
	where <G as AffineCurve>::Projective: VariableBaseMSM<MSMBase=G, Scalar=<G as AffineCurve>::ScalarField>{
	assert!(n.is_power_of_two(), "n is not pow of 2");
	let arr_l_i = precompute_lags(n, s);	
	let g1 = G::prime_subgroup_generator();
	fixed_msm(g1, &arr_l_i)
}

/// compute all L_i(0) for i in [0, n)
pub fn precompute_lag0_fe<F:FftField>(n: usize)->Vec<F>{
	precompute_lags::<F>(n, F::zero())
}

/// compute all [(L_i(s)-L_i(0))/s] for each i
pub fn precompute_lag_diff<G:AffineCurve>(
	n: usize, 
	s: G::ScalarField)
	->Vec<G> where <G as AffineCurve>::Projective: VariableBaseMSM<MSMBase=G, Scalar=<G as AffineCurve>::ScalarField>{
	let s_inv = s.inverse().unwrap();
	let arr1 = precompute_lags::<G::ScalarField>(n, s);
	let arr2 = precompute_lags::<G::ScalarField>(n, G::ScalarField::zero());
	let res = arr1.into_par_iter().zip(arr2.into_par_iter()).map(|(x,y)|
		(x - y) * s_inv).collect::<Vec<G::ScalarField>>();
	let g1 = G::prime_subgroup_generator();

	fixed_msm(g1, &res)
}

/// compute z_n(omega^i) for each i in [0, n)
pub fn compute_derive_vanish<F:FftField>(n: usize) ->Vec<F>{
	let omega = F::get_root_of_unity(n as u64).unwrap();
	let arr_omega = compute_powers::<F>(n, omega);
	let fe_n = F::from(n as u64);
	let z_n_prime = arr_omega.into_par_iter().map(|x|
		fe_n*x.inverse().unwrap()).collect::<Vec<F>>();

	z_n_prime
}

/// see CQ paper for it: compute all quotient polynomials (group form)
pub fn precompute_quotient_poly<G:AffineCurve>(
	t: &Vec<G::ScalarField>, 
	n: usize, 
	kzg_key: &Vec<G>)->Vec<G> 
	where <G as AffineCurve>::Projective: VariableBaseMSM<MSMBase=G, Scalar=<G as AffineCurve>::ScalarField>{
	let kzg_prfs = precompute_kzg_proof(t, n, kzg_key);

	let z_n_prime = compute_derive_vanish::<G::ScalarField>(n);
	let res = kzg_prfs.into_par_iter().zip(z_n_prime.into_par_iter())
		.map(|(x,y)|
			x.mul(y.inverse().unwrap()).into_affine() ).collect::<Vec<G>>();

	res
}


/* This function calculates Lagrange basis polynomials for a given input polynomial. It first generates
   n points on the unit circle (roots of unity), where n is the degree of the input polynomial + 1. Then,
   for each point x_i, it computes the i-th Lagrange polynomial l_i(x) such that l_i(x) = 1 if x = x_i
   and 0 otherwise. The function returns a vector of all the Lagrange basis polynomials. */
pub fn precompute_lag<F: FftField>(poly: &DensePolynomial<F>) -> Vec<DensePolynomial<F>> {
    
	// Generate n x-values (roots of unity), where n is the degree of the polynomial + 1
    let n = closest_pow2(poly.degree() + 1);
    let root = F::get_root_of_unity(n as u64).unwrap();
    let x_values: Vec<F> = (0..n).map(|i| root.pow([i as u64])).collect();

    // Initialize the vector to hold the basis polynomials
    let mut basis_polynomials = Vec::new();

    // Iterate over each x-value
    for i in 0..x_values.len() {
        let xi = x_values[i];

        // Initialize the i-th Lagrange polynomial to 1
        let mut li = DensePolynomial::from_coefficients_slice(&[F::one()]);

        // Iterate over all x-values
        for j in 0..x_values.len() {
            // Skip when i = j since we don't want to include (x - x_i) in the product
            if i != j {
                let xj = x_values[j];

                // Compute the factor (x - x_j) / (x_i - x_j)
                let mut factor = DensePolynomial::from_coefficients_vec(vec![-xj, F::one()]); // x - xj
                let divisor = (xi - xj).inverse().unwrap(); // 1 / (xi - xj)

                // Multiply the factor by the divisor
                for coeff in factor.coeffs.iter_mut() {
                    *coeff = *coeff * divisor;
                }

                // Multiply the current Lagrange polynomial by the factor
                li = &li * &factor;
            }
        }

        // Add the computed Lagrange polynomial to the vector
        basis_polynomials.push(li);
    }

    // Return the Lagrange basis polynomials
    basis_polynomials

}


/// given vec_exp = [v1, ..., vn]
///    return [ g^v1, ..., v^vn]
pub fn fixed_msm<G: AffineCurve>(g_affine: G,
	vec_exp: &Vec<G::ScalarField>)->Vec<G>{
	let g = g_affine.into_projective();
	let window_size = FixedBase::get_mul_window_size(vec_exp.len()+1);
    let scalar_bits = G::ScalarField::MODULUS_BIT_SIZE as usize;
    let g_table = FixedBase::get_window_table(scalar_bits, window_size, g);
    let powers_proj= FixedBase::msm::<G::Projective>(
            scalar_bits,
            window_size,
            &g_table,
            &vec_exp,
    );
    let res = G::Projective::batch_normalization_into_affine(&powers_proj);
	return res;
}

/// compute variable multiscalar multplication
/// given bases: [g1, g2, ..., gn], and exponents [e1, ..., en]
/// return g1^e1 * g2^e2 * .... * gn^en
/// in groth term: [e1*g1] + ... + [en*gn]
pub fn vmsm<G:AffineCurve>(g: &Vec<G>, arr: &Vec<G::ScalarField>)
-> G where G::Projective: VariableBaseMSM<MSMBase=G, Scalar=G::ScalarField> {
		assert!(g.len()>=arr.len(), "exponent length must be <= base.len()!");
        let res: _ = <G::Projective as VariableBaseMSM>::msm(
            &g[..],
            arr
        );
        let res = res.into_affine();
        return res;
}

/// the trap door of KZG Polynomial Commitment
#[derive(Clone)]
pub struct KzgTrapDoor<F: FftField>{
	/// the trap door for generating KZG key series
	pub s: F
}

/// KZG Prover Key (for KZG Polynomial Commitment)
/// SLIGHLY DIFFERENT from the original scheme in the CQ paper,
/// we moved lag_all and lag_diff from the preprocessed info (by prover)
/// to the prover key generated by trusted up (as lag_all and lag_diff)
/// is not related to the problem statement (i.e., general purpose)
#[derive(Clone,CanonicalSerialize,CanonicalDeserialize)]
pub struct KzgProverKey<PE:PairingEngine>{
	/// the upper limit of the degree supported (size of lookup table)
	pub n: usize,
	/// the size of query table
	pub n2: usize,
	/// [s^0]_1, [s^1]_1, ..., [s^n]_1
	pub vec_g1: Vec<PE::G1Affine>,
	/// [s^0]_2, [s^1]_2, ..., [s^n]_2
	pub vec_g2: Vec<PE::G2Affine>,
	/// lag_all: L_i(s) for all i
	pub lag_all: Vec<PE::G1Affine>,
	/// lag_all: L_i(s) for all i in [0,n2] (for query table)
	pub lag_all_n2: Vec<PE::G1Affine>,
	/// lag_all: L_i(s) for all i, over group G2
	pub lag_all_2: Vec<PE::G2Affine>,
	/// lag_diff: ( L_i(s) - L_i(0) )/s for all i
	pub lag_diff: Vec<PE::G1Affine>,
	/// all fe elements of L_i(0)
	pub lag0_fe: Vec<PE::Fr>,
	/// [z_V(s)]_1, where z_V is the vanishing poly for all roots of unity
	pub zv_s1: PE::G1Affine,
	/// [z_V(s) * s]_1
	pub zv_s_s1: PE::G1Affine,
	/// [z_H(s) * s]_1
	pub zv_s_s1_n2: PE::G1Affine,
	/// [z_V(s)]_1, for domain [0,n2) - query table
	pub zv_s1_n2: PE::G1Affine,
	/// [z_V(s)]_2 over G2
	pub zv_s2: PE::G2Affine
}

/// KZG Verifier Key
#[derive(Clone,CanonicalSerialize,CanonicalDeserialize)]
pub struct KzgVerifierKey<PE:PairingEngine>{
	/// the upper limit of the degree supported (size of lookup table)
	pub n: usize,
	/// the size of query table
	pub n2: usize,
	/// generator of G1
	pub g1: PE::G1Affine,
	/// [s]_1
	pub s1: PE::G1Affine,
	/// generator of G2,
	pub g2: PE::G2Affine,
	/// s2: [s]_2
	pub s2: PE::G2Affine,
	/// zv_s1: [z_V(s)]_1. z_V the vanishing poly of all roots of unity
	pub zv_s1: PE::G1Affine,
	/// zv_s2: [z_V(s)]_2. z_V the vanishing poly of all roots of unity
	pub zv_s2: PE::G2Affine,
	/// [s^{N-n-1}]_2
	pub s2_n_diff: PE::G2Affine,
	/// [zv(s)*s]_1
	pub zv_s_s1: PE::G1Affine,
	/// [z_V(s)]_1, for domain [0,n2) - query table
	pub zv_s1_n2: PE::G1Affine,
	/// [z_H(s) * s]_1
	pub zv_s_s1_n2: PE::G1Affine,
}

/// return the default trapdoor
pub fn default_trapdoor<PE:PairingEngine>() -> KzgTrapDoor<PE::Fr>{
	let s = PE::Fr::from(GLOBAL_TRAPDOOR_S);
	KzgTrapDoor{s: s}
}

/// generate the KZG polynomial commitment
/// prover and verifier keys
/// n: the size of lookup table, n2: the size of query table
/// trapdoor: the s for generating KZG keys
pub fn setup_kzg<PE:PairingEngine>(n: usize, n2: usize, 
	trapdoor: &KzgTrapDoor<PE::Fr>) 
	-> (KzgProverKey<PE>, KzgVerifierKey<PE>) 
where
<<PE as PairingEngine>::G1Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G1Affine, Scalar=<<PE as PairingEngine>::G1Affine as AffineCurve>::ScalarField>,
 <<PE as PairingEngine>::G2Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G2Affine, Scalar=<<PE as PairingEngine>::G2Affine as AffineCurve>::ScalarField>{
	//1. power series [s^0, s^1, ..., s^n] 
	assert!(n.is_power_of_two(), "n is not power of 2");
	assert!(n2.is_power_of_two(), "n2 is not power of 2");
	let b_perf = false;
	let mut timer = Timer::new();
	if b_perf{println!("\n** setup_kzg keys");}
	let s = trapdoor.s;
	let mut vec_exp = vec![PE::Fr::one(); n+1];
	for i in 1..n+1 {vec_exp[i] = vec_exp[i-1] * s}
	if b_perf {log_perf(LOG1, "-- build field elemnts: {s^i} --", &mut timer);}

	//2. build ( [s^0]_1, ..., [s^n]_1 ) on G1
	let g1 = PE::G1Affine::prime_subgroup_generator();
	let vec_g1 = fixed_msm(g1, &vec_exp);
	if b_perf {log_perf(LOG1, "-- build [s^i]_1 --", &mut timer);}

	//3. build ( [s^0]_1, ..., [s^n]_1 ) on G2
	let g2 = PE::G2Affine::prime_subgroup_generator();
	let vec_g2 = fixed_msm(g2, &vec_exp);
	if b_perf {log_perf(LOG1, "-- build [s^i]_2 --", &mut timer);}

	//4. build the zv_s2 and zv_s1
	let f_zv = s.pow(&[n as u64]) - PE::Fr::one();
	let f_zv_n2 = s.pow(&[n2 as u64]) - PE::Fr::one();
	let zv_s2 = g2.mul(f_zv).into_affine();
	let zv_s1 = g1.mul(f_zv).into_affine();
	let zv_s_s1 = g1.mul(f_zv * s).into_affine();
	let zv_s1_n2 = g1.mul(f_zv_n2).into_affine();
	let zv_s_s1_n2 = g1.mul(f_zv_n2*s).into_affine();
	let s2_n_diff = vec_g2[n-n2-1]; 
	if b_perf {log_perf(LOG1, "-- build zv_s --", &mut timer);}

	//5. build [L_i(s)]_1 and [(L_i(s)-L_i(0))/s]_1
	let all_lag = precompute_lag_group::<PE::G1Affine>(n,s);
	if b_perf {log_perf(LOG1, "-- build [L_i(s)]_1 --", &mut timer);}
	let all_lag_n2 = precompute_lag_group::<PE::G1Affine>(n2,s);
	if b_perf {log_perf(LOG1, "-- build [L_i(s)]_1 for n2--", &mut timer);}
	let all_lag_2 = precompute_lag_group::<PE::G2Affine>(n,s);
	if b_perf {log_perf(LOG1, "-- build [L_i(s)]_2 --", &mut timer);}
	let lag_diff = precompute_lag_diff::<PE::G1Affine>(n,s);
	if b_perf {log_perf(LOG1, "-- build [(L_i(s)-L_i(0))/s]_1 --", &mut timer);}
	let lag0_fe = precompute_lag0_fe::<PE::Fr>(n);
	if b_perf {log_perf(LOG1, "-- build [(L_i(0)]_1 --", &mut timer);}

	//6. build and return
	let s1 = vec_g1[1].clone();
	let s2 = vec_g2[1].clone();
	let pkey = KzgProverKey{n: n, n2: n2, vec_g1: vec_g1, vec_g2: vec_g2, 
		lag_all: all_lag, lag_all_2: all_lag_2, 
		lag_all_n2: all_lag_n2, lag0_fe: lag0_fe,
		lag_diff: lag_diff, zv_s1: zv_s1, zv_s2: zv_s2, zv_s1_n2: zv_s1_n2,
		zv_s_s1: zv_s_s1, zv_s_s1_n2: zv_s_s1_n2};
	let vkey = KzgVerifierKey{n: n, n2: n2,
			g1: g1, s1: s1, g2: g2, s2: s2, zv_s2: zv_s2,
			s2_n_diff: s2_n_diff, zv_s1: zv_s1, zv_s_s1: zv_s_s1,
			zv_s1_n2: zv_s1_n2,
			zv_s_s1_n2: zv_s_s1_n2};
	return (pkey, vkey);
}

// ----------------------------------------------------
// -- Utility Functions -------------------------------
// ----------------------------------------------------
pub fn print_arr_fe<F:FftField>(s: &str, a: &Vec<F>){
	println!("==== {}: {} ====", s, a.len());
	for i in 0..a.len(){
		println!("-- i: {}, val: {}", i, a[i]);
	}
}
pub fn print_arr<G:AffineCurve>(s: &str, a: &Vec<G>){
	println!("==== {}: {} ====", s, a.len());
	for i in 0..a.len(){
		println!("-- i: {}, val: {}", i, a[i]);
	}
}



// the following functions are VERBATIM from 
// ark_works/poly/domain/utils (related to compute_powers)
const MIN_PARALLEL_CHUNK_SIZE: usize = 1 << 7;

pub(crate) fn compute_powers_serial<F: Field>(size: usize, root: F) -> Vec<F> {
    compute_powers_and_mul_by_const_serial(size, root, F::one())
}
pub(crate) fn compute_powers_and_mul_by_const_serial<F: Field>(
    size: usize,
    root: F,
    c: F,
) -> Vec<F> {
    let mut value = c;
    (0..size)
        .map(|_| {
            let old_value = value;
            value *= root;
            old_value
        })
        .collect()
}
pub(crate) fn compute_powers<F: Field>(size: usize, g: F) -> Vec<F> {
	assert!(size.is_power_of_two(), "size: {} is not power of 2", size);
    if size < MIN_PARALLEL_CHUNK_SIZE {
        return compute_powers_serial(size, g);
    }
    // compute the number of threads we will be using.
    use self::ark_std::cmp::{max, min};
    let num_cpus_available = rayon::current_num_threads();
    let num_elem_per_thread = max(size / num_cpus_available, MIN_PARALLEL_CHUNK_SIZE);
    let num_cpus_used = size / num_elem_per_thread;

    // Split up the powers to compute across each thread evenly.
    let res: Vec<F> = (0..num_cpus_used)
        .into_par_iter()
        .flat_map(|i| {
            let offset = g.pow(&[(i * num_elem_per_thread) as u64]);
            // Compute the size that this chunks' output should be
            // (num_elem_per_thread, unless there are less than num_elem_per_thread elements remaining)
            let num_elements_to_compute = min(size - i * num_elem_per_thread, num_elem_per_thread);
           let res = compute_powers_and_mul_by_const_serial(num_elements_to_compute, g, offset);
            res
        })
        .collect();
    res
}
// -------- the above code are VERBATIM from ark_work/poly/domain/utils.rs

#[cfg(test)]
pub mod tests {
	extern crate acc;
	extern crate ark_bls12_381;

	
	use self::ark_ec::{AffineCurve, PairingEngine, ProjectiveCurve};
	use izpr::serial_poly_utils::*;
	use izpr::serial_logical_poly_utils::*;
	use izpr::serial_rng::rand_arr_fe;
    use self::acc::poly::serial::{serial_fft, serial_ifft};
    use izpr::serial_group_fft2::{serial_group_fft, serial_logical_group_fft, serial_group_ifft, serial_logical_group_ifft};

	use self::ark_bls12_381::Bls12_381;
	type Fr381 = ark_bls12_381::Fr;
	type PE381 = Bls12_381;
	type G1_381= ark_bls12_381::G1Affine;

/*

    #[test]
    fn test_lagrange() {
        let poly = rand_poly::<Fr381>(3, 123456789);
        let lag = precompute_lag(&poly);

        // Generate x-values (roots of unity)
        let n = closest_pow2(poly.degree() + 1); // We want n points for interpolation, where n is degree of poly + 1
        let root = Fr381::get_root_of_unity(n as u64).unwrap();
        let x_values: Vec<Fr381> = (0..n).map(|i| root.pow([i as u64])).collect();

        for (i, poly) in lag.iter().enumerate() {
            // Evaluate the polynomial at each root of unity
            for (j, &xj) in x_values.iter().enumerate() {
                let val = poly.evaluate(&xj);

                // Check if the result is close to 0 or 1 (with some tolerance for numerical errors)
                if i == j {
                    assert!(
                        (val - Fr381::one()).is_zero(),
                        "Value at root {} should be 1",
                        j + 1
                    );
                } else {
                    assert!(
                        (val - Fr381::zero()).is_zero(),
                        "Value at root {} should be 0",
                        j + 1
                    );
                }
            }
        }
    }

	
    #[test]
    fn test_fft() {
		//1. pick a small power of 2
		//2. call logical gropu_fft
		//3. call group_fft
		//4. fail if not the same.
	
	}
*/

	#[test]
	fn test_precompute_kzg_proof(){
		let n = 32;
		let n2 = 4;
		let bits = 64;
		let n3 = 16;
		let mut t = rand_arr_fe::<Fr381>(n3, bits);
		//let mut t = vec![Fr381::from(1u32),Fr381::from(0u32)];
		let trap = default_trapdoor::<PE381>();
		let (pk, vk) = setup_kzg::<PE381>(n, n2, &trap); 

		let arr = precompute_kzg_proof::<G1_381>(&t, n3, &pk.vec_g1);
		let arr2 = logical_precompute_kzg_proof::<G1_381>(&t, n3, trap.s);
		assert!(arr==arr2, "precompute_kzg_proof nok");
	}

	#[test]
	fn test_precompute_lag_group(){
		let n = 512;
		let s = default_trapdoor::<PE381>().s;

		let arr1 = precompute_lag_group::<G1_381>(n, s);
		let arr2 = logical_precompute_lag_group::<G1_381>(n, s);
		assert!(arr1==arr2, "precompute_lag group nok");
	}

	#[test]
	fn test_precompute_lag0_fe(){
		let n = 32;

		let arr1 = precompute_lag0_fe::<Fr381>(n);
		let arr2 = logical_precompute_lag0_fe::<Fr381>(n);
		assert!(arr1==arr2, "precompute_lag0_fe nok");
	}

	#[test]
	fn test_precompute_lag_diff(){
		let n = 32;
		let s = default_trapdoor::<PE381>().s;

		let arr1 = precompute_lag_diff::<G1_381>(n,s);
		let arr2 = logical_precompute_lag_diff::<G1_381>(n, s);
		assert!(arr1==arr2, "precompute_lag_diff nok");
	}

	#[test]
	fn test_precompute_quotient_poly(){
		let n = 32;
		let n2 = 4;
		let bits = 64;
		let mut t = rand_arr_fe::<Fr381>(n, bits);
		let trap = default_trapdoor::<PE381>();
		let (pk, vk) = setup_kzg::<PE381>(n, n2, &trap); 

		let arr1 = precompute_quotient_poly::<G1_381>(&t, n, &pk.vec_g1);
		let arr2 = logical_precompute_quotient_poly::<G1_381>(&t, n, trap.s);
		assert!(arr1==arr2, "precompute_quotient_poly nok");
	}

}

