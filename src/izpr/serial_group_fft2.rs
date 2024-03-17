/*
    Copyright Dr. Xiang Fu

    Author: Trevor Conley, Xiang Fu
    All Rights Reserved.
    Created: 09/02/2023

i	 *** This code is a PORT of ark_work EvaluationDomain and FFT code
	 for field elements.  (see ark_works: poly/src/domain/mod.rs and
	radix2/mod.rs  and radix2/fft.rs ***
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

use self::ark_ff::{FftField,One,Zero };
use izpr::serial_poly_utils::{vmsm, compute_powers,compute_powers_serial};
use self::ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Read, SerializationError, Write};

use self::ark_ec::{AffineCurve, ProjectiveCurve};
use self::ark_ff::{Field};
use self::ark_ec::msm::{VariableBaseMSM};
//use self::ark_std::{cfg_chunks_mut,cfg_into_iter, cfg_iter, cfg_iter_mut, vec::Vec};

const MIN_NUM_CHUNKS_FOR_COMPACTION: usize = 1 << 7;
const MIN_GAP_SIZE_FOR_PARALLELISATION: usize = 1 << 10;
const LOG_ROOTS_OF_UNITY_PARALLEL_SIZE: u32 = 7;



/// Types that can be ProjectiveCurve must implement this trait.
pub trait GroupDomainCoeff<G: ProjectiveCurve>:
    Copy
    + Send
    + Sync
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::AddAssign
    + core::ops::SubAssign
    + ark_ff::Zero
    + core::ops::MulAssign<G::ScalarField>
{
}

impl<T, G> GroupDomainCoeff<G> for T
where
    G: ProjectiveCurve,
    T: Copy
        + Send
        + Sync
        + core::ops::Add<Output = Self>
        + core::ops::Sub<Output = Self>
        + core::ops::AddAssign
        + core::ops::SubAssign
        + ark_ff::Zero
        + core::ops::MulAssign<G::ScalarField>,
{
}

#[derive(PartialEq, Eq, Debug)]
enum GroupFFTOrder {
    /// Both the input and the output of the FFT must be in-order.
    II,
    /// The input of the FFT must be in-order, but the output does not have to
    /// be.
    IO,
    /// The input of the FFT can be out of order, but the output must be
    /// in-order.
    OI,
}

#[derive(Copy, Clone, Hash, Eq, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct Radix2GroupEvaluationDomain<G: ProjectiveCurve> {
	pub size: u64,
	pub root: G::ScalarField,
	pub size_inv: G::ScalarField,
}

impl <G: ProjectiveCurve>  Radix2GroupEvaluationDomain <G> {
	fn new(size: u64) -> Radix2GroupEvaluationDomain<G>{
		assert!(size.is_power_of_two(), "size is not power of 2");
		Radix2GroupEvaluationDomain::<G>{
			size: size,
			root: G::ScalarField::get_root_of_unity(size as u64).unwrap(),
			size_inv: G::ScalarField::from(size).inverse().unwrap()
		}
	}
	
	fn group_gen(&self) -> G::ScalarField{
		self.root
	}
	fn group_gen_inv(&self) -> G::ScalarField{
		self.root.inverse().unwrap()
	}

	fn fft_in_place<T:GroupDomainCoeff<G>>(&self, coefs: &mut Vec<T>){
		coefs.resize(self.size as usize, T::zero());
		self.in_order_group_fft_in_place(coefs);
	}

	fn ifft_in_place<T:GroupDomainCoeff<G>>(&self, coefs: &mut Vec<T>){
		coefs.resize(self.size as usize, T::zero());
		self.in_order_group_ifft_in_place(coefs);
	}

	fn in_order_group_fft_in_place<T:GroupDomainCoeff<G>>(&self, x_s: &mut [T]){
		self.fft_helper_in_place(x_s, GroupFFTOrder::II)
	}

	fn in_order_group_ifft_in_place<T:GroupDomainCoeff<G>>(&self, 
	x_s: &mut [T]){
		self.ifft_helper_in_place(x_s, GroupFFTOrder::II);
		x_s.par_iter_mut().for_each(|val| *val *= self.size_inv);
	}

	fn fft_helper_in_place<T:GroupDomainCoeff<G>>(&self, x_s: &mut [T], ord: GroupFFTOrder){
		use self::GroupFFTOrder::*;
        let log_len = ark_std::log2(x_s.len());

        if ord == OI {
            self.oi_helper(x_s, self.group_gen());
        } else {
            self.io_helper(x_s, self.group_gen());
        }

        if ord == II {
            Self::derange(x_s, log_len);
        }
	}	

	fn ifft_helper_in_place<T:GroupDomainCoeff<G>>(&self, x_s: &mut [T], ord: GroupFFTOrder){
		use self::GroupFFTOrder::*;

        let log_len = ark_std::log2(x_s.len());
        if ord == II { Self::derange(x_s, log_len); }

        if ord == IO {
            self.io_helper(x_s, self.group_gen_inv());
        } else {
            self.oi_helper(x_s, self.group_gen_inv());
        }

	}
	
	fn oi_helper<T: GroupDomainCoeff<G>>(&self, xi: &mut [T], root: G::ScalarField) {
  		let roots_cache = self.roots_of_unity(root);
        // The `cmp::min` is only necessary for the case where
        // `MIN_NUM_CHUNKS_FOR_COMPACTION = 1`. Else, notice that we compact
        // the roots cache by a stride of at least `MIN_NUM_CHUNKS_FOR_COMPACTION`.

        let compaction_max_size = core::cmp::min(
            roots_cache.len() / 2,
            roots_cache.len() / MIN_NUM_CHUNKS_FOR_COMPACTION,
        );
        let mut compacted_roots = vec![G::ScalarField::default(); compaction_max_size];
		let max_threads = rayon::current_num_threads();
       	let mut gap = 1;
        while gap < xi.len() {
            // each butterfly cluster uses 2*gap positions
            let chunk_size = 2 * gap;
            let num_chunks = xi.len() / chunk_size;

            // Only compact roots to achieve cache locality/compactness if
            // the roots lookup is done a significant amount of times
            // Which also implies a large lookup stride.
            let (roots, step) = if num_chunks >= MIN_NUM_CHUNKS_FOR_COMPACTION && gap < xi.len() / 2
            {
                compacted_roots[..gap].par_iter_mut()
                    .zip(roots_cache[..(gap * num_chunks)].par_iter()
					.step_by(num_chunks))
                    .for_each(|(a, b)| *a = *b);
                (&compacted_roots[..gap], 1)
            } else {
                (&roots_cache[..], num_chunks)
            };
            Self::apply_butterfly(
                Self::butterfly_fn_oi,
                xi,
                roots,
                step,
                chunk_size,
                num_chunks,
                max_threads,
                gap,
            );

            gap *= 2;
        }
    }

	#[inline(always)]
    fn butterfly_fn_io<T: GroupDomainCoeff<G>>(((lo, hi), root): ((&mut T, &mut T), &G::ScalarField)) {
        let neg = *lo - *hi;
        *lo += *hi;
        *hi = neg;
        *hi *= *root;
    }

	#[inline(always)]
    fn butterfly_fn_oi<T: GroupDomainCoeff<G>>(((lo, hi), root): ((&mut T, &mut T), &G::ScalarField)) {
        *hi *= *root;
        let neg = *lo - *hi;
        *lo += *hi;
        *hi = neg;
    }


	fn apply_butterfly<T: GroupDomainCoeff<G>, Gf: Fn(((&mut T, &mut T), &G::ScalarField)) + Copy + Sync + Send>(
        g: Gf,
        xi: &mut [T],
        roots: &[G::ScalarField],
        step: usize,
        chunk_size: usize,
        num_chunks: usize,
        max_threads: usize,
        gap: usize,
    ) {
       xi.par_chunks_mut(chunk_size).for_each(|cxi| {
            let (lo, hi) = cxi.split_at_mut(gap);
            // If the chunk is sufficiently big that parallelism helps,
            // we parallelize the butterfly operation within the chunk.

            if gap > MIN_GAP_SIZE_FOR_PARALLELISATION && num_chunks < max_threads {
				lo.par_iter_mut()
					.zip(hi)
					.zip(roots.par_iter().step_by(step))
					.for_each(g);
            } else {
                lo.iter_mut()
                    .zip(hi)
                    .zip(roots.iter().step_by(step))
                    .for_each(g);
            }
        });
    }


	fn roots_of_unity(&self, root: G::ScalarField) -> Vec<G::ScalarField> {
       let log_size = ark_std::log2(self.size as usize);
        // early exit for short inputs
        if log_size <= LOG_ROOTS_OF_UNITY_PARALLEL_SIZE {
            compute_powers_serial((self.size as usize) / 2, root)
        } else {
            let mut temp = root;
            // w, w^2, w^4, w^8, ..., w^(2^(log_size - 1))
            let log_powers: Vec<G::ScalarField> = (0..(log_size - 1))
                .map(|_| {
                    let old_value = temp;
                    temp.square_in_place();
                    old_value
                })
                .collect();
           // allocate the return array and start the recursion
            let mut powers = vec![G::ScalarField::zero(); 1 << (log_size - 1)];
            Self::roots_of_unity_recursive(&mut powers, &log_powers);
            powers
        }
    }


    fn roots_of_unity_recursive(out: &mut [G::ScalarField], log_powers: &[G::ScalarField]) {
        assert_eq!(out.len(), 1 << log_powers.len());
        // base case: just compute the powers sequentially,
        // g = log_powers[0], out = [1, g, g^2, ...]
        if log_powers.len() <= LOG_ROOTS_OF_UNITY_PARALLEL_SIZE as usize {
            out[0] = G::ScalarField::one();
            for idx in 1..out.len() {
                out[idx] = out[idx - 1] * log_powers[0];
            }
            return;
        }

        // recursive case:
        // 1. split log_powers in half
        let (lr_lo, lr_hi) = log_powers.split_at((1 + log_powers.len()) / 2);
        let mut scr_lo = vec![G::ScalarField::default(); 1 << lr_lo.len()];
        let mut scr_hi = vec![G::ScalarField::default(); 1 << lr_hi.len()];
        // 2. compute each half individually
        rayon::join(
            || Self::roots_of_unity_recursive(&mut scr_lo, lr_lo),
            || Self::roots_of_unity_recursive(&mut scr_hi, lr_hi),
        );
        // 3. recombine halves
        // At this point, out is a blank slice.
        out.par_chunks_mut(scr_lo.len())
            .zip(&scr_hi)
            .for_each(|(out_chunk, scr_hi)| {
                for (out_elem, scr_lo) in out_chunk.iter_mut().zip(&scr_lo) {
                    *out_elem = *scr_hi * scr_lo;
                }
            });
    }



	fn io_helper<T: GroupDomainCoeff<G>>(&self, xi: &mut [T], root: G::ScalarField) {
       	let mut roots = self.roots_of_unity(root);
        let mut step = 1;
        let mut first = true;
        let max_threads = rayon::current_num_threads();

		let mut gap = xi.len() / 2;
        while gap > 0 {
            // each butterfly cluster uses 2*gap positions
            let chunk_size = 2 * gap;
            let num_chunks = xi.len() / chunk_size;

            // Only compact roots to achieve cache locality/compactness if
            // the roots lookup is done a significant amount of times
            // Which also implies a large lookup stride.
            if num_chunks >= MIN_NUM_CHUNKS_FOR_COMPACTION {
                if !first {
                    roots = roots.into_par_iter().step_by(step * 2).collect()
                }
                step = 1;
                roots.shrink_to_fit();
            } else {
                step = num_chunks;
            }
            first = false;
            Self::apply_butterfly(
                Self::butterfly_fn_io,
                xi,
                &roots[..],
                step,
                chunk_size,
                num_chunks,
                max_threads,
                gap,
            );

            gap /= 2;
        }
    }

	fn derange<T>(xi: &mut [T], log_len: u32) {
	    for idx in 1..(xi.len() as u64 - 1) {
	        let ridx = Self::bitrev(idx, log_len);
	        if idx < ridx {
	            xi.swap(idx as usize, ridx as usize);
        	}
    	}
	}

	#[inline]
	fn bitrev(a: u64, log_len: u32) -> u64 {
    	a.reverse_bits() >> (64 - log_len)
	}


}

	




/** group_fft: given a vector of group elements:
	( [a_1]_1, ..., [a_n]_1 )
	[a_1] is like g^{a_1}
	The function return an array
	( [b_1]_1, ..., [b_n]_1 )
	vector b is the result of running standard FFT over vector a
*/
pub fn serial_group_fft<G: AffineCurve>(a: &Vec<G>) -> Vec<G>
where <G as AffineCurve>::Projective: VariableBaseMSM<MSMBase=G, Scalar=<G as AffineCurve>::ScalarField>{
	let n  = a.len();
	assert!(n.is_power_of_two(), "n is not pow of 2");
	let mut proj_a: Vec<G::Projective> = a.into_par_iter().map(|x| x.into_projective()). collect::<Vec<G::Projective>>();
	//in_order_group_fft_in_place::<G::Projective, dyn GroupDomainCoeff<G::Projective>>(&mut proj_a);
	let domain = Radix2GroupEvaluationDomain::<G::Projective>::new(n as u64);
	domain.fft_in_place(&mut proj_a);
	G::Projective::batch_normalization_into_affine(&proj_a)	
}

/** logical version of the group fft 
	vec_a = [a_1, ..., a_n]
	vec_b = standard_fft(vec_a); //say vec_b = [b_1, ..., b_n]
	return
	[g^b_1, ..., g^b_n]
	let g = G::prime_subgroup_generator();
*/
pub fn serial_logical_group_fft<G: AffineCurve>(a: &Vec<G>) 
	-> Vec<G>
where <G as AffineCurve>::Projective: VariableBaseMSM<MSMBase=G, Scalar=<G as AffineCurve>::ScalarField>{
	let n = a.len();
	let omega = G::ScalarField::get_root_of_unity(n as u64).unwrap();
	let arr_omega = compute_powers::<G::ScalarField>(n, omega);
	let res = arr_omega.into_par_iter().map(|x|
		vmsm(a, &compute_powers(n, x)) ).collect::<Vec<G>>();

	res
}


/** similar applies to the group operations ifft */
pub fn serial_group_ifft<G: AffineCurve>(a: &Vec<G>) -> Vec<G>
where <G as AffineCurve>::Projective: VariableBaseMSM<MSMBase=G, Scalar=<G as AffineCurve>::ScalarField>{
	let n  = a.len();
	assert!(n.is_power_of_two(), "n is not pow of 2");
	let mut proj_a: Vec<G::Projective> = a.into_par_iter().map(|x| x.into_projective()). collect::<Vec<G::Projective>>();
	let domain = Radix2GroupEvaluationDomain::<G::Projective>::new(n as u64);
	domain.ifft_in_place(&mut proj_a);
	G::Projective::batch_normalization_into_affine(&proj_a)	
}

pub fn serial_logical_group_ifft<G: AffineCurve>(a: &Vec<G>) -> Vec<G>
where <G as AffineCurve>::Projective: VariableBaseMSM<MSMBase=G, Scalar=<G as AffineCurve>::ScalarField>{
	let n = a.len();
	let inv_omega = G::ScalarField::get_root_of_unity(n as u64)
		.unwrap().inverse().unwrap();
	let arr_omega = compute_powers::<G::ScalarField>(n, inv_omega);
	let fe_n = G::ScalarField::from(n as u64).inverse().unwrap();
	let res = arr_omega.into_par_iter().map(|x|
		vmsm(a, &compute_powers(n, x)).mul(fe_n).into_affine() )
		.collect::<Vec<G>>();

	res
}


#[cfg(test)]
pub mod tests {
	extern crate ark_ff;
	extern crate ark_ec;
	extern crate ark_bls12_381;
	extern crate acc;

	use izpr::serial_poly_utils::*;
	use izpr::serial_rng::rand_arr_fe;
	use self::ark_ec::AffineCurve;
    use self::acc::poly::serial::{serial_fft, serial_ifft};
    use izpr::serial_group_fft2::{serial_group_fft, serial_group_ifft, serial_logical_group_fft, serial_logical_group_ifft};
    use izpr::serial_poly_utils::{fixed_msm, vmsm};

	use self::ark_bls12_381::Bls12_381;
	type Fr381 = ark_bls12_381::Fr;
	type PE381 = Bls12_381;
	type G1_381= ark_bls12_381::G1Affine;
	

	#[test]
	fn test_group_ifft(){
		let n = 32;
		let bits = 250;
		let mut t = rand_arr_fe::<Fr381>(n, bits);
		let g1 = G1_381::prime_subgroup_generator();
		let arr_g = fixed_msm(g1, &t);

		let res1 = serial_group_ifft::<G1_381>(&arr_g);
		let res2 = serial_logical_group_ifft::<G1_381>(&arr_g);
		assert!(res1==res2, "ifft result nok!");
	}

	#[test]
	fn test_group_fft(){
		let n = 32;
		let bits = 250;
		let mut t = rand_arr_fe::<Fr381>(n, bits);
		let g1 = G1_381::prime_subgroup_generator();
		let arr_g = fixed_msm(g1, &t);

		let res1 = serial_group_fft::<G1_381>(&arr_g);
		let res2 = serial_logical_group_fft::<G1_381>(&arr_g);
		assert!(res1==res2, "fft result nok!");
	}

	#[test]
	fn test_logical_group_fft(){
		let n = 32;
		let bits = 250;
		let mut t = rand_arr_fe::<Fr381>(n, bits);
		let g1 = G1_381::prime_subgroup_generator();
		let arr_g = fixed_msm(g1, &t);
		let mut t2 = t.clone();
		serial_fft(&mut t2);
		let res1 = fixed_msm(g1, &t2);	
		let res2 = serial_logical_group_fft::<G1_381>(&arr_g);
		assert!(res1==res2, "logical_group_fft fails");
	}

	#[test]
	fn test_logical_group_ifft(){
		let n = 32;
		let bits = 250;
		let mut t = rand_arr_fe::<Fr381>(n, bits);
		let g1 = G1_381::prime_subgroup_generator();
		let arr_g = fixed_msm(g1, &t);
		let mut t2 = t.clone();
		serial_ifft(&mut t2);
		let res1 = fixed_msm(g1, &t2);	
		let res2 = serial_logical_group_ifft::<G1_381>(&arr_g);
		assert!(res1==res2, "logical_group_ifft fails");
	}
}
