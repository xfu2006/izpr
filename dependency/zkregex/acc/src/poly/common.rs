/** 
	Copyright Dr. Xiang Fu

	Author: Dr. Xiang Fu
	All Rights Reserved.
	Created: 06/05/2022
	Reviesed: 03/07/2023 -> Added subvec related functions
	Common Utility Functions for Poly package
*/
extern crate ark_ff;
extern crate ark_poly;
extern crate ark_std;
//extern crate mpi;
extern crate ark_serialize;

use crate::tools::*;
//use ark_ff::{FftField};
use self::ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
use self::ark_poly::{univariate::DensePolynomial};
use self::ark_ff::PrimeField;
//use self::mpi::point_to_point::Status;
//use self::mpi::request::*;
//use self::mpi::traits::*;
//use self::mpi::environment::*;
use self::ark_std::log2;
use profiler::config::*;

const COUNT: usize = 256;
/** return the i'th share of start and end. Static version
	return (start, end) in the entire logical sequence.
	end index is ACTUALLY not included.
	In another word: len_partition = end-start
*/
pub fn get_share_start_end(i: u64, n: u64, total_len: u64) -> (usize, usize){
	let share_size = total_len/n;
	let start = (share_size*i) as usize;
	let end = if i==n-1 {total_len as usize} else {start+share_size as usize};
	return (start as usize, end as usize);
}

/// get the intersection: [start, end), end is actually not included.
pub fn share_intersect(sh1: &(usize, usize), sh2: &(usize, usize))->(usize, usize){
	let start = if sh1.0>sh2.0 {sh1.0} else {sh2.0};
	let mut end = if sh1.1<sh2.1 {sh1.1} else {sh2.1};
	if end<start {end = start;}
	return (start, end);
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
pub fn gen_rescale_plan(src: usize, dst: usize, np: usize, cur_len: usize, target_len: usize) ->(usize, usize, usize, usize){
	let (src_start, src_end) = get_share_start_end(src as u64, np as u64, cur_len as u64);
	let (dest_start, dest_end) = get_share_start_end(dst as u64, np as u64, target_len as u64);
	let start = if src_start<dest_start {dest_start} else {src_start};
	let end = if src_end<dest_end {src_end} else {dest_end};
	//println!("DEBUG USE 889 --- gen_repart_plan: src: {}, dest: {}, target_len{}. src_start: {}, src_end{}, dest_start: {}, dest_end: {} => start: {}, end: {}", src, dst, target_len, src_start, src_end, dest_start, dest_end, start, end);
	if end>start{
		let start_off = start-src_start;
		let end_off = end-src_start;
		return (start_off, end_off, start, end);
	}else{
		return (0,0,0,0); //nothing to send
	}
}  




/// from ark_poly_commit/kzg10
pub fn skip_leading_zeros_and_convert_to_bigints<F: PrimeField>(
    p: &DensePolynomial<F>,
) -> (usize, Vec<F>) {
    let mut num_leading_zeros = 0;
    while num_leading_zeros < p.coeffs.len() && p.coeffs[num_leading_zeros].is_zero() {
        num_leading_zeros += 1;
    }
    let coeffs = &p.coeffs[num_leading_zeros..];
    (num_leading_zeros, coeffs.to_vec())
}

/// get the closet power of 2
pub fn closest_pow2(n: usize)->usize{
	let k = log2(n);
	let n2 = 1<<k;
	return n2;
}

/*
/// from ark_poly_commit/kzg10
pub fn convert_to_bigints<F: PrimeField>(p: &[F]) -> Vec<F::BigInt> {
    let coeffs = ark_std::cfg_iter!(p)
        .map(|s| s.into_repr())
        .collect::<Vec<_>>();
    coeffs
}
*/

fn bigger(a: usize, b: usize) -> usize{
	return if a>b {a} else {b};
}

fn smaller(a: usize, b: usize) -> usize{
	return if a>b {b} else {a};
}

fn intersect(r1: (usize, usize), r2: (usize, usize))-> (usize,usize){
	let b1 = bigger(r1.0, r2.0);
	let b2 = smaller(r1.1, r2.1);
	return if b1<b2 {(b1, b2)} else {(0,0)};
}

/// assuming there is a DisVec or GroupDisVec of total_len
/// for generating a sublist of [start, end)
/// generate a transfer data plan
/// res[i][j] is a tuple (me_start, me_end, dest_start, dest_end)
/// (me_start, me_end) is the the PARITION range to copy from 
/// (dest_start, dest_end) is the PARITTION range to copy to (new)
/// both ends are not really included 
pub fn gen_subvec_plan(total_len: usize, np: usize, start: usize, end: usize) ->Vec<Vec<(usize, usize, usize, usize)>>{
	assert!(end>=start, "gen_subvec ERR: end<start");
	assert!(end<=total_len, "gen_subvec ERR: end>total_len");
	
	let mut res = vec![vec![(0,0,0,0); np]; np];
	let new_len = end-start;
	for src in 0..np{
		let src_start = total_len/np*src; 
		let src_end = if src<np-1 {total_len/np*(src+1)} else {total_len};
		//in sublist
		let new_src_start = if src_start<start {0} else {src_start-start};
		let new_src_end = if src_end<start {0} else {src_end-start}; 
		for dest in 0..np{
			let dest_start = new_len/np*dest;
			let dest_end= if dest<np-1 {new_len/np*(dest+1)} else {new_len};
			let (b1, b2) = intersect((new_src_start, new_src_end), (dest_start, dest_end)); //in new list
			if b2>b1{
				//log(LOG1, &format!("REMOVE LATER 101: total_len: {}, start: {}, end: {}, src: {}, src_start: {}, src_end: {}, new_src_start: {}, new_src_end: {}, dest: {}, dest_start: {}, dest_end: {}: b1: {}, b2: {}", total_len, start, end, src, src_start, src_end, new_src_start, new_src_end, dest, dest_start, dest_end, b1, b2));
				res[src][dest] =(b1+start-src_start, b2+start-src_start, b1-dest_start, b2-dest_start);
				//log(LOG1, &format!("REMOVE LATER 102: res[{}][{}]: {:?}", src, dest, res[src][dest]));
			}
		}
	}
	return res;
}

