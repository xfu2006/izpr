/** 
	Copyright Dr. Xiang Fu
	Author: Dr. Xiang Fu
	All Rights Reserved.
	Created: 06/22/2022
	Wraps a number of serial polynomial functions (from ark-poly).
	Provides conversion functions to and from dis_poly
	Provides functions for half-GCD.
	Provides functions for derivative.
*/
extern crate ark_ff;
extern crate ark_poly;
extern crate ark_std;
//extern crate mpi;
extern crate ark_serialize;

use crate::tools::*;
use crate::poly::common::*;
use std::collections::HashSet;
use self::ark_ff::{FftField,Zero};
use self::ark_std::log2;
use self::ark_serialize::{CanonicalSerialize};
use self::ark_poly::{Polynomial, DenseUVPolynomial, univariate::DensePolynomial, univariate::DenseOrSparsePolynomial};
use self::ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
//use ark_ff::FftField;
//extern crate mpi;
//use mpi::traits::*;
//use mpi::environment::*;
use profiler::config::*;

/** wrapper of DenseOrSparsePolynomial.divide_with_a_and_r from ark-poly */
//#[inline(always)]
pub fn old_divide_with_q_and_r<F: FftField>(
	f: &DensePolynomial<F>, 
	g: &DensePolynomial<F>)
	->(DensePolynomial<F>, DensePolynomial<F>){
	let self_f = DenseOrSparsePolynomial::from(f);
	let self_g = DenseOrSparsePolynomial::from(g);
	let res =  self_f.divide_with_q_and_r(&self_g).unwrap();
	return res;
}

/* check whether the polys mul together is 1, mod x^k */ 
fn _check_inv<F:FftField>(v: &Vec<DensePolynomial<F>>, k: usize, msg: &str){
	let pone = get_poly(vec![F::from(1u64)]);
	let mut prod = pone.clone();
	for i in 0..v.len(){
		let p = &v[i];
		prod = &prod * p;
	}
	let res = mod_by_xk::<F>(&prod, k);
	if res!=pone{
		panic!("check_inv failed: {}", msg);
	}else{
		println!("PASSED: {}", msg);
	}
}

pub fn rand_poly<F:FftField>(n: usize, seed: u128)->DensePolynomial<F>{
	let mut rng = gen_rng_from_seed(seed); 
	let f = DensePolynomial::<F>::rand(n, & mut rng);
	return f;
}

/** faster divide_with_q_and_r using the Hensel lift
(1) http://people.seas.harvard.edu/~madhusudan/MIT/ST15/scribe/lect06.pdf
(2) http://people.csail.mit.edu/madhu/ST12/scribe/lect06.pdf
*/
//#[inline(always)]
pub fn new_divide_with_q_and_r<F: FftField>(
	f: &DensePolynomial<F>, 
	g: &DensePolynomial<F>)
	->(DensePolynomial<F>, DensePolynomial<F>){
	if g.degree()>f.degree() {
		return (DensePolynomial::<F>::zero(),g.clone()); 
	}

	let rev_f = rev::<F>(&f, f.degree());
	let rev_g = rev::<F>(&g, g.degree());
	let diff_deg = f.degree()-g.degree();


	let log_diff_deg = ceil_log2(diff_deg);
	let inv_rev_g = inv(&rev_g, log_diff_deg+1);
	let prod_inv_rev_g_f = &inv_rev_g * &rev_f;
	let rev_q = mod_by_xk::<F>(&prod_inv_rev_g_f, diff_deg+1); 
	let q = rev::<F>(&rev_q, rev_q.degree());
	let degree_diff = diff_deg - q.degree();
	let q = mul_by_xk(&q, degree_diff);
	let qg = mul_poly(&q , g);
	let r = f - &qg;
	return (q,r);
}

/** adaptive: when the expected items is less than 512*256 use old;
otherwise use new_div_with_q_and_r */
//#[inline(always)]
pub fn adapt_divide_with_q_and_r<F: FftField>(
	f: &DensePolynomial<F>, 
	g: &DensePolynomial<F>)
	->(DensePolynomial<F>, DensePolynomial<F>){
	let prods = (f.degree()-g.degree())*g.degree();
	if prods<=792*396{
		return old_divide_with_q_and_r(f, g);
	}else{
		return new_divide_with_q_and_r(f, g);
	}
}

/** return the ceil of log2(n) */
pub fn ceil_log2(n: usize) -> usize{
	let mut k = log2(n);
	let res = 1<<k;
	if res<n{ k = k + 1; }
	return k as usize;
}

/** constructr polynomial */
fn build_poly<F:FftField>(v: &Vec<u64>)->DensePolynomial<F>{
	let mut v2: Vec<F> = vec![];
	for i in 0..v.len(){
		v2.push(F::from(v[i]));
	}
	let p = DensePolynomial::from_coefficients_vec(v2);
	return p;
} 

/** 
	Return the coef at the highest degree, i.e.,
	given f = a_0 + .... a_i x^i, returning a_i
*/
pub fn lead_coef<F:FftField>(f: &DensePolynomial<F>) -> F{
	if f.is_zero() {return F::zero();}
	let res = f.coeffs[f.coeffs.len()-1];
	return res;
}

/** print it nicely */
pub fn print_poly<F:FftField>(s: &str, f: &DensePolynomial<F>){
	if f.is_zero() {println!("\n==== {}: zero!", s); return;}
	println!("\n==== {}: degree: {} ====", s, f.degree());
	for i in 0..f.degree()+1{
		let j = f.degree() - i;
		let c = f.coeffs[j];
		if j==0{
			println!("{}", c);
		}else{
			println!("{}x^{}", c, j);
		} 
	}
	println!("\n");
}

/* Divide the polynomial by x^k.
	Basically just leave the portion above x^k (included)
	E.g., let p = 1 + 2x + 3x^2
	div_by_xk(p, 2) returns 3
	div_by_xk(p, 1) returnx 3x + 2
	div_by_xk(p, 4) returns 0
	div_by_xk(p, 0) returns p itself
*/
pub fn div_by_xk<F:FftField>(f: &DensePolynomial<F>, k: usize) 
	-> DensePolynomial<F>{
	let degree = f.degree();
	let pzero = DensePolynomial::<F>::zero();
	if k==0{
		return f + &pzero; 
	}else if k>degree{
		return pzero;
	}else{
		let old_len = f.coeffs.len();
		let new_coefs = &f.coeffs[k..old_len];
		let newp = DensePolynomial::<F>::from_coefficients_vec(new_coefs.to_vec());
		return newp;
	}
}

/** returning the polynomial with degree lower than k
	e.g., mod_by_xk(1 + 2x + 3x^2, 2) should return 1 + 2x
	mod_by_xk + div_by_xk should always return the original polynomial
*/
pub fn mod_by_xk<F:FftField>(f: &DensePolynomial<F>, k: usize) 
	-> DensePolynomial<F>{
	let degree = f.degree();
	let pzero = DensePolynomial::<F>::zero();
	if k==0{
		return pzero;
	}else if k>degree{
		return f + &pzero;
	}else{
		let new_coefs = &f.coeffs[0..k];
		let newp = DensePolynomial::<F>::from_coefficients_vec(new_coefs.to_vec());
		return newp;
	}
}

/** multiply polynomial by x^k */
pub fn mul_by_xk<F:FftField>(f: &DensePolynomial<F>, k: usize) -> DensePolynomial<F>{
	let pzero = DensePolynomial::<F>::zero();
	if k==0 || f.is_zero(){ return f + &pzero;}
	else{
		let mut v = vec![F::zero(); k + f.degree() + 1];
		for i in 0..k{
			v[i] = F::zero();
		}
		for i in k..k+f.degree()+1{
			v[i] = f.coeffs[i-k];
		}
		let pnew = DensePolynomial::from_coefficients_vec(v);
		return pnew;
	}
}

/** build x^k. If k is 0, return 1 */
pub fn build_xk<F:FftField>(k: usize) -> DensePolynomial<F>{
	let mut v = vec![];
	for _i in 0..k{
		v.push(F::zero());
	}
	v.push(F::one());
	let p = DensePolynomial::<F>::from_coefficients_vec(v);
	return p;
} 

/** compute the inverse of f mod x^(2^k).
	Ref: (1) http://people.seas.harvard.edu/~madhusudan/MIT/ST15/scribe/lect06.pdf
*/
pub fn inv<F:FftField>(g: &DensePolynomial<F>, k: usize)->DensePolynomial<F>{
	let mut timer = Timer::new();
	let mut timer2 = Timer::new();
	timer.start();
	timer2.start();

	//1. k = 0 case
	if g.coeffs.len()==0 || g.coeffs[0].is_zero(){panic!("INV err: coef0 can't be zero!");}
	let c0 = g.coeffs[0].inverse().unwrap();
	let zero= DensePolynomial::<F>::from_coefficients_vec(vec![F::from(0u64)]);
	let one= DensePolynomial::<F>::from_coefficients_vec(vec![F::from(1u64)]);
	let mut a = DensePolynomial::<F>::from_coefficients_vec(vec![c0]);
	let mut t = 1;
	let mut b;

	//2. iterative case
	for _u in 0..k{
		//1. compute b
		let m_a = mod_by_xk(&a, 2*t);
		let m_g = mod_by_xk(&g, 2*t);
		let ag = mul_poly(&m_a , &m_g);
		let ag_1 = &ag - &one;
		b = div_by_xk(&ag_1, t);
		b = mod_by_xk(&b, t);
		
		//2. compute a
		let a1 = mod_by_xk(&(&zero - &(&b * &a)), t);
		let xt_a1 = mul_by_xk(&a1, t);
		a = a + xt_a1;
		a = mod_by_xk(&a, 2*t);
		t = 2*t;
	}	

	return a;
}

/** Reverse the co-ef list. n may be higher than degree of f.
n is the TARGET degree
*/
pub fn rev<F:FftField>(f: &DensePolynomial<F>, deg: usize) 
	-> DensePolynomial<F>{
	if f.is_zero() {return DensePolynomial::<F>::zero();}
	let fdeg= f.degree();
	assert!(fdeg<=deg, "error: f.degree() > deg!");
	let n_zero = deg-fdeg;
	let mut v = vec![];
	for i in 0..deg+1{
		if i>=n_zero{
			v.push(f.coeffs[fdeg-(i-n_zero)]);
		}else{
			v.push(F::zero());
		}
	}
	let g = DensePolynomial::<F>::from_coefficients_vec(v);
	return g;
}

/// get the upper half of a usize
pub fn half(n: usize) -> usize{
	if n%2==0{
		return n/2;
	}else{
		return n/2 + 1;
	}
}


/// print 2x2 matrix
pub fn printm4<F: FftField>(s: &str, m: &[DensePolynomial<F>; 4]){
	print!("{}: ", s);
	for i in 0..4{
		print_poly("", &m[i]);
	}	
	println!("");
}

/// print 1x1 matrix
pub fn printm2<F: FftField>(s: &str, m: &[DensePolynomial<F>; 2]){
	println!("{}: ===== ", s);
	for i in 0..2{
		println!("{}: -----", i);
		print_poly("", &m[i]);
	}	
	println!("");
}

/// m2 x m2 -> m2
/// m2 is represented as a 4 elements 1-d array layed all elements
/// row by row
pub fn m2xm2<F: FftField>(m1: &[DensePolynomial<F>; 4], m2: &[DensePolynomial<F>; 4]) -> [DensePolynomial<F>; 4]{
/*
	let res = [
		&(&m1[0] * &m2[0]) + &(&m1[1] * &m2[2]),
		&(&m1[0] * &m2[1]) + &(&m1[1] * &m2[3]),
		&(&m1[2] * &m2[0]) + &(&m1[3] * &m2[2]),
		&(&m1[2] * &m2[1]) + &(&m1[3] * &m2[3]),
	];
*/
	let res = [
		&mul_poly(&m1[0] , &m2[0]) + &mul_poly(&m1[1] , &m2[2]),
		&mul_poly(&m1[0] , &m2[1]) + &mul_poly(&m1[1] , &m2[3]),
		&mul_poly(&m1[2] , &m2[0]) + &mul_poly(&m1[3] , &m2[2]),
		&mul_poly(&m1[2] , &m2[1]) + &mul_poly(&m1[3] , &m2[3]),
	];
	return res;
}

/// 2x2 matrix multiply with 1x1
pub fn m2xm1<F: FftField>(m1: &[DensePolynomial<F>; 4], m2: &[DensePolynomial<F>; 2]) -> [DensePolynomial<F>; 2]{
	let res = [
		&mul_poly(&m1[0],&m2[0]) + &mul_poly(&m1[1] , &m2[1]),
		&mul_poly(&m1[2],&m2[0]) + &mul_poly(&m1[3] , &m2[1]),
	];
	return res;
}

/** build a dense polynomial given coefs vector */
pub fn get_poly<F:FftField>(v: Vec<F>)-> DensePolynomial<F>{
	return DensePolynomial::<F>::from_coefficients_vec(v);
}

/// make a copy of a polynomial
pub fn cp<F:FftField>(a: &DensePolynomial<F>) -> DensePolynomial<F>{
	let total = a.degree() + 1;
	let mut res = vec![F::zero(); total];
	if a.coeffs.len()==0{
		return DensePolynomial::<F>::zero();
	}
	for i in 0..total{
		res[i] = a.coeffs[i];
	}
	let p2 = get_poly(res);
	return p2;	
}

// make the leading_cof of gcd to be 1.
pub fn get_modic<F:FftField>(gcd: &DensePolynomial<F>, s: &DensePolynomial<F>, t: &DensePolynomial<F>)->(DensePolynomial<F>, DensePolynomial<F>, DensePolynomial<F>){
	let lc = lead_coef(gcd);
	if lc.is_zero() {return (gcd.clone(), s.clone(), t.clone());}
	let inv = lc.inverse().unwrap();
	let pfactor = get_poly::<F>(vec![inv]);
	let g2 = gcd * &pfactor;
	let s2 = s * &pfactor;
	let t2 = t * &pfactor;
	return (g2, s2, t2);
}

/** perform check of the hgcd function. (a,b) is the input and res is the output
panic if error.
*/
pub fn check_hgcd<F:FftField>(a: &DensePolynomial<F>, b: &DensePolynomial<F>, res: &[DensePolynomial<F>;4]){

	let d = a.degree();
	let m = half(d);
	let res = m2xm1(res, &[cp(a),cp(b)]);	
	let c = &res[0];
	let d = &res[1];
	//println!("DEBUG USE 300: d: {}, m: {}, c.degree: {}, d.degree: {}", a.degree(), m, c.degree(), d.degree());
	let (gcd1, s, t)  = xgcd_rec(a, b);
	let (gcd1, _, _) = get_modic(&gcd1, &s, &t);
	let (gcd2, s, t)  = xgcd_rec(c, d);
	let (gcd2, _, _) = get_modic(&gcd2, &s, &t);
	if gcd1!=gcd2 {panic!("gcd1 != gcd2");}
	if c.degree()<m {panic!("c's degree < half(d)!");}
	if d.degree()>=m {panic!("d's degree >= half(d)!");}
}

/** schoolbook O(n^2) complexity multiplication. But faster than FFT
for small degree up to 64 
*/
//#[inline(always)]
pub fn small_mul_poly<F:FftField>(a: &DensePolynomial<F>, b: &DensePolynomial<F>) -> DensePolynomial<F>{
	if a.is_zero() || b.is_zero() {return DensePolynomial::<F>::zero();}
	let n1 = a.degree();
	let n2 = b.degree();
	let n = n1 + n2+1;
	let mut vec:Vec<F> = vec![F::zero();n];
	let c1 = &a.coeffs;
	let c2 = &b.coeffs;
	for i in 0..n1+1{
		for j in 0..n2+1{
			vec[i+j] += c1[i] * c2[j];
		}
	} 	
	let res = DensePolynomial::<F>::from_coefficients_vec(vec);
	return res;
}

//#[inline(always)]
pub fn mul_poly<F:FftField>(a: &DensePolynomial<F>, b: &DensePolynomial<F>) -> DensePolynomial<F>{
	if a.degree() + b.degree()<=196{
		return small_mul_poly(a, b);
	}else{
		return a * b;
	}
}

/// return the half gcd matrix accoring to ref
/// Performance: 10k -> 12 seconds, 100k->124 seconds, 1 million: 1287 seconds
/// *** NOTE *** somehow ONLY works when the two polynomials are co-prime
/// For a and b have factors, occasinally returned polynomials
/// DO NOT STRADDLE over a.degree()/2, strangely (need the sequence of
/// egcd-matrix to be nomal!
pub fn hgcd_worker<F:FftField>(a: &DensePolynomial<F>, b: &DensePolynomial<F>) -> [DensePolynomial<F>;4]{
	let d = a.degree();
	let m = half(d);
	let zerop = DensePolynomial::<F>::zero();
	let one = build_poly(&vec![1u64]);
	let zero= build_poly(&vec![0u64]);
	if b.degree()<m || b.degree()==0{
		let res = [cp(&one), cp(&zero), cp(&zero), cp(&one)];	
		return res;
	}

	let a1 = div_by_xk(&a, m);
	let b1 = div_by_xk(&b, m);
	let mat1 = hgcd(&a1, &b1);
	let cp_a = cp(a);
	let cp_b = cp(b);
	let res= m2xm1(&mat1, &[cp_a, cp_b]);
	let t = &res[0];
	let s = &res[1];
	if s.degree()<m{ 
		return mat1;
	}
	let (q,r) = adapt_divide_with_q_and_r(&t, &s);
	if r==zerop{
		let negq = &zerop - &q;
		let mat2 = [cp(&zero), cp(&one), cp(&one), cp(&negq)]; 
		let res = m2xm2(&mat2, &mat1);
		return res;
	}

	let v = lead_coef(&r);
	let pv = get_poly(vec![v]);
	let rbar = mul_poly(&r , &pv);
	let vq = mul_poly(&pv , &q);
	let negvq = &zerop - &vq;
	let mat2 = [cp(&zero), cp(&one), cp(&pv), cp(&negvq)];
	let l = 2*m - s.degree();
	let s1 = div_by_xk(&s, l);
	let r1 = div_by_xk(&rbar, l);
	let mat3 = hgcd(&s1, &r1);
	let res1 = m2xm2(&mat3, &mat2);
	let res = m2xm2(&res1, &mat1);
	return res;
}

/** call xgcd_special if degree is low */
//#[inline(always)]
pub fn hgcd<F:FftField>(a: &DensePolynomial<F>, b: &DensePolynomial<F>) -> [DensePolynomial<F>;4]{
	if a.degree()<=256{
		let (_,_,s1,t1,s2,t2) = xgcd_special(a, b, half(a.degree()));
		return [s1, t1, s2, t2];
	}else{
		return hgcd_worker(a, b);
	}	
}
// /** check if mgcd is doing the job */
//pub fn check_mgcd(a: &DensePolynomial<F>, b: &DensePolynomial<F>, res: &[DensePolynomial<F>;4]){
//}
/// requires that degree of a>=b. Output 2x2 M s.t. M(a b) = (gcd(a,b) 0)
pub fn mgcd<F:FftField>(a: &DensePolynomial<F>, b: &DensePolynomial<F>) -> [DensePolynomial<F>;4]{
	let zerop = DensePolynomial::<F>::zero();
	let one = build_poly(&vec![1u64]);
	let zero= build_poly(&vec![0u64]);
	let mat1 = hgcd(a, b);
	let res = m2xm1(&mat1, &[cp(a), cp(b)]);
	let t = &res[0];
	let s = &res[1];
	if *s==zerop{ 
		return mat1;
	}

	let (q,r) = adapt_divide_with_q_and_r(t, s);
	if r==zerop{
		let negq = &zerop - &q;
		let mat2 = [cp(&zero), cp(&one), cp(&one), cp(&negq)]; 
		let res = m2xm2(&mat2, &mat1);
		return res;
	}

	let v = lead_coef(&r);
	let pv = get_poly(vec![v]);
	let rbar = mul_poly(&r , &pv);
	let vq = mul_poly(&pv , &q);
	let negvq = &zerop - &vq;
	let mat2 = [cp(&zero), cp(&one), cp(&pv), cp(&negvq)];
	let mat3 = mgcd(&s, &rbar);
	let res = m2xm2(&m2xm2(&mat3, &mat2), &mat1);
	return res;
}

/// returns gcd(a,b), s and t s.t. sa + tb = gcd(a,b)
/// Performance: asm 0.3.0 560 sec for 1M entries
/// used to have 450 sec for 1M in 0.2.0 though
pub fn feea_worker<F:FftField>(a: &DensePolynomial<F>, b: &DensePolynomial<F>) -> (DensePolynomial<F>, DensePolynomial<F>, DensePolynomial<F>){
//	let mut timer = Timer::new();
//	timer.start();
	if a.degree()>b.degree(){
		let mat= mgcd(a, b);
		let s = &mat[0];
		let t = &mat[1];
		let g = &mul_poly(s,a) +  &mul_poly(t,b);
//		timer.stop();
//		println!("feea worker case 1. size: {}, time: {} us", a.degree(), timer.time_us);
		return (g, cp(s), cp(t));
	}else if a.degree()==b.degree(){
		//THIS PART of the algorithm should be fixed
		//Let q = a/b (a constant as degree is the same)
		//Let sb + t(a - qb) = g  // here s,t is the return of feea(b,&modres)
		//Let s'b + ta = g
		// This leads to s' = s-tq
		let (q, r) = adapt_divide_with_q_and_r(a, b);
		let (g, s, t) = feea_worker(b, &r); 
		let sprime = &s - &mul_poly(&t , &q); 
//		timer.stop();
//		println!("feea worker case 2. size: {}, time: {} us", a.degree(), timer.time_us);
		return (g, t, sprime);
	}else{
		//THIS PART of the algorithm of the original program should be fixed
		let (g, s, t) = feea_worker(b, a);
//		timer.stop();
//		println!("feea worker case 3. size: {}, time: {} us", a.degree(), timer.time_us);
		return (g,t,s);
	}
}

//THIS IS THE ONE TO CALL: fast Eucledian algorithm
/// returns gcd(a,b), s and t s.t. sa + tb = gcd(a,b)
/// Performance: 1M entries - 500 seconds sec (100 times of mul)
pub fn feea<F:FftField>(a: &DensePolynomial<F>, b: &DensePolynomial<F>) -> (DensePolynomial<F>, DensePolynomial<F>, DensePolynomial<F>){
//let mut timer = Timer::new();
//	timer.start();
	let (g,s,t) = feea_worker(a, b);
	let v = lead_coef(&g);
	if v.is_zero() {
		return (g,s,t);
	}

	let pv2 = get_poly(vec![v]);
	let g2 = &g / &pv2;
	let s2 = &s / &pv2;
	let t2 = &t / &pv2; 
//	timer.stop();
//	println!("FEEA size: {}, time: {} us. Node: {}", a.degree(), timer.time_us, RUN_CONFIG.my_rank);
	return (g2, s2, t2);
}

/// The classical extended Eucledian algorithm
pub fn xgcd<F:FftField>(a: &DensePolynomial<F>, b: &DensePolynomial<F>) 
	-> (DensePolynomial<F>, DensePolynomial<F>, DensePolynomial<F>){
	let mut p1 = a;
	let mut p2 = b;
	if a.degree()<b.degree(){
		p1 = b;
		p2 = a;
	}

	//1. init
	let mut old_r = p1.clone();
	let mut r = p2.clone();
	let mut old_s = get_poly::<F>(vec![F::one()]);
	let mut s = get_poly::<F>(vec![F::zero()]);
	let mut old_t = get_poly::<F>(vec![F::zero()]);
	let mut t = get_poly::<F>(vec![F::one()]);


	//2. loop
	while !r.is_zero() {
		let (q,_) = adapt_divide_with_q_and_r(&old_r, &r);

		let new_r = &old_r - &(&q * &r);
		old_r = r;
		r = new_r;

		let new_s = &old_s - &(&q * &s);
		old_s = s;
		s = new_s;
		
		let new_t = &old_t - &(&q * &t);
		old_t = t;
		t = new_t;
	}

	return (old_r, old_s, old_t);
}

/* recursive version */
pub fn xgcd_rec<F:FftField>(a: &DensePolynomial<F>, b: &DensePolynomial<F>) 
	-> (DensePolynomial<F>, DensePolynomial<F>, DensePolynomial<F>){
	let mut p1 = a;
	let mut p2 = b;
	if a.degree()<b.degree(){
		p1 = b;
		p2 = a;
	}
	if p2.is_zero(){
		return (p1.clone(), get_poly::<F>(vec![F::one()]), get_poly::<F>(vec![F::zero()]));
	}
	let (q, r) = adapt_divide_with_q_and_r(&p1, &p2);
	let (gcd, s, t) = xgcd_rec(&p2, &r);
	let t1 = &s - &(&t * &q);
	let s1 = t;
	return (gcd, s1, t1);
}

/* recursive version , STOP when the degree of b is LESS THAN the bar
	return c = s*a + t*b, d = s2*a + t2*b.
	d.degree()<bar
*/
pub fn xgcd_rec_special<F:FftField>(a: &DensePolynomial<F>, b: &DensePolynomial<F>, bar: usize) 
	-> (DensePolynomial<F>, DensePolynomial<F>, DensePolynomial<F>){
	let mut p1 = a;
	let mut p2 = b;
	if a.degree()<b.degree(){
		p1 = b;
		p2 = a;
	}
	if p2.is_zero(){
		return (p1.clone(), get_poly::<F>(vec![F::one()]), get_poly::<F>(vec![F::zero()]));
	}
	if  p2.degree()<bar{
		if p1.degree()<bar {panic!("a's degree: {} is also < bar: {}", p1.degree(), bar);}
		//the value of gcd is unkonwn but the (s,t) is ok
		return (get_poly::<F>(vec![F::zero()]), get_poly::<F>(vec![F::one()]), get_poly::<F>(vec![F::one()]));
	}
	let (q, r) = adapt_divide_with_q_and_r(&p1, &p2);
	let (gcd, s, t) = xgcd_rec_special(&p2, &r, bar);
	let t1 = &s - &(&t * &q);
	let s1 = t;
	return (gcd, s1, t1);
}

/** returns (c, d, s1, t1, s2, t2) s.t. (s1*a + t1*b = c and s2*a+t2*b=d)
and then gcd(c,d) is gcd(a,b) and d.degree<limit and c.degree>=limit
When limit is ceil(a.degree/2), it is equivalent to hgcd.
Assumption: *** a.degree>=b.degree ***
Use the traiditonal extended eucledian alg
*/
//#[inline(always)]
pub fn xgcd_special_worker<F:FftField>(a: &DensePolynomial<F>, b: &DensePolynomial<F>, limit: usize) -> (DensePolynomial<F>, DensePolynomial<F>, DensePolynomial<F>, DensePolynomial<F>, DensePolynomial<F>, DensePolynomial<F>){ 
	//1. Initlize: each row is [a, s, t]
	let pzero = DensePolynomial::<F>::zero();
	let pone = get_poly::<F>(vec![F::one()]);
	let mut row2d = [ 
		[a.clone(), pone.clone(), pzero.clone()],
		[b.clone(), pzero.clone(), pone.clone()],
		[pzero.clone(), pzero.clone(), pzero.clone()]
	];
	let mut idx  = 0;

	while row2d[(idx+1)%3][0].degree()>=limit{
		let (q,r) = adapt_divide_with_q_and_r(
			&row2d[idx][0], &row2d[(idx+1)%3][0]);
		row2d[(idx+2)%3][0] = r;
		row2d[(idx+2)%3][1] = &row2d[idx][1] - &mul_poly(&row2d[(idx+1)%3][1], &q);
		row2d[(idx+2)%3][2] = &row2d[idx][2] - &mul_poly(&row2d[(idx+1)%3][2], &q);
		idx = (idx+1)%3;
	}
	
	return (
		row2d[idx][0].clone(), //c
		row2d[(idx+1)%3][0].clone(), //d
		row2d[idx][1].clone(), //s1
		row2d[idx][2].clone(), //t1
		row2d[(idx+1)%3][1].clone(), //s2
		row2d[(idx+1)%3][2].clone(), //t2
	);
}

//#[inline(always)]
pub fn xgcd_special<F:FftField>(a: &DensePolynomial<F>, b: &DensePolynomial<F>, limit: usize) -> (DensePolynomial<F>, DensePolynomial<F>, DensePolynomial<F>, DensePolynomial<F>, DensePolynomial<F>, DensePolynomial<F>){ 
	if a.degree()>=b.degree(){
		return xgcd_special_worker(a, b, limit);
	}else{
		return xgcd_special_worker(b, a, limit);
	}
}






/* write the coefficients to the given file. If not enough, pad with zero.
The 1st line is the coef 0 (lowest degree */
pub fn write_poly_to_file<F:FftField+CanonicalSerialize>(p: &DensePolynomial<F>, fname: &str, lines: usize){
	let n = p.coeffs.len();
	if lines<n{panic!("lines: {} < n: {}", lines, n);}
	let mut vec = p.coeffs.clone();
	for _i in 0..lines-n{
		vec.push(F::zero());
	}	
	println!("DEBUG USE 500: write_arr_fr: {}, size: {}, [0]: {}, [1]: {}",
		&fname, vec.len(), vec[0], vec[1]);
	write_arr_fr::<F>(&vec, &fname.to_string());
}

pub fn get_derivative<F:FftField>(p: &DensePolynomial<F>)->DensePolynomial<F>{
	let old_coefs = &p.coeffs;
	let d = (&p).degree();
	let mut coefs = vec![F::zero(); d];
	for i in 0..d{ //d-1 included
		let newv = old_coefs[i+1].mul(F::from((i+1) as u64));
		coefs[i] = newv;
	}	
	if coefs.len()==0{
		return DensePolynomial::<F>::zero();
	}
	let p2 = get_poly::<F>(coefs);
	return p2;
}

// ---------------------------------------------------
// -------------- FFT RELATED FUNCTIONS ---------------
// ---------------------------------------------------
/** Just a awrapper of ark-poly fft, in place. changes vec, require
vec size to be power of 2. */
pub fn serial_fft<F: FftField>(vec: &mut Vec<F>){
	let b_perf = false;
	let mut t1 = Timer::new();
	t1.start();

	let n = vec.len();
	assert!(n.is_power_of_two(), "vec size is not power of 2");
	let domain = GeneralEvaluationDomain::<F>::new(n).
		expect("field is not smooth enough to construct domain");
	domain.fft_in_place(vec);
	if b_perf{log_perf(LOG1, &format!("------ ---- -- serial_fft: {}", vec.len()), &mut t1);}
}

/** Just a wrapper of ark-poly ifft, in place. Change vec. require
vec size to be power of 2 */
pub fn serial_ifft<F: FftField>(vec: &mut Vec<F>){
	let n = vec.len();
	assert!(n.is_power_of_two(), "vec size is not power of 2");
	let domain = GeneralEvaluationDomain::<F>::new(vec.len()).
		expect("field is not smooth enough to construct domain");
	domain.ifft_in_place(vec);
}
/** return a set of it */
pub fn get_set<F:FftField>(v: &Vec<F>)->Vec<F>{
	let mut res:Vec<F> = vec![];
	let mut set:HashSet<F> = HashSet::new();
	for ele in v{
		if !set.contains(ele){
			set.insert(*ele);
			res.push(*ele);
		}
	}
	return res;
}

/// apply ifft on v. If v's length is not power of 2, expand it to
/// power of 2. Given a list of y values, it generates the
/// co-efficients of a polynomial p(x) which p(omega^i) = y_i
/// where omega is the n'th root of unity.
pub fn ifft<F:FftField>(v: &Vec<F>) -> Vec<F>{
	//1. expand to power of 2 if necessary
	let pow2 = closest_pow2(v.len());	
	let mut res = v.clone();
	if pow2>v.len(){ res.resize(pow2, F::zero());}
	
	//2. cal the ifft wrapper
	serial_ifft(&mut res);
	return res;
}

/// applying fft on (omega*t)
/// calculates \sum a_i (omega*t)^i
pub fn fft_coset<F:FftField>(v: &Vec<F>, t: F) -> Vec<F>{
	//1. expand to power of 2 if necessary
	let pow2 = closest_pow2(v.len());	
	let mut res = v.clone();
	if pow2>v.len(){ res.resize(pow2, F::zero());}

	//2. call the fft wrapper
	let mut coset = t; 
	for i in 1..pow2{
		res[i] = res[i] * coset;
		coset = coset * t;
	}
	serial_fft(&mut res);
	return res;
}

pub fn inplace_fft_coset<F:FftField>(res: &mut Vec<F>, t: F){
	assert!(res.len().is_power_of_two(), "input size not power of 2");
	let mut coset = t; 
	let res_len = res.len();
	for i in 1..res_len{
		res[i] = res[i] * coset;
		coset = coset * t;
	}
	serial_fft(res);
}
/// applying ifft on (omega*t)
pub fn ifft_coset<F:FftField>(v: &Vec<F>, t: F) -> Vec<F>{
	//1. expand to power of 2 if necessary
	let pow2 = closest_pow2(v.len());	
	let mut res = v.clone();
	if pow2>v.len(){ res.resize(pow2, F::zero());}
	
	//2. cal the ifft wrapper
	serial_ifft(&mut res);
	let t = t.inverse().unwrap();
	let mut coset = t; 
	for i in 1..pow2{
		res[i] = res[i] * coset;
		coset = coset * t;
	}
	return res;
}

/// applying ifft on (omega*t)
pub fn inplace_ifft_coset<F:FftField>(res: &mut Vec<F>, t: F){
	assert!(res.len().is_power_of_two(), "input size not power of 2");
	serial_ifft(res);
	let t = t.inverse().unwrap();
	let mut coset = t; 
	let pow2 = res.len();
	for i in 1..pow2{
		res[i] = res[i] * coset;
		coset = coset * t;
	}
}

#[cfg(test)]
mod tests {
	extern crate ark_std;
	extern crate ark_ff;

	use self::ark_std::rand::Rng;
	use self::ark_ff::{UniformRand,Field};
	use crate::poly::dis_poly::*;
	use crate::tools::*;
	use crate::poly::serial::*;
	use self::ark_poly::{univariate::DensePolynomial};
	type Fr381=ark_bls12_381::Fr;


	fn test_div<F:FftField>(f: &DensePolynomial<F>, g: &DensePolynomial<F>){
		let (f1,g1) = new_divide_with_q_and_r(&f, &g);
		let (f2,g2) = old_divide_with_q_and_r(&f, &g);
		assert_eq!(f1, f2, "f1!=f2");
		assert_eq!(g1, g2, "g1!=g2");
		let f3 = &(&f1*g) + &g1;
		assert_eq!(*f, f3, "f1!=f3");
	}

	#[test]
	fn quick_test_div(){
		let n = 32;
  		let mut rng = gen_rng();
		for n2 in 1..n-1{
    		let f = DensePolynomial::<Fr381>::rand(n, & mut rng);
			let g = DensePolynomial::<Fr381>::rand(n2, & mut rng);
			test_div(&f, &g);
		}
	}

	#[test]
	fn quick_test_div2(){
		let n = 29;
  		let mut rng = gen_rng();
		let mut vec = vec![];
		for _i in 0..n{
			let v:u64 = rng.gen::<u64>() % 8u64+ 1;
			vec.push(Fr381::from(v));
		}
		let set_vec = get_set(&vec);
		let f = DisPoly::<Fr381>::binacc_poly(&vec);
		let g = DisPoly::<Fr381>::binacc_poly(&set_vec);
		test_div(&f, &g);
	}

	#[test]
	fn quick_test_div3(){
		let v64:Vec<u64> = vec![0 , 0 , 560 , 0 , 0 , 589 , 0 , 570 , 571 , 572 , 0 , 0 , 0 , 560 , 0 , 0 , 0 , 293 , 0 , 589 , 589 , 589 , 589 , 589 , 589 , 589 , 589 , 589 , 589];
		let n = v64.len();
		let mut vec = vec![];
		for i in 0..n{
			let v:u64 = v64[i];
			vec.push(Fr381::from(v));
		}
		let set_vec = get_set(&vec);
		let f = DisPoly::<Fr381>::binacc_poly(&vec);
		let g = DisPoly::<Fr381>::binacc_poly(&set_vec);
		test_div(&f, &g);
	}


	#[test]
	fn quick_test_utility_funcs(){
		//1. test the lead_coef
		let f = build_poly::<Fr381>(&vec![1u64, 2u64, 3u64]);
		let c = lead_coef(&f);
		assert_eq!(c, Fr381::from(3u64), "lead_coef err!");

		//2. test the div_by_xk
		let g = div_by_xk(&f, 2);
		assert_eq!(0, g.degree(), "div_by_xk not working, error on degree");
		let c = lead_coef(&g);
		assert_eq!(c, Fr381::from(3u64), "div_by_xk not working, error on lead co-ef");

		//3. test the mod_by_xk
		let _g2 = mod_by_xk(&f, 4);

		//4. test mod_by_xk + div_by_xk always get the same
  		let mut rng = gen_rng();
		let degree = 25;
		let f = DensePolynomial::<Fr381>::rand(degree, &mut rng);
		for i in 0..degree+5{
			let g1 = div_by_xk(&f, i);
			let g2 = mod_by_xk(&f, i);
			let g3 = &g2 + &mul_by_xk(&g1, i);
			//println!("\n\n ----!!!! i: {} !!! ----", i);
			//print_poly::<Fr381>("f", &f);
			//print_poly::<Fr381>("g1", &g1);
			//print_poly::<Fr381>("g2", &g2);
			//print_poly::<Fr381>("g3", &g3);
			assert_eq!(g3, f, "integration testing of mod + div_by_xk failed");
		}
	}

	#[test]
	fn quick_test_inv(){
  		let mut rng = gen_rng();
		let degree = 128;
		let f = DensePolynomial::<Fr381>::rand(degree, &mut rng);
		let pone = build_poly::<Fr381>(&vec![1u64]);
		let k = ceil_log2(f.degree()+1);
		for i in 0..(k+1) {
			let invf = inv::<Fr381>(&f, i);
			let prod_1 = &f * &invf;
			let prod = mod_by_xk(&prod_1, 1<<i);
			//println!("---------- i: {} ----------", i);
			//print_poly::<Fr381>("invf", &invf);
			//print_poly::<Fr381>("prod_1", &prod_1);
			//print_poly::<Fr381>("prod", &prod);
			//print_poly::<Fr381>("pone", &pone);
			assert_eq!(prod, pone, "failed quick_test_inv");	
		}
	}

	#[test]
	pub fn quick_test_feea(){
		//test the fast euclidean algorithm
  		let mut rng = gen_rng();
		let n = 32;
/*
  		let mut rng = gen_rng();
		let mut vec = vec![];
		for i in 0..n{
			let v:u64 = rng.gen::<u64>() % 3u64+ 1;
			vec.push(Fr381::from(v));
		}

		let pone = get_poly(vec![Fr381::from(1u64)]);
    	let p1 = DisPoly::<Fr381>::binacc_poly(&vec);
		let p2 = get_derivative(&p1);
*/
    	let p1 = DensePolynomial::<Fr381>::rand(n, & mut rng);
    	let p2 = DensePolynomial::<Fr381>::rand(n, & mut rng);
		let (p3, s, t) = feea(&p1, &p2);
		let (p3, s, t) = get_modic(&p3, &s, &t);
		let g3 = &(&s * &p1) + &(&t * &p2); 
		assert_eq!(p3, g3, "failed fast eucledian test!");

		let (p4, s2, t2) = xgcd_rec(&p1, &p2);
		let (p4, s2, t2) = get_modic(&p4, &s2, &t2);
		let g2 = &(&s2 * &p1) + &(&t2 * &p2); 
		assert_eq!(p4, g2, "xgcd failed!");
		assert_eq!(p3, p4, "failed fast eucledian test! p3!=p4");	
	}

	#[test]
	pub fn quick_test_set_support(){
		//test if the computing of support is correct.
		let n = 123;
  		let mut rng = gen_rng();
		let mut vec = vec![];
		for _i in 0..n{
			let v:u64 = rng.gen::<u64>() % 32u64 + 1;
			vec.push(Fr381::from(v));
		}
		let vec_set = get_set(&vec);

    	let p = DisPoly::<Fr381>::binacc_poly(&vec);
		let pd = get_derivative::<Fr381>(&p);
		let pset= DisPoly::<Fr381>::binacc_poly(&vec_set);
		let (gcd2, s3, t3) = xgcd_rec(&p, &pd);
		let (gcd2, _, _) = get_modic(&gcd2, &s3, &t3);
		let (pset2, r1) = adapt_divide_with_q_and_r(&p, &gcd2);
		assert!(pset2==pset, "ERROR in generating gcd");
		assert!(r1.is_zero(), "r1 is not zero!");

		let p_gcd = pset;
		let (pd_gcd, r2) = adapt_divide_with_q_and_r(&pd, &gcd2);
		assert!(r2.is_zero(), "r2 is not zero!");

		let (_g3, s, t) = feea(&p_gcd, &pd_gcd);
		let g3 = &(&s * &p_gcd) + &(&t * &pd_gcd); 
		let pone = get_poly(vec![Fr381::from(1u64)]);
		assert_eq!(g3, pone, "g3 is not one!");
		
	}

	#[test]
	pub fn test_small_mul(){
		let n = 64;
  		let mut rng = gen_rng();
    	let p1 = DensePolynomial::<Fr381>::rand(n, & mut rng);
    	let p2 = DensePolynomial::<Fr381>::rand(n, & mut rng);
		
		let r1 = small_mul_poly(&p1, &p2);
		let r2 = &p1 * &p2;
		assert!(r1==r2, "failed test small mul");
	}

	#[test]
	pub fn test_xgcd_special(){
		let n = 64;
  		let mut rng = gen_rng();
    	let p1 = DensePolynomial::<Fr381>::rand(n, & mut rng);
    	let p2 = DensePolynomial::<Fr381>::rand(n, & mut rng);
		let (c,d,s1,t1,s2,t2) = xgcd_special(&p1, &p2, 1);
		let c2 = &mul_poly(&s1, &p1) + &mul_poly(&t1, &p2);
		let d2 = &mul_poly(&s2, &p1) + &mul_poly(&t2, &p2);
		assert!(c==c2, "failed c==c2for test_xgcd");	
		assert!(d==d2, "failed d==d2for test_xgcd");	
		assert!(d.degree()==0, "failed d.degree()==0 for test_xgcd");	

		let (gcd, s,t) = xgcd_rec(&p1, &p2);
		let (gcd, _, _) = get_modic(&gcd, &s, &t);
		let (gcd2, s2,t2) = xgcd_rec(&c, &d);
		let (gcd2, _, _) = get_modic(&gcd2, &s2, &t2);
		assert!(gcd2==gcd, "failed gcd2==gcd for test_xgcd");	
	}

	#[test]
	pub fn test_xgcd_hgcd(){ //verify it can be used as hgcd
		let n = 64;
  		let mut rng = gen_rng();
    	let p1 = DensePolynomial::<Fr381>::rand(n, & mut rng);
    	let p2 = DensePolynomial::<Fr381>::rand(n, & mut rng);
		let limit = half(p1.degree());
		let (c,d,_,_,_,_) = xgcd_special(&p1, &p2, limit);
		let mat = hgcd(&p1, &p2);
		let res = m2xm1(&mat, &[cp(&p1), cp(&p2)]);
		let (gcd1, s1, t1) = xgcd_rec(&c, &d);	
		let (gcd2, s2, t2) = xgcd_rec(&res[0], &res[1]);	
		let (gcd1, _, _) = get_modic(&gcd1, &s1, &t1);
		let (gcd2, _, _) = get_modic(&gcd2, &s2, &t2);
		assert!(gcd1==gcd2, "failed gcd1=gcd2 for test_xgcd_hgcd");	
		assert!(c.degree()==res[0].degree(), "c1.degree!=c2.degree");
		assert!(d.degree()==res[1].degree(), "d1.degree!=d2.degree");
	}

	/// generate a random vector of field elements
	fn rand_vec_fe(n: usize)->Vec<Fr381>{
		let mut vec = vec![Fr381::zero(); n];
		let mut rng = gen_rng();
		for i in 0..n{
			vec[i] = Fr381::rand(&mut rng);
		}
		return vec;
	}

	#[test]
	fn test_ifft(){
		let n = 12;
		let vec = rand_vec_fe(n);
		let vec_coef = ifft(&vec);
		let n2 = vec_coef.len() as u64;
		let omega = Fr381::get_root_of_unity(n2).unwrap();
		let p = DensePolynomial::<Fr381>::from_coefficients_vec(vec_coef);
		for i in 0..n{
			let omega_i = omega.pow(&[i as u64]);
			let v2 = p.evaluate(&omega_i);
			assert!(v2==vec[i], "ifft test fails on i: {}, v1: {}, v2: {}",
				i, vec[i], v2);
		}
	}	
}
