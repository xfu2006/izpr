/** 
	Copyright Dr. Xiang Fu

	Author: Dr. Xiang Fu
	All Rights Reserved.
	Created: 02/2022
	Revised: 08/23/2022: Added functions for publisher

	This is the main commnd line console file for processing various 
application senarios (profiling, publisher, prover, verifier etc.)
*/
extern crate acc;
extern crate ark_ff;
extern crate ark_ec;
extern crate ark_poly;
extern crate ark_std;
extern crate ark_bls12_381;
extern crate ark_serialize;
extern crate ark_bn254;

#[cfg(feature = "parallel")]
use self::ark_std::cmp::max;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::env;


//use self::mpi::traits::*;
use ark_ec::bls12::Bls12;
use ark_ec::ProjectiveCurve;

/*
use acc::profiler::config::*;
use acc::profiler::profile_pairing::*;
use acc::profiler::profile_fft::*;
use acc::profiler::profile_poly::*;
use acc::profiler::profile_proto::*;
use acc::profiler::profile_r1cs::*;
use acc::profiler::profile_groth16::*;
use acc::profiler::profile_group::*;
use acc::poly::dis_poly::*;
use acc::poly::group_dis_vec::*;
use acc::poly::dis_vec::*;
use acc::poly::serial::*;
use acc::proto::*;
use acc::proto::zk_subset_v3::*;
use acc::proto::zk_kzg_v2::*;
use acc::proto::proto_tests::{get_max_test_size_for_key};
use acc::proto::nonzk_sigma::*;
use acc::circwitness::serial_circ_gen::*;
use acc::circwitness::modular_circ_gen::*;
use acc::groth16::dis_prover::*;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
//use acc::groth16::common::*;
use acc::groth16::verifier::*;
use acc::groth16::aggregate::*;
use acc::groth16::dis_prove_key::*;
use acc::groth16::new_dis_qap::*;
use acc::zkregex::prover::*;
use acc::zkregex::batch_prover::*;
//use acc::jsnark_driver::new_jd_tools::*;
use acc::poly::common::*;
use acc::proto::ripp_driver::*;
use acc::zkregex::aggregate::*;
*/

extern crate once_cell;
use std::rc::Rc;
//use std::borrow::Borrow;

use self::ark_ec::{PairingEngine};
use self::ark_ff::{PrimeField,UniformRand,to_bytes};


//use acc::poly::disfft::*;
use ark_ff::{Zero};
//use acc::poly::serial::*;
use acc::tools::*;
use std::marker::PhantomData;
use self::ark_ec::{AffineCurve};
use self::ark_ec::msm::{VariableBaseMSM};
use self::ark_poly::Polynomial;
//use self::ark_poly::{EvaluationDomain};

use self::ark_bn254::Bn254;
type Fr = ark_bn254::Fr;
type PE= Bn254;
use self::ark_bls12_381::Bls12_381;
//use acc::proto::ripp_driver::*;

type Fr381=ark_bls12_381::Fr;
type PE381=Bls12_381;

/// just to use some of the definitions if some code is commented out
fn phantom_func(){
	let _d1: PhantomData<Fr381>;
	let _d2: PhantomData<PE381>;
}


fn main() {
}

