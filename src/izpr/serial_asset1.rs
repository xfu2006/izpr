/*
    Copyright Dr. Xiang Fu

    Author: Xiang Fu
    All Rights Reserved.
    Created: 08/03/2023

	This files implements the zk-asset1 protocol
*/
extern crate ark_ff;
extern crate ark_ec;
extern crate ark_poly;
extern crate ark_serialize;
extern crate ark_std;
extern crate mpi;
extern crate ark_bls12_381;
extern crate acc;

use self::ark_ec::{
  	//AffineCurve, 
	PairingEngine, 
	//ProjectiveCurve
};
//use self::ark_ff::{
	//FftField, 
	//One,	
//};


//use self::ark_bls12_381::Bls12_381;
//type Fr381 = ark_bls12_381::Fr;
//type PE381 = Bls12_381;

//use self::ark_poly::{
 //   univariate::DenseOrSparsePolynomial, univariate::DensePolynomial, DenseUVPolynomial, Polynomial,
//};
//use self::ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
//use self::ark_serialize::CanonicalSerialize;
//use self::ark_std::log2;
//use izpr::serial_poly_utils::acc::poly::common::*;
//use izpr::serial_poly_utils::acc::tools::*;
//use std::collections::HashSet;
//use izpr::serial_poly_utils::acc::poly::common::closest_pow2;

//use self::ark_ec::{AffineCurve, ProjectiveCurve};
//use self::ark_ff::{UniformRand};
//use self::ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
//use self::ark_ec::msm::{VariableBaseMSM};
use izpr::serial_poly_utils::{KzgProverKey};


/// The Prover Key of Asset Protoocl 1
#[derive(Clone)]
pub struct AssetOneProverKey<PE:PairingEngine>{
	pub log_bound: usize,
	pub _del_later: PE::G1Affine,
}

/// The Verifier Key
#[derive(Clone)]
pub struct AssetOneVerifierKey<PE:PairingEngine>{
	pub _del_later: PE::G1Affine,
}

/// Proof
#[derive(Clone)]
pub struct AssetOneProof<PE:PairingEngine>{
	pub _del_later: PE::G1Affine,
}


/// generate the prover and verifier keys
pub fn setup_asset_one<PE:PairingEngine>(_pk: &KzgProverKey<PE>, 
	_log_bound: usize,
	_asset_table: &Vec<PE::Fr>) 
	-> (AssetOneProverKey<PE>, AssetOneVerifierKey<PE>){
	unimplemented!("not done");
}

/// prove the asset change is the claimed commit_total
/// note that the asset table of the organization is already
/// cotained in prover key.
/// r_a, r_c are the randoms used for generating commitments to
///  accounts and changes. r_total is the random nonce for 
///  the commitment to total
/// return (commit_total, proof)
pub fn prove_asset_one<PE:PairingEngine>(
	_pk: &AssetOneProverKey<PE>, 
	_accounts: &Vec<PE::Fr>, 
	_r_a: PE::Fr, 
	_changes: &Vec<PE::Fr>, 
	_r_c: PE::Fr, 
	_r_total: PE::Fr)
	-> (PE::G1Affine, AssetOneProof<PE>){
	unimplemented!("not done");
} 

/// verify the proof (the total change is the sum of
///  those changes in accounts that belong to assets table
pub fn verify_asset_one<PE:PairingEngine>(_vk: &AssetOneVerifierKey<PE>,
	_commit_assets: PE::G1Affine, 
	_commit_accounts: PE::G1Affine, 
	_commit_changes: PE::G1Affine, 
	_commit_total_change: PE::G1Affine, 
	_prf: &AssetOneProof<PE>)-> bool{
	unimplemented!("not done");
}

