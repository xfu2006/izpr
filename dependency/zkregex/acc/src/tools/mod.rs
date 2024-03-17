/* 
	Copyright Dr. Xiang Fu

	Author: Dr. Xiang Fu
	All Rights Reserved.
	Created: 02/15/2022
*/

/* Thie file defines several utility classes such as timer,
logger etc.
*/
extern crate ark_ff;
extern crate ark_ec;
extern crate ark_poly;
extern crate ark_std;
extern crate ark_serialize;
//extern crate mpi;
extern crate sha2;
extern crate num_bigint;
extern crate sysinfo;

//use std::thread;
use std::time::Instant;
//use self::mpi::traits::*;
use std::{path::{Path,PathBuf},process};
use self::sysinfo::{Pid,ProcessExt, System, SystemExt};
use profiler::config::*;
use self::ark_std::rand::rngs::StdRng;
use self::ark_std::rand::Rng;
use self::ark_std::rand::SeedableRng;
use std::time::{SystemTime, UNIX_EPOCH};
use self::ark_ff::{FftField,PrimeField};
use std::fs;
use std::io::{Read,BufRead, BufReader, BufWriter, Write};
use std::collections::HashSet;
use std::collections::HashMap;
use std::process::Command;
use std::str::from_utf8;
use std::{thread,time};
use self::ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
//use hex_literal::hex;
use self::sha2::{Sha256,Digest};
use self::ark_ec::{AffineCurve,PairingEngine};
use self::num_bigint::BigUint;
use profiler::config::{RUN_CONFIG};
use poly::common::*;

pub const WORK_DIR:usize = 101;
pub const SWORK_DIR:&str = "101";

/// timer class for recording time in microseconds
pub struct Timer{
	///running time in micro-second (accumulated)
	pub time_us: usize,

	// --- private data members --
	start_time: Instant,
}

/// timer class for recording time in microseconds
impl Timer{
	/// constructor
	pub fn new() -> Timer{
		return Timer{time_us: 0, start_time: Instant::now()};
	}

	/// start recording
	pub fn start(&mut self){
		self.start_time = Instant::now();
	}

	pub fn clear_start(&mut self){
		self.time_us = 0;
		self.start_time = Instant::now();
	}

	pub fn stop(&mut self){
		self.time_us += self.start_time.elapsed().as_micros() as usize;
	}

	pub fn get_time(&self)->usize{
		let time_us = self.start_time.elapsed().as_micros() as usize;
		time_us
	}

	pub fn clear(&mut self){
		self.time_us = 0;
	}
}

///get the avaialble RAM in bytes
pub fn get_sys_mem() -> usize{
	let mut sys = System::new_all();
	sys.refresh_all();

	let total = sys.available_memory();
	return total as usize;
}


/// get the current time in u64
pub fn get_time() -> u128{
  let start = SystemTime::now();
  let val = start.duration_since(UNIX_EPOCH).expect("Err!");
  return val.as_nanos();
}

pub fn rand_u64() ->u64{
	let mut rng = gen_rng();
	let y:u64 = rng.gen();
	return y;
}


/// generate a random generator 
pub fn gen_rng()->StdRng{
	let mut seed = [
        1, 0, 0, 0, 23, 0, 0, 0, 200, 1, 0, 0, 210, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ];
	let mut newts = get_time();
	for i in 0..5{
		seed[i] = (newts%128) as u8;
		newts = newts/128;
	}
    let rng = ark_std::rand::rngs::StdRng::from_seed(seed);
	return rng;
}

/** call get_time() and use it as seed */
pub fn gen_rng_from_seed(useed: u128)->StdRng{
	let mut seed = [
        1, 0, 0, 0, 23, 0, 0, 0, 200, 1, 0, 0, 210, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ];
	let mut newts = useed;
	for i in 0..5{
		seed[i] = (newts%128) as u8;
		newts = newts/128;
	}
    let rng = ark_std::rand::rngs::StdRng::from_seed(seed);
	return rng;
}

/* Return an Random Array of Field Elements */
pub fn rand_arr_field_ele<F: FftField>(size: usize, seed: u128)->Vec<F>{
	let mut vec = vec![F::zero(); size];
	let mut rng = gen_rng_from_seed(seed);
	for i in 0..size{
		vec[i] = F::rand(&mut rng);
	}	
	return vec;
}


/** convert to array of field elements */
pub fn arru64_to_arrft<F:FftField>(arr: &Vec<u64>) -> Vec<F>{
	if arr.len()<1 {panic!("arru64_arr_arrft ERROR: input arr length() is 0");}
	let mut vec = vec![F::zero(); arr.len()];
	for i in 0..arr.len(){
		vec[i] = F::from(arr[i]);
	}
	return vec;
}


/** Return a random array of u64 */
pub fn rand_arr_unique_u64(size: usize, max_bits: usize, seed: u128) -> Vec<u64>{
	let mut vec:Vec<u64> = vec![];
	let mut rng = gen_rng_from_seed(seed);
	let mut added = HashSet::new();
	let mut i = 0;
	let modulus :u64 = 1<<max_bits;
	while i<size{
		let ele = rng.gen::<u64>() % modulus;
	
		if !added.contains(&ele){
			vec.push(ele);
			added.insert(ele);
			i += 1;
		}
	}	
	return vec;
}


/** return true if two are equal */
pub fn vec_equals<F: FftField>(v1: &Vec<F>, v2: &Vec<F>)->bool{
	if v1.len()!=v2.len(){return false;}
	for i in 0..v1.len(){
		if v1[i]!=v2[i]{
			println!("at {}: v1[i] is {}, v2[i] is {}", i, v1[i], v2[i]);
			return false;
		}
	}
	return true;
}

/** dump string */
pub fn dump_vec<F: FftField>(prefix: &str, v: &Vec<F>){
	print!("{}: ", prefix);
	for i in 0..v.len(){
		print!("{} ", v[i]);
	}	
	println!("");
}

/** dump as string */
pub fn dump_slice<F: FftField>(prefix: &str, v: &[F]){
	print!("{}: ", prefix);
	for i in 0..v.len(){
		print!("{} ", v[i]);
	}	
	println!("");
}


/** serlialize a vec of T into vec of u8 */
pub fn old_to_vecu8<F: CanonicalSerialize>(v: &Vec<F>)->Vec<u8>{
	if v.len()==0 {return vec![];}
	let unit_size = v[0].uncompressed_size();
	//let total_size = unit_size * v.len();
	let mut v2 = vec![0; 0];
	for u in 0..v.len(){
		let mut rec = vec![0; unit_size];
		let _x = v[u].serialize_uncompressed(&mut rec[..]);
		v2.append(&mut rec);
	}
	return v2;
}

/** serlialize a vec of T into vec of u8 */
pub fn to_vecu8<F: CanonicalSerialize>(v: &Vec<F>)->Vec<u8>{
	if v.len()==0 {return vec![];}
	let unit_size = v[0].uncompressed_size();
	let mut v2 = vec![0; unit_size*v.len()];
	for u in 0..v.len(){
		let slice = &mut v2[unit_size*u..unit_size*(u+1)];
		let _x = v[u].serialize_uncompressed(slice);
	}
	return v2;
}
/** serlialize a vec of T into vec of u8 */
pub fn slice_to_vecu8<F: CanonicalSerialize>(v: &[F])->Vec<u8>{
	if v.len()==0 {return vec![];}
	let unit_size = v[0].uncompressed_size();
	let mut v2 = vec![0; unit_size*v.len()];
	for u in 0..v.len(){
		let slice = &mut v2[unit_size*u..unit_size*(u+1)];
		let _x = v[u].serialize_uncompressed(slice);
	}
	return v2;
}

/** ASSUMPTION, dest should PREALLOCATED exactly the same size as expected */
pub fn to_vecu8_in_place<F: CanonicalSerialize>(v: &Vec<F>, dest: &mut Vec<u8>){
	if v.len()==0 {return;}
	let unit_size = v[0].uncompressed_size();
	if unit_size*v.len()!=dest.len() {panic!("to_vecu8_in_place ERR: dest len is not: {}", unit_size*v.len());}
	for u in 0..v.len(){
		let slice = &mut dest[unit_size*u..unit_size*(u+1)];
		let _x = v[u].serialize_uncompressed(slice);
	}
}

/** serlialize a vec of T into vec of u8, compressed */
pub fn to_vecu8_compressed<F: CanonicalSerialize>(v: &Vec<F>)->Vec<u8>{
	if v.len()==0 {return vec![];}
	let unit_size = v[0].uncompressed_size();
	//let total_size = unit_size * v.len();
	let mut v2 = vec![0; 0];
	for u in 0..v.len(){
		let mut rec = vec![0; unit_size];
		let _x = v[u].serialize(&mut rec[..]);
		v2.append(&mut rec);
	}
	return v2;
}

/// return the concat of all, will destroy input vecs
pub fn concat_vecu8(v: &mut Vec<Vec<u8>>) -> Vec<u8>{
	let mut vecres = vec![];
	for i in 0..v.len(){
		let vec = &mut v[i];
		vecres.append(vec);
	}
	return vecres;
}

/** deserialize from vec of u8 to vec of T, zero is a default element */
pub fn old_from_vecu8<F:CanonicalDeserialize + CanonicalSerialize + std::clone::Clone>(v: &Vec<u8>, zero: F)  -> Vec<F>{
	let unit_size = zero.uncompressed_size();
	let units = v.len()/unit_size;
	let mut v2 = vec![zero; units];
	for i in 0..units{
		let ele = &v[unit_size*i..unit_size*(i+1)];
		let f1 = F::deserialize_uncompressed(&ele[..]).unwrap();
		v2[i] = f1;
	}
	return v2;
}

/** deserialize from vec of u8 to vec of T, zero is a default element */
#[inline]
pub fn from_vecu8<F:CanonicalDeserialize + CanonicalSerialize + std::clone::Clone>(v: &Vec<u8>, zero: F)  -> Vec<F>{
	let unit_size = zero.uncompressed_size();
	let units = v.len()/unit_size;
	let mut v2 = vec![zero; units];
	for i in 0..units{
		let ele = &v[unit_size*i..unit_size*(i+1)];
		//v2[i] = F::deserialize_uncompressed(&ele[..]).unwrap();
		v2[i] = F::deserialize_unchecked(&ele[..]).unwrap();
	}
	return v2;
}

//return how many lines are there
pub fn get_num_lines(fname: &str)->usize{
	let file = fs::File::open(fname).unwrap();
	let reader = BufReader::new(file);
	let val =  reader.lines().count();
	return val;
}

/** read an array of numbers as field elements from start_line to end_line
(both included). ASSUMPTION: all values are u64 (not longer than 64-bits!) 
*/
pub fn read_arr_from<F:FftField>(fname: &String, start_line:usize, end_line:usize) -> Vec<F>{
	if end_line<start_line {return vec![];}
	let mut vec:Vec<F>  = vec![F::from(0u64); end_line-start_line+1];
	let file = match fs::File::open(fname){
		Ok(file) => file,
		Err(_) => panic!("NODE: {} Unable to open: {}", RUN_CONFIG.my_rank, fname)
	};
	let buf = BufReader::new(file);
	let mut lines = buf.lines();
	vec[0]= F::from(str_to_u64(&(lines.nth(start_line).unwrap().unwrap())));
	for i in 0..end_line-start_line{
		//Use index 0 because nth has already consumed all previous
		let line = lines.nth(0).unwrap().unwrap(); 
		let num = str_to_u64(&line);
		vec[i+1] = F::from(num);
	}
	return vec;
}

/// parse string as fe
pub fn str_to_fe<F:PrimeField>(v: &String) -> F {
	let s: &str = &v[..];
	let res = F::from_str(s);
	match res{
		Ok(res_f) => return res_f,
		Err(_e) => {println!("ERROR parsing v: {}", v); return F::zero();}
	}
}

/** read arr of lines from file. Assumption: file is small! */
pub fn read_arrlines(fname: &String) -> Vec<String>{
	let file = match fs::File::open(fname){
		Ok(file) => file,
		Err(_) => panic!("Unable to open: {}", fname)
	};
	let mut vec = vec![];
	let buf = BufReader::new(file);
	let lines = buf.lines();
	for line in lines{
		let sline = line.unwrap();
		vec.push(sline);
	}
	return vec;
}

/** now read the full field elements from file,
assuming the 1st line is the count. No restriction on u64
*/
pub fn read_arr_fe_from<F:PrimeField>(fname: &String)
 -> Vec<F>{
	let file = match fs::File::open(fname){
		Ok(file) => file,
		Err(_) => panic!("Unable to open: {}", fname)
	};
	let buf = BufReader::new(file);
	let mut lines = buf.lines();
	let line = lines.nth(0).unwrap().unwrap();
	let num = str_to_u64(&line) as usize;
	let mut vec = vec![F::from(0u64); num]; 
	for i in 0..num{
		//Use index 0 because nth has already consumed all previous
		let line = lines.nth(0).unwrap().unwrap(); 
		let fe = str_to_fe::<F>(&line);
		vec[i] = fe;
	}
	return vec;
}
/** now read the full field elements from file,
assuming the 1st line is the count. No restriction on u64
	start_line (excluding the 1st line) - count from 0
	end_line: not included
*/
pub fn read_slice_fe_from<F:PrimeField>(fname: &String, start_idx: usize, end_idx: usize)
 -> Vec<F>{

	let file = match fs::File::open(fname){
		Ok(file) => file,
		Err(_) => panic!("Unable to open: {}", fname)
	};
	let buf = BufReader::new(file);
	let mut lines = buf.lines();
	let num = end_idx - start_idx;
	let mut vec = vec![F::zero(); num];
	let _temp = lines.nth(start_idx);
	for i in 0..num{
		//Use index 0 because nth has already consumed all previous
		let line = lines.nth(0).unwrap().unwrap(); 
		vec[i] = str_to_fe::<F>(&line);
	}
	return vec;
}

// write number in parsable format
pub fn write_arr_fe_to<F:PrimeField>(vec: &Vec<F>, fname: &String){
	let file = match fs::File::create(fname){
		Ok(file) => file,
		Err(_) => panic!("Unable to open: {}", fname)
	};
	let mut buf = BufWriter::new(file);
	let num = vec.len();
	write!(&mut buf, "{}\n", &num).unwrap();
	let zero = F::zero();
	for x in vec{
		if *x==zero{
			write!(&mut buf, "0\n").unwrap();
		}else{
			write!(&mut buf, "{}\n", &x).unwrap();
		}
	}
	buf.flush().unwrap();
}

// write number in parsable format (0 is explicityly outputted)
pub fn write_arr_fe_to_with_zero<F:PrimeField>(vec: &Vec<F>, fname: &String){
	let file = match fs::File::create(fname){
		Ok(file) => file,
		Err(_) => panic!("Unable to open: {}", fname)
	};
	let mut buf = BufWriter::new(file);
	let num = vec.len();
	write!(&mut buf, "{}\n", &num).unwrap();
	for x in vec{
		if !x.is_zero(){
			write!(&mut buf, "{}\n", &x).unwrap();
		}else{
			write!(&mut buf, "0\n").unwrap();
		}
	}
	buf.flush().unwrap();
}
/** read arr of u64 from */
pub fn read_arr_u64_from(fname: &String, start_line:usize, end_line:usize) -> Vec<u64>{
	let mut vec:Vec<u64>  = vec![0u64; end_line-start_line+1];
	let file = match fs::File::open(fname){
		Ok(file) => file,
		Err(_) => panic!("Unable to open: {}", fname)
	};
	let buf = BufReader::new(file);
	let mut lines = buf.lines();
	vec[0]= str_to_u64(&(lines.nth(start_line).unwrap().unwrap()));
	for i in 0..end_line-start_line{
		//Use index 0 because nth has already consumed all previous
		let line = lines.nth(0).unwrap().unwrap(); 
		let num = str_to_u64(&line);
		vec[i+1] = num;
	}
	return vec;
}

/** return a set of it */
pub fn get_set_u64(v: &Vec<u64>)->Vec<u64>{
	let mut res:Vec<u64> = vec![];
	let mut set:HashSet<u64> = HashSet::new();
	for ele in v{
		if !set.contains(ele){
			set.insert(*ele);
			res.push(*ele);
		}
	}
	return res;
}

/** String to u64 */
pub fn str_to_u64(s: &String)->u64{
	let res = s.trim().parse::<u64>().unwrap();
	return res;
}

/** read the 1st line of the file as u64 */
pub fn read_1st_line_as_u64(fname: &String) -> u64{
	let file = match fs::File::open(fname){
		Ok(file) => file,
		Err(_) => panic!("Unable to open: {}", fname)
	};
	let mut buf = BufReader::new(file);
	let mut line = String::new();
	buf.read_line(&mut line).expect("Unable to read 1st line");
	return str_to_u64(&line);
}


/** write the array of numbers to file */
pub fn write_arr(v: &Vec<u64>, fname: &String){
	let file = match fs::File::create(fname){
		Ok(file) => file,
		Err(_) => panic!("Unable to open: {}", fname)
	};
	let mut buf = BufWriter::new(file);
	for num in v{
		write!(&mut buf, "{}\n", &num).unwrap();
	}
	buf.flush().unwrap();
}

/** the first line is the total number of elements */
pub fn write_arr_with_size(v: &Vec<u64>, fname: &String){
	let total:u64 = v.len() as u64;
	let mut a1 = vec![total; v.len()+1];
	for i in 0..v.len(){
		a1[i+1] = v[i];
	}
	write_arr(&a1, fname);
	
}

/** write the string to file path*/
pub fn write_file(fpath: &str, str: String){
	fs::write(fpath, str).unwrap();
}

/** write the array of field elements to file */
pub fn write_arr_fr<F:FftField+CanonicalSerialize>(v: &Vec<F>, fname: &String){
	let _file = match fs::File::create(fname){
		Ok(file) => file,
		Err(_) => panic!("Unable to open: {}", fname)
	};
	let v8 = to_vecu8(v);
	write_vecu8(&v8, fname);
}


/** write the array of numbers to file */
pub fn write_vecu8(v: &Vec<u8>, fname: &String){
	let mut file = match fs::File::create(fname){
		Ok(file) => file,
		Err(_) => panic!(
			"NODE: {}. Unable to open: {}",RUN_CONFIG.my_rank,fname)
	};
	file.write_all(&v).unwrap();
}

/** read the array of numbers from file */
pub fn read_vecu8(fname: &String)->Vec<u8>{
	let mut file = match fs::File::open(fname){
		Ok(file) => file,
		Err(_) => panic!("NODE: {}, Unable to open: {}", RUN_CONFIG.my_rank, fname)
	};
	let metadata = fs::metadata(fname);
	let size = metadata.expect("can't open file").len() as usize;
	let mut buf:Vec<u8> = vec![0; size];
	file.read(&mut buf).unwrap();
	return buf;
}

/** read and return ONE group element */
pub fn read_ge<G: AffineCurve>(fpath: &str) -> G{
	let g= G::prime_subgroup_generator();
	let v8 =  read_vecu8(&fpath.to_string());
	let ele = from_vecu8::<G>(&v8, g);
	return ele[0];
}

/// use default 128-bit security generate 128-bit field element
pub fn hash<F:PrimeField>(barr: &Vec<u8>) -> F{
	return hash_worker(barr, 128);
}

/// hash the given byte array into a field element
pub fn hash_worker<F:PrimeField>(barr: &Vec<u8>, bits: usize)->F{
	let mut t1 = Timer::new();
	t1.start();
	let mut hasher = Sha256::new();
	hasher.update(barr);
	let result = hasher.finalize();
	let bi = BigUint::from_bytes_le(&result);
	let bi250 = BigUint::from(1u64) << bits; 
	let bres = bi % bi250;
	let sres = bres.to_str_radix(10);
	let f = str_to_fe::<F>(&sres);
	t1.stop();
	return f;
}

/// tell if two vectors of G points are equal
pub fn eq_arr_g<G:AffineCurve>(v1: &Vec<G>, v2: &Vec<G>)->bool{
	if v1.len()!=v2.len() {return false;}
	let n = v1.len();
	for i in 0..n{
		if v1[i]!=v2[i] {return false;}
	}
	return true;
}

pub fn file_size(fpath: &str)->usize{
	let fsize = std::fs::metadata(fpath).unwrap().len();
	return fsize as usize;
}

/// tuple: (idx, file_size)
/// sort by file size
/// simple bubble sort - don't apply for big array
pub fn sort_by_file_size(vec: &mut Vec<(usize, usize)>){
	if vec.len() > 1024 {panic!("DON'T apply bubble sort to big arrays!");}
	let n = vec.len();
	for _i in 0..n-1{
		for j in 0..n-1{
			if vec[j].1>vec[j+1].1{
				let tmp= vec[j];	
				vec[j] = vec[j+1];
				vec[j+1] = tmp;	
			}	
		}
	}
}

/// create the directory if not exists
pub fn new_dir_if_not_exists(sdir: &str){
	fs::create_dir_all(sdir).unwrap();
}

/// remove the directory
pub fn remove_dir(sdir: &str){
	//let sdir = &get_absolute_path(sdir);
	if exists(sdir) { fs::remove_dir_all(sdir).unwrap();}
}

/// rename the folder name
pub fn rename_dir(sbase_path:&str, old_name: &str, new_name: &str){
	let oldpath = &format!("{}/{}", sbase_path, old_name);
	let newpath = &format!("{}/{}", sbase_path, new_name);
	assert!(exists(oldpath), "ERROR: rename_dir @{}: path not exist: {}", 
		RUN_CONFIG.my_rank, oldpath);
	if exists(newpath){
		remove_dir(newpath);
	}
	//assert!(!exists(newpath), "{} -> {} but new_path already exist", oldpath, newpath);
	fs::rename(oldpath, newpath).unwrap();
}

/// copy over the folder name
pub fn copy_dir(sbase_path:&str, old_name: &str, new_name: &str){
	let oldpath = format!("{}/{}", sbase_path, old_name);
	let newpath = format!("{}/{}", sbase_path, new_name);
	assert!(exists(&oldpath), "ERROR: copy_dir @{}: path not exist: {}", 
		RUN_CONFIG.my_rank, &oldpath);
	if exists(&newpath){
		remove_dir(&newpath);
	}
	//assert!(!exists(newpath), "{} -> {} but new_path already exist", oldpath, newpath);
	run("cp", &vec![tos("-R"), oldpath, newpath]);
}
pub fn move_file(oldpath: &str, newpath: &str){
	assert!(exists(oldpath), "path not exist: {}", oldpath);
	fs::rename(oldpath, newpath).unwrap();
}

pub fn copy_file(oldpath: &str, newpath: &str){
	assert!(exists(oldpath), "path not exist: {}", oldpath);
	fs::copy(oldpath, newpath).unwrap();
}

/// remove file if it exists
pub fn remove_file(fpath: &str){
	let fpath = get_absolute_path(fpath);
	if exists(&fpath) {fs::remove_file(fpath).unwrap();}
}

/// return the absolute path of it
/// assuming that spath EXISTS!
pub fn get_absolute_path(spath: &str)->String{
	let path = PathBuf::from(spath);
	let abspath = fs::canonicalize(&path).unwrap();
	let sabs = abspath.as_path().display().to_string();
	return String::from(sabs);	
}


/// return True if file exists
pub fn exists(sdir: &str)->bool{
	return	Path::new(sdir).exists();
}

/// remove it and then create as new
pub fn new_dir(sdir: &str){
	if exists(sdir){ remove_dir(sdir);}
	new_dir_if_not_exists(sdir);
}

/// assume base dir exists. create the file and return
/// the file handler in append mode
pub fn new_file_append(spath: &str)->fs::File{
	if exists(spath) {remove_file(spath);}
	fs::File::create(spath).unwrap();
	let f = fs::File::options().append(true).open(spath).unwrap();
	return f;
}




/// log function worker
pub fn log_worker(log_level: usize, bmatch_node: bool, node_id: usize, msg: &String){
	if log_level>=RUN_CONFIG.log_level && (!bmatch_node || node_id==RUN_CONFIG.my_rank){
		println!("LOG: {}", msg);
	}
}

/// log function worker, append to a file
pub fn flog_worker(log_level: usize, bmatch_node: bool, node_id: usize, msg: &String, file_ref: &mut fs::File){

	if log_level>=RUN_CONFIG.log_level && (!bmatch_node || node_id==RUN_CONFIG.my_rank){
		let mut sprefix = format!("");
		for _i in 1..log_level{ sprefix += "  "; }
		let s1 = format!("{}{}", &sprefix, msg);
		println!("{}", &s1);
		let s2 = format!("{}{}\n", &sprefix, msg);
		file_ref.write_all(&s2.as_bytes()).unwrap();
	}
}

/// assuming it's DUMPING main node only
pub fn log(log_level: usize, msg: &String){
	log_worker(log_level, true, 0, msg);
}

/// assuming it's DUMPING main node only
pub fn flog(log_level: usize, msg: &String, file: &mut fs::File){
	flog_worker(log_level, true, 0, msg, file);
}

/// log theperformance print "log_title time_us"
pub fn log_perf(log_level: usize, log_title: &str, timer: &mut Timer){
	timer.stop();
	if timer.time_us<1000{
		log(log_level, &format!("{} {} us", log_title, timer.time_us));
	}else{
		log(log_level, &format!("{} {} ms", log_title, timer.time_us/1000));
	}
	timer.clear_start();
}

/// log theperformance print "log_title time_us"
pub fn flog_perf(log_level: usize, log_title: &str, timer: &mut Timer, 
		fd: &mut fs::File){
	timer.stop();
	if timer.time_us<1000{
		flog(log_level, &format!("{} {} us", log_title, timer.time_us), fd);
	}else{
		flog(log_level, &format!("{} {} ms", log_title, timer.time_us/1000),fd);
	}
	timer.clear_start();
}


/// return a vector of String
/// Assumption: string is not big!
pub fn split_str(s: &str)->Vec<String>{
	let v1 = s.split(" ");
	let mut vec:Vec<String> = vec![];
	for x in v1{
		let sitem = format!("{}", x).trim().to_string();
		if sitem.len() > 0 {vec.push(sitem);}
	}
	return vec;
}

/// assumption: it's a regular 2d array (all row size the same)
pub fn vec2d_to_vec<F:CanonicalSerialize+CanonicalDeserialize+Clone>(vec2d: &Vec<Vec<F>>) -> Vec<F>{
	let cols = vec2d[0].len();
	let rows = vec2d.len();
	let total = rows * cols;
	let sample= vec2d[0][0].clone();
	let mut res = vec![sample; total];
	for i in 0..rows{
		assert!(vec2d[i].len()==cols, "ERROR: vec2d row {} size: {} != cols: {}", i, vec2d[i].len(), cols);
		for j in 0..cols{ res[i*cols + j] = vec2d[i][j].clone();}
	}
	return res;
}

//convert 1d array to 2d, assuming its size is rows* cols
pub fn vec_to_vec2d<F:CanonicalSerialize+CanonicalDeserialize+Clone>(vec: &Vec<F>, rows: usize) -> Vec<Vec<F>>{
	assert!(vec.len()%rows==0, "vec.len(): {} % rows: {} !=0", vec.len(), rows);
	let cols = vec.len()/rows;
	let sample = vec[0].clone();
	let mut res = vec![vec![sample; cols]; rows];
	for i in 0..rows{
		for j in 0..cols{
			res[i][j] = vec[i*cols + j].clone();
		}
	}	
	return res;
}

/// syntax sugar
pub fn tos(s: &str) -> String{
	return String::from(s);
}

/// run a command with string
pub fn run(scmd: &str, args: &Vec<String>) -> String{
	let b_debug = false;
	let me = RUN_CONFIG.my_rank;
	let limit = 3;
	for attempt in 0..limit{
		if b_debug{
			print!("DEBUG USE 101:-- RUN: {} ", scmd);
			for arg in args{
				print!(" {}", arg);
			}
			println!("");
		}
		let output = Command::new(scmd).args(args).output().expect("failed");
		let s_fullcmd = join_str(scmd, args);
		let status = output.status;
		let vu8 = output.stdout.as_slice();
		let s = from_utf8(&vu8).unwrap();
		let vu8_err = output.stderr.as_slice();
		let s_err = from_utf8(&vu8_err).unwrap();
		if !status.success(){
			println!("WARN 101: FAILED RUN attempt: {}!  Node: {}, CMD: {} failed: {}. stderr: {}", attempt, me, s_fullcmd, s, s_err);
			if attempt==limit{
				assert!(status.success(), "FAILED THE LAST CHANCE!");
			}else{
				let three_sec = time::Duration::from_millis(5*1000);
				thread::sleep(three_sec); //sleep for 3 seconds
			}
		}else{
			return String::from(s);
		}
	}
	
	return String::from("ERROR");
}
/// run a command with string, launch the command in a specified dir
pub fn run_in_dir(cmd: &str, args: &Vec<String>, dir: &String) -> String{
	//let _scmd = format!("{} {}", cmd, args.join(" "));
	//println!("DEBUG USE 301: {}", _scmd);
	let output = Command::new(cmd).current_dir(dir).args(args)
		.output().expect("failed");
	let vu8 = output.stdout.as_slice();
	let vu8_2 = output.stderr.as_slice();
	let s = format!("OUTPUT: {}\n", from_utf8(&vu8).unwrap());
	let s2 = format!("ERR: {}\n", from_utf8(&vu8_2).unwrap());
	return s + &s2;
}

/// try to get ALL my list of IPs, if there is one
/// excluding 127.0.0.1, return it (first);
/// otherwise return 127.0.0.1
pub fn get_my_ip(vecip: &Vec<u64>)->u64{
	let sip_self = "127.0.0.1";
	let ip_self = ip_from_str(&String::from(sip_self));
	let sout = run("hostname", &vec![String::from("-I")]);
	let arr = sout.split(" ").collect::<Vec<&str>>();
	for x in arr{
		let sword = String::from(x.trim());
		if sword.len()<4 {continue;}
		let ip = ip_from_str(&sword);
		if vecip.contains(&ip) && ip!=ip_self {
			return ip;
		}
	}
	return ip_self;
	
}



/// check if this node is the first node of some server
pub fn is_1st_node_of_server_by_arch(netarch: &(Vec<u64>, Vec<usize>, Vec<Vec<usize>>)) -> bool{
	let arr1st = &netarch.1;
	let me = RUN_CONFIG.my_rank;
	let bres = arr1st.contains(&me);
	return bres;
}

/// get the ID of the server this node belongs to
/// NODE: only return valid answer when this node is the 1st node
pub fn get_server_id(netarch: &(Vec<u64>, Vec<usize>, Vec<Vec<usize>>)) -> usize{
	let arr1st = &netarch.1;
	let me = RUN_CONFIG.my_rank;
	for idx in 0..arr1st.len(){
		if arr1st[idx]==me{ return idx; }
	}
	panic!("shoudl ONLY call get_server_id at nodes that are the 1st node of a server! Node id: {}", me);
}

/// return the (server_id, relative_id_in_server)
pub fn get_identity_in_server(netarch: &(Vec<u64>,Vec<usize>,Vec<Vec<usize>>))
	-> (usize, usize){
	let node_list = &netarch.2;
	let me = RUN_CONFIG.my_rank;
	for server_id in 0..node_list.len(){
		let vecids = &node_list[server_id];
		for relative_id in 0..vecids.len(){
			let node_id = vecids[relative_id];
			if node_id==me{
				return (server_id, relative_id);
			}
		}
	}
	panic!("CANNOT find server identity for: {}", me);
}

/// get the parent directory of the given path
/// by chopping off the last component after the
/// last "/"
pub fn get_parent_dir(spath: &str) -> String{
	let idx = String::from(spath).rfind("/").unwrap();
	let par_dir = spath[0..idx].to_string();
	return par_dir;
}

/// convert ip address from a string like "1.32.24.25"
pub fn ip_from_str(sip: &String) -> u64{
	let arr = sip.split(".").collect::<Vec<&str>>();
	assert!(arr.len()==4, "ip addr invalid: {}", sip); 
	let mut val = 0u64;
	let base = 256;
	for s in arr{
		let v:u64= s.parse().unwrap();
		val = val*base + v;
	}
	return val;
}


/// convert ip address from u32 to string
pub fn ip_to_str(ip: u64) -> String{
	let mut s = String::from("");
	let base = 256;
	let mut val = ip;
	for i in 0..4{
		let v = val % base;
		let s_i = if i==0 {format!("{}", v)} else {format!("{}.", v)};
		s = s_i + &s;	
		val = val / base;
	} 
	return s;
}


/// read the mpi_hosts_file and parse the list of IP addresses
pub fn get_ip_list(mpi_hosts_file: &String) -> Vec<u64>{
	let lines = read_lines(mpi_hosts_file);
	let mut vec = vec![];
	let mut i_vec = vec![];
	for line in lines{
		let arr = line.split(" ").collect::<Vec<&str>>();
		let ip = String::from(String::from(arr[0]).trim());
		let v_ip = ip_from_str(&ip);
		vec.push(ip);
		i_vec.push(v_ip);
	}
	if vec[0]!="127.0.0.1" {panic!("WE assume in hosts the 1st IP is 127.0.0.1");}
	return i_vec;
}


/** read lines from a file */
pub fn read_lines(fname: &String) -> Vec<String>{
	let file = match fs::File::open(fname){
		Ok(file) => file,
		Err(_) => panic!("Unable to open: {}", fname)
	};
	let reader = BufReader::new(file);
	let mut vec = vec![];
	for line in reader.lines(){
		vec.push(line.unwrap());
	}
	return vec;
}

/// extract the ip_list from the full_ip list
fn extract_iplist(vec: &Vec<u64>) -> Vec<u64>{
	let num: usize = vec[0] as usize;
	let mut v = vec![0u64; num];
	for i in 0..num{
		v[i] = vec[i+1];
	}
	return v;
}

/// expand the array to given number
fn expand_iplist(vip: &Vec<u64>, desired_len: usize) -> Vec<u64>{
	let mut vec = vec![0u64; desired_len];
	let num = vip.len();
	vec[0] = num as u64;
	for i in 0..num{
		vec[i+1] = vip[i];
	}
	return vec;
}



/// list the folder names in a dir
pub fn list_dir(fpath: &str)->Vec<String>{
	let paths = fs::read_dir(fpath).unwrap();
	let mut vres = vec![];
	for path in paths{
		let spath = format!("{}", path.unwrap().path().display());
		if spath.starts_with("/tmp2/batchprove/101") {continue};
		vres.push(spath);
	}
	return vres;
}

/// do not ignore /tmp2/batchprove/101
pub fn list_dir_full(fpath: &str)->Vec<String>{
	let paths = fs::read_dir(fpath).unwrap();
	let mut vres = vec![];
	for path in paths{
		let spath = format!("{}", path.unwrap().path().display());
		vres.push(spath);
	}
	return vres;
}

/// get all recursively embedded
pub fn list_dir_recursively(fpath: &str, vres: &mut Vec<String>){
	let paths = fs::read_dir(fpath).unwrap();
	for path in paths{
		let spath = format!("{}", path.unwrap().path().display());
		let spath = get_absolute_path(&spath);
		if vres.contains(&spath) {continue;}

		let p_path = Path::new(&spath);
		if p_path.is_dir(){
			list_dir_recursively(&spath, vres);
		}else{
			vres.push(spath);
		}
	}
}

/// check if it is a directory
pub fn is_dir(fpath: &str) -> bool{
	let p_path = Path::new(fpath);
	return p_path.is_dir();
}

/// get the last section
/// ASSUMING spath has at least 1 "/"
pub fn get_fname(spath: &str) -> String{
	let idx = String::from(spath).rfind("/").unwrap();
	let fname = spath[idx+1..].to_string();
	return fname;
}

/// pack the given dir and generate the tar file for it with
/// the same name in the target_dir 
pub fn tar_dir(dirpath: &str, target_dir: &str, target_name: &str){
	let parent_path = get_absolute_path(&get_parent_dir(dirpath));
	let tar_path = format!("{}/{}.tar", target_dir, target_name);
	let dirname = get_fname(dirpath);
	
	run("tar", &vec![tos("-cvf"), tar_path.clone(), 
			tos("-C"), parent_path.clone(), dirname]);   	
}

/// extract the tar file at the specified dir
/// the 1ST layer of dir name is renamed to target_dir
pub fn extract_tar(tarfile: &str, parent_dir: &str, target_dir_name: &str){
	let b_debug = false;

	if b_debug{println!("extract_tar: me: {}, tarfile: {}, parent_dir: {}, target: {}", RUN_CONFIG.my_rank, tarfile, parent_dir, target_dir_name);}

	assert!(tarfile.starts_with("/"), 
		"extract_tar: the tarfile has to be absolute path!");
	let target_path = format!("{}/{}", parent_dir, target_dir_name);
	//let src_path= format!("{}/{}", parent_dir, fname_no_tar);
	if exists(&target_path){ remove_dir(&target_path); }
	let res = run("tar", &vec![tos("-xvf"), tos(tarfile), 
				tos("-C"), parent_dir.to_string()]);
	let src_dir= res.lines().nth(0).unwrap();
	let result_dir = format!("{}/{}", parent_dir, target_dir_name);
	if !exists(&result_dir){
		rename_dir(parent_dir, &src_dir, target_dir_name);
	}
	assert!(exists(&result_dir), "extract_tar destination: {} does not exist!",
		result_dir);
}


fn join_str(prefix: &str, v: &Vec<String>)->String{
	let mut s = format!("{}", prefix);
	for u in v{
		s = format!("{} {}", s, u);
	}
	return s;
}


pub fn vec_affine_to_proj<E:PairingEngine>(v: &Vec<E::G1Affine>) -> 
	Vec<E::G1Projective>{
	let mut vres = vec![];
	for x in v{
		vres.push(x.into_projective());
	}
	return vres;
}

pub fn v2d_affine_to_proj<E:PairingEngine>(v: &Vec<Vec<E::G1Affine>>) -> 
	Vec<Vec<E::G1Projective>>{
	let mut vres = vec![];
	for vec in v{
		vres.push(vec_affine_to_proj::<E>(&vec));
	}
	return vres;
}

pub fn vec_affine2_to_proj<E:PairingEngine>(v: &Vec<E::G2Affine>) -> 
	Vec<E::G2Projective>{
	let mut vres = vec![];
	for x in v{
		vres.push(x.into_projective());
	}
	return vres;
}

pub fn v2d_affine2_to_proj<E:PairingEngine>(v: &Vec<Vec<E::G2Affine>>) -> 
	Vec<Vec<E::G2Projective>>{
	let mut vres = vec![];
	for vec in v{
		vres.push(vec_affine2_to_proj::<E>(&vec));
	}
	return vres;
}
