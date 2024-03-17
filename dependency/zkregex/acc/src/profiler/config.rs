/** 
	Copyright Dr. Xiang Fu

	Author: Dr. Xiang Fu
	All Rights Reserved.
	Created: 05/12/2022
	This represent the data structure of the running environment
*/
extern crate ark_ff;
extern crate ark_ec;
extern crate ark_poly;
extern crate ark_std;
//extern crate mpi;


//use self::mpi::traits::*;
//use mpi::point_to_point as p2p;
//use mpi::topology::Rank;
//use self::mpi::environment::Universe;
//use self::mpi::topology::Communicator;
use std::fmt;
extern crate once_cell;
use self::once_cell::sync::Lazy;

pub static RUN_CONFIG: Lazy<RunConfig> = Lazy::new(||
	RunConfig::create()
);

pub struct RunConfig{
	/** number of processors */
	pub n_proc: usize, 	
	/** number of partitions of data */
	pub n_part: usize,
	/** universe */
	//pub univ: Universe,
	/** my rank of the current node */
	pub my_rank: usize,
	/** max number of vec size on dis_vec and other relavant structures */
	pub max_vec_size: usize,
	/** log level, numeric values 0 to 5. The higher, the more URGENT */
	pub log_level: usize,
	/** min bar for distributed gcd */
	pub minbar_dis_feea: usize,
	/** the uname for rsync */
	pub rsync_uname: String,
	/** lower than this bar, do serial inv. 20 stands for 2^20 - 1 million */
	pub log_inv_bar: usize, 
	/** e.g., if < 1<<log_div_bar, do serial division instead */
	pub log_div_bar: usize, 
	/** speed up points. element 0 is where dis_mul has the same
		speed as serial, elment 1 is dis_mul is twice the speed,
		element 2 is dis_mul has 4 times of the speed of serial etc.
		the last element represents the point that it has the 
		max speed points.
		All represents as log2 value.
		Run get log_speedup_points in acc to retrieve the data.
		It's depending on the system (number of nodes)
		NOT USED ANY MOR.
	*/
	pub log_speed_up_points: Vec<usize>,
	/* switch point for switching from alg2 to _old agorithm for nonblock_broadcast */
	pub log_broadcast_alg_switch: usize, //0 for old, 50 for worker2 for nonblock_broadcast
	/* the IDs of subset of kzg in publisher's set */
	pub subset_ids: Vec<usize>
}

pub const LOG2:usize = 0;
pub const LOG1:usize = 1;
pub const WARN:usize = 2;
pub const ERR:usize = 3;

impl RunConfig{
	/** get the current or create a new run config.
	NEVER CALL this function. It has been called
	in setting up RUN_CONFIG  */
	fn create() -> RunConfig{
		//let univ = mpi::initialize().unwrap();
		//let world = univ.world();
		let n = 1;
		let rank = 1;
		return RunConfig{
			n_proc: n,
			n_part: n,
			my_rank: rank, 
		//	univ: univ,
			max_vec_size: 32 * 1024 * 1024, //32m fe = 1GB  
			log_level: LOG1,
			rsync_uname: String::from("xiang"),
			//minbar_dis_feea: 1024*32, //32k,
			minbar_dis_feea: 1024*256, //256k = 2^18
			//log_inv_bar: 1, //debug
			//log_inv_bar: 16, //32-cpu setting
			log_inv_bar: 15, //256-cpu setting
			log_div_bar: 15, //256-cpu setting
			//log_inv_bar: 10, //LOCAL setting
			//log_div_bar: 10, //LOCAL setting
			log_speed_up_points: vec![1, 2, 3, 4, 5, 6, 7],
			//log_speed_up_points: vec![10,10,10,10,10,10,10]
			//log_broadcast_alg_switch: 0, //use old_worker
			//log_broadcast_alg_switch: 50, //use woker1 
			log_broadcast_alg_switch: 27, //use woker1 
			subset_ids: vec![10, 15, 20, 30, 40, 50, 300],
		};
	}

	/// BETTER barrier: allows us to track if misfired.
	pub fn better_barrier(&self, _msg: &str){
		//println!("DEBUG USE 8888: Barrier: my_rank: {}, Msg: {}", self.my_rank, _msg);
	//	self.univ.world().barrier();
	}
} 


impl fmt::Display for RunConfig{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "Config: Processors: {}", self.n_proc)
    }
}
