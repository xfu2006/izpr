use ark_ff::fields::{Fp768, MontBackend, MontConfig};

#[derive(MontConfig)]
#[modulus = "5237311370989869175293026848905079641021338739994243633972937865128169101571388346632361720473792365177258871486054600656048925740061347509722287043067341250552640264308621296888446513816907173362124418513727200975392177480577"]
#[generator = "5"]
pub struct FrConfig;
pub type Fr = Fp768<MontBackend<FrConfig, 12>>;
