# ----
# try to debug subset2 problem (often div not by 0)
#------

MODULUS = 52435875175126190479447740508185965837690552500527637822603658699938581184513; #for BLS12-381 

def read_set(fpath):
	f1 = open(fpath, "r");
	arrlines = f1.readlines();
	f1.close();
	myset = set();
	for x in arrlines[1:]:
		num = int(x);
		myset.add(num);
	return myset;

# return set1 - set2
def diff_set(set1, set2):
	diffset = set();
	for x in set1:
		if not x in set2:
			diffset.add(x);
	return diffset;


# field elements multiplication
def mul(f1, f2):
	return f1*f2%MODULUS;

# compute union set
def union(set1, set2):
	uset = set();
	for x in set1:
		uset.add(x);
	for x in set2:
		uset.add(x);
	return uset;

# let p be the vanishing poly for the given set
# evaluate p(r)
def eval_set_r(set1, r):
	global MODULUS;
	res = 1;
	for x in set1:
		res = res * (x + MODULUS + r) % MODULUS
	return res;
			

def debug():
	r = 16734420230514147543759851145390091614309083903938751647154439920460571301761;
	vec = [729234, 234234234, 98234234234, 3723423423, 23232323, 4343434322222, 123123123123, 2339999]; 
	res1 = eval_set_r(vec, r);
	print("RESULT is: ", res1); 

# MAIN
r = 34908738280573886629627856637074963004;
subset_id = 10;

set_trans = read_set("/tmp2/batchprove/101/generated_set_trans.dat.dat");
set_states = read_set("/tmp2/batchprove/101/generated_set_states.dat.dat");
st = union(set_trans, set_states);
subset = read_set("/home/xiang/Desktop/ProofCarryExec/Code/zkregex/DATA/anti_virus_output/clamav_100/st_subset_" + str(subset_id) +".dat");
witness = diff_set(subset, st);
print("set_states: ", len(set_states));
print("set_trans: ", len(set_trans));
print("st: ", len(st));
print("subset: ", len(subset));
print("witness: ", len(witness));

diffset = diff_set(st, subset);
print("diffset st-subset:", len(diffset));

v_states = eval_set_r(set_states, r);
v_trans= eval_set_r(set_trans, r);
v_st = eval_set_r(st, r);
v_sub = eval_set_r(subset, r);
v_wit = eval_set_r(witness, r);

print("v_st * v_wit: ", mul(v_st, v_wit));
print("v_sub:", v_sub);
bpass = mul(v_st,v_wit)==v_sub;
print("VALUES: states: ", v_states, "v_trans: ", v_trans, "v_subset", v_sub);
print("CHECK completed. PASS: ", bpass);

