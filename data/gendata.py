# -------------------------------
# emulate the data generation
# -------------------------------

from random import *;
from numpy.random import seed
from numpy.random import normal
from numpy.random import poisson 
import re;

OUTPUT_DIR = "raw_data/";

# write 2 dimensional array into given file
def write_2darr_to_file(arr2d, fname):
	f = open(OUTPUT_DIR + fname, "w");
	for row in arr2d:
		line = "";
		for x in row:
			s = str(x) + "\t";
			line += s;
		line += "\n";
		f.write(line);	
	f.close();	

def write_1darr_to_file(arr1d, fname):
	f = open(OUTPUT_DIR + fname, "w");
	for x in arr1d:
		line = str(x) + "\n";
		f.write(line);	
	f.close();	


# extract a collection of records from a given file
# generate records <op, size, time, np>
# time is in ms
def extract_records(keyname, fname):
	f1 = open(fname, "r");
	arrlines = f1.readlines();
	f1.close();

	arrRec = [];
	for line in arrlines:
		if line.find("REPORT_")==0: #process it
			arr = line.split(",");
			rec = {};
			for item in arr:
				a1 = item.split(":");
				skey = a1[0].strip();
				if skey.find("REPORT_")>=0:
					skey = skey[7:];
				val = a1[1].strip().split()[0];
				if val.isnumeric():
					rec[skey] = int(val);
				else:
					rec[skey] = val;
			if rec["op"]==keyname:
				arrRec.append(rec);
	return arrRec;

# return all values in sorted (ascending)
def get_sorted(arrRecs, opname, skey):
	arr = [];
	for rec in arrRecs:
		if opname==rec["op"]:
			val = rec[skey];
			if not val in arr:
				arr.append(val);
	arr.sort();
	return arr;

# given the opname, size and np retrieve the time
# slow (but ok for smaller data set)
def get_time(arrRec, opname, size, np):
	for rec in arrRec:
		if rec["op"]==opname and rec["size"]==size and rec["np"]==np:
			return rec["time"];
	print("CANNOT find record for: " + opname + ", size: " + str(size) + ", np: " + str(np));

# extract all records of the op
# write a 2-dimensioal table into fname
def write_recs(arrRecs, opname, fname):
	arr_np = get_sorted(arrRecs, opname, "np");
	arr_size = get_sorted(arrRecs, opname, "size");

	title_row = ["SIZE/np"] + arr_np;
	data2d = [title_row];	
	for size in arr_size:
		row = [size];
		for np in arr_np:
			itime = get_time(arrRecs, opname, size, np);
			row.append(itime);
		data2d.append(row);

	write_2darr_to_file(data2d, fname);	
		

def main():
	arr_cq= extract_records("cq", "run_dumps/all_gcp.dat");
	write_recs(arr_cq, "cq", "cq.dat");

	arr_range= extract_records("range", "run_dumps/all_gcp.dat");
	write_recs(arr_range, "range", "range.dat");

	arr_pn= extract_records("pn_lookup", "run_dumps/all_gcp.dat");
	write_recs(arr_pn, "pn_lookup", "pn_lookup.dat");
	
	arr_asset1= extract_records("asset1", "run_dumps/all_gcp.dat");
	write_recs(arr_asset1, "asset1", "asset1.dat");

main();
	



