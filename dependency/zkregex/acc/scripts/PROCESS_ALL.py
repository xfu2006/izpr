# -------------------------------
# Dr. Xiang Fu
# Sunday 04/02/2022
# This file is the UI for generating all proofs
# -------------------------------

#job has two entrypes: group_id and subset_id
import os;
import time;
import subprocess;
from pathlib import Path;

# generate a unique order ID for job for sorting
def get_order(job):
	return job["group_id"]*10000 + job["subset_id"]*100 + job["part_id"];

def job_to_fname(job):
	if job["part_id"]==0:
		return "job_" + str(job["group_id"]) + "_" + str(job["subset_id"]) + ".txt";
	else:
		return "job_" + str(job["group_id"]) + "_" + str(job["subset_id"]) + "_" + str(job["part_id"]) + ".txt";


# read the directory and return all jobs
# each job is a dictionary of {group_id, subset_id}
# group_id is essentially log2(file_size_limit)
def get_all_jobs_sorted(dir_name):
	dirs = os.listdir(dir_name);
	res = [];	
	for fname in dirs:
		if fname.find("job")<0: continue;
		arr = fname.split(".")[0].split("_");
		group_id = int(arr[1]);
		subset_id = int(arr[2]);
		if len(arr)>3:
			part_id = int(arr[3]);
		else:
			part_id = 0;
		#print(group_id, subset_id);
		entry = {"group_id": group_id, "subset_id": subset_id, "part_id": part_id};
		res.append(entry);

	res.sort(key=get_order);
	return res;

# generate the batch file -> batchscripts/batch_prove2.sh
def gen_batch_file(batchscript_dir, template_name, job_dir, job):
	f1 = open(batchscript_dir+ template_name, "r");
	s = f1.read();
	f1.close();
	
	s = s.replace("FILE_NAME", batchscript_dir + job_dir + job_to_fname(job));
	f2 = open(batchscript_dir + "batch_prove2.sh", "w");
	f2.write(s);
	f2.close();

# process the job
def process_job(job, report_file, batchscript_dir, template_name, job_dir):
	os.system("rm -fr /tmp2/batchprove");
	gen_batch_file(batchscript_dir, template_name, job_dir, job);
	time1 = time.time();
	sres = run_file(batchscript_dir + "batch_prove2.sh");
	print(sres);
	time2 = time.time();
	elapsed = int(time2 - time1);
	line = "PROCESS: " + job_to_fname(job) + ": START: " + str(time1) +  ", END: " + str(time2) + ", ELAPSED: " + str(elapsed) + " seconds";
	f1 = open(report_file, "a");
	f1.write(line + "\n");
	f1.close();

def run_file(fpath):
	p = subprocess.Popen([fpath], shell=False, stdout=subprocess.PIPE);
	sAll = "";
	while p.poll() is None:
		out = p.stdout.readline();
		print(out);
		sAll += str(out);
	return sAll;

def get_all_files_in_job(job_path):
	f1 = open(job_path, "r");
	lines = f1.readlines();
	f1.close();

	files = [];
	for line in lines:
		if line.find("#")<0:
			fname = line.split()[0];
			files.append(fname);
	return files;

def get_proved_files(report_path):
	f1 = open(report_path, "r");
	lines = f1.readlines();
	f1.close();

	files = [];
	for line in lines:
		if line.find("END_PROVE")>=0:
			fname = line.split()[1];
			files.append(fname);
	return files;		
	
	
	
JOB_DIR = "jobs/";
BATCH_DIR = "batchscripts/";
TEMPLATE_NAME = "batch_prove.template";
RESULT_DIR = "batchscripts/results/";
REPORT_PATH = RESULT_DIR + "proc_all.report";
jobs = get_all_jobs_sorted(BATCH_DIR + JOB_DIR);

# used for cases where only need to generate a particular set
def main_ui():
	print("============== MAIN UI ==============");
	for i in range(len(jobs)):
		print(i, ":", job_to_fname(jobs[i]));
	start_idx = int(input("Enter start_idx: "));
	end_idx = int(input("Enter end_idx of LAST FILE (included): "));
	new_arr = jobs[start_idx: end_idx+1];
	for job in new_arr:
		process_job(job, REPORT_PATH, BATCH_DIR, TEMPLATE_NAME, JOB_DIR);
	print("========================");
	print("REPORT in " + REPORT_PATH);

# process all without asking, better suited for nohup
# nohup python3 -u scripts/PROCESS_ALL.py &
# idx_end (actually NOT included)
def main_no_ui(idx_start, idx_end):
	for job in jobs[idx_start: idx_end]:
		process_job(job, REPORT_PATH, BATCH_DIR, TEMPLATE_NAME, JOB_DIR);

# check if data is complete
def check_data_complete():
	b_full = False;
	for job in jobs:
		fname = job_to_fname(job);
		fpath = BATCH_DIR + JOB_DIR + "/" + fname;
		files = get_all_files_in_job(fpath);  

		report_path = RESULT_DIR + fname + ".report";
		pobj = Path(report_path);
		if not pobj.is_file(): 
			print("### REPORT not exists: " + fname);
			continue;
		proved = get_proved_files(report_path);

		unhandled = [];
		for sfile in files:
			sfile2 = sfile.replace("/", "_");
			if sfile2 not in proved:
				unhandled.append(sfile);
		if len(unhandled)>0:
			if b_full:
				print("!!!! MISSING the following in " + job_to_fname(job) + " !!!!!!");
				for unf in unhandled:
					print(unf); 	
			else:
				print("!!!! MISSING " + str(len(unhandled)) + " files in " + job_to_fname(job) + ", e.g., " + unhandled[0]);
	print("CHECK DATA complete ...");

# get the list of unhandled
def get_unhandled(files, proved):
	unhandled = [];
	for sfile in files:
		sfile2 = sfile.replace("/", "_");
		if sfile2 not in proved:
			unhandled.append(sfile);
	return unhandled;

# analyze report and generate a dictionary of the following entries
# job_name
# num_files 
# num_files_processed
# setup_time: one time setup time
# file_details: dictionary [prove_time]
# arr_err: list of files not proved
def analyze_report(job):
	#1. get correct and error files
	jobname = job_to_fname(job);
	fpath = BATCH_DIR + JOB_DIR + "/" + jobname;
	files = get_all_files_in_job(fpath);  
	num_files = len(files);
	report_path = RESULT_DIR + jobname + ".report";
	pobj = Path(report_path);
	if not pobj.is_file(): 
		return {"job_name": jobname,
			"num_files": num_files, "num_files_processed": 0};
	proved = get_proved_files(report_path);
	arr_err = get_unhandled(files, proved);

	#2. get the details
	f1 = open(report_path, "r");
	lines = f1.readlines();
	f1.close();
	gcd_time = -1; # in sec
	setup_time = -1;
	file_details = {};
	for line in lines:
		if line.find("PERF_USE_Batch_GCD")>=0:
			gcd_time = int(line.split()[1])//1000;
		if line.find("PERF_USE_OnetimeSetup")>=0:
			setup_time = int(line.split()[1])//1000;
		if line.find("START_PROVE")>=0:
			fname  = line.split()[1];
			cur_rec = {"file": fname};
			file_details[fname] = cur_rec;
		if line.find("PERF_USE_Proof_File")>=0:
			arrwords = line.split();
			fname = arrwords[1];
			prove_time = int(arrwords[2])//1000;
			if fname==cur_rec["file"]:
				cur_rec["prove_time"] = prove_time;
			else:
				print("UNDOCUMENTED file: " + fname);
	
	res = {"job_name": jobname, "num_files": num_files, 
		"num_files_processed": len(proved),
		"gcd_time": gcd_time,
		"setup_time": setup_time,
		"file_details": file_details,
		"arr_err": arr_err};
	return res;

def display_report(id, rpt, level):
	#1. line 1
	print("--------------------------");
	if rpt["num_files_processed"]==0: 
		state = "[X]"; 
	elif rpt["num_files_processed"] == rpt["num_files"]:
		state = "[OK]";
	else:
		state = "[" + str(rpt["num_files_processed"]) + "/" + str(rpt["num_files"]) + "]";
	print("Job " + str(id) + ": " + rpt["job_name"] + "\tState:" + state + "\t Files: " + str(rpt["num_files"])); 
	if rpt["num_files_processed"]==0:
		return;
	if rpt["num_files_processed"]<rpt["num_files"]:
		print("  FIRST err/incomplete file: ", rpt["arr_err"][0]);
		return;

	#2. total time
	if level>=2:
		total_prove_time = 0;
		for fname in rpt["file_details"].keys():
			entry = rpt["file_details"][fname];	
			if "prove_time" in entry:
				prove_time = entry["prove_time"];
				total_prove_time += prove_time;
		print("  Files: ", rpt["num_files"], ", SetUp (sec):", rpt["setup_time"], ", TotalProve (Sec):", total_prove_time, ", TOTAL (sec):", total_prove_time + rpt["setup_time"]);

	#3. avg
	if level>=3:
		gcd_time = rpt["gcd_time"];
		num_processed = rpt["num_files_processed"];
		avg_prv_time = total_prove_time//num_processed;
		avg_gcd = gcd_time//num_processed;
		print("  TotalGCD:", gcd_time, ", AvgProveTime:", avg_prv_time, 
			", AvgGCDTime:", avg_gcd, ", AdjustedProve:", 
			avg_prv_time + avg_gcd);

# level higher the more details
def report_all(level):
	id = 0;
	total_prove = 0;
	total_setup = 0;
	total_files = 0;
	for job in jobs:
		report = analyze_report(job);
		if "setup_time" not in report: 
			total_setup += 0;
		else:	
			total_setup += report["setup_time"];
		if "file_details" in report:
			dict_recs = report["file_details"];
			for fname in dict_recs.keys():
				item = dict_recs[fname];
				if "prove_time" in item:
					total_prove+= item["prove_time"];
		if "num_files_processed" in report:
			total_files += report["num_files_processed"];

		display_report(id, report, level);	
		id += 1;
	# REPORT ALL
	print("==================================================");
	print("Files: "	 + str(total_files) +
			", TotalSetup: " + str(total_setup) + " sec" + 
			", TotalProve: " + str(total_prove) + " sec");
	print("================= END REPORT =====================");
		
# -----------------------
# MAIN PROGRAM 
# -----------------------
#input("MAKE SURE TO run acc/scripts/setup_ramfs.sh AT EACH NODE!\nEnter to confirm: ");
#main_ui();
main_no_ui(3, 4);

level = 5;
report_all(level);

