# Performance Analysis EXTRACTS the itemized PERF_XXX cost
# Assuming cost in us
f1 = open("dump.txt", "r");
lines = f1.readlines();
f1.close();

import re;
setItem = [];
hashTotal = {};
sum = 0;
for line in lines:
	if line.find("PERF_USE")==0:
		arr = line.split();
		num = int(arr[1]);
		item = arr[0].split("_")[2];
		if not item in setItem:
			setItem.append(item);
			hashTotal[item] = num;
		else:
			hashTotal[item] += num;

	if line.find("REPORT_op")>=0:
		arritems = line.split(",");
		print(arritems);
		item = arritems[2];
		val_tal = item.split(":")[1].strip().split(" ")[0].strip();
		val_total = int(val_tal);
		
total_sum = 0;
for item in setItem:
	total = hashTotal[item]/1000;
	print(item + ": " + str(total) + " ms");
	total_sum += total;
	
print("TOTAL time: " + str(val_total) + ", reported total: " + str(total_sum) + ", reported rate: " + str(total_sum*100.0/val_total)[0:5] + "%");

