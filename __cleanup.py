import os, shutil, glob

path = "/Users/wangye/Documents/courses/thermo_stats/dist_lattmc/"
i = 0
for infile in glob.iglob(os.path.join(path, '*')):
	
	bn = os.path.basename(infile)
	if bn.startswith("cubic_shape=32x32x32_temp=4.6"):
		shutil.move(bn, "/Users/wangye/Documents/courses/thermo_stats/dist_lattmc/sample_exp_2/cubic_shape/"+bn)
		i += 1
print("moved " + str(i) + " files")
