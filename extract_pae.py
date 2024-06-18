"""
    Extract statistics of PPI predictions by multiprocessing
"""

from plot_pae import *
from multiprocessing import Pool


ncpu = 16
#ncpu = 1

if ncpu>1:
    pool = Pool(ncpu)

dcut = 7
pcut = 15
show = "N"

f = open(f"pae_dcut{dcut}_pcut{pcut}.csv","w")
header = "pair,iptm,pae,pdockq,interfaceA,interfaceB,plddtA,plddtB,interface_pairs,ppi"
f.write(f"{header}\n")

results = []
lines = open("preflist.txt","r")
#lines = open("test.txt","r")
for line in lines:
    model = os.path.join("../AF2_PPI/",line[:-1])
    if ncpu == 1:
        out_str = plot(model,show,pcut,dcut)
        results.append(out_str)
    else:
        pool.apply_async(plot,args=(model,show,pcut,dcut,),
                            callback=results.append)

if ncpu>1:
    pool.close()
    pool.join()

for out_str in results:
    if out_str:
        f.write(f"{out_str}\n")

f.close()
