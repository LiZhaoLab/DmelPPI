"""
    Extract coil IDR in monomers
    03/14/2024, by jpeng
"""

import pandas as pd
import numpy as np

lowerbound = 5
dssp_code = "EHBGTSI-"
state3_code = "EHEHCCCC"
replace_tbl = dict(zip(dssp_code,state3_code))

def longIDR(ss_seqs,ds_scores,lddt_scores,cutoff=20):
    seqlen = len(ds_scores)
    longidrs = []
    plddts = []
    ss = []

    diso = [0] + [1 if s=="C" else 0 for s in ss_seqs] + [0]
    ## add 0 in the beginning and end to better assign flags ##
    ## so that I only need to flag indexes with 
    ## diso[idx-1]==0 & diso[idx]==1 or
    ## diso[idx]==1 & diso[idx+1]==0
    ## e.g.
    ## 01234567890123456789012
    ## 01111111110000111111110
    ## flags = [1,9,14,21]

    flags = []
    for i in range(1,seqlen+1):
        if diso[i-1]==0 and diso[i]==1:
            flags.append(i)
        if diso[i]==1 and diso[i+1]==0:
            flags.append(i)

    ## if length of ordered region is smaller than 3
    ## consider to connect the two adjacent IDRs ##
    nf = len(flags)//2

    if nf>1:
        flags2remove = []
        for i in range(nf-1):
            i1 = 2*i+1
            i2 = 2*i+2
            if flags[i2]-flags[i1]<lowerbound:
                flags2remove.append(flags[i1])
                flags2remove.append(flags[i2])

        for i in flags2remove:
            flags.remove(i)

    ## now check if there are long IDRs based on the new flags ##
    ## remove long IDRs with secondary structures ##
    nf = len(flags)//2
    longidrs = []
    idrstrs = []
    plddts = []
    ss = []
    if nf:
        for i in range(nf):
            i2 = flags[2*i+1]
            i1 = flags[2*i]
            length_idr = i2-i1

            ## long IDRs should be at least 30 aa
            if length_idr>cutoff:
                ssi = [ss_seqs[s] for s in range(i1-1,i2)]
                noss = [s=="C" for s in ssi]
                if sum(noss)/len(ssi)>0.3:
                    ss += ["".join([ss_seqs[s] for s in range(i1-1,i2)])]
                    idrstrs += ['%d-%d'%(i1,i2)]
                    plddt = [lddt_scores[s] for s in range(i1-1,i2)]
                    plddt = np.mean(plddt)
                    longidrs += [s for s in range(i1,i2+1)]
                    #plddts += [lddt_scores[s] for s in range(m,M+1)]
                    #ss += [ss_seqs[s] for s in range(m,M+1)]
                    plddts += ["%.3f"%plddt]

    idrstrs = ';'.join(idrstrs)
    ss = ";".join(ss)
    plddts = ";".join(plddts)
    return ss,longidrs,idrstrs,plddts

df = pd.read_csv("fbpp_out_pred.tsv",sep="\t")

data = {"fbpp":[],"ss":[]}
fbpp2disorder = {}
cols = df.columns
for fbpp,i,res,pl,d,rsa,ss,ds,bs in zip(*[df[col] for col in cols]):
    ss = replace_tbl[ss]
    if fbpp not in fbpp2disorder:
        fbpp2disorder[fbpp] = [[ss],[ds],[pl]]
    else:
        fbpp2disorder[fbpp][0].append(ss)
        fbpp2disorder[fbpp][1].append(ds)
        fbpp2disorder[fbpp][2].append(pl)

for fbpp in fbpp2disorder:
    data["fbpp"].append(fbpp)
    data["ss"].append("".join(fbpp2disorder[fbpp][0]))

data = pd.DataFrame(data)
data.to_csv("fbpp_ss.csv",index=False)


f = open("fbpp_longidrs_byCoil.csv","w")
f.write("fbpp,longidrs,ss,plddt\n")
coil_cutoff = 5
for fbpp in fbpp2disorder:
    ss,longidrs,idrstrs,plddts = longIDR(*fbpp2disorder[fbpp],cutoff=coil_cutoff)
    ss_all = fbpp2disorder[fbpp][0]
    ss_all = "".join(ss_all)
    if idrstrs:
        print(f"{fbpp},{idrstrs},{ss},{plddts}")
        f.write(f"{fbpp},{idrstrs},{ss},{plddts}\n")
f.close()
