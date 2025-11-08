

import os,sys,time
import pandas as pd
import numpy as np

from scipy.spatial import distance
from scipy.stats import fisher_exact,hypergeom

from multiprocessing import Pool

neighbor = 0

mass = {'H': 1.008,
        'C': 12.01,
        'N': 14.01,
        'O': 16.00,
        'S': 32.06,
        'P': 30.97,
        'M': 0.000,
        'ZN':65.0 }

elements = {
         '1H2\'' :'H',
         '1H5\'' :'H',
         '2H2\'' :'H',
         '2H5\'' :'H',
         '1HD1':'H',
         '1HD2':'H',
         '2HD1':'H',
         '2HD2':'H',
         '3HD1':'H',
         '3HD2':'H',
         '1HE2':'H',
         '2HE2':'H',
         '1HG1':'H',
         '1HG2':'H',
         '1HH1':'H',
         '1HH2':'H',
         '2HG1':'H',
         '2HG2':'H',
         '3HG1':'H',
         '3HG2':'H',
         '2HH1':'H',
         '2HH2':'H',
         '3HG2':'H',
         '0C21':'C',
         '1C21':'C',
         '2C21':'C',
         '3C21':'C',
         '4C21':'C',
         '5C21':'C',
         '6C21':'C',
         '7C21':'C',
         '8C21':'C',
         '0C31':'C',
         '1C31':'C',
         '2C31':'C',
         '3C31':'C',
         '4C31':'C',
         '5C31':'C',
         '6C31':'C',
         'ZN':'ZN'
           }

def get_elem(atname):
    if atname[0] in mass.keys():
        return atname[0]
    else:
        return elements[atname]

class atom:
    def __init__(self,atid,atname,resname,resi,coori):
        self.atid = atid
        self.atname = atname
        self.resname = resname
        self.resi = int(resi)
        self.coori = coori

def read_coor(fpdb):
    # read pdb #
    lines = open(fpdb,"r")
    coors = {}
    nat = 0
    chains = []
    for line in lines:
        if line.startswith('ATOM '):
            atid = int(line[6:11])
            atname = line[12:16].strip()
            resname = line[17:21].strip()
            resi = line[22:26].strip()
            chain = line[21]
            #resname = "/".join([resname,resi,chain])
            x = line[30:38]
            y = line[38:46]
            z = line[46:54]

            if chain not in chains:
                chains.append(chain)
                coors[chain] = {}

            if get_elem(atname)!='H':
                coori = [float(s) for s in [x,y,z]]
                atomi = atom(atid,atname,resname,resi,coori)

                if resi not in coors[chain]:
                    coors[chain][resi] = [atomi]
                else:
                    coors[chain][resi].append(atomi)

            nat += 1
    return coors

def contact(residues1,residues2,cutoff=7):
    coor1 = [ai.coor for resi in residues1 for ai in resi]
    coor2 = [ai.coor for resi in residues2 for ai in resi]
    pairwise_distances = distance.cdist(coor1,coor2,'euclidean')
    dmin = np.min(pairwise_distances)
    if dmin>cutoff:
        return False
    return True

class IDR():
    def __init__(self,residue_range,i,j,category="idr"):
        self.category = category
        self.start = i
        self.end = j
        self.residue_range = residue_range


def numlist_to_str(numlist):
    if numlist:
        numstr = []
        idx = 0
        r0 = numlist[0]
        while idx<len(numlist)-1:
            r1 = numlist[idx+1]
            if r1-numlist[idx]>=5:
                numstr.append('%d-%d'%(r0,numlist[idx]))
                r0 = r1
            idx += 1
        if idx:
            numstr.append('%d-%d'%(r0,r1))
        else:
            numstr.append('%d-%d'%(r0,r0))
        return ';'.join(numstr)
    else:
        return ''
    
def rangeStrToList(residue_ranges):
    residues = []

    if residue_ranges == "":
        return residues

    for residue_range in residue_ranges.split(";"):
        i,j = residue_range.split("-")
        i,j = int(i),int(j)
        for m in range(i,j+1):
            residues.append(m)

    return residues

def trueIndex(binding_array):
    m = -1
    for b in binding_array[::-1]:
        if b==0:
            m -= 1
        else:
            break
    return m

## if there are significant changes in SS ##
def fisher(ss1,ss2):
    n1 = len(ss1)
    n2 = len(ss2)
    coil1 = [s=="C" for s in ss1]
    coil2 = [s=="C" for s in ss2]
    ncoil1 = sum(coil1)
    ncoil2 = sum(coil2)

    tbl = [[n2-ncoil2,ncoil2],[n1-ncoil1,ncoil1]]
    r, p = fisher_exact(tbl, alternative='greater')
    #r, p = fisher_exact(tbl)
    #print(tbl,r,p)
    return p

def hypergeom_test(ss1,ss2):
    coil1 = [s=="C" for s in ss1]
    coil2 = [s=="C" for s in ss2]
    n1 = sum(coil1)
    n2 = sum(coil2)

    M = len(ss1) ## total populations 
    n = M-n1 ## no. category of interest
    N = M ## no. of selection
    k = M-n2

    p = hypergeom(M=M,n=n,N=N).sf(k-1)
    return p

# USE CONSERVATION SCORE INSTEAD #
#def extract_dnds(fbpp,dnds_dir = "/ru-auth/local/home/jpeng/scratch/DmelPPI/alignments/"):
#    dnds = {fbpp:{}}
#    fdnds = os.path.join(dnds_dir,"raw_hit/%s.dnds.txt"%fbpp)
#    if os.path.isfile(fdnds):
#        lines = open(fdnds,"r")
#        for line in lines:
#            i,w = line[:-1].split()
#            w = float(w) if w!="-" else w
#            if w!="-":
#                dnds[fbpp][int(i)] = w
#    return dnds

def extract_conservation_score(fbpp,datadir="/ru-auth/local/home/jpeng/scratch/DmelPPI/binding_categories/af_dis2order/conservationIDR/newblastp_hits/"):
    conservation_score = {fbpp:{}}
    lines = open(os.path.join(datadir,f"{fbpp}_conservation.txt"),"r")
    for line in lines:
        site,score = line[:-1].split()
        site = int(site.split("_")[1])
        conservation_score[fbpp][site] = int(score)
    return conservation_score

def read_conservation(inp = "fbpp.txt"):
    proteins = np.loadtxt(inp,dtype=str)
    conservation = {}
    #dnds = {}
    for fbpp in proteins:
        conservation = {**conservation,**extract_conservation_score(fbpp)}
        #dnds = {**dnds,**extract_dnds(fbpp)}
    #return conservation,dnds
    return conservation

def read_coil(fcoil = "fbpp_longidrs_byCoil.csv"):
    ## hash long IDRs (COIL) into dictions ##
    df = pd.read_csv(fcoil)
    pro_with_coil = {}
    idx_with_coil = {}
    #pro_ss = {}
    for pro,idrs in zip(df.fbpp,df.longidrs):
        idr_residues = {}
        residue_ranges = idrs.split(";")
        for residue_range in residue_ranges:
            i,j = residue_range.split("-")
            idrtype = IDR(residue_range,i,j,category="coil")
            for m in range(int(i),int(j)+1):
                idr_residues[m] = idrtype
        idx_with_coil[pro] = idr_residues
        pro_with_coil[pro] = idrs

    return pro_with_coil,idx_with_coil

def read_CF(fpred = "fbpp_longidrs_byPred.csv"):
    ## hash long IDRs (AlphaFold_Disorder) into dictions ##
    df = pd.read_csv(fpred)
    pro_with_CF = {}
    idx_with_CF = {}
    for pro,idrs in zip(df.fbpp,df.longidrs):
        idr_residues = {}
        residue_ranges = idrs.split(";")
        for residue_range in residue_ranges:
            i,j = residue_range.split("-") 
            idrtype = IDR(residue_range,i,j,category="CF")
            for m in range(int(i),int(j)+1):
                idr_residues[m] = idrtype
        idx_with_CF[pro] = idr_residues
        pro_with_CF[pro] = idrs

    return pro_with_CF,idx_with_CF

def read_diso(fpred = "fbpp_longidrs_byPred_noCF.csv"):
    ## hash long IDRs (AlphaFold_Disorder) into dictions ##
    df = pd.read_csv(fpred)
    pro_with_diso = {}
    idx_with_diso = {}
    for pro,idrs in zip(df.fbpp,df.longidrs):
        idr_residues = {}
        residue_ranges = idrs.split(";")
        for residue_range in residue_ranges:
            i,j = residue_range.split("-")
            idrtype = IDR(residue_range,i,j,category="diso")
            for m in range(int(i),int(j)+1):
                idr_residues[m] = idrtype
        idx_with_diso[pro] = idr_residues
        pro_with_diso[pro] = idrs

    return pro_with_diso,idx_with_diso

def combine_idr(*idr_ranges):

    residues = []
    for idr_range in idr_ranges:
        if type(idr_range) == str:
            residues += rangeStrToList(idr_range)

    residues = sorted(list(set(residues)))

    ranges = numlist_to_str(residues)

    return ranges

def write_longidrs(pro_with_coil,idx_with_coil,pro_with_CF,idx_with_CF,pro_with_diso,idx_with_diso,cutoff=30):
    ## now we have all the index of IDR from coil and disorder predictions ##
    idx_with_longidrs = {}
    keys1 = [p for p in pro_with_coil]
    keys2 = [p for p in pro_with_diso]
    keys3 = [p for p in pro_with_CF]
    keys = list(set(keys1+keys2+keys3))

    #data = {"fbpp":[],"IDR_coil":[],"IDR_diso":[],"IDR_CF":[]}
    data = {"fbpp":[],"IDR":[],"coil":[],"CF":[],"category":[]}
    idx_with_idrs = {}
    pro_with_idrs = {}
    for key in keys:
        idx_with_idrs[key] = {}
        pro_with_idrs[key] = ""

        idr_coil = ""
        idr_CF = ""
        idr_diso = ""

        if key in pro_with_coil:
            idr_coil = pro_with_coil[key] 
            #idx_with_longidrs[key].append(idx_with_coil[key])

        if key in pro_with_CF:
            idr_CF = pro_with_CF[key]
            #idx_with_longidrs[key].append(idx_with_CF[key])

        if key in pro_with_diso:
            idr_diso = pro_with_diso[key]
            #idx_with_longidrs[key].append(idx_with_diso[key])
            

        idr_ranges = combine_idr(idr_CF,idr_diso,idr_coil)
        pro_with_idrs[key] = idr_ranges
        idr_strs = idr_ranges.split(";")
        n_idr = len(idr_strs)
        idrs = ["" for i in range(n_idr)]
        coils = [[] for i in range(n_idr)]
        CFs = [[] for i in range(n_idr)]

        for n in range(n_idr):
            idr_range = idr_strs[n]
            i,j = idr_range.split("-")
            i,j = int(i),int(j)
            length = j-i+1
            category = "longIDR" if length>=cutoff else "shortIDR"
            idrtype = IDR(idr_range,i,j,category)
            
            idrs[n] = idrtype # type: ignore
            
            for m in range(i,j+1):
                idx_with_idrs[key][m] = idrtype
                if idr_coil:
                    if m in idx_with_coil[key]:
                        coiltype = idx_with_coil[key][m]
                        if coiltype.residue_range not in coils[n]:
                            coils[n].append(coiltype.residue_range)
                if idr_CF:
                    if m in idx_with_CF[key]:
                        cftype = idx_with_CF[key][m]
                        if cftype.residue_range not in CFs[n]:
                            CFs[n].append(cftype.residue_range)
            coils[n] = ";".join(coils[n]) # type: ignore
            CFs[n] = ";".join(CFs[n]) # type: ignore
                
        
            ## DISORDER
            data["fbpp"].append(key)
            data["IDR"].append(idr_range)
            data["coil"].append(coils[n])
            data["CF"].append(CFs[n])
            data["category"].append(category)
            #print(key,idr_coil,idr_diso)

    data = pd.DataFrame(data)
    data.to_csv("fbpp_idrs.csv",index=False)
    
    
    return pro_with_idrs,idx_with_idrs

def extract_SS(fmonomer = "fbpp_ss.csv", fcomplex = "pairs_ss.csv"):
    ## hash monomer secondary structure information into dictions ##
    ##
    dssp_code   = "HGIEBTSCP-"
    state3_code = "HHHEECCCCC"
    replace_tbl = dict(zip(dssp_code,state3_code))
    #pro_ss = dict(zip(df.fbpp,df.ss))
    df = pd.read_csv(fmonomer)
    pro_ss = {}
    for fbpp,ss in zip(df.fbpp,df.ss):
        for k in replace_tbl:
            ss = ss.replace(k,replace_tbl[k])
        pro_ss[fbpp] = ss

    ## hash complex secondary structure information into dictions ## 
    ## 
    pair_ss = {}
    df = pd.read_csv(fcomplex)
    for pair,ss1,ss2 in zip(df.pair,df.ss1,df.ss2):
        p1,p2 = pair.split("_")
        #p1 = p1[5:]
        #p2 = p2[5:]
        #print(pair,ss1,ss2)
        for k in replace_tbl:
            ss1 = ss1.replace(k,replace_tbl[k])
            ss2 = ss2.replace(k,replace_tbl[k])

        pair_ss[(p1,p2)] = {p1:ss1,p2:ss2}

    return pro_ss,pair_ss


def check_idr_binding(p1,p2,interface1,interface2,pdockq,pair_ss,pro_ss,idx_with_coil,idx_with_CF,idx_with_idrs):
    ## read in coordinates ##
    fpdb = f"./PDB/7227.{p1}_7227.{p2}.pdb"
    #coors = read_coor(fpdb)
    chainkeys = [ch for ch in coors]
    #coors_p1 = coors[chainkeys[0]]
    #coors_p2 = coors[chainkeys[1]]
    #coors = {p1:coors_p1,p2:coors_p2}
    #print(interface1)
    #print(interface2)

    ## determine whether long IDRs are in interfaces &
    ## whether longIDRs involves disorder to order transitions
    ## by default, proteins are not invloved in IDR to order transitions
    #pair = f"{p1}|{p2}"
    pair = (p1,p2)
    #interface1 = interface1.split(":")[1]
    #interface2 = interface2.split(":")[1]
    interface = {p1:interface1,p2:interface2}
    FLAG_idrBinding = {p1:False,p2:False}
    FLAG_coilBinding = {p1:False,p2:False}
    FLAG_coil2order = {p1:False,p2:False}
    FLAG_CFBinding = {p1:False,p2:False} ## binding with conditionally folding regions
    FLAG_orderBinding = {p1:False,p2:False}

    coil_interface = {p1:"",p2:""}
    idr_interface = {p1:"",p2:""}
    idr_type = {p1:"",p2:""}
    CF_interface = {p1:"",p2:""}
    order_interface = {p1:"",p2:""}
    coil2order_if = {p1:[],p2:[]}
    coil2order_pvalues = {p1:"",p2:""}

    ss_monomer = {p1:"",p2:""}
    ss_interface = {p1:"",p2:""}
    idr_regions = {p1:"",p2:""}
    order_regions = {p1:"",p2:""}

    for pro in interface:
        #print(pair,pro)
        pro_len = len(pair_ss[pair][pro])

        ## extract interface residues 
        ## use a diction for fast lookup ##
        interface_residues = {}
        residue_ranges = interface[pro].split(";")
        for residue_range in residue_ranges:
            try:
                i,j = residue_range.split("-") 
                for m in range(int(i),int(j)+1):
                    if m>0 and m<=pro_len:
                        interface_residues[m] = 1
            except ValueError:
                pass
        ##

        flag_coilBinding = 0
        flag_idrBinding = 0
        flag_coil2order = 0
        flag_orderBinding = 0
        flag_CFBinding = 0
        ssm = []
        ssc = []
        #idr_idx = []
        pvaluesm = []

        coil_if_residues = [] ## interface residues in IDR_coil
        coil_if_ranges = [] ## determine whether interface residues are in coil IDRs

        CF_if_residues = [] ## interface residues in CF
        CF_if_ranges = [] ## determine whether interface residues are in diso IDRs

        idr_if_residues = [] ## interface residues in IDR_coil
        idr_if_ranges = [] ## determine whether interface residues are in coil IDRs

        order_if_residues = [] ## interface residues in IDR_diso
        
        idr_types = []

        order_residues = []
        idr_residues = []
        idx_order = {}
        for i in range(1,pro_len+1):
            if pro not in idx_with_idrs:
                order_residues.append(i)
                idx_order[i] = 1
            else:
                if i not in idx_with_idrs[pro]:
                    order_residues.append(i)
                    idx_order[i] = 1
                else:
                    idr_residues.append(i)
        order_regions[pro] = numlist_to_str(order_residues)
        idr_regions[pro] = numlist_to_str(idr_residues)

        for m in interface_residues:
            if pro in idx_with_coil:
                ## IDR that appeared to be coil in AF2 predictions
                if m in idx_with_coil[pro]:
                    m_range = idx_with_coil[pro][m].residue_range
                    coil_if_residues.append(m)
                    flag_coilBinding += 1

            if pro in idx_with_CF:
                if m in idx_with_CF[pro]:
                    ## conditionally folding IDR (IDR in prediction with plddt>70)
                    m_range = idx_with_CF[pro][m].residue_range
                    CF_if_residues.append(m)
                    flag_CFBinding += 1
                    
            if pro in idx_with_idrs:
                if m in idx_with_idrs[pro]:
                    # IDR from both predictions and coil in AF2 prediction ##
                    m_range = idx_with_idrs[pro][m].residue_range
                    idr_if_residues.append(m)
                    flag_idrBinding += 1
                    idr_types.append(idx_with_idrs[pro][m].category)

            if m in idx_order:
                order_if_residues.append(m)
                flag_orderBinding += 1

        
        ## finalizing ordered interface
        order_if_residues_new = sorted(list(set(order_if_residues)))
        order_if_ranges = numlist_to_str(order_if_residues_new)

        ## finalizing IDR interface
        idr_if_residues_new = sorted(list(set(idr_if_residues)))
        idr_if_ranges = numlist_to_str(idr_if_residues_new)
        
        #print(pro,idr_types)
        ##types of IDR
        if len(idr_types)==0:
            idr_type[pro] = "None"
        else:
            if len(list(set(idr_types))):
                idr_type[pro] = idr_types[0]
            else:
                idr_type[pro] = "Mix"

        ## finalizing conditionally folding IDR interface
        CF_if_residues_new = sorted(list(set(CF_if_residues)))
        CF_if_ranges = numlist_to_str(CF_if_residues_new)

        ## finalizing coil IDR interface
        coil_if_residues_new = sorted(list(set(coil_if_residues)))
        coil_if_ranges = numlist_to_str(coil_if_residues_new)


        #if pair==("FBpp0070584","FBpp0086468"): # jpeng to debug
        if pair==("FBpp0087498","FBpp0088489"): # jpeng to debug
            print(interface_residues)
            print(coil2order_if)

        #print(pair,p1,p2,pro_len,if_ranges)
        ## if coil IDR, is it coil2order?
        if coil_if_ranges:
            #print(pair,pro,if_ranges,flag)
            for idr_range in coil_if_ranges.split(";"):
                i,j = idr_range.split("-")
                i,j = int(i),int(j)
         
                ssm_i = "".join([pro_ss[pro][m] for m in range(i-1,j)])
                ssc_i = "".join([pair_ss[pair][pro][m] for m in range(i-1,j)])
         
                p = fisher(ssm_i,ssc_i)
                if p<0.05:
                    flag_coil2order = True
                    coil2order_if[pro].append(idr_range)
                #p = hypergeom_test(ssm_i,ssc_i)
                    pvaluesm.append("%.1e"%p)
         
                    ssm.append(ssm_i)
                    ssc.append(ssc_i)
         
                #print(pair,pro,idr_range,ssm_i,ssc_i,p)
        coil2order_if[pro] = ";".join(coil2order_if[pro]) # type: ignore


        ## Finalizing all
        FLAG_CFBinding[pro] = flag_CFBinding >0
        FLAG_coilBinding[pro] = flag_coilBinding>0
        FLAG_orderBinding[pro] = flag_orderBinding>0
        FLAG_idrBinding[pro] = flag_idrBinding>0
        FLAG_coil2order[pro] = flag_coil2order>0
        ss_monomer[pro] = ";".join(ssm)
        ss_interface[pro] = ";".join(ssc)
        #idr_interface[pro] = numlist_to_str(idr_idx)
        idr_interface[pro] = idr_if_ranges
        coil_interface[pro] = coil_if_ranges
        CF_interface[pro] = CF_if_ranges
        order_interface[pro] = order_if_ranges
        coil2order_pvalues[pro] = ";".join(pvaluesm)
    
    category = ["order","idr","coil","CF","coil2order"]
         
    idrBinding = [FLAG_idrBinding[pro] for pro in pair]
    coilBinding = [FLAG_coilBinding[pro] for pro in pair]
    CFBinding = [FLAG_CFBinding[pro] for pro in pair]
    orderBinding = [FLAG_orderBinding[pro] for pro in pair]
    coil2order = [FLAG_coil2order[pro] for pro in pair]

    index1 = [binding[0] for binding in [orderBinding,idrBinding,coilBinding,CFBinding,coil2order]]
    index2 = [binding[1] for binding in [orderBinding,idrBinding,coilBinding,CFBinding,coil2order]]

    idx1 = trueIndex(index1)
    idx2 = trueIndex(index2)
    
    interaction_type = f"{category[idx1]} to {category[idx2]}"
    

    result_dict = {
        "p1":p1,
        "p2":p2,
        "length1":len(pair_ss[pair][p1]),
        "length2":len(pair_ss[pair][p2]),
        "pdockq":pdockq,
        "idr_binding1":FLAG_idrBinding[p1],
        "idr_binding2":FLAG_idrBinding[p2],
        "coil_binding1":FLAG_coilBinding[p1],
        "coil_binding2":FLAG_coilBinding[p2],
        "CF_binding1":FLAG_CFBinding[p1],
        "CF_binding2":FLAG_CFBinding[p2],
        "coil2order_binding1":FLAG_coil2order[p1],
        "coil2order_binding2":FLAG_coil2order[p2],
        "order_binding1":FLAG_orderBinding[p1],
        "order_binding2":FLAG_orderBinding[p2],
        "interface1":interface[p1],
        "interface2":interface[p2],
        "idr_if1":idr_interface[p1],
        "idr_if2":idr_interface[p2],
        "coil_if1":coil_interface[p1],
        "coil_if2":coil_interface[p2],
        "CF_if1":CF_interface[p1],
        "CF_if2":CF_interface[p2],
        "coil2order_if1":coil2order_if[p1],
        "coil2order_if2":coil2order_if[p2],
        "order_if1":order_interface[p1],
        "order_if2":order_interface[p2],
        "monomer_ss_coil_if1":ss_monomer[p1],
        "monomer_ss_coil_if2":ss_monomer[p2],
        "complex_ss_coil_if1":ss_interface[p1],
        "complex_ss_coil_if2":ss_interface[p2],
        "pvalue_coil_if1":coil2order_pvalues[p1],
        "pvalue_coil_if2":coil2order_pvalues[p2],
        "idr_region1":idr_regions[p1],
        "idr_region2":idr_regions[p2],
        "order_region1":order_regions[p1],
        "order_region2":order_regions[p2],
        "idr_type1":idr_type[p1],
        "idr_type2":idr_type[p2],
        "interaction_type":interaction_type,
    }

    #results = [p1,p2,n1,n2,i1,i2,pdockq,ib1,ib2,cb1,cb2,co1,co2,db1,db2,ob1,ob2,coil1,coil1c,coil2,coil2c,df1,df1c,df2,df2c,cof1,cof1c,cof2,cof2c,cf1,cf1c,cf2,cf2c,of1,of1c,of2,of2c,sm1,sm2,sc1,sc2,pv1,pv2]
    #line = f"{p1},{p2},{n1},{n2},{i1},{i2},{pdockq},{ib1},{ib2},{cb1},{cb2},{co1},{co2},{db1},{db2},{ob1},{ob2},{coil1},{coil1c},{coil2},{coil2c},{df1},{df1c},{df2},{df2c},{cof1},{cof1c},{cof2},{cof2c},{cf1},{cf1c},{cf2},{cf2c},{of1},{of1c},{of2},{of2c},{sm1},{sm2},{sc1},{sc2},{pv1},{pv2}"
    #return results,line
    #print(p1,p2)
    return result_dict


if __name__ == "__main__":

    t0 = time.time()
    #cut = 0.5
    cut = 0
    if cut==0.5:
        conf = "highconf"
    elif cut==0.23:
        conf = "acceptable"
    elif cut==0:
        conf = "possible"
    else:
        sys.exit("cut can only be 0, 0.5, or 0.23")

    ## loop over AF multimer predictions ##
    ## order with order interaction ##
    ## I think there are two =categories as following ##
    ## 1. PPI with one or more order-order interfaces
    ## 2. PPI with all order-order interfaces (no disorder binding)
    pae = pd.read_csv("pae_summary.csv")
    if cut!=0:
        pae = pae.loc[pae.pdockq>=cut].copy()
    else:
        pae = pae.loc[(pae.iptm>=50) & (pae.pae<=10.0) & (pae.pdockq<0.23)].copy()
    pae.to_csv("pae_%s.csv"%conf,index=False)
    t1 = time.time()
    t = t1-t0
    t0 = t1
    print(f"Finished loading AF2 multimer predictions in {t: .1f} seconds")

    ## read in IDR (coil and AlphaFold_disorder) ##
    pro_with_coil,idx_with_coil = read_coil(fcoil = "fbpp_longidrs_byCoil.csv")
    pro_with_CF,idx_with_CF = read_CF(fpred = "fbpp_longidrs_byPred.csv")
    pro_with_diso,idx_with_diso = read_diso(fpred = "fbpp_longidrs_byPred_noCF.csv")
    pro_with_idrs,idx_with_idrs = write_longidrs(pro_with_coil,idx_with_coil,pro_with_CF,idx_with_CF,pro_with_diso,idx_with_diso)

    ## read in secondary structures ##
    pro_ss,pair_ss = extract_SS(fmonomer = "fbpp_ss.csv", fcomplex = "pairs_ss.csv")
    t1 = time.time()
    t = t1-t0
    t0 = t1
    print(f"Finished loading disorder predictions in {t: .1f} seconds")

    ## read conservation score ##
    #conservation_score,dnds = read_conservation()
    conservation_score = {}
    #dnds = {}
    t1 = time.time()
    t = t1-t0
    t0 = t1
    print(f"Finished loading conservation per site in {t: .1f} seconds")

    fout = open("pairs_%s_with_longidrs.csv"%conf,"w")
    fcfb = open("pairs_%s_with_longidrs_cf.csv"%conf,"w")
    fidr = open("pairs_%s_with_longidrs_coil.csv"%conf,"w")
    ford = open("pairs_%s_with_longidrs_order.csv"%conf,"w")
    fsig = open("pairs_%s_with_longidrs.significant.csv"%conf,"w")

    #header = "p1,p2,length1,length2,interface1,interface2,pdockq,coil_binding1,coil_binding2,cf_binding1,cf_binding2,coil2order1,coil2order2,diso_binding1,diso_binding2,order_binding1,order_binding2,coil_if1,conservation_coil_if1,coil_if2,conservation_coil_if2,diso_if1,conservation_diso_if1,diso_if2,conservation_diso_if2,coil2order_if1,coil2order_if1c,coil2order_if2,coil2order_if2c,cf_if1,conservation_cf_if1,cf_if2,conservation_cf_if2,order_if1,conservation_order_if1,order_if2,conservation_order_if2,ss_monomer1,ss_monomer2,ss_complex1,ss_complex2,pval1,pval2"
    #p1,p2,n1,n2,i1,i2,ib1,ib2,cb1,cb2,db1,db2,ob1,ob2,coil1,coil2,cf1,cf2,of1,of2,sm1,sm2,sc1,sc2,pv1,pv2 = results

    result_keys = [
        "p1",
        "p2",
        "length1",
        "length2",
        "pdockq",
        "idr_binding1",
        "idr_binding2",
        "coil_binding1",
        "coil_binding2",
        "CF_binding1",
        "CF_binding2",
        "coil2order_binding1",
        "coil2order_binding2",
        "order_binding1",
        "order_binding2",
        "interface1",
        "interface2",
        "idr_if1",
        "idr_if2",
        "coil_if1",
        "coil_if2",
        "CF_if1",
        "CF_if2",
        "coil2order_if1",
        "coil2order_if2",
        "order_if1",
        "order_if2",
        "monomer_ss_coil_if1",
        "monomer_ss_coil_if2",
        "complex_ss_coil_if1",
        "complex_ss_coil_if2",
        "pvalue_coil_if1",
        "pvalue_coil_if2",
        "idr_region1",
        "idr_region2",
        "order_region1",
        "order_region2",
        "idr_type1",
        "idr_type2",
        "interaction_type",
    ]
    header = ",".join(result_keys)

    fout.write(header+"\n")
    fsig.write(header+"\n")
    fcfb.write(header+"\n")
    fidr.write(header+"\n")
    ford.write(header+"\n")
    #print(header)

    print("Start analyzing interfaces")
    results = []

    ncpu = 1
    if ncpu>1:
        pool = Pool(ncpu)

    for pair,pdockq,interface_pairs in zip(pae.pair,pae.pdockq,pae.interface_pairs):
        ##
        p1,p2 = pair.split("_")
        p1 = p1[5:]
        p2 = p2[5:]
        pair = (p1,p2)
        print(interface_pairs.split(";"))
        
        
        for interface_pair in interface_pairs.split(";"):
            interface1, interface2 = interface_pair.split("|")
            #results,line = check_idr_binding(p1,p2,interface1,interface2,pdockq,pair_ss,pro_ss,idx_with_coil,idx_with_CF,idx_with_diso,conservation_score)
            if ncpu==1:
                result_dict = check_idr_binding(p1,p2,interface1,interface2,pdockq,pair_ss,pro_ss,idx_with_coil,idx_with_CF,idx_with_idrs)
                results.append(result_dict)

            else:
                #print(f"entered pool {pair}")
                pool.apply_async(check_idr_binding,args=(p1,p2,interface1,interface2,pdockq,pair_ss,pro_ss,idx_with_coil,idx_with_CF,idx_with_idrs,),
                                 callback=results.append)

            #p1,p2,n1,n2,i1,i2,pdq,ib1,ib2,cb1,cb2,db1,db2,ob1,ob2,coil1,coil1c,coil2,coil2c,df1,df1c,df2,df2c,cf1,cf1c,cf2,cf2c,of1,of1c,of2,of2c,sm1,sm2,sc1,sc2,pv1,pv2 = results
            #p1,p2,n1,n2,i1,i2,pdockq,ib1,ib2,cb1,cb2,co1,co2,db1,db2,ob1,ob2,coil1,coil1c,coil2,coil2c,df1,df1c,df2,df2c,cof1,cof1c,cof2,cof2c,cf1,cf1c,cf2,cf2c,of1,of1c,of2,of2c,sm1,sm2,sc1,sc2,pv1,pv2 = results
        break
            
    if ncpu>1:
        pool.close()
        pool.join()

    for result_dict in results:
        line = []
        for key in result_dict:
            if type(result_dict)==float:
                line.append(f"{result_dict[key]:.1e}") # type: ignore
            else:
                line.append(f"{result_dict[key]}") # type: ignore
        line = ",".join(line)
        fout.write(line+"\n")

        if result_dict["idr_binding1"] or result_dict["idr_binding2"]: # type: ignore
            # IDR binding (either coil or AlphaFold_disorder)
            #line = f"{p1},{p2},{n1},{n2},{ib1},{ib2},{sm1},{sm2},{sc1},{sc2},{pv1},{pv2},{d1},{d2}"
            fidr.write(line+"\n")
            #print(line)

        if result_dict["coil2order_binding1"] or result_dict["coil2order_binding2"]: # type: ignore
            #(f"{p1},{p2},{n1},{n2},{ib1},{ib2},{sm1},{sm2},{sc1},{sc2},{pv1},{pv2},{d1},{d2}")
            fsig.write(line+"\n")

        if result_dict["CF_binding1"] or result_dict["CF_binding2"]: # type: ignore
            fcfb.write(line+"\n")

        if result_dict["order_binding1"] or result_dict["order_binding2"]: # type: ignore
            ford.write(line+"\n")         
    
    t1 = time.time()
    t = t1-t0
    t0 = t1
    print(f"Finished analyzing in {t: .1f} seconds")


    fout.close()
    fsig.close()
    fcfb.close()
    fidr.close()
    ford.close()
