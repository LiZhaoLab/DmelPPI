"""
 Version 2.0.0, 06/10/2024

 In v2.0.0 of plot_pae.py, we can have pairs of interacting interfaces
 for example,the previous interface of 7227.FBpp0304269_7227.FBpp0305117 was:
   A:1-21;70-82;86-86;102-102|B:34-35;51-52;63-71;87-92;97-97;105-126;135-152;
    156-156
 In this form, we cannot know which interfaces are interacting with each other,
 Now, the interface is:
   1-21|34-35;1-21|63-71;1-21|87-92;1-21|105-126;1-21|135-152;1-21|156-156;
    70-82|51-52;70-82|63-71;70-82|87-92;70-82|135-152;86-86|63-71;
    102-102|63-71;102-102|97-97
 In this form, we can know all the pairs of interacting interfaces
"""


from scipy.spatial.distance import cdist
from pdb2seq import *

import os,sys,shutil,json,pickle,matplotlib,argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

neighbor = 3 ## if two interface residues are within 3 residues, e.g. 1 and 3, then the interface will be 1-3

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

def Union(lst1, lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list

def get_elem(atname):
    if atname[0] in mass.keys():
        return atname[0]
    else:
        return elements[atname]

def numlist_to_str(numlist):
    if numlist:
        numstr = []
        idx = 0
        r0 = numlist[0]
        while idx<len(numlist)-1:
            r1 = numlist[idx+1]
            if r1-numlist[idx]>neighbor:
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

def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

class atom:
    def __init__(self,atid,atname,resname,resi,coori):
        self.atid = atid
        self.atname = atname
        self.resname = resname
        self.resi = int(resi)
        self.coori = coori

# adapted from Burke et al, NSMB, 2023 #
def sigmoid(X,dcut):
    if dcut <=5:
        #5 SpearmanrResult(correlation=0.765, pvalue=4.75e-280)
        parm=[6.96234405e-01, 2.35483775e+02, 2.25322970e-02, 2.88445245e-02]
        #0.7805034405869632
    elif dcut <=6:
        #6 SpearmanrResult(correlation=0.771, pvalue=2.71e-287)
        parm=[7.02605033e-01, 2.91749822e+02, 2.70621128e-02, 2.25416051e-02]
        #0.7871982094514278
    elif dcut <=7:
        #7 SpearmanrResult(correlation=0.771, pvalue=2.24e-287)
        parm=[7.06385097e-01, 3.32456259e+02, 2.97005237e-02, 2.24488132e-02]
        #0.7859609807320201
    elif dcut <=8:
        #8 SpearmanrResult(correlation=0.763, pvalue=2.34e-278)
        parm=[7.18442739e-01,3.60791204e+02,3.01635944e-02, 2.04076969e-02]
        #0.7764648775754815
    elif dcut <=9:
        #9 SpearmanrResult(correlation=0.750, pvalue=4.54e-263)
        parm=[7.23328534e-01, 3.80036094e+02, 3.06316084e-02, 1.98471192e-02]
        #0.7608417399783565
    elif dcut <=10:
        #10 SpearmanrResult(correlation=0.733, pvalue=7.99e-246)
        parm=[7.20293782e-01, 3.95627723e+02, 3.15235037e-02, 2.37304238e-02]
        #0.7431426093979494
    elif dcut <=11:
        #11 SpearmanrResult(correlation=0.713, pvalue=1.75e-226)
        parm=[7.22015998e-01, 4.09095024e+02, 3.11905555e-02, 2.59467513e-02]
        #0.7219615906164123
    else:
        #12 SpearmanrResult(correlation=0.694, pvalue=9.28e-210)
        parm=[7.20555781e-01, 4.21033584e+02, 3.09024241e-02, 2.88659629e-02]
        #0.7023000652310362

    L,x0,k,b = parm
    Q = L / (1 + np.exp(-k*(X-x0)))+b
    return Q

def pdockq(fpdb,fpkl,PAE,pcut,dcut):
    ## load pickle file ##
    data = pickle.load(open(fpkl,"rb"))

    ## get interface ##
    # read pdb #
    lines = open(fpdb,"r")
    pdbatoms = []
    nat = 0
    chains = {}
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

            if get_elem(atname)!='H':
                coori = [float(s) for s in [x,y,z]]
                atomi = atom(atid,atname,resname,resi,coori)
                pdbatoms += [atomi]

                if chain not in chains:
                    chains[chain] = [atomi]
                else:
                    chains[chain].append(atomi)
            nat += 1

    chainkeys = [key for key in chains]
    chainA = chains[chainkeys[0]]
    chainB = chains[chainkeys[1]]

    coorA = np.array([ai.coori for ai in chainA])
    coorB = np.array([ai.coori for ai in chainB])

    # compute distance for two list #
    dmat = cdist(coorA,coorB,'euclidean')
    #print(dmat.shape)

    # extract pairs with distance < dcut #
    imat = np.where(dmat<dcut)
    indexA = imat[0]
    indexB = imat[1]
    #print(imat)
    #print(len(indexA),len(indexB))
    residuesA = np.array([ai.resi for ai in chainA])
    residuesB = np.array([ai.resi for ai in chainB])
    uni_residuesA = sorted(np.unique(residuesA))
    uni_residuesB = sorted(np.unique(residuesB))
    #print(residuesA)

    nresA = len(uni_residuesA)
    nresB = len(uni_residuesB)
    
    interfaceA = []
    interfaceB = []

    interacting = False
    if_respair = {}
    for atpair in zip(indexA,indexB):
        atn1,atn2 = atpair
        res1,res2 = residuesA[atn1],residuesB[atn2]
        idx1,idx2 = res1,nresA+res2
        atpair_pae = min(PAE[idx1-1,idx2-1],PAE[idx2-1,idx1-1])
        if atpair_pae < pcut: ## interface with PAE < pcut
            #print(atpair,res1,res2,idx2,atpair_pae)
            interfaceA.append(res1)
            interfaceB.append(res2)
            if_respair[(res1,res2)] = 1
            interacting = True

    #for pair in if_respair:
    #    print(pair)
    #
    interfaceA = sorted(list(set(interfaceA)))
    interfaceB = sorted(list(set(interfaceB)))

    ifA_str = numlist_to_str(interfaceA)
    ifB_str = numlist_to_str(interfaceB)

    if not interacting:
        interacting_pairs = ""
        Q = sigmoid(0,dcut)
        plddtA = np.nan
        plddtB = np.nan
        return(Q,ifA_str,ifB_str,plddtA,plddtB,interacting_pairs)

    if_pairs = []
    for irange_A in ifA_str.split(";"):
        for irange_B in ifB_str.split(";"):
            binding = False
            i1,j1 = irange_A.split("-")
            i2,j2 = irange_B.split("-")
            for iA in range(int(i1),int(j1)+1):
                for iB in range(int(i2),int(j2)+1):
                    if (iA,iB) in if_respair:                    
                        #print(iA,iB)
                        binding = True
                        interacting = True
                        break
            if binding:
                if_pairs.append(f"{irange_A}|{irange_B}")

    num_interface_res = len(interfaceA) + len(interfaceB)
    interacting_res = 'A:%s|B:%s'%(ifA_str,ifB_str)
    interacting_pairs = ";".join(if_pairs)
    #print(interacting_pairs)

    ## number of interface residues ##
    #ifA_str = numlist_to_str(interfaceA)
    #ifB_str = numlist_to_str(interfaceB)
    #interacting_res = 'A:%s|B:%s'%(ifA_str,ifB_str)

    plddt = data['plddt']
    iptm  = data['iptm']

    if interacting:
        plddtA = plddt[[i-1 for i in interfaceA]]
        plddtB = plddt[[i+nresA-1 for i in interfaceB]]
        ave_plddt = np.mean([p for p in plddtA]+[p for p in plddtB])
        plddtA = np.mean(plddtA)
        plddtB = np.mean(plddtB)
        #print(ave_plddt,num_interface_res)
        X = ave_plddt*np.log(num_interface_res)
    else:
        X = 0

    Q = sigmoid(X,dcut)

    return(Q,ifA_str,ifB_str,plddtA,plddtB,interacting_pairs)
    
def plot(modelname,showfigure,pcut,dcut):
    ## check if the simulation is finished ##
    name = os.path.basename(modelname)
    rankingf = "%s/ranking_debug.json"%modelname
    if os.path.isfile(rankingf):
        data = json.load(open(rankingf))
        topm = data["order"][0]
        pkl = "%s/result_%s.pkl"%(modelname,topm)
        pdb = "%s/ranked_0.pdb"%modelname
        shutil.copy(pdb,"%s.pdb"%modelname)

        ## determine if it's multimer or monomer ##
        sequence = pdb2seq(pdb)
        if len(sequence)>1:
            model = "multimer"
            prolen = [len(sequence[key]) for key in sequence]
        else:
            model = "monomer"
        #print("Prediction finished, plotting best model")

    ## if the prediction is not finished, check if there are any outputs ##
    else:
        max_pdb = ""
        max_pkl = ""
        max_score = 0
        for f in os.listdir(modelname):
            if f.endswith("pkl"):
                pkl = "%s/%s"%(modelname,f)
                data = pickle.load(open(pkl,"rb"))
                if "iptm" in data:
                    key = "iptm"
                elif "ptm" in data:
                    key = "ptm"
                else:
                    key = ""
                if key:
                    if data[key] > max_score:
                        max_pkl = pkl
                        max_score = data[key]
        if max_pkl == "":
            #sys.stderr.write("%s prediction with no outputs, quit\n"%name)
            #sys.exit(0)
            return ""

        ## if there are outputs, check the first output model ##
        pkl = max_pkl
        max_name = os.path.basename(pkl)[7:-4]
        max_pdb = "%s/relaxed_%s.pdb"%(modelname,max_name)
        if os.path.isfile(max_pdb):
            pdb = max_pdb
        else:
            pdb = "%s/unrelaxed_%s.pdb"%(modelname,max_name)
        if showfigure=="Y":
            print("Not finished, plot current best model: %s"%max_name)
        sequence = pdb2seq(pdb)
        if len(sequence)>1:
            model = "multimer"
            prolen = [len(sequence[key]) for key in sequence]
        else:
            model = "monomer"

        ## still extract current best model ##
        shutil.copy(pdb,"%s.pdb"%modelname)

    data = pickle.load(open(pkl,"rb"))
    PAE  = data["predicted_aligned_error"]
    #print(PAE.shape)
    maxp = data["max_predicted_aligned_error"]
    ndim = PAE.shape[0]
    min_PAE = maxp

    if model=="multimer":
        iptm = data["iptm"] * 100
        diag_PAE1 = PAE[0:prolen[0],prolen[0]:ndim]
        min1 = np.min(diag_PAE1)
        diag_PAE2 = PAE[prolen[0]:ndim,0:prolen[0]].T
        diag_PAE = np.concatenate((diag_PAE1,diag_PAE2))
        min2 = np.min(diag_PAE2)
        min_PAE = np.max([min1,min2])

        Q,ifA_str,ifB_str,plddtA,plddtB,interacting_pairs = pdockq(pdb,pkl,PAE,
                                                                   pcut,dcut)

        ppi = "no"
        if Q>=0.5:
            ppi = "highconf"
        elif Q>=0.23:
            ppi = "acceptable"
        #out_str = "%s ipTM: %2d PAE: %.1f pDockQ: %.3f %s %s"%(modelname,score,
        #                                        min_PAE,Q,interacting_res,ppi)
        #print(out_str)
        #out_str = "%s,%2d,%.1f,%.3f,%s,%s,%"%(modelname,score,
        #                                    min_PAE,Q,ifA_str,ifB_str,
        #                                    interacting_pairs,ppi)
        #print(plddtA,plddtB)
        out_str = (f"{name},{iptm:.1f},{min_PAE:.1f},{Q:.3f},"
                   f"{ifA_str},{ifB_str},{plddtA:.1f},{plddtB:.1f},"
                   f"{interacting_pairs},{ppi}"
        )

        print(out_str)
    else:
        score = data["plddt"]
        print(modelname,"PLDDT:%2d"%np.mean(score))

    i5   = 1
    l5   = 50
    n5   = ndim//(i5*l5)
    while n5>7:
        if i5>=2:
          i5 += 2
        else:
          i5 += 1
        n5 = ndim//(i5*l5)
        #print(n5,i5,i5*l5)
    n5   = ndim//(i5*l5)
    ticks= [n*(i5*l5) for n in range(n5+1)]
    ticklabels = [human_format(n) for n in ticks]
    
    ## sns heatmap ##
    if showfigure=="Y":
        fig,ax = plt.subplots(figsize=(4,3.1))
        darkgreen  = "#004b00"
        lightgreen = "#fafffa"
        colors = [darkgreen,lightgreen]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("",colors)
        #sns.heatmap(PAE,cmap="Greens_r",vmin=0,vmax=maxp,
        #            ax=ax)
        #plt.imshow(PAE,cmap="Greens_r",vmin=0,vmax=maxp)
        cmap = "bwr"
        sns.heatmap(PAE,cmap=cmap,vmin=0,vmax=maxp,
                    ax=ax,cbar=True,
                    cbar_kws={"shrink":0.85,"label":"Expected position error (Å)"})
        lc = "k"
        #ax.axhline(y=0, color=lc,linewidth=1)
        #ax.axhline(y=ndim, color=lc,linewidth=1)
        #ax.axvline(x=0, color=lc,linewidth=1)
        #ax.axvline(x=ndim, color=lc,linewidth=1)
        if model=="multimer":
            for i in range(len(prolen)-1):
                ax.axvline(x=prolen[i]-.5, color=lc,linewidth=1.5,linestyle="--")
                ax.axhline(y=prolen[i]-.5, color=lc,linewidth=1.5,linestyle="--")
        ax.set_ylabel("Aligned residue") 
        ax.set_xlabel("Scored residue") 
        ax.tick_params(axis='both', which='both', length=1.5)
        plt.xticks(ticks,ticklabels,rotation=0)
        plt.yticks(ticks,ticklabels)
        #plt.axis('on')
        fig.tight_layout()
        #plt.savefig("%s_pae.pdf"%modelname)
        plt.savefig("%s_pae.png"%modelname,dpi=300)
        plt.show()

    return out_str

def main():
    parser = argparse.ArgumentParser(description='Plot AlphaFold PAE')
    parser.add_argument('-m','--model',
                        help='Name prefix of the model to plot',
                        required=True)
    parser.add_argument('-s','--show',
                        help='Whether to show and save figure',
                        default="Y",choices=["Y", "N"])
    parser.add_argument('-d','--dcut',
                        help='distance cutoff to determine interactions',
                        default=7)
    parser.add_argument('-p','--pcut',
                        help='PAE cutoff to determine possible interface',
                        default=15) ## random disorder regions may ocassionally
                                    ## be placed in nearby regions in AF2
                                    ## predictions
    args = parser.parse_args()
    plot(args.model,args.show,float(args.pcut),float(args.dcut))

if __name__ == "__main__":
    main()
