# DmelPPI: A predicted structural interactome reveals binding interference from intrinsically disordered regions  

## Required inputs  
1. `fbpp_out_pred.tsv` - the output of AlphaFold-Disorder  
    - These columns are used  
        `name` - the FlyBase identifiers of proteins  
        `pos`  - residue index  
        `lddt` - Monomer pLDDT by AlphaFold2  
        `ss`   - Secondary structure predicted by DSSP  
        `disorder-25` - Intrinsic structural disorder (ISD) predicted by AlphaFold-Disorder  

2. `fbpp_ss.csv, complex_ss.csv` - The secondary structure of monomers (from AF database) and predicted complex models  

3. `fbpp2fbgnlist.txt,fbpp2symbol.txt,fbpp2uniprot.csv` - Name mapping between FBpp, FBgn, Gene name, and UniProt accession numbers   

4. Predicted models (pdb format). Available at figshare  

5. Predicted metrics (pkl format). Available at figshare

## Scripts  
### 1. extract different types of IDRs from monomers  
- `extract_idr_byCoil.py`  
    - input: `fbpp_out_pred.tsv`  
    - output: `fbpp_longidrs_byCoil.csv`  
        - The output contains coiled IDR regions (ISD>0.5, pLDDT<0.7, no defined secondary structures) in monomer proteins  

- `extract_idr_byPred.py`  
    - input: `fbpp_out_pred.tsv`  
    - output: `fbpp_longidrs_byPred.csv`  
        - The output contains conditionally folded IDR regions (ISD>0.5, pLDDT>0.7, defined secondary structures) in monomer proteins  

- `extract_idr_byPred_notCF.py`  
    - input: `fbpp_out_pred.tsv`  
    - output: `fbpp_longidrs_byPred_noCF.csv`  
        - The output contains non coil and non conditionally folded regions (ISD>0.5, pLDDT<0.7, defined secondary structures) in monomer proteins  

The scripts also generate `fbpp_ss.csv`, which is the secondary structure of monomers.  

### 2. Extract interfaces from predicted model   
- `plot_pae.py` - A standalone script that can be used to calculate pDockQ, extract interface, plot PAE matrix

```
usage: plot_pae.py [-h] -m MODEL [-s {Y,N}] [-d DCUT] [-p PCUT]

Plot AlphaFold PAE

options:
    -h, --help            show this help message and exit
    -m MODEL, --model MODEL Name prefix of the model to plot
    -s {Y,N}, --show {Y,N}  Whether to show and save figure
    -d DCUT, --dcut DCUT  distance cutoff to determine interactions
    -p PCUT, --pcut PCUT  PAE cutoff to determine possible interface
```  

- `extract_pae.py`  - calls `plot_pae.py` in parallel  
    - input: A list of model names  
    - output: `pae_summary.csv`  
        - The output contains following columns: `pair,iptm,pae,pdockq,interfaceA,interfaceB,plddtA,plddtB,interface_pairs,ppi`  

- `best_models.py` - extract the best predicted models from five random models by AF2 multimer

### 3. Statistics of interfaces  
<p align="center">
<img src="interface_illustration.png" alt="Interface illustration" width="88%"> </p>
  <sub><b>Figure 1.</b> Scheme of interface analysis. We check interface pairs by scanning clustered interfaces. 
  In this example, there are three interfaces (1, 2, and 3). For each interface, we examine whether an IDR 
  is involved, and if so, whether it represents coil binding, coil-to-order transition binding, or CF binding.</sub>
</p>

- `pairs_orderWithorder.py`
    - inputs  
        - blastp output  
        - Different types of IDR regions: `fbpp_longidrs_byCoil.csv, fbpp_longidrs_byPred.csv, fbpp_longidrs_byPred_noCF.csv`  
        - `pae_summary.csv` - the output from `extract_pae.py`  
        - Secondary structures in predicted hetero-dimer models  
        - Secondary structures in predicted monomer models  
    - output: `pairs_highconf_with_idrs.csv`  
        - The output contains following columns for each interface (e.g. interface 1, 2, or 3 in the figure above), `p1,p2,length1,length2,pdockq,idr_binding1,idr_binding2,coil_binding1,coil_binding2,CF_binding1,CF_binding2,coil2order_binding1,coil2order_binding2,order_binding1,order_binding2,interface1,interface2,idr_if1,idr_if2,coil_if1,coil_if2,CF_if1,CF_if2,coil2order_if1,coil2order_if2,order_if1,order_if2,monomer_ss_coil_if1,monomer_ss_coil_if2,complex_ss_coil_if1,complex_ss_coil_if2,pvalue_coil_if1,pvalue_coil_if2,idr_region1,idr_region2,order_region1,order_region2,idr_type1,idr_type2,interaction_type`  
        - some explainations of the columns:  
            - `idr_binding1,idr_binding2` - possible values: True or False. True if the interaction involves disordered binding else False  
            - `coil_bindg1,coil_binding2` - possible values: True or False. True if the interaction involves coil binding else False  
            - `CF_bindg1,CF_binding2` - possible values: True or False. True if the interaction involves conditionally folded binding else False  
            - `coil2order_bindg1,coil2order_binding2` - possible values: True or False. True if the interaction involves coil to order transition binding else False  
            - `order_bindg1,order_binding2` - possible values: True or False. True if the interaction involves ordered regions else False  
            - `idr_if1,idr_if2,coil_if1,coil_if2,CF_if1,CF_if2,coil2order_if1,coil2order_if2,order_if1,order_if2` - the corresponding IDR or ordered interfaces for the above interfaces. The value is empty if the interface does not exist  
            - `monomer_ss_coil_if1,monomer_ss_coil_if2` - The secondary structure of coil interfaces in monomer states  
            - `complex_ss_coil_if1,complex_ss_coil_if2` - The secondary structure of coil interfaces in the hetero-dimer complex states  
            - `pvalue_coil_if1,pvalue_coil_if2` - The p-value whether secondary structure contents increase significantly from monomer state to complex state. This is to check whether there is coil-to-order transition binding  
            - `idr_region1,idr_region2,order_region1,order_region2` - The IDR regions and ordered regions in monomer state  
            - `idr_type1,idr_type2` - Short IDR (5-29 residues) or Long IDR (>= 30 residues)  
            - `interaction_type` - Type of IDR binding   

## Reference
Peng, Junhui and Li Zhao. “A predicted structural interactome reveals binding interference from intrinsically disordered regions.” bioRxiv (2025): doi.org/10.1101/2025.08.15.670535.  


