# DmelPPI

*Required inputs:  
1. `fbpp_out_pred.tsv`  
    Output of AlphaFold-Disorder  
    These columns are used  
    `name` - the FlyBase identifiers of proteins  
    `pos`  - residue index  
    `lddt` - Monomer pLDDT by AlphaFold2 
    `ss`   - Secondary structure predicted by DSSP 
    `disorder-25` - Intrinsic structural disorder (ISD) predicted by AlphaFold-Disorder

*Scripts:  

1. extract different types of IDRs  
    `extract_idr_byCoil.py`  
    - input: `fbpp_out_pred.tsv`  
    - output: `fbpp_longidrs_byCoil.csv`  
        - coiled IDR regions (ISD>0.5, no defined secondary structures) in monomer proteins 
    
    `extract_idr_byPred.py`  
    - input: `fbpp_out_pred.tsv`  
    - output: `fbpp_longidrs_byPred.csv`  
        - conditionally folded IDR regions (ISD>0.5, pLDDT>0.7, defined secondary structures) in monomer proteins  
    
    `extract_idr_byPred_notCF.py`  
    - input: `fbpp_out_pred.tsv`  
    - output: `fbpp_longidrs_byPred_noCF.csv`  
        - non coil and non conditionally folded regions (ISD>0.5, pLDDT<0.7, defined secondary structures) in monomer proteins  

2. Extract interfaces

3. Statistics of interfaces

