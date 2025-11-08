# DmelPPI

## Required inputs  
1. `fbpp_out_pred.tsv`  
    Output of AlphaFold-Disorder  
    These columns are used  
    `name` - the FlyBase identifiers of proteins  
    `pos`  - residue index  
    `lddt` - Monomer pLDDT by AlphaFold2 
    `ss`   - Secondary structure predicted by DSSP 
    `disorder-25` - Intrinsic structural disorder (ISD) predicted by AlphaFold-Disorder  

2. Predicted models (pdb format). Available at figshare  

3. Predicted metrics (pkl format). Available at figshare

## Scripts  
1. extract different types of IDRs  
    `extract_idr_byCoil.py`  
    - input: `fbpp_out_pred.tsv`  
    - output: `fbpp_longidrs_byCoil.csv`  
        - The output contains coiled IDR regions (ISD>0.5, pLDDT<0.7, no defined secondary structures) in monomer proteins 
    
    `extract_idr_byPred.py`  
    - input: `fbpp_out_pred.tsv`  
    - output: `fbpp_longidrs_byPred.csv`  
        - The output contains conditionally folded IDR regions (ISD>0.5, pLDDT>0.7, defined secondary structures) in monomer proteins  
    
    `extract_idr_byPred_notCF.py`  
    - input: `fbpp_out_pred.tsv`  
    - output: `fbpp_longidrs_byPred_noCF.csv`  
        - The output contains non coil and non conditionally folded regions (ISD>0.5, pLDDT<0.7, defined secondary structures) in monomer proteins  

2. Extract interfaces  
    `plot_pae.py`
    - A standalone script that can be used to calculate pDockQ, extract interface, plot PAE matrix

    ```
    usage: plot_pae.py [-h] -m MODEL [-s {Y,N}] [-d DCUT] [-p PCUT]

    Plot AlphaFold PAE

    options:
        -h, --help            show this help message and exit
        -m MODEL, --model MODEL
                            Name prefix of the model to plot
        -s {Y,N}, --show {Y,N}
                        Whether to show and save figure
        -d DCUT, --dcut DCUT  distance cutoff to determine interactions
        -p PCUT, --pcut PCUT  PAE cutoff to determine possible interface
    ```


3. Statistics of interfaces

