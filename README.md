# TransMHCII
Code repository for manuscript: TransMHCII: a novel MHC-II binding prediction model built using a protein language model and an image classifier

# Files
1. **input_with_features.csv**: Same as the Supp.Table 1, but formatted in CSV. 
2. **embed.ipynb**: example embedding using PLM model `prot_t5_xl_uniref50`. Inputs the CSV file above, and outputs a pickle file. Pickle file is not uploaded to this repository due to size. 
3. **train_ProtT5_efficientnet.py**: example code for constructing and training the ProtT5 EfficinetNet v2b0 model. Inputs the CSV file and pickle file from above. A tensorflow model is built in the process. We used a g4dn.12xlarge instance on AWS to train the model in ~ 4 hours. 
