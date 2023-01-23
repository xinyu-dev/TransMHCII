# TransMHCII
Code repository for manuscript: TransMHCII: a novel MHC-II binding prediction model built using a protein language model and an image classifier

# Files
1. **embed.ipynb**: example code for embedding using PLM. Input: a CSV file with sequences and alleles. This is the same as the Supp.Table 1, but formatted a bit differently. Output: an embedded pickle file.
2. **train_ProtT5_efficientnet.py**: example code for constructing and training the ProtT5 EfficinetNet v2b0 model. Input: same CSV file and pickle file from above. Output: tensorflow model.
