pip install git+https://github.com/facebookresearch/esm.git
python $CONDA_PREFIX/lib/python3.10/site-packages/esm/scripts/extract.py esm1_t6_43M_UR50S sequences.fasta embeddings --include mean
