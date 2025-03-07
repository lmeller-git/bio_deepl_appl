# bio_deepl_appl

repository for the course Applications of Deep Learning in Biology

## Installation  

```$ git clone https://github.com/lmeller-git/bio_deepl_appl```

## Usage

load correctly organized data into some directory  
cd into the repo

```$ cd bio_deepl_appl```

add src to you PYTHONPATH  

```$ export PYTHONPATH="$PYTHONPATH:."```

```$ python src/main.py --data <path-to-data-dir>```

The data directory should have this structure:  
```
data
└── project_data
    ├── mega_test.csv
    ├── mega_test_embeddings
    ├── mega_train.csv
    ├── mega_train_embeddings
    ├── mega_val.csv
    └── mega_val_embeddings
```

## References

H. Dieckhaus, M. Brocidiacono, N.Z. Randolph, & B. Kuhlman, Transfer learning to leverage larger datasets for improved prediction of protein stability changes, Proc. Natl. Acad. Sci. U.S.A. 121 (6) e2314853121, https://doi.org/10.1073/pnas.2314853121 (2024). 
