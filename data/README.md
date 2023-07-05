# Prepare Data

RecSys Challenge dataset provided by sharechat is organized as follows.

```
├── README.txt
├── test
│   └── 000000000000.csv
└── train
    ├── 000000000000.csv
    ├── ...
    └── 000000000029.csv
```

For ease of use, we aggregated all data into one parquet file.

```
python concat_data.py
```
