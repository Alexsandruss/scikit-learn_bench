# Data handling in benchmarks

Data handling steps:
1. Load data:
   - If not cached and not available at specified `DATASETSROOT`: download/generate dataset and put it in raw and/or usual cache
   - If cached: load from cached files
   - If `DATASETSROOT` path to indicated file exists: load from specified CSV
2. Split data into subsets if requested
3. Convert to requested form (data type, format, order, etc.)

There are two level of cache with corresponding directories: `raw cache` for files downloaded from external sources, and just `cache` for files applicable for fast-loading in benchmarks.

Each dataset has few associated files in usual `cache`: data component files (`x`, `y`, `weights`, etc.) and JSON file with dataset properties (number of classes, clusters, default split arguments).
For example:
```
data_cache/
...
├── mnist.json
├── mnist_x.parq
├── mnist_y.npz
...
```

Cached file formats:
| Format | File extension | Associated Python types |
| --- | --- | --- |
| [Parquet](https://parquet.apache.org) | `.parq` | pandas.DataFrame |
| Numpy uncompressed binary dense data | `.npz` | numpy.ndarray, pandas.Series |
| Numpy uncompressed binary CSR data | `.csr.npz` | scipy.sparse.csr_matrix |

Using `DATASETSROOT` to load data:
1. Set up config file `data`:`dataset` and `data`:`dataset_kwargs`:`train_file` (and `test_file` if applicable) to appropriate dataset name and csv file name(s).
2. Set `DATASETSROOT` environment variable such that the path to the indicated CSV files matches `$DATASETSROOT`/workloads/`data`:`dataset`/dataset/`data`:`dataset_kwargs`:`train_file`.
3. The loading mechanisms will load the csv via `pandas`.

Existing data sources:
 - Synthetic data from sklearn
 - OpenML datasets
 - Custom loaders for named datasets
 - Pre-downloaded datasets located in specified `DATASETSROOT`
