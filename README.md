# EnsembleSet

[![PyPI release](https://github.com/gperdrizet/ensembleset/actions/workflows/publish_pypi.yml/badge.svg)](https://github.com/gperdrizet/ensembleset/actions/workflows/publish_pypi.yml) [![Python CI](https://github.com/gperdrizet/ensembleset/actions/workflows/python_ci.yml/badge.svg)](https://github.com/gperdrizet/ensembleset/actions/workflows/python_ci.yml)

EnsembleSet generates dataset ensembles by applying a randomized sequence of feature engineering methods to a randomized subset of input features.

## 1. Installation

Install the pre-release alpha from PyPI with:

```bash
pip install ensembleset
```

## 2. Usage

See the [example usage notebook](https://github.com/gperdrizet/ensembleset/blob/main/examples/regression_calorie_burn.ipynb).

Initialize an EnsembleSet class instance, passing in the label name and training DataFrame. Optionally, include a test DataFrame and/or list of any string features and the path where you want EnsembleSet to put data. Then call the `make_datasets()` to generate an EnsembleSet, specifying:

1. The number of individual datasets to generate.
2. The fraction of features to randomly select for each feature engineering step.
3. The number of feature engineering steps to run.

```python
import ensembleset.dataset as ds

data_ensemble=ds.DataSet(
    label='label_column_name',                       # Required
    train_data=train_df,                             # Required
    test_data=test_df,                               # Optional, defaults to None
    string_features=['string_feature_column_names'], # Optional, defaults to None
    data_directory='path/to/ensembleset/data'        # Optional, defaults to ./data
)

data_ensemble.make_datasets(
    n_datasets=10,         # Required
    fraction_features=0.1, # Required
    n_steps=5              # Required
)
```

The above call to `make_datasets()` will generate 10 different datasets using a random sequence of 5 feature engineering techniques applied to a randomly selected 10% of features. The feature selection is re-calculated after each feature engineering step. Each feature engineering step is applied to the test set if one is provided with a minimum of data leakage (e.g. gaussian KDE is calculated from training data only and then applied to training and testing data).

By default, generated datasets will be saved to HDF5 in `data/dataset.h5` using the following structure:

```text
dataset.h5
├──train
│   ├── labels
|   ├── 1
|   ├── .
|   ├── .
|   ├── .
|   └── n
│
└──test
    ├── labels
    ├── 1
    ├── .
    ├── .
    ├── .
    └── n
```

## 3. Feature engineering

The currently implemented pool of feature engineering methods are:

1. **One-hot encoding** for string features
2. **Ordinal encoding** for string features
3. **Log features** with bases 2, e or 10
4. **Ratio features**
5. **Exponential features** with base 2 or e
6. **Sum features** with 2, 3, or 4
7. **Difference features** with 2, 3 or 4 subtrahends
8. **Polynomial features** with degree 2 or 3
9. **Spline features** with degree 2, 3 or 4
10. **Quantized features** with using randomly selected k-bins
11. **Smoothed features** with gaussian kernel density estimation

Major feature engineering parameters are also randomly selected for each step.

