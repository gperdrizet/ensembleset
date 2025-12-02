Quick Start Guide
=================

Basic Usage
-----------

Here's a simple example to get started with EnsembleSet:

.. code-block:: python

   import pandas as pd
   import ensembleset.dataset as ds

   # Create or load your training data
   train_df = pd.DataFrame({
       'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
       'feature2': [10.0, 20.0, 30.0, 40.0, 50.0],
       'feature3': [100.0, 200.0, 300.0, 400.0, 500.0],
       'label': [0, 1, 0, 1, 0]
   })

   # Initialize the DataSet
   data_ensemble = ds.DataSet(
       label='label',
       train_data=train_df,
       data_directory='./ensemble_output'
   )

   # Generate ensemble datasets
   output_file = data_ensemble.make_datasets(
       n_datasets=10,         # Generate 10 different datasets
       frac_features=0.1,     # Use 10% of features per step
       n_steps=5              # Apply 5 engineering steps per dataset
   )

   print(f"Datasets saved to: {output_file}")

With Test Data
--------------

Include test data to generate aligned training and testing datasets:

.. code-block:: python

   import ensembleset.dataset as ds

   # Initialize with both training and testing data
   data_ensemble = ds.DataSet(
       label='label_column_name',
       train_data=train_df,
       test_data=test_df,
       data_directory='./ensemble_output'
   )

   # Generate ensemble datasets
   output_file = data_ensemble.make_datasets(
       n_datasets=10,
       frac_features=0.1,
       n_steps=5
   )

The same feature engineering pipeline applied to training data will be applied to testing data, with all transformations fitted on training data only to prevent data leakage.

With String Features
--------------------

Specify categorical string features that need encoding:

.. code-block:: python

   import ensembleset.dataset as ds

   # Initialize with string features specified
   data_ensemble = ds.DataSet(
       label='target',
       train_data=train_df,
       test_data=test_df,
       string_features=['category_col', 'group_col'],
       data_directory='./ensemble_output',
       ensembleset_base_name='my_ensemble'
   )

   # Generate ensemble datasets
   output_file = data_ensemble.make_datasets(
       n_datasets=10,
       frac_features=0.2,
       n_steps=3
   )

String features will be encoded using either one-hot or ordinal encoding before numerical feature engineering methods are applied.

Understanding Parameters
------------------------

``n_datasets``
   Number of dataset variations to generate. Each will have a unique random sequence of feature engineering methods applied.

``frac_features``
   Fraction of features (0.0 to 1.0) to randomly select for each feature engineering step. For example, 0.1 means 10% of available features. The selection is re-randomized for each step.

``n_steps``
   Number of feature engineering steps to apply in sequence for each dataset. Each step randomly selects a method from the available techniques.

Output Format
-------------

Generated datasets are saved to HDF5 format with the following structure:

.. code-block:: text

   ensembleset.h5
   ├── train
   │   ├── labels          # Training labels array
   │   ├── dataset_0       # First training dataset
   │   ├── dataset_1       # Second training dataset
   │   └── ...
   └── test
       ├── labels          # Testing labels array
       ├── dataset_0       # First testing dataset
       ├── dataset_1       # Second testing dataset
       └── ...

Reading Generated Datasets
---------------------------

Load datasets from the HDF5 file using h5py:

.. code-block:: python

   import h5py
   import numpy as np

   with h5py.File('ensemble_output/ensembleset.h5', 'r') as f:
       # Load training data for first dataset
       train_labels = np.array(f['train/labels'])
       train_features = np.array(f['train/dataset_0'])
       
       # Load testing data for first dataset
       test_labels = np.array(f['test/labels'])
       test_features = np.array(f['test/dataset_0'])

Next Steps
----------

* See :doc:`api` for detailed API documentation
* See :doc:`feature_catalog` for descriptions of all feature engineering methods
* See :doc:`configuration` for configuration options
