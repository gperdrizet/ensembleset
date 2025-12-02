Configuration
=============

EnsembleSet uses configuration dictionaries to define available feature engineering methods and their parameter options.

String Encoding Methods
------------------------

String features are encoded before numerical feature engineering methods are applied.

.. py:data:: ensembleset.feature_engineerings.STRING_ENCODINGS

   Dictionary of available string encoding methods and their default parameters.

   **Available Methods:**

   * ``onehot_encoding`` - One-hot encoding using sklearn's OneHotEncoder
      
     * ``sparse_output``: False - Return dense arrays

   * ``ordinal_encoding`` - Ordinal encoding using sklearn's OrdinalEncoder
      
     * ``handle_unknown``: 'use_encoded_value' - Handle unknown categories
     * ``unknown_value``: np.nan - Value to use for unknown categories

Numerical Feature Engineering Methods
--------------------------------------

.. py:data:: ensembleset.feature_engineerings.NUMERICAL_METHODS

   Dictionary of available numerical feature engineering methods and their parameter options.

   **Available Methods:**

   * ``poly_features`` - Polynomial feature generation
      
     * ``degree``: [2, 3] - Polynomial degree options
     * ``interaction_only``: [True, False] - Include only interaction features
     * ``include_bias``: [True, False] - Include bias column

   * ``spline_features`` - Spline transformation
      
     * ``n_knots``: [5] - Number of knots
     * ``degree``: [2, 3, 4] - Spline degree options
     * ``knots``: ['uniform', 'quantile'] - Knot placement strategy
     * ``extrapolation``: ['constant', 'linear', 'continue', 'periodic'] - Extrapolation method
     * ``include_bias``: [True, False] - Include bias column

   * ``log_features`` - Logarithmic transformation
      
     * ``base``: ['2', 'e', '10'] - Logarithm base options

   * ``ratio_features`` - Ratio/division features
      
     * ``div_zero_value``: [np.nan] - Value to use for division by zero

   * ``exponential_features`` - Exponential transformation
      
     * ``base``: ['2', 'e'] - Exponential base options

   * ``sum_features`` - Sum of feature combinations
      
     * ``n_addends``: [2, 3, 4] - Number of features to sum

   * ``difference_features`` - Difference of feature combinations
      
     * ``n_subtrahends``: [2, 3, 4] - Number of features in subtraction

   * ``kde_smoothing`` - Gaussian kernel density estimation smoothing
      
     * ``bandwidth``: ['scott', 'silverman'] - Bandwidth selection method
     * ``sample_size``: [1000] - Number of samples for KDE

   * ``kbins_quantization`` - Quantization into bins
      
     * ``n_bins``: [4, 8, 16] - Number of bins options
     * ``encode``: ['ordinal'] - Encoding method
     * ``strategy``: ['uniform', 'quantile', 'kmeans'] - Binning strategy

Parameter Selection
-------------------

During dataset generation, parameters are randomly selected from the available options. For example:

* Polynomial features may use degree 2 or 3
* Spline features may use degree 2, 3, or 4
* Log features may use base 2, e, or 10

This randomization ensures diversity across the generated ensemble datasets.

Customization
-------------

While the configuration dictionaries are predefined, users can modify them by accessing the module attributes if custom parameter ranges are desired. However, this is not typically necessary for standard use cases.

Example
-------

.. code-block:: python

   import ensembleset.feature_engineerings as engineerings
   
   # View available string encoding methods
   print(engineerings.STRING_ENCODINGS)
   
   # View available numerical methods
   print(engineerings.NUMERICAL_METHODS)
   
   # Check polynomial feature options
   print(engineerings.NUMERICAL_METHODS['poly_features'])
