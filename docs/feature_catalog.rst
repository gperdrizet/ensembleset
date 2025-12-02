Feature Engineering Catalog
============================

This page provides detailed descriptions of all 11 feature engineering methods available in EnsembleSet.

String Feature Encoding
------------------------

1. One-Hot Encoding
^^^^^^^^^^^^^^^^^^^

Converts categorical string features into binary indicator columns.

**Mathematical Description:**

For a categorical feature with :math:`k` unique categories, one-hot encoding creates :math:`k` binary columns where each column represents one category. For a given sample, exactly one column has value 1 (the category present) and all others are 0.

**Use Cases:**

* Nominal categorical features without inherent ordering
* Features with low to moderate cardinality
* When treating each category as independent is appropriate

**Example:**

.. code-block:: python

   # Input: ['A', 'B', 'A', 'C']
   # Output: 
   #   A  B  C
   #   1  0  0
   #   0  1  0
   #   1  0  0
   #   0  0  1

2. Ordinal Encoding
^^^^^^^^^^^^^^^^^^^^

Converts categorical string features into integer codes.

**Mathematical Description:**

Each unique category is mapped to an integer. For :math:`k` unique categories, integers from 0 to :math:`k-1` are assigned.

**Use Cases:**

* Ordinal categorical features with inherent ordering
* High-cardinality categorical features where one-hot encoding would create too many columns
* Tree-based models that can handle encoded categories

**Example:**

.. code-block:: python

   # Input: ['low', 'medium', 'high', 'low']
   # Output: [0, 1, 2, 0]

Numerical Feature Engineering
------------------------------

3. Polynomial Features
^^^^^^^^^^^^^^^^^^^^^^

Generates polynomial and interaction features from existing features.

**Mathematical Description:**

For degree :math:`d`, polynomial features include all monomials of degree :math:`\leq d`. For two features :math:`x_1` and :math:`x_2` with degree 2:

.. math::

   [1, x_1, x_2, x_1^2, x_1 x_2, x_2^2]

**Use Cases:**

* Capturing non-linear relationships
* Modeling feature interactions
* Polynomial regression models

**Parameters:**

* Degree: 2 or 3
* Interaction only: Include only cross-products
* Include bias: Add constant term

4. Spline Features
^^^^^^^^^^^^^^^^^^

Applies spline basis transformations to features.

**Mathematical Description:**

Spline transformations create piecewise polynomial functions. B-splines of degree :math:`d` with :math:`k` knots create smooth curves defined by control points.

**Use Cases:**

* Flexible non-linear transformations
* Smoother than polynomial features
* Capturing complex non-linear patterns

**Parameters:**

* Degree: 2, 3, or 4
* Knots: Number and placement (uniform or quantile)
* Extrapolation: Behavior outside knot range

5. Logarithmic Features
^^^^^^^^^^^^^^^^^^^^^^^

Applies logarithmic transformations to compress large value ranges.

**Mathematical Description:**

.. math::

   y = \log_b(x)

where :math:`b \in \{2, e, 10\}`

**Use Cases:**

* Features with exponential distributions or heavy right tails
* Reducing the impact of outliers
* Making multiplicative relationships additive

**Parameters:**

* Base: 2, e (natural log), or 10

**Note:** Handles zero and negative values by preprocessing.

6. Ratio Features
^^^^^^^^^^^^^^^^^

Creates ratio features from all pairwise divisions of selected features.

**Mathematical Description:**

For features :math:`x_1, x_2, ..., x_n`, creates:

.. math::

   r_{ij} = \frac{x_i}{x_j} \quad \forall i \neq j

**Use Cases:**

* Capturing relative relationships between features
* Normalizing features by reference values
* Financial ratios (e.g., price/earnings)

**Parameters:**

* Division by zero value: Replacement value (default: NaN)

7. Exponential Features
^^^^^^^^^^^^^^^^^^^^^^^^

Applies exponential transformations to features.

**Mathematical Description:**

.. math::

   y = b^x

where :math:`b \in \{2, e\}`

**Use Cases:**

* Inverse of logarithmic transformation
* Amplifying small differences
* Modeling exponential growth

**Parameters:**

* Base: 2 or e (natural exponential)

**Note:** Handles overflow by preprocessing.

8. Sum Features
^^^^^^^^^^^^^^^

Creates features by summing combinations of selected features.

**Mathematical Description:**

For :math:`n` addends, creates sums of all combinations:

.. math::

   s = x_{i_1} + x_{i_2} + ... + x_{i_n}

where :math:`n \in \{2, 3, 4\}`

**Use Cases:**

* Capturing aggregate effects
* Total or cumulative values
* Additive relationships

**Parameters:**

* Number of addends: 2, 3, or 4

9. Difference Features
^^^^^^^^^^^^^^^^^^^^^^

Creates features by computing differences of feature combinations.

**Mathematical Description:**

For :math:`n` subtrahends, creates:

.. math::

   d = x_{i_1} - x_{i_2} - ... - x_{i_n}

where :math:`n \in \{2, 3, 4\}`

**Use Cases:**

* Change or delta features
* Comparing related measurements
* Removing baseline effects

**Parameters:**

* Number of subtrahends: 2, 3, or 4

10. Gaussian KDE Smoothing
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Applies Gaussian kernel density estimation to smooth features.

**Mathematical Description:**

For each feature value :math:`x`, estimates the probability density:

.. math::

   \hat{f}(x) = \frac{1}{nh} \sum_{i=1}^{n} K\left(\frac{x - x_i}{h}\right)

where :math:`K` is the Gaussian kernel and :math:`h` is the bandwidth.

**Use Cases:**

* Noise reduction
* Identifying underlying distributions
* Smoothing irregular patterns

**Parameters:**

* Bandwidth: 'scott' or 'silverman' method
* Sample size: Number of samples for KDE calculation

**Note:** Fitted on training data only, then applied to both train and test.

11. K-Bins Quantization
^^^^^^^^^^^^^^^^^^^^^^^^

Discretizes continuous features into bins.

**Mathematical Description:**

Divides the feature range into :math:`k` bins and assigns each value to a bin:

.. math::

   y = \text{bin}(x) \in \{0, 1, ..., k-1\}

**Use Cases:**

* Converting continuous to categorical features
* Reducing sensitivity to small variations
* Handling non-linear relationships with linear models

**Parameters:**

* Number of bins: 4, 8, or 16
* Strategy: uniform, quantile, or k-means
* Encoding: ordinal

Feature Engineering Pipeline
-----------------------------

During ensemble generation, these methods are:

1. **Randomly selected** - Each dataset gets a unique sequence
2. **Applied in sequence** - Methods build on previous transformations
3. **Applied to random subsets** - Only a fraction of features are transformed at each step
4. **Fitted on training data** - All transformations use training data statistics to prevent leakage
5. **Applied to test data** - The same fitted transformations are applied to test data

This randomization strategy creates diverse datasets suitable for training ensemble models while maintaining consistent transformations between training and testing data.
