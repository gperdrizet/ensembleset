Installation
============

Requirements
------------

EnsembleSet requires Python 3.10 or higher and has the following dependencies:

* h5py - HDF5 file I/O
* numpy - Numerical operations
* pandas - DataFrame handling
* scikit-learn - ML preprocessing and transformations

Install from PyPI
-----------------

Install the latest release from PyPI using pip:

.. code-block:: bash

   pip install ensembleset

Install from Source
-------------------

To install from source, clone the repository and install using pip:

.. code-block:: bash

   git clone https://github.com/gperdrizet/ensembleset.git
   cd ensembleset
   pip install .

Development Installation
------------------------

For development, install with the optional development dependencies:

.. code-block:: bash

   pip install ensembleset[dev]

This includes:

* ipykernel - Jupyter notebook support
* matplotlib - Visualization

Documentation Dependencies
--------------------------

To build the documentation locally, install the documentation dependencies:

.. code-block:: bash

   pip install ensembleset[docs]

This includes:

* sphinx
* sphinx-rtd-theme
* nbsphinx
* pandoc

Verifying Installation
----------------------

You can verify your installation by importing the package:

.. code-block:: python

   import ensembleset
   print(ensembleset.__version__)
