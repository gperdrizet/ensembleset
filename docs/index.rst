EnsembleSet Documentation
=========================

.. image:: https://github.com/gperdrizet/ensembleset/actions/workflows/publish-to-pypi.yml/badge.svg
   :target: https://github.com/gperdrizet/ensembleset/actions/workflows/publish-to-pypi.yml
   :alt: Publish to PyPI

.. image:: https://github.com/gperdrizet/ensembleset/actions/workflows/publish-to-testpypi.yml/badge.svg
   :target: https://github.com/gperdrizet/ensembleset/actions/workflows/publish-to-testpypi.yml
   :alt: Publish to TestPyPI

.. image:: https://github.com/gperdrizet/ensembleset/actions/workflows/pr-validation.yml/badge.svg
   :target: https://github.com/gperdrizet/ensembleset/actions/workflows/pr-validation.yml
   :alt: PR Validation

.. image:: https://github.com/gperdrizet/ensembleset/actions/workflows/pages/pages-build-deployment/badge.svg
   :target: https://github.com/gperdrizet/ensembleset/actions/workflows/pages/pages-build-deployment
   :alt: pages-build-deployment

.. image:: https://img.shields.io/badge/docs-GitHub%20Pages-blue
   :target: https://gperdrizet.github.io/ensembleset/
   :alt: Documentation

EnsembleSet generates dataset ensembles by applying a randomized sequence of feature engineering methods to a randomized subset of input features.

**Version:** |version|

Overview
--------

EnsembleSet is a Python package designed for generating ensemble datasets through randomized feature engineering. It's particularly useful for training ensemble machine learning models on tabular data prediction and modeling projects.

Key features:

* Generates multiple dataset variations from a single input dataset
* Applies 11 different feature engineering techniques in random sequences
* Supports both training and testing datasets with minimal data leakage
* Outputs ensembles to HDF5 format for efficient storage
* Uses multiprocessing for parallel dataset generation

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   configuration
   feature_catalog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
