"""Data preprocessing utilities for feature engineering pipelines.

This module provides preprocessing functions used to clean and prepare data
before and after feature engineering operations. Functions include handling
missing values, removing constant features, scaling, type conversions, and
managing extreme values.

See Also
--------
ensembleset.feature_methods : Feature engineering operations
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.width', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

logging.captureWarnings(True)


def preprocess_features(
        features: list,
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        preprocessing_steps:list
) -> Tuple[list, pd.DataFrame, pd.DataFrame]:

    '''Runs feature preprocessing steps.'''

    logger = logging.getLogger(__name__ + '.preprocess_features')
    logger.addHandler(logging.NullHandler())

    train_working_df=train_df.copy()

    if test_df is not None:
        test_working_df = test_df.copy()

    else:
        test_working_df = None

    for preprocessing_step in preprocessing_steps:
        preprocessing_func = globals().get(preprocessing_step)

        if features is not None and len(features) >= 1:

            logger.debug('Preprocessor running %s', preprocessing_step)

            features, train_working_df, test_working_df = preprocessing_func(
                features,
                train_working_df,
                test_working_df
            )

        else:

            logger.debug('Preprocessor step %s received no features', preprocessing_step)


    return features, train_working_df, test_working_df


def exclude_string_features(
        features: list,
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
) -> Tuple[list, pd.DataFrame, pd.DataFrame]:

    '''Removes string features from features list.'''

    if features is not None and len(features) >= 1:

        for feature in features:
            if test_df is not None:
                if (is_numeric_dtype(train_df[feature]) is False or
                    is_numeric_dtype(test_df[feature]) is False):

                    train_df.drop(feature, axis=1, inplace=True, errors='ignore')
                    test_df.drop(feature, axis=1, inplace=True, errors='ignore')
                    features.remove(feature)

            elif test_df is None:
                if is_numeric_dtype(train_df[feature]) is False:
                    train_df.drop(feature, axis=1, inplace=True, errors='ignore')
                    features.remove(feature)

    return features, train_df, test_df


def enforce_floats(
        features: list,
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
) -> Tuple[list, pd.DataFrame, pd.DataFrame]:

    '''Changes features to float dtype.'''

    if features is not None and len(features) >= 1:

        train_df[features]=train_df[features].astype(float).copy()

        if test_df is not None:
            test_df[features]=test_df[features].astype(float).copy()

    return features, train_df, test_df


def remove_inf(
        features: list,
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
) -> Tuple[list, pd.DataFrame, pd.DataFrame]:

    '''Replaces any np.inf values with np.NAN.'''

    if features is not None and len(features) >= 1:

        # Get rid of np.inf
        train_df[features]=train_df[features].replace(
            [np.inf, -np.inf],
            np.nan
        )

        if test_df is not None:
            test_df[features]=test_df[features].replace(
                [np.inf, -np.inf],
                np.nan
            )

    return features, train_df, test_df


def remove_large_nums(
        features: list,
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
) -> Tuple[list, pd.DataFrame, pd.DataFrame]:

    '''Replaces numbers larger than the cube root of the float64 limit with np.nan.'''

    if features is not None and len(features) >= 1:

        # Get rid of large values
        train_df[features] = train_df[features].mask(
            abs(train_df[features]) > 1.0*10**102
        )

        if test_df is not None:
            test_df[features] = test_df[features].mask(
                test_df[features] > 1.0*10**102
            )

    return features, train_df, test_df


def remove_small_nums(
        features: list,
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
) -> Tuple[list, pd.DataFrame, pd.DataFrame]:

    '''Replaces values smaller than the float64 limit with zero.'''

    if features is not None and len(features) >= 1:

        # Get rid of small values
        train_df[features] = train_df[features].mask(
            abs(train_df[features]) < 1.0-102
        ).fillna(0.0)

        if test_df is not None:
            test_df[features] = test_df[features].mask(
                abs(test_df[features]) < 1.0-102
            ).fillna(0.0)

    return features, train_df, test_df


def knn_impute(
        features: list,
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
) -> Tuple[list, pd.DataFrame, pd.DataFrame]:

    '''Uses SciKit-lean's KNN imputer to fill np.nan.'''

    if features is not None and len(features) >= 1:

        imputer=KNNImputer()
        train_df[features] = imputer.fit_transform(train_df[features])

        if test_df is not None:
            test_df[features] = imputer.transform(test_df[features])

    return features, train_df, test_df


def remove_constants(
        features: list,
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
) -> Tuple[list, pd.DataFrame, pd.DataFrame]:

    '''Removes constant valued features.'''

    constant_features=train_df.loc[:,train_df.nunique(dropna=False) == 1]
    train_df.drop(constant_features, axis=1, inplace=True)

    if test_df is not None:
        test_df.drop(constant_features, axis=1, inplace=True)

    new_features = list(set(features) & set(train_df.columns.to_list()))

    if len(new_features) == 0:
        new_features = None
        train_df = None
        test_df = None

    return new_features, train_df, test_df


def scale_to_range(
        features: list,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        min_val: float = 0.0,
        max_val: float = 1.0
) -> Tuple[list, pd.DataFrame, pd.DataFrame]:

    '''Scales features into range'''

    if features is not None:

        for feature in features:

            scaler=MinMaxScaler(feature_range=(min_val, max_val))
            train_df[feature]=scaler.fit_transform(train_df[feature].to_frame())

            if test_df is not None:
                test_df[feature]=scaler.transform(test_df[feature].to_frame())

    return features, train_df, test_df


def add_new_features(
    new_train_features,
    new_test_features,
    train_df:pd.DataFrame,
    test_df:pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Adds new features to dataframes'''

    if isinstance(new_test_features, dict):
        new_train_features=pd.DataFrame.from_dict(new_train_features)

        if new_test_features is not None:
            new_test_features=pd.DataFrame.from_dict(new_test_features)

    train_df=pd.concat(
        [train_df.reset_index(drop=True), new_train_features.reset_index(drop=True)],
        axis=1
    )

    train_df = train_df.loc[:,~train_df.columns.duplicated()].copy()
    train_df.sort_index(axis=1, inplace=True)
    train_df.reset_index(inplace=True, drop=True)

    if test_df is not None:
        test_df=pd.concat(
            [test_df.reset_index(drop=True), new_test_features.reset_index(drop=True)],
            axis=1
        )

        test_df = test_df.loc[:,~test_df.columns.duplicated()].copy()
        test_df.sort_index(axis=1, inplace=True)
        test_df.reset_index(inplace=True, drop=True)

    return train_df, test_df
