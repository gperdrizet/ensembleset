"""Collection of functions to run feature engineering operations.

This module provides feature engineering functions for generating ensemble
datasets. Functions include polynomial features, splines, logarithmic and
exponential transformations, ratio and arithmetic operations, Gaussian KDE
smoothing, and binning. String features can be encoded using one-hot or
ordinal encoding.

All functions follow a consistent interface accepting training and testing
DataFrames, feature lists, and keyword arguments, returning the transformed
DataFrames with minimal data leakage.

See Also
--------
ensembleset.preprocessing_methods : Data preprocessing utilities
ensembleset.feature_engineerings : Configuration dictionaries for methods
"""

import logging
import multiprocessing as mp
from random import choices
from math import e
from itertools import permutations, combinations
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    SplineTransformer,
    KBinsDiscretizer
)

from ensembleset.preprocessing_methods import preprocess_features, scale_to_range, add_new_features

pd.set_option('display.width', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

logging.captureWarnings(True)


def onehot_encoding(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        features:list,
        kwargs:dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply one-hot encoding to categorical string features.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data containing features to encode.
    test_df : pd.DataFrame or None
        Testing data to apply fitted encoder to.
    features : list of str or None
        Names of string feature columns to encode.
    kwargs : dict
        Keyword arguments passed to sklearn.preprocessing.OneHotEncoder.
    
    Returns
    -------
    train_df : pd.DataFrame
        Training data with one-hot encoded features replacing originals.
    test_df : pd.DataFrame or None
        Testing data with one-hot encoded features, or None if input was None.
    
    See Also
    --------
    ordinal_encoding : Alternative ordinal encoding for string features
    """

    logger = logging.getLogger(__name__ + '.onehot_encoding')
    logger.addHandler(logging.NullHandler())
    logger.debug('One-hot encoding string features')

    if features is not None:

        encoder=OneHotEncoder(**kwargs)

        encoded_data=encoder.fit_transform(train_df[features])
        encoded_df=pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())
        train_df=pd.concat(
            [train_df.reset_index(drop=True), encoded_df.reset_index(drop=True)],
            axis=1
        )
        train_df.drop(features, axis=1, inplace=True)

        if test_df is not None:
            encoded_data=encoder.transform(test_df[features])
            encoded_df=pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())
            test_df=pd.concat(
                [test_df.reset_index(drop=True), encoded_df.reset_index(drop=True)],
                axis=1
            )
            test_df.drop(features, axis=1, inplace=True)

    return train_df, test_df


def ordinal_encoding(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        features:list,
        kwargs:dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply ordinal encoding to categorical string features.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data containing features to encode.
    test_df : pd.DataFrame or None
        Testing data to apply fitted encoder to.
    features : list of str or None
        Names of string feature columns to encode.
    kwargs : dict
        Keyword arguments passed to sklearn.preprocessing.OrdinalEncoder.
    
    Returns
    -------
    train_df : pd.DataFrame
        Training data with ordinal encoded features in place.
    test_df : pd.DataFrame or None
        Testing data with ordinal encoded features, or None if input was None.
    
    See Also
    --------
    onehot_encoding : Alternative one-hot encoding for string features
    """

    logger = logging.getLogger(__name__ + '.ordinal_encoding')
    logger.addHandler(logging.NullHandler())
    logger.debug('Ordinal encoding string features')

    if features is not None:

        encoder=OrdinalEncoder(**kwargs)

        train_df[features]=encoder.fit_transform(train_df[features])

        if test_df is not None:
            test_df[features]=encoder.transform(test_df[features])

    return train_df, test_df


def poly_features(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        features:list,
        kwargs:dict,
        shortcircuit_preprocessing:bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate polynomial features from selected feature combinations.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data containing features to transform.
    test_df : pd.DataFrame or None
        Testing data to apply fitted transformer to.
    features : list of str
        Names of feature columns to generate polynomial features from.
    kwargs : dict
        Keyword arguments passed to sklearn.preprocessing.PolynomialFeatures.
        Common keys: 'degree' (2 or 3).
    shortcircuit_preprocessing : bool, default=False
        If True, skip preprocessing steps.
    
    Returns
    -------
    train_df : pd.DataFrame
        Training data with polynomial features added.
    test_df : pd.DataFrame or None
        Testing data with polynomial features added, or None if input was None.
    
    See Also
    --------
    spline_features : Alternative spline-based feature transformation
    """

    logger = logging.getLogger(__name__ + '.poly_features')
    logger.addHandler(logging.NullHandler())
    logger.debug('Adding polynomial features')

    if shortcircuit_preprocessing is False:
        features, train_working_df, test_working_df=preprocess_features(
            features=features,
            train_df=train_df,
            test_df=test_df,
            preprocessing_steps=[
                'exclude_string_features',
                'enforce_floats',
                'remove_inf', 
                'remove_large_nums',
                'remove_small_nums',
                'knn_impute',
                'remove_constants',
                'scale_to_range'
            ]
        )

    else:
        train_working_df = train_df.copy()

        if test_df is not None:
            test_working_df = test_df.copy()

        else:
            test_working_df = None

    if features is not None and len(features) > 0:
        for feature in features:

            transformer=PolynomialFeatures(**kwargs)

            try:
                transformed_data=transformer.fit_transform(train_working_df[feature].to_frame())
                new_columns=transformer.get_feature_names_out()
                transformed_train_df=pd.DataFrame(transformed_data, columns=new_columns)

                transformed_test_df = None

                if test_df is not None:

                    transformed_data=transformer.transform(test_working_df[feature].to_frame())
                    new_columns=transformer.get_feature_names_out()
                    transformed_test_df=pd.DataFrame(transformed_data, columns=new_columns)

            except ValueError:
                logger.error('ValueError in poly feature transformer')

        train_df, test_df = add_new_features(
            new_train_features = transformed_train_df,
            new_test_features = transformed_test_df,
            train_df = train_df,
            test_df = test_df
        )

    return train_df, test_df


def spline_features(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        features:list,
        kwargs:dict,
        shortcircuit_preprocessing:bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Runs sklearn's polynomial feature transformer.'''

    logger = logging.getLogger(__name__ + '.spline_features')
    logger.addHandler(logging.NullHandler())
    logger.debug('Adding spline features')

    if shortcircuit_preprocessing is False:
        features, train_working_df, test_working_df=preprocess_features(
            features=features,
            train_df=train_df,
            test_df=test_df,
            preprocessing_steps=[
                'exclude_string_features',
                'enforce_floats',
                'remove_inf',
                'remove_large_nums',
                'remove_small_nums',
                'knn_impute',
                'remove_constants',
                'scale_to_range'
            ]
        )

    else:
        train_working_df = train_df.copy()

        if test_df is not None:
            test_working_df = test_df.copy()

        else:
            test_working_df = None

    if features is not None and len(features) > 0:
        for feature in features:

            transformer=SplineTransformer(**kwargs)

            try:
                transformed_data=transformer.fit_transform(train_working_df[feature].to_frame())
                new_columns=transformer.get_feature_names_out()
                transformed_train_df=pd.DataFrame(transformed_data, columns=new_columns)

                transformed_test_df = None

                if test_df is not None:

                    transformed_data=transformer.transform(test_working_df[feature].to_frame())
                    new_columns=transformer.get_feature_names_out()
                    transformed_test_df=pd.DataFrame(transformed_data, columns=new_columns)

                train_df, test_df = add_new_features(
                    new_train_features = transformed_train_df,
                    new_test_features = transformed_test_df,
                    train_df = train_df,
                    test_df = test_df
                )

            except ValueError:
                logger.error('ValueError in spline feature transformer')

    return train_df, test_df


def log_features(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        features:list,
        kwargs:dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Takes log of feature, uses sklearn min-max scaler if needed
    to avoid undefined log errors.'''

    logger = logging.getLogger(__name__ + '.log_features')
    logger.addHandler(logging.NullHandler())
    logger.debug('Adding log features')

    features, train_working_df, test_working_df = preprocess_features(
        features=features,
        train_df=train_df,
        test_df=test_df,
        preprocessing_steps=[
            'exclude_string_features',
            'enforce_floats',
            'remove_inf', 
            'remove_large_nums',
            'remove_small_nums',
            'knn_impute',
            'remove_constants'
        ]
    )

    if features is not None and len(features) > 0:

        logger.debug(
            'Will compute log for %s features', len(features)
        )

        features, train_working_df, test_working_df = scale_to_range(
            features=features,
            train_df=train_working_df,
            test_df=test_working_df,
            min_val=1,
            max_val=10
        )

        if test_working_df is not None:
            logger.debug('Input test features for log:')
            logger.debug('\n%s', test_working_df[features].describe().transpose())

        new_train_feature_names = []
        new_train_features = []
        new_test_feature_names = []
        new_test_features = []

        for i, feature in enumerate(features):

            logger.debug('Taking log of feature %s of %s', i + 1, len(features))

            # try:

            if kwargs['base'] == '2':
                new_train_feature_names.append(f'{feature}_log2')
                new_train_features.append(
                    np.log2(
                        train_working_df[feature],
                        out=np.zeros_like(train_working_df[feature], dtype=np.float64),
                        where=(np.array(train_working_df[feature]) > 0)
                    )
                )

                if test_df is not None:
                    new_test_feature_names.append(f'{feature}_log2')
                    new_test_features.append(
                        np.log2(
                            test_working_df[feature],
                            out=np.zeros_like(test_working_df[feature], dtype=np.float64),
                            where=(np.array(test_working_df[feature]) > 0)
                        )
                    )

            if kwargs['base'] == 'e':
                new_train_feature_names.append(f'{feature}_loge')
                new_train_features.append(
                    np.log(
                        train_working_df[feature],
                        out=np.zeros_like(train_working_df[feature], dtype=np.float64),
                        where=(np.array(train_working_df[feature]) > 0)
                    )
                )

                if test_df is not None:
                    new_test_feature_names.append(f'{feature}_loge')
                    new_test_features.append(
                        np.log(
                            test_working_df[feature],
                            out=np.zeros_like(test_working_df[feature], dtype=np.float64),
                            where=(np.array(test_working_df[feature]) > 0)
                        )
                    )

            if kwargs['base'] == '10':
                new_train_feature_names.append(f'{feature}_log10')
                new_train_features.append(
                    np.log10(
                        train_working_df[feature],
                        out=np.zeros_like(train_working_df[feature], dtype=np.float64),
                        where=(np.array(train_working_df[feature]) > 0)
                    )
                )

                if test_df is not None:
                    new_test_feature_names.append(f'{feature}_log10')
                    new_test_features.append(
                        np.log10(
                            test_working_df[feature],
                            out=np.zeros_like(test_working_df[feature], dtype=np.float64),
                            where=(np.array(test_working_df[feature]) > 0)
                        )
                    )

        logger.debug('New train features shape: %s', np.array(new_train_features).shape)
        logger.debug('New train feature names shape: %s', len(new_train_feature_names))

        new_train_features_df = pd.DataFrame(
            np.array(new_train_features).T,
            columns=new_train_feature_names
        )

        if test_df is not None:
            logger.debug('New test features shape: %s', np.array(new_test_features).shape)
            logger.debug('New test feature names shape: %s', len(new_test_feature_names))

            new_test_features_df = pd.DataFrame(
                np.array(new_test_features).T,
                columns=new_test_feature_names
            )

        else:
            new_test_features_df = None

        train_df, test_df = add_new_features(
            new_train_features=new_train_features_df,
            new_test_features=new_test_features_df,
            train_df=train_df,
            test_df=test_df
        )

    return train_df, test_df


def ratio_features(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        features:list,
        kwargs:dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Adds every possible ratio feature, replaces divide by zero errors
    with np.nan.'''

    logger = logging.getLogger(__name__ + '.ratio_features')
    logger.addHandler(logging.NullHandler())
    logger.debug('Adding ratio features')

    features, train_working_df, test_working_df=preprocess_features(
        features=features,
        train_df=train_df,
        test_df=test_df,
        preprocessing_steps=[
            'exclude_string_features',
            'enforce_floats',
            'remove_inf', 
            'remove_large_nums',
            'remove_small_nums',
            'knn_impute',
            'remove_constants'
        ]
    )

    if features is not None and len(features) > 1:

        features, train_working_df, test_working_df = scale_to_range(
            features=features,
            train_df=train_working_df,
            test_df=test_working_df,
            min_val=1,
            max_val=10
        )

        feature_pairs=list(permutations(features, 2))

        if len(feature_pairs) > 100:
            feature_pairs = choices(feature_pairs, k=100)

        logger.debug(
            'Will compute quotients for %s pairs of features', len(feature_pairs)
        )

        new_train_feature_names = []
        new_train_features = []
        new_test_feature_names = []
        new_test_features = []

        for i, (feature_a, feature_b) in enumerate(feature_pairs):
            logger.debug('Dividing feature pair %s of %s', i + 1, len(feature_pairs))

            quotient = np.divide(
                np.array(train_working_df[feature_a]),
                np.array(train_working_df[feature_b]),
                out=np.array([kwargs['div_zero_value']]*len(train_working_df[feature_a])),
                where=np.array(train_working_df[feature_b]) != 0
            )

            new_train_feature_names.append(f'{feature_a}_over_{feature_b}')
            new_train_features.append(quotient)

            if test_df is not None:

                quotient = np.divide(
                    np.array(test_working_df[feature_a]),
                    np.array(test_working_df[feature_b]),
                    out=np.array([kwargs['div_zero_value']]*len(test_working_df[feature_a])),
                    where=np.array(test_working_df[feature_b]) != 0
                )

                new_test_feature_names.append(f'{feature_a}_over_{feature_b}')
                new_test_features.append(quotient)

        logger.debug('New train features shape: %s', np.array(new_train_features).shape)
        logger.debug('New train feature names shape: %s', len(new_train_feature_names))

        new_train_features_df = pd.DataFrame(
            np.array(new_train_features).T,
            columns=new_train_feature_names
        )

        if test_df is not None:
            new_test_features_df = pd.DataFrame(
                np.array(new_test_features).T,
                columns=new_test_feature_names
            )

        else:
            new_test_features_df = None

        train_df, test_df = add_new_features(
            new_train_features=new_train_features_df,
            new_test_features=new_test_features_df,
            train_df=train_df,
            test_df=test_df
        )

    else:
        logger.debug('No features to divide')

    return train_df, test_df


def exponential_features(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        features:list,
        kwargs:dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Adds exponential features with base 2 or base e.'''

    logger = logging.getLogger(__name__ + '.exponential_features')
    logger.addHandler(logging.NullHandler())
    logger.debug('Adding exponential features')

    features, train_working_df, test_working_df=preprocess_features(
        features=features,
        train_df=train_df,
        test_df=test_df,
        preprocessing_steps=[
            'exclude_string_features',
            'enforce_floats',
            'remove_inf', 
            'remove_large_nums',
            'remove_small_nums',
            'knn_impute',
            'remove_constants',
            'scale_to_range'
        ]
    )

    if features is not None and len(features) > 0:

        new_train_feature_names = []
        new_train_features = []
        new_test_feature_names = []
        new_test_features = []

        for feature in features:

            if kwargs['base'] == 'e':
                new_train_feature_names.append(f'{feature}_exp_base_e')
                new_train_features.append(
                    e**train_working_df[feature].astype(float)
                )

                if test_df is not None:
                    new_test_feature_names.append(f'{feature}_exp_base_e')
                    new_test_features.append(
                        e**test_working_df[feature].astype(float)
                    )

            elif kwargs['base'] == '2':
                new_train_feature_names.append(f'{feature}_exp_base_2')
                new_train_features.append(
                    2**train_working_df[feature].astype(float))

                if test_df is not None:
                    new_test_feature_names.append(f'{feature}_exp_base_2')
                    new_test_features.append(
                        2**test_working_df[feature].astype(float)
                    )

        new_train_features_df = pd.DataFrame(
            np.array(new_train_features).T,
            columns=new_train_feature_names
        )

        if test_df is not None:
            new_test_features_df = pd.DataFrame(
                np.array(new_test_features).T,
                columns=new_test_feature_names
            )

        else:
            new_test_features_df = None

        train_df, test_df = add_new_features(
            new_train_features=new_train_features_df,
            new_test_features=new_test_features_df,
            train_df=train_df,
            test_df=test_df
        )

    return train_df, test_df


def sum_features(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        features:list,
        kwargs:dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Adds sum features for variable number of addends.'''

    logger = logging.getLogger(__name__ + '.sum_features')
    logger.addHandler(logging.NullHandler())
    logger.debug('Adding sum features')

    features, train_working_df, test_working_df=preprocess_features(
        features=features,
        train_df=train_df,
        test_df=test_df,
        preprocessing_steps=[
            'exclude_string_features',
            'enforce_floats',
            'remove_inf', 
            'remove_large_nums',
            'remove_small_nums',
            'knn_impute',
            'remove_constants',
            'scale_to_range'
        ]
    )

    if features is not None and len(features) > 1:

        if kwargs['n_addends'] > len(features):
            n_addends=len(features)

        else:
            n_addends=kwargs['n_addends']

        new_train_feature_names = []
        new_train_features = []
        new_test_feature_names = []
        new_test_features = []

        addend_sets=list(combinations(features, n_addends))

        if len(addend_sets) > 100:
            addend_sets = choices(addend_sets, k=100)

        logger.debug(
            'Will compute sums for %s sets of %s features', len(addend_sets), n_addends
        )

        for i, addend_set in enumerate(addend_sets):

            logger.debug('Adding feature set %s of %s', i + 1, len(addend_sets))

            train_sum = np.array([0.0]*len(train_working_df))

            for addend in addend_set:

                train_sum += train_working_df[addend].astype(float).to_numpy()

            new_train_feature_names.append(f'sum_feature_{i}')
            new_train_features.append(train_sum)

            if test_df is not None:

                test_sum = np.array([0.0]*len(test_working_df))

                for addend in addend_set:

                    test_sum += test_working_df[addend].astype(float).to_numpy()

                new_test_feature_names.append(f'sum_feature_{i}')
                new_test_features.append(test_sum)

        logger.debug('New train features shape: %s', np.array(new_train_features).shape)
        logger.debug('New train feature names shape: %s', len(new_train_feature_names))

        new_train_features_df = pd.DataFrame(
            np.array(new_train_features).T,
            columns=new_train_feature_names
        )

        if test_df is not None:
            new_test_features_df = pd.DataFrame(
                np.array(new_test_features).T,
                columns=new_test_feature_names
            )

        else:
            new_test_features_df = None

        train_df, test_df=add_new_features(
            new_train_features = new_train_features_df,
            new_test_features = new_test_features_df,
            train_df = train_df,
            test_df = test_df
        )

    else:
        logger.debug('No features to sum')

    return train_df, test_df


def difference_features(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        features:list,
        kwargs:dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Adds difference features for variable number of subtrahends.'''

    logger = logging.getLogger(__name__ + '.difference_features')
    logger.addHandler(logging.NullHandler())
    logger.debug('Adding difference features')

    features, train_working_df, test_working_df=preprocess_features(
        features=features,
        train_df=train_df,
        test_df=test_df,
        preprocessing_steps=[
            'exclude_string_features',
            'enforce_floats',
            'remove_inf', 
            'remove_large_nums',
            'remove_small_nums',
            'knn_impute',
            'remove_constants',
            'scale_to_range'
        ]
    )

    if features is not None and len(features) > 1:

        if kwargs['n_subtrahends'] > len(features):
            n_subtrahends=len(features)

        else:
            n_subtrahends=kwargs['n_subtrahends']

        new_train_feature_names = []
        new_train_features = []
        new_test_feature_names = []
        new_test_features = []

        subtrahend_sets=list(combinations(features, n_subtrahends))

        if len(subtrahend_sets) > 100:
            subtrahend_sets = choices(subtrahend_sets, k=100)

        logger.debug(
            'Will compute differences for %s sets of %s features',
            len(subtrahend_sets),
            n_subtrahends
        )

        for i, subtrahend_set in enumerate(subtrahend_sets):

            logger.debug('Subtracting feature set %s of %s', i + 1, len(subtrahend_sets))
            train_difference = np.array(train_working_df[subtrahend_set[0]])

            for subtrahend in subtrahend_set[1:]:
                train_difference -= train_working_df[subtrahend].astype(float).to_numpy()

            new_train_feature_names.append('-'.join(subtrahend_set))
            new_train_features.append(train_difference)

            if test_df is not None:
                test_difference = np.array(test_working_df[subtrahend_set[0]])

                for subtrahend in subtrahend_set[1:]:
                    test_difference -= test_working_df[subtrahend].astype(float).to_numpy()

                new_test_feature_names.append('-'.join(subtrahend_set))
                new_test_features.append(test_difference)

        logger.debug('New train features shape: %s', np.array(new_train_features).shape)
        logger.debug('New train feature names shape: %s', len(new_train_feature_names))

        new_train_features_df = pd.DataFrame(
            np.array(new_train_features).T,
            columns=new_train_feature_names
        )

        if test_df is not None:
            new_test_features_df = pd.DataFrame(
                np.array(new_test_features).T,
                columns=new_test_feature_names
            )

        else:
            new_test_features_df = None

        train_df, test_df = add_new_features(
            new_train_features=new_train_features_df,
            new_test_features=new_test_features_df,
            train_df=train_df,
            test_df=test_df
        )

    else:
        logger.debug('No features to subtract')

    return train_df, test_df


def kde_smoothing(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        features:list,
        kwargs:dict,
        shortcircuit_preprocessing:bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Uses kernel density estimation to smooth features.'''

    logger = logging.getLogger(__name__ + '.kde_smoothing')
    logger.addHandler(logging.NullHandler())
    logger.debug('Adding kernel density estimate smoothed features')

    if shortcircuit_preprocessing is False:
        features, train_working_df, test_working_df=preprocess_features(
            features=features,
            train_df=train_df,
            test_df=test_df,
            preprocessing_steps=[
                'exclude_string_features',
                'enforce_floats',
                'remove_inf', 
                'remove_large_nums',
                'remove_small_nums',
                'knn_impute',
                'remove_constants',
                'scale_to_range'
            ]
        )

    else:
        train_working_df = train_df.copy()

        if test_df is not None:
            test_working_df = test_df.copy()

        else:
            test_working_df = None

    if features is not None and len(features) > 0:

        new_test_features={}
        new_train_features={}

        if len(train_working_df) > kwargs['sample_size']:
            sample_df=train_working_df.sample(n=kwargs['sample_size'])

        else:
            sample_df=train_working_df

        workers = 2

        for feature in features:

            try:
                scipy_kde = gaussian_kde(
                    sample_df[feature].to_numpy().flatten(),
                    bw_method = kwargs['bandwidth']
                )

                with mp.Pool(workers) as p:
                    new_train_features[f'{feature}_kde'] = np.concatenate(p.map(
                        scipy_kde,
                        np.array_split(train_working_df[feature].to_numpy().flatten(), workers)
                    ))

                if test_df is not None:
                    with mp.Pool(workers) as p:
                        new_test_features[f'{feature}_kde'] = np.concatenate(p.map(
                            scipy_kde,
                            np.array_split(test_working_df[feature].to_numpy().flatten(), workers)
                        ))

            except TypeError:
                logger.error('Typeerror in KDE smoother')

            except np.linalg.LinAlgError:
                logger.error('Numpy linear algebra error in gaussian KDE')

            except ValueError:
                logger.error('Valueerror in KDE smoother')

        train_df, test_df=add_new_features(
            new_train_features = new_train_features,
            new_test_features = new_test_features,
            train_df = train_df,
            test_df = test_df
        )

    return train_df, test_df


def kbins_quantization(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        features:list,
        kwargs:dict,
        shortcircuit_preprocessing:bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Discretizes feature with Kbins quantization.'''

    logger = logging.getLogger(__name__ + '.kbins_quantization')
    logger.addHandler(logging.NullHandler())
    logger.debug('Adding k-bins quantized features')

    if shortcircuit_preprocessing is False:
        features, train_working_df, test_working_df=preprocess_features(
            features=features,
            train_df=train_df,
            test_df=test_df,
            preprocessing_steps=[
                'exclude_string_features',
                'enforce_floats',
                'remove_inf', 
                'remove_large_nums',
                'remove_small_nums',
                'knn_impute',
                'remove_constants',
                'scale_to_range'
            ]
        )

    else:
        train_working_df = train_df.copy()

        if test_df is not None:
            test_working_df = test_df.copy()

        else:
            test_working_df = None

    new_train_features = {}
    new_test_features = {}

    if features is not None and len(features) > 0:
        for feature in features:

            kbins = KBinsDiscretizer(**kwargs)

            try:
                binned_feature = kbins.fit_transform(train_working_df[feature].to_frame())
                binned_feature_name = f'{kbins.get_feature_names_out()}_bins'
                new_train_features[binned_feature_name] = binned_feature.flatten()

                if test_df is not None:
                    binned_feature = kbins.transform(test_working_df[feature].to_frame())
                    new_test_features[binned_feature_name] = binned_feature.flatten()

            except ValueError:
                logger.error('Caught ValueError in KbinsDiscretizer')

        train_df, test_df = add_new_features(
            new_train_features = new_train_features,
            new_test_features = new_test_features,
            train_df = train_df,
            test_df = test_df
        )

    return train_df, test_df
