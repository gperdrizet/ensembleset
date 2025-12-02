'''Unittests for dataset class.'''

import logging
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ensembleset import DataSet
import dummy_dataframe as test_data

# pylint: disable=protected-access

Path('tests/logs').mkdir(parents=True, exist_ok=True)

logging.captureWarnings(True)

logger = logging.getLogger()

logging.basicConfig(
    filename='tests/logs/test_dataset.log',
    filemode='w',
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)

class TestDataSetInit(unittest.TestCase):
    '''Tests for main data set generator class initialization.'''

    def setUp(self):
        '''Dummy DataFrames for tests.'''

        self.dummy_df = test_data.DUMMY_DF


    def test_class_arguments(self):
        '''Tests assignments of class attributes from user arguments.'''

        dataset = DataSet(
            label='floats_pos',
            train_data=self.dummy_df.copy(),
            test_data=self.dummy_df.copy(),
            string_features=['strings']
        )

        self.assertTrue(isinstance(dataset.label, str))
        self.assertTrue(isinstance(dataset.train_data, pd.DataFrame))
        self.assertTrue(isinstance(dataset.test_data, pd.DataFrame))
        self.assertTrue(isinstance(dataset.string_features, list))
        self.assertEqual(dataset.string_features[0], 'strings')
        self.assertEqual(dataset.data_directory, 'ensembleset_data')
        self.assertEqual(dataset.ensembleset_base_name, 'ensembleset')

        dataset = DataSet(
            label='floats_pos',
            train_data=self.dummy_df.copy(),
            test_data=self.dummy_df.copy(),
            string_features=['strings'],
            data_directory='test',
            ensembleset_base_name='test'
        )

        self.assertEqual(dataset.data_directory, 'test')
        self.assertEqual(dataset.ensembleset_base_name, 'test')

        with self.assertRaises(TypeError):
            DataSet(
                label=2, # Bad label
                train_data=self.dummy_df.copy(),
                test_data=self.dummy_df.copy(),
                string_features=['strings']
            )

        with self.assertRaises(TypeError):
            DataSet(
                label='float_pos',
                train_data='Not a Pandas Dataframe', # Bad train data
                test_data=self.dummy_df.copy(),
                string_features=['strings']
            )

        with self.assertRaises(TypeError):
            DataSet(
                label='float_pos',
                train_data=self.dummy_df.copy(),
                test_data='Not a Pandas Dataframe', # Bad test data
                string_features=['strings']
            )

        with self.assertRaises(TypeError):
            DataSet(
                label='float_pos',
                train_data=self.dummy_df.copy(),
                test_data=self.dummy_df.copy(),
                string_features='Not a list of features' # Bad string features
            )

        with self.assertRaises(TypeError):
            DataSet(
                label='float_pos',
                train_data=self.dummy_df.copy(),
                test_data=None,
                string_features='strings' # Bad string features
            )

        with self.assertRaises(TypeError):
            DataSet(
                label='float_pos',
                train_data=self.dummy_df.copy(),
                test_data=None,
                string_features=['strings'],
                data_directory=0 # bad data directory type
            )

        with self.assertRaises(TypeError):
            DataSet(
                label='float_pos',
                train_data=self.dummy_df.copy(),
                test_data=None,
                string_features=['strings'],
                ensembleset_base_name=0 # bad ensembleset basename type
            )

        with self.assertRaises(OSError):
            DataSet(
                label='float_pos',
                train_data=self.dummy_df.copy(),
                test_data=None,
                string_features=['strings'],
                data_directory='/test' # Sketchy data directory
            )


    def test_label_assignment(self):
        '''Tests assigning and saving labels.'''

        dataset = DataSet(
            label='floats_pos',
            train_data=self.dummy_df.copy(),
            test_data=self.dummy_df.copy(),
            string_features=['strings']
        )

        self.assertEqual(dataset.train_labels[-1], 7.0)
        self.assertEqual(dataset.test_labels[-1], 7.0)

        dataset=DataSet(
            label='bad_label_feature',
            train_data=self.dummy_df,
            test_data=self.dummy_df,
            string_features=['strings']
        )

        self.assertTrue(np.isnan(dataset.train_labels[-1]))
        self.assertTrue(np.isnan(dataset.test_labels[-1]))


    def test_pipeline_options(self):
        '''Tests the creation of feature engineering pipeline options'''

        dataset = DataSet(
            label='floats_pos',
            train_data=self.dummy_df.copy(),
            test_data=self.dummy_df.copy(),
            string_features=['strings']
        )

        self.assertTrue(isinstance(dataset.string_encodings, dict))
        self.assertTrue(isinstance(dataset.numerical_methods, dict))


class TestDataPipelineGen(unittest.TestCase):
    '''Tests for data pipeline generator function.'''

    def setUp(self):
        '''Dummy DataFrames and datasets for tests.'''

        self.dummy_df = test_data.DUMMY_DF

        self.dataset = DataSet(
            label='floats_pos',
            train_data=self.dummy_df.copy(),
            test_data=self.dummy_df.copy(),
            string_features=['strings']
        )


    def test_generate_data_pipeline(self):
        '''Tests the data pipeline generation function.'''

        pipeline=self.dataset._generate_data_pipeline(2)

        self.assertEqual(len(pipeline), 3)

        for operation, parameters in pipeline.items():
            self.assertTrue(isinstance(operation, str))
            self.assertTrue(isinstance(parameters, dict))


class TestFeatureSelection(unittest.TestCase):
    '''Tests for data pipeline generator function.'''

    def setUp(self):
        '''Dummy DataFrames and datasets for tests.'''

        self.dummy_df = test_data.DUMMY_DF

        self.dataset = DataSet(
            label='floats_pos',
            train_data=self.dummy_df.copy(),
            test_data=self.dummy_df.copy(),
            string_features=['strings']
        )


    def test_select_features(self):
        '''Tests feature selection function.'''

        features=self.dataset._select_features(3, self.dummy_df)
        self.assertEqual(len(features), 3)


class TestDatasetGeneration(unittest.TestCase):
    '''Tests dataset generation.'''

    def setUp(self):
        '''Dummy DataFrames and datasets for tests.'''

        self.n_datasets = 3
        self.frac_features = 0.1
        self.n_steps = 3

        self.dummy_df = test_data.DUMMY_DF


    def test_make_datasets(self):
        '''Tests generation of datasets.'''

        for test_df in [None, self.dummy_df.copy()]:
            dataset = DataSet(
                label='floats_pos',
                train_data=self.dummy_df.copy(),
                test_data=test_df,
                string_features=['strings'],
            )

            ensemble_file = dataset.make_datasets(
                n_datasets=self.n_datasets,
                frac_features=self.frac_features,
                n_steps=self.n_steps
            )

            with h5py.File(f'{dataset.data_directory}/{ensemble_file}', 'a') as hdf:

                self.assertEqual(len(hdf['train']), self.n_datasets + 1)

                if test_df is not None:
                    self.assertEqual(len(hdf['test']), self.n_datasets + 1)
