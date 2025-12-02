"""Generates variations of a dataset using a pool of feature engineering techniques.

This module provides the DataSet class for generating ensemble datasets through
randomized application of feature engineering methods. Each generated dataset
applies a different sequence of feature engineering techniques to randomly
selected subsets of features, enabling training of diverse ensemble models.

The feature engineering pipeline includes methods such as polynomial features,
splines, logarithmic transformations, ratios, exponentials, sums, differences,
Gaussian KDE smoothing, and binning. String features can be encoded using
one-hot or ordinal encoding.

Examples
--------
>>> import pandas as pd
>>> import ensembleset.dataset as ds
>>> 
>>> # Create sample data
>>> train_df = pd.DataFrame({'feature1': [1, 2, 3], 'label': [0, 1, 0]})
>>> test_df = pd.DataFrame({'feature1': [4, 5], 'label': [1, 0]})
>>> 
>>> # Initialize DataSet
>>> data_ensemble = ds.DataSet(
...     label='label',
...     train_data=train_df,
...     test_data=test_df,
...     data_directory='./data'
... )
>>> 
>>> # Generate 10 ensemble datasets
>>> data_ensemble.make_datasets(n_datasets=10, frac_features=0.1, n_steps=5)

See Also
--------
ensembleset.feature_methods : Feature engineering function implementations
ensembleset.preprocessing_methods : Data preprocessing utilities
"""

import time
import logging
from multiprocessing import Manager, Process, cpu_count
from pathlib import Path
from random import choice, shuffle

import h5py
import numpy as np
import pandas as pd

import ensembleset.feature_engineerings as engineerings
import ensembleset.feature_methods as fm

logging.captureWarnings(True)


class DataSet:
    """Dataset ensemble generator using randomized feature engineering.
    
    This class generates multiple dataset variations by applying randomized
    sequences of feature engineering methods to randomized subsets of input
    features. Each dataset in the ensemble undergoes a unique transformation
    pipeline, making it suitable for training diverse ensemble models.
    
    The class handles both training and testing data with minimal data leakage
    by fitting transformations on training data and applying them to both
    training and testing sets. Generated datasets are saved to HDF5 format
    for efficient storage and retrieval.
    
    Parameters
    ----------
    label : str
        Name of the label column in the training and testing DataFrames.
        This column will be extracted and stored separately from the features.
    train_data : pd.DataFrame
        Training dataset containing both features and the label column.
        Must include the column specified by `label` parameter.
    test_data : pd.DataFrame, optional
        Testing dataset containing both features and the label column.
        If provided, transformations fitted on training data will be
        applied to this data. Default is None.
    string_features : list of str, optional
        List of column names containing string/categorical features that
        require encoding (one-hot or ordinal) before numerical feature
        engineering. Default is None.
    data_directory : str, optional
        Path to directory where generated ensemble datasets will be saved
        in HDF5 format. Directory will be created if it doesn't exist.
        Default is 'ensembleset_data'.
    ensembleset_base_name : str, optional
        Base name for the output HDF5 file. The file will be named
        '{ensembleset_base_name}.h5'. Default is 'ensembleset'.
    
    Attributes
    ----------
    label : str
        Name of the label column.
    train_data : pd.DataFrame
        Training features (without label column).
    test_data : pd.DataFrame or None
        Testing features (without label column).
    train_labels : np.ndarray
        Training labels extracted from train_data.
    test_labels : np.ndarray or None
        Testing labels extracted from test_data.
    string_features : list of str or None
        List of string feature column names.
    data_directory : str
        Path to output directory.
    ensembleset_base_name : str
        Base name for output files.
    string_encodings : dict
        Dictionary of available string encoding methods.
    numerical_methods : dict
        Dictionary of available numerical feature engineering methods.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import ensembleset.dataset as ds
    >>> 
    >>> # Create sample training and testing data
    >>> train_df = pd.DataFrame({
    ...     'num_feature1': [1.0, 2.0, 3.0, 4.0],
    ...     'num_feature2': [10.0, 20.0, 30.0, 40.0],
    ...     'cat_feature': ['A', 'B', 'A', 'B'],
    ...     'target': [0, 1, 0, 1]
    ... })
    >>> test_df = pd.DataFrame({
    ...     'num_feature1': [5.0, 6.0],
    ...     'num_feature2': [50.0, 60.0],
    ...     'cat_feature': ['A', 'B'],
    ...     'target': [1, 0]
    ... })
    >>> 
    >>> # Initialize DataSet with string feature specification
    >>> ensemble = ds.DataSet(
    ...     label='target',
    ...     train_data=train_df,
    ...     test_data=test_df,
    ...     string_features=['cat_feature'],
    ...     data_directory='./ensemble_output',
    ...     ensembleset_base_name='my_ensemble'
    ... )
    >>> 
    >>> # Generate 5 datasets with 3 feature engineering steps each,
    >>> # using 20% of features per step
    >>> output_file = ensemble.make_datasets(
    ...     n_datasets=5,
    ...     frac_features=0.2,
    ...     n_steps=3
    ... )
    
    Notes
    -----
    - Label columns are automatically removed from feature DataFrames and
      stored separately as numpy arrays.
    - String features are encoded before numerical transformations are applied.
    - All transformations are fitted on training data only to prevent data
      leakage, even when test data is provided.
    - Generated datasets are saved in HDF5 format with the structure:
      dataset.h5/train/labels, dataset.h5/train/dataset_N,
      dataset.h5/test/labels, dataset.h5/test/dataset_N
    - Uses multiprocessing for parallel dataset generation.
    
    See Also
    --------
    make_datasets : Generate ensemble datasets with specified parameters
    ensembleset.feature_engineerings : Configuration of available methods
    """

    def __init__(
            self,
            label: str,
            train_data: pd.DataFrame,
            test_data: pd.DataFrame = None,
            string_features: list = None,
            data_directory: str = 'ensembleset_data',
            ensembleset_base_name: str = 'ensembleset'
        ):

        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())

        # Check user argument types
        type_check = self._check_argument_types(
            label=label,
            train_data=train_data,
            test_data=test_data,
            string_features=string_features,
            data_directory=data_directory,
            ensembleset_base_name=ensembleset_base_name
        )

        # If the type check passed, assign arguments to attributes
        if type_check is True:
            self.label = label
            self.train_data = train_data.copy()

            if test_data is not None:
                self.test_data = test_data.copy()

            else:
                self.test_data = None

            self.string_features = string_features
            self.data_directory = data_directory
            self.ensembleset_base_name = ensembleset_base_name

        # Create the HDF5 output
        try:
            Path(self.data_directory).mkdir(parents=True, exist_ok=True)

        except OSError as exc:
            logger.error('Could not create output directory: %s', self.data_directory)
            raise OSError from exc

        logger.info("Training label: '%s'", self.label)
        logger.info('Training data: %s', type(self.train_data))
        logger.info('Testing data: %s', type(self.test_data))
        logger.info('String features: %s', self.string_features)
        logger.info('Data directory: %s', self.data_directory)
        logger.info('Ensembleset basename: %s', self.ensembleset_base_name)

        # Enforce string type on DataFrame columns
        self.train_data.columns = self.train_data.columns.astype(str)

        if self.test_data is not None:
            self.test_data.columns = self.test_data.columns.astype(str)

        # Retrieve and assign the training labels, set NAN if they don't exist
        # then remove them from the training data
        if self.label in self.train_data.columns:
            self.train_labels=np.array(self.train_data[label])
            self.train_data.drop(self.label, axis=1, inplace=True)

        else:
            self.train_labels=[np.nan] * len(self.train_data)

        # Retrieve and assign the testing labels, set NAN if they don't exist
        # then remove them from the training data
        if self.test_data is not None:
            if self.label in self.test_data.columns:
                self.test_labels=np.array(self.test_data[label])
                self.test_data.drop(self.label, axis=1, inplace=True)

            else:
                self.test_labels=[np.nan] * len(self.test_data)

        else:
            self.test_labels = None

        # Define the feature engineering pipeline methods
        self.string_encodings=engineerings.STRING_ENCODINGS
        self.numerical_methods=engineerings.NUMERICAL_METHODS


    def make_datasets(
            self,
            n_datasets: int,
            frac_features: int,
            n_steps: int
    ):
        """Generate ensemble datasets using randomized feature engineering pipelines.
        
        Creates multiple dataset variations by applying randomized sequences of
        feature engineering methods to randomly selected subsets of features.
        Each dataset undergoes a unique transformation pipeline, with feature
        selection re-randomized at each engineering step.
        
        The method uses multiprocessing to generate datasets in parallel, with
        the number of worker processes equal to half the available CPU cores.
        All generated datasets are saved to an HDF5 file in the specified
        data directory.
        
        Parameters
        ----------
        n_datasets : int
            Number of dataset variations to generate. Each dataset will have
            a unique sequence of feature engineering transformations applied.
            Must be a positive integer.
        frac_features : float
            Fraction of features to randomly select for each feature engineering
            step. Must be between 0 and 1. For example, 0.1 means 10% of
            available features are selected at each step. The selection is
            re-randomized for each step in the pipeline.
        n_steps : int
            Number of feature engineering steps to apply in sequence for each
            dataset. Each step randomly selects a method from the available
            feature engineering techniques. Must be a positive integer.
        
        Returns
        -------
        str
            Path to the generated HDF5 file containing all ensemble datasets.
            The file structure is:
            - train/labels: Training labels (np.ndarray)
            - train/dataset_0, train/dataset_1, ...: Training feature sets
            - test/labels: Testing labels (np.ndarray, if test_data provided)
            - test/dataset_0, test/dataset_1, ...: Testing feature sets
        
        Examples
        --------
        >>> import pandas as pd
        >>> import ensembleset.dataset as ds
        >>> 
        >>> # Create sample data
        >>> train_df = pd.DataFrame({
        ...     'feature1': [1, 2, 3, 4, 5],
        ...     'feature2': [10, 20, 30, 40, 50],
        ...     'feature3': [100, 200, 300, 400, 500],
        ...     'label': [0, 1, 0, 1, 0]
        ... })
        >>> 
        >>> # Initialize ensemble
        >>> ensemble = ds.DataSet(
        ...     label='label',
        ...     train_data=train_df,
        ...     data_directory='./output'
        ... )
        >>> 
        >>> # Generate 10 datasets, using 20% of features per step,
        >>> # with 5 engineering steps each
        >>> output_file = ensemble.make_datasets(
        ...     n_datasets=10,
        ...     frac_features=0.2,
        ...     n_steps=5
        ... )
        >>> print(f"Datasets saved to: {output_file}")
        
        >>> # With test data included
        >>> test_df = pd.DataFrame({
        ...     'feature1': [6, 7],
        ...     'feature2': [60, 70],
        ...     'feature3': [600, 700],
        ...     'label': [1, 0]
        ... })
        >>> 
        >>> ensemble_with_test = ds.DataSet(
        ...     label='label',
        ...     train_data=train_df,
        ...     test_data=test_df
        ... )
        >>> 
        >>> output_file = ensemble_with_test.make_datasets(
        ...     n_datasets=5,
        ...     frac_features=0.3,
        ...     n_steps=3
        ... )
        
        Notes
        -----
        - Each dataset has a unique random sequence of feature engineering methods
        - Feature selection is re-randomized after each engineering step
        - At least 1 feature is always selected, even if frac_features * n_features < 1
        - String features are encoded first (if specified), then numerical methods
          are applied
        - All transformations are fitted on training data only to prevent leakage
        - When test_data is provided, fitted transformations are applied to both
          training and testing data
        - Uses multiprocessing with worker count = cpu_count() // 2
        - Progress and debugging information is logged during execution
        
        See Also
        --------
        ensembleset.feature_methods : Individual feature engineering functions
        ensembleset.preprocessing_methods : Data preprocessing utilities
        """

        logger = logging.getLogger(__name__ + '.make_datasets')
        logger.addHandler(logging.NullHandler())

        logger.info('Will make %s datasets', n_datasets)
        logger.info('Running %s feature engineering steps per dataset', n_steps)
        logger.info('Selecting %s percent of features for each step', round(frac_features * 100))

        ensembleset_file = self._create_output(n_datasets, frac_features, n_steps)

        # Start multiprocessing manager and create queues for I/O to dataset worker
        # and logging from workers
        manager=Manager()
        input_queue=manager.Queue()
        output_queue=manager.Queue()
        logging_queue=manager.Queue()

        # Create dataset worker processes
        dataset_worker_processes=[]

        for worker_num in range(cpu_count() // 2):
            dataset_worker_processes.append(
                Process(
                    target=self._dataset_worker,
                    args=(
                        worker_num,
                        input_queue,
                        output_queue,
                        logging_queue,
                    )
                )
            )

        for worker_num, dataset_worker_process in enumerate(dataset_worker_processes):
            logger.info('Starting dataset worker %s', worker_num)
            dataset_worker_process.start()

        # Create output worker process
        output_worker_process = Process(
            target=self._output_worker,
            args=(
                ensembleset_file,
                len(dataset_worker_processes),
                output_queue,
                logging_queue,
            )
        )

        output_worker_process.start()

        # Create worker logging process
        worker_logging_process = Process(
            target=self._logging_worker,
            args=(len(dataset_worker_processes), logging_queue,)
        )

        worker_logging_process.start()

        # Add workunits to the queue
        for n in range(n_datasets):
            input_queue.put(
                {
                    'dataset': n,
                    'frac_features': frac_features,
                    'n_steps': n_steps
                }
            )

        # Send done signals
        for dataset_worker_process in dataset_worker_processes:
            input_queue.put({'dataset': 'done'})

        # Join and close workers
        for dataset_worker_process in dataset_worker_processes:
            dataset_worker_process.join()
            dataset_worker_process.close()

        output_worker_process.join()
        output_worker_process.close()

        worker_logging_process.join()
        worker_logging_process.close()

        return ensembleset_file


    def _dataset_worker(self, worker_num, input_queue, output_queue, logging_queue):
        '''Worker process to generate datasets for a given ensemble.'''

        while True:
            workunit = input_queue.get()
            dataset = workunit['dataset']

            if dataset == 'done':
                output_queue.put({'worker': 'done'})
                logging_queue.put({'worker': 'done'})
                return

            else:
                frac_features = workunit['frac_features']
                n_steps = workunit['n_steps']

            logging_queue.put({
                'worker': str(worker_num),
                'level': 'info',
                'message': f'Input training data shape: {self.train_data.shape}'
            })

            # Take a copy of the training and test data and sample if desired
            train_df = self.train_data.copy()

            if self.test_data is not None:
                test_df = self.test_data.copy()

                logging_queue.put({
                    'worker': str(worker_num),
                    'level': 'info',
                    'message': f'Input testing data shape: {self.test_data.shape}'
                })

            else:
                test_df = None

            # Generate a data pipeline
            pipeline = self._generate_data_pipeline(n_steps)

            # Set input n features for first round
            input_n_features = int(len(train_df.columns.to_list()) * frac_features)
            input_n_features = max([input_n_features, 1])

            # Loop on and apply each method in the pipeline
            for method, arguments in pipeline.items():
                func = getattr(fm, method)

                if method in self.string_encodings:

                    logging_queue.put({
                        'worker': str(worker_num),
                        'level': 'info',
                        'message': f'Applying {method} to {self.string_features}'
                    })

                    train_df, test_df = func(
                        train_df,
                        test_df,
                        self.string_features,
                        arguments
                    )

                else:

                    n_features = int(len(train_df.columns.to_list()) * frac_features)
                    n_features = max([n_features, 1])
                    n_features = min([n_features, 2 * input_n_features])
                    input_n_features = n_features

                    features = self._select_features(n_features, train_df)

                    logging_queue.put({
                        'worker': str(worker_num),
                        'level': 'info',
                        'message': f'Applying {method} to {len(features)} features'
                    })

                    train_df, test_df = func(
                        train_df,
                        test_df,
                        features,
                        arguments
                    )

                    logging_queue.put({
                        'worker': str(worker_num),
                        'level': 'info',
                        'message': f'New training data shape: {train_df.shape}'
                    })

                    if test_df is not None:
                        logging_queue.put({
                            'worker': str(worker_num),
                            'level': 'info',
                            'message': f'New testing data shape: {test_df.shape}'
                        })

            logging_queue.put({
                'worker': str(worker_num),
                'level': 'info',
                'message': f'Final training data shape: {train_df.shape}'
            })

            if test_df is not None:
                logging_queue.put({
                    'worker': str(worker_num),
                    'level': 'info',
                    'message': f'Final testing data shape: {test_df.shape}'
                })

            output_queue.put({
                'worker': worker_num,
                'dataset': dataset,
                'train_df': train_df,
                'test_df': test_df
            })

            time.sleep(0.1)


    def _output_worker(self, ensembleset_file, n_workers, output_queue, logging_queue):
        '''Worker process to collect completed datasets and write them to HDF5.'''

        done_count = 0
        dataset = None
        train_df = None
        test_df = None

        while True:
            workunit = output_queue.get()
            worker = workunit['worker']

            if worker == 'done':
                done_count += 1

                if done_count == n_workers:
                    logging_queue.put({'worker': 'done'})

                    return

            else:
                dataset = workunit['dataset']
                train_df = workunit['train_df']
                test_df = workunit['test_df']

                with h5py.File(f'{self.data_directory}/{ensembleset_file}', 'a') as hdf:

                    # Save the results to HDF5 output
                    _ = hdf.create_dataset(
                        f'train/{dataset}',
                        data=np.array(train_df).astype(np.float64)
                    )

                    if test_df is not None:
                        _ = hdf.create_dataset(
                            f'test/{dataset}',
                            data=np.array(test_df).astype(np.float64)
                        )

                logging_queue.put({
                    'worker': 'output',
                    'level': 'info',
                    'message': f'Saved dataset {dataset} from worker {worker}'
                })

            time.sleep(0.1)


    def _logging_worker(self, n_workers, logging_queue):
        '''Worker process to log messages from dataset and output workers.'''

        logger = logging.getLogger(__name__ + '.dataset_worker')
        logger.addHandler(logging.NullHandler())

        done_count = 0
        level = None
        message = None

        while True:
            workunit = logging_queue.get()
            worker = workunit['worker']

            if worker == 'done':
                done_count += 1

                if done_count == n_workers + 1:
                    return

            else:

                level = workunit['level']
                message = workunit['message']

                if level == 'info':
                    logger.info(message)

                elif level == 'debug':
                    logger.debug(message)

                elif level == 'warning':
                    logger.warning(message)

                elif level == 'error':
                    logger.error(message)


    def _select_features(self, n_features:int, data_df:pd.DataFrame):
        '''Selects a random subset of features.'''

        features = data_df.columns.to_list()
        shuffle(features)
        features = features[:n_features]

        return features


    def _generate_data_pipeline(self, n_steps:int):
        '''Generates one random sequence of feature engineering operations. Starts with
        a string encoding method if we have string features.'''

        pipeline={}

        # Choose a string encoding method, if needed
        if self.string_features is not None:
            options = list(self.string_encodings.keys())
            selection = choice(options)
            pipeline[selection] = self.string_encodings[selection]

        # Construct a random sequence of numerical feature engineering methods
        methods = list(self.numerical_methods.keys())
        shuffle(methods)
        methods = methods[:n_steps]

        for method in methods:

            pipeline[method] = {}
            parameters = self.numerical_methods[method]

            for parameter, values in parameters.items():

                value = choice(values)
                pipeline[method][parameter] = value

        return pipeline


    def _check_argument_types(
            self,
            label: str,
            train_data: pd.DataFrame,
            test_data: pd.DataFrame,
            string_features: list,
            data_directory: str,
            ensembleset_base_name: str
    ) -> bool:

        '''Checks user argument types, returns true or false for all passing.'''

        check_pass = False

        if isinstance(label, str):
            check_pass = True

        else:
            raise TypeError('Label is not a string.')

        if isinstance(train_data, pd.DataFrame):
            check_pass = True

        else:
            raise TypeError('Train data is not a Pandas DataFrame.')

        if isinstance(test_data, pd.DataFrame) or test_data is None:
            check_pass = True

        else:
            raise TypeError('Test data is not a Pandas DataFrame.')

        if isinstance(string_features, list) or string_features is None:
            check_pass = True

        else:
            raise TypeError('String features is not a list.')

        if isinstance(data_directory, str):
            check_pass = True

        else:
            raise TypeError('Data directory is not a string.')

        if isinstance(ensembleset_base_name, str):
            check_pass = True

        else:
            raise TypeError('Ensembleset base name is not a string.')

        return check_pass


    def _create_output(self, n_datasets:int, frac_features:int, n_steps:int):
        '''Creates HDF5 output sink for ensembleset.'''

        # Create groups for training and testing datasets
        ensembleset_file = (f'{self.ensembleset_base_name}-' +
                            f'{n_datasets}-{frac_features}-{n_steps}.h5')

        with h5py.File(f'{self.data_directory}/{ensembleset_file}', 'a') as hdf:

            # Create groups for train and test datasets
            _ = hdf.require_group('train')

            if self.test_data is not None:
                _ = hdf.require_group('test')

        # Add the training and testing labels
        with h5py.File(f'{self.data_directory}/{ensembleset_file}', 'w') as hdf:

            _ = hdf.create_dataset('train/labels', data=self.train_labels)

            if self.test_data is not None:
                _ = hdf.create_dataset('test/labels', data=self.test_labels)

        return ensembleset_file
