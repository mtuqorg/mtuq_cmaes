import copy
import numpy as np
import pandas as pd
from mpi4py import MPI

from mtuq import MTUQDataFrame
from mtuq.dataset import Dataset
from mtuq.event import Origin
# from mtuq.graphics import plot_combined, plot_misfit_force
from mtuq.graphics.uq._matplotlib import _plot_force_matplotlib
from mtuq.io.clients.AxiSEM_NetCDF import Client as AxiSEM_Client
from mtuq.greens_tensor.base import GreensTensorList
from mtuq.misfit import Misfit, PolarityMisfit, WaveformMisfit
from mtuq.misfit.waveform import level2 
from mtuq_cmaes.misfit import level2_replacer 
level2.misfit = level2_replacer.misfit

from mtuq.misfit.waveform import c_ext_L2, calculate_norm_data
from mtuq.process_data import ProcessData
from mtuq_cmaes.cmaes_init import (
    _initialize_mpi_communicator,
    _initialize_logging,
    _initialize_parameters,
    _initialize_ipop,
    _restart_ipop,
    _setup_caches,
)
from mtuq_cmaes.cmaes_mutants import (
    draw_mutants,
    gather_mutants,
    transform_mutants,
    generate_sources,
)
from mtuq_cmaes.cmaes_plotting import result_plots
from mtuq_cmaes.cmaes_utils import (
    linear_transform,
    logarithmic_transform,
    inverse_linear_transform,
    validate_inputs,
)

import matplotlib
matplotlib.use('Agg')


class CMA_ES(object):
    """
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES) class for moment tensor and force inversion.

    This class implements the CMA-ES algorithm for seismic inversion. It accepts a list of `CMAESParameters` objects containing the options and tuning of each of the inverted parameters. The inversion is carried out automatically based on the content of the `CMAESParameters` list.

    Attributes
    ----------
    rank : int
        The rank of the current MPI process.
    size : int
        The total number of MPI processes.
    comm : MPI.Comm
        The MPI communicator.
    event_id : str
        The event ID for the inversion.
    verbose_level : int
        The verbosity level for logging.
    _parameters : list
        A list of `CMAESParameters` objects.
    _parameters_names : list
        A list of parameter names.
    n : int
        The number of parameters.
    lmbda : int
        The number of mutants to be generated.
    catalog_origin : Origin
        The origin of the event to be inverted.
    callback : function
        The callback function used for the inversion.
    xmean : numpy.ndarray
        The mean of the parameter distribution.
    sigma : float
        The step size for the CMA-ES algorithm.
    iteration : int
        The current iteration number.
    counteval : int
        The number of function evaluations.
    ipop : bool
        Whether to use the IPOP method (Increasing Population CMA-ES).
    max_restart : int
        The maximum number of restarts for IPOP.
    lambda_increase_factor : float
        The factor by which to increase lambda at each restart.
    patience : int
        The number of iterations to wait before restarting under no significant improvement.
    _greens_tensors_cache : dict
        A cache for Green's tensors.
    mu : int
        The number of top-performing mutants used for recombination.
    weights : numpy.ndarray
        The weights for recombination.
    mueff : float
        The effective number of top-performing mutants.
    cs : float
        The learning rate for step-size control.
    damps : float
        The damping parameter for step-size control.
    cc : float
        The learning rate for covariance matrix adaptation.
    acov : float
        The covariance matrix adaptation parameter.
    c1 : float
        The learning rate for rank-one update of the covariance matrix.
    cmu : float
        The learning rate for rank-mu update of the covariance matrix.
    ps : numpy.ndarray
        The evolution path for step-size control.
    pc : numpy.ndarray
        The evolution path for covariance matrix adaptation.
    B : numpy.ndarray
        The matrix of eigenvectors of the covariance matrix.
    D : numpy.ndarray
        The matrix of eigenvalues of the covariance matrix.
    C : numpy.ndarray
        The covariance matrix.
    invsqrtC : numpy.ndarray
        The inverse square root of the covariance matrix.
    eigeneval : int
        The number of eigenvalue evaluations.
    chin : float
        The expected length of the evolution path.
    mutants : numpy.ndarray
        The array of mutants.
    mutants_logger_list : pandas.DataFrame
        The logger for mutants.
    mean_logger_list : pandas.DataFrame
        The logger for the mean solution.
    _misfit_holder : numpy.ndarray
        The holder for misfit values.
    fig : matplotlib.figure.Figure
        The figure object for plotting.
    ax : matplotlib.axes.Axes
        The axis object for plotting.

    Methods
    -------
    __init__(self, parameters_list, origin, lmbda=None, callback_function=None, event_id='', verbose_level=0)
        Initializes the CMA-ES class with the given parameters.
    eval_fitness(self, data, stations, misfit, db_or_greens_list, process=None, wavelet=None, verbose=False)
        Evaluates the misfit for each mutant of the population.
    _eval_fitness_db(self, data, stations, misfit, db_or_greens_list, process, wavelet)
        Helper function to evaluate fitness for 'db' mode.
    _eval_fitness_greens(self, data, stations, misfit, db_or_greens_list)
        Helper function to evaluate fitness for 'greens' mode.
    fitness_sort(self, misfit)
        Sorts the mutants by fitness and updates the misfit_holder.
    update_step_size(self)
        Updates the step size for the CMA-ES algorithm.
    update_covariance(self)
        Updates the covariance matrix for the CMA-ES algorithm.
    update_mean(self)
        Updates the mean of the parameter distribution.
    circular_mean(self, id)
        Computes the circular mean for a given parameter.
    smallestAngle(self, targetAngles, currentAngles)
        Calculates the smallest angle (in degrees) between two given sets of angles.
    mean_diff(self, new, old)
        Computes the mean change and applies circular difference for wrapped repair methods.
    create_origins(self)
        Creates a list of origins for each mutant.
    return_candidate_solution(self, id=None)
        Returns the candidate solution for a given mutant.
    _datalogger(self, mean=False)
        Logs the coordinates and misfit values of the mutants.
    _prep_and_cache_C_arrays(self, data, greens, misfit, stations)
        Prepares and caches C compatible arrays for the misfit function evaluation.
    Solve(self, data_list, stations, misfit_list, process_list, db_or_greens_list, max_iter=100, wavelet=None, plot_interval=10, iter_count=0, misfit_weights=None, **kwargs)
        Solves for the best-fitting source model using the CMA-ES algorithm.
    _transform_mutants(self)
        Transforms local mutants on each process based on the parameters scaling and projection settings.
    _generate_sources(self)
        Generates sources by calling the callback function on transformed data according to the set mode.
    _get_greens_tensors_key(self, process)
        Gets the body-wave or surface-wave key for the GreensTensors object from the ProcessData object.
    _check_greens_input_combination(self, db, process, wavelet)
        Checks the validity of the given parameters.
    _check_Solve_inputs(self, data_list, stations, misfit_list, process_list, db_or_greens_list, max_iter=100, wavelet=None, plot_interval=10, iter_count=0, **kwargs)
        Checks the validity of input arguments for the Solve method.
    _get_data_norm(self, data, misfit)
        Computes the norm of the data using the calculate_norm_data function.
    """

    def __init__(
            self, 
            parameters_list: list, 
            origin: Origin, lmbda: int = None, 
            callback_function=None, 
            event_id: str = '', 
            verbose_level: int = 0, 
            ipop: bool = False, 
            max_restart: int = 5,
            lambda_increase_factor: float = 2,
            patience: int = 20,):
        """
        Initializes the CMA-ES class with the given parameters.

        Parameters
        ----------
        parameters_list : list
            A list of `CMAESParameters` objects containing the options and tuning of each of the inverted parameters.
        origin : Origin
            The origin of the event to be inverted.
        lmbda : int, optional
            The number of mutants to be generated. If None, the default value is set to 4 + np.floor(3 * np.log(len(parameters_list))).
        callback_function : function, optional
            The callback function used for the inversion.
        event_id : str, optional
            The event ID for the inversion.
        verbose_level : int, optional
            The verbosity level for logging.
        """

        # Initialize basic properties
        _initialize_mpi_communicator(self)
        _initialize_logging(self, event_id, verbose_level)
        _initialize_parameters(self, parameters_list, lmbda, origin, callback_function)
        self.ipop = ipop
        if self.ipop:
            _initialize_ipop(self, max_restart, lambda_increase_factor, patience)
        
        # Set up caches and storage for logging
        _setup_caches(self)

    # Where the mutants are evaluated ... --------------------------------------------------------------
    def eval_fitness(self, data, stations, misfit, db_or_greens_list, process=None, wavelet=None, verbose=False):
        """
        Evaluates the misfit for each mutant of the population.

        Parameters
        ----------
        data : mtuq.Dataset
            The data to fit (body waves, surface waves).
        stations : list
            The list of stations.
        misfit : mtuq.WaveformMisfit
            The associated mtuq.Misfit object.
        db_or_greens_list : mtuq.AxiSEM_Client or mtuq.GreensTensorList
            Preprocessed Greens functions or local database (for origin search).
        process : mtuq.ProcessData, optional
            The processing function to apply to the Greens functions.
        wavelet : mtuq.wavelet, optional
            The wavelet to convolve with the Greens functions.
        verbose : bool, optional
            Whether to print debug information.

        Returns
        -------
        numpy.ndarray
            The misfit values for each mutant of the population.
        """
        # Use consistent coding style and formatting
        mode = 'db' if isinstance(db_or_greens_list, AxiSEM_Client) else 'greens'

        # Transform the mutant from [0, 10] to the correct parameter space bounds
        transform_mutants(self)
        # Generate valid sources from the transformed mutants using the callback function
        generate_sources(self)

        if mode == 'db':
            return self._eval_fitness_db(data, stations, misfit, db_or_greens_list, process, wavelet)
        elif mode == 'greens':
            return self._eval_fitness_greens(data, stations, misfit, db_or_greens_list)

    def _eval_fitness_db(self, data, stations, misfit, db_or_greens_list, process, wavelet):
        """
        Helper function to evaluate fitness for 'db' mode.

        Parameters
        ----------
        data : mtuq.Dataset
            The data to fit (body waves, surface waves).
        stations : list
            The list of stations.
        misfit : mtuq.WaveformMisfit
            The associated mtuq.Misfit object.
        db_or_greens_list : mtuq.AxiSEM_Client
            Preprocessed Greens functions or local database (for origin search).
        process : mtuq.ProcessData
            The processing function to apply to the Greens functions.
        wavelet : mtuq.wavelet
            The wavelet to convolve with the Greens functions.

        Returns
        -------
        numpy.ndarray
            The misfit values for each mutant of the population.
        """
        # Check if latitude longitude AND depth are absent from the parameters list
        if not any(x in self._parameters_names for x in ['depth', 'latitude', 'longitude']):
            # If so, use the catalog origin, and make one copy per mutant to match the number of mutants.
            if self.rank == 0 and self.verbose_level >= 1:
                print('using catalog origin')
            self.origins = [self.catalog_origin]

            key = self._get_greens_tensors_key(process)

            # Only rank 0 fetches the data from the database
            if self.rank == 0:
                if key not in self._greens_tensors_cache:
                    self._greens_tensors_cache[key] = db_or_greens_list.get_greens_tensors(stations, self.origins)
                    self._greens_tensors_cache[key].convolve(wavelet)
                    self._greens_tensors_cache[key] = self._greens_tensors_cache[key].map(process)
            else:
                self._greens_tensors_cache[key] = None

            # Rank 0 broadcasts the data to the others
            self.local_greens = self.comm.bcast(self._greens_tensors_cache[key], root=0)
            
            self.local_misfit_val = misfit(data, self.local_greens, self.sources)
            self.local_misfit_val = np.asarray(self.local_misfit_val).T
            if self.verbose_level >= 2:
                print('local misfit is :', self.local_misfit_val)  # DEBUG PRINT

            # Gather local misfit values
            self.misfit_val = self.comm.gather(self.local_misfit_val.T, root=0)
            # Broadcast the gathered values and concatenate to return across processes.
            self.misfit_val = self.comm.bcast(self.misfit_val, root=0)
            self.misfit_val = np.asarray(np.concatenate(self.misfit_val)).T
            return self.misfit_val.T
        # If one of the three is present, create a list of origins (one for each mutant), and load the corresponding local greens functions.
        else:
            if self.rank == 0 and self.verbose_level >= 1:
                print('creating new origins list')
            self.create_origins()
            if self.verbose_level >= 2:
                for X in self.origins:
                    print(X)
            # Load, convolve and process local greens function
            start_time = MPI.Wtime()
            self.local_greens = db_or_greens_list.get_greens_tensors(stations, self.origins)
            end_time = MPI.Wtime()
            if self.rank == 0:
                print('Fetching greens tensor: ' + str(end_time-start_time))
            start_time = MPI.Wtime()
            self.local_greens.convolve(wavelet)
            end_time = MPI.Wtime()
            if self.rank == 0:
                print('Convolution: ' + str(end_time-start_time))
            start_time = MPI.Wtime()
            self.local_greens = self.local_greens.map(process)
            end_time = MPI.Wtime()
            if self.rank == 0:
                print('Processing: ' + str(end_time-start_time))
            # DEBUG PRINT to check what is happening on each process: print the number of greens functions loaded on each process
            if self.verbose_level >= 2:
                print('Number of greens functions loaded on process', self.rank, ':', len(self.local_greens))

            # Compute misfit
            start_time = MPI.Wtime()
            self.local_misfit_val = [misfit(data, self.local_greens.select(origin), np.array([self.sources[_i]])) for _i, origin in enumerate(self.origins)]
            self.local_misfit_val = np.asarray(self.local_misfit_val).T[0]
            end_time = MPI.Wtime()

            if self.verbose_level >= 2:
                print('local misfit is :', self.local_misfit_val)  # DEBUG PRINT

            if self.rank == 0:
                print('Misfit: ' + str(end_time-start_time))
            # Gather local misfit values
            self.misfit_val = self.comm.gather(self.local_misfit_val.T, root=0)
            # Broadcast the gathered values and concatenate to return across processes.
            self.misfit_val = self.comm.bcast(self.misfit_val, root=0)
            self.misfit_val = np.asarray(np.concatenate(self.misfit_val))
            return self.misfit_val

    def _eval_fitness_greens(self, data, stations, misfit, db_or_greens_list):
        """
        Helper function to evaluate fitness for 'greens' mode.

        Parameters
        ----------
        data : mtuq.Dataset
            The data to fit (body waves, surface waves).
        stations : list
            The list of stations.
        misfit : mtuq.WaveformMisfit
            The associated mtuq.Misfit object.
        db_or_greens_list : mtuq.GreensTensorList
            Preprocessed Greens functions or local database (for origin search).

        Returns
        -------
        numpy.ndarray
            The misfit values for each mutant of the population.
        """
        # Check if latitude longitude AND depth are absent from the parameters list
        if not any(x in self._parameters_names for x in ['depth', 'latitude', 'longitude']):
            # If so, use the catalog origin, and make one copy per mutant to match the number of mutants.
            if self.rank == 0 and self.verbose_level >= 1:
                print('using catalog origin')
            self.local_greens = db_or_greens_list

            # Get cached arrays using misfit hash and run c_ext_L2.misfit using values from the cache
            # For some reason, the results are not the same if we use the cache from iteration 0. 
            # This is why the cache is create and used from iteration 1 onwards.
            if hasattr(self, 'data_cache'):
                if self.verbose_level >= 1:
                    print('Using cached data')
                hashkey = hash(misfit)
                cache = self.data_cache.get(hashkey)
                data_data = cache['data_data']
                greens_data = cache['greens_data']
                greens_greens = cache['greens_greens']
                groups = cache['groups']
                weights = cache['weights']
                hybrid_norm = cache['hybrid_norm']
                dt = cache['dt']
                padding = cache['padding']
                debug_level = 0
                msg_args = [0, 0, 0]
                self.local_misfit_val = c_ext_L2.misfit(data_data, greens_data, greens_greens, self.sources, groups, weights, hybrid_norm, dt, padding[0], padding[1], debug_level, *msg_args)
            else:
                if self.verbose_level >= 1:
                    print('First iteration, calculating misfit using misfit object')
                self.local_misfit_val = misfit(data, self.local_greens, self.sources)

            self.local_misfit_val = np.asarray(self.local_misfit_val).T
            if self.verbose_level >= 2:
                print('local misfit is :', self.local_misfit_val)

            # Gather local misfit values
            self.misfit_val = self.comm.gather(self.local_misfit_val.T, root=0)
            # Broadcast the gathered values and concatenate to return across processes.
            self.misfit_val = self.comm.bcast(self.misfit_val, root=0)
            self.misfit_val = np.asarray(np.concatenate(self.misfit_val)).T
            return self.misfit_val.T
        # If one of the three is present, issue a warning and break.
        else:
            if self.rank == 0:
                print('WARNING: Greens mode is not compatible with latitude, longitude or depth parameters. Consider using a local Axisem database instead.')
            return None

    def fitness_sort(self, misfit):
        """
        Sorts the mutants by fitness, and updates the misfit_holder.

        Parameters
        ----------
        misfit : array
            The misfit array to sort the mutants by. Can be the sum of body and surface wave misfits, or the misfit of a single wave type.

        Attributes
        ----------
        self.mutants : array
            The sorted mutants.
        self.transformed_mutants : array
            The sorted transformed mutants.
        self._misfit_holder : array
            The updated misfit_holder. Reset to 0 after sorting.
        """
        if self.rank == 0:
            sorted_indices = np.argsort(misfit.T)[0]
            self.mutants = self.mutants[:, sorted_indices]
            self.transformed_mutants = self.transformed_mutants[:, sorted_indices]

            # Update best solution
            if self.ipop:
                if misfit.min() < self.best_misfit:
                    self.best_misfit = misfit.min()
                    self.best_solution = self.mutants[:, 0].copy()
                    self.no_improve_counter = 0  # Reset patience counter
                else:
                    self.no_improve_counter += 1
            else:
                self.best_misfit = misfit.min()
                self.best_solution = self.mutants[:, 0].copy()

        # Broadcast the updated variables to all ranks
        self.best_misfit = self.comm.bcast(self.best_misfit, root=0)
        self.best_solution = self.comm.bcast(self.best_solution, root=0)
        if self.ipop:
            self.no_improve_counter = self.comm.bcast(self.no_improve_counter, root=0)

        self._misfit_holder *= 0

    def update_step_size(self):
        """Updates the step size for the CMA-ES algorithm."""
        if self.rank == 0:
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * self.invsqrtC @ (self.mean_diff(self.xmean, self.xold) / self.sigma)

    def update_covariance(self):
        """Updates the covariance matrix for the CMA-ES algorithm."""
        if self.rank == 0:
            ps_norm = np.linalg.norm(self.ps)
            condition = ps_norm / np.sqrt(1 - (1 - self.cs) ** (2 * (self.counteval // self.lmbda + 1))) / self.chin
            threshold = 1.4 + 2 / (self.n + 1)

            if condition < threshold:
                self.hsig = 1
            else:
                self.hsig = 0

            self.dhsig = (1 - self.hsig) * self.cc * (2 - self.cc)

            self.pc = (1 - self.cc) * self.pc + self.hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * self.mean_diff(self.xmean, self.xold) / self.sigma

            artmp = (1 / self.sigma) * self.mean_diff(self.mutants[:, 0:int(self.mu)], self.xold)
            # Old version - from the pureCMA Matlab implementation on Wikipedia
            # self.C = (1-self.c1-self.cmu) * self.C + self.c1 * (self.pc@self.pc.T + (1-self.hsig) * self.cc*(2-self.cc) * self.C) + self.cmu * artmp @ np.diag(self.weights.T[0]) @ artmp.T
            # Old version - from CMA-ES tutorial by Hansen et al. (2016) - https://arxiv.org/pdf/1604.00772.pdf
            # self.C = (1 + self.c1*self.dhsig - self.c1 - self.cmu*np.sum(self.weights)) * self.C + self.c1 * self.pc@self.pc.T + self.cmu * artmp @ np.diag(self.weights.T[0]) @ artmp.T

            # New version - from the purecma python implementation on GitHub - September, 2017, version 3.0.0
            # https://github.com/CMA-ES/pycma/blob/development/cma/purecma.py
            self.C *= 1 + self.c1 * self.dhsig - self.c1 - self.cmu * sum(self.weights)  # Discount factor
            self.C += self.c1 * self.pc @ self.pc.T  # Rank one update (pc.pc^T is a matrix of rank 1)
            self.C += self.cmu * artmp @ np.diag(self.weights.T[0]) @ artmp.T  # Rank mu update, supported by the mu best individuals

            # Adapt step size
            # We do sigma_i+1 = sigma_i * exp((cs/damps)*(||ps||/E[N(0,I)]) - 1) only now as artmp needs sigma_i
            self.sigma = self.sigma * np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chin - 1))

            if self.counteval - self.eigeneval > self.lmbda / (self.c1 + self.cmu) / self.n / 10:
                self.eigeneval = self.counteval
                self.C = np.triu(self.C) + np.triu(self.C, 1).T
                self.D, self.B = np.linalg.eig(self.C)
                self.D = np.array([self.D]).T
                self.D = np.sqrt(self.D)
                self.invsqrtC = self.B @ np.diag(self.D[:, 0]**-1) @ self.B.T

        self.iteration = self.counteval // self.lmbda

    def update_mean(self):
        """Updates the mean of the parameter distribution."""
        if self.rank == 0:
            self.xold = self.xmean.copy()
            self.xmean = np.dot(self.mutants[:, 0:len(self.weights)], self.weights)
            for _i, param in enumerate(self._parameters):
                if param.repair == 'wrapping':
                    print('computing wrapped mean for parameter:', param.name)
                    self.xmean[_i] = self.circular_mean(_i)
            
            # Update the mean datalogger
            current_mean_df = self._datalogger(mean=True)

            # If self.mean_logger_list is empty, initialize it with the current DataFrame
            if self.mean_logger_list.empty:
                self.mean_logger_list = current_mean_df
            else:
                # Concatenate the current DataFrame to the logger list
                self.mean_logger_list = pd.concat([self.mean_logger_list, current_mean_df], ignore_index=True)

    def circular_mean(self, id):
        """
        Compute the circular mean on the "id"th parameter. Ciruclar mean allows to compute mean of the samples on a periodic space.

        Parameters
        ----------
        id : int
            The index of the parameter.

        Returns
        -------
        float
            The circular mean of the parameter.
        """
        param = self.mutants[id]
        a = linear_transform(param, 0, 360) - 180
        mean = np.rad2deg(np.arctan2(np.sum(np.sin(np.deg2rad(a[range(int(self.mu))])) * self.weights.T), np.sum(np.cos(np.deg2rad(a[range(int(self.mu))])) * self.weights.T))) + 180
        mean = inverse_linear_transform(mean, 0, 360)
        return mean

    def smallestAngle(self, targetAngles, currentAngles) -> np.ndarray:
        """
        Calculates the smallest angle (in degrees) between two given sets of angles. It computes the difference between the target and current angles, making sure the result stays within the range [0, 360). If the resulting difference is more than 180, it is adjusted to go in the shorter, negative direction.

        Parameters
        ----------
        targetAngles : np.ndarray
            An array containing the target angles in degrees.
        currentAngles : np.ndarray
            An array containing the current angles in degrees.

        Returns
        -------
        np.ndarray
            An array containing the smallest difference in degrees between the target and current angles.
        """
        # Subtract the angles, constraining the value to [0, 360)
        diffs = (targetAngles - currentAngles) % 360

        # If we are more than 180 we're taking the long way around.
        # Let's instead go in the shorter, negative direction
        diffs[diffs > 180] = -(360 - diffs[diffs > 180])
        return diffs

    def mean_diff(self, new, old):
        """
        Computes the mean change and applies circular difference for wrapped repair methods.

        Parameters
        ----------
        new : np.ndarray
            The new mean values.
        old : np.ndarray
            The old mean values.

        Returns
        -------
        np.ndarray
            The mean difference.
        """
        diff = new - old
        for _i, param in enumerate(self._parameters):
            if param.repair == 'wrapping':
                angular_diff = self.smallestAngle(linear_transform(new[_i], 0, 360), linear_transform(old[_i], 0, 360))
                angular_diff = inverse_linear_transform(angular_diff, 0, 360)
                diff[_i] = angular_diff
        return diff

    def create_origins(self):
        """Creates a list of origins for each mutant."""
        # Check which of the three origin modifiers are in the parameters
        if 'depth' in self._parameters_names:
            depth = self.transformed_mutants[self._parameters_names.index('depth')]
        else:
            depth = self.catalog_origin.depth_in_m
        if 'latitude' in self._parameters_names:
            latitude = self.transformed_mutants[self._parameters_names.index('latitude')]
        else:
            latitude = self.catalog_origin.latitude
        if 'longitude' in self._parameters_names:
            longitude = self.transformed_mutants[self._parameters_names.index('longitude')]
        else:
            longitude = self.catalog_origin.longitude
        
        self.origins = []
        for i in range(len(self.scattered_mutants[0])):
            self.origins += [self.catalog_origin.copy()]
            if 'depth' in self._parameters_names:
                setattr(self.origins[-1], 'depth_in_m', depth[i])
            if 'latitude' in self._parameters_names:
                setattr(self.origins[-1], 'latitude', latitude[i])
            if 'longitude' in self._parameters_names:
                setattr(self.origins[-1], 'longitude', longitude[i])

    def return_candidate_solution(self, id=None):
        """
        Returns the candidate solution for a given mutant.

        Parameters
        ----------
        id : int, optional
            The index of the mutant. If None, returns the mean solution.

        Returns
        -------
        tuple
            The transformed mean and the origins.
        """
        # Only required on rank 0
        if self.rank == 0:
            if not id == None:
                return_x = np.array([self.mutants[:, id]]).T
            else:
                return_x = np.asarray([self.best_solution.copy()]).T
            self.transformed_mean = np.zeros_like(return_x)
            for _i, param in enumerate(self._parameters):
                # Print paramter scaling if verbose
                if param.scaling == 'linear':
                    self.transformed_mean[_i] = linear_transform(return_x[_i], param.lower_bound, param.upper_bound)
                elif param.scaling == 'log':
                    self.transformed_mean[_i] = logarithmic_transform(return_x[_i], param.lower_bound, param.upper_bound)
                else:
                    raise ValueError('Unrecognized scaling, must be linear or log')
                # Apply optional projection operator to each parameter
                if not param.projection is None:
                    self.transformed_mean[_i] = np.asarray(list(map(param.projection, self.transformed_mean[_i])))

            # Check which of the three origin modifiers are in the parameters
            if 'depth' in self._parameters_names:
                depth = self.transformed_mean[self._parameters_names.index('depth')]
            else:
                depth = self.catalog_origin.depth_in_m
            if 'latitude' in self._parameters_names:
                latitude = self.transformed_mean[self._parameters_names.index('latitude')]
            else:
                latitude = self.catalog_origin.latitude
            if 'longitude' in self._parameters_names:
                longitude = self.transformed_mean[self._parameters_names.index('longitude')]
            else:
                longitude = self.catalog_origin.longitude
            
            self.origins = []
            for i in range(len(self.transformed_mean[0])):
                self.origins += [self.catalog_origin.copy()]
                if 'depth' in self._parameters_names:
                    setattr(self.origins[-1], 'depth_in_m', depth[i])
                if 'latitude' in self._parameters_names:
                    setattr(self.origins[-1], 'latitude', latitude[i])
                if 'longitude' in self._parameters_names:
                    setattr(self.origins[-1], 'longitude', longitude[i])

            return self.transformed_mean, self.origins

    def _datalogger(self, mean=False):
        """
        Logs the coordinates and misfit values of the mutants or the mean solution.

        Parameters
        ----------
        mean : bool, optional
            If True, logs the mean coordinates. Default is False.

        Returns
        -------
        MTUQDataFrame
            The logged data.
        """
        if self.rank == 0:
            if not mean:
                coordinates = self.transformed_mutants.T
                misfit_values = self._misfit_holder
                results = np.hstack((coordinates, misfit_values))
                columns_labels = self._parameters_names + [0]
                da = pd.DataFrame(
                    data=results,
                    columns=columns_labels
                )
                return MTUQDataFrame(da)

            if mean:
                transformed_mean = np.zeros_like(self.xmean)
                for _i, param in enumerate(self._parameters):
                    if param.scaling == 'linear':
                        transformed_mean[_i] = linear_transform(self.xmean[_i], param.lower_bound, param.upper_bound)
                    elif param.scaling == 'log':
                        transformed_mean[_i] = logarithmic_transform(self.xmean[_i], param.lower_bound, param.upper_bound)
                    else:
                        raise ValueError('Unrecognized scaling, must be linear or log')
                    # Apply optional projection operator to each parameter
                    if not param.projection is None:
                        transformed_mean[_i] = np.asarray(list(map(param.projection, transformed_mean[_i])))

                # Include restart information
                results = transformed_mean.T
                columns_labels = self._parameters_names

                # Add a 'restart' column
                df = pd.DataFrame(
                    data=results,
                    columns=columns_labels
                )
                if self.ipop:
                    df['restart'] = self.current_restarts  # Tag with current restart number

                return MTUQDataFrame(df)


    def _prep_and_cache_C_arrays(self, data, greens, misfit, stations):
        """
        Prepares and caches C compatible arrays for the misfit function evaluation.

        It is responsible for preparing the data arrays for the inversion, in a format expected by the lower-level 
        c-code for misfit evaluation. Mostly copy-pasted from mtuq.misfit.waveform.level2

        Only used when the mode is 'greens'.

        Parameters
        ----------
        data : mtuq.Dataset
            The data to fit (body waves, surface waves).
        greens : mtuq.GreensTensorList
            Preprocessed Greens functions or local database (for origin search).
        misfit : mtuq.WaveformMisfit
            The associated mtuq.Misfit object.
        stations : list
            The list of stations.
        """
        from mtuq.misfit.waveform.level2 import _get_time_sampling, _get_stations, _get_components, _get_weights, _get_groups, _get_data, _get_greens, _get_padding, _autocorr_1, _autocorr_2, _corr_1_2
        from mtuq.util import Null

        msg_handle = Null()
        # If no attributes are present, create the dictionaries
        if not hasattr(self, 'data_cache'):
            self.data_cache = {}

        # Use the misfit object __hash__ method to create a unique key for the data_cache dictionary.
        key = hash(misfit)

        # If the key is not present in the data_cache, prepare the data arrays for the inversion before caching them.
        if key not in self.data_cache:

            # Precompute the data arrays for the inversion before caching them.
            nt, dt = _get_time_sampling(data)
            stations = _get_stations(data)
            components = _get_components(data)

            weights = _get_weights(data, stations, components)

            # which components will be used to determine time shifts (boolean array)?
            groups = _get_groups(misfit.time_shift_groups, components)

            # Set include_mt and include_force based on the mode
            if self.mode in ['mt', 'mt_dc', 'mt_dev']:
                for g in greens:
                    g.include_mt = True
                    g.include_force = False
            elif self.mode == 'force':
                for g in greens:
                    g.include_mt = False
                    g.include_force = True
            else:
                raise ValueError('Invalid mode. Supported modes: "mt", "mt_dc", "mt_dev", "force".')

            # collapse main structures into NumPy arrays
            data = _get_data(data, stations, components)
            greens = _get_greens(greens, stations, components)

            # cross-correlate data and synthetics
            padding = _get_padding(misfit.time_shift_min, misfit.time_shift_max, dt)
            data_data = _autocorr_1(data)
            greens_greens = _autocorr_2(greens, padding)
            greens_data = _corr_1_2(data, greens, padding)

            if misfit.norm == 'hybrid':
                hybrid_norm = 1
            else:
                hybrid_norm = 0

            # collect message attributes
            try:
                msg_args = [getattr(msg_handle, attrib) for attrib in ['start', 'stop', 'percent']]
            except:
                msg_args = [0, 0, 0]

            # Cache the data arrays for the inversion to be called by c_ext_L2.misfit(data_data, greens_data, greens_greens, sources, groups, weights, hybrid_norm, dt, padding[0], padding[1], debug_level, *msg_args)
            self.data_cache[key] = {
                'data_data': data_data,
                'greens_data': greens_data,
                'greens_greens': greens_greens,
                'groups': groups,
                'weights': weights,
                'hybrid_norm': hybrid_norm,
                'dt': dt,
                'padding': padding,
                'msg_args': msg_args
            }
        
        elif key in self.data_cache:
            if self.verbose_level >= 2:
                print('Data arrays already cached. Nothing to do here.')
            pass

    def Solve(self, data_list, stations, misfit_list, process_list, db_or_greens_list, max_iter=100, wavelet=None, plot_interval=10, iter_count=0, misfit_weights=None, **kwargs):
        """
        Solves for the best-fitting source model using the CMA-ES algorithm. This is the master method used in inversions. 

        This method iteratively draws mutants, evaluates their fitness based on misfits between synthetic and observed data, and updates the mean and covariance of the CMA-ES distribution. At specified intervals, it also plots mean waveforms and results for visualization.

        Parameters
        ----------
        data_list : list
            List of observed data sets. (e.g. [data_sw] or [data_bw, data_sw])
        stations : list
            List of stations (generally obtained from mtuq method data.get_stations())
        misfit_list : list
            List of mtuq misfit objects (e.g. [misfit_sw] or [misfit_bw, misfit_sw]).
        process_list : list
            List of mtuq ProcessData objects to apply to data (e.g. [process_sw] or [process_bw, process_sw]).
        db_or_greens_list : list or AxiSEM_Client object
            Either an AxiSEM database client or a mtuq GreensTensorList.
        max_iter : int, optional
            Maximum number of iterations to perform. Default is 100. A stoping criterion will be implemented in the future.
        wavelet : object, optional
            Wavelet for source time function. Default is None. Required when db_or_greens_list is an AxiSEM database client.
        plot_interval : int, optional
            Interval at which plots of mean waveforms and results should be generated. Default is 10.
        iter_count : int, optional
            Current iteration count, should be useful for resuming. Default is 0.
        src_type : str, optional
            Type of source model, one of ['full', 'deviatoric', 'dc']. Default is full.
        misfit_weights : list, optional
            List of misfit weights. Default is None for equal weights.
        **kwargs
            Additional keyword arguments passed to eval_fitness method.

        Returns
        -------
        None

        Note
        ----
        This method is the wrapper that automate the execution of the CMA-ES algorithm. It is the default workflow for Moment tensor and Force inversion and should not work with a "custom" inversion (multiple-sub events, source time function, etc.). It interacts with the  `draw_mutants`, `eval_fitness`, `gather_mutants`, `fitness_sort`, `update_mean`, `update_step_size` and `update_covariance`. 
        """
        greens_cache = {}
        data_cache = {}

        if self.rank == 0:
            # Check Solve inputs
            validate_inputs(data_list, stations, misfit_list, process_list, db_or_greens_list, max_iter, wavelet, plot_interval, iter_count)

        # Handling of the misfit weights. If not provided, equal weights are used, otherwise the weights are used to derive percentages.
        if misfit_weights is None:
            misfit_weights = [1.0] * len(data_list)
        elif len(misfit_weights) != len(data_list):
            raise ValueError('Length of misfit_weights must match the length of data_list.')

        total_weight = sum(misfit_weights)
        if total_weight == 0:
            raise ValueError('Sum of weights cannot be zero.')

        misfit_weights = [w / total_weight for w in misfit_weights]

        iteration = 0
        while iteration < max_iter:
            if self.rank == 0:
                print('Iteration %d\n' % (iteration + iter_count + 1))
            
            draw_mutants(self)

            misfits = []
            for j, (current_data, current_misfit, process) in enumerate(zip(data_list, misfit_list, process_list)):
                mode = 'db' if isinstance(db_or_greens_list, AxiSEM_Client) else 'greens'
                # get greens[j] or db depending on mode from db_or_greens_list
                greens = db_or_greens_list[j] if mode == 'greens' else None
                db = db_or_greens_list if mode == 'db' else None

                if mode == 'db':
                    misfit_values = self.eval_fitness(current_data, stations, current_misfit, db, process_list[j], wavelet, **kwargs)
                elif mode == 'greens':
                    if iteration == 1 and type(current_misfit) == WaveformMisfit:
                        raw_greens_to_cache = copy.deepcopy(greens)
                        raw_data_to_cache = copy.deepcopy(current_data)
                        self._prep_and_cache_C_arrays(raw_data_to_cache, raw_greens_to_cache, current_misfit, stations)
                    misfit_values = self.eval_fitness(current_data, stations, current_misfit, greens, **kwargs)

                norm = self._get_data_norm(current_data, current_misfit)
                misfit_values /= norm
                misfits.append(misfit_values)

            weighted_misfits = [w * m for w, m in zip(misfit_weights, misfits)]
            total_missfit = sum(weighted_misfits)
            self._misfit_holder += total_missfit
            gather_mutants(self)
            self.fitness_sort(total_missfit)
            self.update_mean()
            self.update_step_size()
            self.update_covariance()

            if self.ipop:
                # Condition to trigger a restart:
                # - Either no improvement over 'patience' iterations
                # - Or reached the maximum number of iterations
                restart_condition = self.no_improve_counter >= self.patience or iteration >= max_iter - 1

                if restart_condition and self.current_restarts < self.max_restarts:
                    if self.rank == 0:
                        reason = "no improvement" if self.no_improve_counter >= self.patience else "reached max_iter"
                        print(f"No improvement ({reason}). Initiating IPOP restart #{self.current_restarts + 1}.")
                    _restart_ipop(self)
                    iteration = 0  # Reset iteration counter for the new run
                    continue  # Skip plotting in this iteration

                elif restart_condition and self.current_restarts >= self.max_restarts:
                    if self.rank == 0:
                        print(f"No improvement and reached maximum number of IPOP restarts ({self.max_restarts}). Terminating optimization.")
                        self.iteration = len(self.mean_logger_list)
                        self.ipop_terminated = True
                    break


            result_plots(self, data_list, stations, misfit_list, process_list, db_or_greens_list, max_iter, plot_interval, iter_count, iteration)
            
            iteration += 1
        # Trigger a final plotting at the end of the optimization
        result_plots(self, data_list, stations, misfit_list, process_list, db_or_greens_list, max_iter, plot_interval, iter_count, iteration)

    def _get_greens_tensors_key(self, process):
        """
        Gets the body-wave or surface-wave key for the GreensTensors object from the ProcessData object.

        Parameters
        ----------
        process : ProcessData
            The ProcessData object.

        Returns
        -------
        str
            The body-wave or surface-wave key.
        """
        return process.window_type

    def _check_greens_input_combination(self, db, process, wavelet):
        """
        Checks the validity of the given parameters.

        Raises a ValueError if the database object is not an AxiSEM_Client or GreensTensorList, 
        or if the process function and wavelet are not defined when the database object is an AxiSEM_Client.

        Parameters
        ----------
        db : AxiSEM_Client or GreensTensorList
            The database object to check, expected to be an instance of either AxiSEM_Client or GreensTensorList.
        process : ProcessData
            The process function to be used if the database is an AxiSEM_Client.
        wavelet : mtuq.wavelet
            The wavelet to be used if the database is an AxiSEM_Client.

        Raises
        ------
        ValueError
            If the input combination of db, process, and wavelet is invalid.
        """
        if not isinstance(db, (AxiSEM_Client, GreensTensorList)):
            raise ValueError('database must be either an AxiSEM_Client object or a GreensTensorList object')
        if isinstance(db, AxiSEM_Client) and (process is None or wavelet is None):
            raise ValueError('process_function and wavelet must be specified if database is an AxiSEM_Client')

    def _check_Solve_inputs(self, data_list, stations, misfit_list, process_list, db_or_greens_list, max_iter=100, wavelet=None, plot_interval=10, iter_count=0, **kwargs):
        """
        Checks the validity of input arguments for the Solve method.

        Parameters
        ----------
        data_list : list
            List of observed data sets. (e.g. [data_sw] or [data_bw, data_sw])
        stations : list
            List of stations (generally obtained from mtuq method data.get_stations())
        misfit_list : list
            List of mtuq misfit objects (e.g. [misfit_sw] or [misfit_bw, misfit_sw]).
        process_list : list
            List of mtuq ProcessData objects to apply to data (e.g. [process_sw] or [process_bw, process_sw]).
        db_or_greens_list : list or AxiSEM_Client object
            Either an AxiSEM database client or a mtuq GreensTensorList.
        max_iter : int, optional
            Maximum number of iterations to perform. Default is 100. A stoping criterion will be implemented in the future.
        wavelet : object, optional
            Wavelet for source time function. Default is None. Required when db_or_greens_list is an AxiSEM database client.
        plot_interval : int, optional
            Interval at which plots of mean waveforms and results should be generated. Default is 10.
        iter_count : int, optional
            Current iteration count, should be useful for resuming. Default is 0.
        **kwargs
            Additional keyword arguments passed to eval_fitness method.

        Raises
        ------
        ValueError
            If any of the inputs are invalid.
        """
        if not isinstance(data_list, list):
            if isinstance(data_list, Dataset):
                data_list = [data_list]
            else:
                raise ValueError('`data_list` should be a list of mtuq Dataset or an array containing polarities.')
        if not isinstance(stations, list):
            raise ValueError('`stations` should be a list of mtuq Station objects.')
        if not isinstance(misfit_list, list):
            if isinstance(misfit_list, PolarityMisfit) or isinstance(misfit_list, Misfit):
                misfit_list = [misfit_list]
            else:
                raise ValueError('`misfit_list` should be a list of mtuq Misfit objects.')
        if not isinstance(process_list, list):
            if isinstance(process_list, ProcessData):
                process_list = [process_list]
            else:
                raise ValueError('`process_list` should be a list of mtuq ProcessData objects.')
        if not isinstance(db_or_greens_list, list):
            if isinstance(db_or_greens_list, AxiSEM_Client) or isinstance(db_or_greens_list, GreensTensorList):
                db_or_greens_list = [db_or_greens_list]
            else:
                raise ValueError('`db_or_greens_list` should be a list of either mtuq AxiSEM_Client or GreensTensorList objects.')
        if not isinstance(max_iter, int):
            raise ValueError('`max_iter` should be an integer.')
        if any(isinstance(db, AxiSEM_Client) for db in db_or_greens_list) and wavelet is None:
            raise ValueError('wavelet must be specified if database is an AxiSEM_Client')
        if not isinstance(plot_interval, int):
            raise ValueError('`plot_interval` should be an integer.')
        if iter_count is not None and not isinstance(iter_count, int):
            raise ValueError('`iter_count` should be an integer or None.')

    def _get_data_norm(self, data, misfit):
        """
        Computes the norm of the data using the calculate_norm_data function.

        Parameters
        ----------
        data : mtuq.Dataset
            The evaluated processed data.
        misfit : mtuq.Misfit
            The misfit object used to evaluate the data object.

        Returns
        -------
        float
            The norm of the data.
        """
        # If misfit type is Polarity misfit, use the sum of the absolute values of the data as number of used stations.
        if isinstance(misfit, PolarityMisfit):
            return np.sum(np.abs(data))
        # Else, use the calculate_norm_data function.
        else:
            if isinstance(misfit.time_shift_groups, str):
                components = list(misfit.time_shift_groups)
            elif isinstance(misfit.time_shift_groups, list):
                components = list(''.join(misfit.time_shift_groups))
            
            return calculate_norm_data(data, misfit.norm, components)