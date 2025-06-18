import numpy as np
from mpi4py import MPI
import pandas as pd
from mtuq.event import Origin
from mtuq.util.math import to_mij, to_rtp

def _initialize_mpi_communicator(cmaes_instance):
    """Initializes the MPI communicator and sets the process rank and size."""
    cmaes_instance.rank = 0
    cmaes_instance.size = 1
    cmaes_instance.comm = MPI.COMM_WORLD
    cmaes_instance.rank = cmaes_instance.comm.Get_rank()
    cmaes_instance.size = cmaes_instance.comm.Get_size()

def _initialize_logging(cmaes_instance, event_id, verbose_level):
    """Sets up logging properties like event_id and verbose level."""
    cmaes_instance.event_id = event_id
    cmaes_instance.verbose_level = verbose_level
    if cmaes_instance.rank == 0:
        print(f'Initializing CMA-ES inversion for event {cmaes_instance.event_id}')

def _initialize_parameters(cmaes_instance, parameters_list, lmbda, origin, callback_function):
    """
    Initializes the parameters for the CMA-ES algorithm, including population size, callback function,
    and variables related to step-size control and covariance matrix adaptation.

    Parameters
    ----------
    parameters_list : list
        A list of `CMAESParameters` objects containing the options and tuning of each of the inverted parameters.
    lmbda : int, optional
        The number of mutants to be generated. If None, the default value is set to 4 + np.floor(3 * np.log(len(parameters_list))).
    origin : Origin
        The origin of the event to be inverted.
    callback_function : function, optional
        The callback function used for the inversion.
    """
    cmaes_instance._parameters = parameters_list
    cmaes_instance._parameters_names = [parameter.name for parameter in parameters_list]
    cmaes_instance.n = len(cmaes_instance._parameters)

    # Set the default number of mutants (lambda) if not specified
    if lmbda is None:
        cmaes_instance.lmbda = int(4 + np.floor(3 * np.log(cmaes_instance.n)))
    else:
        cmaes_instance.lmbda = lmbda

    # Initialize the misfit holder which will store the misfit values for each mutant for a given iteration
    cmaes_instance._misfit_holder = np.zeros((int(cmaes_instance.lmbda), 1))

    # Ensure that the number of MPI processes is not greater than the population size
    if cmaes_instance.size > cmaes_instance.lmbda and cmaes_instance.rank == 0:
        import warnings
        warnings.warn(f'Number of MPI processes ({cmaes_instance.size}) exceeds population size ({cmaes_instance.lmbda}). Some processes may be idle.')

    # Validate the origin parameter
    if not isinstance(origin, Origin):
        raise ValueError('The "origin" parameter must be an instance of mtuq.event.Origin. Please provide a valid object to be used as catalog origin.')

    # Set the origin for the inversion
    cmaes_instance.catalog_origin = origin

    # Handle callback function initialization
    cmaes_instance.callback = callback_function
    if cmaes_instance.callback is None:
        _set_default_callback(cmaes_instance)

    # Initialize parameters tied to the CMA-ES algorithm
    cmaes_instance.xmean = np.asarray([[param.initial for param in cmaes_instance._parameters]]).T
    cmaes_instance.sigma = 1.67  # Default initial Gaussian variance for all parameters
    cmaes_instance.iteration = 0
    cmaes_instance.counteval = 0
    cmaes_instance._greens_tensors_cache = {}

    # Weight initialization for recombination
    cmaes_instance.mu = np.floor(cmaes_instance.lmbda / 2)
    a = 1  # Use 1/2 in tutorial and 1 in publication
    cmaes_instance.weights = np.array([np.log(cmaes_instance.mu + a) - np.log(np.arange(1, cmaes_instance.mu + 1))]).T
    cmaes_instance.weights /= sum(cmaes_instance.weights)
    cmaes_instance.mueff = sum(cmaes_instance.weights)**2 / sum(cmaes_instance.weights**2)

    # Step-size control parameters
    cmaes_instance.cs = (cmaes_instance.mueff + 2) / (cmaes_instance.n + cmaes_instance.mueff + 5)
    cmaes_instance.damps = 1 + 2 * max(0, np.sqrt((cmaes_instance.mueff - 1) / (cmaes_instance.n + 1)) - 1) + cmaes_instance.cs

    # Covariance matrix adaptation parameters
    cmaes_instance.cc = (4 + cmaes_instance.mueff / cmaes_instance.n) / (cmaes_instance.n + 4 + 2 * cmaes_instance.mueff / cmaes_instance.n)
    cmaes_instance.acov = 2
    cmaes_instance.c1 = cmaes_instance.acov / ((cmaes_instance.n + 1.3)**2 + cmaes_instance.mueff)
    cmaes_instance.cmu = min(1 - cmaes_instance.c1, cmaes_instance.acov * (cmaes_instance.mueff - 2 + 1 / cmaes_instance.mueff) / ((cmaes_instance.n + 2)**2 + cmaes_instance.acov * cmaes_instance.mueff / 2))

    # Initialize step-size and covariance-related variables
    cmaes_instance.ps = np.zeros_like(cmaes_instance.xmean)
    cmaes_instance.pc = np.zeros_like(cmaes_instance.xmean)
    cmaes_instance.B = np.eye(cmaes_instance.n, cmaes_instance.n)
    cmaes_instance.D = np.ones((cmaes_instance.n, 1))
    cmaes_instance.C = cmaes_instance.B @ np.diag(cmaes_instance.D[:, 0]**2) @ cmaes_instance.B.T
    cmaes_instance.invsqrtC = cmaes_instance.B @ np.diag(cmaes_instance.D[:, 0]**-1) @ cmaes_instance.B.T
    cmaes_instance.eigeneval = 0
    cmaes_instance.chin = cmaes_instance.n**0.5 * (1 - 1 / (4 * cmaes_instance.n) + 1 / (21 * cmaes_instance.n**2))
    cmaes_instance.mutants = np.zeros((cmaes_instance.n, cmaes_instance.lmbda))

def _initialize_ipop(cmaes_instance, max_restarts: int = 10, lambda_increase_factor: float = 2.0, patience: int = 20):
    """
    Initializes the IPOP (Increasing Population) strategy for the CMA-ES algorithm.

    Parameters
    ----------
    max_restarts : int, optional
        Maximum number of restarts allowed. Default is 10.
    lambda_increase_factor : float, optional
        Factor by which to increase lambda upon each restart. Default is 2.0.
    patience : int, optional
        Number of iterations to wait without improvement before restarting. Default is 20.
    """
    cmaes_instance.ipop = True
    cmaes_instance.ipop_terminated = False
    cmaes_instance.max_restarts = max_restarts
    cmaes_instance.lambda_increase_factor = lambda_increase_factor
    cmaes_instance.patience = patience
    cmaes_instance.current_restarts = 0
    cmaes_instance.no_improve_counter = 0
    cmaes_instance.best_misfit = np.inf
    cmaes_instance.best_solution = None
    cmaes_instance.best_origins = None

    if cmaes_instance.rank == 0:
        print("IPOP strategy initialized with the following parameters:")
        print(f"  Max Restarts: {cmaes_instance.max_restarts}")
        print(f"  Lambda Increase Factor: {cmaes_instance.lambda_increase_factor}")
        print(f"  Patience: {cmaes_instance.patience} iterations without improvement")

def _restart_ipop(cmaes_instance):
    """
    Handles the IPOP restart by increasing the population size and reinitializing necessary attributes.
    Ensures synchronization across all MPI ranks.
    """
    # Only rank 0 manages the restart logic
    if cmaes_instance.rank == 0:
        if cmaes_instance.current_restarts >= cmaes_instance.max_restarts:
            print(f"Rank {cmaes_instance.rank}: Maximum number of restarts ({cmaes_instance.max_restarts}) reached. Terminating optimization.")
            cmaes_instance.ipop_terminated = True
        else:
            # Increase lambda by the specified factor
            new_lambda = int(cmaes_instance.lmbda * cmaes_instance.lambda_increase_factor)
            print(f"Rank {cmaes_instance.rank}: Restarting CMA-ES with increased population size: {new_lambda}")
            cmaes_instance.lmbda = new_lambda
            cmaes_instance.current_restarts += 1
    else:
        # Initialize variables on non-zero ranks
        new_lambda = None
        cmaes_instance.ipop_terminated = None
        current_restarts = None

    # Broadcast the termination flag
    try:
        cmaes_instance.ipop_terminated = cmaes_instance.comm.bcast(cmaes_instance.ipop_terminated, root=0)
        if cmaes_instance.verbose_level >= 1: print(f"Rank {cmaes_instance.rank}: Received ipop_terminated={cmaes_instance.ipop_terminated}")
    except Exception as e:
        print(f"Rank {cmaes_instance.rank}: Error during bcast of ipop_terminated: {e}")
        cmaes_instance.comm.Abort(1)

    # Broadcast the new_lambda
    try:
        new_lambda = cmaes_instance.comm.bcast(cmaes_instance.lmbda if cmaes_instance.rank == 0 else None, root=0)
        if cmaes_instance.verbose_level >= 1: print(f"Rank {cmaes_instance.rank}: Received lambda={new_lambda}")
    except Exception as e:
        print(f"Rank {cmaes_instance.rank}: Error during bcast of lambda: {e}")
        cmaes_instance.comm.Abort(1)

    # Broadcast the current_restarts
    try:
        cmaes_instance.current_restarts = cmaes_instance.comm.bcast(cmaes_instance.current_restarts if cmaes_instance.rank == 0 else None, root=0)
        if cmaes_instance.verbose_level >= 1: print(f"Rank {cmaes_instance.rank}: Received current_restarts={cmaes_instance.current_restarts}")
    except Exception as e:
        print(f"Rank {cmaes_instance.rank}: Error during bcast of current_restarts: {e}")
        cmaes_instance.comm.Abort(1)

    # If termination flag is set, do not proceed with restart
    if cmaes_instance.ipop_terminated:
        print(f"Rank {cmaes_instance.rank}: Terminating optimization due to maximum restarts.")
        return

    # Update lambda on all ranks
    cmaes_instance.lmbda = new_lambda

    # Reinitialize misfit_holder based on the new lambda
    cmaes_instance._misfit_holder = np.zeros((cmaes_instance.lmbda, 1))
    if cmaes_instance.verbose_level >= 1: print(f"Rank {cmaes_instance.rank}: Reinitialized _misfit_holder with shape {cmaes_instance._misfit_holder.shape}")

    # Reinitialize CMA-ES parameters
    cmaes_instance.mutants = np.zeros((cmaes_instance.n, cmaes_instance.lmbda))
    cmaes_instance.transformed_mutants = np.zeros((cmaes_instance.n, cmaes_instance.lmbda))
    if cmaes_instance.verbose_level >= 1: print(f"Rank {cmaes_instance.rank}: Reinitialized mutants and transformed_mutants with shape ({cmaes_instance.n}, {cmaes_instance.lmbda})")

    # Reset step-size and covariance-related variables
    restart_from_best = False  # Set to True if you want to restart from the best solution
    if restart_from_best:
        cmaes_instance.xmean = np.asarray([cmaes_instance.best_solution.copy()]).T
        if cmaes_instance.verbose_level >= 1: print(f"Rank {cmaes_instance.rank}: Restarting from best_solution.")
    else:
        # Restart at random within the parameter bounds (assuming [0, 10] scaling as per your code)
        cmaes_instance.xmean = np.random.uniform(0, 10, (cmaes_instance.n, 1))
        if cmaes_instance.verbose_level >= 1: print(f"Rank {cmaes_instance.rank}: Restarting with random xmean.")

    cmaes_instance.sigma = 1.67  # Adjust based on your problem
    cmaes_instance.B = np.eye(cmaes_instance.n)
    cmaes_instance.D = np.ones((cmaes_instance.n, 1))
    cmaes_instance.C = cmaes_instance.B @ np.diag(cmaes_instance.D[:, 0]**2) @ cmaes_instance.B.T
    cmaes_instance.invsqrtC = cmaes_instance.B @ np.diag(cmaes_instance.D[:, 0]**-1) @ cmaes_instance.B.T
    cmaes_instance.ps = np.zeros_like(cmaes_instance.xmean)
    cmaes_instance.pc = np.zeros_like(cmaes_instance.xmean)
    cmaes_instance.eigeneval = 0
    if cmaes_instance.verbose_level >= 1: print(f"Rank {cmaes_instance.rank}: Reset sigma, B, D, C, invsqrtC, ps, pc, and eigenval.")

    # Update weights and related parameters based on new lambda
    cmaes_instance.mu = int(np.floor(cmaes_instance.lmbda / 2))
    a = 1  # Use 1 in publication
    cmaes_instance.weights = np.log(cmaes_instance.mu + a) - np.log(np.arange(1, cmaes_instance.mu + 1))
    cmaes_instance.weights /= np.sum(cmaes_instance.weights)
    cmaes_instance.weights = cmaes_instance.weights.reshape(-1, 1)  # Ensure column vector
    cmaes_instance.mueff = (np.sum(cmaes_instance.weights))**2 / np.sum(cmaes_instance.weights**2)
    if cmaes_instance.verbose_level >= 1: print(f"Rank {cmaes_instance.rank}: Updated weights and mueff.")

    # Step-size control parameters
    cmaes_instance.cs = (cmaes_instance.mueff + 2) / (cmaes_instance.n + cmaes_instance.mueff + 5)
    cmaes_instance.damps = 1 + 2 * max(0, np.sqrt((cmaes_instance.mueff - 1) / (cmaes_instance.n + 1)) - 1) + cmaes_instance.cs
    if cmaes_instance.verbose_level >= 1: print(f"Rank {cmaes_instance.rank}: Updated cs and damps.")

    # Covariance matrix adaptation parameters
    cmaes_instance.cc = (4 + cmaes_instance.mueff / cmaes_instance.n) / (cmaes_instance.n + 4 + 2 * cmaes_instance.mueff / cmaes_instance.n)
    cmaes_instance.acov = 2
    cmaes_instance.c1 = cmaes_instance.acov / ((cmaes_instance.n + 1.3)**2 + cmaes_instance.mueff)
    cmaes_instance.cmu = min(
        1 - cmaes_instance.c1,
        cmaes_instance.acov * (cmaes_instance.mueff - 2 + 1 / cmaes_instance.mueff) / ((cmaes_instance.n + 2)**2 + cmaes_instance.acov * cmaes_instance.mueff / 2)
    )
    if cmaes_instance.verbose_level >= 1: print(f"Rank {cmaes_instance.rank}: Updated cc, c1, and cmu.")

    # Reset patience counter
    cmaes_instance.no_improve_counter = 0
    if cmaes_instance.verbose_level >= 1: print(f"Rank {cmaes_instance.rank}: Reset no_improve_counter.")

    # Synchronize all ranks before proceeding
    try:
        if cmaes_instance.verbose_level >= 1: print(f"Rank {cmaes_instance.rank}: Waiting at Barrier.")
        cmaes_instance.comm.Barrier()
        if cmaes_instance.verbose_level >= 1: print(f"Rank {cmaes_instance.rank}: Passed Barrier. Restart complete.")
    except Exception as e:
        if cmaes_instance.verbose_level >= 1: print(f"Rank {cmaes_instance.rank}: Error during Barrier: {e}")
        cmaes_instance.comm.Abort(1)

def _set_default_callback(cmaes_instance):
    """Sets the default callback function based on the parameter names."""
    if 'Mw' in cmaes_instance._parameters_names or 'kappa' in cmaes_instance._parameters_names:
        cmaes_instance.callback = to_mij
        cmaes_instance.mij_args = ['rho', 'v', 'w', 'kappa', 'sigma', 'h']
        cmaes_instance.mode = 'mt'
        if 'w' not in cmaes_instance._parameters_names and 'v' in cmaes_instance._parameters_names:
            cmaes_instance.callback = to_mij
            cmaes_instance.mode = 'mt_dev'
            cmaes_instance.mij_args = ['rho', 'w', 'kappa', 'sigma', 'h']
        elif 'v' not in cmaes_instance._parameters_names and 'w' not in cmaes_instance._parameters_names:
            cmaes_instance.callback = to_mij
            cmaes_instance.mode = 'mt_dc'
            cmaes_instance.mij_args = ['rho', 'kappa', 'sigma', 'h']
    elif 'F0' in cmaes_instance._parameters_names:
        cmaes_instance.callback = to_rtp
        cmaes_instance.mode = 'force'

def _setup_caches(cmaes_instance):
    """Initializes pandas DataFrames for logging. Used for post-processing and plotting."""
    cmaes_instance.mutants_logger_list = pd.DataFrame()
    cmaes_instance.mean_logger_list = pd.DataFrame()

    # Define holder variables for plotting
    cmaes_instance.fig = None
    cmaes_instance.ax = None