import numpy as np
import pandas as pd
from mtuq_cmaes.cmaes_utils import array_in_bounds, in_bounds, Repair, linear_transform, logarithmic_transform


def generate_mutants(cmaes_instance):
    """
    Generates all mutants from a Gaussian distribution.
    """
    for i in range(cmaes_instance.lmbda):
        mutant = draw_single_mutant(cmaes_instance)
        cmaes_instance.mutants[:, i] = mutant.flatten()

def draw_single_mutant(cmaes_instance):
    """
    Draws a single mutant from the Gaussian distribution.
    """
    return cmaes_instance.xmean + cmaes_instance.sigma * cmaes_instance.B @ (cmaes_instance.D * np.random.randn(cmaes_instance.n, 1))

def draw_mutants(cmaes_instance):
    """
    Draws mutants from a Gaussian distribution and scatters them across MPI processes.
    """
    if cmaes_instance.rank == 0:
        generate_mutants(cmaes_instance)
        repair_and_redraw_mutants(cmaes_instance)
        scatter_mutants(cmaes_instance)
    else:
        receive_mutants(cmaes_instance)
    
    cmaes_instance.scattered_mutants = cmaes_instance.mutant_slice
    cmaes_instance.counteval += cmaes_instance.lmbda

def repair_and_redraw_mutants(cmaes_instance):
    """
    Applies repair methods and redraws to all mutants, parameter by parameter.
    """
    bounds = [0, 10]  # Define the hardcoded bounds
    redraw_counter = 0
    for param_idx in range(cmaes_instance.n):
        param_values = cmaes_instance.mutants[param_idx, :]
        if array_in_bounds(param_values, bounds[0], bounds[1]):
            continue
        if cmaes_instance._parameters[param_idx].repair is None:
            cmaes_instance.mutants[param_idx, :], was_redrawn = redraw_param_until_valid(cmaes_instance, param_values, bounds)
            if was_redrawn:
                redraw_counter += 1
            print(f'Redrawn {redraw_counter} out-of-bounds mutants for parameter {cmaes_instance._parameters[param_idx].name}')
        else:
            cmaes_instance.mutants[param_idx, :] = apply_repair_to_param(cmaes_instance, param_values, bounds, param_idx)

def redraw_param_until_valid(cmaes_instance, param_values, bounds):
    """
    Redraws the out-of-bounds values of a parameter array until they are within bounds.
    """
    was_redrawn = False
    for i in range(len(param_values)):
        while not in_bounds(param_values[i], bounds[0], bounds[1]):
            param_values[i] = np.random.randn()
            was_redrawn = True
    return param_values, was_redrawn

def apply_repair_to_param(cmaes_instance, param_values, bounds, param_idx):
    """
    Applies a repair method to the full array of parameter values if defined.
    """
    param = cmaes_instance._parameters[param_idx]
    printed_repair = False
    while not array_in_bounds(param_values, bounds[0], bounds[1]):
        if cmaes_instance.verbose_level >= 0 and not printed_repair:
            print(f'Repairing parameter {param.name} using method {param.repair}')
            printed_repair = True
        Repair(param.repair, param_values, cmaes_instance.xmean[param_idx])
    return param_values

def transform_mutants(cmaes_instance):
    """
    Transforms local mutants on each process based on the parameters scaling and projection settings.

    For each parameter, depending on its scaling setting ('linear' or 'log'), 
    it applies a transformation to the corresponding elements of scattered_mutants.
    If a projection is specified, it applies this projection to the transformed values.

    Attributes
    ----------
    scattered_mutants : np.ndarray
        A 2D numpy array with the original mutant data. When MPI is used, correspond to the local mutants on each process.
    _parameters : list
        A list of Parameter objects, each with attributes 'scaling', 'lower_bound', 'upper_bound', 
        and 'projection' specifying how to transform the corresponding scattered_mutants.

    Raises
    ------
    ValueError
        If an unrecognized scaling is provided.
    """
    cmaes_instance.transformed_mutants = np.zeros_like(cmaes_instance.scattered_mutants)
    for i, param in enumerate(cmaes_instance._parameters):
        if param.scaling == 'linear':
            cmaes_instance.transformed_mutants[i] = linear_transform(cmaes_instance.scattered_mutants[i], param.lower_bound, param.upper_bound)
        elif param.scaling == 'log':
            cmaes_instance.transformed_mutants[i] = logarithmic_transform(cmaes_instance.scattered_mutants[i], param.lower_bound, param.upper_bound)
        else:
            raise ValueError('Unrecognized scaling, must be linear or log')
        if param.projection is not None:
            cmaes_instance.transformed_mutants[i] = np.asarray(list(map(param.projection, cmaes_instance.transformed_mutants[i])))

def generate_sources(cmaes_instance):
    """
    Generates sources by calling the callback function on transformed data according to the set mode.

    Depending on the mode, the method selects a subset of transformed mutants, possibly extending
    it with zero-filled columns at specific positions, and then passes the processed data to the
    callback function. The results are stored in a contiguous NumPy array in cmaes_instance.sources.

    Raises
    ------
    ValueError
        If an unsupported mode is provided.

    Attributes
    ----------
    mode : str
        A string representing the mode of operation, which can be 'mt', 'mt_dev', 'mt_dc', or 'force'.
    transformed_mutants : np.ndarray
        A 2D numpy array that contains the transformed data to be processed.
    callback : callable
        A callable that is used to process the data.
    """
    # Mapping between modes and respective slices or insertion positions for processing.
    mode_to_indices = {
        'mt': (0, 6),  # For 'mt', a slice from the first 6 elements of transformed_mutants is used.
        'mt_dev': (0, 5, 2),  # For 'mt_dev', a zero column is inserted at position 2 after slicing the first 5 elements.
        'mt_dc': (0, 4, 1, 2),  # For 'mt_dc', zero columns are inserted at positions 1 and 2 after slicing the first 4 elements.
        'force': (0, 3),  # For 'force', a slice from the first 3 elements of transformed_mutants is used.
    }

    # Check the mode's validity. Raise an error if the mode is unsupported.
    if cmaes_instance.mode not in mode_to_indices:
        raise ValueError(f'Invalid mode: {cmaes_instance.mode}')

    # Get the slice or insertion positions based on the current mode.
    indices = mode_to_indices[cmaes_instance.mode]

    # If the mode is 'mt' or 'force', take a slice from transformed_mutants and pass it to the callback.
    if cmaes_instance.mode in ['mt', 'force']:
        cmaes_instance.sources = np.ascontiguousarray(cmaes_instance.callback(*cmaes_instance.transformed_mutants[indices[0]:indices[1]]))
    else:
        # For 'mt_dev' and 'mt_dc' modes, insert zeros at specific positions after slicing transformed_mutants.
        cmaes_instance.extended_mutants = cmaes_instance.transformed_mutants[indices[0]:indices[1]]
        for insertion_index in indices[2:]:
            cmaes_instance.extended_mutants = np.insert(cmaes_instance.extended_mutants, insertion_index, 0, axis=0)
        # Pass the processed data to the callback, and save the result as a contiguous array in cmaes_instance.sources.
        cmaes_instance.sources = np.ascontiguousarray(cmaes_instance.callback(*cmaes_instance.extended_mutants[0:6]))

def scatter_mutants(cmaes_instance):
    """
    Splits and scatters the mutants across processes.
    """
    cmaes_instance.mutant_lists = np.array_split(cmaes_instance.mutants, cmaes_instance.size, axis=1)
    cmaes_instance.mutant_slice = cmaes_instance.comm.scatter(cmaes_instance.mutant_lists, root=0)

def receive_mutants(cmaes_instance):
    """
    Receives scattered mutants on non-root processes.
    """
    cmaes_instance.mutant_slice = cmaes_instance.comm.scatter(None, root=0)

def gather_mutants(cmaes_instance):
    """
    Gathers mutants from all processes into the root process. It also uses the datalogger to construct the mutants_logger_list.

    Attributes
    ----------
    cmaes_instance.mutants : array
        The gathered and concatenated mutants. This attribute is set to None for non-root processes after gathering.
    cmaes_instance.transformed_mutants : array
        The gathered and concatenated transformed mutants. This attribute is set to None for non-root processes after gathering.
    cmaes_instance.mutants_logger_list : list
        The list to which the datalogger is appended.
    """
    # Printing the mutants on each process, their shapes and types for debugging purposes
    if cmaes_instance.verbose_level >= 2:
        print(f"Process {cmaes_instance.rank}: scattered_mutants = {cmaes_instance.scattered_mutants}")
        print(f"Process {cmaes_instance.rank}: shape = {np.shape(cmaes_instance.scattered_mutants)}")
        print(f"Process {cmaes_instance.rank}: type = {type(cmaes_instance.scattered_mutants)}")

    cmaes_instance.mutants = cmaes_instance.comm.gather(cmaes_instance.scattered_mutants, root=0)
    if cmaes_instance.rank == 0:
        cmaes_instance.mutants = np.concatenate(cmaes_instance.mutants, axis=1)
        if cmaes_instance.verbose_level >= 2:
            print("Gathered mutants on root process:")
            print("Mutants array:\n", cmaes_instance.mutants)
            print("Shape of mutants array:", np.shape(cmaes_instance.mutants))
            print("Type of mutants array:", type(cmaes_instance.mutants))
    else:
        cmaes_instance.mutants = None

    cmaes_instance.transformed_mutants = cmaes_instance.comm.gather(cmaes_instance.transformed_mutants, root=0)
    if cmaes_instance.rank == 0:
        cmaes_instance.transformed_mutants = np.concatenate(cmaes_instance.transformed_mutants, axis=1)
        if cmaes_instance.verbose_level >= 2:
            print(cmaes_instance.transformed_mutants, '\n', 'shape is', np.shape(cmaes_instance.transformed_mutants), '\n', 'type is', type(cmaes_instance.transformed_mutants))  # DEBUG PRINT
    else:
        cmaes_instance.transformed_mutants = None

    if cmaes_instance.comm.rank == 0:
        current_df = cmaes_instance._datalogger(mean=False)
    # Log the mutants from _datalogger object
    # If cmaes_instance.mutants_logger_list is empty, initialize it with the current DataFrame
        if cmaes_instance.mutants_logger_list.empty:
            cmaes_instance.mutants_logger_list = current_df
        else:
            # Concatenate the current DataFrame to the logger list
            cmaes_instance.mutants_logger_list = pd.concat([cmaes_instance.mutants_logger_list, current_df], ignore_index=True)
