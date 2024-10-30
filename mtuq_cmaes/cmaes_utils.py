import numpy as np
from mtuq.io.clients.AxiSEM_NetCDF import Client as AxiSEM_Client
from mtuq.misfit import Misfit, PolarityMisfit
from mtuq.dataset import Dataset
from mtuq.process_data import ProcessData
from mtuq.greens_tensor.base import GreensTensorList

def validate_inputs(data_list, stations, misfit_list, process_list, db_or_greens_list, max_iter, wavelet, plot_interval, iter_count):
    """
    Centralized input validation for the CMA-ES algorithm.
    Raises ValueError if any of the input parameters are invalid.
    """
    if not isinstance(data_list, list):
        if isinstance(data_list, Dataset):
            data_list = [data_list]
        else:
            raise ValueError('`data_list` should be a list of mtuq Dataset or an array containing polarities.')

    if not isinstance(stations, list):
        raise ValueError('`stations` should be a list of mtuq Station objects.')

    if not isinstance(misfit_list, list):
        if isinstance(misfit_list, (PolarityMisfit, Misfit)):
            misfit_list = [misfit_list]
        else:
            raise ValueError('`misfit_list` should be a list of mtuq Misfit objects.')

    if not isinstance(process_list, list):
        if isinstance(process_list, ProcessData):
            process_list = [process_list]
        else:
            raise ValueError('`process_list` should be a list of mtuq ProcessData objects.')

    if not isinstance(db_or_greens_list, list):
        if isinstance(db_or_greens_list, (AxiSEM_Client, GreensTensorList)):
            db_or_greens_list = [db_or_greens_list]
        else:
            raise ValueError('`db_or_greens_list` should be a list of mtuq AxiSEM_Client or GreensTensorList objects.')

    if any(isinstance(db, AxiSEM_Client) for db in db_or_greens_list) and wavelet is None:
        raise ValueError('wavelet must be specified if database is an AxiSEM_Client')

    if not isinstance(max_iter, int) or not isinstance(plot_interval, int):
        raise ValueError('`max_iter` and `plot_interval` should be integers.')

    if iter_count is not None and not isinstance(iter_count, int):
        raise ValueError('`iter_count` should be an integer or None.')


def linear_transform(i, a, b):
    """ 
    Linear map suggested by N. Hansen for appropriate parameter scaling/variable encoding in CMA-ES.

    Linear map from [0;10] to [a,b].

    Parameters
    ----------
    i : float
        The input value to be transformed.
    a : float
        The lower bound of the target range.
    b : float
        The upper bound of the target range.

    Returns
    -------
    float
        The transformed value.

    Example
    -------
    >>> linear_transform(5, 0, 100)
    50.0

    source:
    (https://cma-es.github.io/cmaes_sourcecode_page.html)
    """
    transformed = a + (b-a) * i / 10
    return transformed

def inverse_linear_transform(transformed, a, b):
    """ 
    Inverse linear mapping to reproject the variable in the [0; 10] range, from its original transformation bounds.

    Parameters
    ----------
    transformed : float
        The transformed value to be inversely transformed.
    a : float
        The lower bound of the original range.
    b : float
        The upper bound of the original range.

    Returns
    -------
    float
        The inversely transformed value.

    Example
    -------
    >>> inverse_linear_transform(50, 0, 100)
    5.0
    """
    i = (10*(transformed-a))/(b-a)
    return i

def logarithmic_transform(i, a, b):
    """ 
    Logarithmic mapping suggested by N. Hansen. Particularly adapted to define Magnitude ranges of [1e^(n),1e^(n+3)].

    Logarithmic map from [0,10] to [a,b], with `a` and `b` typically spaced by 3 to 4 orders of magnitudes.

    Parameters
    ----------
    i : float
        The input value to be transformed.
    a : float
        The lower bound of the target range.
    b : float
        The upper bound of the target range.

    Returns
    -------
    float
        The transformed value.

    Example
    -------
    >>> logarithmic_transform(5, 1e1, 1e4)
    316.22776601683796

    source:
    (https://cma-es.github.io/cmaes_sourcecode_page.html)
    """
    d=np.log10(b)-np.log10(a)
    transformed = 10**(np.log10(a)) * 10**(d*i/10)
    return transformed

def in_bounds(value, a=0, b=10):
    """
    Check if a value is within the specified bounds.

    Parameters
    ----------
    value : float
        The value to check.
    a : float, optional
        The lower bound (default is 0).
    b : float, optional
        The upper bound (default is 10).

    Returns
    -------
    bool
        True if the value is within bounds, False otherwise.

    Example
    -------
    >>> in_bounds(5, 0, 10)
    True
    """
    return value >= a and value <= b

def array_in_bounds(array, a=0, b=10):
    """
    Check if all elements of an array are in bounds.

    Parameters
    ----------
    array : array-like
        The array to check.
    a : float, optional
        The lower bound (default is 0).
    b : float, optional
        The upper bound (default is 10).

    Returns
    -------
    bool
        True if all elements of the array are in bounds, False otherwise.

    Example
    -------
    >>> array_in_bounds([5, 6, 7], 0, 10)
    True
    """
    for i in range(len(array)):
        if not in_bounds(array[i], a, b):
            return False
    return True

class Repair:
    """
    Repair class to define all the boundary handling constraint methods implemented in R. Biedrzycki 2019, https://doi.org/10.1016/j.swevo.2019.100627.

    These methods are invoked whenever a CMA-ES mutant infringes a boundary.

    Parameters
    ----------
    method : str
        The repair method to use.
    data_array : array-like
        The array of data to repair.
    mean : float
        The mean value of the distribution.
    lower_bound : float, optional
        The lower bound (default is 0).
    upper_bound : float, optional
        The upper bound (default is 10).

    Methods
    -------
    reinitialize()
        Redraw all out of bounds values from a uniform distribution [0,10].
    projection()
        Project all out of bounds values to the violated bounds.
    reflection()
        Reflect the infeasible coordinate value back from the boundary by the amount of constraint violation.
    wrapping()
        Shift the infeasible coordinate value by the feasible interval.
    transformation()
        Apply adaptive transformation based on 90%-tile.
    rand_based()
        Redraw the out of bound mutants between the base vector and the violated boundary.
    midpoint_base()
        Replace the infeasible coordinate value by the midpoint between its current position and the violated boundary.
    midpoint_target()
        Replace the infeasible coordinate value by the average of the target individual and the violated constraint.
    conservative()
        Replace the infeasible solution by the distribution mean.

    Example
    -------
    >>> data = np.array([12, -3, 5, 8])
    >>> repair = Repair('projection', data, mean=5)
    >>> repair.projection()
    >>> data
    array([10,  0,  5,  8])
    """
    def __init__(self, method, data_array, mean, lower_bound=0, upper_bound=10):
        self.method = method
        self.data_array = data_array
        self.mean = mean
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # Out of boundary violators mask. l_oob for lower_out-of-boundary and u_oob for upper_out-of-boundary.
        self.l_oob = self.data_array < self.lower_bound
        self.u_oob = self.data_array > self.upper_bound


        if self.method == 'reinitialize':
            # Call the reinitialize function
            self.reinitialize()
        elif self.method == 'projection':
            # Call the projection function
            self.projection()
        elif self.method == 'reflection':
            # Call the reflection function
            self.reflection()
        elif self.method == 'wrapping':
            # Call the wrapping function
            self.wrapping()
        elif self.method == 'transformation':
            # Call the transformation function
            self.transformation()
        elif self.method == 'projection_to_midpoint':
            # Call the projection_to_midpoint function
            print('Projection to midpoint repair method not implemented')
        elif self.method == 'rand_based':
            # Call the rand_based function
            self.rand_based()
        elif self.method == 'midpoint_base':
            # Call the rebound function
            self.midpoint_base()
        elif self.method == 'midpoint_target':
            # Call the rebound function
            self.midpoint_target()
        elif self.method == 'conservative':
            # Call the conservative function
            self.conservative()
        else:
            print('Repair method not recognized')

    def reinitialize(self):
        """
        Redraw all out of bounds values from a uniform distribution [0,10].
        """
        self.data_array[self.l_oob] = np.random.uniform(self.lower_bound,self.upper_bound, len(self.data_array[self.l_oob]))
        self.data_array[self.u_oob] = np.random.uniform(self.lower_bound,self.upper_bound, len(self.data_array[self.u_oob]))

    def projection(self):
        """
        Project all out of bounds values to the violated bounds.
        """
        self.data_array[self.l_oob] = self.lower_bound
        self.data_array[self.u_oob] = self.upper_bound

    def reflection(self):
        """
        The infeasible coordinate value of the solution is reflected back from the boundary by the amount of constraint violation.
        This method may be called several times if the points are out of bound for more than the length of the [0,10] domain.
        """
        self.data_array[self.l_oob] = 2*self.lower_bound - self.data_array[self.l_oob]
        self.data_array[self.u_oob] = 2*self.upper_bound - self.data_array[self.u_oob]

    def wrapping(self):
        """
        The infeasible coordinate value is shifted by the feasible interval.
        """
        self.data_array[self.l_oob] = self.data_array[self.l_oob] + self.upper_bound - self.lower_bound
        self.data_array[self.u_oob] = self.data_array[self.u_oob] - self.upper_bound + self.lower_bound

    def transformation(self):
        """
        Adaptive transformation based on 90%-tile.
        Nonlinear transform defined by R. Biedrzycki 2019, https://doi.org/10.1016/j.swevo.2019.100627
        """
        al = np.min([(self.upper_bound - self.lower_bound)/2,(1+np.abs(self.lower_bound-2))/20])
        au = np.min([(self.upper_bound - self.lower_bound)/2,(1+np.abs(self.upper_bound))/20])
        # Create the masks for the values out of self.lower_bound - al and self.upper_bound + au bounds
        mask_1 = self.data_array > (self.upper_bound + au)
        mask_2 = self.data_array < (self.lower_bound - al)
        # Reflect out of bounds values according to self.lower_bound - al and self.upper_bound + au.
        self.data_array[mask_1] = (2*self.upper_bound + au) - self.data_array[mask_1]
        self.data_array[mask_2] = 2*self.lower_bound - al - self.data_array[mask_2]

        # Create masks for the nonlinear transformation.
        mask_3 = (self.data_array >= (self.lower_bound + al)) & (self.data_array <= (self.upper_bound - au))
        mask_4 = (self.data_array >= (self.lower_bound - al)) & (self.data_array < (self.lower_bound + al))
        mask_5 = (self.data_array > (self.upper_bound - au)) & (self.data_array <= (self.upper_bound + au))

        # Note that reflected data are transformed according to the same principle which results in a periodic transformation
        self.data_array[mask_3] = self.data_array[mask_3]
        self.data_array[mask_4] = self.lower_bound + (self.data_array[mask_4] - (self.lower_bound-al))**2/(4*al)
        self.data_array[mask_5] = self.upper_bound - (self.data_array[mask_5] - (self.upper_bound-au))**2/(4*al)

    def rand_based(self):
        """
        Redraw the out of bound mutants between the base vector (the CMA_ES.xmean used in draw_mutants()) and the violated boundary.
        The base vector is the mean of the population.
        """
        self.data_array[self.l_oob] = np.random.uniform(self.mean, self.lower_bound, len(self.data_array[self.l_oob]))
        self.data_array[self.u_oob] = np.random.uniform(self.mean, self.upper_bound, len(self.data_array[self.u_oob]))

    def midpoint_base(self):
        """
        The infeasible coordinate value is replaced by the midpoint between its current position and the violated boundary.
        Ensures that the resulting points are within the bounds.
        """
        # For lower out-of-bounds values, move them towards the midpoint between their position and the lower bound
        self.data_array[self.l_oob] = (self.data_array[self.l_oob] + self.lower_bound) / 2
        
        # Ensure they are within bounds
        self.data_array[self.l_oob] = np.maximum(self.data_array[self.l_oob], self.lower_bound)
        
        # For upper out-of-bounds values, move them towards the midpoint between their position and the upper bound
        self.data_array[self.u_oob] = (self.data_array[self.u_oob] + self.upper_bound) / 2
        
        # Ensure they are within bounds
        self.data_array[self.u_oob] = np.minimum(self.data_array[self.u_oob], self.upper_bound)

    def midpoint_target(self):
        """
        The average of the target individual and the violated constraint replaces the infeasible coordinate value.
        """
        target = 5
        self.data_array[self.l_oob] = (target + self.lower_bound)/2
        self.data_array[self.u_oob] = (target + self.upper_bound)/2

    def conservative(self):
        """
        The infeasible solution is replaced by the distribution mean.
        """
        self.data_array[self.l_oob] = self.mean
        self.data_array[self.u_oob] = self.mean
