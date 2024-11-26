#!/usr/bin/env python

import os
import numpy as np

from mtuq import read, open_db, download_greens_tensors
from mtuq.event import Origin
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath
from mtuq.util.cap import parse_station_codes, Trapezoid
from mtuq_cmaes import initialize_mt
from mtuq_cmaes.cmaes import CMA_ES
from mtuq_cmaes.cmaes_plotting import _cmaes_scatter_plot, _cmaes_scatter_plot_dc

if __name__=='__main__':
    #
    # Carries out CMA-ES inversion over moment tensor parameters
    #
    # USAGE
    #   mpirun -n <NPROC> python mtuq_cmaes_fmt.py
    # ---------------------------------------------------------------------
    # The code is intended to be run either sequentially if using the `greens`
    #  mode) or in parallel (highly recomanded for `database` mode). 
    # ---------------------------------------------------------------------
    # The `greens` mode with 24 ~ 120 mutants per generation (CMAES parameter 'lambda')
    # should only take a few seconds to run on a single core, and achieves better 
    # results than when using a grid search. (No restriction of being on a grid, including 
    # finer Mw search).
    # The algorithm can converge with as low as 6 mutants per generation, but this is
    # not recommended as it will take more steps to converge, and is more prone to
    # getting stuck in local minima. This could be useful if you are trying to find
    # other minima, but is not recommended for general use. 
    # ---------------------------------------------------------------------
    # The 'database' should be used when searching over depth / hypocenter.
    # I also recommend anything between 24 to 120 mutants per generation, (CMAES parameter 'lambda')
    # Each mutant will require its own greens functions, meaning the most compute time will be
    # spent fetching and pre-processing greens functions. This can be sped up by using a
    # larger number of cores, but the scaling is not perfect. (e.g. 24 cores is not 24x faster)
    # The use of the ipop restart strategy has not been tested in this mode, so mileage may vary.
    # ---------------------------------------------------------------------
    # CMA-ES algorithm
    # 1 - Initialise the CMA-ES algorithm with a set of mutants
    # 2 - Evaluate the misfit of each mutant
    # 3 - Sort the mutants by misfit (best to worst), the best mutants are used to update the
    #     mean and covariance matrix of the next generation (50% of the population retained)
    # 4 - Update the mean and covariance matrix of the next generation
    # 5 - Repeat steps 2-4 until the ensemble of mutants converges

    path_data=    fullpath('data/examples/20090407201255351/*.[zrt]')
    path_weights= fullpath('data/examples/20090407201255351/weights.dat')
    event_id=     '20090407201255351'
    model=        'ak135'
    mode =        'greens' # 'database' or 'greens'

    #
    # Body and surface wave measurements will be made separately
    #

    process_bw = ProcessData(
        filter_type='Bandpass',
        freq_min= 0.1,
        freq_max= 0.333,
        pick_type='taup',
        taup_model=model,
        window_type='body_wave',
        window_length=15.,
        capuaf_file=path_weights,
        )

    process_sw = ProcessData(
        filter_type='Bandpass',
        freq_min=0.025,
        freq_max=0.0625,
        pick_type='taup',
        taup_model=model,
        window_type='surface_wave',
        window_length=150.,
        capuaf_file=path_weights,
        )


    #
    # For our objective function, we will use a sum of body and surface wave
    # contributions
    #

    misfit_bw = Misfit(
        norm='L2',
        time_shift_min=-2.,
        time_shift_max=+2.,
        time_shift_groups=['ZR'],
        )

    misfit_sw = Misfit(
        norm='L2',
        time_shift_min=-10.,
        time_shift_max=+10.,
        time_shift_groups=['ZR','T'],
        )


    #
    # User-supplied weights control how much each station contributes to the
    # objective function. Note that these should be functional in the CMAES
    # mode.
    #

    station_id_list = parse_station_codes(path_weights)


    #
    # Next, we specify the source wavelet. 
    #

    wavelet = Trapezoid(
        magnitude=4.5)


    #
    # The Origin time and hypocenter are defined as in the grid-search codes
    # It will either be fixed and used as-is by the CMA-ES mode (typically `greens` mode)
    # or will be used as a starting point for hypocenter search (using the `database` mode)
    #
    # See also Dataset.get_origins(), which attempts to create Origin objects
    # from waveform metadata
    #

    origin = Origin({
        'time': '2009-04-07T20:12:55.000000Z',
        'latitude': 61.454200744628906,
        'longitude': -149.7427978515625,
        'depth_in_m': 33033.599853515625,
        })


    from mpi4py import MPI
    comm = MPI.COMM_WORLD


    #
    # The main I/O work starts now
    #

    if comm.rank==0:
        print('Reading data...\n')
        data = read(path_data, format='sac', 
            event_id=event_id,
            station_id_list=station_id_list,
            tags=['units:m', 'type:velocity']) 


        data.sort_by_distance()
        stations = data.get_stations()


        print('Processing data...\n')
        data_bw = data.map(process_bw)
        data_sw = data.map(process_sw)


        if mode == 'greens':
            print('Reading Greens functions...\n')
            greens = download_greens_tensors(stations, origin, model)
            # ------------------
            # Alternatively, if you have a local AxiSEM database, you can use:
            # db = open_db('/Path/To/Axisem/Database/ak135f/', format='AxiSEM')
            # greens = db.get_greens_tensors(stations, origin, model)
            # ------------------            
            greens.convolve(wavelet)
            greens_bw = greens.map(process_bw)
            greens_sw = greens.map(process_sw)


    else:
        stations = None
        data_bw = None
        data_sw = None
        if mode == 'greens':
            db = None
            greens_bw = None
            greens_sw = None
            greens = None

    stations = comm.bcast(stations, root=0)
    data_bw = comm.bcast(data_bw, root=0)
    data_sw = comm.bcast(data_sw, root=0)

    if mode == 'greens':
        greens_bw = comm.bcast(greens_bw, root=0)
        greens_sw = comm.bcast(greens_sw, root=0)
        greens = comm.bcast(greens, root=0)
    elif mode == 'database':
        # This mode expects the path to a local AxiSEM database to be specified 
        db = open_db('/Path/To/Axisem/Database/ak135f/', format='AxiSEM')
    #
    # The main computational work starts now
    #
    
    # Defining source type (full moment tensor, deviatoric moment tensor or double couple)
    src_type = 'full' # 'full', 'deviatoric' or 'dc'

    if mode == 'database':
        # For a full search with depth, latitutde and longitude, use:
        parameter_list = initialize_mt(Mw_range=[4,6], depth_range=[30000, 55000], latitude_range=[61.0, 61.8], longitude_range=[-150.0, -149.0], src_type=src_type)
        # Alternatively, to fix the depth, latitude and longitude, use:
        # parameter_list = initialize_mt(Mw_range=[4,6], src_type=src_type) # -- Note: This is not the recommanded use for fixed origin, prefer using the 'greens' mode
    elif mode == 'greens':
        parameter_list = initialize_mt(Mw_range=[4,6], src_type=src_type)

    # Creating list of important objects to be passed to solving and plotting functions later
    DATA = [data_bw, data_sw]  # add more as needed
    MISFIT = [misfit_bw, misfit_sw]  # add more as needed
    PROCESS = [process_bw, process_sw]  # add more as needed
    GREENS = [greens_bw, greens_sw] if mode == 'greens' else None  # add more as needed

    ipop = True # -- IPOP automatically restarts the algorithm, with an increased population size. It generally improves the exploration of the parameter space.
    
    if not ipop:
        popsize = 240 # -- CMA-ES population size - number of mutants (you can play with this value, 24 to 120 is a good range)
        CMA = CMA_ES(parameter_list , origin=origin, lmbda=popsize, event_id=event_id, ipop=False)
    else:
        popsize = None # -- If popsize it None, it will use the smallest population size, based on the empirical rule of thumb: popsize = 4 + int(3 * np.log(N)), where N is the number of parameters to be optimized.
        CMA = CMA_ES(parameter_list , origin=origin, lmbda=popsize, event_id=event_id, ipop=True)
    CMA.sigma = 1.67 # -- CMA-ES step size, defined as the standard deviation of the population can be ajusted here (1.67 seems to provide a balanced exploration/exploitation and avoid getting stuck in local minima). 
    # The default value is otherwise 1 standard deviation (you can play with this value)
    iter = 60 # -- Number of iterations (you can play with this value, 60 to 120 is a good range. If Using IPOP, you can use a lower number of iterations, as restart will potentially supersed the need for more iterations).

    if mode == 'database':
        CMA.Solve(DATA, stations, MISFIT, PROCESS, db, iter, wavelet, plot_interval=10, misfit_weights=[1., 12.])
    elif mode == 'greens':
        CMA.Solve(DATA, stations, MISFIT, PROCESS, GREENS, iter, plot_interval=10, misfit_weights=[1., 12.])

    if comm.rank==0:
        result = CMA.mutants_logger_list # -- This is the list of mutants (i.e. the population) at each iteration
        # This is a mtuq.grid_search.MTUQDataFrame object, which is the same as when conducting a random grid-search
        # It is therefore compatible with the "regular" plotting functions in mtuq.graphics 
        fig = _cmaes_scatter_plot(CMA) # -- This is a scatter plot of the mutants at the last iteration
        fig.savefig(event_id+'CMA-ES_final_step.pdf')
        fig2 = _cmaes_scatter_plot_dc(CMA)
        fig2.savefig(event_id+'CMA-ES_final_step_dc.pdf')

    if comm.rank==0:
        print("Total number of misfit evaluations: ", CMA.counteval)
        print('\nFinished\n')

    # ================================================================================================
    # FOR EDUCATIONAL PURPOSE -- This is what is happening under the hood in the Solve function
    #                              . . . not an actual code to run . . .
    # ================================================================================================
    # for i in range(iter):
    #     # ------------------
    #     # The CMA-ES Algorithm is described in:
    #     # Hansen, N. (2016) The CMA Evolution Strategy: A Tutorial. arXiv:1604.00772
    #     # ------------------
    #     CMA.draw_mutants() # -- Draw mutants from the current distribution
    #     if mode == 'database':
    #         # It using the database mode, the catalog origin and process functions are required.
    #         # As with the grid-search, we can separate Body-wave and Surface waves misfit. It is also possible to
    #         # Split the misfit into different time-shift groups (e.g. b-ZR, s-ZR, s-T, etc.)
    #         mis_bw = CMA.eval_fitness(data_bw, stations, misfit_bw, db, origin,  process_bw, wavelet, verbose=False)
    #         mis_sw = CMA.eval_fitness(data_sw, stations, misfit_sw, db, origin,  process_sw, wavelet, verbose=False)
    #     elif mode == 'greens':
    #         mis_bw = CMA.eval_fitness(data_bw, stations, misfit_bw, greens_bw)
    #         mis_sw = CMA.eval_fitness(data_sw, stations, misfit_sw, greens_sw)
    #
    #     CMA.gather_mutants() # -- Gather mutants from all processes 
    #     CMA.fitness_sort(mis_bw+mis_sw) # -- Sort mutants by fitness
    #     CMA.update_mean() # -- Update the mean of the distribution
    #     CMA.update_step_size() # -- Update the step size
    #     CMA.update_covariance()   # -- Update the covariance matrix
    #
    #     # ------------------  Plotting results ------------------
    #     # if i multiple of `plot_interval` and Last iteration:
    #     if i % plot_interval == 0 or i == iter-1:
    #         if mode == 'database':
    #             cmaes_instance.plot_mean_waveforms(DATA, PROCESS, MISFIT, stations, db)
    #         elif mode == 'greens':
    #             cmaes_instance.plot_mean_waveforms(DATA, PROCESS, MISFIT, stations, db=greens)
    #
    #         if src_type == 'full' or src_type == 'deviatoric' or src_type == 'dc':
    #             if CMA.comm.rank==0:
    #                 result = CMA.mutants_logger_list # This one is an important one! 
    #                 It returns a DataFrame, the same as when using a random grid search and is therefore compatible with the default mtuq plotting tools.
    #                 result_plots(CMA, data_list, stations, misfit_list, process_list, db_or_greens_list, max_iter, plot_interval, iter_count, iteration)
    # ================================================================================================