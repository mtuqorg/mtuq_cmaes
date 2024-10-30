import numpy as np
from mtuq.graphics import plot_data_greens2, plot_data_greens1
from mtuq.graphics.uq._matplotlib import _hammer_projection, _generate_lune, _generate_sphere
from mtuq.util.math import wrap_180, to_gamma, to_delta
from mtuq.event import MomentTensor, Force
from mtuq.grid.moment_tensor import UnstructuredGrid
from mtuq.grid.force import UnstructuredGrid
from mtuq.misfit import Misfit, PolarityMisfit
from mtuq.io.clients.AxiSEM_NetCDF import Client as AxiSEM_Client
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mtuq.graphics import plot_combined, plot_misfit_force

def result_plots(cmaes_instance, data_list, stations, misfit_list, process_list, db_or_greens_list, max_iter, plot_interval, iter_count, iteration):
    if (iteration + 1) % plot_interval == 0 or iteration == max_iter - 1 or (cmaes_instance.ipop and cmaes_instance.ipop_terminated):
        if cmaes_instance.rank == 0:
            plot_mean_waveforms(cmaes_instance, data_list, process_list, misfit_list, stations, db_or_greens_list, iteration)
            if cmaes_instance.mode in ['mt', 'mt_dc', 'mt_dev']:
                print('Plotting results for iteration %d\n' % (iteration + 1 + iter_count))
                result = cmaes_instance.mutants_logger_list

                # Handling the mean solution
                V,W = cmaes_instance.return_candidate_solution()[0][1:3]

                # If mode is mt, mt_dev or mt_dc, plot the misfit map
            if cmaes_instance.mode in ['mt', 'mt_dev', 'mt_dc']:
                plot_combined(cmaes_instance.event_id + '_combined_misfit_map.png', result, colormap='viridis', best_vw=(V, W))
            elif cmaes_instance.mode == 'force':
                print('Plotting results for iteration %d\n' % (iteration + 1 +iter_count))
                result = cmaes_instance.mutants_logger_list
                plot_misfit_force(cmaes_instance.event_id + '_misfit_map.png', result, colormap='viridis', backend=_plot_force_matplotlib, plot_type='colormesh', best_force=cmaes_instance.return_candidate_solution()[0][1::])


def plot_mean_waveforms(cmaes_instance, data_list, process_list, misfit_list, stations, db_or_greens_list, iteration):
    """
    Plots the mean waveforms using the base mtuq waveform plots (mtuq.graphics.waveforms).

    Depending on the mode, different parameters are inserted into the mean solution (padding w or v with 0s for instance)
    If green's functions a provided directly, they are used as is. Otherwise, extrace green's function from Axisem database and preprocess them.
    Support only 1 or 2 waveform groups (body and surface waves, or surface waves only)

    Parameters
    ----------
    data_list : list
        A list of data to be plotted (typically `data_bw` and `data_sw`).
    process_list : list
        A list of processes for each data (typically `process_bw` and `process_sw`).
    misfit_list : list
        A list of misfits for each data (typically `misfit_bw` and `misfit_sw`).
    stations : list
        A list of stations.
    db_or_greens_list : list
        Either an AxiSEM_Client instance or a list of GreensTensors (typically `greens_bw` and `greens_sw`).
    mean_solution : numpy.ndarray
        The mean solution to be plotted.
    final_origin : list
        The final origin to be plotted.
    mode : str
        The mode of the inversion ('mt', 'mt_dev', 'mt_dc', or 'force').
    callback : function
        The callback function used for the inversion.
    event_id : str
        The event ID for the inversion.
    iteration : int
        The current iteration of the inversion.
    rank : int
        The rank of the current process.

    Raises
    ------
    ValueError
        If the mode is not 'mt', 'mt_dev', 'mt_dc', or 'force'.
    """

    if cmaes_instance.rank != 0:
        return  # Exit early if not rank 0

    mean_solution, final_origin = cmaes_instance.return_candidate_solution()

    # Solution grid will change depending on the mode (mt, mt_dev, mt_dc, or force)
    modes = {
        'mt': ('rho', 'v', 'w', 'kappa', 'sigma', 'h'),
        'mt_dev': ('rho', 'v', 'w', 'kappa', 'sigma', 'h'),
        'mt_dc': ('rho', 'v', 'w', 'kappa', 'sigma', 'h'),
        'force': ('F0', 'phi', 'h'),
    }

    if cmaes_instance.mode not in modes:
        raise ValueError('Invalid mode. Supported modes for the plotting functions in the Solve method: "mt", "mt_dev", "mt_dc", "force"')

    mode_dimensions = modes[cmaes_instance.mode]

    # Pad mean_solution based on moment tensor mode (deviatoric or double couple)
    if cmaes_instance.mode == 'mt_dev':
        mean_solution = np.insert(mean_solution, 2, 0, axis=0)
    elif cmaes_instance.mode == 'mt_dc':
        mean_solution = np.insert(mean_solution, 1, 0, axis=0)
        mean_solution = np.insert(mean_solution, 2, 0, axis=0)

    solution_grid = UnstructuredGrid(dims=mode_dimensions, coords=mean_solution, callback=cmaes_instance.callback)

    final_origin = final_origin[0]
    if cmaes_instance.mode.startswith('mt'):
        best_source = MomentTensor(solution_grid.get(0))
    elif cmaes_instance.mode == 'force':
        best_source = Force(solution_grid.get(0))

    lune_dict = solution_grid.get_dict(0)

    # Assignments for brevity (might be removed later)
    data = data_list.copy()
    process = process_list.copy()
    misfit = misfit_list.copy()
    greens_or_db = db_or_greens_list.copy() if isinstance(db_or_greens_list, list) else db_or_greens_list

    mode = 'db' if isinstance(greens_or_db, AxiSEM_Client) else 'greens'
    if mode == 'db':
        _greens = greens_or_db.get_greens_tensors(stations, final_origin)
        greens = [None] * len(process_list)
        for i in range(len(process_list)):
            greens[i] = _greens.map(process_list[i])
            greens[i][0].tags[0] = 'model:ak135f_2s'
    elif mode == 'greens':
        greens = greens_or_db

    # Check for the occurences of mtuq.misfit.polarity.PolarityMisfit in misfit_list:
    # if present, remove the corresponding data, greens, process and misfit from the lists
    # Run backward to avoid index errors
    for i in range(len(misfit) - 1, -1, -1):
        if isinstance(misfit[i], PolarityMisfit):
            del data[i]
            del process[i]
            del misfit[i]
            del greens[i]

    # Include the restart information in the filename _iteration_restart
    name_string = cmaes_instance.event_id + '_waveforms_mean_restart_' + str(cmaes_instance.current_restarts) + '_' + str(iteration + 1) if cmaes_instance.ipop == True else cmaes_instance.event_id + '_waveforms_mean_' + str(iteration + 1)

    # Plot based on the number of ProcessData objects in the process_list
    if len(process) == 2:
        plot_data_greens2(name_string + '.png',
                            data[0], data[1], greens[0], greens[1], process[0], process[1],
                            misfit[0], misfit[1], stations, final_origin, best_source, lune_dict)
    elif len(process) == 1:
        plot_data_greens1(name_string + '.png',
                            data[0], greens[0], process[0], misfit[0], stations, final_origin, best_source, lune_dict)


def _cmaes_scatter_plot(cmaes_instance):
    """
    Generates a scatter plot of the mutants and the current mean solution
    
    Parameters
    ----------
    cmaes_instance : CMA_ES    
        The CMA_ES object containing the necessary information for plotting.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object for the plot.
    """
    if cmaes_instance.rank == 0:
        # Check if mode is mt, mt_dev or mt_dc or force
        if cmaes_instance.mode in ['mt', 'mt_dev', 'mt_dc']:
            if cmaes_instance.fig is None:
                cmaes_instance.fig, cmaes_instance.ax = _generate_lune()

            # Define v as by values from cmaes_instance.mutants_logger_list if it exists, otherwise pad with values of zeroes
            m = np.asarray(cmaes_instance.mutants_logger_list['misfit'])
            sorted_indices = np.argsort(m)[::-1]

            if 'v' in cmaes_instance.mutants_logger_list:
                v = np.asarray(cmaes_instance.mutants_logger_list['v'])
            else:
                v = np.zeros_like(m)

            if 'w' in cmaes_instance.mutants_logger_list:
                w = np.asarray(cmaes_instance.mutants_logger_list['w'])
            else:
                w = np.zeros_like(m)

            # Handling the mean solutions
            V, W = cmaes_instance.mean_logger_list['v'], cmaes_instance.mean_logger_list['w']
            if cmaes_instance.ipop:
                restart = cmaes_instance.mean_logger_list['restart']
            else:
                restart = np.zeros_like(V)

            # Projecting the mean solution onto the lune
            V, W = _hammer_projection(to_gamma(V), to_delta(W))
            cmaes_instance.ax.scatter(V, W, c=restart, marker='x', zorder=10000, cmap='tab10', s=6)
            # Projecting the mutants onto the lune
            v, w = _hammer_projection(to_gamma(v), to_delta(w))

            vmin, vmax = np.percentile(np.asarray(m), [0, 90])

            cmaes_instance.ax.scatter(v[sorted_indices], w[sorted_indices], c=m[sorted_indices], s=3, vmin=vmin, vmax=vmax, zorder=100)

            # Add the mean solution to the plot



            cmaes_instance.fig.canvas.draw()
            return cmaes_instance.fig

        elif cmaes_instance.mode == 'force':
            if cmaes_instance.fig is None:
                cmaes_instance.fig, cmaes_instance.ax = _generate_sphere()

            # phi and h will always be present in the mutants_logger_list
            m = np.asarray(cmaes_instance.mutants_logger_list['misfit'])
            phi, h = np.asarray(cmaes_instance.mutants_logger_list['phi']), np.asarray(cmaes_instance.mutants_logger_list['h'])
            latitude = np.degrees(np.pi / 2 - np.arccos(h))
            longitude = wrap_180(phi + 90)
            # Getting mean solution
            PHI, H = cmaes_instance.mean_logger_list['phi'], cmaes_instance.mean_logger_list['h']
            LATITUDE = np.asarray(np.degrees(np.pi / 2 - np.arccos(H)))
            LONGITUDE = wrap_180(np.asarray(PHI + 90))
            if cmaes_instance.ipop:
                restart = cmaes_instance.mean_logger_list['restart']
            else:
                restart = np.zeros_like(LATITUDE)

            # Projecting the mean solution onto the sphere
            LONGITUDE, LATITUDE = _hammer_projection(LONGITUDE, LATITUDE)
            # Projecting the mutants onto the sphere
            longitude, latitude = _hammer_projection(longitude, latitude)

            vmin, vmax = np.percentile(np.asarray(m), [0, 90])

            cmaes_instance.ax.scatter(longitude, latitude, c=m, s=3, vmin=vmin, vmax=vmax, zorder=100, cmap='Greys_r')
            cmaes_instance.ax.scatter(LONGITUDE, LATITUDE, c=restart, marker='x', zorder=10000, cmap='tab10', s=6)

            cmaes_instance.fig.tight_layout()

            return cmaes_instance.fig
        


def _cmaes_scatter_plot_dc(cmaes_instance):
    """
    Generates a scatter plot of the mutants and the current mean solution on the three DC quadrants
    
    Parameters
    ----------
    cmaes_instance : CMA_ES    
        The CMA_ES object containing the necessary information for plotting.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object for the plot.
    """
    if cmaes_instance.rank == 0:
        # Define v as by values from cmaes_instance.mutants_logger_list if it exists, otherwise pad with values of zeroes
        m = np.asarray(cmaes_instance.mutants_logger_list['misfit'])
        sorted_indices = np.argsort(m)[::-1]

        kappa = np.asarray(cmaes_instance.mutants_logger_list['kappa'])
        sigma = np.asarray(cmaes_instance.mutants_logger_list['sigma'])
        h = np.asarray(cmaes_instance.mutants_logger_list['h'])
        # omega = np.rad2deg(np.arccos(h))
        omega = h

        # Handling the mean solutions
        KAPPA, SIGMA, H = cmaes_instance.mean_logger_list['kappa'], cmaes_instance.mean_logger_list['sigma'], cmaes_instance.mean_logger_list['h']
        # OMEGA = np.rad2deg(np.arccos(H))
        OMEGA = H
        if cmaes_instance.ipop:
            restart = cmaes_instance.mean_logger_list['restart']
        else:
            restart = np.zeros_like(KAPPA)

        vmin, vmax = np.percentile(np.asarray(m), [0, 90])

        fig = plt.figure(figsize=(5, 5))
        gs = GridSpec(2, 2, figure=fig)

        # Turn off the unused subplot (bottom-left)
        ax_unused = fig.add_subplot(gs[1, 0])
        ax_unused.axis('off')

        kappa_ticks = [0, 45, 90, 135, 180, 225, 270, 315, 360]
        kappa_ticklabels = ['0', '', '90', '', '180', '', '270', '', '360']

        sigma_ticks = [-90, -67.5, -45, -22.5, 0, 22.5, 45, 67.5, 90]
        sigma_ticklabels = ['-90', '', '-45', '', '0', '', '45', '', '90']

        h_ticks = [np.cos(np.radians(tick)) for tick in [0, 15, 30, 45, 60, 75, 90]]
        h_ticklabels = ['0', '', '30', '', '60', '', '90']
        

        # Plotting kappa vs omega in the top-right subplot, sharing x-axis with ax1
        ax1 = fig.add_subplot(gs[0, 0])
        scatter1 = ax1.scatter(omega[sorted_indices], kappa[sorted_indices], c=m[sorted_indices], s=3,
                            vmin=vmin, vmax=vmax, zorder=100)
        ax1.scatter(OMEGA, KAPPA, c=restart, marker='x', zorder=10000, cmap='tab10', s=6)
        ax1.set_xlabel('Dip')
        ax1.set_ylabel('Strike')

        ax1.set_xlim(1, 0)
        ax1.set_ylim(0, 360)

        ax1.set_xticks(h_ticks)
        ax1.set_xticklabels(h_ticklabels)
        ax1.set_yticks(kappa_ticks)
        ax1.set_yticklabels(kappa_ticklabels)


        # Plotting kappa vs sigma in the top-left subplot
        ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
        scatter2 = ax2.scatter(sigma[sorted_indices], kappa[sorted_indices], c=m[sorted_indices], s=3,
                            vmin=vmin, vmax=vmax, zorder=100)
        ax2.scatter(SIGMA, KAPPA, c=restart, marker='x', zorder=10000, cmap='tab10', s=6)
        ax2.set_ylabel('Strike')
        ax2.set_xlabel('Rake')

        ax2.set_xlim(-90, 90)

        ax2.set_xticks(sigma_ticks)
        ax2.set_xticklabels(sigma_ticklabels)
        ax2.set_yticks(kappa_ticks)
        ax2.set_yticklabels(kappa_ticklabels)

        # Plotting sigma vs omega in the bottom-right subplot, sharing y-axis with ax2
        ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
        scatter3 = ax3.scatter(sigma[sorted_indices], omega[sorted_indices], c=m[sorted_indices], s=3,
                            vmin=vmin, vmax=vmax, zorder=100)
        ax3.scatter(SIGMA, OMEGA, c=restart, marker='x', zorder=10000, cmap='tab10', s=6)
        ax3.set_xlabel('Rake')
        ax3.set_ylabel('Dip')

        ax3.set_ylim(0, 1)


        ax3.set_xticks(sigma_ticks)
        ax3.set_xticklabels(sigma_ticklabels)
        ax3.set_yticks(h_ticks)
        ax3.set_yticklabels(h_ticklabels)

        # Adjust layout to prevent overlap
        fig.tight_layout()

        # Return the figure
        return fig
        
