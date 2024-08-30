"""Plots hyperparameter grids for bias-clustering extension to Experiment 9.

One hyperparameter grid = one evaluation metric vs. all hyperparams.
"""

import os
import sys
import argparse
import numpy
from scipy.stats import rankdata
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import imagemagick_utils
import file_system_utils
import gg_plotting_utils
import evaluation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

BIAS_DISCRETIZATION_INTERVAL_INTERVALS_AXIS1 = numpy.array([0.025, 0.050, 0.075, 0.100])
MIN_CLUSTER_SIZES_PX_AXIS2 = numpy.array([10, 25, 50, 100, 250], dtype=int)
BUFFER_DISTANCES_PX_AXIS3 = numpy.array([0, 2, 4, 6], dtype=int)
DO_BACKWARDS_FLAGS_AXIS4 = numpy.array([0, 1], dtype=int)

BEST_MARKER_TYPE = '*'
BEST_MARKER_SIZE_GRID_CELLS = 0.075
WHITE_COLOUR = numpy.full(3, 1.)
BLACK_COLOUR = numpy.full(3, 0.)

SELECTED_MARKER_TYPE = 'o'
SELECTED_MARKER_SIZE_GRID_CELLS = 0.075
SELECTED_MARKER_INDICES = numpy.array([0, 0, 0, 0], dtype=int)

MAIN_COLOUR_MAP_OBJECT = pyplot.get_cmap(name='viridis', lut=20)
BIAS_COLOUR_MAP_OBJECT = pyplot.get_cmap(name='seismic', lut=20)
MAIN_COLOUR_MAP_OBJECT.set_bad(numpy.full(3, 152. / 255))
BIAS_COLOUR_MAP_OBJECT.set_bad(numpy.full(3, 152. / 255))

FONT_SIZE = 26
AXIS_LABEL_FONT_SIZE = 26
TICK_LABEL_FONT_SIZE = 14

pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

EXPERIMENT_DIR_ARG_NAME = 'experiment_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

EXPERIMENT_DIR_HELP_STRING = 'Name of top-level directory with models.'
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXPERIMENT_DIR_ARG_NAME, type=str, required=True,
    help=EXPERIMENT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _finite_percentile(input_array, percentile_level):
    """Takes percentile of input array, considering only finite values.

    :param input_array: numpy array.
    :param percentile_level: Percentile level, ranging from 0...100.
    :return: output_percentile: Percentile value.
    """

    return numpy.percentile(
        input_array[numpy.isfinite(input_array)], percentile_level
    )


def _plot_scores_2d(
        score_matrix, min_colour_value, max_colour_value, x_tick_labels,
        y_tick_labels):
    """Plots scores on 2-D grid.

    M = number of rows in grid
    N = number of columns in grid

    :param score_matrix: M-by-N numpy array of scores.
    :param min_colour_value: Minimum value in colour scheme.
    :param max_colour_value: Max value in colour scheme.
    :param x_tick_labels: length-N list of tick labels.
    :param y_tick_labels: length-M list of tick labels.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    if min_colour_value is None:
        colour_map_object = BIAS_COLOUR_MAP_OBJECT
        min_colour_value = -1 * max_colour_value
    else:
        colour_map_object = MAIN_COLOUR_MAP_OBJECT

    axes_object.imshow(
        score_matrix, cmap=colour_map_object, origin='lower',
        vmin=min_colour_value, vmax=max_colour_value
    )

    x_tick_values = numpy.linspace(
        0, score_matrix.shape[1] - 1, num=score_matrix.shape[1], dtype=float
    )
    y_tick_values = numpy.linspace(
        0, score_matrix.shape[0] - 1, num=score_matrix.shape[0], dtype=float
    )

    pyplot.xticks(
        x_tick_values, x_tick_labels,
        rotation=90., fontsize=TICK_LABEL_FONT_SIZE
    )
    pyplot.yticks(y_tick_values, y_tick_labels, fontsize=TICK_LABEL_FONT_SIZE)

    colour_norm_object = matplotlib.colors.Normalize(
        vmin=min_colour_value, vmax=max_colour_value, clip=False
    )

    colour_bar_object = gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object,
        data_matrix=score_matrix[numpy.invert(numpy.isnan(score_matrix))],
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', extend_min=False, extend_max=False,
        font_size=FONT_SIZE, fraction_of_axis_length=0.6
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.2g}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    return figure_object, axes_object


def _print_ranking_one_score(score_matrix, score_name):
    """Prints ranking for one score.

    I = number of bias-discretization interval intervals
    M = number of minimum cluster sizes
    D = number of buffer distances
    B = number of backwards flags

    :param score_matrix: I-by-M-by-D-by-B numpy array of score values.
    :param score_name: Name of score.
    """

    scores_1d = numpy.ravel(score_matrix)
    scores_1d[numpy.isnan(scores_1d)] = numpy.inf
    sort_indices_1d = numpy.argsort(scores_1d)
    i_sort_indices, m_sort_indices, d_sort_indices, b_sort_indices = (
        numpy.unravel_index(sort_indices_1d, score_matrix.shape)
    )

    for k in range(len(i_sort_indices)):
        i = i_sort_indices[k]
        m = m_sort_indices[k]
        d = d_sort_indices[k]
        b = b_sort_indices[k]

        print((
            r'{0:d}th-lowest {1:s} = {2:.4g} ... '
            r'discretization $\Delta\Delta$ = {3:.3f} ... '
            r'min size = {4:d} ... '
            r'buffer dist = {5:d} px ... '
            r'backwards = {6:d}'
        ).format(
            k + 1, score_name, score_matrix[i, m, d, b],
            BIAS_DISCRETIZATION_INTERVAL_INTERVALS_AXIS1[i],
            MIN_CLUSTER_SIZES_PX_AXIS2[m],
            BUFFER_DISTANCES_PX_AXIS3[d],
            DO_BACKWARDS_FLAGS_AXIS4[b]
        ))


def _print_ranking_all_scores(
        dwmse_matrix_celsius03, rmse_matrix_kelvins01, mae_matrix_kelvins01,
        bias_matrix_kelvins01, stdev_bias_matrix_kelvins01,
        spatial_min_bias_matrix_kelvins01, spatial_max_bias_matrix_kelvins01,
        correlation_matrix, kge_matrix, reliability_matrix_kelvins02):
    """Prints ranking for all scores.

    I = number of bias-discretization interval intervals
    M = number of minimum cluster sizes
    D = number of buffer distances
    B = number of backwards flags

    :param dwmse_matrix_celsius03: I-by-M-by-D-by-B numpy array of DWMSE
        (dual-weighted mean squared error) values.
    :param rmse_matrix_kelvins01: I-by-M-by-D-by-B numpy array of root mean
        squared error (RMSE) values.
    :param mae_matrix_kelvins01: I-by-M-by-D-by-B numpy array of mean absolute
        errors (MAE).
    :param bias_matrix_kelvins01: I-by-M-by-D-by-B numpy array of bias values.
    :param stdev_bias_matrix_kelvins01: I-by-M-by-D-by-B numpy array of
        standard-deviation biases.
    :param spatial_min_bias_matrix_kelvins01: I-by-M-by-D-by-B numpy array of
        spatial-minimum biases.
    :param spatial_max_bias_matrix_kelvins01: I-by-M-by-D-by-B numpy array of
        spatial-maximum biases.
    :param correlation_matrix: I-by-M-by-D-by-B numpy array of Pearson
        correlations.
    :param kge_matrix: I-by-M-by-D-by-B numpy array of Kling-Gupta efficiency
        (KGE) values.
    :param reliability_matrix_kelvins02: I-by-M-by-D-by-B numpy array of
        reliability values.
    """

    these_scores = numpy.ravel(dwmse_matrix_celsius03)
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    dwmse_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        dwmse_matrix_celsius03.shape
    )

    these_scores = numpy.ravel(rmse_matrix_kelvins01)
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    rmse_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        rmse_matrix_kelvins01.shape
    )

    these_scores = numpy.ravel(mae_matrix_kelvins01)
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    mae_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        mae_matrix_kelvins01.shape
    )

    these_scores = numpy.ravel(numpy.absolute(bias_matrix_kelvins01))
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    bias_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        bias_matrix_kelvins01.shape
    )

    these_scores = numpy.ravel(numpy.absolute(stdev_bias_matrix_kelvins01))
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    stdev_bias_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        stdev_bias_matrix_kelvins01.shape
    )

    these_scores = numpy.ravel(
        numpy.absolute(spatial_min_bias_matrix_kelvins01)
    )
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    spatial_min_bias_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        spatial_min_bias_matrix_kelvins01.shape
    )

    these_scores = numpy.ravel(
        numpy.absolute(spatial_max_bias_matrix_kelvins01)
    )
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    spatial_max_bias_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        spatial_max_bias_matrix_kelvins01.shape
    )

    these_scores = numpy.ravel(-1 * correlation_matrix)
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    correlation_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        correlation_matrix.shape
    )

    these_scores = numpy.ravel(-1 * kge_matrix)
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    kge_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        kge_matrix.shape
    )

    these_scores = numpy.ravel(reliability_matrix_kelvins02)
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    reliability_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        reliability_matrix_kelvins02.shape
    )

    overall_rank_matrix = numpy.mean(
        numpy.stack([
            dwmse_rank_matrix, rmse_rank_matrix, mae_rank_matrix,
            bias_rank_matrix, stdev_bias_rank_matrix,
            spatial_min_bias_rank_matrix, spatial_max_bias_rank_matrix,
            correlation_rank_matrix, kge_rank_matrix, reliability_rank_matrix
        ], axis=-1),
        axis=-1
    )

    sort_indices_1d = numpy.argsort(numpy.ravel(overall_rank_matrix))
    i_sort_indices, m_sort_indices, d_sort_indices, b_sort_indices = (
        numpy.unravel_index(sort_indices_1d, overall_rank_matrix.shape)
    )

    for k in range(len(i_sort_indices)):
        i = i_sort_indices[k]
        m = m_sort_indices[k]
        d = d_sort_indices[k]
        b = b_sort_indices[k]

        print((
            r'Discretization $\Delta\Delta$ = {0:.3f} ... '
            r'min size = {1:d} ... '
            r'buffer dist = {2:d} px ... '
            r'backwards = {3:d}:\n'
            'DWMSE/RMSE/MAE ranks = {4:.1f}, {5:.1f}, {6:.1f} ... '
            'bias and stdev-bias ranks = {7:.1f}, {8:.1f} ... '
            'spatial-min- and spatial-max-bias ranks = {9:.1f}, {10:.1f} ... '
            'correlation/KGE/reliability ranks = {11:.1f}, {12:.1f}, {13:.1f}'
        ).format(
            BIAS_DISCRETIZATION_INTERVAL_INTERVALS_AXIS1[i],
            MIN_CLUSTER_SIZES_PX_AXIS2[m],
            BUFFER_DISTANCES_PX_AXIS3[d],
            DO_BACKWARDS_FLAGS_AXIS4[b],
            dwmse_rank_matrix[i, m, d, b],
            rmse_rank_matrix[i, m, d, b],
            mae_rank_matrix[i, m, d, b],
            bias_rank_matrix[i, m, d, b],
            stdev_bias_rank_matrix[i, m, d, b],
            spatial_min_bias_rank_matrix[i, m, d, b],
            spatial_max_bias_rank_matrix[i, m, d, b],
            correlation_rank_matrix[i, m, d, b],
            kge_rank_matrix[i, m, d, b],
            reliability_rank_matrix[i, m, d, b]
        ))


def _run(experiment_dir_name, output_dir_name):
    """Plots hyperparameter grids for bias-clustering extension to Experiment 9.

    This is effectively the main method.

    :param experiment_dir_name: See documentation at top of this script.
    :param output_dir_name: Same.
    """

    axis1_length = len(BIAS_DISCRETIZATION_INTERVAL_INTERVALS_AXIS1)
    axis2_length = len(MIN_CLUSTER_SIZES_PX_AXIS2)
    axis3_length = len(BUFFER_DISTANCES_PX_AXIS3)
    axis4_length = len(DO_BACKWARDS_FLAGS_AXIS4)

    y_tick_labels = [
        '{0:.3f}'.format(b)
        for b in BIAS_DISCRETIZATION_INTERVAL_INTERVALS_AXIS1
    ]
    x_tick_labels = ['{0:d}'.format(m) for m in MIN_CLUSTER_SIZES_PX_AXIS2]

    y_axis_label = r'Bias-discretization $\Delta\Delta$ ($^{\circ}$C)'
    x_axis_label = 'Min cluster size (pixels)'

    dimensions = (axis1_length, axis2_length, axis3_length, axis4_length)

    dwmse_matrix_celsius03 = numpy.full(dimensions, numpy.nan)
    rmse_matrix_kelvins01 = numpy.full(dimensions, numpy.nan)
    mae_matrix_kelvins01 = numpy.full(dimensions, numpy.nan)
    bias_matrix_kelvins01 = numpy.full(dimensions, numpy.nan)
    stdev_bias_matrix_kelvins01 = numpy.full(dimensions, numpy.nan)
    spatial_min_bias_matrix_kelvins01 = numpy.full(dimensions, numpy.nan)
    spatial_max_bias_matrix_kelvins01 = numpy.full(dimensions, numpy.nan)
    correlation_matrix = numpy.full(dimensions, numpy.nan)
    kge_matrix = numpy.full(dimensions, numpy.nan)
    reliability_matrix_kelvins02 = numpy.full(dimensions, numpy.nan)

    for i in range(axis1_length):
        for j in range(axis2_length):
            for k in range(axis3_length):
                for m in range(axis4_length):
                    this_eval_file_name = (
                        '{0:s}/validation_full_grid/isotonic_regression/'
                        'bias-discretization-interval-interval={1:.3f}_'
                        'min-cluster-size-px={2:03d}_'
                        'buffer-distance-px={3:d}_'
                        'do-backwards-clustering={4:d}/'
                        'ungridded_evaluation.nc'
                    ).format(
                        experiment_dir_name,
                        BIAS_DISCRETIZATION_INTERVAL_INTERVALS_AXIS1[i],
                        MIN_CLUSTER_SIZES_PX_AXIS2[j],
                        BUFFER_DISTANCES_PX_AXIS3[k],
                        DO_BACKWARDS_FLAGS_AXIS4[m]
                    )

                    if not os.path.isfile(this_eval_file_name):
                        continue

                    print('Reading data from: "{0:s}"...'.format(
                        this_eval_file_name
                    ))
                    this_eval_table_xarray = evaluation.read_file(
                        this_eval_file_name
                    )
                    etx = this_eval_table_xarray

                    dwmse_matrix_celsius03[i, j, k, m] = numpy.mean(
                        etx[evaluation.DWMSE_KEY].values[0, :]
                    )
                    rmse_matrix_kelvins01[i, j, k, m] = numpy.sqrt(numpy.mean(
                        etx[evaluation.MSE_KEY].values[0, :]
                    ))
                    mae_matrix_kelvins01[i, j, k, m] = numpy.mean(
                        etx[evaluation.MAE_KEY].values[0, :]
                    )
                    bias_matrix_kelvins01[i, j, k, m] = numpy.mean(
                        etx[evaluation.BIAS_KEY].values[0, :]
                    )
                    stdev_bias_matrix_kelvins01[i, j, k, m] = numpy.mean(
                        etx[evaluation.PREDICTION_STDEV_KEY].values[0, :] -
                        etx[evaluation.TARGET_STDEV_KEY].values[0, :]
                    )
                    spatial_min_bias_matrix_kelvins01[i, j, k, m] = numpy.mean(
                        etx[evaluation.SPATIAL_MIN_BIAS_KEY].values[0, :]
                    )
                    spatial_max_bias_matrix_kelvins01[i, j, k, m] = numpy.mean(
                        etx[evaluation.SPATIAL_MAX_BIAS_KEY].values[0, :]
                    )
                    correlation_matrix[i, j, k, m] = numpy.mean(
                        etx[evaluation.CORRELATION_KEY].values[0, :]
                    )
                    kge_matrix[i, j, k, m] = numpy.mean(
                        etx[evaluation.KGE_KEY].values[0, :]
                    )
                    reliability_matrix_kelvins02[i, j, k, m] = numpy.mean(
                        etx[evaluation.RELIABILITY_KEY].values[0, :]
                    )

    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=dwmse_matrix_celsius03, score_name='DWMSE ([deg C]^-3)'
    )
    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=rmse_matrix_kelvins01,
        score_name='RMSE (Kelvins)'
    )
    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=mae_matrix_kelvins01,
        score_name='MAE (Kelvins)'
    )
    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=numpy.absolute(bias_matrix_kelvins01),
        score_name='absolute bias (Kelvins)'
    )
    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=numpy.absolute(stdev_bias_matrix_kelvins01),
        score_name='absolute stdev bias (Kelvins)'
    )
    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=numpy.absolute(spatial_min_bias_matrix_kelvins01),
        score_name='absolute spatial-min bias (Kelvins)'
    )
    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=numpy.absolute(spatial_max_bias_matrix_kelvins01),
        score_name='absolute spatial-max bias (Kelvins)'
    )
    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=-1 * correlation_matrix,
        score_name='negative correlation'
    )
    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=-1 * kge_matrix,
        score_name='negative KGE'
    )
    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=reliability_matrix_kelvins02,
        score_name='reliability (K^2)'
    )
    print(SEPARATOR_STRING)

    _print_ranking_all_scores(
        dwmse_matrix_celsius03=dwmse_matrix_celsius03,
        rmse_matrix_kelvins01=rmse_matrix_kelvins01,
        mae_matrix_kelvins01=mae_matrix_kelvins01,
        bias_matrix_kelvins01=bias_matrix_kelvins01,
        stdev_bias_matrix_kelvins01=stdev_bias_matrix_kelvins01,
        spatial_min_bias_matrix_kelvins01=spatial_min_bias_matrix_kelvins01,
        spatial_max_bias_matrix_kelvins01=spatial_max_bias_matrix_kelvins01,
        correlation_matrix=correlation_matrix,
        kge_matrix=kge_matrix,
        reliability_matrix_kelvins02=reliability_matrix_kelvins02
    )
    print(SEPARATOR_STRING)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    dimensions = (axis3_length, axis4_length)
    dwmse_panel_file_name_matrix = numpy.full(dimensions, '', dtype=object)
    rmse_panel_file_name_matrix = numpy.full(dimensions, '', dtype=object)
    mae_panel_file_name_matrix = numpy.full(dimensions, '', dtype=object)
    bias_panel_file_name_matrix = numpy.full(dimensions, '', dtype=object)
    stdev_bias_panel_file_name_matrix = numpy.full(dimensions, '', dtype=object)
    spatial_min_bias_panel_file_name_matrix = numpy.full(dimensions, '', dtype=object)
    spatial_max_bias_panel_file_name_matrix = numpy.full(dimensions, '', dtype=object)
    correlation_panel_file_name_matrix = numpy.full(dimensions, '', dtype=object)
    kge_panel_file_name_matrix = numpy.full(dimensions, '', dtype=object)
    reliability_panel_file_name_matrix = numpy.full(dimensions, '', dtype=object)

    for k in range(axis3_length):
        for m in range(axis4_length):

            # Plot DWMSE vs. hyperparameters.
            figure_object, axes_object = _plot_scores_2d(
                score_matrix=dwmse_matrix_celsius03[..., k, m],
                min_colour_value=_finite_percentile(dwmse_matrix_celsius03, 0),
                max_colour_value=_finite_percentile(dwmse_matrix_celsius03, 95),
                x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
            )

            this_index = numpy.nanargmin(numpy.ravel(dwmse_matrix_celsius03))
            best_indices = numpy.unravel_index(
                this_index, dwmse_matrix_celsius03.shape
            )

            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                BEST_MARKER_SIZE_GRID_CELLS / dwmse_matrix_celsius03.shape[1]
            )

            if best_indices[2] == k and best_indices[3] == m:
                axes_object.plot(
                    best_indices[1], best_indices[0],
                    linestyle='None', marker=BEST_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=WHITE_COLOUR,
                    markeredgecolor=WHITE_COLOUR
                )

            if SELECTED_MARKER_INDICES[2] == k and SELECTED_MARKER_INDICES[3] == m:
                axes_object.plot(
                    SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
                    linestyle='None', marker=SELECTED_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=WHITE_COLOUR,
                    markeredgecolor=WHITE_COLOUR
                )

            axes_object.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
            axes_object.set_ylabel(y_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)

            title_string = r'Dual-weighted MSE ([$^{\circ}$C]$^3$)'
            title_string += (
                '\n{0:s} clustering with buffer dist = {1:d} px'
            ).format(
                'BACKWARDS' if DO_BACKWARDS_FLAGS_AXIS4[m] else 'FORWARD',
                BUFFER_DISTANCES_PX_AXIS3[k]
            )
            axes_object.set_title(title_string)

            dwmse_panel_file_name_matrix[k, m] = (
                '{0:s}/dwmse_buffer-distance-px={1:d}_'
                'do-backwards-clustering={2:d}.jpg'
            ).format(
                output_dir_name,
                BUFFER_DISTANCES_PX_AXIS3[k],
                DO_BACKWARDS_FLAGS_AXIS4[m]
            )

            print('Saving figure to: "{0:s}"...'.format(
                dwmse_panel_file_name_matrix[k, m]
            ))
            figure_object.savefig(
                dwmse_panel_file_name_matrix[k, m],
                dpi=FIGURE_RESOLUTION_DPI, pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)

            # Plot RMSE vs. hyperparameters.
            figure_object, axes_object = _plot_scores_2d(
                score_matrix=rmse_matrix_kelvins01[..., k, m],
                min_colour_value=_finite_percentile(rmse_matrix_kelvins01, 0),
                max_colour_value=_finite_percentile(rmse_matrix_kelvins01, 95),
                x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
            )

            this_index = numpy.nanargmin(numpy.ravel(rmse_matrix_kelvins01))
            best_indices = numpy.unravel_index(
                this_index, rmse_matrix_kelvins01.shape
            )

            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                BEST_MARKER_SIZE_GRID_CELLS / rmse_matrix_kelvins01.shape[1]
            )

            if best_indices[2] == k and best_indices[3] == m:
                axes_object.plot(
                    best_indices[1], best_indices[0],
                    linestyle='None', marker=BEST_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=WHITE_COLOUR,
                    markeredgecolor=WHITE_COLOUR
                )

            if SELECTED_MARKER_INDICES[2] == k and SELECTED_MARKER_INDICES[3] == m:
                axes_object.plot(
                    SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
                    linestyle='None', marker=SELECTED_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=WHITE_COLOUR,
                    markeredgecolor=WHITE_COLOUR
                )

            axes_object.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
            axes_object.set_ylabel(y_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)

            title_string = r'RMSE ($^{\circ}$C)'
            title_string += (
                '\n{0:s} clustering with buffer dist = {1:d} px'
            ).format(
                'BACKWARDS' if DO_BACKWARDS_FLAGS_AXIS4[m] else 'FORWARD',
                BUFFER_DISTANCES_PX_AXIS3[k]
            )
            axes_object.set_title(title_string)

            rmse_panel_file_name_matrix[k, m] = (
                '{0:s}/rmse_buffer-distance-px={1:d}_'
                'do-backwards-clustering={2:d}.jpg'
            ).format(
                output_dir_name,
                BUFFER_DISTANCES_PX_AXIS3[k],
                DO_BACKWARDS_FLAGS_AXIS4[m]
            )

            print('Saving figure to: "{0:s}"...'.format(
                rmse_panel_file_name_matrix[k, m]
            ))
            figure_object.savefig(
                rmse_panel_file_name_matrix[k, m],
                dpi=FIGURE_RESOLUTION_DPI, pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)

            # Plot MAE vs. hyperparameters.
            figure_object, axes_object = _plot_scores_2d(
                score_matrix=mae_matrix_kelvins01[..., k, m],
                min_colour_value=_finite_percentile(mae_matrix_kelvins01, 0),
                max_colour_value=_finite_percentile(mae_matrix_kelvins01, 95),
                x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
            )

            this_index = numpy.nanargmin(numpy.ravel(mae_matrix_kelvins01))
            best_indices = numpy.unravel_index(
                this_index, mae_matrix_kelvins01.shape
            )

            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                BEST_MARKER_SIZE_GRID_CELLS / mae_matrix_kelvins01.shape[1]
            )

            if best_indices[2] == k and best_indices[3] == m:
                axes_object.plot(
                    best_indices[1], best_indices[0],
                    linestyle='None', marker=BEST_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=WHITE_COLOUR,
                    markeredgecolor=WHITE_COLOUR
                )

            if SELECTED_MARKER_INDICES[2] == k and SELECTED_MARKER_INDICES[3] == m:
                axes_object.plot(
                    SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
                    linestyle='None', marker=SELECTED_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=WHITE_COLOUR,
                    markeredgecolor=WHITE_COLOUR
                )

            axes_object.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
            axes_object.set_ylabel(y_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)

            title_string = r'MAE ($^{\circ}$C)'
            title_string += (
                '\n{0:s} clustering with buffer dist = {1:d} px'
            ).format(
                'BACKWARDS' if DO_BACKWARDS_FLAGS_AXIS4[m] else 'FORWARD',
                BUFFER_DISTANCES_PX_AXIS3[k]
            )
            axes_object.set_title(title_string)

            mae_panel_file_name_matrix[k, m] = (
                '{0:s}/mae_buffer-distance-px={1:d}_'
                'do-backwards-clustering={2:d}.jpg'
            ).format(
                output_dir_name,
                BUFFER_DISTANCES_PX_AXIS3[k],
                DO_BACKWARDS_FLAGS_AXIS4[m]
            )

            print('Saving figure to: "{0:s}"...'.format(
                mae_panel_file_name_matrix[k, m]
            ))
            figure_object.savefig(
                mae_panel_file_name_matrix[k, m],
                dpi=FIGURE_RESOLUTION_DPI, pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)

            # Plot bias vs. hyperparameters.
            figure_object, axes_object = _plot_scores_2d(
                score_matrix=bias_matrix_kelvins01[..., k, m],
                min_colour_value=None,
                max_colour_value=_finite_percentile(
                    numpy.absolute(bias_matrix_kelvins01), 95
                ),
                x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
            )

            this_index = numpy.nanargmin(numpy.ravel(
                numpy.absolute(bias_matrix_kelvins01)
            ))
            best_indices = numpy.unravel_index(
                this_index, bias_matrix_kelvins01.shape
            )

            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                BEST_MARKER_SIZE_GRID_CELLS / bias_matrix_kelvins01.shape[1]
            )

            if best_indices[2] == k and best_indices[3] == m:
                axes_object.plot(
                    best_indices[1], best_indices[0],
                    linestyle='None', marker=BEST_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=BLACK_COLOUR,
                    markeredgecolor=BLACK_COLOUR
                )

            if SELECTED_MARKER_INDICES[2] == k and SELECTED_MARKER_INDICES[3] == m:
                axes_object.plot(
                    SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
                    linestyle='None', marker=SELECTED_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=BLACK_COLOUR,
                    markeredgecolor=BLACK_COLOUR
                )

            axes_object.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
            axes_object.set_ylabel(y_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)

            title_string = r'Bias ($^{\circ}$C)'
            title_string += (
                '\n{0:s} clustering with buffer dist = {1:d} px'
            ).format(
                'BACKWARDS' if DO_BACKWARDS_FLAGS_AXIS4[m] else 'FORWARD',
                BUFFER_DISTANCES_PX_AXIS3[k]
            )
            axes_object.set_title(title_string)

            bias_panel_file_name_matrix[k, m] = (
                '{0:s}/bias_buffer-distance-px={1:d}_'
                'do-backwards-clustering={2:d}.jpg'
            ).format(
                output_dir_name,
                BUFFER_DISTANCES_PX_AXIS3[k],
                DO_BACKWARDS_FLAGS_AXIS4[m]
            )

            print('Saving figure to: "{0:s}"...'.format(
                bias_panel_file_name_matrix[k, m]
            ))
            figure_object.savefig(
                bias_panel_file_name_matrix[k, m],
                dpi=FIGURE_RESOLUTION_DPI, pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)

            # Plot stdev bias vs. hyperparameters.
            figure_object, axes_object = _plot_scores_2d(
                score_matrix=stdev_bias_matrix_kelvins01[..., k, m],
                min_colour_value=None,
                max_colour_value=_finite_percentile(
                    numpy.absolute(stdev_bias_matrix_kelvins01), 95
                ),
                x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
            )

            this_index = numpy.nanargmin(numpy.ravel(
                numpy.absolute(stdev_bias_matrix_kelvins01)
            ))
            best_indices = numpy.unravel_index(
                this_index, stdev_bias_matrix_kelvins01.shape
            )

            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                BEST_MARKER_SIZE_GRID_CELLS / stdev_bias_matrix_kelvins01.shape[1]
            )

            if best_indices[2] == k and best_indices[3] == m:
                axes_object.plot(
                    best_indices[1], best_indices[0],
                    linestyle='None', marker=BEST_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=BLACK_COLOUR,
                    markeredgecolor=BLACK_COLOUR
                )

            if SELECTED_MARKER_INDICES[2] == k and SELECTED_MARKER_INDICES[3] == m:
                axes_object.plot(
                    SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
                    linestyle='None', marker=SELECTED_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=BLACK_COLOUR,
                    markeredgecolor=BLACK_COLOUR
                )

            axes_object.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
            axes_object.set_ylabel(y_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)

            title_string = r'Standard-deviation bias ($^{\circ}$C)'
            title_string += (
                '\n{0:s} clustering with buffer dist = {1:d} px'
            ).format(
                'BACKWARDS' if DO_BACKWARDS_FLAGS_AXIS4[m] else 'FORWARD',
                BUFFER_DISTANCES_PX_AXIS3[k]
            )
            axes_object.set_title(title_string)

            stdev_bias_panel_file_name_matrix[k, m] = (
                '{0:s}/stdev_bias_buffer-distance-px={1:d}_'
                'do-backwards-clustering={2:d}.jpg'
            ).format(
                output_dir_name,
                BUFFER_DISTANCES_PX_AXIS3[k],
                DO_BACKWARDS_FLAGS_AXIS4[m]
            )

            print('Saving figure to: "{0:s}"...'.format(
                stdev_bias_panel_file_name_matrix[k, m]
            ))
            figure_object.savefig(
                stdev_bias_panel_file_name_matrix[k, m],
                dpi=FIGURE_RESOLUTION_DPI, pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)

            # Plot spatial-minimum bias vs. hyperparameters.
            figure_object, axes_object = _plot_scores_2d(
                score_matrix=spatial_min_bias_matrix_kelvins01[..., k, m],
                min_colour_value=None,
                max_colour_value=_finite_percentile(
                    numpy.absolute(spatial_min_bias_matrix_kelvins01), 95
                ),
                x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
            )

            this_index = numpy.nanargmin(numpy.ravel(
                numpy.absolute(spatial_min_bias_matrix_kelvins01)
            ))
            best_indices = numpy.unravel_index(
                this_index, spatial_min_bias_matrix_kelvins01.shape
            )

            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                BEST_MARKER_SIZE_GRID_CELLS /
                spatial_min_bias_matrix_kelvins01.shape[1]
            )

            if best_indices[2] == k and best_indices[3] == m:
                axes_object.plot(
                    best_indices[1], best_indices[0],
                    linestyle='None', marker=BEST_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=BLACK_COLOUR,
                    markeredgecolor=BLACK_COLOUR
                )

            if SELECTED_MARKER_INDICES[2] == k and SELECTED_MARKER_INDICES[3] == m:
                axes_object.plot(
                    SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
                    linestyle='None', marker=SELECTED_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=BLACK_COLOUR,
                    markeredgecolor=BLACK_COLOUR
                )

            axes_object.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
            axes_object.set_ylabel(y_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)

            title_string = r'Spatial-minimum bias ($^{\circ}$C)'
            title_string += (
                '\n{0:s} clustering with buffer dist = {1:d} px'
            ).format(
                'BACKWARDS' if DO_BACKWARDS_FLAGS_AXIS4[m] else 'FORWARD',
                BUFFER_DISTANCES_PX_AXIS3[k]
            )
            axes_object.set_title(title_string)

            spatial_min_bias_panel_file_name_matrix[k, m] = (
                '{0:s}/spatial_min_bias_buffer-distance-px={1:d}_'
                'do-backwards-clustering={2:d}.jpg'
            ).format(
                output_dir_name,
                BUFFER_DISTANCES_PX_AXIS3[k],
                DO_BACKWARDS_FLAGS_AXIS4[m]
            )

            print('Saving figure to: "{0:s}"...'.format(
                spatial_min_bias_panel_file_name_matrix[k, m]
            ))
            figure_object.savefig(
                spatial_min_bias_panel_file_name_matrix[k, m],
                dpi=FIGURE_RESOLUTION_DPI, pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)

            # Plot spatial-maximum bias vs. hyperparameters.
            figure_object, axes_object = _plot_scores_2d(
                score_matrix=spatial_max_bias_matrix_kelvins01[..., k, m],
                min_colour_value=None,
                max_colour_value=_finite_percentile(
                    numpy.absolute(spatial_max_bias_matrix_kelvins01), 95
                ),
                x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
            )

            this_index = numpy.nanargmin(numpy.ravel(
                numpy.absolute(spatial_max_bias_matrix_kelvins01)
            ))
            best_indices = numpy.unravel_index(
                this_index, spatial_max_bias_matrix_kelvins01.shape
            )

            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                BEST_MARKER_SIZE_GRID_CELLS /
                spatial_max_bias_matrix_kelvins01.shape[1]
            )

            if best_indices[2] == k and best_indices[3] == m:
                axes_object.plot(
                    best_indices[1], best_indices[0],
                    linestyle='None', marker=BEST_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=BLACK_COLOUR,
                    markeredgecolor=BLACK_COLOUR
                )

            if SELECTED_MARKER_INDICES[2] == k and SELECTED_MARKER_INDICES[3] == m:
                axes_object.plot(
                    SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
                    linestyle='None', marker=SELECTED_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=BLACK_COLOUR,
                    markeredgecolor=BLACK_COLOUR
                )

            axes_object.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
            axes_object.set_ylabel(y_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)

            title_string = r'Spatial-maximum bias ($^{\circ}$C)'
            title_string += (
                '\n{0:s} clustering with buffer dist = {1:d} px'
            ).format(
                'BACKWARDS' if DO_BACKWARDS_FLAGS_AXIS4[m] else 'FORWARD',
                BUFFER_DISTANCES_PX_AXIS3[k]
            )
            axes_object.set_title(title_string)

            spatial_max_bias_panel_file_name_matrix[k, m] = (
                '{0:s}/spatial_max_bias_buffer-distance-px={1:d}_'
                'do-backwards-clustering={2:d}.jpg'
            ).format(
                output_dir_name,
                BUFFER_DISTANCES_PX_AXIS3[k],
                DO_BACKWARDS_FLAGS_AXIS4[m]
            )

            print('Saving figure to: "{0:s}"...'.format(
                spatial_max_bias_panel_file_name_matrix[k, m]
            ))
            figure_object.savefig(
                spatial_max_bias_panel_file_name_matrix[k, m],
                dpi=FIGURE_RESOLUTION_DPI, pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)

            # Plot correlation vs. hyperparameters.
            figure_object, axes_object = _plot_scores_2d(
                score_matrix=correlation_matrix[..., k, m],
                min_colour_value=_finite_percentile(correlation_matrix, 5),
                max_colour_value=_finite_percentile(correlation_matrix, 100),
                x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
            )

            this_index = numpy.nanargmax(numpy.ravel(correlation_matrix))
            best_indices = numpy.unravel_index(
                this_index, correlation_matrix.shape
            )

            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                BEST_MARKER_SIZE_GRID_CELLS / correlation_matrix.shape[1]
            )

            if best_indices[2] == k and best_indices[3] == m:
                axes_object.plot(
                    best_indices[1], best_indices[0],
                    linestyle='None', marker=BEST_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=BLACK_COLOUR,
                    markeredgecolor=BLACK_COLOUR
                )

            if SELECTED_MARKER_INDICES[2] == k and SELECTED_MARKER_INDICES[3] == m:
                axes_object.plot(
                    SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
                    linestyle='None', marker=SELECTED_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=BLACK_COLOUR,
                    markeredgecolor=BLACK_COLOUR
                )

            axes_object.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
            axes_object.set_ylabel(y_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)

            title_string = (
                'Correlation\n{0:s} clustering with buffer dist = {1:d} px'
            ).format(
                'BACKWARDS' if DO_BACKWARDS_FLAGS_AXIS4[m] else 'FORWARD',
                BUFFER_DISTANCES_PX_AXIS3[k]
            )
            axes_object.set_title(title_string)

            correlation_panel_file_name_matrix[k, m] = (
                '{0:s}/correlation_buffer-distance-px={1:d}_'
                'do-backwards-clustering={2:d}.jpg'
            ).format(
                output_dir_name,
                BUFFER_DISTANCES_PX_AXIS3[k],
                DO_BACKWARDS_FLAGS_AXIS4[m]
            )

            print('Saving figure to: "{0:s}"...'.format(
                correlation_panel_file_name_matrix[k, m]
            ))
            figure_object.savefig(
                correlation_panel_file_name_matrix[k, m],
                dpi=FIGURE_RESOLUTION_DPI, pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)

            # Plot KGE vs. hyperparameters.
            figure_object, axes_object = _plot_scores_2d(
                score_matrix=kge_matrix[..., k, m],
                min_colour_value=_finite_percentile(kge_matrix, 5),
                max_colour_value=_finite_percentile(kge_matrix, 100),
                x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
            )

            this_index = numpy.nanargmax(numpy.ravel(kge_matrix))
            best_indices = numpy.unravel_index(
                this_index, kge_matrix.shape
            )

            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                BEST_MARKER_SIZE_GRID_CELLS / kge_matrix.shape[1]
            )

            if best_indices[2] == k and best_indices[3] == m:
                axes_object.plot(
                    best_indices[1], best_indices[0],
                    linestyle='None', marker=BEST_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=BLACK_COLOUR,
                    markeredgecolor=BLACK_COLOUR
                )

            if SELECTED_MARKER_INDICES[2] == k and SELECTED_MARKER_INDICES[3] == m:
                axes_object.plot(
                    SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
                    linestyle='None', marker=SELECTED_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=BLACK_COLOUR,
                    markeredgecolor=BLACK_COLOUR
                )

            axes_object.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
            axes_object.set_ylabel(y_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)

            title_string = (
                'KGE\n{0:s} clustering with buffer dist = {1:d} px'
            ).format(
                'BACKWARDS' if DO_BACKWARDS_FLAGS_AXIS4[m] else 'FORWARD',
                BUFFER_DISTANCES_PX_AXIS3[k]
            )
            axes_object.set_title(title_string)

            kge_panel_file_name_matrix[k, m] = (
                '{0:s}/kge_buffer-distance-px={1:d}_'
                'do-backwards-clustering={2:d}.jpg'
            ).format(
                output_dir_name,
                BUFFER_DISTANCES_PX_AXIS3[k],
                DO_BACKWARDS_FLAGS_AXIS4[m]
            )

            print('Saving figure to: "{0:s}"...'.format(
                kge_panel_file_name_matrix[k, m]
            ))
            figure_object.savefig(
                kge_panel_file_name_matrix[k, m],
                dpi=FIGURE_RESOLUTION_DPI, pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)

            # Plot reliability vs. hyperparameters.
            figure_object, axes_object = _plot_scores_2d(
                score_matrix=reliability_matrix_kelvins02[..., k, m],
                min_colour_value=_finite_percentile(reliability_matrix_kelvins02, 0),
                max_colour_value=_finite_percentile(reliability_matrix_kelvins02, 95),
                x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
            )

            this_index = numpy.nanargmin(numpy.ravel(reliability_matrix_kelvins02))
            best_indices = numpy.unravel_index(
                this_index, reliability_matrix_kelvins02.shape
            )

            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                BEST_MARKER_SIZE_GRID_CELLS / kge_matrix.shape[1]
            )

            if best_indices[2] == k and best_indices[3] == m:
                axes_object.plot(
                    best_indices[1], best_indices[0],
                    linestyle='None', marker=BEST_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=WHITE_COLOUR,
                    markeredgecolor=WHITE_COLOUR
                )

            if SELECTED_MARKER_INDICES[2] == k and SELECTED_MARKER_INDICES[3] == m:
                axes_object.plot(
                    SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
                    linestyle='None', marker=SELECTED_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=WHITE_COLOUR,
                    markeredgecolor=WHITE_COLOUR
                )

            axes_object.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
            axes_object.set_ylabel(y_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)

            title_string = r'Reliability ([$^{\circ}$C]$^2$)'
            title_string += (
                '\n{0:s} clustering with buffer dist = {1:d} px'
            ).format(
                'BACKWARDS' if DO_BACKWARDS_FLAGS_AXIS4[m] else 'FORWARD',
                BUFFER_DISTANCES_PX_AXIS3[k]
            )
            axes_object.set_title(title_string)

            reliability_panel_file_name_matrix[k, m] = (
                '{0:s}/reliability_buffer-distance-px={1:d}_'
                'do-backwards-clustering={2:d}.jpg'
            ).format(
                output_dir_name,
                BUFFER_DISTANCES_PX_AXIS3[k],
                DO_BACKWARDS_FLAGS_AXIS4[m]
            )

            print('Saving figure to: "{0:s}"...'.format(
                reliability_panel_file_name_matrix[k, m]
            ))
            figure_object.savefig(
                reliability_panel_file_name_matrix[k, m],
                dpi=FIGURE_RESOLUTION_DPI, pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)

    all_panel_file_names = numpy.concatenate([
        numpy.ravel(dwmse_panel_file_name_matrix),
        numpy.ravel(rmse_panel_file_name_matrix),
        numpy.ravel(mae_panel_file_name_matrix),
        numpy.ravel(bias_panel_file_name_matrix),
        numpy.ravel(stdev_bias_panel_file_name_matrix),
        numpy.ravel(spatial_min_bias_panel_file_name_matrix),
        numpy.ravel(spatial_max_bias_panel_file_name_matrix),
        numpy.ravel(correlation_panel_file_name_matrix),
        numpy.ravel(kge_panel_file_name_matrix),
        numpy.ravel(reliability_panel_file_name_matrix)
    ])

    for this_panel_file_name in all_panel_file_names:
        print('Resizing panel: "{0:s}"...'.format(this_panel_file_name))
        imagemagick_utils.resize_image(
            input_file_name=this_panel_file_name,
            output_file_name=this_panel_file_name,
            output_size_pixels=int(2.5e6)
        )

    concat_figure_file_names = [
        '{0:s}/dwmse.jpg'.format(output_dir_name),
        '{0:s}/rmse.jpg'.format(output_dir_name),
        '{0:s}/mae.jpg'.format(output_dir_name),
        '{0:s}/bias.jpg'.format(output_dir_name),
        '{0:s}/stdev_bias.jpg'.format(output_dir_name),
        '{0:s}/spatial_min_bias.jpg'.format(output_dir_name),
        '{0:s}/spatial_max_bias.jpg'.format(output_dir_name),
        '{0:s}/correlation.jpg'.format(output_dir_name),
        '{0:s}/kge.jpg'.format(output_dir_name),
        '{0:s}/reliability.jpg'.format(output_dir_name)
    ]

    panel_file_name_matrices = [
        dwmse_panel_file_name_matrix,
        rmse_panel_file_name_matrix,
        mae_panel_file_name_matrix,
        bias_panel_file_name_matrix,
        stdev_bias_panel_file_name_matrix,
        spatial_min_bias_panel_file_name_matrix,
        spatial_max_bias_panel_file_name_matrix,
        correlation_panel_file_name_matrix,
        kge_panel_file_name_matrix,
        reliability_panel_file_name_matrix
    ]

    num_panel_rows = int(numpy.floor(
        numpy.sqrt(axis3_length)
    ))
    num_panel_columns = int(numpy.ceil(
        float(axis4_length) / num_panel_rows
    ))

    for i in range(len(concat_figure_file_names)):
        print('Concatenating panels to: "{0:s}"...'.format(
            concat_figure_file_names[i]
        ))
        imagemagick_utils.concatenate_images(
            input_file_names=numpy.ravel(panel_file_name_matrices[i]).tolist(),
            output_file_name=concat_figure_file_names[i],
            num_panel_rows=num_panel_rows,
            num_panel_columns=num_panel_columns
        )
        imagemagick_utils.resize_image(
            input_file_name=concat_figure_file_names[i],
            output_file_name=concat_figure_file_names[i],
            output_size_pixels=int(1e7)
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        experiment_dir_name=getattr(INPUT_ARG_OBJECT, EXPERIMENT_DIR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
