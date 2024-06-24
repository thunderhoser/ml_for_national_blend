"""Plots hyperparameter grids for Experiment 4.

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

import file_system_utils
import gg_plotting_utils
import make_templates_exp04_temp_only_more_models as make_templates_exp04
import evaluation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NWP_MODEL_SET_STRINGS_AXIS1 = make_templates_exp04._get_hyperparams()[0]
NWP_MODEL_SET_STRINGS_AXIS1 = NWP_MODEL_SET_STRINGS_AXIS1[::4]
PREDICTOR_SET_STRINGS_AXIS2 = (
    make_templates_exp04.UNIQUE_PREDICTOR_SET_DESCRIPTIONS
)

BEST_MARKER_TYPE = '*'
BEST_MARKER_SIZE_GRID_CELLS = 0.075
WHITE_COLOUR = numpy.full(3, 1.)
BLACK_COLOUR = numpy.full(3, 0.)

SELECTED_MARKER_TYPE = 'o'
SELECTED_MARKER_SIZE_GRID_CELLS = 0.075
SELECTED_MARKER_INDICES = numpy.array([0, 0], dtype=int)

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


def _nwp_model_set_string_to_fancy(nwp_model_set_string):
    """Fancifies string describing a set of NWP models.

    :param nwp_model_set_string: Original string (not fancy).
    :return: fancy_set_string: New string (fancy).
    """

    fancy_set_string = nwp_model_set_string.replace('-', '/')
    fancy_set_string = fancy_set_string.replace('wrf_arw', 'WRF')
    fancy_set_string = fancy_set_string.replace('nam_nest', 'NAMN')
    fancy_set_string = fancy_set_string.replace('gridded_gfs_mos', 'GMOS')
    fancy_set_string = fancy_set_string.replace('nam', 'NAM')
    fancy_set_string = fancy_set_string.replace('rap', 'RAP')
    fancy_set_string = fancy_set_string.replace('gfs', 'GFS')
    fancy_set_string = fancy_set_string.replace('hrrr', 'HRRR')
    fancy_set_string = fancy_set_string.replace('gefs', 'GEFS')
    fancy_set_string = fancy_set_string.replace('ecmwf', 'ECMWF')
    fancy_set_string = fancy_set_string.replace('gridded_lamp', 'GLAMP')

    return fancy_set_string


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
        font_size=FONT_SIZE
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.2g}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    return figure_object, axes_object


def _print_ranking_one_score(score_matrix, score_name):
    """Prints ranking for one score.

    M = number of NWP-model sets
    P = number of predictor sets

    :param score_matrix: M-by-P numpy array of score values.
    :param score_name: Name of score.
    """

    scores_1d = numpy.ravel(score_matrix)
    scores_1d[numpy.isnan(scores_1d)] = numpy.inf
    sort_indices_1d = numpy.argsort(scores_1d)
    i_sort_indices, j_sort_indices = numpy.unravel_index(
        sort_indices_1d, score_matrix.shape
    )

    for m in range(len(i_sort_indices)):
        i = i_sort_indices[m]
        j = j_sort_indices[m]

        print((
            '{0:d}th-lowest {1:s} = {2:.4g} ... '
            'NWP models = {3:s} ... predictors = {4:s}'
        ).format(
            m + 1, score_name, score_matrix[i, j],
            _nwp_model_set_string_to_fancy(NWP_MODEL_SET_STRINGS_AXIS1[i]),
            PREDICTOR_SET_STRINGS_AXIS2[j]
        ))


def _print_ranking_all_scores(
        dwmse_matrix_celsius03, rmse_matrix_kelvins01, mae_matrix_kelvins01,
        bias_matrix_kelvins01, stdev_bias_matrix_kelvins01,
        spatial_min_bias_matrix_kelvins01, spatial_max_bias_matrix_kelvins01,
        correlation_matrix, kge_matrix, reliability_matrix_kelvins02):
    """Prints ranking for all scores.

    M = number of NWP-model sets
    P = number of predictor sets

    :param dwmse_matrix_celsius03: M-by-P numpy array of DWMSE (dual-weighted
        mean squared error) values.
    :param rmse_matrix_kelvins01: M-by-P numpy array of root mean squared error
        (RMSE) values.
    :param mae_matrix_kelvins01: M-by-P numpy array of mean absolute errors
        (MAE).
    :param bias_matrix_kelvins01: M-by-P numpy array of bias values.
    :param stdev_bias_matrix_kelvins01: M-by-P numpy array of standard-deviation
        biases.
    :param spatial_min_bias_matrix_kelvins01: M-by-P numpy array of
        spatial-minimum biases.
    :param spatial_max_bias_matrix_kelvins01: M-by-P numpy array of
        spatial-maximum biases.
    :param correlation_matrix: M-by-P numpy array of Pearson correlations.
    :param kge_matrix: M-by-P numpy array of Kling-Gupta efficiency (KGE)
        values.
    :param reliability_matrix_kelvins02: M-by-P numpy array of reliability
        values.
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
    i_sort_indices, j_sort_indices = numpy.unravel_index(
        sort_indices_1d, overall_rank_matrix.shape
    )

    for m in range(len(i_sort_indices)):
        i = i_sort_indices[m]
        j = j_sort_indices[m]

        print((
            'NWP models = {0:s} ... predictors = {1:s} ... '
            'DWMSE/RMSE/MAE ranks = {2:.1f}, {3:.1f}, {4:.1f} ... '
            'bias and stdev-bias ranks = {5:.1f}, {6:.1f} ... '
            'spatial-min- and spatial-max-bias ranks = {7:.1f}, {8:.1f} ... '
            'correlation/KGE/reliability ranks = {9:.1f}, {10:.1f}, {11:.1f}'
        ).format(
            _nwp_model_set_string_to_fancy(NWP_MODEL_SET_STRINGS_AXIS1[i]),
            PREDICTOR_SET_STRINGS_AXIS2[j],
            dwmse_rank_matrix[i, j], rmse_rank_matrix[i, j], mae_rank_matrix[i, j],
            bias_rank_matrix[i, j], stdev_bias_rank_matrix[i, j],
            spatial_min_bias_rank_matrix[i, j], spatial_max_bias_rank_matrix[i, j],
            correlation_rank_matrix[i, j], kge_rank_matrix[i, j], reliability_rank_matrix[i, j]
        ))


def _run(experiment_dir_name, output_dir_name):
    """Plots hyperparameter grids for Experiment 4.

    This is effectively the main method.

    :param experiment_dir_name: See documentation at top of this script.
    :param output_dir_name: Same.
    """

    num_nwp_model_sets = len(NWP_MODEL_SET_STRINGS_AXIS1)
    num_predictor_sets = len(PREDICTOR_SET_STRINGS_AXIS2)

    y_tick_labels = [
        _nwp_model_set_string_to_fancy(n) for n in NWP_MODEL_SET_STRINGS_AXIS1
    ]
    x_tick_labels = [
        '{0:s}'.format(p) for p in PREDICTOR_SET_STRINGS_AXIS2
    ]
    x_tick_labels = [l.replace('surface', 'Surface') for l in x_tick_labels]
    x_tick_labels = [l.replace('-', ', ') for l in x_tick_labels]

    y_axis_label = 'NWP models'
    x_axis_label = 'Predictor variables'

    dimensions = (num_nwp_model_sets, num_predictor_sets)

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

    for i in range(num_nwp_model_sets):
        for j in range(num_predictor_sets):
            this_eval_file_name = (
                '{0:s}/nwp-model-names={1:s}_nwp-fields={2:s}/validation/'
                'ungridded_evaluation.nc'
            ).format(
                experiment_dir_name,
                NWP_MODEL_SET_STRINGS_AXIS1[i],
                PREDICTOR_SET_STRINGS_AXIS2[j]
            )

            if not os.path.isfile(this_eval_file_name):
                continue

            print('Reading data from: "{0:s}"...'.format(this_eval_file_name))
            this_eval_table_xarray = evaluation.read_file(this_eval_file_name)
            etx = this_eval_table_xarray

            dwmse_matrix_celsius03[i, j] = numpy.mean(
                etx[evaluation.DWMSE_KEY].values[0, :]
            )
            rmse_matrix_kelvins01[i, j] = numpy.sqrt(numpy.mean(
                etx[evaluation.MSE_KEY].values[0, :]
            ))
            mae_matrix_kelvins01[i, j] = numpy.mean(
                etx[evaluation.MAE_KEY].values[0, :]
            )
            bias_matrix_kelvins01[i, j] = numpy.mean(
                etx[evaluation.BIAS_KEY].values[0, :]
            )
            stdev_bias_matrix_kelvins01[i, j] = numpy.mean(
                etx[evaluation.PREDICTION_STDEV_KEY].values[0, :] -
                etx[evaluation.TARGET_STDEV_KEY].values[0, :]
            )
            spatial_min_bias_matrix_kelvins01[i, j] = numpy.mean(
                etx[evaluation.SPATIAL_MIN_BIAS_KEY].values[0, :]
            )
            spatial_max_bias_matrix_kelvins01[i, j] = numpy.mean(
                etx[evaluation.SPATIAL_MAX_BIAS_KEY].values[0, :]
            )
            correlation_matrix[i, j] = numpy.mean(
                etx[evaluation.CORRELATION_KEY].values[0, :]
            )
            kge_matrix[i, j] = numpy.mean(
                etx[evaluation.KGE_KEY].values[0, :]
            )
            reliability_matrix_kelvins02[i, j] = numpy.mean(
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

    # Plot DWMSE vs. hyperparameters.
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=dwmse_matrix_celsius03,
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

    axes_object.plot(
        best_indices[1], best_indices[0],
        linestyle='None', marker=BEST_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=WHITE_COLOUR,
        markeredgecolor=WHITE_COLOUR
    )
    axes_object.plot(
        SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
        linestyle='None', marker=SELECTED_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=WHITE_COLOUR,
        markeredgecolor=WHITE_COLOUR
    )

    axes_object.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
    axes_object.set_ylabel(y_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
    axes_object.set_title(r'Dual-weighted MSE ([$^{\circ}$C]$^3$)')

    output_file_name = '{0:s}/dwmse.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot RMSE vs. hyperparameters.
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=rmse_matrix_kelvins01,
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

    axes_object.plot(
        best_indices[1], best_indices[0],
        linestyle='None', marker=BEST_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=WHITE_COLOUR,
        markeredgecolor=WHITE_COLOUR
    )
    axes_object.plot(
        SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
        linestyle='None', marker=SELECTED_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=WHITE_COLOUR,
        markeredgecolor=WHITE_COLOUR
    )

    axes_object.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
    axes_object.set_ylabel(y_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
    axes_object.set_title(r'RMSE ($^{\circ}$C)')

    output_file_name = '{0:s}/rmse.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot MAE vs. hyperparameters.
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=mae_matrix_kelvins01,
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

    axes_object.plot(
        best_indices[1], best_indices[0],
        linestyle='None', marker=BEST_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=WHITE_COLOUR,
        markeredgecolor=WHITE_COLOUR
    )
    axes_object.plot(
        SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
        linestyle='None', marker=SELECTED_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=WHITE_COLOUR,
        markeredgecolor=WHITE_COLOUR
    )

    axes_object.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
    axes_object.set_ylabel(y_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
    axes_object.set_title(r'MAE ($^{\circ}$C)')

    output_file_name = '{0:s}/mae.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot bias vs. hyperparameters.
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=bias_matrix_kelvins01,
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

    axes_object.plot(
        best_indices[1], best_indices[0],
        linestyle='None', marker=BEST_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=BLACK_COLOUR,
        markeredgecolor=BLACK_COLOUR
    )
    axes_object.plot(
        SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
        linestyle='None', marker=SELECTED_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=BLACK_COLOUR,
        markeredgecolor=BLACK_COLOUR
    )

    axes_object.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
    axes_object.set_ylabel(y_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
    axes_object.set_title(r'Bias ($^{\circ}$C)')

    output_file_name = '{0:s}/bias.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot stdev bias vs. hyperparameters.
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=stdev_bias_matrix_kelvins01,
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

    axes_object.plot(
        best_indices[1], best_indices[0],
        linestyle='None', marker=BEST_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=BLACK_COLOUR,
        markeredgecolor=BLACK_COLOUR
    )
    axes_object.plot(
        SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
        linestyle='None', marker=SELECTED_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=BLACK_COLOUR,
        markeredgecolor=BLACK_COLOUR
    )

    axes_object.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
    axes_object.set_ylabel(y_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
    axes_object.set_title(r'Standard-deviation bias ($^{\circ}$C)')

    output_file_name = '{0:s}/stdev_bias.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot spatial-minimum bias vs. hyperparameters.
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=spatial_min_bias_matrix_kelvins01,
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
        BEST_MARKER_SIZE_GRID_CELLS / spatial_min_bias_matrix_kelvins01.shape[1]
    )

    axes_object.plot(
        best_indices[1], best_indices[0],
        linestyle='None', marker=BEST_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=BLACK_COLOUR,
        markeredgecolor=BLACK_COLOUR
    )
    axes_object.plot(
        SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
        linestyle='None', marker=SELECTED_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=BLACK_COLOUR,
        markeredgecolor=BLACK_COLOUR
    )

    axes_object.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
    axes_object.set_ylabel(y_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
    axes_object.set_title(r'Spatial-minimum bias ($^{\circ}$C)')

    output_file_name = '{0:s}/spatial_min_bias.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot spatial-maximum bias vs. hyperparameters.
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=spatial_max_bias_matrix_kelvins01,
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
        BEST_MARKER_SIZE_GRID_CELLS / spatial_max_bias_matrix_kelvins01.shape[1]
    )

    axes_object.plot(
        best_indices[1], best_indices[0],
        linestyle='None', marker=BEST_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=BLACK_COLOUR,
        markeredgecolor=BLACK_COLOUR
    )
    axes_object.plot(
        SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
        linestyle='None', marker=SELECTED_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=BLACK_COLOUR,
        markeredgecolor=BLACK_COLOUR
    )

    axes_object.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
    axes_object.set_ylabel(y_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
    axes_object.set_title(r'Spatial-maximum bias ($^{\circ}$C)')

    output_file_name = '{0:s}/spatial_max_bias.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot correlation vs. hyperparameters.
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=correlation_matrix,
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

    axes_object.plot(
        best_indices[1], best_indices[0],
        linestyle='None', marker=BEST_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=BLACK_COLOUR,
        markeredgecolor=BLACK_COLOUR
    )
    axes_object.plot(
        SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
        linestyle='None', marker=SELECTED_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=BLACK_COLOUR,
        markeredgecolor=BLACK_COLOUR
    )

    axes_object.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
    axes_object.set_ylabel(y_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
    axes_object.set_title('Correlation')

    output_file_name = '{0:s}/correlation.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot KGE vs. hyperparameters.
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=kge_matrix,
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

    axes_object.plot(
        best_indices[1], best_indices[0],
        linestyle='None', marker=BEST_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=BLACK_COLOUR,
        markeredgecolor=BLACK_COLOUR
    )
    axes_object.plot(
        SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
        linestyle='None', marker=SELECTED_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=BLACK_COLOUR,
        markeredgecolor=BLACK_COLOUR
    )

    axes_object.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
    axes_object.set_ylabel(y_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
    axes_object.set_title('Kling-Gupta efficiency')

    output_file_name = '{0:s}/kge.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot reliability vs. hyperparameters.
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=reliability_matrix_kelvins02,
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
        BEST_MARKER_SIZE_GRID_CELLS / reliability_matrix_kelvins02.shape[1]
    )

    axes_object.plot(
        best_indices[1], best_indices[0],
        linestyle='None', marker=BEST_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=WHITE_COLOUR,
        markeredgecolor=WHITE_COLOUR
    )
    axes_object.plot(
        SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
        linestyle='None', marker=SELECTED_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=WHITE_COLOUR,
        markeredgecolor=WHITE_COLOUR
    )

    axes_object.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
    axes_object.set_ylabel(y_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
    axes_object.set_title(r'RMSE ([$^{\circ}$C]$^2$)')

    output_file_name = '{0:s}/reliability.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        experiment_dir_name=getattr(INPUT_ARG_OBJECT, EXPERIMENT_DIR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
