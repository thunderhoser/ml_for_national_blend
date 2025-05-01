"""Plots hyperparameter grids for Experiment 21b.

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
import urma_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

WEIGHT_DECAY_VALUES_AXIS1 = numpy.array([1e-5, 3e-5, 5e-5, 7e-5, 9e-5])
ANNEALING_CYCLE_LENGTHS_AXIS2 = numpy.array([100, 200, 300, 400, 500], dtype=int)
ANNEALING_MIN_LEARNING_RATES_AXIS3 = numpy.array([1e-5, 3e-5, 5e-5, 7e-5, 9e-5])

TARGET_FIELD_NAMES = [
    urma_utils.TEMPERATURE_2METRE_NAME, urma_utils.DEWPOINT_2METRE_NAME,
    urma_utils.U_WIND_10METRE_NAME, urma_utils.V_WIND_10METRE_NAME,
    urma_utils.WIND_GUST_10METRE_NAME
]

TARGET_FIELD_TO_ENGLISH = {
    urma_utils.TEMPERATURE_2METRE_NAME: 'temperature',
    urma_utils.DEWPOINT_2METRE_NAME: 'dewpoint',
    urma_utils.U_WIND_10METRE_NAME: 'zonal wind',
    urma_utils.V_WIND_10METRE_NAME: 'meridional wind',
    urma_utils.WIND_GUST_10METRE_NAME: 'gust'
}

TARGET_FIELD_TO_UNITS = {
    urma_utils.TEMPERATURE_2METRE_NAME: 'K',
    urma_utils.DEWPOINT_2METRE_NAME: 'K',
    urma_utils.U_WIND_10METRE_NAME: r'm s$^{-1}$',
    urma_utils.V_WIND_10METRE_NAME: r'm s$^{-1}$',
    urma_utils.WIND_GUST_10METRE_NAME: r'm s$^{-1}$'
}

TARGET_FIELD_TO_SQUARED_UNITS = {
    urma_utils.TEMPERATURE_2METRE_NAME: r'K$^2$',
    urma_utils.DEWPOINT_2METRE_NAME: r'K$^2$',
    urma_utils.U_WIND_10METRE_NAME: r'm$^2$ s$^{-2}$',
    urma_utils.V_WIND_10METRE_NAME: r'm$^2$ s$^{-2}$',
    urma_utils.WIND_GUST_10METRE_NAME: r'm$^2$ s$^{-2}$'
}

TARGET_FIELD_TO_CUBED_UNITS = {
    urma_utils.TEMPERATURE_2METRE_NAME: r'K$^3$',
    urma_utils.DEWPOINT_2METRE_NAME: r'K$^3$',
    urma_utils.U_WIND_10METRE_NAME: r'm$^3$ s$^{-3}$',
    urma_utils.V_WIND_10METRE_NAME: r'm$^3$ s$^{-3}$',
    urma_utils.WIND_GUST_10METRE_NAME: r'm$^3$ s$^{-3}$'
}

BEST_MARKER_TYPE = '*'
BEST_MARKER_SIZE_GRID_CELLS = 0.075
WHITE_COLOUR = numpy.full(3, 1.)
BLACK_COLOUR = numpy.full(3, 0.)

SELECTED_MARKER_TYPE = 'o'
SELECTED_MARKER_SIZE_GRID_CELLS = 0.075
SELECTED_MARKER_INDICES = numpy.array([0, 0, 0], dtype=int)

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

    W = number of weight-decay values
    L = number of annealing-cycle lengths
    M = number of minimum learning rates

    :param score_matrix: W-by-L-by-M numpy array of score values.
    :param score_name: Name of score.
    """

    scores_1d = numpy.ravel(score_matrix)
    scores_1d[numpy.isnan(scores_1d)] = numpy.inf
    sort_indices_1d = numpy.argsort(scores_1d)
    w_sort_indices, l_sort_indices, m_sort_indices = (
        numpy.unravel_index(sort_indices_1d, score_matrix.shape)
    )

    for k in range(len(w_sort_indices)):
        w = w_sort_indices[k]
        l = l_sort_indices[k]
        m = m_sort_indices[k]

        print((
            r'{0:d}th-lowest {1:s} = {2:.4g} ... '
            r'weight decay = {3:.5f} ... '
            r'cycle length = {4:d} ... '
            r'min learning rate = {5:.5f}'
        ).format(
            k + 1, score_name, score_matrix[w, l, m],
            WEIGHT_DECAY_VALUES_AXIS1[w],
            ANNEALING_CYCLE_LENGTHS_AXIS2[l],
            ANNEALING_MIN_LEARNING_RATES_AXIS3[m]
        ))


def _print_ranking_all_scores(
        dwmse_matrix, rmse_matrix, mae_matrix, bias_matrix, stdev_bias_matrix,
        spatial_min_bias_matrix, spatial_max_bias_matrix,
        correlation_matrix, kge_matrix, reliability_matrix, target_field_name):
    """Prints ranking for all scores.

    This method might rank all scores for one target field, or all scores for
    all target fields.

    W = number of weight-decay values
    L = number of annealing-cycle lengths
    M = number of minimum learning rates
    F = number of target fields

    :param dwmse_matrix: W-by-L-by-M or W-by-L-by-M-by-F numpy array of dual-
        weighted mean squared errors.
    :param rmse_matrix: W-by-L-by-M or W-by-L-by-M-by-F numpy array of root
        mean squared errors.
    :param mae_matrix: W-by-L-by-M or W-by-L-by-M-by-F numpy array of mean
        absolute errors.
    :param bias_matrix: W-by-L-by-M or W-by-L-by-M-by-F numpy array of biases.
    :param stdev_bias_matrix: W-by-L-by-M or W-by-L-by-M-by-F numpy array of
        standard-deviation biases.
    :param spatial_min_bias_matrix: W-by-L-by-M or W-by-L-by-M-by-F numpy array
        of spatial-minimum biases.
    :param spatial_max_bias_matrix: W-by-L-by-M or W-by-L-by-M-by-F numpy array
        of spatial-max biases.
    :param correlation_matrix: W-by-L-by-M or W-by-L-by-M-by-F numpy array of
        correlations.
    :param kge_matrix: W-by-L-by-M or W-by-L-by-M-by-F numpy array of
        Kling-Gupta efficiencies.
    :param reliability_matrix: W-by-L-by-M or W-by-L-by-M-by-F numpy array of
        reliabilities.
    :param target_field_name: Name of target field represented by input arrays
        (could be "all").
    """

    if len(dwmse_matrix.shape) == 3:
        dwmse_matrix = numpy.stack([dwmse_matrix, dwmse_matrix], axis=-1)
        rmse_matrix = numpy.stack([rmse_matrix, rmse_matrix], axis=-1)
        mae_matrix = numpy.stack([mae_matrix, mae_matrix], axis=-1)
        bias_matrix = numpy.stack([bias_matrix, bias_matrix], axis=-1)
        stdev_bias_matrix = numpy.stack(
            [stdev_bias_matrix, stdev_bias_matrix], axis=-1
        )
        spatial_min_bias_matrix = numpy.stack(
            [spatial_min_bias_matrix, spatial_min_bias_matrix], axis=-1
        )
        spatial_max_bias_matrix = numpy.stack(
            [spatial_max_bias_matrix, spatial_max_bias_matrix], axis=-1
        )
        correlation_matrix = numpy.stack(
            [correlation_matrix, correlation_matrix], axis=-1
        )
        kge_matrix = numpy.stack([kge_matrix, kge_matrix], axis=-1)
        reliability_matrix = numpy.stack(
            [reliability_matrix, reliability_matrix], axis=-1
        )
    
    def rank_one_metric(metric_matrix):
        """Ranks values of one metric across all hyperparams.
        
        And potentially across all target fields.
        
        :param metric_matrix: W-by-L-by-M or W-by-L-by-M-by-F numpy array of
            metric values.
        :return: rank_matrix: W-by-L-by-M numpy array of ranks.
        """

        num_target_fields = metric_matrix.shape[-1]
        mm = metric_matrix
        
        rank_matrix_by_field = [
            numpy.reshape(
                rankdata(
                    numpy.nan_to_num(numpy.ravel(mm[..., f]), nan=numpy.inf),
                    method='average'
                ),
                mm[..., 0].shape
            )
            for f in range(num_target_fields)
        ]

        print('SHAPE of metric_matrix = {0:s}'.format(str(metric_matrix.shape)))
        print('SHAPE of rank_matrix_by_field = {0:s}'.format(str([rm.shape for rm in rank_matrix_by_field])))
        print('SHAPE of rank_matrix = {0:s}'.format(str(numpy.mean(numpy.stack(rank_matrix_by_field), axis=-1).shape)))
    
        return numpy.mean(numpy.stack(rank_matrix_by_field), axis=-1)

    dwmse_rank_matrix = rank_one_metric(dwmse_matrix)
    rmse_rank_matrix = rank_one_metric(rmse_matrix)
    mae_rank_matrix = rank_one_metric(mae_matrix)
    bias_rank_matrix = rank_one_metric(numpy.absolute(bias_matrix))
    stdev_bias_rank_matrix = rank_one_metric(numpy.absolute(stdev_bias_matrix))
    spatial_min_bias_rank_matrix = rank_one_metric(
        numpy.absolute(spatial_min_bias_matrix)
    )
    spatial_max_bias_rank_matrix = rank_one_metric(
        numpy.absolute(spatial_max_bias_matrix)
    )
    correlation_rank_matrix = rank_one_metric(-1 * correlation_matrix)
    kge_rank_matrix = rank_one_metric(-1 * kge_matrix)
    reliability_rank_matrix = rank_one_metric(reliability_matrix)

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
    w_sort_indices, l_sort_indices, m_sort_indices = (
        numpy.unravel_index(sort_indices_1d, overall_rank_matrix.shape)
    )

    for k in range(len(w_sort_indices)):
        w = w_sort_indices[k]
        l = l_sort_indices[k]
        m = m_sort_indices[k]

        print((
            'Weight decay = {0:.5f} ... '
            'cycle length = {1:d} ... '
            'min learning rate = {2:.5f}:\n'
            'DWMSE/RMSE/MAE ranks for {3:s} = {4:.1f}, {5:.1f}, {6:.1f} ... '
            'bias and stdev-bias ranks for {3:s} = {7:.1f}, {8:.1f} ... '
            'spatial-min- and spatial-max-bias ranks for {3:s} = {9:.1f}, {10:.1f} ... '
            'correlation/KGE/reliability ranks for {3:s} = {11:.1f}, {12:.1f}, {13:.1f}'
        ).format(
            WEIGHT_DECAY_VALUES_AXIS1[w],
            ANNEALING_CYCLE_LENGTHS_AXIS2[l],
            ANNEALING_MIN_LEARNING_RATES_AXIS3[m],
            target_field_name,
            dwmse_rank_matrix[w, l, m],
            rmse_rank_matrix[w, l, m],
            mae_rank_matrix[w, l, m],
            bias_rank_matrix[w, l, m],
            stdev_bias_rank_matrix[w, l, m],
            spatial_min_bias_rank_matrix[w, l, m],
            spatial_max_bias_rank_matrix[w, l, m],
            correlation_rank_matrix[w, l, m],
            kge_rank_matrix[w, l, m],
            reliability_rank_matrix[w, l, m]
        ))


def _run(experiment_dir_name, output_dir_name):
    """Plots hyperparameter grids for Experiment 21b.

    This is effectively the main method.

    :param experiment_dir_name: See documentation at top of this script.
    :param output_dir_name: Same.
    """

    axis1_length = len(WEIGHT_DECAY_VALUES_AXIS1)
    axis2_length = len(ANNEALING_CYCLE_LENGTHS_AXIS2)
    axis3_length = len(ANNEALING_MIN_LEARNING_RATES_AXIS3)
    num_target_fields = len(TARGET_FIELD_NAMES)

    y_tick_labels = [
        '{0:.5f}'.format(w) for w in WEIGHT_DECAY_VALUES_AXIS1
    ]
    x_tick_labels = ['{0:d}'.format(m) for m in ANNEALING_CYCLE_LENGTHS_AXIS2]

    y_axis_label = 'Weight decay'
    x_axis_label = 'Length of annealing cycle'

    dimensions = (axis1_length, axis2_length, axis3_length, num_target_fields)

    dwmse_matrix = numpy.full(dimensions, numpy.nan)
    rmse_matrix = numpy.full(dimensions, numpy.nan)
    mae_matrix = numpy.full(dimensions, numpy.nan)
    bias_matrix = numpy.full(dimensions, numpy.nan)
    stdev_bias_matrix = numpy.full(dimensions, numpy.nan)
    spatial_min_bias_matrix = numpy.full(dimensions, numpy.nan)
    spatial_max_bias_matrix = numpy.full(dimensions, numpy.nan)
    correlation_matrix = numpy.full(dimensions, numpy.nan)
    kge_matrix = numpy.full(dimensions, numpy.nan)
    reliability_matrix = numpy.full(dimensions, numpy.nan)

    for i in range(axis1_length):
        for j in range(axis2_length):
            for k in range(axis3_length):
                for m in range(num_target_fields):
                    this_eval_file_name = (
                        '{0:s}/weight-decay={1:.5f}_'
                        'annealing={2:03d}epochs-min{3:.5f}/'
                        'validation_full_grid/isotonic_regression/'
                        'ungridded_evaluation_{4:s}.nc'
                    ).format(
                        experiment_dir_name,
                        WEIGHT_DECAY_VALUES_AXIS1[i],
                        ANNEALING_CYCLE_LENGTHS_AXIS2[j],
                        ANNEALING_MIN_LEARNING_RATES_AXIS3[k],
                        TARGET_FIELD_NAMES[m]
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

                    dwmse_matrix[i, j, k, m] = numpy.mean(
                        etx[evaluation.DWMSE_KEY].values[0, :]
                    )
                    rmse_matrix[i, j, k, m] = numpy.sqrt(numpy.mean(
                        etx[evaluation.MSE_KEY].values[0, :]
                    ))
                    mae_matrix[i, j, k, m] = numpy.mean(
                        etx[evaluation.MAE_KEY].values[0, :]
                    )
                    bias_matrix[i, j, k, m] = numpy.mean(
                        etx[evaluation.BIAS_KEY].values[0, :]
                    )
                    stdev_bias_matrix[i, j, k, m] = numpy.mean(
                        etx[evaluation.PREDICTION_STDEV_KEY].values[0, :] -
                        etx[evaluation.TARGET_STDEV_KEY].values[0, :]
                    )
                    spatial_min_bias_matrix[i, j, k, m] = numpy.mean(
                        etx[evaluation.SPATIAL_MIN_BIAS_KEY].values[0, :]
                    )
                    spatial_max_bias_matrix[i, j, k, m] = numpy.mean(
                        etx[evaluation.SPATIAL_MAX_BIAS_KEY].values[0, :]
                    )
                    correlation_matrix[i, j, k, m] = numpy.mean(
                        etx[evaluation.CORRELATION_KEY].values[0, :]
                    )
                    kge_matrix[i, j, k, m] = numpy.mean(
                        etx[evaluation.KGE_KEY].values[0, :]
                    )
                    reliability_matrix[i, j, k, m] = numpy.mean(
                        etx[evaluation.RELIABILITY_KEY].values[0, :]
                    )

    print(SEPARATOR_STRING)

    for m in range(num_target_fields):
        _print_ranking_one_score(
            score_matrix=dwmse_matrix[..., m],
            score_name='DWMSE for {0:s}'.format(TARGET_FIELD_NAMES[m])
        )
        print(SEPARATOR_STRING)

        _print_ranking_one_score(
            score_matrix=rmse_matrix[..., m],
            score_name='RMSE for {0:s}'.format(TARGET_FIELD_NAMES[m])
        )
        print(SEPARATOR_STRING)

        _print_ranking_one_score(
            score_matrix=mae_matrix[..., m],
            score_name='MAE for {0:s}'.format(TARGET_FIELD_NAMES[m])
        )
        print(SEPARATOR_STRING)

        _print_ranking_one_score(
            score_matrix=numpy.absolute(bias_matrix[..., m]),
            score_name='absolute bias for {0:s}'.format(TARGET_FIELD_NAMES[m])
        )
        print(SEPARATOR_STRING)

        _print_ranking_one_score(
            score_matrix=numpy.absolute(stdev_bias_matrix[..., m]),
            score_name='abs stdev bias for {0:s}'.format(TARGET_FIELD_NAMES[m])
        )
        print(SEPARATOR_STRING)

        _print_ranking_one_score(
            score_matrix=numpy.absolute(spatial_min_bias_matrix[..., m]),
            score_name='abs spatial-min bias for {0:s}'.format(TARGET_FIELD_NAMES[m])
        )
        print(SEPARATOR_STRING)

        _print_ranking_one_score(
            score_matrix=numpy.absolute(spatial_max_bias_matrix[..., m]),
            score_name='abs spatial-max bias for {0:s}'.format(TARGET_FIELD_NAMES[m])
        )
        print(SEPARATOR_STRING)

        _print_ranking_one_score(
            score_matrix=-1 * correlation_matrix[..., m],
            score_name='neg correlation for {0:s}'.format(TARGET_FIELD_NAMES[m])
        )
        print(SEPARATOR_STRING)

        _print_ranking_one_score(
            score_matrix=-1 * kge_matrix[..., m],
            score_name='neg KGE for {0:s}'.format(TARGET_FIELD_NAMES[m])
        )
        print(SEPARATOR_STRING)

        _print_ranking_one_score(
            score_matrix=reliability_matrix[..., m],
            score_name='reliability for {0:s}'.format(TARGET_FIELD_NAMES[m])
        )
        print(SEPARATOR_STRING)

        _print_ranking_all_scores(
            dwmse_matrix=dwmse_matrix[..., m],
            rmse_matrix=rmse_matrix[..., m],
            mae_matrix=mae_matrix[..., m],
            bias_matrix=bias_matrix[..., m],
            stdev_bias_matrix=stdev_bias_matrix[..., m],
            spatial_min_bias_matrix=spatial_min_bias_matrix[..., m],
            spatial_max_bias_matrix=spatial_max_bias_matrix[..., m],
            correlation_matrix=correlation_matrix[..., m],
            kge_matrix=kge_matrix[..., m],
            reliability_matrix=reliability_matrix[..., m],
            target_field_name=TARGET_FIELD_NAMES[m]
        )
        print(SEPARATOR_STRING)

    _print_ranking_all_scores(
        dwmse_matrix=dwmse_matrix,
        rmse_matrix=rmse_matrix,
        mae_matrix=mae_matrix,
        bias_matrix=bias_matrix,
        stdev_bias_matrix=stdev_bias_matrix,
        spatial_min_bias_matrix=spatial_min_bias_matrix,
        spatial_max_bias_matrix=spatial_max_bias_matrix,
        correlation_matrix=correlation_matrix,
        kge_matrix=kge_matrix,
        reliability_matrix=reliability_matrix,
        target_field_name='all fields'
    )
    print(SEPARATOR_STRING)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    dimensions = (axis3_length, num_target_fields)
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
        for m in range(num_target_fields):

            # Plot DWMSE vs. hyperparameters.
            figure_object, axes_object = _plot_scores_2d(
                score_matrix=dwmse_matrix[..., k, m],
                min_colour_value=_finite_percentile(dwmse_matrix[..., m], 0),
                max_colour_value=_finite_percentile(dwmse_matrix[..., m], 95),
                x_tick_labels=x_tick_labels,
                y_tick_labels=y_tick_labels
            )

            this_index = numpy.nanargmin(numpy.ravel(dwmse_matrix[..., m]))
            best_indices = numpy.unravel_index(
                this_index, dwmse_matrix[..., m].shape
            )

            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                BEST_MARKER_SIZE_GRID_CELLS / dwmse_matrix.shape[1]
            )

            if best_indices[2] == k:
                axes_object.plot(
                    best_indices[1], best_indices[0],
                    linestyle='None', marker=BEST_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=WHITE_COLOUR,
                    markeredgecolor=WHITE_COLOUR
                )

            if SELECTED_MARKER_INDICES[2] == k:
                axes_object.plot(
                    SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
                    linestyle='None', marker=SELECTED_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=WHITE_COLOUR,
                    markeredgecolor=WHITE_COLOUR
                )

            axes_object.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
            axes_object.set_ylabel(y_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)

            title_string = (
                'Dual-weighted MSE for {0:s} ({1:s})\n'
                'Min learning rate = {2:.5f}'
            ).format(
                TARGET_FIELD_TO_ENGLISH[TARGET_FIELD_NAMES[m]],
                TARGET_FIELD_TO_CUBED_UNITS[TARGET_FIELD_NAMES[m]],
                ANNEALING_MIN_LEARNING_RATES_AXIS3[k]
            )
            axes_object.set_title(title_string)

            dwmse_panel_file_name_matrix[k, m] = (
                '{0:s}/dwmse_min-learning-rate={1:.5f}_{2:s}.jpg'
            ).format(
                output_dir_name,
                ANNEALING_MIN_LEARNING_RATES_AXIS3[k],
                TARGET_FIELD_NAMES[m].replace('_', '-')
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
                score_matrix=rmse_matrix[..., k, m],
                min_colour_value=_finite_percentile(rmse_matrix[..., m], 0),
                max_colour_value=_finite_percentile(rmse_matrix[..., m], 95),
                x_tick_labels=x_tick_labels,
                y_tick_labels=y_tick_labels
            )

            this_index = numpy.nanargmin(numpy.ravel(rmse_matrix[..., m]))
            best_indices = numpy.unravel_index(
                this_index, rmse_matrix[..., m].shape
            )

            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                BEST_MARKER_SIZE_GRID_CELLS / rmse_matrix.shape[1]
            )

            if best_indices[2] == k:
                axes_object.plot(
                    best_indices[1], best_indices[0],
                    linestyle='None', marker=BEST_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=WHITE_COLOUR,
                    markeredgecolor=WHITE_COLOUR
                )

            if SELECTED_MARKER_INDICES[2] == k:
                axes_object.plot(
                    SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
                    linestyle='None', marker=SELECTED_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=WHITE_COLOUR,
                    markeredgecolor=WHITE_COLOUR
                )

            axes_object.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
            axes_object.set_ylabel(y_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)

            title_string = (
                'RMSE for {0:s} ({1:s})\n'
                'Min learning rate = {2:.5f}'
            ).format(
                TARGET_FIELD_TO_ENGLISH[TARGET_FIELD_NAMES[m]],
                TARGET_FIELD_TO_UNITS[TARGET_FIELD_NAMES[m]],
                ANNEALING_MIN_LEARNING_RATES_AXIS3[k]
            )
            axes_object.set_title(title_string)

            rmse_panel_file_name_matrix[k, m] = (
                '{0:s}/rmse_min-learning-rate={1:.5f}_{2:s}.jpg'
            ).format(
                output_dir_name,
                ANNEALING_MIN_LEARNING_RATES_AXIS3[k],
                TARGET_FIELD_NAMES[m].replace('_', '-')
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
                score_matrix=mae_matrix[..., k, m],
                min_colour_value=_finite_percentile(mae_matrix[..., m], 0),
                max_colour_value=_finite_percentile(mae_matrix[..., m], 95),
                x_tick_labels=x_tick_labels,
                y_tick_labels=y_tick_labels
            )

            this_index = numpy.nanargmin(numpy.ravel(mae_matrix[..., m]))
            best_indices = numpy.unravel_index(
                this_index, mae_matrix[..., m].shape
            )

            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                BEST_MARKER_SIZE_GRID_CELLS / mae_matrix.shape[1]
            )

            if best_indices[2] == k:
                axes_object.plot(
                    best_indices[1], best_indices[0],
                    linestyle='None', marker=BEST_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=WHITE_COLOUR,
                    markeredgecolor=WHITE_COLOUR
                )

            if SELECTED_MARKER_INDICES[2] == k:
                axes_object.plot(
                    SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
                    linestyle='None', marker=SELECTED_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=WHITE_COLOUR,
                    markeredgecolor=WHITE_COLOUR
                )

            axes_object.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
            axes_object.set_ylabel(y_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)

            title_string = (
                'MAE for {0:s} ({1:s})\n'
                'Min learning rate = {2:.5f}'
            ).format(
                TARGET_FIELD_TO_ENGLISH[TARGET_FIELD_NAMES[m]],
                TARGET_FIELD_TO_UNITS[TARGET_FIELD_NAMES[m]],
                ANNEALING_MIN_LEARNING_RATES_AXIS3[k]
            )
            axes_object.set_title(title_string)

            mae_panel_file_name_matrix[k, m] = (
                '{0:s}/mae_min-learning-rate={1:.5f}_{2:s}.jpg'
            ).format(
                output_dir_name,
                ANNEALING_MIN_LEARNING_RATES_AXIS3[k],
                TARGET_FIELD_NAMES[m].replace('_', '-')
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
                score_matrix=bias_matrix[..., k, m],
                min_colour_value=None,
                max_colour_value=_finite_percentile(
                    numpy.absolute(bias_matrix), 95
                ),
                x_tick_labels=x_tick_labels,
                y_tick_labels=y_tick_labels
            )

            this_index = numpy.nanargmin(numpy.ravel(
                numpy.absolute(bias_matrix[..., m])
            ))
            best_indices = numpy.unravel_index(
                this_index, bias_matrix[..., m].shape
            )

            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                BEST_MARKER_SIZE_GRID_CELLS / bias_matrix.shape[1]
            )

            if best_indices[2] == k:
                axes_object.plot(
                    best_indices[1], best_indices[0],
                    linestyle='None', marker=BEST_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=BLACK_COLOUR,
                    markeredgecolor=BLACK_COLOUR
                )

            if SELECTED_MARKER_INDICES[2]:
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
                'Bias for {0:s} ({1:s})\n'
                'Min learning rate = {2:.5f}'
            ).format(
                TARGET_FIELD_TO_ENGLISH[TARGET_FIELD_NAMES[m]],
                TARGET_FIELD_TO_UNITS[TARGET_FIELD_NAMES[m]],
                ANNEALING_MIN_LEARNING_RATES_AXIS3[k]
            )
            axes_object.set_title(title_string)

            bias_panel_file_name_matrix[k, m] = (
                '{0:s}/bias_min-learning-rate={1:.5f}_{2:s}.jpg'
            ).format(
                output_dir_name,
                ANNEALING_MIN_LEARNING_RATES_AXIS3[k],
                TARGET_FIELD_NAMES[m].replace('_', '-')
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
                score_matrix=stdev_bias_matrix[..., k, m],
                min_colour_value=None,
                max_colour_value=_finite_percentile(
                    numpy.absolute(stdev_bias_matrix), 95
                ),
                x_tick_labels=x_tick_labels,
                y_tick_labels=y_tick_labels
            )

            this_index = numpy.nanargmin(numpy.ravel(
                numpy.absolute(stdev_bias_matrix[..., m])
            ))
            best_indices = numpy.unravel_index(
                this_index, stdev_bias_matrix[..., m].shape
            )

            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                BEST_MARKER_SIZE_GRID_CELLS / stdev_bias_matrix.shape[1]
            )

            if best_indices[2] == k:
                axes_object.plot(
                    best_indices[1], best_indices[0],
                    linestyle='None', marker=BEST_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=BLACK_COLOUR,
                    markeredgecolor=BLACK_COLOUR
                )

            if SELECTED_MARKER_INDICES[2]:
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
                'Stdev bias for {0:s} ({1:s})\n'
                'Min learning rate = {2:.5f}'
            ).format(
                TARGET_FIELD_TO_ENGLISH[TARGET_FIELD_NAMES[m]],
                TARGET_FIELD_TO_UNITS[TARGET_FIELD_NAMES[m]],
                ANNEALING_MIN_LEARNING_RATES_AXIS3[k]
            )
            axes_object.set_title(title_string)

            stdev_bias_panel_file_name_matrix[k, m] = (
                '{0:s}/stdev_bias_min-learning-rate={1:.5f}_{2:s}.jpg'
            ).format(
                output_dir_name,
                ANNEALING_MIN_LEARNING_RATES_AXIS3[k],
                TARGET_FIELD_NAMES[m].replace('_', '-')
            )

            print('Saving figure to: "{0:s}"...'.format(
                stdev_bias_panel_file_name_matrix[k, m]
            ))
            figure_object.savefig(
                stdev_bias_panel_file_name_matrix[k, m],
                dpi=FIGURE_RESOLUTION_DPI, pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)

            # Plot spatial-min bias vs. hyperparameters.
            figure_object, axes_object = _plot_scores_2d(
                score_matrix=spatial_min_bias_matrix[..., k, m],
                min_colour_value=None,
                max_colour_value=_finite_percentile(
                    numpy.absolute(spatial_min_bias_matrix), 95
                ),
                x_tick_labels=x_tick_labels,
                y_tick_labels=y_tick_labels
            )

            this_index = numpy.nanargmin(numpy.ravel(
                numpy.absolute(spatial_min_bias_matrix[..., m])
            ))
            best_indices = numpy.unravel_index(
                this_index, spatial_min_bias_matrix[..., m].shape
            )

            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                BEST_MARKER_SIZE_GRID_CELLS / spatial_min_bias_matrix.shape[1]
            )

            if best_indices[2] == k:
                axes_object.plot(
                    best_indices[1], best_indices[0],
                    linestyle='None', marker=BEST_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=BLACK_COLOUR,
                    markeredgecolor=BLACK_COLOUR
                )

            if SELECTED_MARKER_INDICES[2]:
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
                'Spatial-min bias for {0:s} ({1:s})\n'
                'Min learning rate = {2:.5f}'
            ).format(
                TARGET_FIELD_TO_ENGLISH[TARGET_FIELD_NAMES[m]],
                TARGET_FIELD_TO_UNITS[TARGET_FIELD_NAMES[m]],
                ANNEALING_MIN_LEARNING_RATES_AXIS3[k]
            )
            axes_object.set_title(title_string)

            spatial_min_bias_panel_file_name_matrix[k, m] = (
                '{0:s}/spatial_min_bias_min-learning-rate={1:.5f}_{2:s}.jpg'
            ).format(
                output_dir_name,
                ANNEALING_MIN_LEARNING_RATES_AXIS3[k],
                TARGET_FIELD_NAMES[m].replace('_', '-')
            )

            print('Saving figure to: "{0:s}"...'.format(
                spatial_min_bias_panel_file_name_matrix[k, m]
            ))
            figure_object.savefig(
                spatial_min_bias_panel_file_name_matrix[k, m],
                dpi=FIGURE_RESOLUTION_DPI, pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)

            # Plot spatial-max bias vs. hyperparameters.
            figure_object, axes_object = _plot_scores_2d(
                score_matrix=spatial_max_bias_matrix[..., k, m],
                min_colour_value=None,
                max_colour_value=_finite_percentile(
                    numpy.absolute(spatial_max_bias_matrix), 95
                ),
                x_tick_labels=x_tick_labels,
                y_tick_labels=y_tick_labels
            )

            this_index = numpy.nanargmin(numpy.ravel(
                numpy.absolute(spatial_max_bias_matrix[..., m])
            ))
            best_indices = numpy.unravel_index(
                this_index, spatial_max_bias_matrix[..., m].shape
            )

            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                BEST_MARKER_SIZE_GRID_CELLS / spatial_max_bias_matrix.shape[1]
            )

            if best_indices[2] == k:
                axes_object.plot(
                    best_indices[1], best_indices[0],
                    linestyle='None', marker=BEST_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=BLACK_COLOUR,
                    markeredgecolor=BLACK_COLOUR
                )

            if SELECTED_MARKER_INDICES[2]:
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
                'Spatial-max bias for {0:s} ({1:s})\n'
                'Min learning rate = {2:.5f}'
            ).format(
                TARGET_FIELD_TO_ENGLISH[TARGET_FIELD_NAMES[m]],
                TARGET_FIELD_TO_UNITS[TARGET_FIELD_NAMES[m]],
                ANNEALING_MIN_LEARNING_RATES_AXIS3[k]
            )
            axes_object.set_title(title_string)

            spatial_max_bias_panel_file_name_matrix[k, m] = (
                '{0:s}/spatial_max_bias_min-learning-rate={1:.5f}_{2:s}.jpg'
            ).format(
                output_dir_name,
                ANNEALING_MIN_LEARNING_RATES_AXIS3[k],
                TARGET_FIELD_NAMES[m].replace('_', '-')
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
                min_colour_value=_finite_percentile(correlation_matrix[..., m], 5),
                max_colour_value=_finite_percentile(correlation_matrix[..., m], 100),
                x_tick_labels=x_tick_labels,
                y_tick_labels=y_tick_labels
            )

            this_index = numpy.nanargmax(numpy.ravel(correlation_matrix[..., m]))
            best_indices = numpy.unravel_index(
                this_index, correlation_matrix[..., m].shape
            )

            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                BEST_MARKER_SIZE_GRID_CELLS / correlation_matrix.shape[1]
            )

            if best_indices[2] == k:
                axes_object.plot(
                    best_indices[1], best_indices[0],
                    linestyle='None', marker=BEST_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=BLACK_COLOUR,
                    markeredgecolor=BLACK_COLOUR
                )

            if SELECTED_MARKER_INDICES[2] == k:
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
                'Correlation for {0:s}\n'
                'Min learning rate = {1:.5f}'
            ).format(
                TARGET_FIELD_TO_ENGLISH[TARGET_FIELD_NAMES[m]],
                ANNEALING_MIN_LEARNING_RATES_AXIS3[k]
            )
            axes_object.set_title(title_string)

            correlation_panel_file_name_matrix[k, m] = (
                '{0:s}/correlation_min-learning-rate={1:.5f}_{2:s}.jpg'
            ).format(
                output_dir_name,
                ANNEALING_MIN_LEARNING_RATES_AXIS3[k],
                TARGET_FIELD_NAMES[m].replace('_', '-')
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
                min_colour_value=_finite_percentile(kge_matrix[..., m], 5),
                max_colour_value=_finite_percentile(kge_matrix[..., m], 100),
                x_tick_labels=x_tick_labels,
                y_tick_labels=y_tick_labels
            )

            this_index = numpy.nanargmax(numpy.ravel(kge_matrix[..., m]))
            best_indices = numpy.unravel_index(
                this_index, kge_matrix[..., m].shape
            )

            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                BEST_MARKER_SIZE_GRID_CELLS / kge_matrix.shape[1]
            )

            if best_indices[2] == k:
                axes_object.plot(
                    best_indices[1], best_indices[0],
                    linestyle='None', marker=BEST_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=BLACK_COLOUR,
                    markeredgecolor=BLACK_COLOUR
                )

            if SELECTED_MARKER_INDICES[2] == k:
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
                'KGE for {0:s}\n'
                'Min learning rate = {1:.5f}'
            ).format(
                TARGET_FIELD_TO_ENGLISH[TARGET_FIELD_NAMES[m]],
                ANNEALING_MIN_LEARNING_RATES_AXIS3[k]
            )
            axes_object.set_title(title_string)

            kge_panel_file_name_matrix[k, m] = (
                '{0:s}/kge_min-learning-rate={1:.5f}_{2:s}.jpg'
            ).format(
                output_dir_name,
                ANNEALING_MIN_LEARNING_RATES_AXIS3[k],
                TARGET_FIELD_NAMES[m].replace('_', '-')
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
                score_matrix=reliability_matrix[..., k, m],
                min_colour_value=_finite_percentile(reliability_matrix[..., m], 0),
                max_colour_value=_finite_percentile(reliability_matrix[..., m], 95),
                x_tick_labels=x_tick_labels,
                y_tick_labels=y_tick_labels
            )

            this_index = numpy.nanargmin(numpy.ravel(reliability_matrix[..., m]))
            best_indices = numpy.unravel_index(
                this_index, reliability_matrix[..., m].shape
            )

            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                BEST_MARKER_SIZE_GRID_CELLS / reliability_matrix.shape[1]
            )

            if best_indices[2] == k:
                axes_object.plot(
                    best_indices[1], best_indices[0],
                    linestyle='None', marker=BEST_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=WHITE_COLOUR,
                    markeredgecolor=WHITE_COLOUR
                )

            if SELECTED_MARKER_INDICES[2] == k:
                axes_object.plot(
                    SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
                    linestyle='None', marker=SELECTED_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=WHITE_COLOUR,
                    markeredgecolor=WHITE_COLOUR
                )

            axes_object.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
            axes_object.set_ylabel(y_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)

            title_string = (
                'Reliability for {0:s} ({1:s})\n'
                'Min learning rate = {2:.5f}'
            ).format(
                TARGET_FIELD_TO_ENGLISH[TARGET_FIELD_NAMES[m]],
                TARGET_FIELD_TO_SQUARED_UNITS[TARGET_FIELD_NAMES[m]],
                ANNEALING_MIN_LEARNING_RATES_AXIS3[k]
            )
            axes_object.set_title(title_string)

            reliability_panel_file_name_matrix[k, m] = (
                '{0:s}/reliability_min-learning-rate={1:.5f}_{2:s}.jpg'
            ).format(
                output_dir_name,
                ANNEALING_MIN_LEARNING_RATES_AXIS3[k],
                TARGET_FIELD_NAMES[m].replace('_', '-')
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

    for m in range(num_target_fields):
        concat_figure_file_name = '{0:s}/dwmse_{1:s}.jpg'.format(
            output_dir_name, TARGET_FIELD_NAMES[m].replace('_', '-')
        )
        print('Concatenating panels to: "{0:s}"...'.format(
            concat_figure_file_name
        ))
        imagemagick_utils.concatenate_images(
            input_file_names=dwmse_panel_file_name_matrix[:, m].tolist(),
            output_file_name=concat_figure_file_name,
            num_panel_rows=3,
            num_panel_columns=2
        )
        imagemagick_utils.resize_image(
            input_file_name=concat_figure_file_name,
            output_file_name=concat_figure_file_name,
            output_size_pixels=int(1e7)
        )

        concat_figure_file_name = '{0:s}/rmse_{1:s}.jpg'.format(
            output_dir_name, TARGET_FIELD_NAMES[m].replace('_', '-')
        )
        print('Concatenating panels to: "{0:s}"...'.format(
            concat_figure_file_name
        ))
        imagemagick_utils.concatenate_images(
            input_file_names=rmse_panel_file_name_matrix[:, m].tolist(),
            output_file_name=concat_figure_file_name,
            num_panel_rows=3,
            num_panel_columns=2
        )
        imagemagick_utils.resize_image(
            input_file_name=concat_figure_file_name,
            output_file_name=concat_figure_file_name,
            output_size_pixels=int(1e7)
        )

        concat_figure_file_name = '{0:s}/mae_{1:s}.jpg'.format(
            output_dir_name, TARGET_FIELD_NAMES[m].replace('_', '-')
        )
        print('Concatenating panels to: "{0:s}"...'.format(
            concat_figure_file_name
        ))
        imagemagick_utils.concatenate_images(
            input_file_names=mae_panel_file_name_matrix[:, m].tolist(),
            output_file_name=concat_figure_file_name,
            num_panel_rows=3,
            num_panel_columns=2
        )
        imagemagick_utils.resize_image(
            input_file_name=concat_figure_file_name,
            output_file_name=concat_figure_file_name,
            output_size_pixels=int(1e7)
        )

        concat_figure_file_name = '{0:s}/bias_{1:s}.jpg'.format(
            output_dir_name, TARGET_FIELD_NAMES[m].replace('_', '-')
        )
        print('Concatenating panels to: "{0:s}"...'.format(
            concat_figure_file_name
        ))
        imagemagick_utils.concatenate_images(
            input_file_names=bias_panel_file_name_matrix[:, m].tolist(),
            output_file_name=concat_figure_file_name,
            num_panel_rows=3,
            num_panel_columns=2
        )
        imagemagick_utils.resize_image(
            input_file_name=concat_figure_file_name,
            output_file_name=concat_figure_file_name,
            output_size_pixels=int(1e7)
        )

        concat_figure_file_name = '{0:s}/stdev_bias_{1:s}.jpg'.format(
            output_dir_name, TARGET_FIELD_NAMES[m].replace('_', '-')
        )
        print('Concatenating panels to: "{0:s}"...'.format(
            concat_figure_file_name
        ))
        imagemagick_utils.concatenate_images(
            input_file_names=stdev_bias_panel_file_name_matrix[:, m].tolist(),
            output_file_name=concat_figure_file_name,
            num_panel_rows=3,
            num_panel_columns=2
        )
        imagemagick_utils.resize_image(
            input_file_name=concat_figure_file_name,
            output_file_name=concat_figure_file_name,
            output_size_pixels=int(1e7)
        )

        concat_figure_file_name = '{0:s}/spatial_min_bias_{1:s}.jpg'.format(
            output_dir_name, TARGET_FIELD_NAMES[m].replace('_', '-')
        )
        print('Concatenating panels to: "{0:s}"...'.format(
            concat_figure_file_name
        ))
        imagemagick_utils.concatenate_images(
            input_file_names=spatial_min_bias_panel_file_name_matrix[:, m].tolist(),
            output_file_name=concat_figure_file_name,
            num_panel_rows=3,
            num_panel_columns=2
        )
        imagemagick_utils.resize_image(
            input_file_name=concat_figure_file_name,
            output_file_name=concat_figure_file_name,
            output_size_pixels=int(1e7)
        )

        concat_figure_file_name = '{0:s}/spatial_max_bias_{1:s}.jpg'.format(
            output_dir_name, TARGET_FIELD_NAMES[m].replace('_', '-')
        )
        print('Concatenating panels to: "{0:s}"...'.format(
            concat_figure_file_name
        ))
        imagemagick_utils.concatenate_images(
            input_file_names=spatial_max_bias_panel_file_name_matrix[:, m].tolist(),
            output_file_name=concat_figure_file_name,
            num_panel_rows=3,
            num_panel_columns=2
        )
        imagemagick_utils.resize_image(
            input_file_name=concat_figure_file_name,
            output_file_name=concat_figure_file_name,
            output_size_pixels=int(1e7)
        )

        concat_figure_file_name = '{0:s}/correlation_{1:s}.jpg'.format(
            output_dir_name, TARGET_FIELD_NAMES[m].replace('_', '-')
        )
        print('Concatenating panels to: "{0:s}"...'.format(
            concat_figure_file_name
        ))
        imagemagick_utils.concatenate_images(
            input_file_names=correlation_panel_file_name_matrix[:, m].tolist(),
            output_file_name=concat_figure_file_name,
            num_panel_rows=3,
            num_panel_columns=2
        )
        imagemagick_utils.resize_image(
            input_file_name=concat_figure_file_name,
            output_file_name=concat_figure_file_name,
            output_size_pixels=int(1e7)
        )

        concat_figure_file_name = '{0:s}/kge_{1:s}.jpg'.format(
            output_dir_name, TARGET_FIELD_NAMES[m].replace('_', '-')
        )
        print('Concatenating panels to: "{0:s}"...'.format(
            concat_figure_file_name
        ))
        imagemagick_utils.concatenate_images(
            input_file_names=kge_panel_file_name_matrix[:, m].tolist(),
            output_file_name=concat_figure_file_name,
            num_panel_rows=3,
            num_panel_columns=2
        )
        imagemagick_utils.resize_image(
            input_file_name=concat_figure_file_name,
            output_file_name=concat_figure_file_name,
            output_size_pixels=int(1e7)
        )

        concat_figure_file_name = '{0:s}/reliability_{1:s}.jpg'.format(
            output_dir_name, TARGET_FIELD_NAMES[m].replace('_', '-')
        )
        print('Concatenating panels to: "{0:s}"...'.format(
            concat_figure_file_name
        ))
        imagemagick_utils.concatenate_images(
            input_file_names=reliability_panel_file_name_matrix[:, m].tolist(),
            output_file_name=concat_figure_file_name,
            num_panel_rows=3,
            num_panel_columns=2
        )
        imagemagick_utils.resize_image(
            input_file_name=concat_figure_file_name,
            output_file_name=concat_figure_file_name,
            output_size_pixels=int(1e7)
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        experiment_dir_name=getattr(INPUT_ARG_OBJECT, EXPERIMENT_DIR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
