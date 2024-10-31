"""Plots gridded model evaluation."""

import os
import sys
import argparse
import numpy
import xarray
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import matplotlib.colors

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import grids
import time_conversion
import longitude_conversion as lng_conversion
import file_system_utils
import error_checking
import border_io
import urma_utils
import evaluation
import neural_net
import plotting_utils
import target_plotting

TOLERANCE = 1e-6

TARGET_FIELD_NAME_TO_VERBOSE_UNITLESS = {
    urma_utils.TEMPERATURE_2METRE_NAME: '2-m temperature',
    urma_utils.DEWPOINT_2METRE_NAME: '2-m dewpoint',
    urma_utils.U_WIND_10METRE_NAME: '10-m zonal wind',
    urma_utils.V_WIND_10METRE_NAME: '10-m meridional wind',
    urma_utils.WIND_GUST_10METRE_NAME: '10-m wind gust'
}

TARGET_FIELD_NAME_TO_VERBOSE = {
    urma_utils.TEMPERATURE_2METRE_NAME: r'2-m temperature ($^{\circ}$C)',
    urma_utils.DEWPOINT_2METRE_NAME: r'2-m dewpoint ($^{\circ}$C)',
    urma_utils.U_WIND_10METRE_NAME: r'10-m zonal wind (m s$^{-1}$)',
    urma_utils.V_WIND_10METRE_NAME: r'10-m meridional wind (m s$^{-1}$)',
    urma_utils.WIND_GUST_10METRE_NAME: r'10-m wind gust (m s$^{-1}$)'
}

TARGET_FIELD_NAME_TO_VERBOSE_SQUARED = {
    urma_utils.TEMPERATURE_2METRE_NAME: r'2-m temperature ($^{\circ}$C$^2$)',
    urma_utils.DEWPOINT_2METRE_NAME: r'2-m dewpoint ($^{\circ}$C$^2$)',
    urma_utils.U_WIND_10METRE_NAME: r'10-m zonal wind (m$^2$ s$^{-2}$)',
    urma_utils.V_WIND_10METRE_NAME: r'10-m meridional wind (m$^2$ s$^{-2}$)',
    urma_utils.WIND_GUST_10METRE_NAME: r'10-m wind gust (m$^2$ s$^{-2}$)'
}

TARGET_FIELD_NAME_TO_VERBOSE_CUBED = {
    urma_utils.TEMPERATURE_2METRE_NAME: r'2-m temperature ($^{\circ}$C$^3$)',
    urma_utils.DEWPOINT_2METRE_NAME: r'2-m dewpoint ($^{\circ}$C$^3$)',
    urma_utils.U_WIND_10METRE_NAME: r'10-m zonal wind (m$^3$ s$^{-3}$)',
    urma_utils.V_WIND_10METRE_NAME: r'10-m meridional wind (m$^3$ s$^{-3}$)',
    urma_utils.WIND_GUST_10METRE_NAME: r'10-m wind gust (m$^3$ s$^{-3}$)'
}

RMSE_KEY = 'root_mean_squared_error'

METRIC_NAME_TO_VERBOSE = {
    evaluation.TARGET_STDEV_KEY: 'stdev of actual value',
    evaluation.PREDICTION_STDEV_KEY: 'stdev of prediction',
    evaluation.TARGET_MEAN_KEY: 'mean actual value',
    evaluation.PREDICTION_MEAN_KEY: 'mean prediction',
    RMSE_KEY: 'RMSE',
    evaluation.MSE_BIAS_KEY: 'bias part of MSE',
    evaluation.MSE_VARIANCE_KEY: 'variance part of MSE',
    evaluation.MSE_SKILL_SCORE_KEY: 'MSE skill score',
    evaluation.DWMSE_KEY: 'DWMSE',
    evaluation.DWMSE_SKILL_SCORE_KEY: 'DWMSE skill score',
    evaluation.KS_STATISTIC_KEY: 'K-S statistic',
    evaluation.KS_P_VALUE_KEY: 'K-S p-value',
    evaluation.MAE_KEY: 'MAE',
    evaluation.MAE_SKILL_SCORE_KEY: 'MAE skill score',
    evaluation.BIAS_KEY: 'bias',
    evaluation.CORRELATION_KEY: 'correlation',
    evaluation.KGE_KEY: 'Kling-Gupta efficiency',
    evaluation.RELIABILITY_KEY: 'reliability',
    evaluation.SSRAT_KEY: 'spread-skill ratio',
    evaluation.SSDIFF_KEY: 'spread-skill difference',
    evaluation.SSREL_KEY: 'spread-skill reliability',
    evaluation.SPATIAL_MIN_BIAS_KEY: 'spatial-min bias',
    evaluation.SPATIAL_MAX_BIAS_KEY: 'spatial-max bias'
}

UNITLESS_METRIC_NAMES = [
    evaluation.MSE_SKILL_SCORE_KEY, evaluation.DWMSE_SKILL_SCORE_KEY,
    evaluation.MAE_SKILL_SCORE_KEY, evaluation.CORRELATION_KEY,
    evaluation.KGE_KEY, evaluation.SSRAT_KEY
]
SQUARED_METRIC_NAMES = [
    evaluation.MSE_BIAS_KEY, evaluation.MSE_VARIANCE_KEY,
    evaluation.RELIABILITY_KEY
]
CUBED_METRIC_NAMES = [evaluation.DWMSE_KEY]

METRIC_NAME_TO_COLOUR_MAP_OBJECT = {
    evaluation.TARGET_STDEV_KEY: pyplot.get_cmap('cividis'),
    evaluation.PREDICTION_STDEV_KEY: pyplot.get_cmap('cividis'),
    evaluation.TARGET_MEAN_KEY: pyplot.get_cmap('cividis'),
    evaluation.PREDICTION_MEAN_KEY: pyplot.get_cmap('cividis'),
    RMSE_KEY: pyplot.get_cmap('viridis'),
    evaluation.MSE_BIAS_KEY: pyplot.get_cmap('viridis'),
    evaluation.MSE_VARIANCE_KEY: pyplot.get_cmap('viridis'),
    evaluation.MSE_SKILL_SCORE_KEY: pyplot.get_cmap('viridis'),
    evaluation.DWMSE_KEY: pyplot.get_cmap('viridis'),
    evaluation.DWMSE_SKILL_SCORE_KEY: pyplot.get_cmap('viridis'),
    evaluation.KS_STATISTIC_KEY: pyplot.get_cmap('viridis'),
    evaluation.KS_P_VALUE_KEY: pyplot.get_cmap('viridis'),
    evaluation.MAE_KEY: pyplot.get_cmap('viridis'),
    evaluation.MAE_SKILL_SCORE_KEY: pyplot.get_cmap('viridis'),
    evaluation.BIAS_KEY: pyplot.get_cmap('seismic'),
    evaluation.CORRELATION_KEY: pyplot.get_cmap('viridis'),
    evaluation.KGE_KEY: pyplot.get_cmap('seismic'),
    evaluation.RELIABILITY_KEY: pyplot.get_cmap('viridis'),
    evaluation.SSRAT_KEY: pyplot.get_cmap('seismic'),
    evaluation.SSDIFF_KEY: pyplot.get_cmap('seismic'),
    evaluation.SSREL_KEY: pyplot.get_cmap('viridis'),
    evaluation.SPATIAL_MIN_BIAS_KEY: pyplot.get_cmap('seismic'),
    evaluation.SPATIAL_MAX_BIAS_KEY: pyplot.get_cmap('seismic')
}

METRIC_NAME_TO_COLOUR_NORM_TYPE_STRING = {
    evaluation.TARGET_STDEV_KEY: 'sequential',
    evaluation.PREDICTION_STDEV_KEY: 'sequential',
    evaluation.TARGET_MEAN_KEY: 'sequential',
    evaluation.PREDICTION_MEAN_KEY: 'sequential',
    RMSE_KEY: 'sequential',
    evaluation.MSE_BIAS_KEY: 'sequential',
    evaluation.MSE_VARIANCE_KEY: 'sequential',
    evaluation.MSE_SKILL_SCORE_KEY: 'sequential',
    evaluation.DWMSE_KEY: 'sequential',
    evaluation.DWMSE_SKILL_SCORE_KEY: 'sequential',
    evaluation.KS_STATISTIC_KEY: 'sequential',
    evaluation.KS_P_VALUE_KEY: 'sequential',
    evaluation.MAE_KEY: 'sequential',
    evaluation.MAE_SKILL_SCORE_KEY: 'sequential',
    evaluation.BIAS_KEY: 'diverging',
    evaluation.CORRELATION_KEY: 'sequential',
    evaluation.KGE_KEY: 'diverging_weird',
    evaluation.RELIABILITY_KEY: 'sequential',
    evaluation.SSRAT_KEY: 'ssrat',
    evaluation.SSDIFF_KEY: 'diverging',
    evaluation.SSREL_KEY: 'sequential',
    evaluation.SPATIAL_MIN_BIAS_KEY: 'diverging',
    evaluation.SPATIAL_MAX_BIAS_KEY: 'diverging'
}

NAN_COLOUR = numpy.full(3, 152. / 255)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_eval_file_name_or_pattern'
BY_MONTH_ARG_NAME = 'by_month'
BY_HOUR_ARG_NAME = 'by_hour'
TARGET_FIELD_ARG_NAME = 'target_field_name'
METRICS_ARG_NAME = 'metric_names'
MIN_VALUES_ARG_NAME = 'min_colour_values'
MAX_VALUES_ARG_NAME = 'max_colour_values'
MIN_PERCENTILES_ARG_NAME = 'min_colour_percentiles'
MAX_PERCENTILES_ARG_NAME = 'max_colour_percentiles'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Evaluation scores will be read from here by '
    '`evaluation.read_file`.  If `{0:s} == 1`, there will be 12 input files, '
    'each produced by replacing the ".nc" at the end of `{1:s}` with '
    '"_month01.nc" or "_month02.nc" or... "_month12.nc".  If `{2:s} == 1`, '
    'there will be 24 input files, each produced by replacing the ".nc" at the '
    'end of `{1:s}` with "_hour00.nc" or "_hour01.nc" or... "_hour23.nc".'
).format(
    BY_MONTH_ARG_NAME,
    INPUT_FILE_ARG_NAME,
    BY_HOUR_ARG_NAME
)
BY_MONTH_HELP_STRING = (
    'Boolean flag.  If 1, will produce a set of plots for forecasts valid in '
    'every month.  If both this argument and `{0:s}` are 0, will produce a '
    'single set of plots for all forecasts.'
).format(
    BY_HOUR_ARG_NAME
)
BY_HOUR_HELP_STRING = (
    'Boolean flag.  If 1, will produce a set of plots for forecasts valid at '
    'every UTC hour.  If both this argument and `{0:s}` are 0, will produce a '
    'single set of plots for all forecasts.'
).format(
    BY_MONTH_ARG_NAME
)
TARGET_FIELD_HELP_STRING = (
    'Name of target field for which to plot metrics.  This field name must be '
    'accepted by `urma_utils.check_field_name`.'
)
METRICS_HELP_STRING = (
    'List of metrics to plot.  Each metric must be in the following list:'
    '\n{0:s}'
).format(
    str(list(METRIC_NAME_TO_VERBOSE.keys()))
)
MIN_VALUES_HELP_STRING = (
    'List of minimum values for each colour bar (one per metric in the list '
    '`{0:s}`).  If you would rather specify min/max values by percentile, '
    'leave this argument alone; use `{1:s}` and `{2:s}`, instead.'
).format(
    METRICS_ARG_NAME,
    MIN_PERCENTILES_ARG_NAME,
    MAX_PERCENTILES_ARG_NAME
)
MAX_VALUES_HELP_STRING = 'See documentation for `{0:s}`.'.format(
    MIN_VALUES_ARG_NAME
)
MIN_PERCENTILES_HELP_STRING = (
    'List of minimum percentiles for each colour bar (one per metric in the '
    'list `{0:s}`).  For example, suppose that the second value in `{0:s}` is '
    '"mean_squared_error" and the second value in this list is 0.5.  Then the '
    'minimum value in the colour bar for MSE will be the 0.5th-percentile MSE '
    'over all grid points.  If you would rather specify min/max values '
    'directly, leave this argument alone; use `{1:s}` and `{2:s}`, instead.'
).format(
    METRICS_ARG_NAME,
    MIN_PERCENTILES_ARG_NAME,
    MAX_PERCENTILES_ARG_NAME
)
MAX_PERCENTILES_HELP_STRING = 'See documentation for `{0:s}`.'.format(
    MIN_PERCENTILES_ARG_NAME
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BY_HOUR_ARG_NAME, type=int, required=False, default=0,
    help=BY_HOUR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BY_MONTH_ARG_NAME, type=int, required=False, default=0,
    help=BY_MONTH_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_FIELD_ARG_NAME, type=str, required=True,
    help=TARGET_FIELD_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + METRICS_ARG_NAME, type=str, nargs='+', required=True,
    help=METRICS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_VALUES_ARG_NAME, type=float, nargs='+', required=False,
    default=[101.], help=MIN_VALUES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_VALUES_ARG_NAME, type=float, nargs='+', required=False,
    default=[-1.], help=MAX_VALUES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_PERCENTILES_ARG_NAME, type=float, nargs='+', required=False,
    default=[101.], help=MIN_PERCENTILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILES_ARG_NAME, type=float, nargs='+', required=False,
    default=[-1.], help=MAX_PERCENTILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _get_ssrat_colour_scheme(max_colour_value):
    """Returns colour scheme for spread-skill ratio (SSRAT).

    :param max_colour_value: Max value in colour scheme.
    :return: colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
    :return: colour_norm_object: Colour-normalizer (maps from data space to
        colour-bar space, which goes from 0...1).  This is an instance of
        `matplotlib.colors.Normalize`.
    """

    orig_colour_map_object = pyplot.get_cmap('seismic')

    negative_values = numpy.linspace(0, 1, num=1001, dtype=float)
    positive_values = numpy.linspace(1, max_colour_value, num=1001, dtype=float)
    bias_values = numpy.concatenate((negative_values, positive_values))

    normalized_values = numpy.linspace(0, 1, num=len(bias_values), dtype=float)
    rgb_matrix = orig_colour_map_object(normalized_values)[:, :-1]

    colour_map_object = matplotlib.colors.ListedColormap(rgb_matrix)
    colour_map_object.set_bad(NAN_COLOUR)
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        bias_values, colour_map_object.N
    )

    return colour_map_object, colour_norm_object


def _plot_one_score(
        score_matrix, grid_latitude_matrix_deg_n, grid_longitude_matrix_deg_e,
        border_latitudes_deg_n, border_longitudes_deg_e,
        colour_map_object, colour_norm_object, title_string, output_file_name):
    """Plots one score.

    M = number of rows in grid
    N = number of columns in grid
    P = number of points in border file

    :param score_matrix: M-by-N numpy array of data values.
    :param grid_latitude_matrix_deg_n: M-by-N numpy array of grid-point
        latitudes (deg north).
    :param grid_longitude_matrix_deg_e: M-by-N numpy array of grid-point
        longitudes (deg east).
    :param border_latitudes_deg_n: length-P numpy array of latitudes (deg
        north).
    :param border_longitudes_deg_e: length-P numpy array of longitudes (deg
        east).
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.colors.ListedColormap` or similar).
    :param colour_norm_object: Colour-normalizer, used to map from physical
        values to colours (instance of `matplotlib.colors.BoundaryNorm` or
        similar).
    :param title_string: Title.
    :param output_file_name: Path to output file.
    """

    plotting_with_borders = not (
        numpy.all(numpy.isnan(grid_latitude_matrix_deg_n)) and
        numpy.all(numpy.isnan(grid_longitude_matrix_deg_e))
    )

    if not plotting_with_borders:
        num_rows = score_matrix.shape[0]
        grid_latitudes_deg_n = numpy.linspace(-1, 1, num=num_rows, dtype=float)

        num_columns = score_matrix.shape[1]
        grid_longitudes_deg_e = numpy.linspace(
            0, 1, num=num_columns, dtype=float
        )

        grid_latitude_matrix_deg_n, grid_longitude_matrix_deg_e = (
            grids.latlng_vectors_to_matrices(
                unique_latitudes_deg=grid_latitudes_deg_n,
                unique_longitudes_deg=grid_longitudes_deg_e
            )
        )

    border_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        border_longitudes_deg_e
    )
    grid_longitude_matrix_deg_e = lng_conversion.convert_lng_positive_in_west(
        grid_longitude_matrix_deg_e
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    colour_map_object.set_bad(NAN_COLOUR)

    target_plotting.plot_field(
        data_matrix=score_matrix,
        latitude_matrix_deg_n=grid_latitude_matrix_deg_n,
        longitude_matrix_deg_e=grid_longitude_matrix_deg_e,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        axes_object=axes_object,
        plot_colour_bar=True
    )

    if plotting_with_borders:
        plotting_utils.plot_borders(
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            axes_object=axes_object,
            line_colour=numpy.full(3, 0.)
        )
        plotting_utils.plot_grid_lines(
            plot_latitudes_deg_n=numpy.ravel(grid_latitude_matrix_deg_n),
            plot_longitudes_deg_e=numpy.ravel(grid_longitude_matrix_deg_e),
            axes_object=axes_object,
            meridian_spacing_deg=20.,
            parallel_spacing_deg=10.
        )

    axes_object.set_xlim(
        numpy.min(grid_longitude_matrix_deg_e),
        numpy.max(grid_longitude_matrix_deg_e)
    )
    axes_object.set_ylim(
        numpy.min(grid_latitude_matrix_deg_n),
        numpy.max(grid_latitude_matrix_deg_n)
    )
    axes_object.set_title(title_string)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(eval_file_name_or_pattern, by_month, by_hour,
         target_field_name, metric_names, min_colour_values, max_colour_values,
         min_colour_percentiles, max_colour_percentiles, output_dir_name):
    """Plots gridded model evaluation.

    This is effectively the main method.

    :param eval_file_name_or_pattern: See documentation at top of file.
    :param by_month: Same.
    :param by_hour: Same.
    :param target_field_name: Same.
    :param metric_names: Same.
    :param min_colour_values: Same.
    :param max_colour_values: Same.
    :param min_colour_percentiles: Same.
    :param max_colour_percentiles: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if the input file contains ungridded, rather than
        gridded, evaluation.
    """

    # Check input args.
    assert not (by_month and by_hour)

    valid_metric_names = list(METRIC_NAME_TO_VERBOSE.keys())
    assert all([m in valid_metric_names for m in metric_names])
    num_metrics = len(metric_names)

    if (
            len(min_colour_values) == 1 and
            len(max_colour_values) == 1 and
            max_colour_values[0] < min_colour_values[0]
    ):
        min_colour_values = None
        max_colour_values = None

    if (
            len(min_colour_percentiles) == 1 and
            len(max_colour_percentiles) == 1 and
            max_colour_percentiles[0] < min_colour_percentiles[0]
    ):
        min_colour_percentiles = None
        max_colour_percentiles = None

    if min_colour_values is not None:
        error_checking.assert_is_numpy_array(
            min_colour_values,
            exact_dimensions=numpy.array([num_metrics], dtype=int)
        )
        error_checking.assert_is_greater_numpy_array(
            max_colour_values - min_colour_values, 0.
        )

    if min_colour_percentiles is not None:
        error_checking.assert_is_numpy_array(
            max_colour_percentiles,
            exact_dimensions=numpy.array([num_metrics], dtype=int)
        )

        error_checking.assert_is_leq_numpy_array(max_colour_percentiles, 100.)
        error_checking.assert_is_geq_numpy_array(min_colour_percentiles, 0.)
        error_checking.assert_is_greater_numpy_array(
            max_colour_percentiles - min_colour_percentiles, 0.
        )

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Do actual stuff.
    if by_month:
        eval_file_pattern = eval_file_name_or_pattern
        assert eval_file_pattern.endswith('.nc')

        months = numpy.linspace(1, 12, num=12, dtype=int)
        evaluation_file_names = [
            '{0:s}_month{1:02d}.nc'.format(eval_file_pattern, m)
            for m in months
        ]
    elif by_hour:
        eval_file_pattern = eval_file_name_or_pattern
        assert eval_file_pattern.endswith('.nc')

        hours = numpy.linspace(0, 23, num=24, dtype=int)
        evaluation_file_names = [
            '{0:s}_hour{1:02d}.nc'.format(eval_file_pattern, h)
            for h in hours
        ]
    else:
        evaluation_file_names = [eval_file_name_or_pattern]
        assert os.path.isfile(evaluation_file_names[0])

    num_files = len(evaluation_file_names)
    evaluation_tables_xarray = [xarray.Dataset()] * num_files
    target_field_names = None

    for i in range(num_files):
        if not os.path.isfile(evaluation_file_names[i]):
            continue

        print('Reading data from: "{0:s}"...'.format(evaluation_file_names[i]))
        evaluation_tables_xarray[i] = evaluation.read_file(
            evaluation_file_names[i]
        )

        if evaluation.ROW_DIM not in evaluation_tables_xarray[i].coords:
            error_string = (
                'File "{0:s}" contains ungridded evaluation.  This script '
                'handles only gridded evaluation.'
            ).format(
                evaluation_file_names[i]
            )

            raise ValueError(error_string)

        model_file_name = (
            evaluation_tables_xarray[i].attrs[evaluation.MODEL_FILE_KEY]
        )

        model_metafile_name = neural_net.find_metafile(
            model_file_name=model_file_name, raise_error_if_missing=True
        )

        print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
        model_metadata_dict = neural_net.read_metafile(model_metafile_name)
        generator_option_dict = model_metadata_dict[
            neural_net.TRAINING_OPTIONS_KEY
        ]
        goptd = generator_option_dict

        if target_field_names is None:
            target_field_names = goptd[neural_net.TARGET_FIELDS_KEY]

        assert target_field_names == goptd[neural_net.TARGET_FIELDS_KEY]

    num_target_fields = len(target_field_names)
    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    for i in range(num_files):
        if not evaluation_tables_xarray[i].data_vars:
            continue

        etx = evaluation_tables_xarray[i]

        if by_hour:
            file_name_suffix = '_hour{0:02d}'.format(i)
        elif by_month:
            file_name_suffix = '_month{0:02d}'.format(i + 1)
        else:
            file_name_suffix = ''

        if by_month:
            title_suffix = ' in {0:s}'.format(
                time_conversion.string_to_unix_sec(
                    '2000-{0:02d}-01'.format(i + 1), '%b'
                )
            )
        elif by_hour:
            title_suffix = ' at {0:02d}Z'.format(i)
        else:
            title_suffix = ''

        for k in range(num_target_fields):
            if target_field_names[k] != target_field_name:
                continue

            for m in range(len(metric_names)):
                if metric_names[m] in [
                        evaluation.SPATIAL_MIN_BIAS_KEY,
                        evaluation.SPATIAL_MAX_BIAS_KEY
                ]:
                    continue

                if metric_names[m] == RMSE_KEY:
                    this_score_matrix = numpy.sqrt(numpy.nanmean(
                        etx[evaluation.MSE_KEY].values[:, :, k, ...], axis=-1
                    ))
                else:
                    this_score_matrix = (
                        etx[metric_names[m]].values[:, :, k, ...] + 0.
                    )
                    if len(this_score_matrix.shape) > 2:
                        this_score_matrix = numpy.nanmean(
                            this_score_matrix, axis=-1
                        )

                if metric_names[m] in [
                        evaluation.TARGET_MEAN_KEY,
                        evaluation.PREDICTION_MEAN_KEY
                ]:
                    first_matrix = numpy.nanmean(
                        etx[evaluation.TARGET_MEAN_KEY].values[:, :, k, ...],
                        axis=-1
                    )
                    second_matrix = numpy.nanmean(
                        etx[evaluation.PREDICTION_MEAN_KEY].values[
                            :, :, k, ...
                        ],
                        axis=-1
                    )
                    score_matrix_for_cnorm = numpy.stack(
                        [first_matrix, second_matrix], axis=-1
                    )
                elif metric_names[m] in [
                        evaluation.TARGET_STDEV_KEY,
                        evaluation.PREDICTION_STDEV_KEY
                ]:
                    first_matrix = numpy.nanmean(
                        etx[evaluation.TARGET_STDEV_KEY].values[:, :, k, ...],
                        axis=-1
                    )
                    second_matrix = numpy.nanmean(
                        etx[evaluation.PREDICTION_STDEV_KEY].values[
                            :, :, k, ...
                        ],
                        axis=-1
                    )
                    score_matrix_for_cnorm = numpy.stack(
                        [first_matrix, second_matrix], axis=-1
                    )
                else:
                    score_matrix_for_cnorm = this_score_matrix

                colour_norm_type_string = (
                    METRIC_NAME_TO_COLOUR_NORM_TYPE_STRING[
                        metric_names[m]
                    ]
                )
                colour_map_object = METRIC_NAME_TO_COLOUR_MAP_OBJECT[
                    metric_names[m]
                ]

                if min_colour_values is None:
                    if colour_norm_type_string == 'sequential':
                        this_min_colour_value = numpy.nanpercentile(
                            score_matrix_for_cnorm, min_colour_percentiles[m]
                        )
                        this_max_colour_value = numpy.nanpercentile(
                            score_matrix_for_cnorm, max_colour_percentiles[m]
                        )
                    elif colour_norm_type_string == 'diverging_weird':
                        this_max_colour_value = numpy.nanpercentile(
                            score_matrix_for_cnorm, max_colour_percentiles[m]
                        )
                        this_min_colour_value = -1 * this_max_colour_value
                    elif colour_norm_type_string == 'diverging':
                        this_max_colour_value = numpy.nanpercentile(
                            numpy.absolute(score_matrix_for_cnorm),
                            max_colour_percentiles[m]
                        )
                        this_min_colour_value = -1 * this_max_colour_value
                    elif colour_norm_type_string == 'ssrat':
                        this_offset = numpy.nanpercentile(
                            numpy.absolute(score_matrix_for_cnorm - 1.),
                            max_colour_percentiles[m]
                        )
                        this_max_colour_value = 1. + this_offset
                        this_min_colour_value = 0.
                    else:
                        this_max_colour_value = numpy.nanpercentile(
                            score_matrix_for_cnorm, max_colour_percentiles[m]
                        )
                        this_min_colour_value = -1 * this_max_colour_value

                    if numpy.isnan(this_max_colour_value):
                        this_min_colour_value = 0.
                        this_max_colour_value = 1.

                    this_max_colour_value = max([
                        this_max_colour_value,
                        this_min_colour_value + TOLERANCE
                    ])
                else:
                    if colour_norm_type_string in [
                            'diverging', 'diverging_weird'
                    ]:
                        this_max_colour_value = max([
                            numpy.absolute(min_colour_values[m]),
                            numpy.absolute(max_colour_values[m])
                        ])
                        this_min_colour_value = -1 * this_max_colour_value
                    else:
                        this_min_colour_value = min_colour_values[m]
                        this_max_colour_value = max_colour_values[m]

                if colour_norm_type_string == 'ssrat':
                    colour_map_object, colour_norm_object = (
                        _get_ssrat_colour_scheme(this_max_colour_value)
                    )
                else:
                    colour_norm_object = pyplot.Normalize(
                        vmin=this_min_colour_value, vmax=this_max_colour_value
                    )

                if metric_names[m] in UNITLESS_METRIC_NAMES:
                    this_fancy_field_name = (
                        TARGET_FIELD_NAME_TO_VERBOSE_UNITLESS[
                            target_field_names[k]
                        ]
                    )
                elif metric_names[m] in SQUARED_METRIC_NAMES:
                    this_fancy_field_name = (
                        TARGET_FIELD_NAME_TO_VERBOSE_SQUARED[
                            target_field_names[k]
                        ]
                    )
                elif metric_names[m] in CUBED_METRIC_NAMES:
                    this_fancy_field_name = TARGET_FIELD_NAME_TO_VERBOSE_CUBED[
                        target_field_names[k]
                    ]
                else:
                    this_fancy_field_name = TARGET_FIELD_NAME_TO_VERBOSE[
                        target_field_names[k]
                    ]

                title_string = (
                    '{0:s}{1:s} for {2:s}{3:s}\n'
                    'Min/avg/max = {4:.2g}, {5:.2g}, {6:.2g}'
                ).format(
                    METRIC_NAME_TO_VERBOSE[metric_names[m]][0].upper(),
                    METRIC_NAME_TO_VERBOSE[metric_names[m]][1:],
                    this_fancy_field_name,
                    title_suffix,
                    numpy.nanmin(this_score_matrix),
                    numpy.nanmean(this_score_matrix),
                    numpy.nanmax(this_score_matrix),
                )

                output_file_name = '{0:s}/{1:s}_{2:s}{3:s}.jpg'.format(
                    output_dir_name,
                    target_field_names[k].replace('_', '-'),
                    metric_names[m],
                    file_name_suffix
                )

                _plot_one_score(
                    score_matrix=this_score_matrix,
                    grid_latitude_matrix_deg_n=
                    etx[evaluation.LATITUDE_KEY].values + 0.,
                    grid_longitude_matrix_deg_e=
                    etx[evaluation.LONGITUDE_KEY].values + 0.,
                    border_latitudes_deg_n=border_latitudes_deg_n,
                    border_longitudes_deg_e=border_longitudes_deg_e,
                    colour_map_object=colour_map_object,
                    colour_norm_object=colour_norm_object,
                    title_string=title_string,
                    output_file_name=output_file_name
                )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        eval_file_name_or_pattern=getattr(
            INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME
        ),
        by_month=bool(getattr(INPUT_ARG_OBJECT, BY_MONTH_ARG_NAME)),
        by_hour=bool(getattr(INPUT_ARG_OBJECT, BY_HOUR_ARG_NAME)),
        target_field_name=getattr(INPUT_ARG_OBJECT, TARGET_FIELD_ARG_NAME),
        metric_names=getattr(INPUT_ARG_OBJECT, METRICS_ARG_NAME),
        min_colour_values=numpy.array(
            getattr(INPUT_ARG_OBJECT, MIN_VALUES_ARG_NAME),
            dtype=float
        ),
        max_colour_values=numpy.array(
            getattr(INPUT_ARG_OBJECT, MAX_VALUES_ARG_NAME),
            dtype=float
        ),
        min_colour_percentiles=numpy.array(
            getattr(INPUT_ARG_OBJECT, MIN_PERCENTILES_ARG_NAME),
            dtype=float
        ),
        max_colour_percentiles=numpy.array(
            getattr(INPUT_ARG_OBJECT, MAX_PERCENTILES_ARG_NAME),
            dtype=float
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
