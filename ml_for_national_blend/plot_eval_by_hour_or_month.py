"""Plots evaluation metrics for one model by hour or by month.

Keep in mind that this script works only with ungridded evaluation files as
input.
"""

import os
import sys
import re
import copy
import argparse
import numpy
import xarray
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import evaluation
import plot_gridded_evaluation as plot_gridded_eval

DEFAULT_FONT_SIZE = 30
pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

MARKER_TYPE = 'o'
MARKER_SIZE = 16
LINE_WIDTH = 5

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

LINE_COLOURS = [
    numpy.array([27, 158, 119], dtype=float) / 255,
    numpy.array([217, 95, 2], dtype=float) / 255,
    numpy.array([117, 112, 179], dtype=float) / 255
]

TARGET_FIELD_NAME_TO_VERBOSE_UNITLESS = (
    plot_gridded_eval.TARGET_FIELD_NAME_TO_VERBOSE_UNITLESS
)
TARGET_FIELD_NAME_TO_VERBOSE = (
    plot_gridded_eval.TARGET_FIELD_NAME_TO_VERBOSE
)
TARGET_FIELD_NAME_TO_VERBOSE_SQUARED = (
    plot_gridded_eval.TARGET_FIELD_NAME_TO_VERBOSE_SQUARED
)
TARGET_FIELD_NAME_TO_VERBOSE_CUBED = (
    plot_gridded_eval.TARGET_FIELD_NAME_TO_VERBOSE_CUBED
)

RMSE_KEY = plot_gridded_eval.RMSE_KEY

METRIC_NAME_TO_VERBOSE = plot_gridded_eval.METRIC_NAME_TO_VERBOSE
UNITLESS_METRIC_NAMES = plot_gridded_eval.UNITLESS_METRIC_NAMES
SQUARED_METRIC_NAMES = plot_gridded_eval.SQUARED_METRIC_NAMES
CUBED_METRIC_NAMES = plot_gridded_eval.CUBED_METRIC_NAMES

METRIC_NAME_TO_COLOUR_MAP_OBJECT = (
    plot_gridded_eval.METRIC_NAME_TO_COLOUR_MAP_OBJECT
)
METRIC_NAME_TO_COLOUR_NORM_TYPE_STRING = (
    plot_gridded_eval.METRIC_NAME_TO_COLOUR_NORM_TYPE_STRING
)

METRIC_NAMES_BY_GROUP = [
    [evaluation.TARGET_STDEV_KEY, evaluation.PREDICTION_STDEV_KEY],
    [evaluation.TARGET_MEAN_KEY, evaluation.PREDICTION_MEAN_KEY],
    [evaluation.MSE_BIAS_KEY, evaluation.MSE_VARIANCE_KEY, evaluation.RELIABILITY_KEY],
    [RMSE_KEY, evaluation.MAE_KEY, evaluation.BIAS_KEY],
    [evaluation.SPATIAL_MIN_BIAS_KEY, evaluation.SPATIAL_MAX_BIAS_KEY],
    [evaluation.DWMSE_KEY],
    [evaluation.MAE_SKILL_SCORE_KEY, evaluation.MSE_SKILL_SCORE_KEY, evaluation.DWMSE_SKILL_SCORE_KEY],
    [evaluation.KS_P_VALUE_KEY, evaluation.KS_STATISTIC_KEY],
    [evaluation.CORRELATION_KEY, evaluation.KGE_KEY],
    [evaluation.SSRAT_KEY],
    [evaluation.SSDIFF_KEY, evaluation.SSREL_KEY]
]

INPUT_FILE_PATTERN_ARG_NAME = 'input_eval_file_pattern'
BY_HOUR_ARG_NAME = 'by_hour'
BY_MONTH_ARG_NAME = 'by_month'
TARGET_FIELDS_ARG_NAME = 'target_field_names'
METRICS_ARG_NAME = 'metric_names'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_PATTERN_HELP_STRING = (
    'Path to input file.  Evaluation scores will be read from here by '
    '`evaluation.read_file`.  If `{0:s} == 1`, there will be 12 input files, '
    'each produced by replacing the ".nc" at the end of `{1:s}` with '
    '"_month01.nc" or "_month02.nc" or... "_month12.nc".  If `{2:s} == 1`, '
    'there will be 24 input files, each produced by replacing the ".nc" at the '
    'end of `{1:s}` with "_hour00.nc" or "_hour01.nc" or... "_hour23.nc".'
).format(
    BY_MONTH_ARG_NAME,
    INPUT_FILE_PATTERN_ARG_NAME,
    BY_HOUR_ARG_NAME
)
BY_MONTH_HELP_STRING = (
    'Boolean flag.  If 1, will plot metrics by month instead of hour.'
)
BY_HOUR_HELP_STRING = (
    'Boolean flag.  If 1, will plot metrics by hour instead of month.'
)
TARGET_FIELDS_HELP_STRING = (
    'List of target fields.  For each pair of target field / metric, this '
    'script will produce one plot.  Each target field must be accepted by '
    '`urma_utils.check_field_name`.'
)
METRICS_HELP_STRING = (
    'List of metrics.  For each pair of target field / metric, this script '
    'will produce one plot.  Each metric must be in the following list:\n{0:s}'
).format(
    str(list(METRIC_NAME_TO_VERBOSE.keys()))
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory (figures will be saved here).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_PATTERN_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_PATTERN_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BY_HOUR_ARG_NAME, type=int, required=True,
    help=BY_HOUR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BY_MONTH_ARG_NAME, type=int, required=True,
    help=BY_MONTH_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_FIELDS_ARG_NAME, type=str, nargs='+', required=True,
    help=TARGET_FIELDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + METRICS_ARG_NAME, type=str, nargs='+', required=True,
    help=METRICS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_one_metric_group(metric_matrix, target_field_name, metric_names,
                           title_string, output_file_name):
    """Plots one group of metrics, for one target field, vs. month or hour.

    M = number of metrics
    D = number of time divisions (hours or months)

    :param metric_matrix: M-by-D numpy array of metric values.
    :param target_field_name: Name of target field.
    :param metric_names: length-M list of metric names.
    :param title_string: Title.
    :param output_file_name: Path to output file.  Figure will be saved here.
    """

    num_time_divisions = metric_matrix.shape[1]
    if num_time_divisions == 24:
        x_tick_values = numpy.linspace(0, 23, num=24, dtype=float)
        x_tick_labels = ['{0:d}'.format(x) for x in x_tick_values]
    else:
        x_tick_values = numpy.linspace(1, 12, num=12, dtype=float)
        x_tick_labels = [
            'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
            'Oct', 'Nov', 'Dec'
        ]

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    num_metrics = len(metric_names)
    legend_handles = [None] * num_metrics
    legend_strings = [''] * num_metrics

    for m in range(num_metrics):
        this_handle = axes_object.plot(
            x_tick_values,
            metric_matrix[m, :],
            color=LINE_COLOURS[m],
            linestyle='solid',
            linewidth=LINE_WIDTH,
            marker=MARKER_TYPE,
            markersize=MARKER_SIZE,
            markerfacecolor=LINE_COLOURS[m],
            markeredgecolor=LINE_COLOURS[m],
            markeredgewidth=0
        )[0]

        this_label = '{0:s}{1:s}'.format(
            metric_names[m][0].upper(),
            metric_names[m][1:]
        )

        if metric_names[m] in SQUARED_METRIC_NAMES:
            unit_string = re.search(
                r"\(.*\)",
                TARGET_FIELD_NAME_TO_VERBOSE_SQUARED[target_field_name]
            ).group()
        elif metric_names[m] in CUBED_METRIC_NAMES:
            unit_string = re.search(
                r"\(.*\)", TARGET_FIELD_NAME_TO_VERBOSE_CUBED[target_field_name]
            ).group()
        elif metric_names[m] in UNITLESS_METRIC_NAMES:
            unit_string = ''
        else:
            unit_string = re.search(
                r"\(.*\)", TARGET_FIELD_NAME_TO_VERBOSE[target_field_name]
            ).group()

        if unit_string != '':
            unit_string = ' ' + unit_string

        this_label = '{0:s}{1:s}'.format(this_label, unit_string)
        legend_handles.append(this_handle)
        legend_strings.append(this_label)

    axes_object.set_xlim([
        numpy.min(x_tick_values) - 0.5,
        numpy.max(x_tick_values) + 0.5
    ])

    axes_object.legend(
        legend_handles, legend_strings,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.1),
        fancybox=True,
        shadow=True,
        ncol=1
    )

    axes_object.set_xticks(x_tick_values)
    axes_object.set_xticklabels(x_tick_labels)
    axes_object.set_title(title_string)

    if num_time_divisions == 24:
        axes_object.set_xlabel('UTC hour')
    else:
        axes_object.set_xlabel('Month')

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name,
        dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(evaluation_file_pattern, by_month, by_hour,
         target_field_names, metric_names, output_dir_name):
    """Plots evaluation metrics for one model by hour or by month.

    This is effectively the main method.

    :param evaluation_file_pattern: See documentation at top of this script.
    :param by_month: Same.
    :param by_hour: Same.
    :param target_field_names: Same.
    :param metric_names: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    assert not (by_month and by_hour)

    if by_month:
        assert evaluation_file_pattern.endswith('.nc')

        months = numpy.linspace(1, 12, num=12, dtype=int)
        evaluation_file_names = [
            '{0:s}_month{1:02d}.nc'.format(evaluation_file_pattern, m)
            for m in months
        ]
    else:
        assert evaluation_file_pattern.endswith('.nc')

        hours = numpy.linspace(0, 23, num=24, dtype=int)
        evaluation_file_names = [
            '{0:s}_hour{1:02d}.nc'.format(evaluation_file_pattern, h)
            for h in hours
        ]

    num_files = len(evaluation_file_names)
    evaluation_tables_xarray = [xarray.Dataset()] * num_files
    model_file_name = None

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(evaluation_file_names[i]))
        evaluation_tables_xarray[i] = evaluation.read_file(
            evaluation_file_names[i]
        )
        etx_i = evaluation_tables_xarray[i]

        if evaluation.ROW_DIM in etx_i.coords:
            raise ValueError(
                'This script handles only ungridded (full-domain) evaluation, '
                'not per-grid-point evaluation.'
            )

        this_model_file_name = etx_i.attrs[evaluation.MODEL_FILE_KEY]
        if i == 0:
            model_file_name = copy.deepcopy(this_model_file_name)

        assert model_file_name == this_model_file_name

    num_target_fields = len(target_field_names)
    num_metrics = len(metric_names)

    for f in range(num_target_fields):
        field_index_by_file = numpy.array([
            numpy.where(
                etx.coords[evaluation.FIELD_DIM].values ==
                target_field_names[f]
            )[0][0]
            for etx in evaluation_tables_xarray
        ], dtype=int)

        metric_matrix = numpy.full((num_metrics, num_files), numpy.nan)

        for m in range(num_metrics):
            if metric_names[m] == RMSE_KEY:
                these_values = numpy.array([
                    numpy.sqrt(numpy.mean(
                        etx[evaluation.MSE_KEY].values[f_new, :], axis=1
                    ))
                    for etx, f_new in zip(
                        evaluation_tables_xarray, field_index_by_file
                    )
                ])
            elif metric_names[m] in [
                    evaluation.KS_P_VALUE_KEY, evaluation.KS_STATISTIC_KEY
            ]:
                these_values = numpy.array([
                    etx[metric_names[m]].values[f_new]
                    for etx, f_new in zip(
                        evaluation_tables_xarray, field_index_by_file
                    )
                ])
            else:
                these_values = numpy.array([
                    numpy.mean(etx[metric_names[m]].values[f_new, :], axis=1)
                    for etx, f_new in zip(
                        evaluation_tables_xarray, field_index_by_file
                    )
                ])

            metric_matrix[m, :] = these_values

        have_plotted_metric = numpy.full(num_metrics, False, dtype=bool)

        for m in range(num_metrics):
            if have_plotted_metric[m]:
                continue

            group_flags = numpy.array(
                [metric_names[m] in group for group in METRIC_NAMES_BY_GROUP],
                dtype=bool
            )
            group_index = numpy.where(group_flags)[0][0]

            metric_indices_to_plot = []
            for this_metric_name in METRIC_NAMES_BY_GROUP[group_index]:
                if this_metric_name not in metric_names:
                    continue

                metric_indices_to_plot.append(
                    metric_names.index(this_metric_name)
                )

            metric_indices_to_plot = numpy.array(
                metric_indices_to_plot, dtype=int
            )
            have_plotted_metric[metric_indices_to_plot] = True

            output_file_name = '{0:s}/{1:s}_{2:s}_{3:s}.jpg'.format(
                output_dir_name,
                target_field_names[f].replace('_', '-'),
                metric_names[m].replace('_', '-'),
                'hourly' if by_hour else 'monthly'
            )

            _plot_one_metric_group(
                metric_matrix[metric_indices_to_plot, :],
                target_field_name=target_field_names[f],
                metric_names=[metric_names[k] for k in metric_indices_to_plot],
                title_string='Metrics for {0:s}'.format(
                    TARGET_FIELD_NAME_TO_VERBOSE[target_field_names[f]]
                ),
                output_file_name=output_file_name
            )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        evaluation_file_pattern=getattr(
            INPUT_ARG_OBJECT, INPUT_FILE_PATTERN_ARG_NAME
        ),
        by_month=bool(getattr(INPUT_ARG_OBJECT, BY_MONTH_ARG_NAME)),
        by_hour=bool(getattr(INPUT_ARG_OBJECT, BY_HOUR_ARG_NAME)),
        target_field_names=getattr(INPUT_ARG_OBJECT, TARGET_FIELDS_ARG_NAME),
        metric_names=getattr(INPUT_ARG_OBJECT, METRICS_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
