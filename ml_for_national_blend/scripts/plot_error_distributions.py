
"""Plots error distributions.

Specifically, this script creates 4 plots for every target variable:

#1 Error distribution as a function of target value (boxplot or violin plot for
   each bin of target values)
#2 Error distribution as a function of predicted value (boxplot or violin plot
   for each bin of predicted values)
#3 Comparison of right tails (PDF plot for right tail of both distributions
   [target and predicted] on the same axes)
#4 Comparison of left tails
#5 Comparison of full distributions (PDF plot of both target and predicted
   distributions, not restricted to just the tails)
"""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from scipy.stats import gaussian_kde
from ml_for_national_blend.io import prediction_io
from ml_for_national_blend.utils import urma_utils
from ml_for_national_blend.utils import evaluation
from ml_for_national_blend.machine_learning import neural_net
from ml_for_national_blend.outside_code import time_conversion
from ml_for_national_blend.outside_code import file_system_utils
from ml_for_national_blend.outside_code import error_checking

HOURS_TO_SECONDS = 3600
TIME_FORMAT = '%Y-%m-%d-%H'

TARGET_FIELD_NAME_TO_VERBOSE = {
    urma_utils.TEMPERATURE_2METRE_NAME: r'2-m temperature ($^{\circ}$C)',
    urma_utils.DEWPOINT_2METRE_NAME: r'2-m dewpoint ($^{\circ}$C)',
    urma_utils.U_WIND_10METRE_NAME: r'10-m zonal wind (m s$^{-1}$)',
    urma_utils.V_WIND_10METRE_NAME: r'10-m meridional wind (m s$^{-1}$)',
    urma_utils.WIND_GUST_10METRE_NAME: r'10-m wind gust (m s$^{-1}$)'
}

VIOLIN_LINE_WIDTH = 1.5
VIOLIN_LINE_COLOUR = numpy.full(3, 0.)

VIOLIN_FACE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
PREDICTION_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
TARGET_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

DEFAULT_FONT_SIZE = 30
pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

INPUT_DIR_ARG_NAME = 'input_prediction_dir_name'
INIT_TIME_LIMITS_ARG_NAME = 'init_time_limit_strings'
EVALUATE_MONTH_ARG_NAME = 'evaluate_month'
EVALUATE_HOUR_ARG_NAME = 'evaluate_hour'
TARGET_FIELDS_ARG_NAME = 'target_field_names'
NUM_BINS_ARG_NAME = 'num_bins_by_target'
MIN_BIN_EDGES_ARG_NAME = 'min_bin_edge_by_target'
MAX_BIN_EDGES_ARG_NAME = 'max_bin_edge_by_target'
VIOLIN_OR_BOX_ARG_NAME = 'violin_or_box_plots'
LEFT_TAIL_PERCENTILE_ARG_NAME = 'left_tail_percentile'
RIGHT_TAIL_PERCENTILE_ARG_NAME = 'right_tail_percentile'
MAX_NUM_PDF_VALUES_ARG_NAME = 'max_num_pdf_values'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory, containing one prediction file per init time.  '
    'Files therein will be found by `prediction_io.find_file` and read by '
    '`prediction_io.read_file`.'
)
INIT_TIME_LIMITS_HELP_STRING = (
    'List of two initialization times, specifying the beginning and end of the '
    'evaluation period.  Time format is "yyyy-mm-dd-HH".'
)
EVALUATE_MONTH_HELP_STRING = (
    'Will evaluate only forecasts valid in this month (ranging from '
    '1...12).  If you want to evaluate forecasts regardless of month, leave '
    'this argument alone.'
)
EVALUATE_HOUR_HELP_STRING = (
    'Will evaluate only forecasts valid at this UTC hour (ranging from '
    '0...23).  If you want to evaluate forecasts regardless of hour, leave '
    'this argument alone.'
)
TARGET_FIELDS_HELP_STRING = (
    'List of target fields to be evaluated.  Each one must be accepted by '
    '`urma_utils.check_field_name`.'
)
NUM_BINS_HELP_STRING = (
    'length-T list with number of bins in violin/box plot for each target '
    'variable, where T = length of {0:s}.'
).format(
    TARGET_FIELDS_ARG_NAME
)
MIN_BIN_EDGES_HELP_STRING = (
    'length-T list with lower edge of lowest bin in violin/box plot for each '
    'target variable, where T = length of {0:s}.'
).format(
    TARGET_FIELDS_ARG_NAME
)
MAX_BIN_EDGES_HELP_STRING = (
    'length-T list with upper edge of highest bin in violin/box plot for each '
    'target variable, where T = length of {0:s}.'
).format(
    TARGET_FIELDS_ARG_NAME
)
VIOLIN_OR_BOX_HELP_STRING = (
    'Boolean flag.  If 1, the first two plots (error distribution as a '
    'function of target value, error dist AAFO predicted value) will be violin '
    'plots.  If 0, they will be boxplots.'
)
LEFT_TAIL_PERCENTILE_HELP_STRING = (
    'Percentile (ranging from 0...100) used to define left tail of '
    'distribution.  For example, if {0:s} = 0.5, then the left tail of the '
    'distribution for target variable y will be defined as lowest 0.5% of '
    'y-values.  The percentile will be computed over all values of y, both '
    'targets and predictions.'
).format(
    LEFT_TAIL_PERCENTILE_ARG_NAME
)
RIGHT_TAIL_PERCENTILE_HELP_STRING = 'Same as {0:s} but for right tail.'.format(
    LEFT_TAIL_PERCENTILE_ARG_NAME
)
MAX_NUM_PDF_VALUES_HELP_STRING = (
    'Maximum number of values to include in one PDF.  This is the number of '
    'atomic examples (one atomic example = one grid point at one time step).  '
    'If you want to use all values, make this argument -1.  But be warned that '
    'kernel-density estimation (KDE) might take a *really* long time if you '
    'have many values.  For example, 100 time steps with the NBM grid leads to '
    '~400M atomic examples (because the NBM grid has ~4M grid points).  In '
    'this case, for producing plot #5 (full distributions and not just the '
    'tails), KDE will take over an hour for each variable.'
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INIT_TIME_LIMITS_ARG_NAME, type=str, nargs=2, required=True,
    help=INIT_TIME_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EVALUATE_MONTH_ARG_NAME, type=int, required=False, default=-1,
    help=EVALUATE_MONTH_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EVALUATE_HOUR_ARG_NAME, type=int, required=False, default=-1,
    help=EVALUATE_HOUR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_FIELDS_ARG_NAME, type=str, nargs='+', required=True,
    help=TARGET_FIELDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_BINS_ARG_NAME, type=int, nargs='+', required=True,
    help=NUM_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_BIN_EDGES_ARG_NAME, type=float, nargs='+', required=True,
    help=MIN_BIN_EDGES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_BIN_EDGES_ARG_NAME, type=float, nargs='+', required=True,
    help=MAX_BIN_EDGES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + VIOLIN_OR_BOX_ARG_NAME, type=int, required=True,
    help=VIOLIN_OR_BOX_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LEFT_TAIL_PERCENTILE_ARG_NAME, type=float, required=False,
    default=0.5, help=LEFT_TAIL_PERCENTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + RIGHT_TAIL_PERCENTILE_ARG_NAME, type=float, required=False,
    default=99.5, help=RIGHT_TAIL_PERCENTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_NUM_PDF_VALUES_ARG_NAME, type=int, required=False, default=-1,
    help=MAX_NUM_PDF_VALUES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_distributions(
        target_values, predicted_values, tail_percentile, target_field_name):
    """Plots distributions (target and predicted) for one variable.

    Specifically, this method creates plot #3 or #4 or #5 (see definitions at
    top of this script).

    E = number of atomic examples, where one "atomic example" is one valid time
        at one grid point.

    :param target_values: length-E numpy array of actual values.
    :param predicted_values: length-E numpy array of predicted values.
    :param tail_percentile: Percentile (ranging from 0...100) used to define
        "tail" of distribution.  If you are plotting the full distributions
        (not just the left or right tail), make this None.
    :param target_field_name: Name of target field.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    if tail_percentile is None:
        tail_percentile = numpy.nan

    plotting_left_tail = tail_percentile < 50.
    plotting_right_tail = tail_percentile > 50.

    if plotting_left_tail or plotting_right_tail:
        tail_threshold = numpy.nanpercentile(
            numpy.concatenate([target_values, predicted_values]),
            tail_percentile
        )
    else:
        tail_threshold = None

    if plotting_left_tail:
        relevant_target_values = target_values[target_values <= tail_threshold]
        relevant_predicted_values = predicted_values[
            predicted_values <= tail_threshold
        ]
    elif plotting_right_tail:
        relevant_target_values = target_values[target_values >= tail_threshold]
        relevant_predicted_values = predicted_values[
            predicted_values >= tail_threshold
        ]
    else:
        relevant_target_values = target_values
        relevant_predicted_values = predicted_values

    target_kde_object = gaussian_kde(relevant_target_values, bw_method='scott')
    prediction_kde_object = gaussian_kde(
        relevant_predicted_values, bw_method='scott'
    )

    x_min = min([
        numpy.min(relevant_target_values),
        numpy.min(relevant_predicted_values)
    ])
    x_max = max([
        numpy.max(relevant_target_values),
        numpy.max(relevant_predicted_values)
    ])

    if plotting_left_tail:
        x_values = numpy.linspace(x_min, tail_threshold, num=1001)
    elif plotting_right_tail:
        x_values = numpy.linspace(tail_threshold, x_max, num=1001)
    else:
        x_values = numpy.linspace(x_min, x_max, num=1001)

    target_y_values = target_kde_object(x_values)
    prediction_y_values = prediction_kde_object(x_values)

    target_y_values = numpy.maximum(
        target_y_values,
        numpy.finfo(target_y_values.dtype).eps
    )
    prediction_y_values = numpy.maximum(
        prediction_y_values,
        numpy.finfo(prediction_y_values.dtype).eps
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    if plotting_left_tail or plotting_right_tail:
        axes_object.set_yscale('log')

    target_handle = axes_object.plot(
        x_values, target_y_values,
        linestyle='solid', linewidth=3, color=TARGET_COLOUR
    )[0]
    prediction_handle = axes_object.plot(
        x_values, prediction_y_values,
        linestyle='solid', linewidth=3, color=PREDICTION_COLOUR
    )[0]

    axes_object.fill_between(
        x_values, target_y_values, color=TARGET_COLOUR, alpha=0.2
    )
    axes_object.fill_between(
        x_values, prediction_y_values, color=PREDICTION_COLOUR, alpha=0.2
    )

    axes_object.set_xlabel(TARGET_FIELD_NAME_TO_VERBOSE[target_field_name])
    axes_object.set_ylabel('Probability density')

    legend_handles = [target_handle, prediction_handle]
    legend_strings = ['Actual', 'Predicted']

    if plotting_left_tail:
        axes_object.legend(
            legend_handles, legend_strings, loc='lower right',
            bbox_to_anchor=(0.9, 0.1), fancybox=True, shadow=False,
            facecolor='white', edgecolor='k', framealpha=1., ncol=1
        )
    elif plotting_right_tail:
        axes_object.legend(
            legend_handles, legend_strings, loc='lower left',
            bbox_to_anchor=(0.1, 0.1), fancybox=True, shadow=False,
            facecolor='white', edgecolor='k', framealpha=1., ncol=1
        )
    else:
        axes_object.legend(
            legend_handles, legend_strings, loc='lower center',
            bbox_to_anchor=(0.5, 0.1), fancybox=True, shadow=False,
            facecolor='white', edgecolor='k', framealpha=1., ncol=1
        )

    axes_object.grid(True, which='both', linestyle='--', linewidth=0.5)

    title_string = 'Comparison of {0:s} for {1:s}'.format(
        'left tails' if plotting_left_tail
        else 'right tails' if plotting_right_tail
        else 'distributions',
        TARGET_FIELD_NAME_TO_VERBOSE[target_field_name]
    )
    axes_object.set_title(title_string)

    return figure_object, axes_object


def _plot_error_distribution(
        target_values, predicted_values, min_bin_edge, max_bin_edge, num_bins,
        bin_by_target_or_predicted_values, violin_or_box_plots,
        target_field_name):
    """Plots error distribution for one target variable.

    Specifically, this method creates either plot #1 or plot #2 (see definitions
    at top of this script).

    E = number of atomic examples, where one "atomic example" is one valid time
        at one grid point.

    :param target_values: length-E numpy array of actual values.
    :param predicted_values: length-E numpy array of predicted values.
    :param min_bin_edge: Lower edge of lowest bin.
    :param max_bin_edge: Upper edge of highest bin.
    :param num_bins: Number of bins.
    :param bin_by_target_or_predicted_values: Boolean flag.  If True (False),
        the distribution plots (violins or boxes) will be stratified by target
        value.  If False, they will be stratified by predicted value.
    :param violin_or_box_plots: Boolean flag.  See documentation at top of this
        script for more explanation.
    :param target_field_name: Name of target field.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    bin_edges = numpy.linspace(
        min_bin_edge, max_bin_edge, num=num_bins + 1, dtype=float
    )
    bin_edges_for_test = bin_edges + 0.
    bin_edges_for_test[0] = -numpy.inf
    bin_edges_for_test[-1] = numpy.inf

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    boxplot_style_dict = {
        'color': 'k',
        'linewidth': 2
    }
    x_values = numpy.linspace(0, num_bins - 1, num=num_bins, dtype=float)
    good_bin_indices = []

    for j in range(num_bins):
        if bin_by_target_or_predicted_values:
            good_flags = numpy.logical_and(
                target_values >= bin_edges_for_test[j],
                target_values < bin_edges_for_test[j + 1]
            )
            good_flags = numpy.logical_and(
                good_flags,
                numpy.invert(numpy.isnan(predicted_values))
            )
        else:
            good_flags = numpy.logical_and(
                predicted_values >= bin_edges_for_test[j],
                predicted_values < bin_edges_for_test[j + 1]
            )
            good_flags = numpy.logical_and(
                good_flags,
                numpy.invert(numpy.isnan(target_values))
            )

        good_indices = numpy.where(good_flags)[0]
        if len(good_indices) == 0:
            continue

        good_bin_indices.append(j)

        if violin_or_box_plots:
            violin_handles = axes_object.violinplot(
                predicted_values[good_indices] - target_values[good_indices],
                widths=1., vert=True, positions=x_values[[j]],
                showmeans=True, showmedians=False, showextrema=True
            )

            for part_name in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
                try:
                    this_handle = violin_handles[part_name]
                except:
                    continue

                this_handle.set_edgecolor(VIOLIN_LINE_COLOUR)

                if part_name == 'cbars':
                    this_handle.set_linewidth(VIOLIN_LINE_WIDTH / 2)
                else:
                    this_handle.set_linewidth(VIOLIN_LINE_WIDTH)

            for this_handle in violin_handles['bodies']:
                this_handle.set_facecolor(VIOLIN_FACE_COLOUR)
                this_handle.set_edgecolor(VIOLIN_FACE_COLOUR)
                this_handle.set_linewidth(0)
                this_handle.set_alpha(1.)
        else:
            axes_object.boxplot(
                predicted_values[good_indices] - target_values[good_indices],
                widths=0.8, vert=True, notch=False, sym='o', whis=(0.5, 99.5),
                medianprops=boxplot_style_dict, boxprops=boxplot_style_dict,
                whiskerprops=boxplot_style_dict, capprops=boxplot_style_dict,
                positions=x_values[[j]]
            )

    good_bin_indices = numpy.array(good_bin_indices, dtype=int)
    good_bin_indices = numpy.linspace(
        numpy.min(good_bin_indices), numpy.max(good_bin_indices),
        num=numpy.max(good_bin_indices) - numpy.min(good_bin_indices) + 1,
        dtype=int
    )

    x_tick_strings = [
        '[{0:.1f}, {1:.1f})'.format(a, b) for a, b in
        zip(bin_edges[:-1], bin_edges[1:])
    ]
    x_tick_strings[0] = '< {0:.1f}'.format(bin_edges[1])
    x_tick_strings[-1] = '>= {0:.1f}'.format(bin_edges[-2])
    x_tick_strings = [x_tick_strings[j] for j in good_bin_indices]

    for j in range(len(x_tick_strings)):
        if j == 0 or j == len(x_tick_strings) - 1:
            continue
        if numpy.mod(j, 3) == 0:
            continue

        x_tick_strings[j] = ' '

    x_values = x_values[good_bin_indices]

    axes_object.set_xticks(x_values)
    axes_object.set_xticklabels(x_tick_strings, rotation=90, fontsize=15)
    axes_object.set_xlim(good_bin_indices[0] - 0.5, good_bin_indices[-1] + 0.5)

    axes_object.set_xlabel('{0:s} value'.format(
        'Actual' if bin_by_target_or_predicted_values
        else 'Predicted'
    ))

    title_string = 'Error distribution for {0:s}'.format(
        TARGET_FIELD_NAME_TO_VERBOSE[target_field_name]
    )
    axes_object.set_title(title_string)

    return figure_object, axes_object


def _run(prediction_dir_name, init_time_limit_strings, evaluate_month,
         evaluate_hour, target_field_names, num_bins_by_target,
         min_bin_edge_by_target, max_bin_edge_by_target,
         violin_or_box_plots, left_tail_percentile, right_tail_percentile,
         max_num_pdf_values, output_dir_name):
    """Plots error distributions.

    This is effectively the main method.

    :param prediction_dir_name: See documentation at top of this script.
    :param init_time_limit_strings: Same.
    :param evaluate_month: Same.
    :param evaluate_hour: Same.
    :param target_field_names: Same.
    :param num_bins_by_target: Same.
    :param min_bin_edge_by_target: Same.
    :param max_bin_edge_by_target: Same.
    :param violin_or_box_plots: Same.
    :param left_tail_percentile: Same.
    :param right_tail_percentile: Same.
    :param max_num_pdf_values: Same.
    :param output_dir_name: Same.
    """

    # Check input args.
    num_target_fields = len(target_field_names)
    expected_dim = numpy.array([num_target_fields], dtype=int)

    error_checking.assert_is_numpy_array(
        num_bins_by_target, exact_dimensions=expected_dim
    )
    error_checking.assert_is_integer_numpy_array(num_bins_by_target)
    error_checking.assert_is_geq_numpy_array(num_bins_by_target, 10)
    error_checking.assert_is_leq_numpy_array(num_bins_by_target, 1000)

    error_checking.assert_is_numpy_array(
        min_bin_edge_by_target, exact_dimensions=expected_dim
    )
    error_checking.assert_is_numpy_array(
        max_bin_edge_by_target, exact_dimensions=expected_dim
    )
    for j in range(num_target_fields):
        error_checking.assert_is_greater(
            max_bin_edge_by_target[j],
            min_bin_edge_by_target[j]
        )

    if evaluate_month < 1:
        evaluate_month = None
    if evaluate_hour < 0:
        evaluate_hour = None

    assert evaluate_month is None or evaluate_hour is None

    if evaluate_month is not None:
        error_checking.assert_is_leq(evaluate_month, 12)
    elif evaluate_hour is not None:
        error_checking.assert_is_leq(evaluate_hour, 23)

    error_checking.assert_is_leq(left_tail_percentile, 10.)
    error_checking.assert_is_greater(left_tail_percentile, 0.)
    error_checking.assert_is_geq(right_tail_percentile, 90.)
    error_checking.assert_is_less_than(right_tail_percentile, 100.)

    if max_num_pdf_values <= 0:
        max_num_pdf_values = int(1e15)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Find input files.
    init_time_limits_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, TIME_FORMAT)
        for t in init_time_limit_strings
    ], dtype=int)

    prediction_file_names = prediction_io.find_files_for_period(
        directory_name=prediction_dir_name,
        first_init_time_unix_sec=init_time_limits_unix_sec[0],
        last_init_time_unix_sec=init_time_limits_unix_sec[1],
        raise_error_if_any_missing=False,
        raise_error_if_all_missing=True
    )

    if len(prediction_file_names) == 0:
        prediction_file_names = prediction_io.find_rap_based_files_for_period(
            directory_name=prediction_dir_name,
            first_init_time_unix_sec=init_time_limits_unix_sec[0],
            last_init_time_unix_sec=init_time_limits_unix_sec[1],
            raise_error_if_any_missing=False,
            raise_error_if_all_missing=True
        )

    # Subset to relevant input files, if necessary.
    # first_prediction_table_xarray = prediction_io.read_file(
    #     prediction_file_names[0]
    # )
    # first_ptx = first_prediction_table_xarray
    #
    # if 'model_lead_time_hours' in first_ptx.attrs:
    #     model_lead_time_hours = first_ptx.attrs['model_lead_time_hours']
    # else:
    #     model_file_name = first_ptx.attrs[prediction_io.MODEL_FILE_KEY]
    #     model_metafile_name = neural_net.find_metafile(
    #         model_file_name=model_file_name, raise_error_if_missing=True
    #     )
    #
    #     print('Reading model metadata from: "{0:s}"...'.format(
    #         model_metafile_name
    #     ))
    #     model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    #     model_lead_time_hours = model_metadata_dict[
    #         neural_net.TRAINING_OPTIONS_KEY
    #     ][neural_net.TARGET_LEAD_TIME_KEY]

    model_lead_time_hours = 48

    init_times_unix_sec = numpy.array([
        prediction_io.file_name_to_init_time(f) for f in prediction_file_names
    ], dtype=int)

    valid_times_unix_sec = (
        init_times_unix_sec + model_lead_time_hours * HOURS_TO_SECONDS
    )

    if evaluate_month is not None:
        valid_months = numpy.array([
            int(time_conversion.unix_sec_to_string(t, '%Y%m%d')[4:6])
            for t in valid_times_unix_sec
        ], dtype=int)

        good_indices = numpy.where(valid_months == evaluate_month)[0]
        prediction_file_names = [prediction_file_names[k] for k in good_indices]

    if evaluate_hour is not None:
        valid_hours = numpy.array([
            int(time_conversion.unix_sec_to_string(t, '%Y%m%d%H')[8:])
            for t in valid_times_unix_sec
        ], dtype=int)

        good_indices = numpy.where(valid_hours == evaluate_hour)[0]
        prediction_file_names = [prediction_file_names[k] for k in good_indices]

    del init_times_unix_sec
    del valid_times_unix_sec

    # Do actual stuff.
    prediction_tables_xarray = evaluation.read_inputs(
        prediction_file_names=prediction_file_names,
        target_field_names=target_field_names,
        take_ensemble_mean=True
    )

    prediction_matrix = numpy.stack([
        ptx[prediction_io.PREDICTION_KEY].values[..., 0]
        for ptx in prediction_tables_xarray
    ], axis=0)

    target_matrix = numpy.stack([
        ptx[prediction_io.TARGET_KEY].values
        for ptx in prediction_tables_xarray
    ], axis=0)

    del prediction_tables_xarray

    prediction_matrix = numpy.reshape(
        prediction_matrix, (-1, num_target_fields)
    )
    target_matrix = numpy.reshape(
        target_matrix, (-1, num_target_fields)
    )

    nan_flags = numpy.logical_or(
        numpy.any(numpy.isnan(prediction_matrix), axis=1),
        numpy.any(numpy.isnan(target_matrix), axis=1)
    )
    real_indices = numpy.where(numpy.invert(nan_flags))[0]
    prediction_matrix = prediction_matrix[real_indices, :]
    target_matrix = target_matrix[real_indices, :]

    for j in range(num_target_fields):
        figure_object, axes_object = _plot_error_distribution(
            target_values=target_matrix[:, j],
            predicted_values=prediction_matrix[:, j],
            min_bin_edge=min_bin_edge_by_target[j],
            max_bin_edge=max_bin_edge_by_target[j],
            num_bins=num_bins_by_target[j],
            bin_by_target_or_predicted_values=True,
            violin_or_box_plots=violin_or_box_plots,
            target_field_name=target_field_names[j]
        )

        figure_file_name = '{0:s}/error_dist_by_target_value_{1:s}.jpg'.format(
            output_dir_name,
            target_field_names[j].replace('_', '-')
        )

        print('Saving figure to: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

    for j in range(num_target_fields):
        figure_object, axes_object = _plot_error_distribution(
            target_values=target_matrix[:, j],
            predicted_values=prediction_matrix[:, j],
            min_bin_edge=min_bin_edge_by_target[j],
            max_bin_edge=max_bin_edge_by_target[j],
            num_bins=num_bins_by_target[j],
            bin_by_target_or_predicted_values=False,
            violin_or_box_plots=violin_or_box_plots,
            target_field_name=target_field_names[j]
        )

        figure_file_name = (
            '{0:s}/error_dist_by_predicted_value_{1:s}.jpg'
        ).format(
            output_dir_name,
            target_field_names[j].replace('_', '-')
        )

        print('Saving figure to: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

    for j in range(num_target_fields):
        figure_object, axes_object = _plot_distributions(
            target_values=target_matrix[:, j],
            predicted_values=prediction_matrix[:, j],
            tail_percentile=left_tail_percentile,
            target_field_name=target_field_names[j]
        )

        figure_file_name = '{0:s}/left_tail_comparison_{1:s}.jpg'.format(
            output_dir_name,
            target_field_names[j].replace('_', '-')
        )

        print('Saving figure to: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

    for j in range(num_target_fields):
        figure_object, axes_object = _plot_distributions(
            target_values=target_matrix[:, j],
            predicted_values=prediction_matrix[:, j],
            tail_percentile=right_tail_percentile,
            target_field_name=target_field_names[j]
        )

        figure_file_name = '{0:s}/right_tail_comparison_{1:s}.jpg'.format(
            output_dir_name,
            target_field_names[j].replace('_', '-')
        )

        print('Saving figure to: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

    if target_matrix.shape[0] > max_num_pdf_values:
        combined_matrix = numpy.stack(
            [target_matrix, prediction_matrix], axis=-1
        )
        del target_matrix
        del prediction_matrix

        numpy.random.shuffle(combined_matrix)
        combined_matrix = combined_matrix[:max_num_pdf_values, ...]
        target_matrix = combined_matrix[..., 0]
        prediction_matrix = combined_matrix[..., 1]
        del combined_matrix

    for j in range(num_target_fields):
        figure_object, axes_object = _plot_distributions(
            target_values=target_matrix[:, j],
            predicted_values=prediction_matrix[:, j],
            tail_percentile=None,
            target_field_name=target_field_names[j]
        )

        figure_file_name = '{0:s}/full_dist_comparison_{1:s}.jpg'.format(
            output_dir_name,
            target_field_names[j].replace('_', '-')
        )

        print('Saving figure to: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        init_time_limit_strings=getattr(
            INPUT_ARG_OBJECT, INIT_TIME_LIMITS_ARG_NAME
        ),
        evaluate_month=getattr(INPUT_ARG_OBJECT, EVALUATE_MONTH_ARG_NAME),
        evaluate_hour=getattr(INPUT_ARG_OBJECT, EVALUATE_HOUR_ARG_NAME),
        target_field_names=getattr(
            INPUT_ARG_OBJECT, TARGET_FIELDS_ARG_NAME
        ),
        num_bins_by_target=numpy.array(
            getattr(INPUT_ARG_OBJECT, NUM_BINS_ARG_NAME), dtype=int
        ),
        min_bin_edge_by_target=numpy.array(
            getattr(INPUT_ARG_OBJECT, MIN_BIN_EDGES_ARG_NAME), dtype=float
        ),
        max_bin_edge_by_target=numpy.array(
            getattr(INPUT_ARG_OBJECT, MAX_BIN_EDGES_ARG_NAME), dtype=float
        ),
        violin_or_box_plots=bool(
            getattr(INPUT_ARG_OBJECT, VIOLIN_OR_BOX_ARG_NAME)
        ),
        left_tail_percentile=getattr(
            INPUT_ARG_OBJECT, LEFT_TAIL_PERCENTILE_ARG_NAME
        ),
        right_tail_percentile=getattr(
            INPUT_ARG_OBJECT, RIGHT_TAIL_PERCENTILE_ARG_NAME
        ),
        max_num_pdf_values=getattr(
            INPUT_ARG_OBJECT, MAX_NUM_PDF_VALUES_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
