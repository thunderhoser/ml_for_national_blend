"""Plots ungridded (averaged over the whole domain) model evaluation."""

import os
import sys
import argparse
import numpy
import xarray
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
import matplotlib.patches
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import temperature_conversions as temperature_conv
import file_system_utils
import error_checking
import urma_io
import prediction_io
import urma_utils
import evaluation
import evaluation_plotting as eval_plotting

TARGET_FIELD_NAME_TO_VERBOSE = {
    urma_utils.TEMPERATURE_2METRE_NAME: r'2-m temperature ($^{\circ}$C)',
    urma_utils.DEWPOINT_2METRE_NAME: r'2-m dewpoint ($^{\circ}$C)',
    urma_utils.U_WIND_10METRE_NAME: r'10-m zonal wind (m s$^{-1}$)',
    urma_utils.V_WIND_10METRE_NAME: r'10-m meridional wind (m s$^{-1}$)',
    urma_utils.WIND_GUST_10METRE_NAME: r'10-m wind gust (m s$^{-1}$)'
}

CELSIUS_FIELD_NAMES = [
    urma_utils.TEMPERATURE_2METRE_NAME, urma_utils.DEWPOINT_2METRE_NAME
]

LINE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
POLYGON_OPACITY = 0.5
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_eval_file_name_or_pattern'
BY_MONTH_ARG_NAME = 'by_month'
BY_HOUR_ARG_NAME = 'by_hour'
TARGET_NORM_FILE_ARG_NAME = 'input_target_norm_file_name'
PLOT_FULL_DISTS_ARG_NAME = 'plot_full_error_distributions'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
METRICS_IN_TITLES_ARG_NAME = 'report_metrics_in_titles'
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
TARGET_NORM_FILE_HELP_STRING = (
    'Path to file with normalization parameters for target fields.  Will be '
    'read by `urma_io.read_normalization_file`.'
)
PLOT_FULL_DISTS_HELP_STRING = (
    'Boolean flag.  If 1, for each evaluation set, will plot full error '
    'distribution with boxplot.'
)
CONFIDENCE_LEVEL_HELP_STRING = (
    'Confidence level (from 0...1).  If you do not want to plot confidence '
    'intervals, leave this alone.'
)
METRICS_IN_TITLES_HELP_STRING = (
    'Boolean flag.  If 1 (0), will (not) report overall metrics in panel '
    'titles.'
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
    '--' + TARGET_NORM_FILE_ARG_NAME, type=str, required=True,
    help=TARGET_NORM_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_FULL_DISTS_ARG_NAME, type=int, required=False, default=0,
    help=PLOT_FULL_DISTS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=-1,
    help=CONFIDENCE_LEVEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + METRICS_IN_TITLES_ARG_NAME, type=int, required=False, default=1,
    help=METRICS_IN_TITLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def __change_file_name_for_nwp_forecasts(file_name):
    """Deals with the fact that I renamed directories manually.

    :param file_name: File name.
    :return: file_name: Updated version of input.
    """

    file_name = file_name.replace(
        'ecmwf/processed/interp_to_nbm_grid/prediction_files',
        'ecmwf/processed/interp_to_nbm_grid/prediction_files_48h'
    )
    file_name = file_name.replace(
        'gefs/processed/interp_to_nbm_grid/prediction_files',
        'gefs/processed/interp_to_nbm_grid/prediction_files_48h'
    )
    file_name = file_name.replace(
        'gfs/processed/interp_to_nbm_grid/prediction_files',
        'gfs/processed/interp_to_nbm_grid/prediction_files_48h'
    )
    file_name = file_name.replace(
        'gridded_gfs_mos/processed/interp_to_nbm_grid/prediction_files',
        'gridded_gfs_mos/processed/interp_to_nbm_grid/prediction_files_48h'
    )
    file_name = file_name.replace(
        'gridded_lamp/processed/interp_to_nbm_grid/prediction_files',
        'gridded_lamp/processed/interp_to_nbm_grid/prediction_files_24h'
    )
    file_name = file_name.replace(
        'hrrr/processed/interp_to_nbm_grid/prediction_files',
        'hrrr/processed/interp_to_nbm_grid/prediction_files_48h'
    )
    file_name = file_name.replace(
        'nam/processed/interp_to_nbm_grid/prediction_files',
        'nam/processed/interp_to_nbm_grid/prediction_files_51h'
    )
    file_name = file_name.replace(
        'nam_nest/processed/interp_to_nbm_grid/prediction_files',
        'nam_nest/processed/interp_to_nbm_grid/prediction_files_48h'
    )
    file_name = file_name.replace(
        'rap/processed/interp_to_nbm_grid/prediction_files',
        'rap/processed/interp_to_nbm_grid/prediction_files_48h'
    )
    file_name = file_name.replace(
        'wrf_arw/processed/interp_to_nbm_grid/prediction_files',
        'wrf_arw/processed/interp_to_nbm_grid/prediction_files_48h'
    )

    file_name = file_name.replace('48h_48h', '48h')
    file_name = file_name.replace('24h_24h', '24h')
    file_name = file_name.replace('51h_51h', '51h')

    return file_name


def _plot_attributes_diagram(
        evaluation_tables_xarray, line_styles, line_colours,
        set_descriptions_abbrev, set_descriptions_verbose, confidence_level,
        climo_mean_target_value, target_field_name, report_reliability_in_title,
        title_suffix, output_dir_name, force_plot_legend=False):
    """Plots attributes diagram for each set and each target variable.

    S = number of evaluation sets
    T_v = number of vector target variables
    T_s = number of scalar target variables
    H = number of heights

    :param evaluation_tables_xarray: length-S list of xarray tables in format
        returned by `evaluation.read_file`.
    :param line_styles: length-S list of line styles.
    :param line_colours: length-S list of line colours.
    :param set_descriptions_abbrev: length-S list of abbreviated descriptions
        for evaluation sets.
    :param set_descriptions_verbose: length-S list of verbose descriptions for
        evaluation sets.
    :param confidence_level: See documentation at top of file.
    :param climo_mean_target_value: Mean target value in training data, i.e.,
        "climatology".
    :param target_field_name: Name of target variable.
    :param report_reliability_in_title: Boolean flag.  If True, will report
        overall reliability in title.
    :param title_suffix: End of figure title (string).
    :param output_dir_name: Name of output directory.  Figures will be saved
        here.
    :param force_plot_legend: Boolean flag.
    """

    target_indices = numpy.array([
        numpy.where(
            etx.coords[evaluation.FIELD_DIM].values == target_field_name
        )[0][0]
        for etx in evaluation_tables_xarray
    ], dtype=int)

    mean_predictions_by_set = [
        etx[evaluation.RELIABILITY_X_KEY].values[k, ...]
        for etx, k in zip(evaluation_tables_xarray, target_indices)
    ]
    mean_observations_by_set = [
        etx[evaluation.RELIABILITY_Y_KEY].values[k, ...]
        for etx, k in zip(evaluation_tables_xarray, target_indices)
    ]
    bin_centers_by_set = [
        etx[evaluation.RELIABILITY_BIN_CENTER_KEY].values[k, :]
        for etx, k in zip(evaluation_tables_xarray, target_indices)
    ]
    example_counts_by_set = [
        etx[evaluation.RELIABILITY_COUNT_KEY].values[k, :]
        for etx, k in zip(evaluation_tables_xarray, target_indices)
    ]
    inverted_bin_centers_by_set = [
        etx[evaluation.INV_RELIABILITY_BIN_CENTER_KEY].values[k, :]
        for etx, k in zip(evaluation_tables_xarray, target_indices)
    ]
    inverted_example_counts_by_set = [
        etx[evaluation.INV_RELIABILITY_COUNT_KEY].values[k, :]
        for etx, k in zip(evaluation_tables_xarray, target_indices)
    ]
    reliabilities_by_set = [
        etx[evaluation.RELIABILITY_KEY].values[k, :]
        for etx, k in zip(evaluation_tables_xarray, target_indices)
    ]
    mse_skill_scores_by_set = [
        etx[evaluation.MSE_SKILL_SCORE_KEY].values[k, :]
        for etx, k in zip(evaluation_tables_xarray, target_indices)
    ]

    concat_values = numpy.concatenate([
        numpy.nanmean(a, axis=-1)
        for a in mean_predictions_by_set + mean_observations_by_set
        if a is not None
    ])

    if numpy.all(numpy.isnan(concat_values)):
        return

    max_value_to_plot = numpy.nanpercentile(concat_values, 100.)
    min_value_to_plot = numpy.nanpercentile(concat_values, 0.)
    num_evaluation_sets = len(evaluation_tables_xarray)

    for main_index in range(num_evaluation_sets):
        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

        legend_handles = []
        legend_strings = []

        this_handle = eval_plotting.plot_attributes_diagram(
            figure_object=figure_object,
            axes_object=axes_object,
            mean_predictions=
            numpy.nanmean(mean_predictions_by_set[main_index], axis=-1),
            mean_observations=
            numpy.nanmean(mean_observations_by_set[main_index], axis=-1),
            mean_value_in_training=climo_mean_target_value,
            min_value_to_plot=min_value_to_plot,
            max_value_to_plot=max_value_to_plot,
            line_colour=line_colours[main_index],
            line_style=line_styles[main_index],
            line_width=4
        )

        if this_handle is not None:
            legend_handles.append(this_handle)
            legend_strings.append(set_descriptions_verbose[main_index])

        num_bootstrap_reps = mean_predictions_by_set[main_index].shape[1]

        if num_bootstrap_reps > 1 and confidence_level is not None:
            polygon_coord_matrix = evaluation.confidence_interval_to_polygon(
                x_value_matrix=mean_predictions_by_set[main_index],
                y_value_matrix=mean_observations_by_set[main_index],
                confidence_level=confidence_level,
                same_order=False
            )

            polygon_colour = matplotlib.colors.to_rgba(
                line_colours[main_index], POLYGON_OPACITY
            )
            patch_object = matplotlib.patches.Polygon(
                polygon_coord_matrix, lw=0, ec=polygon_colour, fc=polygon_colour
            )
            axes_object.add_patch(patch_object)

        eval_plotting.plot_inset_histogram(
            figure_object=figure_object,
            bin_centers=bin_centers_by_set[main_index],
            bin_counts=example_counts_by_set[main_index],
            has_predictions=True,
            bar_colour=line_colours[main_index]
        )

        eval_plotting.plot_inset_histogram(
            figure_object=figure_object,
            bin_centers=inverted_bin_centers_by_set[main_index],
            bin_counts=inverted_example_counts_by_set[main_index],
            has_predictions=False,
            bar_colour=line_colours[main_index]
        )

        # eval_plotting.plot_inset_histogram(
        #     figure_object=figure_object,
        #     bin_centers=bin_centers_by_set[main_index],
        #     bin_counts=inverted_example_counts_by_set[main_index],
        #     has_predictions=False,
        #     bar_colour=line_colours[main_index]
        # )

        axes_object.set_xlabel('Prediction')
        axes_object.set_ylabel('Conditional mean observation')

        title_string = 'Attributes diagram for {0:s}{1:s}'.format(
            TARGET_FIELD_NAME_TO_VERBOSE[target_field_name],
            title_suffix
        )
        if report_reliability_in_title:
            title_string += '\nREL = {0:.2f}; MSESS = {1:.2f}'.format(
                numpy.mean(reliabilities_by_set[main_index]),
                numpy.mean(mse_skill_scores_by_set[main_index])
            )

        axes_object.set_title(title_string)

        for i in range(num_evaluation_sets):
            if i == main_index:
                continue

            this_handle = eval_plotting._plot_reliability_curve(
                axes_object=axes_object,
                mean_predictions=
                numpy.nanmean(mean_predictions_by_set[i], axis=-1),
                mean_observations=
                numpy.nanmean(mean_observations_by_set[i], axis=-1),
                min_value_to_plot=min_value_to_plot,
                max_value_to_plot=max_value_to_plot,
                line_colour=line_colours[i],
                line_style=line_styles[i],
                line_width=4
            )

            if this_handle is not None:
                legend_handles.append(this_handle)
                legend_strings.append(set_descriptions_verbose[i])

            num_bootstrap_reps = mean_predictions_by_set[i].shape[1]

            if num_bootstrap_reps > 1 and confidence_level is not None:
                polygon_coord_matrix = (
                    evaluation.confidence_interval_to_polygon(
                        x_value_matrix=mean_predictions_by_set[i],
                        y_value_matrix=mean_observations_by_set[i],
                        confidence_level=confidence_level,
                        same_order=False
                    )
                )

                polygon_colour = matplotlib.colors.to_rgba(
                    line_colours[i], POLYGON_OPACITY
                )
                patch_object = matplotlib.patches.Polygon(
                    polygon_coord_matrix, lw=0,
                    ec=polygon_colour, fc=polygon_colour
                )
                axes_object.add_patch(patch_object)

        if len(legend_handles) > 1 or force_plot_legend:
            axes_object.legend(
                legend_handles, legend_strings, loc='center left',
                bbox_to_anchor=(0, 0.35), fancybox=True, shadow=False,
                facecolor='white', edgecolor='k', framealpha=0.5, ncol=1
            )

        figure_file_name = '{0:s}/{1:s}_attributes{2:s}.jpg'.format(
            output_dir_name,
            target_field_name.replace('_', '-'),
            set_descriptions_abbrev[main_index]
        )

        print('Saving figure to: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)


def _run(eval_file_name_or_pattern, by_month, by_hour,
         target_normalization_file_name, plot_full_error_distributions,
         confidence_level, report_metrics_in_titles, output_dir_name):
    """Plots ungridded (averaged over the whole domain) model evaluation.

    This is effectively the main method.

    :param eval_file_name_or_pattern: See documentation at top of file.
    :param by_month: Same.
    :param by_hour: Same.
    :param target_normalization_file_name: Same.
    :param plot_full_error_distributions: Same.
    :param confidence_level: Same.
    :param report_metrics_in_titles: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if any input file contains gridded, rather than
        ungridded, evaluation.
    """

    # Check input args.
    assert not (by_month and by_hour)

    if confidence_level < 0:
        confidence_level = None
    if confidence_level is not None:
        error_checking.assert_is_geq(confidence_level, 0.9)
        error_checking.assert_is_less_than(confidence_level, 1.)

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

        if evaluation.ROW_DIM in evaluation_tables_xarray[i].coords:
            error_string = (
                'File "{0:s}" contains gridded evaluation.  This script '
                'handles only ungridded evaluation.'
            ).format(
                evaluation_file_names[i]
            )

            raise ValueError(error_string)

        etx_i = evaluation_tables_xarray[i]

        if target_field_names is None:
            target_field_names = (
                etx_i.coords[evaluation.FIELD_DIM].values.tolist()
            )

        assert (
            target_field_names ==
            etx_i.coords[evaluation.FIELD_DIM].values.tolist()
        )

    print('Reading normalization params from: "{0:s}"...'.format(
        target_normalization_file_name
    ))
    target_norm_param_table_xarray = urma_io.read_normalization_file(
        target_normalization_file_name
    )
    tnpt = target_norm_param_table_xarray

    these_indices = numpy.array([
        numpy.where(tnpt.coords[urma_utils.FIELD_DIM].values == f)[0][0]
        for f in target_field_names
    ], dtype=int)

    climo_mean_target_values = (
        tnpt[urma_utils.MEAN_VALUE_KEY].values[these_indices]
    )

    celsius_indices = numpy.array(
        [f in CELSIUS_FIELD_NAMES for f in target_field_names], dtype=bool
    )
    for j in celsius_indices:
        climo_mean_target_values[j] = temperature_conv.kelvins_to_celsius(
            climo_mean_target_values[j]
        )

    num_target_fields = len(target_field_names)

    for i in range(num_files):
        if not evaluation_tables_xarray[i].data_vars:
            continue

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
            _plot_attributes_diagram(
                evaluation_tables_xarray=[evaluation_tables_xarray[i]],
                line_styles=['solid'],
                line_colours=[LINE_COLOUR],
                set_descriptions_abbrev=[file_name_suffix],
                set_descriptions_verbose=[file_name_suffix],
                confidence_level=confidence_level,
                climo_mean_target_value=climo_mean_target_values[k],
                target_field_name=target_field_names[k],
                report_reliability_in_title=report_metrics_in_titles,
                title_suffix=title_suffix,
                output_dir_name=output_dir_name
            )

            figure_object, axes_object = pyplot.subplots(
                1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
            )
            etx = evaluation_tables_xarray[i]

            eval_plotting.plot_taylor_diagram(
                target_stdev=numpy.nanmean(
                    etx[evaluation.TARGET_STDEV_KEY].values[k, :]
                ),
                prediction_stdev=numpy.nanmean(
                    etx[evaluation.PREDICTION_STDEV_KEY].values[k, :]
                ),
                correlation=numpy.nanmean(
                    etx[evaluation.CORRELATION_KEY].values[k, :]
                ),
                marker_colour=LINE_COLOUR,
                axes_object=axes_object,
                figure_object=figure_object
            )

            title_string = 'Taylor diagram for {0:s}{1:s}'.format(
                TARGET_FIELD_NAME_TO_VERBOSE[target_field_names[k]],
                title_suffix
            )
            if report_metrics_in_titles:
                title_string += (
                    '\nActual/pred stdevs = {0:.2f}, {1:.2f}; corr = {2:.2f}'
                ).format(
                    numpy.nanmean(
                        etx[evaluation.TARGET_STDEV_KEY].values[k, :]
                    ),
                    numpy.nanmean(
                        etx[evaluation.PREDICTION_STDEV_KEY].values[k, :]
                    ),
                    numpy.nanmean(
                        etx[evaluation.CORRELATION_KEY].values[k, :]
                    )
                )

            axes_object.set_title(title_string)

            figure_file_name = '{0:s}/{1:s}_taylor{2:s}.jpg'.format(
                output_dir_name,
                target_field_names[k].replace('_', '-'),
                file_name_suffix
            )

            print('Saving figure to: "{0:s}"...'.format(figure_file_name))
            figure_object.savefig(
                figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)

    if not plot_full_error_distributions:
        return

    for i in range(num_files):
        if not evaluation_tables_xarray[i].data_vars:
            continue

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

        prediction_file_names = evaluation_tables_xarray[i].attrs[
            evaluation.PREDICTION_FILES_KEY
        ]
        prediction_file_names = [
            __change_file_name_for_nwp_forecasts(f)
            for f in prediction_file_names
        ]

        num_times = len(prediction_file_names)
        error_matrix = numpy.array([], dtype=float)

        for j in range(num_times):
            print('Reading data from: "{0:s}"...'.format(
                prediction_file_names[j]
            ))

            this_prediction_table_xarray = evaluation.read_inputs(
                prediction_file_names=[prediction_file_names[j]],
                target_field_names=target_field_names,
                take_ensemble_mean=True
            )[0]
            ptx = this_prediction_table_xarray

            this_prediction_matrix = (
                ptx[prediction_io.PREDICTION_KEY].values[..., 0]
            )
            this_target_matrix = ptx[prediction_io.TARGET_KEY].values

            if error_matrix.size == 0:
                error_matrix = numpy.full(
                    (num_times,) + this_target_matrix.shape, numpy.nan
                )

            error_matrix[j, ...] = this_prediction_matrix - this_target_matrix

        for k in range(num_target_fields):
            error_values = numpy.ravel(error_matrix[..., k])
            error_values = error_values[numpy.invert(numpy.isnan(error_values))]

            figure_object, axes_object = pyplot.subplots(
                1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
            )
            eval_plotting.plot_error_distribution(
                error_values=error_values,
                min_error_to_plot=numpy.percentile(error_values, 0.),
                max_error_to_plot=numpy.percentile(error_values, 100.),
                axes_object=axes_object
            )

            title_string = 'Error distribution for {0:s}{1:s}'.format(
                TARGET_FIELD_NAME_TO_VERBOSE[target_field_names[k]],
                title_suffix
            )
            axes_object.set_title(title_string)

            figure_file_name = (
                '{0:s}/{1:s}_error-distribution{2:s}.jpg'
            ).format(
                output_dir_name,
                target_field_names[k].replace('_', '-'),
                file_name_suffix
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
        eval_file_name_or_pattern=getattr(
            INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME
        ),
        by_month=bool(getattr(INPUT_ARG_OBJECT, BY_MONTH_ARG_NAME)),
        by_hour=bool(getattr(INPUT_ARG_OBJECT, BY_HOUR_ARG_NAME)),
        target_normalization_file_name=getattr(
            INPUT_ARG_OBJECT, TARGET_NORM_FILE_ARG_NAME
        ),
        plot_full_error_distributions=bool(
            getattr(INPUT_ARG_OBJECT, PLOT_FULL_DISTS_ARG_NAME)
        ),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        report_metrics_in_titles=bool(
            getattr(INPUT_ARG_OBJECT, METRICS_IN_TITLES_ARG_NAME)
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
