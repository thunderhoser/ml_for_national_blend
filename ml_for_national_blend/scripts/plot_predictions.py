"""Plots NN-based predictions and targets (actual values)."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from ml_for_national_blend.outside_code import time_conversion
from ml_for_national_blend.outside_code import longitude_conversion as lng_conversion
from ml_for_national_blend.outside_code import file_system_utils
from ml_for_national_blend.outside_code import error_checking
from ml_for_national_blend.io import border_io
from ml_for_national_blend.io import prediction_io
from ml_for_national_blend.utils import nbm_utils
from ml_for_national_blend.utils import urma_utils
from ml_for_national_blend.utils import misc_utils
from ml_for_national_blend.machine_learning import neural_net
from ml_for_national_blend.machine_learning import nwp_input
from ml_for_national_blend.plotting import target_plotting
from ml_for_national_blend.plotting import plotting_utils

TOLERANCE = 1e-6
HOURS_TO_SECONDS = 3600
TIME_FORMAT = '%Y-%m-%d-%H'

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_DIR_ARG_NAME = 'input_prediction_dir_name'
INIT_TIME_ARG_NAME = 'init_time_string'
FIELDS_ARG_NAME = 'field_names'
MIN_VALUES_ARG_NAME = 'min_colour_values'
MAX_VALUES_ARG_NAME = 'max_colour_values'
MIN_PERCENTILES_ARG_NAME = 'min_colour_percentiles'
MAX_PERCENTILES_ARG_NAME = 'max_colour_percentiles'
PLOT_DIFFS_ARG_NAME = 'plot_diffs'
MAX_DIFF_VALUES_ARG_NAME = 'max_colour_values_for_diff'
MAX_DIFF_PERCENTILES_ARG_NAME = 'max_colour_percentiles_for_diff'
BASELINE_MODEL_ARG_NAME = 'baseline_nwp_model_name'
BASELINE_MODEL_DIR_ARG_NAME = 'baseline_nwp_model_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Path to input directory.  The relevant file therein will be found by '
    '`prediction_io.find_file` and read by `prediction_io.read_file`.'
)
INIT_TIME_HELP_STRING = 'Forecast-initialization time (format "yyyy-mm-dd-HH").'
FIELDS_HELP_STRING = (
    '1-D list of fields (will plot predictions and targets for each).  Each '
    'field must be accepted by `urma_utils.check_field_name`.'
)
MIN_VALUES_HELP_STRING = (
    '1-D list of minimum values in colour scheme (must have same length as '
    '{0:s}).  If you would rather specify minimum values by percentile, leave '
    'this argument alone.'
).format(
    FIELDS_ARG_NAME
)
MAX_VALUES_HELP_STRING = 'Same as {0:s} but for max values.'.format(
    MIN_VALUES_ARG_NAME
)
MIN_PERCENTILES_HELP_STRING = (
    '1-D list of percentiles (must have same length as {0:s}).  For the [k]th '
    'field at the [i]th time step, the minimum value in the colour scheme will '
    'be percentile {1:s}[k] of all {0:s}[k] values at the [i]th time step.'
).format(
    FIELDS_ARG_NAME, MIN_PERCENTILES_ARG_NAME
)
MAX_PERCENTILES_HELP_STRING = 'Same as {0:s} but for max values.'.format(
    MIN_PERCENTILES_ARG_NAME
)
PLOT_DIFFS_HELP_STRING = (
    'Boolean flag.  If 1, will plot difference fields (predicted minus actual) '
    'in addition to fundamental fields (predicted and actual).  If 0, will '
    'plot only predicted and actual.'
)
MAX_DIFF_VALUES_HELP_STRING = 'Same as {0:s} but for difference fields.'.format(
    MAX_DIFF_VALUES_ARG_NAME
)
MAX_DIFF_PERCENTILES_HELP_STRING = (
    'Same as {0:s} but for difference fields.'
).format(
    MAX_DIFF_PERCENTILES_ARG_NAME
)
BASELINE_MODEL_HELP_STRING = (
    'Name of baseline model (must be accepted by '
    '`nwp_model_utils.check_model_name`).  If you do not want a baseline '
    'model, leave this argument alone.'
)
BASELINE_MODEL_DIR_HELP_STRING = (
    '[used only if {0:s} is non-empty] Path to directory for baseline model.  '
    'Files therein will be found by `interp_nwp_model_io.find_file` and read '
    'by `interp_nwp_model_io.read_file`.'
).format(BASELINE_MODEL_ARG_NAME)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INIT_TIME_ARG_NAME, type=str, required=True,
    help=INIT_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIELDS_ARG_NAME, type=str, nargs='+', required=True,
    help=FIELDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_VALUES_ARG_NAME, type=float, nargs='+', required=False,
    default=[1.], help=MIN_VALUES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_VALUES_ARG_NAME, type=float, nargs='+', required=False,
    default=[-1.], help=MAX_VALUES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_PERCENTILES_ARG_NAME, type=float, nargs='+', required=False,
    default=[1.], help=MIN_PERCENTILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILES_ARG_NAME, type=float, nargs='+', required=False,
    default=[-1.], help=MAX_PERCENTILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_DIFFS_ARG_NAME, type=int, required=True,
    help=PLOT_DIFFS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_DIFF_VALUES_ARG_NAME, type=float, nargs='+', required=False,
    default=[-1.], help=MAX_DIFF_VALUES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_DIFF_PERCENTILES_ARG_NAME, type=float, nargs='+', required=False,
    default=[-1.], help=MAX_DIFF_PERCENTILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BASELINE_MODEL_ARG_NAME, type=str, required=False, default='',
    help=BASELINE_MODEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BASELINE_MODEL_DIR_ARG_NAME, type=str, required=False, default='',
    help=BASELINE_MODEL_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_one_field(
        data_matrix, latitude_matrix_deg_n, longitude_matrix_deg_e,
        border_latitudes_deg_n, border_longitudes_deg_e,
        colour_map_object, colour_norm_object,
        title_string, output_file_name):
    """Plots one field.

    M = number of rows in grid
    N = number of columns in grid
    P = number of points in border file

    :param data_matrix: M-by-N numpy array of data values.
    :param latitude_matrix_deg_n: M-by-N numpy array of latitudes (deg north).
    :param longitude_matrix_deg_e: M-by-N numpy array of longitudes (deg east).
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

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    target_plotting.plot_field(
        data_matrix=data_matrix,
        latitude_matrix_deg_n=latitude_matrix_deg_n,
        longitude_matrix_deg_e=longitude_matrix_deg_e,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        axes_object=axes_object,
        plot_colour_bar=True
    )

    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object,
        line_colour=numpy.full(3, 0.)
    )
    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=numpy.ravel(latitude_matrix_deg_n),
        plot_longitudes_deg_e=numpy.ravel(longitude_matrix_deg_e),
        axes_object=axes_object,
        meridian_spacing_deg=20.,
        parallel_spacing_deg=10.
    )

    axes_object.set_xlim(
        numpy.min(longitude_matrix_deg_e),
        numpy.max(longitude_matrix_deg_e)
    )
    axes_object.set_ylim(
        numpy.min(latitude_matrix_deg_n),
        numpy.max(latitude_matrix_deg_n)
    )
    axes_object.set_title(title_string)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _check_colour_limit_args(
        min_colour_values, max_colour_values,
        min_colour_percentiles, max_colour_percentiles, num_fields):
    """Error-checks arguments pertaining to colour limits.

    :param min_colour_values: See documentation at top of this script.
    :param max_colour_values: Same.
    :param min_colour_percentiles: Same.
    :param max_colour_percentiles: Same.
    :param num_fields: Number of fields to plot.
    :return: min_colour_values: Same as input but might be modified.
    :return: max_colour_values: Same as input but might be modified.
    :return: min_colour_percentiles: Same as input but might be modified.
    :return: max_colour_percentiles: Same as input but might be modified.
    """

    if (
            len(min_colour_values) == len(max_colour_values) == 1 and
            max_colour_values[0] < min_colour_values[0]
    ):
        min_colour_values = None
        max_colour_values = None

    if (
            len(min_colour_percentiles) == len(max_colour_percentiles) == 1 and
            max_colour_percentiles[0] < min_colour_percentiles[0]
    ):
        min_colour_percentiles = None
        max_colour_percentiles = None

    assert min_colour_values is not None or min_colour_percentiles is not None

    if min_colour_values is not None:
        error_checking.assert_is_numpy_array(
            min_colour_values,
            exact_dimensions=numpy.array([num_fields], dtype=int)
        )
        error_checking.assert_is_greater_numpy_array(
            max_colour_values - min_colour_values, 0.
        )

    if min_colour_percentiles is not None:
        error_checking.assert_is_numpy_array(
            min_colour_percentiles,
            exact_dimensions=numpy.array([num_fields], dtype=int)
        )
        error_checking.assert_is_geq_numpy_array(min_colour_percentiles, 0.)
        error_checking.assert_is_leq_numpy_array(max_colour_percentiles, 100.)
        error_checking.assert_is_greater_numpy_array(
            max_colour_percentiles - min_colour_percentiles, 0.
        )

    return (
        min_colour_values, max_colour_values,
        min_colour_percentiles, max_colour_percentiles
    )


def _plot_everything_1sample(
        prediction_table_xarray, example_index, field_names, lead_time_hours,
        baseline_nwp_model_name, baseline_nwp_model_dir_name,
        min_colour_values, max_colour_values,
        min_colour_percentiles, max_colour_percentiles,
        border_latitudes_deg_n, border_longitudes_deg_e,
        plot_diffs, min_colour_values_for_diff, max_colour_values_for_diff,
        min_colour_percentiles_for_diff, max_colour_percentiles_for_diff,
        output_dir_name):
    """Plots everything for one data sample.

    P = number of points in border set

    :param prediction_table_xarray: xarray table in format returned by
        `prediction_io.read_file`.
    :param example_index: Will plot the [i]th data sample, where
        i = `example_index`.
    :param field_names: See documentation at top of this script.
    :param lead_time_hours: Lead time.
    :param baseline_nwp_model_name: See documentation at top of this script.
    :param baseline_nwp_model_dir_name: Same.
    :param min_colour_values: Same.
    :param max_colour_values: Same.
    :param min_colour_percentiles: Same.
    :param max_colour_percentiles: Same.
    :param border_latitudes_deg_n: length-P numpy array of latitudes (deg
        north).
    :param border_longitudes_deg_e: length-P numpy array of longitudes (deg
        east).
    :param plot_diffs: See documentation at top of this script.
    :param min_colour_values_for_diff: Same.
    :param max_colour_values_for_diff: Same.
    :param min_colour_percentiles_for_diff: Same.
    :param max_colour_percentiles_for_diff: Same.
    :param output_dir_name: Same.
    """

    i = example_index
    ptx = prediction_table_xarray

    init_time_unix_sec = ptx[prediction_io.INIT_TIME_KEY].values[i]
    valid_time_unix_sec = (
        init_time_unix_sec + lead_time_hours * HOURS_TO_SECONDS
    )
    init_time_string = time_conversion.unix_sec_to_string(
        init_time_unix_sec, TIME_FORMAT
    )
    valid_time_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, TIME_FORMAT
    )

    plotting_patches = not (
        len(ptx.coords[prediction_io.ROW_DIM].values) ==
        len(nbm_utils.NBM_Y_COORDS_METRES)
        and len(ptx.coords[prediction_io.COLUMN_DIM].values) ==
        len(nbm_utils.NBM_X_COORDS_METRES)
    )

    if baseline_nwp_model_name is None:
        baseline_prediction_matrix = None
    else:
        if plotting_patches:
            full_latitude_matrix_deg_n, full_longitude_matrix_deg_e = (
                nbm_utils.read_coords()
            )
            full_longitude_matrix_deg_e = (
                lng_conversion.convert_lng_positive_in_west(
                    full_longitude_matrix_deg_e
                )
            )

            patch_start_latitude_deg_n = (
                ptx[prediction_io.LATITUDE_KEY].values[i, 0, 0]
            )
            patch_start_longitude_deg_e = (
                ptx[prediction_io.LONGITUDE_KEY].values[i, 0, 0]
            )
            patch_start_longitude_deg_e = (
                lng_conversion.convert_lng_positive_in_west(
                    patch_start_longitude_deg_e
                )
            )

            good_indices = numpy.where(numpy.logical_and(
                numpy.absolute(
                    full_latitude_matrix_deg_n -
                    patch_start_latitude_deg_n
                ) < TOLERANCE,
                numpy.absolute(
                    full_longitude_matrix_deg_e -
                    patch_start_longitude_deg_e
                ) < TOLERANCE
            ))

            patch_location_dict = misc_utils.determine_patch_locations(
                patch_size_2pt5km_pixels=
                len(ptx.coords[prediction_io.ROW_DIM].values),
                start_row_2pt5km=good_indices[0][0],
                start_column_2pt5km=good_indices[1][0]
            )
        else:
            patch_location_dict = None

        baseline_prediction_matrix = (
            nwp_input.read_residual_baseline_one_example(
                init_time_unix_sec=init_time_unix_sec,
                nwp_model_name=baseline_nwp_model_name,
                nwp_lead_time_hours=lead_time_hours,
                nwp_directory_name=baseline_nwp_model_dir_name,
                target_field_names=field_names,
                patch_location_dict=patch_location_dict,
                predict_dewpoint_depression=False,
                predict_gust_factor=False
            )
        )

    for j in range(len(field_names)):
        this_field_name_fancy = target_plotting.FIELD_NAME_TO_FANCY[
            field_names[j]
        ]
        j_new = numpy.where(
            ptx[prediction_io.FIELD_NAME_KEY].values == field_names[j]
        )[0][0]

        prediction_matrix = (
            ptx[prediction_io.PREDICTION_KEY].values[i, ..., j_new]
        )
        target_matrix = (
            ptx[prediction_io.TARGET_KEY].values[i, ..., j_new]
        )

        title_string = 'Predicted {0:s}\nInit {1:s}, valid {2:s}'.format(
            this_field_name_fancy,
            init_time_string,
            valid_time_string
        )

        output_file_name = (
            '{0:s}/init={1:s}_valid={2:s}_{3:s}_example{4:04d}_predicted.jpg'
        ).format(
            output_dir_name,
            init_time_string,
            valid_time_string,
            field_names[j].replace('_', '-'),
            example_index
        )

        if min_colour_values is None:
            if baseline_prediction_matrix is None:
                these_matrices = [prediction_matrix, target_matrix]
            else:
                these_matrices = [
                    prediction_matrix,
                    target_matrix,
                    baseline_prediction_matrix[..., j]
                ]

            concat_matrix = numpy.stack(these_matrices, axis=-1)

            (
                colour_map_object, colour_norm_object
            ) = target_plotting.field_to_colour_scheme(
                field_name=field_names[j],
                min_value=
                numpy.nanpercentile(concat_matrix, min_colour_percentiles[j]),
                max_value=
                numpy.nanpercentile(concat_matrix, max_colour_percentiles[j]),
            )
        else:
            (
                colour_map_object, colour_norm_object
            ) = target_plotting.field_to_colour_scheme(
                field_name=field_names[j],
                min_value=min_colour_values[j],
                max_value=max_colour_values[j]
            )

        _plot_one_field(
            data_matrix=prediction_matrix,
            latitude_matrix_deg_n=
            ptx[prediction_io.LATITUDE_KEY].values[i, ...],
            longitude_matrix_deg_e=
            ptx[prediction_io.LONGITUDE_KEY].values[i, ...],
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            title_string=title_string,
            output_file_name=output_file_name
        )

        title_string = 'Actual {0:s}\nValid {1:s}'.format(
            this_field_name_fancy, valid_time_string
        )

        output_file_name = (
            '{0:s}/init={1:s}_valid={2:s}_{3:s}_example{4:04d}_actual.jpg'
        ).format(
            output_dir_name,
            init_time_string,
            valid_time_string,
            field_names[j].replace('_', '-'),
            example_index
        )

        _plot_one_field(
            data_matrix=target_matrix,
            latitude_matrix_deg_n=
            ptx[prediction_io.LATITUDE_KEY].values[i, ...],
            longitude_matrix_deg_e=
            ptx[prediction_io.LONGITUDE_KEY].values[i, ...],
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            title_string=title_string,
            output_file_name=output_file_name
        )

        if baseline_prediction_matrix is not None:
            title_string = 'Baseline {0:s}\nValid {1:s}'.format(
                this_field_name_fancy, valid_time_string
            )

            output_file_name = (
                '{0:s}/init={1:s}_valid={2:s}_{3:s}_example{4:04d}_baseline.jpg'
            ).format(
                output_dir_name,
                init_time_string,
                valid_time_string,
                field_names[j].replace('_', '-'),
                example_index
            )

            _plot_one_field(
                data_matrix=baseline_prediction_matrix[..., j],
                latitude_matrix_deg_n=
                ptx[prediction_io.LATITUDE_KEY].values[i, ...],
                longitude_matrix_deg_e=
                ptx[prediction_io.LONGITUDE_KEY].values[i, ...],
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                colour_map_object=colour_map_object,
                colour_norm_object=colour_norm_object,
                title_string=title_string,
                output_file_name=output_file_name
            )

        if not plot_diffs:
            continue

        if max_colour_values_for_diff is None:
            if baseline_prediction_matrix is None:
                relevant_matrix = prediction_matrix - target_matrix
            else:
                relevant_matrix = numpy.stack([
                    prediction_matrix - target_matrix,
                    baseline_prediction_matrix[..., j] - target_matrix
                ], axis=-1)

            this_max = numpy.nanpercentile(
                numpy.absolute(relevant_matrix),
                max_colour_percentiles_for_diff[j]
            )

            (
                colour_map_object, colour_norm_object
            ) = target_plotting.field_to_colour_scheme(
                field_name=urma_utils.U_WIND_10METRE_NAME,
                min_value=-1 * this_max,
                max_value=this_max,
            )
        else:
            (
                colour_map_object, colour_norm_object
            ) = target_plotting.field_to_colour_scheme(
                field_name=field_names[j],
                min_value=min_colour_values_for_diff[j],
                max_value=max_colour_values_for_diff[j]
            )

        title_string = 'Error in {0:s}\nValid {1:s}'.format(
            this_field_name_fancy, valid_time_string
        )

        output_file_name = (
            '{0:s}/init={1:s}_valid={2:s}_{3:s}_example{4:04d}_error.jpg'
        ).format(
            output_dir_name,
            init_time_string,
            valid_time_string,
            field_names[j].replace('_', '-'),
            example_index
        )

        _plot_one_field(
            data_matrix=prediction_matrix - target_matrix,
            latitude_matrix_deg_n=
            ptx[prediction_io.LATITUDE_KEY].values[i, ...],
            longitude_matrix_deg_e=
            ptx[prediction_io.LONGITUDE_KEY].values[i, ...],
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            title_string=title_string,
            output_file_name=output_file_name
        )

        if baseline_prediction_matrix is None:
            continue

        title_string = 'Baseline error in {0:s}\nValid {1:s}'.format(
            this_field_name_fancy, valid_time_string
        )

        output_file_name = (
            '{0:s}/init={1:s}_valid={2:s}_{3:s}_example{4:04d}_'
            'baseline-error.jpg'
        ).format(
            output_dir_name,
            init_time_string,
            valid_time_string,
            field_names[j].replace('_', '-'),
            example_index
        )

        _plot_one_field(
            data_matrix=baseline_prediction_matrix[..., j] - target_matrix,
            latitude_matrix_deg_n=
            ptx[prediction_io.LATITUDE_KEY].values[i, ...],
            longitude_matrix_deg_e=
            ptx[prediction_io.LONGITUDE_KEY].values[i, ...],
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            title_string=title_string,
            output_file_name=output_file_name
        )


def _run(prediction_dir_name, init_time_string, field_names,
         min_colour_values, max_colour_values,
         min_colour_percentiles, max_colour_percentiles, plot_diffs,
         max_colour_values_for_diff, max_colour_percentiles_for_diff,
         baseline_nwp_model_name, baseline_nwp_model_dir_name,
         output_dir_name):
    """Plots NN-based predictions and targets (actual values).

    This is effectively the main method.

    :param prediction_dir_name: See documentation at top of this script.
    :param init_time_string: Same.
    :param field_names: Same.
    :param min_colour_values: Same.
    :param max_colour_values: Same.
    :param min_colour_percentiles: Same.
    :param max_colour_percentiles: Same.
    :param plot_diffs: Same.
    :param max_colour_values_for_diff: Same.
    :param max_colour_percentiles_for_diff: Same.
    :param baseline_nwp_model_name: Same.
    :param baseline_nwp_model_dir_name: Same.
    :param output_dir_name: Same.
    """

    # Check input args.
    init_time_unix_sec = time_conversion.string_to_unix_sec(
        init_time_string, TIME_FORMAT
    )
    assert all([f in urma_utils.ALL_FIELD_NAMES for f in field_names])

    if baseline_nwp_model_name == '':
        baseline_nwp_model_name = None

    (
        min_colour_values,
        max_colour_values,
        min_colour_percentiles,
        max_colour_percentiles
    ) = _check_colour_limit_args(
        min_colour_values=min_colour_values,
        max_colour_values=max_colour_values,
        min_colour_percentiles=min_colour_percentiles,
        max_colour_percentiles=max_colour_percentiles,
        num_fields=len(field_names)
    )

    if (
            len(max_colour_values_for_diff) == 1
            and max_colour_values_for_diff[0] < 0
    ):
        min_colour_values_for_diff = numpy.array([1.])
    else:
        min_colour_values_for_diff = -1 * max_colour_values_for_diff

    if (
            len(max_colour_percentiles_for_diff) == 1
            and max_colour_percentiles_for_diff[0] < 0
    ):
        min_colour_percentiles_for_diff = numpy.array([1.])
    else:
        min_colour_percentiles_for_diff = 100. - max_colour_percentiles_for_diff

    (
        min_colour_values_for_diff,
        max_colour_values_for_diff,
        min_colour_percentiles_for_diff,
        max_colour_percentiles_for_diff
    ) = _check_colour_limit_args(
        min_colour_values=min_colour_values_for_diff,
        max_colour_values=max_colour_values_for_diff,
        min_colour_percentiles=min_colour_percentiles_for_diff,
        max_colour_percentiles=max_colour_percentiles_for_diff,
        num_fields=len(field_names)
    )

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Do actual stuff.
    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    prediction_file_name = prediction_io.find_file(
        directory_name=prediction_dir_name,
        init_time_unix_sec=init_time_unix_sec,
        raise_error_if_missing=True
    )

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_table_xarray = prediction_io.read_file(prediction_file_name)
    ptx = prediction_table_xarray

    model_file_name = ptx.attrs[prediction_io.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=True
    )

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    lead_time_hours = training_option_dict[neural_net.TARGET_LEAD_TIME_KEY]

    num_examples = len(ptx.coords[prediction_io.INIT_TIME_DIM].values)

    for i in range(num_examples):
        _plot_everything_1sample(
            prediction_table_xarray=prediction_table_xarray,
            example_index=i,
            field_names=field_names,
            lead_time_hours=lead_time_hours,
            baseline_nwp_model_name=baseline_nwp_model_name,
            baseline_nwp_model_dir_name=baseline_nwp_model_dir_name,
            min_colour_values=min_colour_values,
            max_colour_values=max_colour_values,
            min_colour_percentiles=min_colour_percentiles,
            max_colour_percentiles=max_colour_percentiles,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            plot_diffs=plot_diffs,
            min_colour_values_for_diff=min_colour_values_for_diff,
            max_colour_values_for_diff=max_colour_values_for_diff,
            min_colour_percentiles_for_diff=min_colour_percentiles_for_diff,
            max_colour_percentiles_for_diff=max_colour_percentiles_for_diff,
            output_dir_name=output_dir_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        init_time_string=getattr(INPUT_ARG_OBJECT, INIT_TIME_ARG_NAME),
        field_names=getattr(INPUT_ARG_OBJECT, FIELDS_ARG_NAME),
        min_colour_values=numpy.array(
            getattr(INPUT_ARG_OBJECT, MIN_VALUES_ARG_NAME), dtype=float
        ),
        max_colour_values=numpy.array(
            getattr(INPUT_ARG_OBJECT, MAX_VALUES_ARG_NAME), dtype=float
        ),
        min_colour_percentiles=numpy.array(
            getattr(INPUT_ARG_OBJECT, MIN_PERCENTILES_ARG_NAME), dtype=float
        ),
        max_colour_percentiles=numpy.array(
            getattr(INPUT_ARG_OBJECT, MAX_PERCENTILES_ARG_NAME), dtype=float
        ),
        plot_diffs=bool(getattr(INPUT_ARG_OBJECT, PLOT_DIFFS_ARG_NAME)),
        max_colour_values_for_diff=numpy.array(
            getattr(INPUT_ARG_OBJECT, MAX_DIFF_VALUES_ARG_NAME), dtype=float
        ),
        max_colour_percentiles_for_diff=numpy.array(
            getattr(INPUT_ARG_OBJECT, MAX_DIFF_PERCENTILES_ARG_NAME),
            dtype=float
        ),
        baseline_nwp_model_name=getattr(
            INPUT_ARG_OBJECT, BASELINE_MODEL_ARG_NAME
        ),
        baseline_nwp_model_dir_name=getattr(
            INPUT_ARG_OBJECT, BASELINE_MODEL_DIR_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
