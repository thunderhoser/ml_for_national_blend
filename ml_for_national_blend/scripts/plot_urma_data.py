"""Plots URMA data."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from ml_for_national_blend.outside_code import time_conversion
from ml_for_national_blend.outside_code import temperature_conversions as temperature_conv
from ml_for_national_blend.outside_code import file_system_utils
from ml_for_national_blend.outside_code import error_checking
from ml_for_national_blend.io import border_io
from ml_for_national_blend.io import urma_io
from ml_for_national_blend.utils import urma_utils
from ml_for_national_blend.plotting import target_plotting
from ml_for_national_blend.plotting import plotting_utils

TIME_FORMAT = '%Y-%m-%d-%H'

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_DIR_ARG_NAME = 'input_urma_dir_name'
VALID_TIME_ARG_NAME = 'valid_time_string'
FIELDS_ARG_NAME = 'field_names'
MIN_VALUES_ARG_NAME = 'min_colour_values'
MAX_VALUES_ARG_NAME = 'max_colour_values'
MIN_PERCENTILES_ARG_NAME = 'min_colour_percentiles'
MAX_PERCENTILES_ARG_NAME = 'max_colour_percentiles'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Path to input directory.  The relevant file therein will be found by '
    '`urma_io.find_file`.'
)
VALID_TIME_HELP_STRING = 'Valid time (format "yyyy-mm-dd-HH").'
FIELDS_HELP_STRING = (
    '1-D list of fields to plot.  Each field must be accepted by '
    '`urma_utils.check_field_name`.'
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
    'field, the minimum value in the colour scheme will be percentile {1:s}[k] '
    'of all {0:s}[k] values.'
).format(
    FIELDS_ARG_NAME, MIN_PERCENTILES_ARG_NAME
)
MAX_PERCENTILES_HELP_STRING = 'Same as {0:s} but for max values.'.format(
    MIN_PERCENTILES_ARG_NAME
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
    '--' + VALID_TIME_ARG_NAME, type=str, required=True,
    help=VALID_TIME_HELP_STRING
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


def _run(urma_directory_name, valid_time_string, field_names,
         min_colour_values, max_colour_values,
         min_colour_percentiles, max_colour_percentiles, output_dir_name):
    """Plots URMA data.

    This is effectively the main method.

    :param urma_directory_name: See documentation at top of this script.
    :param valid_time_string: Same.
    :param field_names: Same.
    :param min_colour_values: Same.
    :param max_colour_values: Same.
    :param min_colour_percentiles: Same.
    :param max_colour_percentiles: Same.
    :param output_dir_name: Same.
    """

    # Check input args.
    valid_time_unix_sec = time_conversion.string_to_unix_sec(
        valid_time_string, TIME_FORMAT
    )
    for f in field_names:
        urma_utils.check_field_name(f)

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

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Do actual stuff.
    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    urma_file_name = urma_io.find_file(
        directory_name=urma_directory_name,
        valid_date_string=time_conversion.unix_sec_to_string(
            valid_time_unix_sec, urma_io.DATE_FORMAT
        ),
        raise_error_if_missing=True
    )

    print('Reading data from: "{0:s}"...'.format(urma_file_name))
    urma_table_xarray = urma_io.read_file(urma_file_name)
    urma_table_xarray = urma_utils.subset_by_time(
        urma_table_xarray=urma_table_xarray,
        desired_times_unix_sec=numpy.array([valid_time_unix_sec], dtype=int)
    )

    for j in range(len(field_names)):
        this_field_name_fancy = (
            target_plotting.FIELD_NAME_TO_FANCY[field_names[j]]
        )

        j_new = numpy.where(
            urma_table_xarray.coords[urma_utils.FIELD_DIM].values ==
            field_names[j]
        )[0][0]

        data_matrix = (
            urma_table_xarray[urma_utils.DATA_KEY].values[0, ..., j_new]
        )

        if field_names[j] in [
                urma_utils.TEMPERATURE_2METRE_NAME,
                urma_utils.DEWPOINT_2METRE_NAME
        ]:
            data_matrix = temperature_conv.kelvins_to_celsius(data_matrix)

        title_string = 'URMA {0:s}\nValid {1:s}'.format(
            this_field_name_fancy, valid_time_string
        )

        output_file_name = (
            '{0:s}/valid={1:s}_{2:s}_urma.jpg'
        ).format(
            output_dir_name,
            valid_time_string,
            field_names[j].replace('_', '-')
        )

        if min_colour_values is None:
            (
                colour_map_object, colour_norm_object
            ) = target_plotting.field_to_colour_scheme(
                field_name=field_names[j],
                min_value=
                numpy.nanpercentile(data_matrix, min_colour_percentiles[j]),
                max_value=
                numpy.nanpercentile(data_matrix, max_colour_percentiles[j]),
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
            data_matrix=data_matrix,
            latitude_matrix_deg_n=
            urma_table_xarray[urma_utils.LATITUDE_KEY].values,
            longitude_matrix_deg_e=
            urma_table_xarray[urma_utils.LONGITUDE_KEY].values,
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
        urma_directory_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        valid_time_string=getattr(INPUT_ARG_OBJECT, VALID_TIME_ARG_NAME),
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
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
