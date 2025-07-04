"""IO methods for grib and grib2 files.

Taken from the GewitterGefahr library:
https://github.com/thunderhoser/gewittergefahr

These methods use wgrib and wgrib2, which are command-line tools.  See
README_grib (in the same directory as this module) for installation
instructions.
"""

import os
import subprocess
import tempfile
import warnings
import numpy
from ml_for_national_blend.outside_code import file_system_utils
from ml_for_national_blend.outside_code import error_checking

TOLERANCE_FOR_SENTINEL_VALUES = 10.

GRIB1_FILE_EXTENSION = '.grb'
GRIB1_FILE_TYPE = 'grib1'
WGRIB_EXE_NAME_DEFAULT = '/usr/bin/wgrib'

GRIB2_FILE_EXTENSION = '.grb2'
GRIB2_FILE_TYPE = 'grib2'
WGRIB2_EXE_NAME_DEFAULT = '/usr/bin/wgrib2'
VALID_FILE_TYPES = [GRIB1_FILE_TYPE, GRIB2_FILE_TYPE]

U_WIND_PREFIX = 'UGRD'
V_WIND_PREFIX = 'VGRD'


def _field_name_grib1_to_grib2(field_name_grib1):
    """Converts field name from grib1 to grib2.

    :param field_name_grib1: Field name in grib1 format.
    :return: field_name_grib2: Field name in grib2 format.
    """

    return field_name_grib1.replace('gnd', 'ground').replace('sfc', 'surface')


def _sentinel_value_to_nan(data_matrix, sentinel_value=None):
    """Replaces all instances of sentinel value with NaN.

    :param data_matrix: numpy array (may contain sentinel values).
    :param sentinel_value: Sentinel value (may be None).
    :return: data_matrix: numpy array without sentinel values.
    """

    if sentinel_value is None:
        return data_matrix

    data_vector = numpy.reshape(data_matrix, data_matrix.size)
    sentinel_flags = numpy.isclose(
        data_vector, sentinel_value, atol=TOLERANCE_FOR_SENTINEL_VALUES)

    sentinel_indices = numpy.where(sentinel_flags)[0]
    data_vector[sentinel_indices] = numpy.nan
    return numpy.reshape(data_vector, data_matrix.shape)


def check_file_type(grib_file_type):
    """Ensures that grib file type is valid.

    :param grib_file_type: Either "grib1" or "grib2".
    :raises: ValueError: if `grib_file_type not in VALID_FILE_TYPES`.
    """

    error_checking.assert_is_string(grib_file_type)
    if grib_file_type not in VALID_FILE_TYPES:
        error_string = (
            '\n\n{0:s}\nValid file types (listed above) do not include "{1:s}".'
        ).format(str(VALID_FILE_TYPES), grib_file_type)

        raise ValueError(error_string)


def file_name_to_type(grib_file_name):
    """Determines file type (either grib1 or grib2) from file name.

    :param grib_file_name: Path to input file.
    :return: grib_file_type: Either "grib1" or "grib2".
    :raises: ValueError: if file type is neither grib1 nor grib2.
    """

    error_checking.assert_is_string(grib_file_name)
    if grib_file_name.endswith(GRIB1_FILE_EXTENSION):
        return GRIB1_FILE_TYPE
    if grib_file_name.endswith(GRIB2_FILE_EXTENSION):
        return GRIB2_FILE_TYPE
    if grib_file_name.endswith('.grib2'):
        return GRIB2_FILE_TYPE

    # TODO(thunderhoser): Hack for GFS data on HPSS.
    file_extension_string = os.path.splitext(grib_file_name)[1]
    if file_extension_string == '':
        return GRIB2_FILE_TYPE

    # TODO(thunderhoser): Hack for URMA data on HPSS.
    if file_extension_string == '.grb2_wexp':
        return GRIB2_FILE_TYPE

    # TODO(thunderhoser): Hack for NBM constants on Hera.
    if file_extension_string == '.gb2':
        return GRIB2_FILE_TYPE

    error_string = (
        'File type should be either "{0:s}" or "{1:s}".  Instead, got: "{2:s}"'
    ).format(GRIB1_FILE_TYPE, GRIB2_FILE_TYPE, grib_file_name)

    raise ValueError(error_string)


def rotate_winds_in_grib_file(
        input_grib_file_name, output_grib_file_name,
        grid_definition_file_name,
        wgrib2_exe_name=WGRIB2_EXE_NAME_DEFAULT, raise_error_if_fails=True):
    """Rotates winds in grib file from grid-relative to Earth-relative.

    :param input_grib_file_name: Path to input file.
    :param output_grib_file_name: Path to output file.
    :param grid_definition_file_name: Path to grid-definition file on local
        machine.  You can download this Perl script from:
        https://ftp.cpc.ncep.noaa.gov/wd51we/wgrib2.scripts/grid_defn.pl
    :param wgrib2_exe_name: Path to wgrib2 executable.
    :param raise_error_if_fails: Boolean flag.  If the rotation fails and
        raise_error_if_fails = True, this method will error out.  If the
        rotation fails and raise_error_if_fails = False, this method will
        return False.
    :return: success: Boolean flag.
    """

    # Error-checking.
    error_checking.assert_file_exists(input_grib_file_name)
    file_system_utils.mkdir_recursive_if_necessary(
        file_name=output_grib_file_name
    )
    error_checking.assert_file_exists(grid_definition_file_name)
    # error_checking.assert_file_exists(wgrib2_exe_name)
    error_checking.assert_is_boolean(raise_error_if_fails)

    assert file_name_to_type(input_grib_file_name) == GRIB2_FILE_TYPE

    # Do actual stuff.
    command_string = (
        '"{0:s}" "{1:s}" -set_grib_type same -new_grid_winds earth '
        '-new_grid_interpolation neighbor -new_grid `"{2:s}" "{1:s}"` "{3:s}"'
    ).format(
        wgrib2_exe_name,
        input_grib_file_name,
        grid_definition_file_name,
        output_grib_file_name
    )

    print(command_string)

    if os.path.isfile(output_grib_file_name):
        os.remove(output_grib_file_name)

    try:
        subprocess.call(command_string, shell=True)
        error_checking.assert_file_exists(output_grib_file_name)
    except OSError as this_exception:
        if raise_error_if_fails:
            raise

        warning_string = (
            '\n\n{0:s}\n\nCommand (shown above) failed (details shown below).'
            '\n\n{1:s}'
        ).format(command_string, str(this_exception))

        warnings.warn(warning_string)
        return False

    return True


def read_field_from_grib_file(
        grib_file_name, field_name_grib1, num_grid_rows, num_grid_columns,
        sentinel_value=None, temporary_dir_name=None,
        wgrib_exe_name=WGRIB_EXE_NAME_DEFAULT,
        wgrib2_exe_name=WGRIB2_EXE_NAME_DEFAULT,
        raise_error_if_fails=True,
        grib_inventory_file_name=None):
    """Reads field from grib file.

    One field = one variable at one time step.

    M = number of rows (unique y-coordinates or latitudes of grid points)
    N = number of columns (unique x-coordinates or longitudes of grid points)

    :param grib_file_name: Path to input file.
    :param field_name_grib1: Field name in grib1 format (example: 500-mb height
        is "HGT:500 mb").
    :param num_grid_rows: Number of rows expected in grid.
    :param num_grid_columns: Number of columns expected in grid.
    :param sentinel_value: Sentinel value (all instances will be replaced with
        NaN).
    :param temporary_dir_name: Name of temporary directory.  An intermediate
        text file will be stored here.
    :param wgrib_exe_name: Path to wgrib executable.
    :param wgrib2_exe_name: Path to wgrib2 executable.
    :param raise_error_if_fails: Boolean flag.  If the extraction fails and
        raise_error_if_fails = True, this method will error out.  If the
        extraction fails and raise_error_if_fails = False, this method will
        return None.
    :param grib_inventory_file_name: Path to inventory file, generated by
        either `wgrib ${grib_file_name} > ${grib_inventory_file_name}` or
        `wgrib2 ${grib_file_name} > ${grib_inventory_file_name}`.  If this does
        not exist, it will be created.
    :return: field_matrix: M-by-N numpy array with values of the given field.
        If the grid is regular in x-y coordinates, x increases towards the right
        (in the positive direction of the second axis), while y increases
        downward (in the positive direction of the first axis).  If the grid is
        regular in lat-long, replace "x" and "y" in the previous sentence with
        "long" and "lat," respectively.
    :return: grib_inventory_file_name: Path to inventory file.
    :raises: ValueError: if extraction fails and raise_error_if_fails = True.
    """

    # Error-checking.
    error_checking.assert_is_string(field_name_grib1)
    error_checking.assert_is_integer(num_grid_rows)
    error_checking.assert_is_greater(num_grid_rows, 0)
    error_checking.assert_is_integer(num_grid_columns)
    error_checking.assert_is_greater(num_grid_columns, 0)
    # error_checking.assert_file_exists(wgrib_exe_name)
    # error_checking.assert_file_exists(wgrib2_exe_name)
    error_checking.assert_is_boolean(raise_error_if_fails)
    if sentinel_value is not None:
        error_checking.assert_is_not_nan(sentinel_value)

    if temporary_dir_name is not None:
        file_system_utils.mkdir_recursive_if_necessary(
            directory_name=temporary_dir_name
        )

    if grib_inventory_file_name is None:
        error_checking.assert_is_string(grib_inventory_file_name)
    else:
        grib_inventory_file_name = tempfile.NamedTemporaryFile(
            dir=temporary_dir_name, delete=False
        ).name

    # Housekeeping.
    grib_file_type = file_name_to_type(grib_file_name)
    if wgrib2_exe_name is None:
        grib_file_type = GRIB1_FILE_TYPE

    temporary_file_name = tempfile.NamedTemporaryFile(
        dir=temporary_dir_name, delete=False
    ).name

    # Create inventory file, if necessary.
    if grib_inventory_file_name is None:
        if grib_file_type == GRIB1_FILE_TYPE:
            command_string = '"{0:s}" "{1:s}" > "{2:s}"'.format(
                wgrib_exe_name, grib_file_name, grib_inventory_file_name
            )
        else:
            command_string = '"{0:s}" "{1:s}" > "{2:s}"'.format(
                wgrib2_exe_name, grib_file_name, grib_inventory_file_name
            )

        print(command_string)

        try:
            subprocess.call(command_string, shell=True)
        except OSError as this_exception:
            os.remove(grib_inventory_file_name)
            if raise_error_if_fails:
                raise

            warning_string = (
                '\n\n'
                '{0:s}\n\n'
                'Command (shown above) failed (details shown below).\n\n'
                '{1:s}'
            ).format(command_string, str(this_exception))

            warnings.warn(warning_string)
            return None, grib_inventory_file_name

    # Extract field to temporary file.
    # if grib_file_type == GRIB1_FILE_TYPE:
    #     command_string = (
    #         '"{0:s}" "{1:s}" -s | grep -w "{2:s}" | grep -v "ens std dev" | "{0:s}" -i "{1:s}" '
    #         '-text -nh -o "{3:s}"'
    #     ).format(
    #         wgrib_exe_name, grib_file_name, field_name_grib1,
    #         temporary_file_name
    #     )
    # else:
    #     command_string = (
    #         '"{0:s}" "{1:s}" -s | grep -w "{2:s}" | grep -v "ens std dev" | "{0:s}" -i "{1:s}" '
    #         '-no_header -text "{3:s}"'
    #     ).format(
    #         wgrib2_exe_name, grib_file_name,
    #         _field_name_grib1_to_grib2(field_name_grib1), temporary_file_name
    #     )

    if grib_file_type == GRIB1_FILE_TYPE:
        command_string = (
            'grep -w "{0:s}" "{1:s}" | grep -v "ens std dev" | '
            '"{2:s}" -i "{3:s}" -text -nh -o "{4:s}"'
        ).format(
            field_name_grib1,
            grib_inventory_file_name,
            wgrib_exe_name,
            grib_file_name,
            temporary_file_name
        )
    else:
        command_string = (
            'grep -w "{0:s}" "{1:s}" | grep -v "ens std dev" | '
            '"{2:s}" -i "{3:s}" -no_header -text "{4:s}"'
        ).format(
            _field_name_grib1_to_grib2(field_name_grib1),
            grib_inventory_file_name,
            wgrib2_exe_name,
            grib_file_name,
            temporary_file_name
        )

    print(command_string)

    try:
        subprocess.call(command_string, shell=True)
    except OSError as this_exception:
        os.remove(temporary_file_name)
        if raise_error_if_fails:
            raise

        warning_string = (
            '\n\n{0:s}\n\nCommand (shown above) failed (details shown below).'
            '\n\n{1:s}'
        ).format(command_string, str(this_exception))

        warnings.warn(warning_string)
        return None, grib_inventory_file_name

    # Read field from temporary file.
    field_vector = numpy.loadtxt(temporary_file_name)
    os.remove(temporary_file_name)

    if len(field_vector) == num_grid_columns * (num_grid_rows - 200):
        field_matrix = numpy.reshape(
            field_vector, (num_grid_columns, num_grid_rows - 200)
        )
        field_matrix = numpy.transpose(field_matrix)
        field_matrix = numpy.pad(
            field_matrix,
            pad_width=((200, 0), (0, 0)),
            mode='constant',
            constant_values=numpy.nan
        )
        field_matrix = _sentinel_value_to_nan(
            data_matrix=field_matrix, sentinel_value=sentinel_value
        )

        return field_matrix, grib_inventory_file_name

    if len(field_vector) == num_grid_columns * num_grid_rows:
        field_matrix = numpy.reshape(
            field_vector, (num_grid_rows, num_grid_columns)
        )
        field_matrix = _sentinel_value_to_nan(
            data_matrix=field_matrix, sentinel_value=sentinel_value
        )
        return field_matrix, grib_inventory_file_name

    try:
        num_values = len(field_vector)
        half_num_values = int(numpy.round(0.5 * num_values))

        if 2 * half_num_values != num_values:
            error_string = (
                'SOMETHING WENT VERY WRONG.  Expected {0:d} values '
                '({1:d} rows x {2:d} columns) in temporary text file from '
                'wgrib, but the file contains {3:d} values instead.'
            ).format(
                num_grid_rows * num_grid_columns,
                num_grid_rows,
                num_grid_columns,
                num_values
            )

            raise ValueError(error_string)

        first_field_matrix = numpy.reshape(
            field_vector[:half_num_values], (num_grid_rows, num_grid_columns)
        )
        second_field_matrix = numpy.reshape(
            field_vector[half_num_values:], (num_grid_rows, num_grid_columns)
        )
        max_diff = numpy.nanmax(
            numpy.absolute(first_field_matrix - second_field_matrix)
        )

        if max_diff > 1e-6:
            error_string = (
                'SOMETHING WENT VERY WRONG.  First and second data matrices in '
                'temporary text file from wgrib should be equal, but found a '
                'max absolute diff of {0:f}.'
            ).format(max_diff)

            raise ValueError(error_string)

        field_matrix = first_field_matrix
    except ValueError as this_exception:
        if raise_error_if_fails:
            raise

        warning_string = (
            '\n\nnumpy.reshape failed (details shown below).\n\n{0:s}'
        ).format(str(this_exception))

        warnings.warn(warning_string)
        return None, grib_inventory_file_name

    field_matrix = _sentinel_value_to_nan(
        data_matrix=field_matrix, sentinel_value=sentinel_value
    )
    return field_matrix, grib_inventory_file_name


def is_u_wind_field(field_name_grib1):
    """Determines whether or not field is a u-wind field.

    :param field_name_grib1: Field name in grib1 format.
    :return: is_u_wind_flag: Boolean flag.
    """

    error_checking.assert_is_string(field_name_grib1)
    return field_name_grib1.startswith(U_WIND_PREFIX)


def is_v_wind_field(field_name_grib1):
    """Determines whether or not field is a v-wind field.

    :param field_name_grib1: Field name in grib1 format.
    :return: is_v_wind_flag: Boolean flag.
    """

    error_checking.assert_is_string(field_name_grib1)
    return field_name_grib1.startswith(V_WIND_PREFIX)


def is_wind_field(field_name_grib1):
    """Determines whether or not field is a wind field.

    :param field_name_grib1: Field name in grib1 format.
    :return: is_wind_flag: Boolean flag.
    """

    return is_u_wind_field(field_name_grib1) or is_v_wind_field(
        field_name_grib1)


def switch_uv_in_field_name(field_name_grib1):
    """Switches u-wind and v-wind in field name.

    In other words, if the original field is u-wind, this method converts it to
    the equivalent v-wind field.  If the original field is v-wind, this method
    converts to the equivalent u-wind field.

    :param field_name_grib1: Field name in grib1 format.
    :return: switched_field_name_grib1: See above discussion.
    """

    if not is_wind_field(field_name_grib1):
        return field_name_grib1
    if is_u_wind_field(field_name_grib1):
        return field_name_grib1.replace(U_WIND_PREFIX, V_WIND_PREFIX)
    return field_name_grib1.replace(V_WIND_PREFIX, U_WIND_PREFIX)


def file_type_to_extension(grib_file_type):
    """Converts grib file type to file extension.

    :param grib_file_type: File type (either "grib1" or "grib2").
    :return: grib_file_extension: Expected file extension for the given type.
    """

    check_file_type(grib_file_type)
    if grib_file_type == GRIB1_FILE_TYPE:
        return GRIB1_FILE_EXTENSION
    if grib_file_type == GRIB2_FILE_TYPE:
        return GRIB2_FILE_EXTENSION

    return None
