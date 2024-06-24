"""Miscellaneous helper methods."""

import os
import numpy
from scipy.ndimage import distance_transform_edt
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml_for_national_blend.utils import nbm_utils

DEGREES_TO_RADIANS = numpy.pi / 180
WIND_DIR_DEFAULT_DEG = 0.

USED_DATE_STRINGS = time_conversion.get_spc_dates_in_range(
    '20170101', '20200225'
)[::5]
USED_DATE_STRINGS += time_conversion.get_spc_dates_in_range(
    '20200302', '20221028'
)[::5]
USED_DATE_STRINGS += time_conversion.get_spc_dates_in_range(
    '20221101', '20231027'
)[::5]
USED_DATE_STRINGS += time_conversion.get_spc_dates_in_range(
    '20231102', '20231227'
)[::5]

ROW_LIMITS_2PT5KM_KEY = 'row_limits_2pt5km'
COLUMN_LIMITS_2PT5KM_KEY = 'column_limits_2pt5km'
ROW_LIMITS_10KM_KEY = 'row_limits_10km'
COLUMN_LIMITS_10KM_KEY = 'column_limits_10km'
ROW_LIMITS_20KM_KEY = 'row_limits_20km'
COLUMN_LIMITS_20KM_KEY = 'column_limits_20km'
ROW_LIMITS_40KM_KEY = 'row_limits_40km'
COLUMN_LIMITS_40KM_KEY = 'column_limits_40km'


def untar_file(tar_file_name, target_dir_name, relative_paths_to_untar=None):
    """Untars a file.

    :param tar_file_name: Path to tar file.
    :param target_dir_name: Path to output directory.
    :param relative_paths_to_untar: List of relative paths to extract from the
        tar file.  If you want to untar the whole thing, leave this argument
        alone.
    :raises: ValueError: if the Unix command fails.
    """

    error_checking.assert_file_exists(tar_file_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=target_dir_name
    )

    if relative_paths_to_untar is not None:
        error_checking.assert_is_string_list(relative_paths_to_untar)

    command_string = 'tar -C "{0:s}" -xvf "{1:s}"'.format(
        target_dir_name, tar_file_name
    )
    for this_path in relative_paths_to_untar:
        command_string += ' "' + this_path + '"'

    exit_code = os.system(command_string)
    if exit_code != 0:
        raise ValueError(
            '\nUnix command failed (log messages shown above should explain '
            'why).'
        )


def untar_zarr_or_netcdf_file(tar_file_name, target_dir_name):
    """Untars zarr or NetCDF file, getting rid of subdirectories in tar.

    :param tar_file_name: Path to tar file.
    :param target_dir_name: Path to output directory.
    :raises: ValueError: if the Unix command fails.
    """

    error_checking.assert_file_exists(tar_file_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=target_dir_name
    )

    command_string = 'tar -tvf "{0:s}"'.format(tar_file_name)
    tar_contents_string = os.popen(command_string).read()
    last_line = tar_contents_string.splitlines()[-1]
    last_path_in_tar_file = last_line.split()[-1]

    zarr_subdir_flags = numpy.array([
        s.endswith('.zarr') or s.endswith('.nc') or s.endswith('.netcdf')
        for s in last_path_in_tar_file.split('/')
    ], dtype=bool)

    zarr_subdir_index = numpy.where(zarr_subdir_flags)[0][0]

    command_string = 'tar -C "{0:s}" -xvf "{1:s}"'.format(
        target_dir_name, tar_file_name
    )

    if zarr_subdir_index > 0:
        command_string += ' --strip={0:d}'.format(zarr_subdir_index)

    exit_code = os.system(command_string)
    if exit_code != 0:
        raise ValueError(
            '\nUnix command failed (log messages shown above should explain '
            'why).'
        )


def create_tar_file(source_paths_to_tar, tar_file_name):
    """Creates a tar file.

    :param source_paths_to_tar: List of paths to files/directories that you want
        to tar.
    :param tar_file_name: Path to output file.
    :raises: ValueError: if the Unix command fails.
    """

    for this_path in source_paths_to_tar:
        try:
            error_checking.assert_file_exists(this_path)
        except:
            error_checking.assert_directory_exists(this_path)

    file_system_utils.mkdir_recursive_if_necessary(file_name=tar_file_name)

    command_string = 'tar -czvf "{0:s}"'.format(tar_file_name)
    for this_path in source_paths_to_tar:
        command_string += ' "' + this_path + '"'

    exit_code = os.system(command_string)
    if exit_code != 0:
        raise ValueError(
            '\nUnix command failed (log messages shown above should explain '
            'why).'
        )


def speed_and_direction_to_uv(wind_speeds_m_s01, wind_directions_deg):
    """Converts wind vectors from speed and direction to u- and v-components.

    :param wind_speeds_m_s01: numpy array of wind speeds (metres per second).
    :param wind_directions_deg: Equivalent-shape numpy array of wind
        directions (direction of origin, as per meteorological convention).
    :return: u_winds_m_s01: Equivalent-shape numpy array of u-components (metres
        per second).
    :return: v_winds_m_s01: Equivalent-shape numpy array of v-components.
    """

    error_checking.assert_is_geq_numpy_array(
        wind_speeds_m_s01, 0., allow_nan=True
    )
    error_checking.assert_is_numpy_array(
        wind_directions_deg,
        exact_dimensions=numpy.array(wind_speeds_m_s01.shape, dtype=int)
    )

    these_wind_directions_deg = wind_directions_deg + 0.
    these_wind_directions_deg[
        numpy.isnan(these_wind_directions_deg)
    ] = WIND_DIR_DEFAULT_DEG

    u_winds_m_s01 = -1 * wind_speeds_m_s01 * numpy.sin(
        these_wind_directions_deg * DEGREES_TO_RADIANS
    )
    v_winds_m_s01 = -1 * wind_speeds_m_s01 * numpy.cos(
        these_wind_directions_deg * DEGREES_TO_RADIANS
    )

    return u_winds_m_s01, v_winds_m_s01


def remove_unused_days(candidate_times_unix_sec):
    """Removes times that occur during an unused day.

    An 'unused day' is just a day for which we did not bother downloading all
    the NWP outputs from HPSS.

    :param candidate_times_unix_sec: 1-D numpy array of candidate times.
    :return: relevant_times_unix_sec: 1-D numpy array of relevant times,
        excluding unused days.
    """

    error_checking.assert_is_numpy_array(
        candidate_times_unix_sec, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(candidate_times_unix_sec)

    candidate_date_strings = [
        time_conversion.unix_sec_to_string(t, time_conversion.SPC_DATE_FORMAT)
        for t in candidate_times_unix_sec
    ]
    good_indices = numpy.where(numpy.isin(
        element=numpy.array(candidate_date_strings),
        test_elements=numpy.array(USED_DATE_STRINGS)
    ))[0]

    return candidate_times_unix_sec[good_indices]


def determine_patch_locations(patch_size_2pt5km_pixels, start_row_2pt5km=None,
                              start_column_2pt5km=None):
    """Determines patch location within NBM grid at every resolution.

    :param patch_size_2pt5km_pixels: Patch size, in number of 2.5-km pixels per
        side.  For example, if patch_size_2pt5km_pixels = 448, then the patch
        size at the finest resolution (2.5 km) is 448 x 448 pixels.  This must
        be a multiple of 16.
    :param start_row_2pt5km: Index of first row at finest resolution (2.5 km).
        If you make this argument None, the start row will be determined
        randomly.
    :param start_column_2pt5km: Same as start_row_2pt5km but for column.
    :return: location_dict: Dictionary with the following keys.
    location_dict["row_limits_2pt5km"]: length-2 numpy array with indices of
        first and last row at finest resolution (2.5 km).
    location_dict["column_limits_2pt5km"]: Same as key "row_limits_2pt5km" but
        for column.
    location_dict["row_limits_10km"]: length-2 numpy array with indices of
        first and last row at second-finest resolution (10 km).
    location_dict["column_limits_10km"]: Same as key "row_limits_10km" but
        for column.
    location_dict["row_limits_20km"]: length-2 numpy array with indices of
        first and last row at third-finest resolution (20 km).
    location_dict["column_limits_20km"]: Same as key "row_limits_20km" but
        for column.
    location_dict["row_limits_40km"]: length-2 numpy array with indices of
        first and last row at coarsest resolution (40 km).
    location_dict["column_limits_40km"]: Same as key "row_limits_40km" but
        for column.
    """

    # TODO(thunderhoser): I will eventually need two different patch sizes,
    # representing an outer domain (for predictors) and an inner domain (for
    # targets).  However, I will do this by including a binary mask in the
    # target_matrix and using the binary mask in the loss function.  This binary
    # mask will prevent the NN from being penalized for any predictions it
    # makes in the outer domain, so that only the inner domain matters.  This
    # will avoid edge effects, allowing the NN to not worry about predictions
    # near the edge of the patch.

    # Check input args.
    error_checking.assert_is_integer(patch_size_2pt5km_pixels)
    error_checking.assert_is_greater(patch_size_2pt5km_pixels, 0)
    error_checking.assert_equals(
        numpy.mod(patch_size_2pt5km_pixels, 16),
        0
    )

    num_rows_in_full_grid = len(nbm_utils.NBM_Y_COORDS_METRES)
    num_columns_in_full_grid = len(nbm_utils.NBM_X_COORDS_METRES)
    min_dimension_of_full_grid = min([
        num_rows_in_full_grid, num_columns_in_full_grid
    ])
    error_checking.assert_is_less_than(
        patch_size_2pt5km_pixels, min_dimension_of_full_grid
    )

    max_possible_start_row = number_rounding.floor_to_nearest(
        num_rows_in_full_grid - patch_size_2pt5km_pixels + 1,
        16
    )
    max_possible_start_row = numpy.round(max_possible_start_row).astype(int)

    max_possible_start_column = number_rounding.floor_to_nearest(
        num_columns_in_full_grid - patch_size_2pt5km_pixels + 1,
        16
    )
    max_possible_start_column = numpy.round(
        max_possible_start_column
    ).astype(int)

    if start_row_2pt5km is None or start_column_2pt5km is None:
        possible_start_rows = numpy.linspace(
            0, max_possible_start_row,
            num=int(numpy.round(float(max_possible_start_row) / 16)) + 1,
            dtype=int
        )
        start_row_2pt5km = numpy.random.choice(possible_start_rows, size=1)[0]

        possible_start_columns = numpy.linspace(
            0, max_possible_start_column,
            num=int(numpy.round(float(max_possible_start_column) / 16)) + 1,
            dtype=int
        )
        start_column_2pt5km = numpy.random.choice(
            possible_start_columns, size=1
        )[0]

    error_checking.assert_is_integer(start_row_2pt5km)
    error_checking.assert_is_geq(start_row_2pt5km, 0)
    error_checking.assert_is_leq(start_row_2pt5km, max_possible_start_row)
    error_checking.assert_equals(
        numpy.mod(start_row_2pt5km, 16),
        0
    )

    error_checking.assert_is_integer(start_column_2pt5km)
    error_checking.assert_is_geq(start_column_2pt5km, 0)
    error_checking.assert_is_leq(start_column_2pt5km, max_possible_start_column)
    error_checking.assert_equals(
        numpy.mod(start_column_2pt5km, 16),
        0
    )

    row_limits_2pt5km = start_row_2pt5km + numpy.array(
        [0, patch_size_2pt5km_pixels], dtype=int
    )
    column_limits_2pt5km = start_column_2pt5km + numpy.array(
        [0, patch_size_2pt5km_pixels], dtype=int
    )

    row_limits_10km = numpy.round(
        row_limits_2pt5km.astype(float) / 4
    ).astype(int)
    column_limits_10km = numpy.round(
        column_limits_2pt5km.astype(float) / 4
    ).astype(int)

    row_limits_20km = numpy.round(
        row_limits_2pt5km.astype(float) / 8
    ).astype(int)
    column_limits_20km = numpy.round(
        column_limits_2pt5km.astype(float) / 8
    ).astype(int)

    row_limits_40km = numpy.round(
        row_limits_2pt5km.astype(float) / 16
    ).astype(int)
    column_limits_40km = numpy.round(
        column_limits_2pt5km.astype(float) / 16
    ).astype(int)

    row_limits_2pt5km[1] -= 1
    column_limits_2pt5km[1] -= 1
    row_limits_10km[1] -= 1
    column_limits_10km[1] -= 1
    row_limits_20km[1] -= 1
    column_limits_20km[1] -= 1
    row_limits_40km[1] -= 1
    column_limits_40km[1] -= 1

    # Do actual stuff.
    return {
        ROW_LIMITS_2PT5KM_KEY: row_limits_2pt5km,
        COLUMN_LIMITS_2PT5KM_KEY: column_limits_2pt5km,
        ROW_LIMITS_10KM_KEY: row_limits_10km,
        COLUMN_LIMITS_10KM_KEY: column_limits_10km,
        ROW_LIMITS_20KM_KEY: row_limits_20km,
        COLUMN_LIMITS_20KM_KEY: column_limits_20km,
        ROW_LIMITS_40KM_KEY: row_limits_40km,
        COLUMN_LIMITS_40KM_KEY: column_limits_40km
    }


def fill_nans_by_nn_interp(data_matrix):
    """Fills NaN's with nearest neighbours.

    This method is adapted from the method `fill`, which you can find here:
    https://stackoverflow.com/posts/9262129/revisions

    :param data_matrix: numpy array of real-valued data.
    :return: data_matrix: Same but without NaN's.
    """

    error_checking.assert_is_real_numpy_array(data_matrix)

    indices = distance_transform_edt(
        numpy.isnan(data_matrix), return_distances=False, return_indices=True
    )
    return data_matrix[tuple(indices)]
