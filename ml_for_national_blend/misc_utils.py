"""Miscellaneous helper methods."""

import os
import sys
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking

DEGREES_TO_RADIANS = numpy.pi / 180
WIND_DIR_DEFAULT_DEG = 0.


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
