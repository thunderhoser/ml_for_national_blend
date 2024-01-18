"""Miscellaneous helper methods."""

import os
import sys

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking


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
