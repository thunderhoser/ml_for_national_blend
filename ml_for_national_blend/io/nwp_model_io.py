"""Input/output methods for processed NWP-model data."""

import os
import re
import copy
import shutil
import xarray
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml_for_national_blend.utils import misc_utils
from ml_for_national_blend.utils import nwp_model_utils

TIME_FORMAT = '%Y-%m-%d-%H'


def find_file(directory_name, init_time_unix_sec, model_name, allow_tar=False,
              raise_error_if_missing=True):
    """Finds zarr file with NWP output (forecasts) for 1 model run (init time).

    :param directory_name: Path to input directory.
    :param init_time_unix_sec: Initialization time.
    :param model_name: Name of NWP model (must be accepted by
        `nwp_model_utils.check_model_name`).
    :param allow_tar: Boolean flag.  If True, will allow tar file (but will
        look for untarred file first).  If False, will look only for untarred
        file.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: nwp_forecast_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    nwp_model_utils.check_init_time(
        init_time_unix_sec=init_time_unix_sec, model_name=model_name
    )
    error_checking.assert_is_boolean(allow_tar)
    error_checking.assert_is_boolean(raise_error_if_missing)

    nwp_forecast_file_name = '{0:s}/{1:s}_{2:s}.zarr'.format(
        directory_name,
        model_name,
        time_conversion.unix_sec_to_string(init_time_unix_sec, TIME_FORMAT)
    )

    if os.path.isdir(nwp_forecast_file_name) or not raise_error_if_missing:
        return nwp_forecast_file_name

    if allow_tar:
        nwp_forecast_file_name = re.sub(
            '.zarr$', '.tar', nwp_forecast_file_name
        )

    if os.path.isfile(nwp_forecast_file_name):
        return nwp_forecast_file_name

    nwp_forecast_file_name = re.sub(
        '.tar$', '.zarr', nwp_forecast_file_name
    )
    error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
        nwp_forecast_file_name
    )
    raise ValueError(error_string)


def find_files_for_period(
        directory_name, model_name,
        first_init_time_unix_sec, last_init_time_unix_sec, allow_tar=False,
        raise_error_if_any_missing=False, raise_error_if_all_missing=True):
    """Finds many zarr files, each with NWP output for 1 model run (init time).

    :param directory_name: Path to input directory.
    :param model_name: Name of NWP model (must be accepted by
        `nwp_model_utils.check_model_name`).
    :param first_init_time_unix_sec: First initialization time in period.
    :param last_init_time_unix_sec: Last initialization time in period.
    :param allow_tar: Boolean flag.  If True, will allow tar file (but will
        look for untarred file first).  If False, will look only for untarred
        file.
    :param raise_error_if_any_missing: Boolean flag.  If any file is missing and
        `raise_error_if_any_missing == True`, will throw error.
    :param raise_error_if_all_missing: Boolean flag.  If all files are missing
        and `raise_error_if_all_missing == True`, will throw error.
    :return: zarr_file_names: 1-D list of paths to zarr files with NWP
        forecasts, one per model run.
    :raises: ValueError: if all files are missing and
        `raise_error_if_all_missing == True`.
    """

    error_checking.assert_is_boolean(allow_tar)
    error_checking.assert_is_boolean(raise_error_if_any_missing)
    error_checking.assert_is_boolean(raise_error_if_all_missing)

    init_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_init_time_unix_sec,
        end_time_unix_sec=last_init_time_unix_sec,
        time_interval_sec=
        nwp_model_utils.model_to_init_time_interval(model_name),
        include_endpoint=True
    )

    zarr_file_names = []

    for this_init_time_unix_sec in init_times_unix_sec:
        try:
            this_file_name = find_file(
                directory_name=directory_name,
                model_name=model_name,
                init_time_unix_sec=this_init_time_unix_sec,
                allow_tar=allow_tar,
                raise_error_if_missing=True
            )
        except ValueError as this_error:
            if raise_error_if_any_missing:
                raise this_error

        if os.path.isdir(this_file_name) or os.path.isfile(this_file_name):
            zarr_file_names.append(this_file_name)

    if raise_error_if_all_missing and len(zarr_file_names) == 0:
        error_string = (
            'Cannot find any file in directory "{0:s}" from init times {1:s} '
            'to {2:s}.'
        ).format(
            directory_name,
            time_conversion.unix_sec_to_string(
                init_times_unix_sec[0], TIME_FORMAT
            ),
            time_conversion.unix_sec_to_string(
                init_times_unix_sec[-1], TIME_FORMAT
            )
        )
        raise ValueError(error_string)

    return zarr_file_names


def file_name_to_init_time(nwp_forecast_file_name):
    """Parses initialization time from name of file with NWP forecasts.

    :param nwp_forecast_file_name: File path.
    :return: init_time_unix_sec: Initialization time.
    """

    pathless_file_name = os.path.split(nwp_forecast_file_name)[1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]
    init_time_string = extensionless_file_name.split('_')[-1]
    model_name = '_'.join(extensionless_file_name.split('_')[:-1])

    init_time_unix_sec = time_conversion.string_to_unix_sec(
        init_time_string, TIME_FORMAT
    )
    nwp_model_utils.check_init_time(
        init_time_unix_sec=init_time_unix_sec, model_name=model_name
    )

    return init_time_unix_sec


def file_name_to_model_name(nwp_forecast_file_name):
    """Parses NWP-model name from name of file with NWP forecasts.

    :param nwp_forecast_file_name: File path.
    :return: model_name: Model name.
    """

    pathless_file_name = os.path.split(nwp_forecast_file_name)[1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]

    model_name = '_'.join(extensionless_file_name.split('_')[:-1])
    nwp_model_utils.check_model_name(model_name)

    return model_name


def read_file(zarr_file_name, allow_tar=False):
    """Reads NWP output (forecasts) from zarr file.

    :param zarr_file_name: Path to input file.
    :param allow_tar: Boolean flag.  If False, input file must be untarred.
        If True and input file is tarred, will untar, read the untarred file,
        then delete the untarred file -- leaving the original tar in place.
    :return: nwp_forecast_table_xarray: xarray table.  Documentation in the
        xarray table should make values self-explanatory.
    """

    error_checking.assert_is_boolean(allow_tar)
    if not (allow_tar and zarr_file_name.endswith('.tar')):
        return xarray.open_zarr(zarr_file_name)

    tar_file_name = copy.deepcopy(zarr_file_name)
    misc_utils.untar_zarr_or_netcdf_file(
        tar_file_name=tar_file_name,
        target_dir_name=os.path.split(tar_file_name)[0]
    )

    zarr_file_name = re.sub('.tar$', '.zarr', tar_file_name)
    nwp_forecast_table_xarray = xarray.open_zarr(zarr_file_name)
    # shutil.rmtree(zarr_file_name)

    return nwp_forecast_table_xarray


def write_file(nwp_forecast_table_xarray, zarr_file_name):
    """Writes NWP output (forecasts) to zarr file.

    :param nwp_forecast_table_xarray: xarray table in format returned by
        `read_file`.
    :param zarr_file_name: Path to output file.
    """

    error_checking.assert_is_string(zarr_file_name)
    if os.path.isdir(zarr_file_name):
        shutil.rmtree(zarr_file_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=zarr_file_name
    )

    encoding_dict = {
        nwp_model_utils.DATA_KEY: {'dtype': 'float32'}
    }
    nwp_forecast_table_xarray.to_zarr(
        store=zarr_file_name, mode='w', encoding=encoding_dict
    )


def write_normalization_file(norm_param_table_xarray, netcdf_file_name):
    """Writes normalization parameters for NWP data to NetCDF file.

    :param norm_param_table_xarray: xarray table (metadata and variable names
        should make the table self-explanatory).
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    norm_param_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def read_normalization_file(netcdf_file_name):
    """Reads normalization parameters for NWP data from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: norm_param_table_xarray: xarray table (metadata and variable names
        should make the table self-explanatory).
    """

    error_checking.assert_file_exists(netcdf_file_name)
    return xarray.open_dataset(netcdf_file_name)
