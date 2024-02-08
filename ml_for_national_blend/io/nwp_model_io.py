"""Input/output methods for processed NWP-model data."""

import os
import shutil
import xarray
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml_for_national_blend.utils import nwp_model_utils

TIME_FORMAT = '%Y-%m-%d-%H'


def find_file(directory_name, init_time_unix_sec, model_name,
              in_zarr_format=True, forecast_hour=None,
              raise_error_if_missing=True):
    """Finds file with NWP output for 1 model run -- or 1 run & 1 forecast hour.

    :param directory_name: Path to input directory.
    :param init_time_unix_sec: Initialization time.
    :param model_name: Name of NWP model (must be accepted by
        `nwp_model_utils.check_model_name`).
    :param in_zarr_format: Boolean flag.  If True (False), will look for data in
        zarr (NetCDF) format.
    :param forecast_hour: [used only if in_zarr_format == False]
        Forecast hour.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: nwp_forecast_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    # TODO(thunderhoser): Build in some flexibility on file format... maybe.

    error_checking.assert_is_string(directory_name)
    nwp_model_utils.check_init_time(
        init_time_unix_sec=init_time_unix_sec, model_name=model_name
    )
    error_checking.assert_is_boolean(in_zarr_format)
    error_checking.assert_is_boolean(raise_error_if_missing)

    if in_zarr_format:
        nwp_forecast_file_name = '{0:s}/{1:s}_{2:s}.zarr'.format(
            directory_name,
            model_name,
            time_conversion.unix_sec_to_string(init_time_unix_sec, TIME_FORMAT)
        )

        if raise_error_if_missing and not os.path.isdir(nwp_forecast_file_name):
            error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
                nwp_forecast_file_name
            )
            raise ValueError(error_string)

        return nwp_forecast_file_name

    error_checking.assert_is_integer(forecast_hour)
    error_checking.assert_is_greater(forecast_hour, 0)

    nwp_forecast_file_name = '{0:s}/{1:s}_{2:s}_hour{3:03d}.nc'.format(
        directory_name,
        model_name,
        time_conversion.unix_sec_to_string(init_time_unix_sec, TIME_FORMAT),
        forecast_hour
    )

    if raise_error_if_missing and not os.path.isfile(nwp_forecast_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            nwp_forecast_file_name
        )
        raise ValueError(error_string)

    return nwp_forecast_file_name


def find_files_for_period(
        directory_name, model_name,
        first_init_time_unix_sec, last_init_time_unix_sec,
        in_zarr_format=True, forecast_hour=None,
        raise_error_if_any_missing=False, raise_error_if_all_missing=True):
    """Finds many files with NWP output.

    :param directory_name: Path to input directory.
    :param model_name: Name of NWP model (must be accepted by
        `nwp_model_utils.check_model_name`).
    :param first_init_time_unix_sec: First initialization time in period.
    :param last_init_time_unix_sec: Last initialization time in period.
    :param in_zarr_format: Boolean flag.  If True (False), will look for data in
        zarr (NetCDF) format.
    :param forecast_hour: [used only if in_zarr_format == False]
        Forecast hour.
    :param raise_error_if_any_missing: Boolean flag.  If any file is missing and
        `raise_error_if_any_missing == True`, will throw error.
    :param raise_error_if_all_missing: Boolean flag.  If all files are missing
        and `raise_error_if_all_missing == True`, will throw error.
    :return: nwp_forecast_file_names: 1-D list of file paths.
    :raises: ValueError: if all files are missing and
        `raise_error_if_all_missing == True`.
    """

    error_checking.assert_is_boolean(in_zarr_format)
    error_checking.assert_is_boolean(raise_error_if_any_missing)
    error_checking.assert_is_boolean(raise_error_if_all_missing)
    init_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_init_time_unix_sec,
        end_time_unix_sec=last_init_time_unix_sec,
        time_interval_sec=
        nwp_model_utils.model_to_init_time_interval(model_name),
        include_endpoint=True
    )

    nwp_forecast_file_names = []

    for this_init_time_unix_sec in init_times_unix_sec:
        this_file_name = find_file(
            directory_name=directory_name,
            model_name=model_name,
            init_time_unix_sec=this_init_time_unix_sec,
            in_zarr_format=in_zarr_format,
            forecast_hour=forecast_hour,
            raise_error_if_missing=raise_error_if_any_missing
        )

        if in_zarr_format and os.path.isdir(this_file_name):
            nwp_forecast_file_names.append(this_file_name)

        if not in_zarr_format and os.path.isfile(this_file_name):
            nwp_forecast_file_names.append(this_file_name)

    if raise_error_if_all_missing and len(nwp_forecast_file_names) == 0:
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

    return nwp_forecast_file_names


def file_name_to_init_time(nwp_forecast_file_name):
    """Parses initialization time from name of file with NWP forecasts.

    :param nwp_forecast_file_name: File path.
    :return: init_time_unix_sec: Initialization time.
    """

    pathless_file_name = os.path.split(nwp_forecast_file_name)[1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]

    if 'hour' in extensionless_file_name.split('_')[-1]:
        init_time_string = extensionless_file_name.split('_')[-2]
        model_name = '_'.join(extensionless_file_name.split('_')[:-2])
    else:
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

    if 'hour' in extensionless_file_name.split('_')[-1]:
        model_name = '_'.join(extensionless_file_name.split('_')[:-2])
    else:
        model_name = '_'.join(extensionless_file_name.split('_')[:-1])

    nwp_model_utils.check_model_name(model_name)

    return model_name


def file_name_to_forecast_hour(nwp_forecast_file_name):
    """Parses forecast hour from name of file with NWP forecasts.

    :param nwp_forecast_file_name: File path.
    :return: forecast_hour: Forecast hour.  If None, this means that the file
        contains multiple forecast hours.
    """

    pathless_file_name = os.path.split(nwp_forecast_file_name)[1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]

    if 'hour' in extensionless_file_name.split('_')[-1]:
        forecast_hour = int(
            extensionless_file_name.split('_')[-1].replace('hour', '')
        )
        error_checking.assert_is_greater(forecast_hour, 0)

        return forecast_hour

    return None


def read_file(input_file_name):
    """Reads NWP output (forecasts) from zarr or NetCDF file.

    :param input_file_name: Path to input file.
    :return: nwp_forecast_table_xarray: xarray table.  Documentation in the
        xarray table should make values self-explanatory.
    """

    if input_file_name.endswith('.zarr'):
        return xarray.open_zarr(input_file_name)

    return xarray.open_dataset(input_file_name)


def write_file(nwp_forecast_table_xarray, zarr_file_name=None,
               netcdf_file_name=None):
    """Writes NWP output (forecasts) to zarr or NetCDF file.

    :param nwp_forecast_table_xarray: xarray table in format returned by
        `read_file`.
    :param zarr_file_name: Path to output file (if zarr format is desired).
    :param netcdf_file_name: Path to output file (if NetCDF format is desired).
    """

    if zarr_file_name is not None:
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

        return

    error_checking.assert_is_string(netcdf_file_name)
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)

    encoding_dict = {
        nwp_model_utils.DATA_KEY: {'dtype': 'float32'}
    }
    nwp_forecast_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT',
        encoding=encoding_dict
    )
