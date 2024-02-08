"""Interpolates NWP data from native grid to NBM grid."""

import os
import sys
import argparse
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import grids
import time_conversion
import nwp_model_io
import nbm_utils
import nwp_model_utils

TIME_FORMAT = '%Y-%m-%d-%H'

MODEL_NAME_TO_DOWNSAMPLING_FACTOR = {
    nwp_model_utils.WRF_ARW_MODEL_NAME: 1,
    nwp_model_utils.NAM_NEST_MODEL_NAME: 1,
    nwp_model_utils.HRRR_MODEL_NAME: 1,
    nwp_model_utils.GRIDDED_LAMP_MODEL_NAME: 1,
    nwp_model_utils.RAP_MODEL_NAME: 4,
    nwp_model_utils.NAM_MODEL_NAME: 4,
    nwp_model_utils.GFS_MODEL_NAME: 8,
    nwp_model_utils.GEFS_MODEL_NAME: 16
}

INPUT_DIR_ARG_NAME = 'input_native_grid_dir_name'
MODEL_ARG_NAME = 'model_name'
FIRST_INIT_TIME_ARG_NAME = 'first_init_time_string'
LAST_INIT_TIME_ARG_NAME = 'last_init_time_string'
OUTPUT_DIR_ARG_NAME = 'output_nbm_grid_dir_name'

INPUT_DIR_HELP_STRING = (
    'Path to input directory.  Data on native model grid will be found in this '
    'directory by `nwp_model_io.find_file`.'
)
MODEL_HELP_STRING = (
    'Name of NWP model (must be accepted by '
    '`nwp_model_utils.check_model_name`).'
)
FIRST_INIT_TIME_HELP_STRING = (
    'First init time (format "yyyy-mm-dd-HH").  This script will process model '
    'runs initialized at all times in the continuous period {0:s}...{1:s}.'
).format(
    FIRST_INIT_TIME_ARG_NAME, LAST_INIT_TIME_ARG_NAME
)
LAST_INIT_TIME_HELP_STRING = 'See documentation for {0:s}.'.format(
    FIRST_INIT_TIME_ARG_NAME
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Data on NBM grid will be written here (one '
    'NetCDF file per model run per lead time) by '
    '`nwp_model_io.write_file_on_nbm_grid`, to exact locations determined by '
    '`nwp_model_io.find_file_on_nbm_grid`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_ARG_NAME, type=str, required=True, help=MODEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_INIT_TIME_ARG_NAME, type=str, required=True,
    help=FIRST_INIT_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_INIT_TIME_ARG_NAME, type=str, required=True,
    help=LAST_INIT_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_dir_name, model_name, first_init_time_string,
         last_init_time_string, output_dir_name):
    """Interpolates NWP data from native grid to NBM grid.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of this script.
    :param model_name: Same.
    :param first_init_time_string: Same.
    :param last_init_time_string: Same.
    :param output_dir_name: Same.
    """

    first_init_time_unix_sec = time_conversion.string_to_unix_sec(
        first_init_time_string, TIME_FORMAT
    )
    last_init_time_unix_sec = time_conversion.string_to_unix_sec(
        last_init_time_string, TIME_FORMAT
    )
    input_file_names = nwp_model_io.find_files_for_period(
        directory_name=input_dir_name,
        model_name=model_name,
        first_init_time_unix_sec=first_init_time_unix_sec,
        last_init_time_unix_sec=last_init_time_unix_sec,
        raise_error_if_any_missing=False,
        raise_error_if_all_missing=True
    )

    downsampling_factor = MODEL_NAME_TO_DOWNSAMPLING_FACTOR[model_name]
    if downsampling_factor == 1:
        nbm_x_coords = nbm_utils.NBM_X_COORDS_METRES
        nbm_y_coords = nbm_utils.NBM_Y_COORDS_METRES
    else:
        nbm_x_coords = nbm_utils.NBM_X_COORDS_METRES[::downsampling_factor][:-1]
        nbm_y_coords = nbm_utils.NBM_Y_COORDS_METRES[::downsampling_factor][:-1]

    nbm_x_coord_matrix, nbm_y_coord_matrix = grids.xy_vectors_to_matrices(
        x_unique_metres=nbm_x_coords, y_unique_metres=nbm_y_coords
    )
    nbm_latitude_matrix_deg_n, nbm_longitude_matrix_deg_e = (
        nbm_utils.project_xy_to_latlng(
            x_coords_metres=nbm_x_coord_matrix,
            y_coords_metres=nbm_y_coord_matrix
        )
    )

    for this_input_file_name in input_file_names:
        print('Reading data on native grid from: "{0:s}"...'.format(
            this_input_file_name
        ))
        nwp_forecast_table_xarray = nwp_model_io.read_file(this_input_file_name)
        nwpft = nwp_forecast_table_xarray

        native_x_coord_matrix, native_y_coord_matrix = (
            nbm_utils.project_latlng_to_xy(
                latitudes_deg_n=nwpft[nwp_model_utils.LATITUDE_KEY].values,
                longitudes_deg_e=nwpft[nwp_model_utils.LONGITUDE_KEY].values
            )
        )

        forecast_hours = nwpft.coords[nwp_model_utils.FORECAST_HOUR_DIM].values

        for j in range(len(forecast_hours)):
            data_matrix = nwpft[nwp_model_utils.DATA_KEY].values[j, ...]
            interp_data_matrix = nbm_utils.interp_data_to_nbm_grid(
                data_matrix=data_matrix,
                x_coord_matrix=native_x_coord_matrix,
                y_coord_matrix=native_y_coord_matrix,
                use_nearest_neigh=True,
                new_x_coords=nbm_x_coords,
                new_y_coords=nbm_y_coords
            )

            coord_dict = {
                nwp_model_utils.FORECAST_HOUR_DIM: numpy.array(
                    [forecast_hours[j]], dtype=int
                ),
                nwp_model_utils.ROW_DIM: numpy.linspace(
                    0, interp_data_matrix.shape[0] - 1,
                    num=interp_data_matrix.shape[0], dtype=int
                ),
                nwp_model_utils.COLUMN_DIM: numpy.linspace(
                    0, interp_data_matrix.shape[1] - 1,
                    num=interp_data_matrix.shape[1], dtype=int
                ),
                nwp_model_utils.FIELD_DIM:
                    nwpft.coords[nwp_model_utils.FIELD_DIM].values
            }

            these_dim = (
                nwp_model_utils.FORECAST_HOUR_DIM, nwp_model_utils.ROW_DIM,
                nwp_model_utils.COLUMN_DIM, nwp_model_utils.FIELD_DIM
            )
            main_data_dict = {
                nwp_model_utils.DATA_KEY: (
                    these_dim, numpy.expand_dims(interp_data_matrix, axis=0)
                )
            }

            these_dim = (nwp_model_utils.ROW_DIM, nwp_model_utils.COLUMN_DIM)
            main_data_dict.update({
                nwp_model_utils.LATITUDE_KEY: (
                    these_dim, nbm_latitude_matrix_deg_n
                ),
                nwp_model_utils.LONGITUDE_KEY: (
                    these_dim, nbm_longitude_matrix_deg_e
                )
            })

            interp_table_xarray = xarray.Dataset(
                data_vars=main_data_dict, coords=coord_dict
            )

            output_file_name = nwp_model_io.find_file(
                directory_name=output_dir_name,
                model_name=model_name,
                init_time_unix_sec=
                nwp_model_io.file_name_to_init_time(this_input_file_name),
                in_zarr_format=False,
                forecast_hour=forecast_hours[j],
                raise_error_if_missing=False
            )

            print('Writing interpolated data to: "{0:s}"...'.format(
                output_file_name
            ))
            nwp_model_io.write_file(
                nwp_forecast_table_xarray=interp_table_xarray,
                netcdf_file_name=output_file_name
            )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        model_name=getattr(INPUT_ARG_OBJECT, MODEL_ARG_NAME),
        first_init_time_string=getattr(
            INPUT_ARG_OBJECT, FIRST_INIT_TIME_ARG_NAME
        ),
        last_init_time_string=getattr(
            INPUT_ARG_OBJECT, LAST_INIT_TIME_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
