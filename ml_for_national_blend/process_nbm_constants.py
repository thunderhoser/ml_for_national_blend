"""Processes NBM (National Blend of Models) time-constant fields.

Each raw file should be a GRIB2 file provided by Geoff Wagner, containing one
field.

The output will contain all fields in one NetCDF file.
"""

import os
import sys
import argparse
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import raw_nbm_constant_io
import nbm_constant_io
import nbm_constant_utils

INPUT_FILES_ARG_NAME = 'input_grib2_file_names'
FIELDS_ARG_NAME = 'field_names'
WGRIB2_EXE_ARG_NAME = 'wgrib2_exe_file_name'
TEMPORARY_DIR_ARG_NAME = 'temporary_dir_name'
OUTPUT_FILE_ARG_NAME = 'output_netcdf_file_name'

INPUT_FILES_HELP_STRING = (
    'List of paths to input files (each a GRIB2 file with one field).'
)
FIELDS_HELP_STRING = (
    'List of field names (one per input file).  Each field name must be '
    'accepted by `nbm_constant_utils.check_field_name`.'
)
WGRIB2_EXE_HELP_STRING = 'Path to wgrib2 executable.'
TEMPORARY_DIR_HELP_STRING = (
    'Path to temporary directory for text files created by wgrib2.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file (one NetCDF file with all fields).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIELDS_ARG_NAME, type=str, nargs='+', required=True,
    help=FIELDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + WGRIB2_EXE_ARG_NAME, type=str, required=True,
    help=WGRIB2_EXE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TEMPORARY_DIR_ARG_NAME, type=str, required=True,
    help=TEMPORARY_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_file_names, field_names, wgrib2_exe_name, temporary_dir_name,
         output_file_name):
    """Processes NBM (National Blend of Models) time-constant fields.

    This is effectively the main method.

    :param input_file_names: See documentation at top of this script.
    :param field_names: Same.
    :param wgrib2_exe_name: Same.
    :param temporary_dir_name: Same.
    :param output_file_name: Same.
    """

    # Check input args.
    num_fields = len(field_names)
    for j in range(num_fields):
        nbm_constant_utils.check_field_name(field_names[j])

    error_checking.assert_is_numpy_array(
        numpy.array(input_file_names),
        exact_dimensions=numpy.array([num_fields], dtype=int)
    )

    # Do actual stuff.
    nbm_constant_tables_xarray = [xarray.Dataset()] * num_fields

    for j in range(num_fields):
        print('Reading data from: "{0:s}"...'.format(input_file_names[j]))
        nbm_constant_tables_xarray[j] = raw_nbm_constant_io.read_file(
            grib2_file_name=input_file_names[j],
            wgrib2_exe_name=wgrib2_exe_name,
            temporary_dir_name=temporary_dir_name,
            field_name=field_names[j]
        )

    nbm_constant_table_xarray = xarray.concat(
        nbm_constant_tables_xarray, dim=nbm_constant_utils.FIELD_DIM,
        data_vars=[nbm_constant_utils.DATA_KEY],
        coords='minimal', compat='identical', join='exact'
    )

    print('Writing data to file: "{0:s}"...'.format(output_file_name))
    nbm_constant_io.write_file(
        nbm_constant_table_xarray=nbm_constant_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        field_names=getattr(INPUT_ARG_OBJECT, FIELDS_ARG_NAME),
        wgrib2_exe_name=getattr(INPUT_ARG_OBJECT, WGRIB2_EXE_ARG_NAME),
        temporary_dir_name=getattr(INPUT_ARG_OBJECT, TEMPORARY_DIR_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
