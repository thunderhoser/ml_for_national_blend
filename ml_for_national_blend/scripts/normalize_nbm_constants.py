"""Normalizes NBM time-constant fields."""

import argparse
import numpy
from ml_for_national_blend.io import nbm_constant_io
from ml_for_national_blend.utils import normalization
from ml_for_national_blend.utils import nbm_constant_utils

INPUT_FILE_ARG_NAME = 'input_nbm_constant_file_name'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
USE_QUANTILE_NORM_ARG_NAME = 'use_quantile_norm'
OUTPUT_FILE_ARG_NAME = 'output_nbm_constant_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing unnormalized NBM constants.  Will be read '
    'by `nbm_constant_io.read_file`.'
)
NORMALIZATION_FILE_HELP_STRING = (
    'Path to file with normalization parameters.  Will be read by '
    '`nbm_constant_io.read_normalization_file`.'
)
USE_QUANTILE_NORM_HELP_STRING = (
    'Boolean flag.  If 1, will do two-step normalization: conversion to '
    'quantiles and then normal distribution (using inverse CDF).  If 0, will '
    'do simple z-score normalization.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Normalized NBM constants will be written here by '
    '`nbm_constant_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NORMALIZATION_FILE_ARG_NAME, type=str, required=True,
    help=NORMALIZATION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + USE_QUANTILE_NORM_ARG_NAME, type=int, required=True,
    help=USE_QUANTILE_NORM_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_file_name, normalization_file_name, use_quantile_norm,
         output_file_name):
    """Normalizes NBM time-constant fields.

    This is effectively the main method.

    :param input_file_name: See documentation at top of this script.
    :param normalization_file_name: Same.
    :param use_quantile_norm: Same.
    :param output_file_name: Same.
    """

    print('Reading unnormalized data from: "{0:s}"...'.format(input_file_name))
    nbm_constant_table_xarray = nbm_constant_io.read_file(input_file_name)

    print('Reading normalization params from: "{0:s}"...'.format(
        normalization_file_name
    ))
    norm_param_table_xarray = nbm_constant_io.read_normalization_file(
        normalization_file_name
    )

    nbm_constant_table_xarray = normalization.normalize_nbm_constants(
        nbm_constant_table_xarray=nbm_constant_table_xarray,
        norm_param_table_xarray=norm_param_table_xarray,
        use_quantile_norm=use_quantile_norm
    )
    nbmct = nbm_constant_table_xarray

    field_names = nbmct.coords[nbm_constant_utils.FIELD_DIM].values
    num_fields = len(field_names)

    for j in range(num_fields):
        this_data_matrix = nbmct[nbm_constant_utils.DATA_KEY].values[..., j]

        print((
            'Min/median/mean/max for normalized {0:s} = '
            '{1:.2g}, {2:.2g}, {3:.2g}, {4:.2g}'
        ).format(
            field_names[j],
            numpy.min(this_data_matrix),
            numpy.median(this_data_matrix),
            numpy.mean(this_data_matrix),
            numpy.max(this_data_matrix)
        ))

    print('Writing normalized data to: "{0:s}"...'.format(output_file_name))
    nbm_constant_io.write_file(
        nbm_constant_table_xarray=nbm_constant_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        normalization_file_name=getattr(
            INPUT_ARG_OBJECT, NORMALIZATION_FILE_ARG_NAME
        ),
        use_quantile_norm=bool(
            getattr(INPUT_ARG_OBJECT, USE_QUANTILE_NORM_ARG_NAME)
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
