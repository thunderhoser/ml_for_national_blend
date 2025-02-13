"""Computes normalization parameters for NBM time-constant fields.

Normalization parameters = mean, stdev, and quantiles for each variable.
"""

import argparse
from ml_for_national_blend.io import nbm_constant_io
from ml_for_national_blend.utils import normalization

INPUT_FILE_ARG_NAME = 'input_nbm_constant_file_name'
NUM_QUANTILES_ARG_NAME = 'num_quantiles'
OUTPUT_FILE_ARG_NAME = 'output_norm_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `nbm_constant_io.read_file`.'
)
NUM_QUANTILES_HELP_STRING = (
    'Number of quantiles to store for each variable.  The quantile levels will '
    'be evenly spaced from 0 to 1 (i.e., the 0th to 100th percentile).'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Will be written by '
    '`nbm_constant_io.write_normalization_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_QUANTILES_ARG_NAME, type=int, required=False, default=1001,
    help=NUM_QUANTILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_file_name, num_quantiles, output_file_name):
    """Computes normalization parameters for NBM time-constant fields.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param num_quantiles: Same.
    :param output_file_name: Same.
    """

    norm_param_table_xarray = (
        normalization.get_normalization_params_for_nbm_const(
            nbm_constant_file_name=input_file_name,
            num_quantiles=num_quantiles
        )
    )

    print('Writing normalization parameters to: "{0:s}"...'.format(
        output_file_name
    ))
    nbm_constant_io.write_normalization_file(
        norm_param_table_xarray=norm_param_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        num_quantiles=getattr(INPUT_ARG_OBJECT, NUM_QUANTILES_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
