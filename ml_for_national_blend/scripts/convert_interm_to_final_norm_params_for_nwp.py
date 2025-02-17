"""Converts intermediate to final normalization parameters for NWP data.

Final normalization parameters = mean, stdev, and quantiles for each variable.
"""

import argparse
from ml_for_national_blend.io import nwp_model_io
from ml_for_national_blend.utils import normalization

INPUT_FILES_ARG_NAME = 'input_file_names'
NUM_QUANTILES_ARG_NAME = 'num_quantiles'
OUTPUT_FILE_ARG_NAME = 'output_norm_file_name'

INPUT_FILES_HELP_STRING = (
    'List of paths to input files, each containing intermediate normalization '
    'params for one data chunk.  Each file will be read by '
    '`nwp_model_io.read_normalization_file`, returning an xarray table in the '
    'format created by `normalization.get_intermediate_norm_params_for_nwp`.'
)
NUM_QUANTILES_HELP_STRING = (
    'Number of quantiles to store for each variable.  The quantile levels will '
    'be evenly spaced from 0 to 1 (i.e., the 0th to 100th percentile).'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Final normalization params will be written by '
    '`nwp_model_io.write_normalization_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_QUANTILES_ARG_NAME, type=int, required=False, default=1001,
    help=NUM_QUANTILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(intermediate_norm_file_names, num_quantiles, final_norm_file_name):
    """Converts intermediate to final normalization parameters for NWP data.

    This is effectively the main method.

    :param intermediate_norm_file_names: See documentation at top of this
        script.
    :param num_quantiles: Same.
    :param final_norm_file_name: Same.
    """

    num_data_chunks = len(intermediate_norm_file_names)
    intermediate_norm_param_tables_xarray = [None] * num_data_chunks

    for i in range(num_data_chunks):
        print((
            'Reading intermediate normalization params from: "{0:s}"...'
        ).format(
            intermediate_norm_file_names[i]
        ))

        intermediate_norm_param_tables_xarray[i] = (
            nwp_model_io.read_normalization_file(
                intermediate_norm_file_names[i]
            )
        )

    final_norm_param_table_xarray = (
        normalization.intermediate_to_final_normalization_params(
            intermediate_norm_param_tables_xarray=
            intermediate_norm_param_tables_xarray,
            num_quantiles=num_quantiles
        )
    )

    print('Writing final normalization params to: "{0:s}"...'.format(
        final_norm_file_name
    ))
    nwp_model_io.write_normalization_file(
        norm_param_table_xarray=final_norm_param_table_xarray,
        netcdf_file_name=final_norm_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        intermediate_norm_file_names=getattr(
            INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME
        ),
        num_quantiles=getattr(INPUT_ARG_OBJECT, NUM_QUANTILES_ARG_NAME),
        final_norm_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
