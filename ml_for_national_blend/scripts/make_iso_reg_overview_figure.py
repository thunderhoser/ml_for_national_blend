"""Creates overview figure to explain isotonic regression."""

import argparse
import numpy
from ml_for_national_blend.outside_code import error_checking
from ml_for_national_blend.io import prediction_io

RAW_FILE_ARG_NAME = 'input_raw_prediction_file_name'
BIAS_CORRECTED_FILE_ARG_NAME = 'input_bc_prediction_file_name'
NUM_ATOMIC_EXAMPLES_ARG_NAME = 'num_atomic_examples'
TARGET_FIELD_ARG_NAME = 'target_field_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

RAW_FILE_HELP_STRING = (
    'Path to file with raw predictions (i.e., from base model).  Will be read '
    'by `prediction_io.read_file`.'
)
BIAS_CORRECTED_FILE_HELP_STRING = (
    'Path to file with bias-corrected predictions (i.e., from isotonic '
    'regression).  Will be read by `prediction_io.read_file`.'
)
NUM_ATOMIC_EXAMPLES_HELP_STRING = (
    'Number of atomic examples (where 1 atomic examples = 1 time step at 1 '
    'pixel) to use in plot.'
)
TARGET_FIELD_HELP_STRING = 'Name of target field.'
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + RAW_FILE_ARG_NAME, type=str, required=True,
    help=RAW_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BIAS_CORRECTED_FILE_ARG_NAME, type=str, required=True,
    help=BIAS_CORRECTED_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_ATOMIC_EXAMPLES_ARG_NAME, type=int, required=True,
    help=NUM_ATOMIC_EXAMPLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_FIELD_ARG_NAME, type=str, required=True,
    help=TARGET_FIELD_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(raw_prediction_file_name, bc_prediction_file_name, num_atomic_examples,
         target_field_name, output_dir_name):
    """Creates overview figure to explain isotonic regression.

    This is effectively the main method.

    :param raw_prediction_file_name: See documentation at top of file.
    :param bc_prediction_file_name: Same.
    :param num_atomic_examples: Same.
    :param target_field_name: Same.
    :param output_dir_name: Same.
    """

    error_checking.assert_is_geq(num_atomic_examples, 10)
    error_checking.assert_is_leq(num_atomic_examples, 1000)

    print('Reading data from: "{0:s}"...'.format(raw_prediction_file_name))
    raw_prediction_table_xarray = prediction_io.read_file(
        raw_prediction_file_name
    )

    inds = numpy.where(
        raw_prediction_table_xarray[prediction_io.FIELD_NAME_KEY].values ==
        target_field_name
    )[0]
    assert len(inds) == 1
    raw_prediction_table_xarray = raw_prediction_table_xarray.isel({
        prediction_io.FIELD_DIM: inds
    })

    print('Reading data from: "{0:s}"...'.format(bc_prediction_file_name))
    bc_prediction_table_xarray = prediction_io.read_file(
        bc_prediction_file_name
    )

    inds = numpy.where(
        bc_prediction_table_xarray[prediction_io.FIELD_NAME_KEY].values ==
        target_field_name
    )[0]
    assert len(inds) == 1
    bc_prediction_table_xarray = bc_prediction_table_xarray.isel({
        prediction_io.FIELD_DIM: inds
    })

    raw_ptx = raw_prediction_table_xarray
    bc_ptx = bc_prediction_table_xarray

    num_grid_rows = len(raw_ptx.coords[prediction_io.ROW_DIM].values)
    num_grid_columns = len(raw_ptx.coords[prediction_io.COLUMN_DIM].values)
    ensemble_size = len(
        raw_ptx.coords[prediction_io.ENSEMBLE_MEMBER_DIM].values
    )

    assert num_grid_rows == len(bc_ptx.coords[prediction_io.ROW_DIM].values)
    assert (
        num_grid_columns ==
        len(bc_ptx.coords[prediction_io.COLUMN_DIM].values)
    )
    assert (
        ensemble_size ==
        len(bc_ptx.coords[prediction_io.ENSEMBLE_MEMBER_DIM].values)
    )
    assert ensemble_size > 1

    raw_prediction_matrix = (
        raw_ptx[prediction_io.PREDICTION_KEY].values[..., 0, :]
    )
    bc_prediction_matrix = (
        bc_ptx[prediction_io.PREDICTION_KEY].values[..., 0, :]
    )
    target_matrix = bc_ptx[prediction_io.TARGET_KEY].values[..., 0]

    real_prediction_flag_matrix = numpy.logical_and(
        numpy.all(numpy.isfinite(raw_prediction_matrix), axis=-1),
        numpy.all(numpy.isfinite(bc_prediction_matrix), axis=-1)
    )
    real_target_flag_matrix = numpy.isfinite(target_matrix)

    good_row_indices, good_column_indices = numpy.where(numpy.logical_and(
        real_prediction_flag_matrix, real_target_flag_matrix
    ))
    linear_indices = numpy.linspace(
        0, len(good_row_indices) - 1, num=len(good_row_indices), dtype=int
    )
    linear_indices = numpy.random.choice(
        linear_indices, size=num_atomic_examples, replace=False
    )

    good_row_indices = good_row_indices[linear_indices]
    good_column_indices = good_column_indices[linear_indices]
    raw_prediction_matrix = (
        raw_prediction_matrix[good_row_indices, good_column_indices, :]
    )
    bc_prediction_matrix = (
        bc_prediction_matrix[good_row_indices, good_column_indices, :]
    )
    target_matrix = target_matrix[good_row_indices, good_column_indices]

    print(raw_prediction_matrix.shape)
    print(bc_prediction_matrix.shape)
    print(target_matrix.shape)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        raw_prediction_file_name=getattr(INPUT_ARG_OBJECT, RAW_FILE_ARG_NAME),
        bc_prediction_file_name=getattr(
            INPUT_ARG_OBJECT, BIAS_CORRECTED_FILE_ARG_NAME
        ),
        num_atomic_examples=getattr(
            INPUT_ARG_OBJECT, NUM_ATOMIC_EXAMPLES_ARG_NAME
        ),
        target_field_name=getattr(INPUT_ARG_OBJECT, TARGET_FIELD_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
