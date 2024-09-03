"""Creates overview figure to explain isotonic regression."""

import os
import sys
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking
import prediction_io
import target_plotting

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

RAW_PREDICTION_LINE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
BC_PREDICTION_LINE_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
TARGET_LINE_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
LINE_WIDTH = 3

PREDICTION_MARKER_TYPE = 'o'
PREDICTION_MARKER_SIZE = 16
TARGET_MARKER_TYPE = '*'
TARGET_MARKER_SIZE = 24

DEFAULT_FONT_SIZE = 30
pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

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

    # Basic error-checking.
    error_checking.assert_is_geq(num_atomic_examples, 10)
    error_checking.assert_is_leq(num_atomic_examples, 1000)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Read files and subset to desired field.
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

    # Make sure files have matching metadata.
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

    assert (
        raw_ptx.attrs[prediction_io.INIT_TIME_KEY] ==
        bc_ptx.attrs[prediction_io.INIT_TIME_KEY]
    )

    # Subset grid points.
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

    # TODO(thunderhoser): This is a HACK.
    # real_target_flag_matrix = numpy.isfinite(target_matrix)
    real_target_flag_matrix = target_matrix < -10.

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
    target_values = target_matrix[good_row_indices, good_column_indices]

    # Create plot to show how IR affects ensemble means.
    sort_indices = numpy.argsort(target_values)
    raw_prediction_matrix = raw_prediction_matrix[sort_indices, :]
    bc_prediction_matrix = bc_prediction_matrix[sort_indices, :]
    target_values = target_values[sort_indices]

    example_indices = numpy.linspace(
        1, num_atomic_examples, num=num_atomic_examples, dtype=float
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    legend_handles = [None] * 3

    legend_handles[0] = axes_object.plot(
        example_indices, numpy.mean(raw_prediction_matrix, axis=-1),
        color=RAW_PREDICTION_LINE_COLOUR,
        linestyle='solid',
        linewidth=LINE_WIDTH,
        marker=PREDICTION_MARKER_TYPE,
        markersize=PREDICTION_MARKER_SIZE,
        markerfacecolor=RAW_PREDICTION_LINE_COLOUR,
        markeredgecolor=RAW_PREDICTION_LINE_COLOUR,
        markeredgewidth=0
    )[0]
    legend_handles[1] = axes_object.plot(
        example_indices, numpy.mean(bc_prediction_matrix, axis=-1),
        color=BC_PREDICTION_LINE_COLOUR,
        linestyle='solid',
        linewidth=LINE_WIDTH,
        marker=PREDICTION_MARKER_TYPE,
        markersize=PREDICTION_MARKER_SIZE,
        markerfacecolor=BC_PREDICTION_LINE_COLOUR,
        markeredgecolor=BC_PREDICTION_LINE_COLOUR,
        markeredgewidth=0
    )[0]
    legend_handles[2] = axes_object.plot(
        example_indices, target_values,
        color=TARGET_LINE_COLOUR,
        linestyle='solid',
        linewidth=LINE_WIDTH,
        marker=TARGET_MARKER_TYPE,
        markersize=TARGET_MARKER_SIZE,
        markerfacecolor=TARGET_LINE_COLOUR,
        markeredgecolor=TARGET_LINE_COLOUR,
        markeredgewidth=0
    )[0]

    legend_strings = ['Raw pred''n', 'Bias-corrected pred''n', 'Actual']

    axes_object.legend(
        legend_handles, legend_strings, loc='upper left',
        bbox_to_anchor=(0, 0.95), fancybox=True, shadow=False,
        facecolor='white', edgecolor='k', framealpha=0.5, ncol=1
    )

    axes_object.set_xticks([], [])
    axes_object.set_xlabel('Data sample')
    axes_object.set_ylabel(
        target_plotting.FIELD_NAME_TO_FANCY[target_field_name]
    )

    output_file_name = '{0:s}/ir_effect_on_ensemble_mean.jpg'.format(
        output_dir_name
    )
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


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
