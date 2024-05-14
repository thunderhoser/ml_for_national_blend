"""Evaluates model."""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import prediction_io
import evaluation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

SENTINEL_VALUE = -1e6
TIME_FORMAT = '%Y-%m-%d-%H'

INPUT_DIR_ARG_NAME = 'input_prediction_dir_name'
INIT_TIME_LIMITS_ARG_NAME = 'init_time_limit_strings'
NUM_BOOTSTRAP_REPS_ARG_NAME = 'num_bootstrap_reps'
TARGET_FIELDS_ARG_NAME = 'target_field_names'
TARGET_NORM_FILE_ARG_NAME = 'input_target_norm_file_name'
NUM_RELIA_BINS_ARG_NAME = 'num_relia_bins_by_target'
MIN_RELIA_BIN_EDGES_ARG_NAME = 'min_relia_bin_edge_by_target'
MAX_RELIA_BIN_EDGES_ARG_NAME = 'max_relia_bin_edge_by_target'
MIN_RELIA_BIN_EDGES_PRCTILE_ARG_NAME = 'min_relia_bin_edge_prctile_by_target'
MAX_RELIA_BIN_EDGES_PRCTILE_ARG_NAME = 'max_relia_bin_edge_prctile_by_target'
PER_GRID_CELL_ARG_NAME = 'per_grid_cell'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory, containing one prediction file per init time.  '
    'Files therein will be found by `prediction_io.find_file` and read by '
    '`prediction_io.read_file`.'
)
INIT_TIME_LIMITS_HELP_STRING = (
    'List of two init times (format "yyyy-mm-dd-HH"), specifying the beginning '
    'and end of the evaluation period.'
)
NUM_BOOTSTRAP_REPS_HELP_STRING = 'Number of bootstrap replicates.'
TARGET_FIELDS_HELP_STRING = (
    'List of target fields to be evaluated.  Each one must be accepted by '
    '`urma_utils.check_field_name`.'
)
TARGET_NORM_FILE_HELP_STRING = (
    'Path to file with normalization parameters for target fields.  Will be '
    'read by `urma_io.read_normalization_file`.'
)
NUM_RELIA_BINS_HELP_STRING = (
    'length-T numpy array with number of bins in reliability curve for each '
    'target, where T = number of target fields.'
)
MIN_RELIA_BIN_EDGES_HELP_STRING = (
    'length-T numpy array with minimum target/predicted value in reliability '
    'curve for each target.  If you instead want minimum values to be '
    'percentiles over the data, leave this argument alone and use {0:s}.'
).format(MIN_RELIA_BIN_EDGES_PRCTILE_ARG_NAME)

MAX_RELIA_BIN_EDGES_HELP_STRING = 'Same as {0:s} but for maximum'.format(
    MIN_RELIA_BIN_EDGES_ARG_NAME
)
MIN_RELIA_BIN_EDGES_PRCTILE_HELP_STRING = (
    'length-T numpy array with percentile level used to determine minimum '
    'target/predicted value in reliability curve for each target.  If you '
    'instead want to specify raw values, leave this argument alone and use '
    '{0:s}.'
).format(MIN_RELIA_BIN_EDGES_ARG_NAME)

MAX_RELIA_BIN_EDGES_PRCTILE_HELP_STRING = (
    'Same as {0:s} but for maximum'
).format(MIN_RELIA_BIN_EDGES_PRCTILE_ARG_NAME)

PER_GRID_CELL_HELP_STRING = (
    'Boolean flag.  If 1, will compute a separate set of scores at each grid '
    'cell.  If 0, will compute one set of scores for the whole domain.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Evaluation scores will be written here by '
    '`evaluation.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INIT_TIME_LIMITS_ARG_NAME, type=str, nargs=2, required=True,
    help=INIT_TIME_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_BOOTSTRAP_REPS_ARG_NAME, type=int, required=True,
    help=NUM_BOOTSTRAP_REPS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_FIELDS_ARG_NAME, type=str, nargs='+', required=True,
    help=TARGET_FIELDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_NORM_FILE_ARG_NAME, type=str, required=True,
    help=TARGET_NORM_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_RELIA_BINS_ARG_NAME, type=int, nargs='+', required=True,
    help=NUM_RELIA_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_RELIA_BIN_EDGES_ARG_NAME, type=float, nargs='+', required=False,
    default=[SENTINEL_VALUE - 1], help=MIN_RELIA_BIN_EDGES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_RELIA_BIN_EDGES_ARG_NAME, type=float, nargs='+', required=False,
    default=[SENTINEL_VALUE - 1], help=MAX_RELIA_BIN_EDGES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_RELIA_BIN_EDGES_PRCTILE_ARG_NAME, type=float, nargs='+',
    required=False, default=[SENTINEL_VALUE - 1],
    help=MIN_RELIA_BIN_EDGES_PRCTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_RELIA_BIN_EDGES_PRCTILE_ARG_NAME, type=float, nargs='+',
    required=False, default=[SENTINEL_VALUE - 1],
    help=MAX_RELIA_BIN_EDGES_PRCTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PER_GRID_CELL_ARG_NAME, type=int, required=True,
    help=PER_GRID_CELL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(prediction_dir_name, init_time_limit_strings, num_bootstrap_reps,
         target_field_names, target_normalization_file_name,
         num_relia_bins_by_target, min_relia_bin_edge_by_target,
         max_relia_bin_edge_by_target,
         min_relia_bin_edge_prctile_by_target,
         max_relia_bin_edge_prctile_by_target,
         per_grid_cell, output_file_name):
    """Evaluates model.

    This is effectively the main method.

    :param prediction_dir_name: See documentation at top of file.
    :param init_time_limit_strings: Same.
    :param num_bootstrap_reps: Same.
    :param target_field_names: Same.
    :param target_normalization_file_name: Same.
    :param num_relia_bins_by_target: Same.
    :param min_relia_bin_edge_by_target: Same.
    :param max_relia_bin_edge_by_target: Same.
    :param min_relia_bin_edge_prctile_by_target: Same.
    :param max_relia_bin_edge_prctile_by_target: Same.
    :param per_grid_cell: Same.
    :param output_file_name: Same.
    """

    if (
            (len(min_relia_bin_edge_by_target) == 1 and
             min_relia_bin_edge_by_target[0] <= SENTINEL_VALUE) or
            (len(max_relia_bin_edge_by_target) == 1 and
             max_relia_bin_edge_by_target[0] <= SENTINEL_VALUE)
    ):
        min_relia_bin_edge_by_target = None
        max_relia_bin_edge_by_target = None

    if (
            (len(min_relia_bin_edge_prctile_by_target) == 1 and
             min_relia_bin_edge_prctile_by_target[0] <= SENTINEL_VALUE) or
            (len(max_relia_bin_edge_prctile_by_target) == 1 and
             max_relia_bin_edge_prctile_by_target[0] <= SENTINEL_VALUE)
    ):
        min_relia_bin_edge_prctile_by_target = None
        max_relia_bin_edge_prctile_by_target = None

    init_time_limits_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, TIME_FORMAT)
        for t in init_time_limit_strings
    ], dtype=int)

    prediction_file_names = prediction_io.find_files_for_period(
        directory_name=prediction_dir_name,
        first_init_time_unix_sec=init_time_limits_unix_sec[0],
        last_init_time_unix_sec=init_time_limits_unix_sec[1],
        raise_error_if_any_missing=False,
        raise_error_if_all_missing=True
    )

    result_table_xarray = evaluation.get_scores_with_bootstrapping(
        prediction_file_names=prediction_file_names,
        num_bootstrap_reps=num_bootstrap_reps,
        target_field_names=target_field_names,
        target_normalization_file_name=target_normalization_file_name,
        num_relia_bins_by_target=num_relia_bins_by_target,
        min_relia_bin_edge_by_target=min_relia_bin_edge_by_target,
        max_relia_bin_edge_by_target=max_relia_bin_edge_by_target,
        min_relia_bin_edge_prctile_by_target=
        min_relia_bin_edge_prctile_by_target,
        max_relia_bin_edge_prctile_by_target=
        max_relia_bin_edge_prctile_by_target,
        per_grid_cell=per_grid_cell
    )
    print(SEPARATOR_STRING)

    t = result_table_xarray
    target_field_names = t.coords[evaluation.FIELD_DIM].values

    for k in range(len(target_field_names)):
        print((
            'Stdev of target and predicted {0:s} = {1:f}, {2:f} ... '
            'MSE and skill score = {3:f}, {4:f} ... '
            'DWMSE and skill score = {5:f}, {6:f} ... '
            'MAE and skill score = {7:f}, {8:f} ... '
            'bias = {9:f} ... correlation = {10:f} ... KGE = {11:f} ... '
            'spatial-min bias = {12:f} ... spatial-max bias = {13:f}'
        ).format(
            target_field_names[k],
            numpy.nanmean(
                t[evaluation.TARGET_STDEV_KEY].values[..., k, :]
            ),
            numpy.nanmean(
                t[evaluation.PREDICTION_STDEV_KEY].values[..., k, :]
            ),
            numpy.nanmean(t[evaluation.MSE_KEY].values[..., k, :]),
            numpy.nanmean(
                t[evaluation.MSE_SKILL_SCORE_KEY].values[..., k, :]
            ),
            numpy.nanmean(t[evaluation.DWMSE_KEY].values[..., k, :]),
            numpy.nanmean(
                t[evaluation.DWMSE_SKILL_SCORE_KEY].values[..., k, :]
            ),
            numpy.nanmean(t[evaluation.MAE_KEY].values[..., k, :]),
            numpy.nanmean(
                t[evaluation.MAE_SKILL_SCORE_KEY].values[..., k, :]
            ),
            numpy.nanmean(t[evaluation.BIAS_KEY].values[..., k, :]),
            numpy.nanmean(t[evaluation.CORRELATION_KEY].values[..., k, :]),
            numpy.nanmean(t[evaluation.KGE_KEY].values[..., k, :]),
            numpy.nanmean(t[evaluation.SPATIAL_MIN_BIAS_KEY].values[k, :]),
            numpy.nanmean(t[evaluation.SPATIAL_MAX_BIAS_KEY].values[k, :])
        ))

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    evaluation.write_file(
        result_table_xarray=result_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        init_time_limit_strings=getattr(
            INPUT_ARG_OBJECT, INIT_TIME_LIMITS_ARG_NAME
        ),
        num_bootstrap_reps=getattr(
            INPUT_ARG_OBJECT, NUM_BOOTSTRAP_REPS_ARG_NAME
        ),
        target_field_names=getattr(
            INPUT_ARG_OBJECT, TARGET_FIELDS_ARG_NAME
        ),
        target_normalization_file_name=getattr(
            INPUT_ARG_OBJECT, TARGET_NORM_FILE_ARG_NAME
        ),
        num_relia_bins_by_target=numpy.array(
            getattr(INPUT_ARG_OBJECT, NUM_RELIA_BINS_ARG_NAME), dtype=int
        ),
        min_relia_bin_edge_by_target=numpy.array(
            getattr(INPUT_ARG_OBJECT, MIN_RELIA_BIN_EDGES_ARG_NAME), dtype=float
        ),
        max_relia_bin_edge_by_target=numpy.array(
            getattr(INPUT_ARG_OBJECT, MAX_RELIA_BIN_EDGES_ARG_NAME), dtype=float
        ),
        min_relia_bin_edge_prctile_by_target=numpy.array(
            getattr(INPUT_ARG_OBJECT, MIN_RELIA_BIN_EDGES_PRCTILE_ARG_NAME),
            dtype=float
        ),
        max_relia_bin_edge_prctile_by_target=numpy.array(
            getattr(INPUT_ARG_OBJECT, MAX_RELIA_BIN_EDGES_PRCTILE_ARG_NAME),
            dtype=float
        ),
        per_grid_cell=bool(getattr(INPUT_ARG_OBJECT, PER_GRID_CELL_ARG_NAME)),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
