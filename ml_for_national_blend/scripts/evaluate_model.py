"""Evaluates model."""

import os
import argparse
import numpy
from ml_for_national_blend.outside_code import time_conversion
from ml_for_national_blend.outside_code import error_checking
from ml_for_national_blend.io import prediction_io
from ml_for_national_blend.utils import evaluation
from ml_for_national_blend.machine_learning import neural_net_utils as nn_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
HOURS_TO_SECONDS = 3600
TIME_FORMAT = '%Y-%m-%d-%H'

INPUT_DIR_ARG_NAME = 'input_prediction_dir_name'
INIT_TIME_LIMITS_ARG_NAME = 'init_time_limit_strings'
EVALUATE_MONTH_ARG_NAME = 'evaluate_month'
EVALUATE_HOUR_ARG_NAME = 'evaluate_hour'
NUM_BOOTSTRAP_REPS_ARG_NAME = 'num_bootstrap_reps'
TARGET_FIELDS_ARG_NAME = 'target_field_names'
TARGET_NORM_FILE_ARG_NAME = 'input_target_norm_file_name'
NUM_RELIA_BINS_ARG_NAME = 'num_relia_bins_by_target'
MIN_RELIA_BIN_EDGES_ARG_NAME = 'min_relia_bin_edge_by_target'
MAX_RELIA_BIN_EDGES_ARG_NAME = 'max_relia_bin_edge_by_target'
PER_GRID_CELL_ARG_NAME = 'per_grid_cell'
KEEP_IT_SIMPLE_ARG_NAME = 'keep_it_simple'
COMPUTE_SSRAT_ARG_NAME = 'compute_ssrat'
COMPUTE_SSREL_ARG_NAME = 'compute_ssrel'
OUTPUT_FILE_ARG_NAME = 'output_file_name_or_pattern'

INPUT_DIR_HELP_STRING = (
    'Name of input directory, containing one prediction file per init time.  '
    'Files therein will be found by `prediction_io.find_file` and read by '
    '`prediction_io.read_file`.'
)
INIT_TIME_LIMITS_HELP_STRING = (
    'List of two initialization times, specifying the beginning and end of the '
    'evaluation period.  Time format is "yyyy-mm-dd-HH".'
)
EVALUATE_MONTH_HELP_STRING = (
    'Will evaluate only forecasts valid in this month (ranging from '
    '1...12).  If you want to evaluate forecasts regardless of month, leave '
    'this argument alone.'
)
EVALUATE_HOUR_HELP_STRING = (
    'Will evaluate only forecasts valid at this UTC hour (ranging from '
    '0...23).  If you want to evaluate forecasts regardless of hour, leave '
    'this argument alone.'
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
    'length-T numpy array with minimum predicted value in reliability curve '
    'for each target.'
)
MAX_RELIA_BIN_EDGES_HELP_STRING = 'Same as {0:s} but for maximum'.format(
    MIN_RELIA_BIN_EDGES_ARG_NAME
)
PER_GRID_CELL_HELP_STRING = (
    'Boolean flag.  If 1, will compute a separate set of scores at each grid '
    'cell.  If 0, will compute one set of scores for the whole domain.'
)
KEEP_IT_SIMPLE_HELP_STRING = (
    'Boolean flag.  If 1, will avoid Kolmogorov-Smirnov test and attributes '
    'diagram.'
)
COMPUTE_SSRAT_HELP_STRING = (
    'Boolean flag.  If 1, will compute spread-skill ratio (SSRAT) and spread-'
    'skill difference (SSDIFF).'
)
COMPUTE_SSREL_HELP_STRING = (
    'Boolean flag.  If 1, will compute spread-skill reliability (SSREL).'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Evaluation scores will be written here by '
    '`evaluation.write_file`.  If `{0:s}` is specified, the full name of the '
    'output file will be produced by replacing the ".nc" at the end of `{1:s}` '
    'with "_month01.nc" or "_month02.nc" or... "_month12.nc".  If `{2:s}` is '
    'specified, the full name of the output file will be produced by replacing '
    'the ".nc" at the end of `{1:s}` with "_hour00.nc" or "_hour01.nc" or... '
    '"_hour23.nc".'
).format(
    EVALUATE_MONTH_ARG_NAME,
    OUTPUT_FILE_ARG_NAME,
    EVALUATE_HOUR_ARG_NAME
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
    '--' + EVALUATE_MONTH_ARG_NAME, type=int, required=False, default=-1,
    help=EVALUATE_MONTH_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EVALUATE_HOUR_ARG_NAME, type=int, required=False, default=-1,
    help=EVALUATE_HOUR_HELP_STRING
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
    '--' + MIN_RELIA_BIN_EDGES_ARG_NAME, type=float, nargs='+', required=True,
    help=MIN_RELIA_BIN_EDGES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_RELIA_BIN_EDGES_ARG_NAME, type=float, nargs='+', required=True,
    help=MAX_RELIA_BIN_EDGES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PER_GRID_CELL_ARG_NAME, type=int, required=True,
    help=PER_GRID_CELL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + KEEP_IT_SIMPLE_ARG_NAME, type=int, required=True,
    help=KEEP_IT_SIMPLE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + COMPUTE_SSRAT_ARG_NAME, type=int, required=False, default=0,
    help=COMPUTE_SSRAT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + COMPUTE_SSREL_ARG_NAME, type=int, required=False, default=0,
    help=COMPUTE_SSREL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(prediction_dir_name, init_time_limit_strings,
         evaluate_month, evaluate_hour, num_bootstrap_reps,
         target_field_names, target_normalization_file_name,
         num_relia_bins_by_target, min_relia_bin_edge_by_target,
         max_relia_bin_edge_by_target,
         per_grid_cell, keep_it_simple, compute_ssrat, compute_ssrel,
         output_file_name_or_pattern):
    """Evaluates model.

    This is effectively the main method.

    :param prediction_dir_name: See documentation at top of file.
    :param init_time_limit_strings: Same.
    :param evaluate_month: Same.
    :param evaluate_hour: Hour.
    :param num_bootstrap_reps: Same.
    :param target_field_names: Same.
    :param target_normalization_file_name: Same.
    :param num_relia_bins_by_target: Same.
    :param min_relia_bin_edge_by_target: Same.
    :param max_relia_bin_edge_by_target: Same.
    :param per_grid_cell: Same.
    :param keep_it_simple: Same.
    :param compute_ssrat: Same.
    :param compute_ssrel: Same.
    :param output_file_name_or_pattern: Same.
    """

    if evaluate_month < 1:
        evaluate_month = None
    if evaluate_hour < 0:
        evaluate_hour = None

    assert evaluate_month is None or evaluate_hour is None

    if evaluate_month is not None:
        error_checking.assert_is_leq(evaluate_month, 12)
        output_file_pattern = output_file_name_or_pattern
        assert output_file_pattern.endswith('.nc')

        output_file_name = '{0:s}_month{1:02d}.nc'.format(
            os.path.splitext(output_file_pattern)[0],
            evaluate_month
        )
    elif evaluate_hour is not None:
        error_checking.assert_is_leq(evaluate_hour, 23)
        output_file_pattern = output_file_name_or_pattern
        assert output_file_pattern.endswith('.nc')

        output_file_name = '{0:s}_hour{1:02d}.nc'.format(
            os.path.splitext(output_file_pattern)[0],
            evaluate_hour
        )
    else:
        output_file_name = output_file_name_or_pattern

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

    if len(prediction_file_names) == 0:
        prediction_file_names = prediction_io.find_rap_based_files_for_period(
            directory_name=prediction_dir_name,
            first_init_time_unix_sec=init_time_limits_unix_sec[0],
            last_init_time_unix_sec=init_time_limits_unix_sec[1],
            raise_error_if_any_missing=False,
            raise_error_if_all_missing=True
        )

    first_prediction_table_xarray = prediction_io.read_file(
        prediction_file_names[0]
    )
    first_ptx = first_prediction_table_xarray

    if 'model_lead_time_hours' in first_ptx.attrs:
        model_lead_time_hours = first_ptx.attrs['model_lead_time_hours']
    else:
        model_file_name = first_ptx.attrs[prediction_io.MODEL_FILE_KEY]
        model_metafile_name = nn_utils.find_metafile(
            model_file_name=model_file_name, raise_error_if_missing=True
        )

        print('Reading model metadata from: "{0:s}"...'.format(
            model_metafile_name
        ))
        model_metadata_dict = nn_utils.read_metafile(model_metafile_name)
        model_lead_time_hours = model_metadata_dict[
            nn_utils.TRAINING_OPTIONS_KEY
        ][nn_utils.TARGET_LEAD_TIME_KEY]

    init_times_unix_sec = numpy.array([
        prediction_io.file_name_to_init_time(f) for f in prediction_file_names
    ], dtype=int)

    valid_times_unix_sec = (
        init_times_unix_sec + model_lead_time_hours * HOURS_TO_SECONDS
    )

    if evaluate_month is not None:
        valid_months = numpy.array([
            int(time_conversion.unix_sec_to_string(t, '%Y%m%d')[4:6])
            for t in valid_times_unix_sec
        ], dtype=int)

        good_indices = numpy.where(valid_months == evaluate_month)[0]
        prediction_file_names = [prediction_file_names[k] for k in good_indices]

    if evaluate_hour is not None:
        valid_hours = numpy.array([
            int(time_conversion.unix_sec_to_string(t, '%Y%m%d%H')[8:])
            for t in valid_times_unix_sec
        ], dtype=int)

        good_indices = numpy.where(valid_hours == evaluate_hour)[0]
        prediction_file_names = [prediction_file_names[k] for k in good_indices]

    del init_times_unix_sec
    del valid_times_unix_sec

    result_table_xarray = evaluation.get_scores_with_bootstrapping(
        prediction_file_names=prediction_file_names,
        num_bootstrap_reps=num_bootstrap_reps,
        target_field_names=target_field_names,
        target_normalization_file_name=target_normalization_file_name,
        num_relia_bins_by_target=num_relia_bins_by_target,
        min_relia_bin_edge_by_target=min_relia_bin_edge_by_target,
        max_relia_bin_edge_by_target=max_relia_bin_edge_by_target,
        min_relia_bin_edge_prctile_by_target=None,
        max_relia_bin_edge_prctile_by_target=None,
        per_grid_cell=per_grid_cell,
        keep_it_simple=keep_it_simple,
        compute_ssrat=compute_ssrat,
        compute_ssrel=compute_ssrel
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
        evaluate_month=getattr(INPUT_ARG_OBJECT, EVALUATE_MONTH_ARG_NAME),
        evaluate_hour=getattr(INPUT_ARG_OBJECT, EVALUATE_HOUR_ARG_NAME),
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
        per_grid_cell=bool(getattr(INPUT_ARG_OBJECT, PER_GRID_CELL_ARG_NAME)),
        keep_it_simple=bool(getattr(INPUT_ARG_OBJECT, KEEP_IT_SIMPLE_ARG_NAME)),
        compute_ssrat=bool(getattr(INPUT_ARG_OBJECT, COMPUTE_SSRAT_ARG_NAME)),
        compute_ssrel=bool(getattr(INPUT_ARG_OBJECT, COMPUTE_SSREL_ARG_NAME)),
        output_file_name_or_pattern=getattr(
            INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME
        )
    )
