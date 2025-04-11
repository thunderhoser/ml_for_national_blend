"""Prepares example files for faster NN-training."""

import copy
import json
import argparse
import numpy
from ml_for_national_blend.io import example_io
from ml_for_national_blend.utils import nwp_model_utils
from ml_for_national_blend.outside_code import time_conversion
from ml_for_national_blend.machine_learning import neural_net_utils as nn_utils
from ml_for_national_blend.machine_learning import \
    neural_net_training_simple as nn_training_simple
from ml_for_national_blend.scripts import training_args

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
TIME_FORMAT = '%Y-%m-%d-%H'

NWP_LEAD_TIMES_ARG_NAME = training_args.NWP_LEAD_TIMES_ARG_NAME
NWP_MODELS_ARG_NAME = training_args.NWP_MODELS_ARG_NAME
NWP_MODEL_TO_FIELDS_ARG_NAME = training_args.NWP_MODEL_TO_FIELDS_ARG_NAME
NWP_NORMALIZATION_FILE_ARG_NAME = training_args.NWP_NORMALIZATION_FILE_ARG_NAME
NWP_RESID_NORM_FILE_ARG_NAME = training_args.NWP_RESID_NORM_FILE_ARG_NAME
NWP_USE_QUANTILE_NORM_ARG_NAME = training_args.NWP_USE_QUANTILE_NORM_ARG_NAME

BACKUP_NWP_MODEL_ARG_NAME = training_args.BACKUP_NWP_MODEL_ARG_NAME
BACKUP_NWP_DIR_ARG_NAME = training_args.BACKUP_NWP_DIR_ARG_NAME

TARGET_LEAD_TIME_ARG_NAME = training_args.TARGET_LEAD_TIME_ARG_NAME
TARGET_FIELDS_ARG_NAME = training_args.TARGET_FIELDS_ARG_NAME
TARGET_LAG_TIMES_ARG_NAME = training_args.TARGET_LAG_TIMES_ARG_NAME
TARGET_NORMALIZATION_FILE_ARG_NAME = training_args.TARGET_NORMALIZATION_FILE_ARG_NAME
TARGET_RESID_NORM_FILE_ARG_NAME = training_args.TARGET_RESID_NORM_FILE_ARG_NAME
TARGETS_USE_QUANTILE_NORM_ARG_NAME = training_args.TARGETS_USE_QUANTILE_NORM_ARG_NAME

RECENT_BIAS_LAG_TIMES_ARG_NAME = training_args.RECENT_BIAS_LAG_TIMES_ARG_NAME
RECENT_BIAS_LEAD_TIMES_ARG_NAME = training_args.RECENT_BIAS_LEAD_TIMES_ARG_NAME

NBM_CONSTANT_FIELDS_ARG_NAME = training_args.NBM_CONSTANT_FIELDS_ARG_NAME
NBM_CONSTANT_FILE_ARG_NAME = training_args.NBM_CONSTANT_FILE_ARG_NAME

COMPARE_TO_BASELINE_ARG_NAME = training_args.COMPARE_TO_BASELINE_ARG_NAME
# BATCH_SIZE_ARG_NAME = training_args.BATCH_SIZE_ARG_NAME
SENTINEL_VALUE_ARG_NAME = training_args.SENTINEL_VALUE_ARG_NAME

PATCH_SIZE_ARG_NAME = training_args.PATCH_SIZE_ARG_NAME
PATCH_BUFFER_SIZE_ARG_NAME = training_args.PATCH_BUFFER_SIZE_ARG_NAME
PATCH_START_ROW_ARG_NAME = training_args.PATCH_START_ROW_ARG_NAME
PATCH_START_COLUMN_ARG_NAME = training_args.PATCH_START_COLUMN_ARG_NAME
REQUIRE_ALL_PREDICTORS_ARG_NAME = training_args.REQUIRE_ALL_PREDICTORS_ARG_NAME

DO_RESIDUAL_PREDICTION_ARG_NAME = training_args.DO_RESIDUAL_PREDICTION_ARG_NAME
RESID_BASELINE_MODEL_ARG_NAME = training_args.RESID_BASELINE_MODEL_ARG_NAME
RESID_BASELINE_LEAD_TIME_ARG_NAME = training_args.RESID_BASELINE_LEAD_TIME_ARG_NAME
RESID_BASELINE_MODEL_DIR_ARG_NAME = training_args.RESID_BASELINE_MODEL_DIR_ARG_NAME

FIRST_TRAINING_TIMES_ARG_NAME = training_args.FIRST_TRAINING_TIMES_ARG_NAME
LAST_TRAINING_TIMES_ARG_NAME = training_args.LAST_TRAINING_TIMES_ARG_NAME
TRAINING_NWP_DIRS_ARG_NAME = training_args.TRAINING_NWP_DIRS_ARG_NAME
TRAINING_TARGET_DIR_ARG_NAME = training_args.TRAINING_TARGET_DIR_ARG_NAME

OUTPUT_DIR_ARG_NAME = 'output_dir_name'

NWP_LEAD_TIMES_HELP_STRING = training_args.NWP_LEAD_TIMES_HELP_STRING
NWP_MODELS_HELP_STRING = training_args.NWP_MODELS_HELP_STRING
NWP_MODEL_TO_FIELDS_HELP_STRING = training_args.NWP_MODEL_TO_FIELDS_HELP_STRING
NWP_NORMALIZATION_FILE_HELP_STRING = training_args.NWP_NORMALIZATION_FILE_HELP_STRING
NWP_RESID_NORM_FILE_HELP_STRING = training_args.NWP_RESID_NORM_FILE_HELP_STRING
NWP_USE_QUANTILE_NORM_HELP_STRING = training_args.NWP_USE_QUANTILE_NORM_HELP_STRING
BACKUP_NWP_MODEL_HELP_STRING = training_args.BACKUP_NWP_MODEL_HELP_STRING
BACKUP_NWP_DIR_HELP_STRING = training_args.BACKUP_NWP_DIR_HELP_STRING
TARGET_LEAD_TIME_HELP_STRING = training_args.TARGET_LEAD_TIME_HELP_STRING
TARGET_FIELDS_HELP_STRING = training_args.TARGET_FIELDS_HELP_STRING
TARGET_LAG_TIMES_HELP_STRING = training_args.TARGET_LAG_TIMES_HELP_STRING
TARGET_NORMALIZATION_FILE_HELP_STRING = training_args.TARGET_NORMALIZATION_FILE_HELP_STRING
TARGET_RESID_NORM_FILE_HELP_STRING = training_args.TARGET_RESID_NORM_FILE_HELP_STRING
TARGETS_USE_QUANTILE_NORM_HELP_STRING = training_args.TARGETS_USE_QUANTILE_NORM_HELP_STRING
RECENT_BIAS_LAG_TIMES_HELP_STRING = training_args.RECENT_BIAS_LAG_TIMES_HELP_STRING
RECENT_BIAS_LEAD_TIMES_HELP_STRING = training_args.RECENT_BIAS_LEAD_TIMES_HELP_STRING
NBM_CONSTANT_FIELDS_HELP_STRING = training_args.NBM_CONSTANT_FIELDS_HELP_STRING
NBM_CONSTANT_FILE_HELP_STRING = training_args.NBM_CONSTANT_FILE_HELP_STRING
COMPARE_TO_BASELINE_HELP_STRING = training_args.COMPARE_TO_BASELINE_HELP_STRING
# BATCH_SIZE_HELP_STRING = training_args.BATCH_SIZE_HELP_STRING
SENTINEL_VALUE_HELP_STRING = training_args.SENTINEL_VALUE_HELP_STRING
PATCH_SIZE_HELP_STRING = training_args.PATCH_SIZE_HELP_STRING
PATCH_BUFFER_SIZE_HELP_STRING = training_args.PATCH_BUFFER_SIZE_HELP_STRING
PATCH_START_ROW_HELP_STRING = training_args.PATCH_START_ROW_HELP_STRING
PATCH_START_COLUMN_HELP_STRING = training_args.PATCH_START_COLUMN_HELP_STRING
REQUIRE_ALL_PREDICTORS_HELP_STRING = training_args.REQUIRE_ALL_PREDICTORS_HELP_STRING
DO_RESIDUAL_PREDICTION_HELP_STRING = training_args.DO_RESIDUAL_PREDICTION_HELP_STRING
RESID_BASELINE_MODEL_HELP_STRING = training_args.RESID_BASELINE_MODEL_HELP_STRING
RESID_BASELINE_LEAD_TIME_HELP_STRING = training_args.RESID_BASELINE_LEAD_TIME_HELP_STRING
RESID_BASELINE_MODEL_DIR_HELP_STRING = training_args.RESID_BASELINE_MODEL_DIR_HELP_STRING
FIRST_TRAINING_TIMES_HELP_STRING = training_args.FIRST_TRAINING_TIMES_HELP_STRING
LAST_TRAINING_TIMES_HELP_STRING = training_args.LAST_TRAINING_TIMES_HELP_STRING
TRAINING_NWP_DIRS_HELP_STRING = training_args.TRAINING_NWP_DIRS_HELP_STRING
TRAINING_TARGET_DIR_HELP_STRING = training_args.TRAINING_TARGET_DIR_HELP_STRING
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Fully processed learning examples will be '
    'written here by `example_io.write_file`, to exact locations determined by '
    '`example_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()

INPUT_ARG_PARSER.add_argument(
    '--' + NWP_LEAD_TIMES_ARG_NAME, type=int, nargs='+', required=True,
    help=NWP_LEAD_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NWP_MODELS_ARG_NAME, type=str, nargs='+', required=True,
    help=NWP_MODELS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NWP_MODEL_TO_FIELDS_ARG_NAME, type=str, required=True,
    help=NWP_MODEL_TO_FIELDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NWP_NORMALIZATION_FILE_ARG_NAME, type=str, required=True,
    help=NWP_NORMALIZATION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NWP_RESID_NORM_FILE_ARG_NAME, type=str, required=False,
    default='', help=NWP_RESID_NORM_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NWP_USE_QUANTILE_NORM_ARG_NAME, type=int, required=False,
    default=1, help=NWP_USE_QUANTILE_NORM_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BACKUP_NWP_MODEL_ARG_NAME, type=str, required=True,
    help=BACKUP_NWP_MODEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BACKUP_NWP_DIR_ARG_NAME, type=str, required=True,
    help=BACKUP_NWP_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_LEAD_TIME_ARG_NAME, type=int, required=True,
    help=TARGET_LEAD_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_FIELDS_ARG_NAME, type=str, nargs='+', required=True,
    help=TARGET_FIELDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_LAG_TIMES_ARG_NAME, type=int, nargs='+', required=True,
    help=TARGET_LAG_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_NORMALIZATION_FILE_ARG_NAME, type=str, required=True,
    help=TARGET_NORMALIZATION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_RESID_NORM_FILE_ARG_NAME, type=str, required=False,
    default='', help=TARGET_RESID_NORM_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGETS_USE_QUANTILE_NORM_ARG_NAME, type=int, required=False,
    default=1, help=TARGETS_USE_QUANTILE_NORM_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + RECENT_BIAS_LAG_TIMES_ARG_NAME, type=int, nargs='+',
    required=False, default=[-1], help=RECENT_BIAS_LAG_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + RECENT_BIAS_LEAD_TIMES_ARG_NAME, type=int, nargs='+',
    required=False, default=[-1], help=RECENT_BIAS_LEAD_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NBM_CONSTANT_FIELDS_ARG_NAME, type=str, nargs='+',
    required=False, default=[''], help=NBM_CONSTANT_FIELDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NBM_CONSTANT_FILE_ARG_NAME, type=str, required=True,
    help=NBM_CONSTANT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + COMPARE_TO_BASELINE_ARG_NAME, type=int, required=True,
    help=COMPARE_TO_BASELINE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SENTINEL_VALUE_ARG_NAME, type=float, required=False,
    default=-10., help=SENTINEL_VALUE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PATCH_SIZE_ARG_NAME, type=int, required=True,
    help=PATCH_SIZE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PATCH_BUFFER_SIZE_ARG_NAME, type=int, required=False, default=0,
    help=PATCH_BUFFER_SIZE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PATCH_START_ROW_ARG_NAME, type=int, required=False, default=-1,
    help=PATCH_START_ROW_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PATCH_START_COLUMN_ARG_NAME, type=int, required=False,
    default=-1, help=PATCH_START_COLUMN_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + REQUIRE_ALL_PREDICTORS_ARG_NAME, type=int, required=True,
    help=REQUIRE_ALL_PREDICTORS_HELP_STRING
)

INPUT_ARG_PARSER.add_argument(
    '--' + DO_RESIDUAL_PREDICTION_ARG_NAME, type=int, required=True,
    help=DO_RESIDUAL_PREDICTION_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + RESID_BASELINE_MODEL_ARG_NAME, type=str, required=False,
    default='', help=RESID_BASELINE_MODEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + RESID_BASELINE_LEAD_TIME_ARG_NAME, type=int, required=False,
    default=-1, help=RESID_BASELINE_LEAD_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + RESID_BASELINE_MODEL_DIR_ARG_NAME, type=str, required=False,
    default='', help=RESID_BASELINE_MODEL_DIR_HELP_STRING
)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TRAINING_TIMES_ARG_NAME, type=str, nargs='+',
    required=True, help=FIRST_TRAINING_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TRAINING_TIMES_ARG_NAME, type=str, nargs='+',
    required=True, help=LAST_TRAINING_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TRAINING_NWP_DIRS_ARG_NAME, type=str, nargs='+', required=True,
    help=TRAINING_NWP_DIRS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TRAINING_TARGET_DIR_ARG_NAME, type=str, required=True,
    help=TRAINING_TARGET_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _process_nwp_directories(nwp_directory_names, nwp_model_names):
    """Processes NWP directories for either training or validation data.

    :param nwp_directory_names: See documentation for input arg
        "nwp_directory_names" to this script.
    :param nwp_model_names: See documentation for input arg to this script.
    :return: nwp_model_to_dir_name: Dictionary, where each key is the name of an
        NWP model and the corresponding value is the input directory.
    """

    # assert len(nwp_model_names) == len(nwp_directory_names)
    nwp_directory_names = nwp_directory_names[:len(nwp_model_names)]

    if len(nwp_directory_names) == 1:
        found_any_model_name_in_dir_name = any([
            m in nwp_directory_names[0]
            for m in nwp_model_utils.ALL_MODEL_NAMES_WITH_ENSEMBLE
        ])
        infer_directories = (
                len(nwp_model_names) > 1 or
                (len(nwp_model_names) == 1 and not found_any_model_name_in_dir_name)
        )
    else:
        infer_directories = False

    if infer_directories:
        top_directory_name = copy.deepcopy(nwp_directory_names[0])
        nwp_directory_names = [
            '{0:s}/{1:s}/processed/interp_to_nbm_grid'.format(
                top_directory_name, m
            ) for m in nwp_model_names
        ]

    return dict(zip(nwp_model_names, nwp_directory_names))


def _run(nwp_lead_times_hours, nwp_model_names, nwp_model_to_field_names,
         nwp_normalization_file_name, nwp_resid_norm_file_name,
         nwp_use_quantile_norm, backup_nwp_model_name, backup_nwp_dir_name,
         target_lead_time_hours, target_field_names, target_lag_times_hours,
         target_normalization_file_name, target_resid_norm_file_name,
         targets_use_quantile_norm,
         recent_bias_init_time_lags_hours, recent_bias_lead_times_hours,
         nbm_constant_field_names, nbm_constant_file_name,
         compare_to_baseline_in_loss, sentinel_value,
         patch_size_2pt5km_pixels, patch_buffer_size_2pt5km_pixels,
         patch_start_row_2pt5km, patch_start_column_2pt5km,
         require_all_predictors,
         do_residual_prediction, resid_baseline_model_name,
         resid_baseline_lead_time_hours, resid_baseline_model_dir_name,
         first_init_time_strings, last_init_time_strings,
         nwp_directory_names, target_dir_name, output_dir_name):
    """Prepares example files for faster NN-training.

    This is effectively the main method.
    
    :param nwp_lead_times_hours: See documentation at top of this script.
    :param nwp_model_names: Same.
    :param nwp_model_to_field_names: Same.
    :param nwp_normalization_file_name: Same.
    :param nwp_resid_norm_file_name: Same.
    :param nwp_use_quantile_norm: Same.
    :param backup_nwp_model_name: Same.
    :param backup_nwp_dir_name: Same.
    :param target_lead_time_hours: Same.
    :param target_field_names: Same.
    :param target_lag_times_hours: Same.
    :param target_normalization_file_name: Same.
    :param target_resid_norm_file_name: Same.
    :param targets_use_quantile_norm: Same.
    :param recent_bias_init_time_lags_hours: Same.
    :param recent_bias_lead_times_hours: Same.
    :param nbm_constant_field_names: Same.
    :param nbm_constant_file_name: Same.
    :param compare_to_baseline_in_loss: Same.
    :param sentinel_value: Same.
    :param patch_size_2pt5km_pixels: Same.
    :param patch_buffer_size_2pt5km_pixels: Same.
    :param patch_start_row_2pt5km: Same.
    :param patch_start_column_2pt5km: Same.
    :param patch_start_column_2pt5km: Same.
    :param require_all_predictors: Same.
    :param do_residual_prediction: Same.
    :param resid_baseline_model_name: Same.
    :param resid_baseline_lead_time_hours: Same.
    :param resid_baseline_model_dir_name: Same.
    :param first_init_time_strings: Same.
    :param last_init_time_strings: Same.
    :param nwp_directory_names: Same.
    :param target_dir_name: Same.
    :param output_dir_name: Same.
    """

    if nwp_resid_norm_file_name == '':
        nwp_resid_norm_file_name = None
    if target_normalization_file_name == '':
        target_normalization_file_name = None
    if target_resid_norm_file_name == '':
        target_resid_norm_file_name = None
    if resid_baseline_model_name == '':
        resid_baseline_model_name = None
    if resid_baseline_model_dir_name == '':
        resid_baseline_model_dir_name = None
    if resid_baseline_lead_time_hours <= 0:
        resid_baseline_lead_time_hours = None

    if patch_size_2pt5km_pixels < 0:
        patch_size_2pt5km_pixels = None
    if patch_buffer_size_2pt5km_pixels < 0:
        patch_buffer_size_2pt5km_pixels = None
    if patch_start_row_2pt5km < 0:
        patch_start_row_2pt5km = None
    if patch_start_column_2pt5km < 0:
        patch_start_column_2pt5km = None

    if (
            patch_size_2pt5km_pixels is None
            or patch_buffer_size_2pt5km_pixels is None
            or patch_start_row_2pt5km is None
            or patch_start_column_2pt5km is None
    ):
        patch_size_2pt5km_pixels = None
        patch_buffer_size_2pt5km_pixels = None
        patch_start_row_2pt5km = None
        patch_start_column_2pt5km = None

    if nbm_constant_file_name == '':
        nbm_constant_file_name = None
        nbm_constant_field_names = []
    if len(nbm_constant_field_names) == 1 and nbm_constant_field_names[0] == '':
        nbm_constant_file_name = None
        nbm_constant_field_names = []
    if len(target_lag_times_hours) == 1 and target_lag_times_hours[0] < 0:
        target_lag_times_hours = None

    if (
            len(recent_bias_init_time_lags_hours) == 1 and
            recent_bias_init_time_lags_hours[0] < 0
    ):
        recent_bias_init_time_lags_hours = None

    if (
            len(recent_bias_lead_times_hours) == 1 and
            recent_bias_lead_times_hours[0] < 0
    ):
        recent_bias_lead_times_hours = None

    nwp_model_to_dir_name = _process_nwp_directories(
        nwp_directory_names=nwp_directory_names,
        nwp_model_names=nwp_model_names
    )

    first_init_times_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, TIME_FORMAT)
        for t in first_init_time_strings
    ], dtype=int)
    last_init_times_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, TIME_FORMAT)
        for t in last_init_time_strings
    ], dtype=int)

    option_dict = {
        nn_utils.FIRST_INIT_TIMES_KEY: first_init_times_unix_sec,
        nn_utils.LAST_INIT_TIMES_KEY: last_init_times_unix_sec,
        nn_utils.NWP_LEAD_TIMES_KEY: nwp_lead_times_hours,
        nn_utils.NWP_MODEL_TO_DIR_KEY: nwp_model_to_dir_name,
        nn_utils.NWP_MODEL_TO_FIELDS_KEY: nwp_model_to_field_names,
        nn_utils.NWP_NORM_FILE_KEY: nwp_normalization_file_name,
        nn_utils.NWP_RESID_NORM_FILE_KEY: nwp_resid_norm_file_name,
        nn_utils.NWP_USE_QUANTILE_NORM_KEY: nwp_use_quantile_norm,
        nn_utils.BACKUP_NWP_MODEL_KEY: backup_nwp_model_name,
        nn_utils.BACKUP_NWP_DIR_KEY: backup_nwp_dir_name,
        nn_utils.TARGET_LEAD_TIME_KEY: target_lead_time_hours,
        nn_utils.TARGET_FIELDS_KEY: target_field_names,
        nn_utils.TARGET_LAG_TIMES_KEY: target_lag_times_hours,
        nn_utils.TARGET_DIR_KEY: target_dir_name,
        nn_utils.TARGET_NORM_FILE_KEY: target_normalization_file_name,
        nn_utils.TARGET_RESID_NORM_FILE_KEY: target_resid_norm_file_name,
        nn_utils.TARGETS_USE_QUANTILE_NORM_KEY: targets_use_quantile_norm,
        nn_utils.RECENT_BIAS_LAG_TIMES_KEY: recent_bias_init_time_lags_hours,
        nn_utils.RECENT_BIAS_LEAD_TIMES_KEY: recent_bias_lead_times_hours,
        nn_utils.NBM_CONSTANT_FIELDS_KEY: nbm_constant_field_names,
        nn_utils.NBM_CONSTANT_FILE_KEY: nbm_constant_file_name,
        nn_utils.COMPARE_TO_BASELINE_IN_LOSS_KEY: compare_to_baseline_in_loss,
        nn_utils.BATCH_SIZE_KEY: 32,
        nn_utils.SENTINEL_VALUE_KEY: sentinel_value,
        nn_utils.DO_RESIDUAL_PREDICTION_KEY: do_residual_prediction,
        nn_utils.RESID_BASELINE_MODEL_KEY: resid_baseline_model_name,
        nn_utils.RESID_BASELINE_LEAD_TIME_KEY: resid_baseline_lead_time_hours,
        nn_utils.RESID_BASELINE_MODEL_DIR_KEY: resid_baseline_model_dir_name,
        nn_utils.PATCH_SIZE_KEY: patch_size_2pt5km_pixels,
        nn_utils.PATCH_BUFFER_SIZE_KEY: patch_buffer_size_2pt5km_pixels,
        nn_utils.PATCH_START_ROW_KEY: patch_start_row_2pt5km,
        nn_utils.PATCH_START_COLUMN_KEY: patch_start_column_2pt5km,
        nn_utils.REQUIRE_ALL_PREDICTORS_KEY: require_all_predictors
    }

    nn_metafile_name = nn_utils.find_metafile(
        model_file_name='{0:s}/model.weights.h5'.format(output_dir_name),
        raise_error_if_missing=False
    )

    print('Writing metadata to: "{0:s}"...'.format(nn_metafile_name))
    nn_utils.write_metafile(
        pickle_file_name=nn_metafile_name,
        num_epochs=100,
        use_exp_moving_average_with_decay=False,
        num_training_batches_per_epoch=32,
        training_option_dict=option_dict,
        num_validation_batches_per_epoch=16,
        validation_option_dict=option_dict,
        loss_function_string='mse',
        optimizer_function_string='keras.optimizers.AdamW()',
        metric_function_strings=['mse'],
        u_net_architecture_dict=None,
        chiu_net_architecture_dict=None,
        chiu_net_pp_architecture_dict=None,
        chiu_next_pp_architecture_dict=option_dict,
        plateau_patience_epochs=10,
        plateau_learning_rate_multiplier=0.6,
        early_stopping_patience_epochs=50,
        patch_overlap_fast_gen_2pt5km_pixels=144
    )

    init_times_unix_sec = nn_utils.find_relevant_init_times(
        first_time_by_period_unix_sec=first_init_times_unix_sec,
        last_time_by_period_unix_sec=last_init_times_unix_sec,
        nwp_model_names=nwp_model_names
    )

    for this_init_time_unix_sec in init_times_unix_sec:
        print(SEPARATOR_STRING)
        data_dict = None

        try:
            data_dict = nn_training_simple.create_data(
                option_dict=option_dict,
                init_time_unix_sec=this_init_time_unix_sec
            )
        except Exception as this_exception:
            print(this_exception)
            pass

        if data_dict is None:
            continue

        output_file_name = example_io.find_file(
            directory_name=output_dir_name,
            init_time_unix_sec=this_init_time_unix_sec,
            raise_error_if_missing=False
        )

        print('Writing NN example to: "{0:s}"...'.format(output_file_name))
        example_io.write_file(
            data_dict=data_dict, npz_file_name=output_file_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        nwp_lead_times_hours=numpy.array(
            getattr(INPUT_ARG_OBJECT, NWP_LEAD_TIMES_ARG_NAME), dtype=int
        ),
        nwp_model_names=getattr(INPUT_ARG_OBJECT, NWP_MODELS_ARG_NAME),
        nwp_model_to_field_names=json.loads(getattr(
            INPUT_ARG_OBJECT, NWP_MODEL_TO_FIELDS_ARG_NAME
        )),
        nwp_normalization_file_name=getattr(
            INPUT_ARG_OBJECT, NWP_NORMALIZATION_FILE_ARG_NAME
        ),
        nwp_resid_norm_file_name=getattr(
            INPUT_ARG_OBJECT, NWP_RESID_NORM_FILE_ARG_NAME
        ),
        nwp_use_quantile_norm=bool(getattr(
            INPUT_ARG_OBJECT, NWP_USE_QUANTILE_NORM_ARG_NAME
        )),
        backup_nwp_model_name=getattr(
            INPUT_ARG_OBJECT, BACKUP_NWP_MODEL_ARG_NAME
        ),
        backup_nwp_dir_name=getattr(
            INPUT_ARG_OBJECT, BACKUP_NWP_DIR_ARG_NAME
        ),
        target_lead_time_hours=getattr(
            INPUT_ARG_OBJECT, TARGET_LEAD_TIME_ARG_NAME
        ),
        target_field_names=getattr(INPUT_ARG_OBJECT, TARGET_FIELDS_ARG_NAME),
        target_lag_times_hours=numpy.array(
            getattr(INPUT_ARG_OBJECT, TARGET_LAG_TIMES_ARG_NAME), dtype=int
        ),
        target_normalization_file_name=getattr(
            INPUT_ARG_OBJECT, TARGET_NORMALIZATION_FILE_ARG_NAME
        ),
        target_resid_norm_file_name=getattr(
            INPUT_ARG_OBJECT, TARGET_RESID_NORM_FILE_ARG_NAME
        ),
        targets_use_quantile_norm=bool(getattr(
            INPUT_ARG_OBJECT, TARGETS_USE_QUANTILE_NORM_ARG_NAME
        )),
        recent_bias_init_time_lags_hours=numpy.array(
            getattr(INPUT_ARG_OBJECT, RECENT_BIAS_LAG_TIMES_ARG_NAME), dtype=int
        ),
        recent_bias_lead_times_hours=numpy.array(
            getattr(INPUT_ARG_OBJECT, RECENT_BIAS_LEAD_TIMES_ARG_NAME),
            dtype=int
        ),
        nbm_constant_field_names=getattr(
            INPUT_ARG_OBJECT, NBM_CONSTANT_FIELDS_ARG_NAME
        ),
        nbm_constant_file_name=getattr(
            INPUT_ARG_OBJECT, NBM_CONSTANT_FILE_ARG_NAME
        ),
        compare_to_baseline_in_loss=bool(getattr(
            INPUT_ARG_OBJECT, COMPARE_TO_BASELINE_ARG_NAME
        )),
        sentinel_value=getattr(INPUT_ARG_OBJECT, SENTINEL_VALUE_ARG_NAME),
        patch_size_2pt5km_pixels=getattr(INPUT_ARG_OBJECT, PATCH_SIZE_ARG_NAME),
        patch_buffer_size_2pt5km_pixels=getattr(
            INPUT_ARG_OBJECT, PATCH_BUFFER_SIZE_ARG_NAME
        ),
        patch_start_row_2pt5km=getattr(
            INPUT_ARG_OBJECT, PATCH_START_ROW_ARG_NAME
        ),
        patch_start_column_2pt5km=getattr(
            INPUT_ARG_OBJECT, PATCH_START_COLUMN_ARG_NAME
        ),
        require_all_predictors=bool(getattr(
            INPUT_ARG_OBJECT, REQUIRE_ALL_PREDICTORS_ARG_NAME
        )),
        do_residual_prediction=bool(getattr(
            INPUT_ARG_OBJECT, DO_RESIDUAL_PREDICTION_ARG_NAME
        )),
        resid_baseline_model_name=getattr(
            INPUT_ARG_OBJECT, RESID_BASELINE_MODEL_ARG_NAME
        ),
        resid_baseline_lead_time_hours=getattr(
            INPUT_ARG_OBJECT, RESID_BASELINE_LEAD_TIME_ARG_NAME
        ),
        resid_baseline_model_dir_name=getattr(
            INPUT_ARG_OBJECT, RESID_BASELINE_MODEL_DIR_ARG_NAME
        ),
        first_init_time_strings=getattr(
            INPUT_ARG_OBJECT, FIRST_TRAINING_TIMES_ARG_NAME
        ),
        last_init_time_strings=getattr(
            INPUT_ARG_OBJECT, LAST_TRAINING_TIMES_ARG_NAME
        ),
        nwp_directory_names=getattr(
            INPUT_ARG_OBJECT, TRAINING_NWP_DIRS_ARG_NAME
        ),
        target_dir_name=getattr(
            INPUT_ARG_OBJECT, TRAINING_TARGET_DIR_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
