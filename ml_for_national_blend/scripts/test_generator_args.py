"""Contains list of input arguments for test_generator.py."""

from ml_for_national_blend.utils import urma_utils
from ml_for_national_blend.utils import nwp_model_utils

OUTPUT_DIR_ARG_NAME = 'output_model_dir_name'

NWP_LEAD_TIMES_ARG_NAME = 'nwp_lead_times_hours'
NWP_MODELS_ARG_NAME = 'nwp_model_names'
NWP_MODEL_TO_FIELDS_ARG_NAME = 'nwp_model_to_field_names'
NWP_NORMALIZATION_FILE_ARG_NAME = 'nwp_normalization_file_name'
NWP_USE_QUANTILE_NORM_ARG_NAME = 'nwp_use_quantile_norm'
BACKUP_NWP_MODEL_ARG_NAME = 'backup_nwp_model_name'
BACKUP_NWP_DIR_ARG_NAME = 'backup_nwp_dir_name'
TARGET_LEAD_TIME_ARG_NAME = 'target_lead_time_hours'
TARGET_FIELDS_ARG_NAME = 'target_field_names'
TARGET_LAG_TIMES_ARG_NAME = 'target_lag_times_hours'
TARGET_NORMALIZATION_FILE_ARG_NAME = 'target_normalization_file_name'
TARGETS_USE_QUANTILE_NORM_ARG_NAME = 'targets_use_quantile_norm'
RECENT_BIAS_LAG_TIMES_ARG_NAME = 'recent_bias_init_time_lags_hours'
RECENT_BIAS_LEAD_TIMES_ARG_NAME = 'recent_bias_lead_times_hours'
NBM_CONSTANT_FIELDS_ARG_NAME = 'nbm_constant_field_names'
NBM_CONSTANT_FILE_ARG_NAME = 'nbm_constant_file_name'
BATCH_SIZE_ARG_NAME = 'num_examples_per_batch'
SENTINEL_VALUE_ARG_NAME = 'sentinel_value'
PATCH_SIZE_ARG_NAME = 'patch_size_2pt5km_pixels'
PATCH_BUFFER_SIZE_ARG_NAME = 'patch_buffer_size_2pt5km_pixels'
PATCH_START_ROW_ARG_NAME = 'patch_start_row_2pt5km'
PATCH_START_COLUMN_ARG_NAME = 'patch_start_column_2pt5km'
USE_FAST_PATCH_GENERATOR_ARG_NAME = 'use_fast_patch_generator'
PATCH_OVERLAP_SIZE_ARG_NAME = 'patch_overlap_size_2pt5km_pixels'
REQUIRE_ALL_PREDICTORS_ARG_NAME = 'require_all_predictors'

DO_RESIDUAL_PREDICTION_ARG_NAME = 'do_residual_prediction'
RESID_BASELINE_MODEL_ARG_NAME = 'resid_baseline_model_name'
RESID_BASELINE_LEAD_TIME_ARG_NAME = 'resid_baseline_lead_time_hours'
RESID_BASELINE_MODEL_DIR_ARG_NAME = 'resid_baseline_model_dir_name'

FIRST_TRAINING_TIMES_ARG_NAME = 'first_init_time_strings_for_training'
LAST_TRAINING_TIMES_ARG_NAME = 'last_init_time_strings_for_training'
TRAINING_NWP_DIRS_ARG_NAME = 'nwp_dir_names_for_training'
TRAINING_TARGET_DIR_ARG_NAME = 'target_dir_name_for_training'

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Trained model will be saved here.'
)

NWP_LEAD_TIMES_HELP_STRING = (
    'List of lead times to use in predictors.  The same set of lead times will '
    'be used for every NWP model.'
)
NWP_MODELS_HELP_STRING = (
    'List of NWP models to use in predictors.  Each model name must be in the '
    'following list:\n{0:s}'
).format(
    str(nwp_model_utils.ALL_MODEL_NAMES_WITH_ENSEMBLE)
)
NWP_MODEL_TO_FIELDS_HELP_STRING = (
    'Dictionary, where each key is the name of an NWP model (from the list '
    '{0:s}) and the corresponding value is a list of predictor variables '
    '(fields).  The dictionary must be formatted as a JSON string.  Each field '
    'name must be in the following list:\n{1:s}'
).format(
    NWP_MODELS_ARG_NAME,
    str(nwp_model_utils.ALL_FIELD_NAMES)
)
NWP_NORMALIZATION_FILE_HELP_STRING = (
    'Path to normalization file for NWP predictors (will be read by '
    '`nwp_model_io.read_normalization_file`).  Use this argument only if you '
    'want to normalize predictors on the fly -- i.e., `{0:s}` contains '
    'unnormalized data, but you want to train with normalized data.'
).format(
    TRAINING_NWP_DIRS_ARG_NAME
)
NWP_USE_QUANTILE_NORM_HELP_STRING = (
    'Boolean flag.  If 1, will do two-step normalization: conversion to '
    'quantiles and then normal distribution (using inverse CDF).  If 0, will '
    'do simple z-score normalization.'
)
BACKUP_NWP_MODEL_HELP_STRING = (
    'Name of backup NWP model, used to fill missing data.'
)
BACKUP_NWP_DIR_HELP_STRING = (
    'Path to data directory for backup NWP model.  Files therein will be found '
    'by `interp_nwp_model_io.find_file`.'
)
TARGET_LEAD_TIME_HELP_STRING = 'Lead time for target variables.'
TARGET_FIELDS_HELP_STRING = (
    'List of target fields (i.e., variables to predict).  Each field name must '
    'be in the following list:\n{0:s}'
).format(
    str(urma_utils.ALL_FIELD_NAMES)
)
TARGET_LAG_TIMES_HELP_STRING = (
    'List of lag times for target fields used in the predictors.  If you do '
    'not want to use target fields in the predictors, make this a 1-element '
    'list with a negative number -- for example, [-1].'
)
TARGET_NORMALIZATION_FILE_HELP_STRING = (
    'Path to normalization file for target variables (will be read by '
    '`urma_io.read_normalization_file`).  Use this argument only if you want '
    'to normalize targets on the fly -- i.e., `{0:s}` contains unnormalized '
    'data, but you want to train with normalized targets.'
).format(
    TRAINING_TARGET_DIR_ARG_NAME
)
TARGETS_USE_QUANTILE_NORM_HELP_STRING = 'Same as {0:s} but for targets.'.format(
    NWP_USE_QUANTILE_NORM_ARG_NAME
)
RECENT_BIAS_LAG_TIMES_HELP_STRING = (
    '1-D list of lag times for recent NWP bias.  If you do not want predictors '
    'to include recent NWP bias, leave this argument alone.'
)
RECENT_BIAS_LEAD_TIMES_HELP_STRING = (
    '1-D list of lead times for recent NWP bias (with same length as `{0:s}`).'
    '  If you do not want predictors to include recent NWP bias, leave this '
    'argument alone.'
).format(
    RECENT_BIAS_LAG_TIMES_ARG_NAME
)
NBM_CONSTANT_FIELDS_HELP_STRING = (
    'List of NBM constant fields to be used as predictors.  Each must be '
    'accepted by `nbm_constant_utils.check_field_name`.'
)
NBM_CONSTANT_FILE_HELP_STRING = (
    'Path to file with NBM constant fields (readable by '
    '`nbm_constant_io.read_file`).  If you do not want NBM-constant '
    'predictors, make this an empty string.'
)
BATCH_SIZE_HELP_STRING = 'Number of data examples per batch.'
SENTINEL_VALUE_HELP_STRING = (
    'All NaN predictors will be replaced with this value.'
)

PATCH_SIZE_HELP_STRING = (
    'Patch size, in units of 2.5-km pixels.  For example, if {0:s} = 448, then '
    'grid dimensions at the finest resolution (2.5 km) are 448 x 448.  If you '
    'want to train with the full grid -- and not the patchwise approach -- '
    'make this argument negative.'
).format(
    PATCH_SIZE_ARG_NAME
)
PATCH_BUFFER_SIZE_HELP_STRING = (
    '[used only if {0:s} is positive] Buffer between the outer domain (used '
    'for predictors) and the inner domain (used to penalize predictions in '
    'loss function).  This must be a non-negative integer.'
).format(
    PATCH_SIZE_ARG_NAME
)
PATCH_START_ROW_HELP_STRING = (
    '[used only if {0:s} is positive] Indicates start row for training patch.  '
    'For example, if {1:s} == k, the patch will always start at the [k]th row '
    'on the 2.5-km grid.  If you want each patch to be at a different '
    'location, leave this alone.'
).format(
    PATCH_SIZE_ARG_NAME, PATCH_START_ROW_ARG_NAME
)
PATCH_START_COLUMN_HELP_STRING = 'Same as {0:s} but for columns.'.format(
    PATCH_START_ROW_ARG_NAME
)
USE_FAST_PATCH_GENERATOR_HELP_STRING = (
    '[used only if {0:s} is positive] Boolean flag.  If 1, will use fast '
    'version of patch-generator.  I HIGHLY RECOMMEND LEAVING THIS ARGUMENT AT 1.'
).format(
    PATCH_SIZE_ARG_NAME
)
PATCH_OVERLAP_SIZE_HELP_STRING = (
    '[used only if {0:s} == 1] Overlap between adjacent patches, measured in '
    'number of pixels on the finest-resolution (2.5-km) grid.'
).format(
    USE_FAST_PATCH_GENERATOR_ARG_NAME
)
REQUIRE_ALL_PREDICTORS_HELP_STRING = (
    'Boolean flag.  If 1, only data samples where all NWP predictors are found '
    'will be used.  If 0, any data sample where *some* NWP predictors are '
    'found will be used.'
)

DO_RESIDUAL_PREDICTION_HELP_STRING = (
    'Boolean flag.  If True, the NN is trained to predict a residual -- i.e., '
    'the departure between URMA truth and a single NWP forecast.  If False, '
    'the NN is trained to predict the URMA target fields directly.'
)
RESID_BASELINE_MODEL_HELP_STRING = (
    'Name of NWP model used to generate residual baseline fields.  If '
    'do_residual_prediction == False, make this argument None.'
)
RESID_BASELINE_LEAD_TIME_HELP_STRING = (
    'Lead time used to generate residual baseline fields.  If '
    'do_residual_prediction == False, make this argument None.'
)
RESID_BASELINE_MODEL_DIR_HELP_STRING = (
    'Directory path for residual baseline fields.  Within this directory,'
    'relevant files will be found by `interp_nwp_model_io.find_file`.'
)

FIRST_TRAINING_TIMES_HELP_STRING = (
    'length-P list (where P = number of continuous training periods), where '
    'each item is the start of a continuous training period (format '
    '"yyyy-mm-dd-HH").'
)
LAST_TRAINING_TIMES_HELP_STRING = (
    'Same as {0:s} but for ends of continuous training periods.'
).format(
    FIRST_TRAINING_TIMES_ARG_NAME
)

TRAINING_NWP_DIRS_HELP_STRING = (
    'This argument can be formatted in two ways.  Option 1: a single path.  In '
    'this case, for each model, "<model_name>/processed/interp_to_nbm_grid" '
    'will be added to the end.  Option 2: a list of paths, with the same '
    'length as {0:s}.  Relevant files in any directory will be found by '
    '`nwp_model_io.find_file` and read by `nwp_model_io.read_file`.'
).format(
    NWP_MODELS_ARG_NAME
)
TRAINING_TARGET_DIR_HELP_STRING = (
    'Path to directory with target data for training period.  Relevant files '
    'in this directory will be found by `urma_io.find_file` and read by '
    '`urma_io.read_file`.'
)


def add_input_args(parser_object):
    """Adds input args to ArgumentParser object.

    :param parser_object: Instance of `argparse.ArgumentParser` (may already
        contain some input args).
    :return: parser_object: Same as input but with new args added.
    """

    parser_object.add_argument(
        '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
        help=OUTPUT_DIR_HELP_STRING
    )

    parser_object.add_argument(
        '--' + NWP_LEAD_TIMES_ARG_NAME, type=int, nargs='+', required=True,
        help=NWP_LEAD_TIMES_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NWP_MODELS_ARG_NAME, type=str, nargs='+', required=True,
        help=NWP_MODELS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NWP_MODEL_TO_FIELDS_ARG_NAME, type=str, required=True,
        help=NWP_MODEL_TO_FIELDS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NWP_NORMALIZATION_FILE_ARG_NAME, type=str, required=False,
        default='', help=NWP_NORMALIZATION_FILE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NWP_USE_QUANTILE_NORM_ARG_NAME, type=int, required=False,
        default=1, help=NWP_USE_QUANTILE_NORM_HELP_STRING
    )
    parser_object.add_argument(
        '--' + BACKUP_NWP_MODEL_ARG_NAME, type=str, required=True,
        help=BACKUP_NWP_MODEL_HELP_STRING
    )
    parser_object.add_argument(
        '--' + BACKUP_NWP_DIR_ARG_NAME, type=str, required=True,
        help=BACKUP_NWP_DIR_HELP_STRING
    )
    parser_object.add_argument(
        '--' + TARGET_LEAD_TIME_ARG_NAME, type=int, required=True,
        help=TARGET_LEAD_TIME_HELP_STRING
    )
    parser_object.add_argument(
        '--' + TARGET_FIELDS_ARG_NAME, type=str, nargs='+', required=True,
        help=TARGET_FIELDS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + TARGET_LAG_TIMES_ARG_NAME, type=int, nargs='+', required=True,
        help=TARGET_LAG_TIMES_HELP_STRING
    )
    parser_object.add_argument(
        '--' + TARGET_NORMALIZATION_FILE_ARG_NAME, type=str, required=False,
        default='', help=TARGET_NORMALIZATION_FILE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + TARGETS_USE_QUANTILE_NORM_ARG_NAME, type=int, required=False,
        default=1, help=TARGETS_USE_QUANTILE_NORM_HELP_STRING
    )
    parser_object.add_argument(
        '--' + RECENT_BIAS_LAG_TIMES_ARG_NAME, type=int, nargs='+',
        required=False, default=[-1], help=RECENT_BIAS_LAG_TIMES_HELP_STRING
    )
    parser_object.add_argument(
        '--' + RECENT_BIAS_LEAD_TIMES_ARG_NAME, type=int, nargs='+',
        required=False, default=[-1], help=RECENT_BIAS_LEAD_TIMES_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NBM_CONSTANT_FIELDS_ARG_NAME, type=str, nargs='+',
        required=False, default=[''], help=NBM_CONSTANT_FIELDS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NBM_CONSTANT_FILE_ARG_NAME, type=str, required=True,
        help=NBM_CONSTANT_FILE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + BATCH_SIZE_ARG_NAME, type=int, required=True,
        help=BATCH_SIZE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + SENTINEL_VALUE_ARG_NAME, type=float, required=False,
        default=-10., help=SENTINEL_VALUE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + PATCH_SIZE_ARG_NAME, type=int, required=True,
        help=PATCH_SIZE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + PATCH_BUFFER_SIZE_ARG_NAME, type=int, required=False, default=0,
        help=PATCH_BUFFER_SIZE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + PATCH_START_ROW_ARG_NAME, type=int, required=False, default=-1,
        help=PATCH_START_ROW_HELP_STRING
    )
    parser_object.add_argument(
        '--' + PATCH_START_COLUMN_ARG_NAME, type=int, required=False,
        default=-1, help=PATCH_START_COLUMN_HELP_STRING
    )
    parser_object.add_argument(
        '--' + USE_FAST_PATCH_GENERATOR_ARG_NAME, type=int, required=False,
        default=1, help=USE_FAST_PATCH_GENERATOR_HELP_STRING
    )
    parser_object.add_argument(
        '--' + PATCH_OVERLAP_SIZE_ARG_NAME, type=int, required=False,
        default=-1, help=PATCH_OVERLAP_SIZE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + REQUIRE_ALL_PREDICTORS_ARG_NAME, type=int, required=True,
        help=REQUIRE_ALL_PREDICTORS_HELP_STRING
    )

    parser_object.add_argument(
        '--' + DO_RESIDUAL_PREDICTION_ARG_NAME, type=int, required=True,
        help=DO_RESIDUAL_PREDICTION_HELP_STRING
    )
    parser_object.add_argument(
        '--' + RESID_BASELINE_MODEL_ARG_NAME, type=str, required=False,
        default='', help=RESID_BASELINE_MODEL_HELP_STRING
    )
    parser_object.add_argument(
        '--' + RESID_BASELINE_LEAD_TIME_ARG_NAME, type=int, required=False,
        default=-1, help=RESID_BASELINE_LEAD_TIME_HELP_STRING
    )
    parser_object.add_argument(
        '--' + RESID_BASELINE_MODEL_DIR_ARG_NAME, type=str, required=False,
        default='', help=RESID_BASELINE_MODEL_DIR_HELP_STRING
    )

    parser_object.add_argument(
        '--' + FIRST_TRAINING_TIMES_ARG_NAME, type=str, nargs='+',
        required=True, help=FIRST_TRAINING_TIMES_HELP_STRING
    )
    parser_object.add_argument(
        '--' + LAST_TRAINING_TIMES_ARG_NAME, type=str, nargs='+',
        required=True, help=LAST_TRAINING_TIMES_HELP_STRING
    )
    parser_object.add_argument(
        '--' + TRAINING_NWP_DIRS_ARG_NAME, type=str, nargs='+', required=True,
        help=TRAINING_NWP_DIRS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + TRAINING_TARGET_DIR_ARG_NAME, type=str, required=True,
        help=TRAINING_TARGET_DIR_HELP_STRING
    )

    return parser_object
