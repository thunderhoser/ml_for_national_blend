"""Contains list of input arguments for training a neural net."""

import os
import sys

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import urma_utils
import nwp_model_utils

TEMPLATE_FILE_ARG_NAME = 'input_template_file_name'
OUTPUT_DIR_ARG_NAME = 'output_model_dir_name'

NWP_LEAD_TIMES_ARG_NAME = 'nwp_lead_times_hours'
NWP_MODELS_ARG_NAME = 'nwp_model_names'
NWP_FIELDS_ARG_NAME = 'nwp_field_names'
NWP_NORMALIZATION_FILE_ARG_NAME = 'nwp_normalization_file_name'
TARGET_LEAD_TIME_ARG_NAME = 'target_lead_time_hours'
TARGET_FIELDS_ARG_NAME = 'target_field_names'
TARGET_NORMALIZATION_FILE_ARG_NAME = 'target_normalization_file_name'
BATCH_SIZE_ARG_NAME = 'num_examples_per_batch'
SENTINEL_VALUE_ARG_NAME = 'sentinel_value'

TRAINING_TIME_LIMITS_ARG_NAME = 'init_time_limit_strings_for_training'
TRAINING_NWP_DIRS_ARG_NAME = 'nwp_dir_names_for_training'
TRAINING_TARGET_DIR_ARG_NAME = 'target_dir_name_for_training'

VALIDATION_TIME_LIMITS_ARG_NAME = 'init_time_limit_strings_for_validation'
VALIDATION_NWP_DIRS_ARG_NAME = 'nwp_dir_names_for_validation'
VALIDATION_TARGET_DIR_ARG_NAME = 'target_dir_name_for_validation'

NUM_EPOCHS_ARG_NAME = 'num_epochs'
NUM_TRAINING_BATCHES_ARG_NAME = 'num_training_batches_per_epoch'
NUM_VALIDATION_BATCHES_ARG_NAME = 'num_validation_batches_per_epoch'

PLATEAU_PATIENCE_ARG_NAME = 'plateau_patience_epochs'
PLATEAU_MULTIPLIER_ARG_NAME = 'plateau_learning_rate_multiplier'
EARLY_STOPPING_PATIENCE_ARG_NAME = 'early_stopping_patience_epochs'

TEMPLATE_FILE_HELP_STRING = (
    'Path to template file, containing model architecture.  This will be read '
    'by `neural_net.read_model`.'
)
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
    str(nwp_model_utils.ALL_MODEL_NAMES)
)
NWP_FIELDS_HELP_STRING = (
    'List of fields to use in predictors.  The same set of fields will be used '
    'for every NWP model.  Each field name must be in the following list:'
    '\n{0:s}'
).format(
    str(nwp_model_utils.ALL_FIELD_NAMES)
)
NWP_NORMALIZATION_FILE_HELP_STRING = (
    'Path to normalization file for NWP predictors (will be read by '
    '`nwp_model_io.read_normalization_file`).  Use this argument only if you '
    'want to normalize predictors on the fly -- i.e., `{0:s}` and `{1:s}` '
    'contain unnormalized data, but you want to train with normalized data.'
).format(
    TRAINING_NWP_DIRS_ARG_NAME, VALIDATION_NWP_DIRS_ARG_NAME
)
TARGET_LEAD_TIME_HELP_STRING = 'Lead time for target variables.'
TARGET_FIELDS_HELP_STRING = (
    'List of target fields (i.e., variables to predict).  Each field name must '
    'be in the following list:\n{0:s}'
).format(
    str(urma_utils.ALL_FIELD_NAMES)
)
TARGET_NORMALIZATION_FILE_HELP_STRING = (
    'Path to normalization file for target variables (will be read by '
    '`urma_io.read_normalization_file`).  Use this argument only if you want '
    'to normalize targets on the fly -- i.e., `{0:s}` and `{1:s}` contain '
    'unnormalized data, but you want to train with normalized targets.'
).format(
    TRAINING_TARGET_DIR_ARG_NAME, VALIDATION_TARGET_DIR_ARG_NAME
)
BATCH_SIZE_HELP_STRING = 'Number of data examples per batch.'
SENTINEL_VALUE_HELP_STRING = (
    'All NaN predictors will be replaced with this value.'
)

TRAINING_TIME_LIMITS_HELP_STRING = (
    'Length-2 list with first and last NWP-model runs (init times in format '
    '"yyyy-mm-dd-HH") to be used for training.'
)
TRAINING_NWP_DIRS_HELP_STRING = (
    'List of directory paths with NWP data for training period.  This list '
    'must have the same length as `{0:s}`.  Relevant files in each directory '
    'will be found by `nwp_model_io.find_file` and read by '
    '`nwp_model_io.read_file`.'
).format(
    NWP_MODELS_ARG_NAME
)
TRAINING_TARGET_DIR_HELP_STRING = (
    'Path to directory with target data for training period.  Relevant files '
    'in this directory will be found by `urma_io.find_file` and read by '
    '`urma_io.read_file`.'
)

VALIDATION_TIME_LIMITS_HELP_STRING = (
    'Same as `{0:s}` but for validation data.'
).format(
    TRAINING_TIME_LIMITS_ARG_NAME
)
VALIDATION_NWP_DIRS_HELP_STRING = (
    'Same as `{0:s}` but for validation data.'
).format(
    TRAINING_NWP_DIRS_ARG_NAME
)
VALIDATION_TARGET_DIR_HELP_STRING = (
    'Same as `{0:s}` but for validation data.'
).format(
    TRAINING_TARGET_DIR_ARG_NAME
)

NUM_EPOCHS_HELP_STRING = 'Number of epochs.'
NUM_TRAINING_BATCHES_HELP_STRING = 'Number of training batches per epoch.'
NUM_VALIDATION_BATCHES_HELP_STRING = 'Number of validation batches per epoch.'

PLATEAU_PATIENCE_HELP_STRING = (
    'Training will be deemed to have reached "plateau" if validation loss has '
    'not decreased in the last N epochs, where N = {0:s}.'
).format(PLATEAU_PATIENCE_ARG_NAME)

PLATEAU_MULTIPLIER_HELP_STRING = (
    'If training reaches "plateau," learning rate will be multiplied by this '
    'value in range (0, 1).'
)
EARLY_STOPPING_PATIENCE_HELP_STRING = (
    'Training will be stopped early if validation loss has not decreased in '
    'the last N epochs, where N = {0:s}.'
).format(EARLY_STOPPING_PATIENCE_ARG_NAME)


def add_input_args(parser_object):
    """Adds input args to ArgumentParser object.

    :param parser_object: Instance of `argparse.ArgumentParser` (may already
        contain some input args).
    :return: parser_object: Same as input but with new args added.
    """

    parser_object.add_argument(
        '--' + TEMPLATE_FILE_ARG_NAME, type=str, required=True,
        help=TEMPLATE_FILE_HELP_STRING
    )
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
        '--' + NWP_FIELDS_ARG_NAME, type=str, nargs='+', required=True,
        help=NWP_FIELDS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NWP_NORMALIZATION_FILE_ARG_NAME, type=str, required=False,
        default='', help=NWP_NORMALIZATION_FILE_HELP_STRING
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
        '--' + TARGET_NORMALIZATION_FILE_ARG_NAME, type=str, required=False,
        default='', help=TARGET_NORMALIZATION_FILE_HELP_STRING
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
        '--' + TRAINING_TIME_LIMITS_ARG_NAME, type=str, nargs=2, required=True,
        help=TRAINING_TIME_LIMITS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + TRAINING_NWP_DIRS_ARG_NAME, type=str, nargs='+', required=True,
        help=TRAINING_NWP_DIRS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + TRAINING_TARGET_DIR_ARG_NAME, type=str, required=True,
        help=TRAINING_TARGET_DIR_HELP_STRING
    )

    parser_object.add_argument(
        '--' + VALIDATION_TIME_LIMITS_ARG_NAME, type=str, nargs=2, required=True,
        help=VALIDATION_TIME_LIMITS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + VALIDATION_NWP_DIRS_ARG_NAME, type=str, nargs='+', required=True,
        help=VALIDATION_NWP_DIRS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + VALIDATION_TARGET_DIR_ARG_NAME, type=str, required=True,
        help=VALIDATION_TARGET_DIR_HELP_STRING
    )

    parser_object.add_argument(
        '--' + NUM_EPOCHS_ARG_NAME, type=int, required=False, default=1000,
        help=NUM_EPOCHS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NUM_TRAINING_BATCHES_ARG_NAME, type=int, required=False,
        default=32, help=NUM_TRAINING_BATCHES_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NUM_VALIDATION_BATCHES_ARG_NAME, type=int, required=False,
        default=16, help=NUM_VALIDATION_BATCHES_HELP_STRING
    )

    parser_object.add_argument(
        '--' + PLATEAU_PATIENCE_ARG_NAME, type=int, required=False,
        default=10, help=PLATEAU_PATIENCE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + PLATEAU_MULTIPLIER_ARG_NAME, type=float, required=False,
        default=0.9, help=PLATEAU_MULTIPLIER_HELP_STRING
    )
    parser_object.add_argument(
        '--' + EARLY_STOPPING_PATIENCE_ARG_NAME, type=int, required=False,
        default=50, help=EARLY_STOPPING_PATIENCE_HELP_STRING
    )

    return parser_object