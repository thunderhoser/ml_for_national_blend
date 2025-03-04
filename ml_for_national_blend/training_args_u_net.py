"""Contains list of input arguments for training a simple U-net."""

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

NWP_LEAD_TIME_ARG_NAME = 'nwp_lead_time_hours'
NWP_MODELS_ARG_NAME = 'nwp_model_names'
NWP_MODEL_TO_FIELDS_ARG_NAME = 'nwp_model_to_field_names'
NWP_NORMALIZATION_FILE_ARG_NAME = 'nwp_normalization_file_name'

BACKUP_NWP_MODEL_ARG_NAME = 'backup_nwp_model_name'
BACKUP_NWP_DIR_ARG_NAME = 'backup_nwp_dir_name'
TARGET_LEAD_TIME_ARG_NAME = 'target_lead_time_hours'
TARGET_FIELDS_ARG_NAME = 'target_field_names'

COMPARE_TO_BASELINE_ARG_NAME = 'compare_to_baseline_in_loss'
BATCH_SIZE_ARG_NAME = 'num_examples_per_batch'
SENTINEL_VALUE_ARG_NAME = 'sentinel_value'

PATCH_SIZE_ARG_NAME = 'patch_size_2pt5km_pixels'
PATCH_BUFFER_SIZE_ARG_NAME = 'patch_buffer_size_2pt5km_pixels'
PATCH_OVERLAP_SIZE_ARG_NAME = 'patch_overlap_size_2pt5km_pixels'

RESID_BASELINE_MODEL_ARG_NAME = 'resid_baseline_model_name'
RESID_BASELINE_LEAD_TIME_ARG_NAME = 'resid_baseline_lead_time_hours'
RESID_BASELINE_MODEL_DIR_ARG_NAME = 'resid_baseline_model_dir_name'

NWP_DIRS_ARG_NAME = 'nwp_directory_names'
TARGET_DIR_ARG_NAME = 'target_dir_name'
TRAINING_TIME_LIMITS_ARG_NAME = 'training_init_time_limit_strings'
VALIDATION_TIME_LIMITS_ARG_NAME = 'validation_init_time_limit_strings'

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
    'Path to output directory.  Trained model will be saved here.'
)

NWP_LEAD_TIME_HELP_STRING = (
    'Lead time for NWP forecasts to be used in predictors.  The same lead time '
    'will be used for every NWP model.'
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
    'Path to normalization file for NWP predictors (readable by '
    '`nwp_model_io.read_normalization_file`), containing params for two-step '
    'z-score normalization.'
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

COMPARE_TO_BASELINE_HELP_STRING = (
    'Boolean flag.  If 1, the loss function involves comparing to the residual '
    'baseline.  In other words, the loss function involves a skill score, '
    'except with the residual baseline instead of climo.'
)
BATCH_SIZE_HELP_STRING = 'Number of data examples per batch.'
SENTINEL_VALUE_HELP_STRING = (
    'All NaN predictors will be replaced with this value.'
)

PATCH_SIZE_HELP_STRING = (
    'Patch size, in units of 2.5-km pixels.  For example, if {0:s} = 448, then '
    'grid dimensions at the finest resolution (2.5 km) are 448 x 448.'
).format(
    PATCH_SIZE_ARG_NAME
)
PATCH_BUFFER_SIZE_HELP_STRING = (
    'Buffer between the outer domain (used for predictors) and the inner '
    'domain (used to penalize predictions in loss function).  This must be a '
    'non-negative integer.'
)
PATCH_OVERLAP_SIZE_HELP_STRING = (
    'Overlap between adjacent patches, measured in number of pixels on the '
    'finest-resolution (2.5-km) grid.'
)


RESID_BASELINE_MODEL_HELP_STRING = (
    '[used only if {0:s} == 1] Name of NWP model used to generate baseline '
    'predictions.'
).format(
    COMPARE_TO_BASELINE_ARG_NAME
)
RESID_BASELINE_LEAD_TIME_HELP_STRING = (
    '[used only if {0:s} == 1] Lead time for baseline predictions.'
).format(
    COMPARE_TO_BASELINE_ARG_NAME
)
RESID_BASELINE_MODEL_DIR_HELP_STRING = (
    '[used only if {0:s} == 1] Path to directory for NWP model used to '
    'generate baseline predictions.'
).format(
    COMPARE_TO_BASELINE_ARG_NAME
)

NWP_DIRS_HELP_STRING = (
    'List of paths with the same length as {0:s}.  Relevant files in each '
    'directory will be found by `nwp_model_io.find_file` and read by '
    '`nwp_model_io.read_file`.'
).format(
    NWP_MODELS_ARG_NAME
)
TARGET_DIR_HELP_STRING = (
    'Path to directory with target data.  Relevant files in this directory '
    'will be found by `urma_io.find_file` and read by `urma_io.read_file`.'
)
TRAINING_TIME_LIMITS_HELP_STRING = (
    'Time limits for training period.  This should be a length-2 list, where '
    'the first (second) is the first (last) forecast-initialization time in '
    'the training period, formatted like "yyyy-mm-dd-HH".'
)
VALIDATION_TIME_LIMITS_HELP_STRING = (
    'Same as {0:s} but for validation period.'
).format(TRAINING_TIME_LIMITS_ARG_NAME)

NUM_EPOCHS_HELP_STRING = 'Number of epochs.'
NUM_TRAINING_BATCHES_HELP_STRING = 'Number of training batches per epoch.'
NUM_VALIDATION_BATCHES_HELP_STRING = 'Number of validation batches per epoch.'

PLATEAU_PATIENCE_HELP_STRING = (
    'Training will be deemed to have reached "plateau" if validation loss has '
    'not decreased in the last N epochs, where N = {0:s}.'
).format(
    PLATEAU_PATIENCE_ARG_NAME
)
PLATEAU_MULTIPLIER_HELP_STRING = (
    'If training reaches "plateau," learning rate will be multiplied by this '
    'value in range (0, 1).'
)
EARLY_STOPPING_PATIENCE_HELP_STRING = (
    'Training will be stopped early if validation loss has not decreased in '
    'the last N epochs, where N = {0:s}.'
).format(
    EARLY_STOPPING_PATIENCE_ARG_NAME
)


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
        '--' + NWP_LEAD_TIME_ARG_NAME, type=int, required=True,
        help=NWP_LEAD_TIME_HELP_STRING
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
        '--' + NWP_NORMALIZATION_FILE_ARG_NAME, type=str, required=True,
        help=NWP_NORMALIZATION_FILE_HELP_STRING
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
        '--' + COMPARE_TO_BASELINE_ARG_NAME, type=int, required=True,
        help=COMPARE_TO_BASELINE_HELP_STRING
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
        '--' + PATCH_BUFFER_SIZE_ARG_NAME, type=int, required=True,
        help=PATCH_BUFFER_SIZE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + PATCH_OVERLAP_SIZE_ARG_NAME, type=int, required=True,
        help=PATCH_OVERLAP_SIZE_HELP_STRING
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
        '--' + NWP_DIRS_ARG_NAME, type=str, nargs='+', required=True,
        help=NWP_DIRS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + TARGET_DIR_ARG_NAME, type=str, required=True,
        help=TARGET_DIR_HELP_STRING
    )
    parser_object.add_argument(
        '--' + TRAINING_TIME_LIMITS_ARG_NAME, type=str, nargs='+',
        required=True, help=TRAINING_TIME_LIMITS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + VALIDATION_TIME_LIMITS_ARG_NAME, type=str, nargs='+',
        required=True, help=VALIDATION_TIME_LIMITS_HELP_STRING
    )

    parser_object.add_argument(
        '--' + NUM_EPOCHS_ARG_NAME, type=int, required=False, default=1000,
        help=NUM_EPOCHS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NUM_TRAINING_BATCHES_ARG_NAME, type=int, required=True,
        help=NUM_TRAINING_BATCHES_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NUM_VALIDATION_BATCHES_ARG_NAME, type=int, required=True,
        help=NUM_VALIDATION_BATCHES_HELP_STRING
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
        default=100, help=EARLY_STOPPING_PATIENCE_HELP_STRING
    )

    return parser_object
