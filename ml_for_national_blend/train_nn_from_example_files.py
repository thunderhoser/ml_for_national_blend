"""Trains neural net from pre-processed example files."""

import os
import sys
import copy
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import neural_net_utils as nn_utils
import neural_net_training_simple as nn_training_simple
import neural_net_training_multipatch as nn_training_multipatch

TIME_FORMAT = '%Y-%m-%d-%H'

TEMPLATE_FILE_ARG_NAME = 'input_template_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_model_dir_name'

BATCH_SIZE_ARG_NAME = 'num_examples_per_batch'
PATCH_SIZE_ARG_NAME = 'patch_size_2pt5km_pixels'
PATCH_BUFFER_SIZE_ARG_NAME = 'patch_buffer_size_2pt5km_pixels'
PATCH_OVERLAP_SIZE_ARG_NAME = 'patch_overlap_size_2pt5km_pixels'

FIRST_TRAINING_TIMES_ARG_NAME = 'first_init_time_strings_for_training'
LAST_TRAINING_TIMES_ARG_NAME = 'last_init_time_strings_for_training'
FIRST_VALIDATION_TIMES_ARG_NAME = 'first_init_time_strings_for_validation'
LAST_VALIDATION_TIMES_ARG_NAME = 'last_init_time_strings_for_validation'

NUM_EPOCHS_ARG_NAME = 'num_epochs'
EMA_DECAY_ARG_NAME = 'use_exp_moving_average_with_decay'
NUM_TRAINING_BATCHES_ARG_NAME = 'num_training_batches_per_epoch'
NUM_VALIDATION_BATCHES_ARG_NAME = 'num_validation_batches_per_epoch'

PLATEAU_PATIENCE_ARG_NAME = 'plateau_patience_epochs'
PLATEAU_MULTIPLIER_ARG_NAME = 'plateau_learning_rate_multiplier'
EARLY_STOPPING_PATIENCE_ARG_NAME = 'early_stopping_patience_epochs'

TEMPLATE_FILE_HELP_STRING = (
    'Path to template file, containing model architecture.  This will be read '
    'by `neural_net.read_model`.'
)
EXAMPLE_DIR_HELP_STRING = (
    'Path to directory with fully processed learning examples, which will be '
    'found by `example_io.find_file` and read by `example_io.read_file`.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Trained model will be saved here.'
)

BATCH_SIZE_HELP_STRING = 'Number of data examples per batch.'
PATCH_SIZE_HELP_STRING = (
    'Patch size, in units of 2.5-km pixels.  For example, if {0:s} = 448, then '
    'grid dimensions at the finest resolution (2.5 km) are 448 x 448.  If you '
    'do not want multi-patch training, make this argument negative.'
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
PATCH_OVERLAP_SIZE_HELP_STRING = (
    '[used only if {0:s} is positive] Overlap between adjacent patches, '
    'measured in number of pixels on the finest-resolution (2.5-km) grid.'
).format(
    PATCH_SIZE_ARG_NAME
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
FIRST_VALIDATION_TIMES_HELP_STRING = (
    'length-P list (where P = number of continuous validation periods), where '
    'each item is the start of a continuous validation period (format '
    '"yyyy-mm-dd-HH").'
)
LAST_VALIDATION_TIMES_HELP_STRING = (
    'Same as {0:s} but for ends of continuous validation periods.'
).format(
    FIRST_VALIDATION_TIMES_ARG_NAME
)

NUM_EPOCHS_HELP_STRING = 'Number of epochs.'
EMA_DECAY_HELP_STRING = (
    'Decay parameter for EMA (exponential moving average) training method.  If '
    'you do not want EMA, leave this argument alone.'
)
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

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + TEMPLATE_FILE_ARG_NAME, type=str, required=True,
    help=TEMPLATE_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)

INPUT_ARG_PARSER.add_argument(
    '--' + BATCH_SIZE_ARG_NAME, type=int, required=True,
    help=BATCH_SIZE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PATCH_SIZE_ARG_NAME, type=int, required=True,
    help=PATCH_SIZE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PATCH_BUFFER_SIZE_ARG_NAME, type=int, required=False, default=-1,
    help=PATCH_BUFFER_SIZE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PATCH_OVERLAP_SIZE_ARG_NAME, type=int, required=False,
    default=-1, help=PATCH_OVERLAP_SIZE_HELP_STRING
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
    '--' + FIRST_VALIDATION_TIMES_ARG_NAME, type=str, nargs='+',
    required=True, help=FIRST_VALIDATION_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_VALIDATION_TIMES_ARG_NAME, type=str, nargs='+',
    required=True, help=LAST_VALIDATION_TIMES_HELP_STRING
)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EPOCHS_ARG_NAME, type=int, required=False, default=1000,
    help=NUM_EPOCHS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EMA_DECAY_ARG_NAME, type=float, required=False, default=-1.,
    help=EMA_DECAY_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TRAINING_BATCHES_ARG_NAME, type=int, required=False,
    default=32, help=NUM_TRAINING_BATCHES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_VALIDATION_BATCHES_ARG_NAME, type=int, required=False,
    default=16, help=NUM_VALIDATION_BATCHES_HELP_STRING
)

INPUT_ARG_PARSER.add_argument(
    '--' + PLATEAU_PATIENCE_ARG_NAME, type=int, required=False,
    default=10, help=PLATEAU_PATIENCE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLATEAU_MULTIPLIER_ARG_NAME, type=float, required=False,
    default=0.9, help=PLATEAU_MULTIPLIER_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EARLY_STOPPING_PATIENCE_ARG_NAME, type=int, required=False,
    default=100, help=EARLY_STOPPING_PATIENCE_HELP_STRING
)


def _run(template_file_name, example_dir_name, output_dir_name,
         num_examples_per_batch, patch_size_2pt5km_pixels,
         patch_buffer_size_2pt5km_pixels, patch_overlap_size_2pt5km_pixels,
         first_init_time_strings_for_training,
         last_init_time_strings_for_training,
         first_init_time_strings_for_validation,
         last_init_time_strings_for_validation,
         num_epochs, use_exp_moving_average_with_decay,
         num_training_batches_per_epoch, num_validation_batches_per_epoch,
         plateau_patience_epochs, plateau_learning_rate_multiplier,
         early_stopping_patience_epochs):
    """Trains neural net from pre-processed example files.

    This is effectively the main method.

    :param template_file_name: See documentation at top of this script.
    :param example_dir_name: Same.
    :param output_dir_name: Same.
    :param num_examples_per_batch: Same.
    :param patch_size_2pt5km_pixels: Same.
    :param patch_buffer_size_2pt5km_pixels: Same.
    :param patch_overlap_size_2pt5km_pixels: Same.
    :param first_init_time_strings_for_training: Same.
    :param last_init_time_strings_for_training: Same.
    :param first_init_time_strings_for_validation: Same.
    :param last_init_time_strings_for_validation: Same.
    :param num_epochs: Same.
    :param use_exp_moving_average_with_decay: Same.
    :param num_training_batches_per_epoch: Same.
    :param num_validation_batches_per_epoch: Same.
    :param plateau_patience_epochs: Same.
    :param plateau_learning_rate_multiplier: Same.
    :param early_stopping_patience_epochs: Same.
    """

    if patch_size_2pt5km_pixels < 0:
        patch_size_2pt5km_pixels = None
    if patch_buffer_size_2pt5km_pixels < 0:
        patch_buffer_size_2pt5km_pixels = None
    if patch_overlap_size_2pt5km_pixels < 0:
        patch_overlap_size_2pt5km_pixels = None

    if (
            patch_size_2pt5km_pixels is None
            or patch_buffer_size_2pt5km_pixels is None
            or patch_overlap_size_2pt5km_pixels is None
    ):
        patch_size_2pt5km_pixels = None
        patch_buffer_size_2pt5km_pixels = None
        patch_overlap_size_2pt5km_pixels = None

    if use_exp_moving_average_with_decay < 0:
        use_exp_moving_average_with_decay = None

    first_init_times_for_training_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, TIME_FORMAT)
        for t in first_init_time_strings_for_training
    ], dtype=int)
    last_init_times_for_training_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, TIME_FORMAT)
        for t in last_init_time_strings_for_training
    ], dtype=int)
    first_init_times_for_validation_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, TIME_FORMAT)
        for t in first_init_time_strings_for_validation
    ], dtype=int)
    last_init_times_for_validation_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, TIME_FORMAT)
        for t in last_init_time_strings_for_validation
    ], dtype=int)

    if patch_size_2pt5km_pixels is None:
        training_generator = (
            nn_training_simple.data_generator_from_example_files(
                example_dir_name=example_dir_name,
                first_init_times_unix_sec=
                first_init_times_for_training_unix_sec,
                last_init_times_unix_sec=last_init_times_for_training_unix_sec,
                num_examples_per_batch=num_examples_per_batch
            )
        )

        validation_generator = (
            nn_training_simple.data_generator_from_example_files(
                example_dir_name=example_dir_name,
                first_init_times_unix_sec=
                first_init_times_for_validation_unix_sec,
                last_init_times_unix_sec=
                last_init_times_for_validation_unix_sec,
                num_examples_per_batch=num_examples_per_batch
            )
        )
    else:
        training_generator = (
            nn_training_multipatch.data_generator_from_example_files(
                example_dir_name=example_dir_name,
                first_init_times_unix_sec=
                first_init_times_for_training_unix_sec,
                last_init_times_unix_sec=last_init_times_for_training_unix_sec,
                num_examples_per_batch=num_examples_per_batch,
                patch_size_2pt5km_pixels=patch_size_2pt5km_pixels,
                patch_buffer_size_2pt5km_pixels=patch_buffer_size_2pt5km_pixels,
                patch_overlap_size_2pt5km_pixels=
                patch_overlap_size_2pt5km_pixels
            )
        )

        validation_generator = (
            nn_training_multipatch.data_generator_from_example_files(
                example_dir_name=example_dir_name,
                first_init_times_unix_sec=
                first_init_times_for_validation_unix_sec,
                last_init_times_unix_sec=
                last_init_times_for_validation_unix_sec,
                num_examples_per_batch=num_examples_per_batch,
                patch_size_2pt5km_pixels=patch_size_2pt5km_pixels,
                patch_buffer_size_2pt5km_pixels=patch_buffer_size_2pt5km_pixels,
                patch_overlap_size_2pt5km_pixels=
                patch_overlap_size_2pt5km_pixels
            )
        )

    example_metafile_name = nn_utils.find_metafile(
        model_file_name='{0:s}/model.weights.h5'.format(example_dir_name),
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(example_metafile_name))
    example_metadata_dict = nn_utils.read_metafile(example_metafile_name)
    option_dict = example_metadata_dict[nn_utils.TRAINING_OPTIONS_KEY]
    option_dict[nn_utils.BATCH_SIZE_KEY] = num_examples_per_batch
    option_dict[nn_utils.PATCH_SIZE_KEY] = patch_size_2pt5km_pixels
    option_dict[nn_utils.PATCH_BUFFER_SIZE_KEY] = (
        patch_buffer_size_2pt5km_pixels
    )

    training_option_dict = copy.deepcopy(option_dict)
    training_option_dict[nn_utils.FIRST_INIT_TIMES_KEY] = (
        first_init_times_for_training_unix_sec
    )
    training_option_dict[nn_utils.LAST_INIT_TIMES_KEY] = (
        last_init_times_for_training_unix_sec
    )

    validation_option_dict = copy.deepcopy(option_dict)
    validation_option_dict[nn_utils.FIRST_INIT_TIMES_KEY] = (
        first_init_times_for_validation_unix_sec
    )
    validation_option_dict[nn_utils.LAST_INIT_TIMES_KEY] = (
        last_init_times_for_validation_unix_sec
    )

    print('Reading model template from: "{0:s}"...'.format(template_file_name))
    model_object = nn_utils.read_model(
        hdf5_file_name=template_file_name, for_inference=False
    )

    model_metafile_name = nn_utils.find_metafile(
        model_file_name=template_file_name, raise_error_if_missing=True
    )
    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = nn_utils.read_metafile(model_metafile_name)
    mmd = model_metadata_dict

    if patch_size_2pt5km_pixels is None:
        nn_training_simple.train_model(
            model_object=model_object,
            num_epochs=num_epochs,
            use_exp_moving_average_with_decay=use_exp_moving_average_with_decay,
            num_training_batches_per_epoch=num_training_batches_per_epoch,
            training_option_dict=training_option_dict,
            num_validation_batches_per_epoch=num_validation_batches_per_epoch,
            validation_option_dict=validation_option_dict,
            loss_function_string=mmd[nn_utils.LOSS_FUNCTION_KEY],
            optimizer_function_string=mmd[nn_utils.OPTIMIZER_FUNCTION_KEY],
            metric_function_strings=mmd[nn_utils.METRIC_FUNCTIONS_KEY],
            u_net_architecture_dict=mmd[nn_utils.U_NET_ARCHITECTURE_KEY],
            chiu_net_architecture_dict=mmd[nn_utils.CHIU_NET_ARCHITECTURE_KEY],
            chiu_net_pp_architecture_dict=
            mmd[nn_utils.CHIU_NET_PP_ARCHITECTURE_KEY],
            chiu_next_pp_architecture_dict=
            mmd[nn_utils.CHIU_NEXT_PP_ARCHITECTURE_KEY],
            plateau_patience_epochs=plateau_patience_epochs,
            plateau_learning_rate_multiplier=plateau_learning_rate_multiplier,
            early_stopping_patience_epochs=early_stopping_patience_epochs,
            output_dir_name=output_dir_name,
            training_generator=training_generator,
            validation_generator=validation_generator
        )
    else:
        nn_training_multipatch.train_model(
            model_object=model_object,
            num_epochs=num_epochs,
            use_exp_moving_average_with_decay=use_exp_moving_average_with_decay,
            num_training_batches_per_epoch=num_training_batches_per_epoch,
            training_option_dict=training_option_dict,
            num_validation_batches_per_epoch=num_validation_batches_per_epoch,
            validation_option_dict=validation_option_dict,
            loss_function_string=mmd[nn_utils.LOSS_FUNCTION_KEY],
            optimizer_function_string=mmd[nn_utils.OPTIMIZER_FUNCTION_KEY],
            metric_function_strings=mmd[nn_utils.METRIC_FUNCTIONS_KEY],
            u_net_architecture_dict=mmd[nn_utils.U_NET_ARCHITECTURE_KEY],
            chiu_net_architecture_dict=mmd[nn_utils.CHIU_NET_ARCHITECTURE_KEY],
            chiu_net_pp_architecture_dict=
            mmd[nn_utils.CHIU_NET_PP_ARCHITECTURE_KEY],
            chiu_next_pp_architecture_dict=
            mmd[nn_utils.CHIU_NEXT_PP_ARCHITECTURE_KEY],
            plateau_patience_epochs=plateau_patience_epochs,
            plateau_learning_rate_multiplier=plateau_learning_rate_multiplier,
            early_stopping_patience_epochs=early_stopping_patience_epochs,
            patch_overlap_fast_gen_2pt5km_pixels=
            patch_overlap_size_2pt5km_pixels,
            output_dir_name=output_dir_name,
            training_generator=training_generator,
            validation_generator=validation_generator
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        template_file_name=getattr(INPUT_ARG_OBJECT, TEMPLATE_FILE_ARG_NAME),
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME),
        num_examples_per_batch=getattr(INPUT_ARG_OBJECT, BATCH_SIZE_ARG_NAME),
        patch_size_2pt5km_pixels=getattr(INPUT_ARG_OBJECT, PATCH_SIZE_ARG_NAME),
        patch_buffer_size_2pt5km_pixels=getattr(
            INPUT_ARG_OBJECT, PATCH_BUFFER_SIZE_ARG_NAME
        ),
        patch_overlap_size_2pt5km_pixels=getattr(
            INPUT_ARG_OBJECT, PATCH_OVERLAP_SIZE_ARG_NAME
        ),
        first_init_time_strings_for_training=getattr(
            INPUT_ARG_OBJECT, FIRST_TRAINING_TIMES_ARG_NAME
        ),
        last_init_time_strings_for_training=getattr(
            INPUT_ARG_OBJECT, LAST_TRAINING_TIMES_ARG_NAME
        ),
        first_init_time_strings_for_validation=getattr(
            INPUT_ARG_OBJECT, FIRST_VALIDATION_TIMES_ARG_NAME
        ),
        last_init_time_strings_for_validation=getattr(
            INPUT_ARG_OBJECT, LAST_VALIDATION_TIMES_ARG_NAME
        ),
        num_epochs=getattr(INPUT_ARG_OBJECT, NUM_EPOCHS_ARG_NAME),
        use_exp_moving_average_with_decay=getattr(
            INPUT_ARG_OBJECT, EMA_DECAY_ARG_NAME
        ),
        num_training_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, NUM_TRAINING_BATCHES_ARG_NAME
        ),
        num_validation_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, NUM_VALIDATION_BATCHES_ARG_NAME
        ),
        plateau_patience_epochs=getattr(
            INPUT_ARG_OBJECT, PLATEAU_PATIENCE_ARG_NAME
        ),
        plateau_learning_rate_multiplier=getattr(
            INPUT_ARG_OBJECT, PLATEAU_MULTIPLIER_ARG_NAME
        ),
        early_stopping_patience_epochs=getattr(
            INPUT_ARG_OBJECT, EARLY_STOPPING_PATIENCE_ARG_NAME
        )
    )
