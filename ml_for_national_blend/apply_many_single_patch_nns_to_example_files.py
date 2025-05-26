"""Applies many single-patch NNs to pre-processed .npz files."""

import os
import sys
import argparse
import traceback
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import time_periods
import prediction_io
import neural_net_utils as nn_utils
import neural_net_training_simple as nn_training_simple

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
NONE_STRINGS = ['', 'none', 'None']

TIME_FORMAT = '%Y-%m-%d-%H'
SYNOPTIC_TIME_INTERVAL_SEC = 6 * 3600

NUM_EXAMPLES_PER_BATCH = 5
MASK_PIXEL_IF_WEIGHT_BELOW = 0.05

MODEL_FILES_ARG_NAME = 'input_model_file_names'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
INIT_TIME_ARG_NAME = 'init_time_string'
FIRST_INIT_TIME_ARG_NAME = 'first_init_time_string'
LAST_INIT_TIME_ARG_NAME = 'last_init_time_string'
USE_EMA_ARG_NAME = 'use_ema'
SAVE_MEAN_ONLY_ARG_NAME = 'save_ensemble_mean_only'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_FILES_HELP_STRING = (
    'List of paths to trained models.  Each model will be read by '
    '`neural_net_utils.read_model`, and each should cover a single patch '
    '(subdomain) of the full NBM grid.'
)
EXAMPLE_DIR_HELP_STRING = (
    'Path to directory with processed .npz files.  The relevant file (for the '
    'given forecast-init time) will be found by `example_io.find_file` and '
    'read by `example_io.read_file`.'
)
INIT_TIME_HELP_STRING = (
    'Forecast-initialization time (format "yyyy-mm-dd-HH").  If you want '
    'multiple times, leave this argument alone; use `{0:s}` and `{1:s}`, '
    'instead.'
).format(
    FIRST_INIT_TIME_ARG_NAME, LAST_INIT_TIME_ARG_NAME
)
FIRST_INIT_TIME_HELP_STRING = (
    'Will apply neural net to all forecast-init times in the period '
    '`{0:s}`...`{1:s}`.  If you want just one init time, use `{2:s}`, instead.'
).format(
    FIRST_INIT_TIME_ARG_NAME, LAST_INIT_TIME_ARG_NAME, INIT_TIME_ARG_NAME
)
LAST_INIT_TIME_HELP_STRING = FIRST_INIT_TIME_HELP_STRING
USE_EMA_HELP_STRING = (
    'Boolean flag.  If {0:s} == 1 and the neural nets were trained with the '
    'exponential moving average (EMA) method, then EMA weights will be used '
    'at inference time.  Otherwise, instantaneous weights will be used at '
    'inference time.'
).format(
    USE_EMA_ARG_NAME
)
SAVE_MEAN_ONLY_HELP_STRING = (
    'Boolean flag.  If 1, will save only the ensemble mean, not the full '
    'ensemble.'
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Predictions and targets for the given '
    'initialization time will be written here by `prediction_io.write_file`, '
    'to an exact location determined by `prediction_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=MODEL_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INIT_TIME_ARG_NAME, type=str, required=False, default='',
    help=INIT_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_INIT_TIME_ARG_NAME, type=str, required=False, default='',
    help=FIRST_INIT_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_INIT_TIME_ARG_NAME, type=str, required=False, default='',
    help=LAST_INIT_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + USE_EMA_ARG_NAME, type=int, required=False, default=0,
    help=USE_EMA_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SAVE_MEAN_ONLY_ARG_NAME, type=int, required=False, default=0,
    help=SAVE_MEAN_ONLY_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _apply_nns_one_init_time(
        model_objects, model_file_names, example_dir_name,
        init_time_unix_sec, save_ensemble_mean_only, output_dir_name):
    """Applies suite of neural nets to one forecast-init time.

    A = number of models

    :param model_objects: length-A list of trained models (instances of
        `keras.models.Model` or `keras.models.Sequential`).
    :param model_file_names: length-A list of paths to model files.
    :param example_dir_name: See documentation at top of this script.
    :param init_time_unix_sec: Same.
    :param save_ensemble_mean_only: Same.
    :param output_dir_name: Same.
    """

    num_models = len(model_file_names)
    model_metadata_dicts = [dict()] * num_models

    for k in range(num_models):
        this_metafile_name = nn_utils.find_metafile(
            model_file_name=model_file_names[k], raise_error_if_missing=True
        )

        print('Reading metadata from: "{0:s}"...'.format(this_metafile_name))
        model_metadata_dicts[k] = nn_utils.read_metafile(this_metafile_name)

    data_dict = nn_training_simple.create_data_from_example_file(
        example_dir_name=example_dir_name,
        init_time_unix_sec=init_time_unix_sec,
        patch_start_row_2pt5km=None,
        patch_start_column_2pt5km=None
    )

    predictor_matrices = data_dict[nn_training_simple.PREDICTOR_MATRICES_KEY]
    target_matrix_with_other_shit = data_dict[
        nn_training_simple.TARGET_MATRIX_KEY
    ]
    init_times_unix_sec = data_dict[nn_training_simple.INIT_TIMES_KEY]
    latitude_matrix_deg_n = data_dict[nn_training_simple.LATITUDE_MATRIX_KEY]
    longitude_matrix_deg_e = data_dict[nn_training_simple.LONGITUDE_MATRIX_KEY]

    validation_option_dict = model_metadata_dicts[0][
        nn_utils.VALIDATION_OPTIONS_KEY
    ]
    num_target_fields = len(validation_option_dict[nn_utils.TARGET_FIELDS_KEY])
    target_matrix = target_matrix_with_other_shit[..., :num_target_fields]
    # mask_matrix = (
    #     target_matrix_with_other_shit[..., -1] >= MASK_PIXEL_IF_WEIGHT_BELOW
    # )

    prediction_matrix = nn_utils.apply_many_single_patch_models(
        model_objects=model_objects,
        full_predictor_matrices=predictor_matrices,
        num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
        model_metadata_dicts=model_metadata_dicts,
        verbose=True
    )

    if save_ensemble_mean_only:
        prediction_matrix = numpy.mean(
            prediction_matrix, axis=-1, keepdims=True
        )

    output_file_name = prediction_io.find_file(
        directory_name=output_dir_name,
        init_time_unix_sec=init_time_unix_sec,
        raise_error_if_missing=False
    )

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    prediction_io.write_file(
        netcdf_file_name=output_file_name,
        target_matrix=target_matrix[0, ...],
        prediction_matrix=prediction_matrix[0, ...],
        latitude_matrix_deg_n=latitude_matrix_deg_n[0, ...],
        longitude_matrix_deg_e=longitude_matrix_deg_e[0, ...],
        field_names=validation_option_dict[nn_utils.TARGET_FIELDS_KEY],
        init_time_unix_sec=init_times_unix_sec[0],
        model_file_name=model_file_names[0],
        isotonic_model_file_names=None,
        uncertainty_calib_model_file_names=None
    )


def _run(model_file_names, example_dir_name, init_time_string,
         first_init_time_string, last_init_time_string,
         use_ema, save_ensemble_mean_only, output_dir_name):
    """Applies many single-patch NNs to pre-processed .npz files.

    This is effectively the main method.

    :param model_file_names: See documentation at top of this script.
    :param example_dir_name: Same.
    :param init_time_string: Same.
    :param first_init_time_string: Same.
    :param last_init_time_string: Same.
    :param use_ema: Same.
    :param save_ensemble_mean_only: Same.
    :param output_dir_name: Same.
    """

    num_models = len(model_file_names)
    model_objects = [None] * num_models

    for k in range(num_models):
        print('Reading model from: "{0:s}"...'.format(model_file_names[k]))
        model_objects[k] = nn_utils.read_model(
            hdf5_file_name=model_file_names[k], for_inference=use_ema
        )

    if init_time_string in NONE_STRINGS:
        first_init_time_unix_sec = time_conversion.string_to_unix_sec(
            first_init_time_string, TIME_FORMAT
        )
        last_init_time_unix_sec = time_conversion.string_to_unix_sec(
            last_init_time_string, TIME_FORMAT
        )
        init_times_unix_sec = time_periods.range_and_interval_to_list(
            start_time_unix_sec=first_init_time_unix_sec,
            end_time_unix_sec=last_init_time_unix_sec,
            time_interval_sec=SYNOPTIC_TIME_INTERVAL_SEC,
            include_endpoint=True
        )
    else:
        init_time_unix_sec = time_conversion.string_to_unix_sec(
            init_time_string, TIME_FORMAT
        )
        init_times_unix_sec = numpy.array([init_time_unix_sec], dtype=int)

    for this_init_time_unix_sec in init_times_unix_sec:
        try:
            _apply_nns_one_init_time(
                model_objects=model_objects,
                model_file_names=model_file_names,
                example_dir_name=example_dir_name,
                init_time_unix_sec=this_init_time_unix_sec,
                save_ensemble_mean_only=save_ensemble_mean_only,
                output_dir_name=output_dir_name
            )
        except Exception:
            traceback.print_exc()


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_names=getattr(INPUT_ARG_OBJECT, MODEL_FILES_ARG_NAME),
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        init_time_string=getattr(INPUT_ARG_OBJECT, INIT_TIME_ARG_NAME),
        first_init_time_string=getattr(
            INPUT_ARG_OBJECT, FIRST_INIT_TIME_ARG_NAME
        ),
        last_init_time_string=getattr(
            INPUT_ARG_OBJECT, LAST_INIT_TIME_ARG_NAME
        ),
        use_ema=bool(getattr(INPUT_ARG_OBJECT, USE_EMA_ARG_NAME)),
        save_ensemble_mean_only=bool(
            getattr(INPUT_ARG_OBJECT, SAVE_MEAN_ONLY_ARG_NAME)
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
