"""Applies trained neural net -- inference time!"""

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
import nbm_utils
import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TIME_FORMAT = '%Y-%m-%d-%H'
NUM_EXAMPLES_PER_BATCH = 3

# TODO(thunderhoser): Need to deal with possibility that targets are normalized,
# i.e., might need to denormalize predictions.

MODEL_FILE_ARG_NAME = 'input_model_file_name'
INIT_TIME_ARG_NAME = 'init_time_string'
NWP_MODELS_ARG_NAME = 'nwp_model_names'
NWP_DIRECTORIES_ARG_NAME = 'input_nwp_directory_names'
TARGET_DIR_ARG_NAME = 'input_target_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained model (will be read by `neural_net.read_model`).'
)
INIT_TIME_HELP_STRING = 'Forecast-initialization time (format "yyyy-mm-dd-HH").'
NWP_MODELS_HELP_STRING = (
    'List of NWP models used to create predictors.  This list must match the '
    'one used for training (although it may be in a different order).'
)
NWP_DIRECTORIES_HELP_STRING = (
    'List of directory paths with NWP data.  This list must have the same '
    'length as `{0:s}`.  Relevant files in each directory will be found by '
    '`nwp_model_io.find_file` and read by `nwp_model_io.read_file`.'
).format(
    NWP_MODELS_ARG_NAME
)
TARGET_DIR_HELP_STRING = (
    'Path to directory with target data.  Relevant files in this directory '
    'will be found by `urma_io.find_file` and read by `urma_io.read_file`.'
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Predictions and targets for the given '
    'initialization time will be written here by `prediction_io.write_file`, '
    'to an exact location determined by `prediction_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INIT_TIME_ARG_NAME, type=str, required=True,
    help=INIT_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NWP_MODELS_ARG_NAME, type=str, nargs='+', required=True,
    help=NWP_MODELS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NWP_DIRECTORIES_ARG_NAME, type=str, nargs='+', required=True,
    help=NWP_DIRECTORIES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_DIR_ARG_NAME, type=str, required=True,
    help=TARGET_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(model_file_name, init_time_string, nwp_model_names,
         nwp_directory_names, target_dir_name, output_dir_name):
    """Applies trained neural net -- inference time!

    This is effectively the main method.

    :param model_file_name: See documentation at top of this script.
    :param init_time_string: Same.
    :param nwp_model_names: Same.
    :param nwp_directory_names: Same.
    :param target_dir_name: Same.
    :param output_dir_name: Same.
    """

    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = neural_net.read_model(model_file_name)
    model_metafile_name = neural_net.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    validation_option_dict = (
        model_metadata_dict[neural_net.VALIDATION_OPTIONS_KEY]
    )

    nwp_model_to_dir_name = dict(
        zip(nwp_model_names, nwp_directory_names)
    )
    init_time_unix_sec = time_conversion.string_to_unix_sec(
        init_time_string, TIME_FORMAT
    )
    validation_option_dict.update({
        neural_net.INIT_TIME_LIMITS_KEY:
            numpy.full(2, init_time_unix_sec, dtype=int),
        neural_net.NWP_MODEL_TO_DIR_KEY: nwp_model_to_dir_name,
        neural_net.TARGET_DIR_KEY: target_dir_name
    })

    data_dict = neural_net.create_data(validation_option_dict)
    predictor_matrices = data_dict[neural_net.PREDICTOR_MATRICES_KEY]
    target_matrix = data_dict[neural_net.TARGET_MATRIX_KEY]
    init_time_unix_sec = data_dict[neural_net.INIT_TIMES_KEY][0]

    # TODO(thunderhoser): HACK.
    latitude_matrix_deg_n, longitude_matrix_deg_e = nbm_utils.read_coords()
    if validation_option_dict[neural_net.SUBSET_GRID_KEY]:
        latitude_matrix_deg_n = latitude_matrix_deg_n[544:993, 752:1201]
        longitude_matrix_deg_e = longitude_matrix_deg_e[544:993, 752:1201]

    prediction_matrix = neural_net.apply_model(
        model_object=model_object,
        predictor_matrices=predictor_matrices,
        num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
        verbose=True,
        predict_dewpoint_depression=
        validation_option_dict[neural_net.PREDICT_DEWPOINT_DEPRESSION_KEY],
        predict_gust_factor=
        validation_option_dict[neural_net.PREDICT_GUST_FACTOR_KEY],
        target_field_names=validation_option_dict[neural_net.TARGET_FIELDS_KEY]
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
        latitude_matrix_deg_n=latitude_matrix_deg_n,
        longitude_matrix_deg_e=longitude_matrix_deg_e,
        field_names=validation_option_dict[neural_net.TARGET_FIELDS_KEY],
        init_time_unix_sec=init_time_unix_sec,
        model_file_name=model_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        init_time_string=getattr(INPUT_ARG_OBJECT, INIT_TIME_ARG_NAME),
        nwp_model_names=getattr(INPUT_ARG_OBJECT, NWP_MODELS_ARG_NAME),
        nwp_directory_names=getattr(INPUT_ARG_OBJECT, NWP_DIRECTORIES_ARG_NAME),
        target_dir_name=getattr(INPUT_ARG_OBJECT, TARGET_DIR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
