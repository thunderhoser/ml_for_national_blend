"""Does inference for neural net trained with patchwise approach."""

import copy
import argparse
import numpy
from ml_for_national_blend.outside_code import time_conversion
from ml_for_national_blend.io import prediction_io
from ml_for_national_blend.utils import nwp_model_utils
from ml_for_national_blend.machine_learning import \
    neural_net_with_fancy_patches as neural_net

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
PATCHES_TO_FULL_GRID_ARG_NAME = 'patches_to_full_grid'
PATCH_START_ROW_ARG_NAME = 'patch_start_row_2pt5km'
PATCH_START_COLUMN_ARG_NAME = 'patch_start_column_2pt5km'
PATCH_OVERLAP_SIZE_ARG_NAME = 'patch_overlap_size_2pt5km_pixels'
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
PATCHES_TO_FULL_GRID_HELP_STRING = (
    '[used only if NN was trained with patchwise approach] Boolean flag.  If '
    '1, will slide patch around the full grid to generate predictions on the '
    'full grid.  If 0, will generate predictions for patches of the same size '
    'used to train.'
)
PATCH_START_ROW_HELP_STRING = (
    '[used only if NN was trained with patchwise approach and {0:s} == 0] For '
    'every data sample -- i.e., every init time -- will apply the NN to a '
    'patch at the same location, starting at this row in the 2.5-km grid.  If '
    'you want the patch location to move around, make this argument negative.'
).format(
    PATCHES_TO_FULL_GRID_ARG_NAME
)
PATCH_START_COLUMN_HELP_STRING = (
    'Same as {0:s}, but for column instead of row.'
).format(
    PATCH_START_ROW_ARG_NAME
)
PATCH_OVERLAP_SIZE_HELP_STRING = (
    '[used only if NN was trained with patchwise approach and {0:s} == 1] '
    'Overlap between adjacent patches, measured in number of pixels on the '
    'finest-resolution (2.5-km) grid.'
).format(
    PATCHES_TO_FULL_GRID_ARG_NAME
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
    '--' + PATCHES_TO_FULL_GRID_ARG_NAME, type=int, required=False, default=0,
    help=PATCHES_TO_FULL_GRID_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PATCH_START_ROW_ARG_NAME, type=int, required=False, default=-1,
    help=PATCH_START_ROW_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PATCH_START_COLUMN_ARG_NAME, type=int, required=False, default=-1,
    help=PATCH_START_COLUMN_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PATCH_OVERLAP_SIZE_ARG_NAME, type=int, required=False, default=-1,
    help=PATCH_OVERLAP_SIZE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _process_nwp_directories(nwp_directory_names, nwp_model_names):
    """Processes NWP directories for either training or validation data.

    :param nwp_directory_names: See documentation for input arg
        "nwp_dir_names_for_training" to this script.
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


def _run(model_file_name, init_time_string, nwp_model_names,
         nwp_directory_names, target_dir_name,
         patches_to_full_grid, patch_start_row_2pt5km,
         patch_start_column_2pt5km, patch_overlap_size_2pt5km_pixels,
         output_dir_name):
    """Applies trained neural net -- inference time!

    Does inference for neural net trained with patchwise approach.

    :param model_file_name: See documentation at top of this script.
    :param init_time_string: Same.
    :param nwp_model_names: Same.
    :param nwp_directory_names: Same.
    :param target_dir_name: Same.
    :param patches_to_full_grid: Same.
    :param patch_start_row_2pt5km: Same.
    :param patch_start_column_2pt5km: Same.
    :param patch_overlap_size_2pt5km_pixels: Same.
    :param output_dir_name: Same.
    """

    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = neural_net.read_model(model_file_name)
    model_metafile_name = neural_net.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    mmd = model_metadata_dict

    validation_option_dict = mmd[neural_net.VALIDATION_OPTIONS_KEY]
    was_nn_trained_patchwise = validation_option_dict[neural_net.PATCH_SIZE_KEY]
    nwp_model_names_for_training = list(
        validation_option_dict[neural_net.NWP_MODEL_TO_DIR_KEY].keys()
    )
    assert set(nwp_model_names) == set(nwp_model_names_for_training)

    nwp_model_to_dir_name = _process_nwp_directories(
        nwp_directory_names=nwp_directory_names,
        nwp_model_names=nwp_model_names
    )
    init_time_unix_sec = time_conversion.string_to_unix_sec(
        init_time_string, TIME_FORMAT
    )
    validation_option_dict.update({
        neural_net.FIRST_INIT_TIMES_KEY: numpy.array(
            [init_time_unix_sec], dtype=int
        ),
        neural_net.LAST_INIT_TIMES_KEY: numpy.array(
            [init_time_unix_sec], dtype=int
        ),
        neural_net.NWP_MODEL_TO_DIR_KEY: nwp_model_to_dir_name,
        neural_net.TARGET_DIR_KEY: target_dir_name
    })

    if not was_nn_trained_patchwise:
        patches_to_full_grid = False
        patch_start_row_2pt5km = None
        patch_start_column_2pt5km = None

    if patches_to_full_grid:
        validation_option_dict[neural_net.PATCH_SIZE_KEY] = None
        patch_start_row_2pt5km = None
        patch_start_column_2pt5km = None

    if patch_start_row_2pt5km < 0 or patch_start_column_2pt5km < 0:
        patch_start_row_2pt5km = None
        patch_start_column_2pt5km = None

    data_dict = neural_net.create_data(
        option_dict=validation_option_dict,
        patch_start_row_2pt5km=patch_start_row_2pt5km,
        patch_start_column_2pt5km=patch_start_column_2pt5km
    )

    predictor_matrices = data_dict[neural_net.PREDICTOR_MATRICES_KEY]
    target_matrix = data_dict[neural_net.TARGET_MATRIX_KEY]
    init_time_unix_sec = data_dict[neural_net.INIT_TIMES_KEY][0]
    latitude_matrix_deg_n = data_dict[neural_net.LATITUDE_MATRIX_KEY][0, ...]
    longitude_matrix_deg_e = data_dict[neural_net.LONGITUDE_MATRIX_KEY][0, ...]

    vod = validation_option_dict

    if patches_to_full_grid:
        prediction_matrix = neural_net.apply_patchwise_model_to_full_grid(
            model_object=model_object,
            full_predictor_matrices=predictor_matrices,
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
            verbose=True,
            predict_dewpoint_depression=
            vod[neural_net.PREDICT_DEWPOINT_DEPRESSION_KEY],
            predict_gust_factor=vod[neural_net.PREDICT_GUST_FACTOR_KEY],
            target_field_names=vod[neural_net.TARGET_FIELDS_KEY],
            patch_overlap_size_2pt5km_pixels=patch_overlap_size_2pt5km_pixels
        )
    else:
        prediction_matrix = neural_net.apply_model(
            model_object=model_object,
            predictor_matrices=predictor_matrices,
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
            verbose=True,
            predict_dewpoint_depression=
            vod[neural_net.PREDICT_DEWPOINT_DEPRESSION_KEY],
            predict_gust_factor=vod[neural_net.PREDICT_GUST_FACTOR_KEY],
            target_field_names=vod[neural_net.TARGET_FIELDS_KEY]
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
        patches_to_full_grid=bool(
            getattr(INPUT_ARG_OBJECT, PATCHES_TO_FULL_GRID_ARG_NAME)
        ),
        patch_start_row_2pt5km=getattr(
            INPUT_ARG_OBJECT, PATCH_START_ROW_ARG_NAME
        ),
        patch_start_column_2pt5km=getattr(
            INPUT_ARG_OBJECT, PATCH_START_COLUMN_ARG_NAME
        ),
        patch_overlap_size_2pt5km_pixels=getattr(
            INPUT_ARG_OBJECT, PATCH_OVERLAP_SIZE_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
