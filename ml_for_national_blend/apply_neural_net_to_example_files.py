"""Applies trained neural network to pre-processed .npz files."""

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
import neural_net_utils as nn_utils
import neural_net_training_simple as nn_training_simple
import neural_net_training_multipatch as nn_training_multipatch

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TIME_FORMAT = '%Y-%m-%d-%H'
NUM_EXAMPLES_PER_BATCH = 5

MASK_PIXEL_IF_WEIGHT_BELOW = 0.05

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
INIT_TIME_ARG_NAME = 'init_time_string'
PATCHES_TO_FULL_GRID_ARG_NAME = 'patches_to_full_grid'
SINGLE_PATCH_START_ROW_ARG_NAME = 'single_patch_start_row_2pt5km'
SINGLE_PATCH_START_COLUMN_ARG_NAME = 'single_patch_start_column_2pt5km'
USE_EMA_ARG_NAME = 'use_ema'
SAVE_MEAN_ONLY_ARG_NAME = 'save_ensemble_mean_only'
USE_TRAPEZOIDAL_WEIGHTING_ARG_NAME = 'use_trapezoidal_weighting'
PATCH_OVERLAP_SIZE_ARG_NAME = 'patch_overlap_size_2pt5km_pixels'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained model (will be read by `neural_net_utils.read_model`).'
)
EXAMPLE_DIR_HELP_STRING = (
    'Path to directory with processed .npz files.  The relevant file (for the '
    'given forecast-init time) will be found by `example_io.find_file` and '
    'read by `example_io.read_file`.'
)
INIT_TIME_HELP_STRING = 'Forecast-initialization time (format "yyyy-mm-dd-HH").'
PATCHES_TO_FULL_GRID_HELP_STRING = (
    '[used only if NN was trained with multiple patches] Boolean flag.  If '
    '1, will slide patch around the full grid to generate predictions on the '
    'full grid.  If 0, will generate predictions for patches of the same size '
    'used to train.'
)
SINGLE_PATCH_START_ROW_HELP_STRING = (
    '[used only if NN was trained with a single patch] Start row of patch.  If '
    '{0:s} == j, this means the patch starts at the [j]th row of the full NBM '
    'grid.'
).format(
    SINGLE_PATCH_START_ROW_ARG_NAME
)
SINGLE_PATCH_START_COLUMN_HELP_STRING = (
    '[used only if NN was trained with a single patch] Start column of patch.  '
    'If {0:s} == k, this means the patch starts at the [k]th column of the '
    'full NBM grid.'
).format(
    SINGLE_PATCH_START_ROW_ARG_NAME
)
USE_EMA_HELP_STRING = (
    'Boolean flag.  If {0:s} == 1 and the neural net was trained with the '
    'exponential moving average (EMA) metadata, then EMA weights will be used '
    'at inference time.  Otherwise, instantaneous weights will be used at '
    'inference time.'
).format(
    USE_EMA_ARG_NAME
)
SAVE_MEAN_ONLY_HELP_STRING = (
    'Boolean flag.  If 1, will save only the ensemble mean, not the full '
    'ensemble.'
)
USE_TRAPEZOIDAL_WEIGHTING_HELP_STRING = (
    '[used only if {0:s} == 1] Boolean flag.  If 1, trapezoidal weighting will '
    'be used, so that predictions in the center of a given patch are given a '
    'higher weight than predictions at the edge.'
).format(
    PATCHES_TO_FULL_GRID_ARG_NAME
)
PATCH_OVERLAP_SIZE_HELP_STRING = (
    '[used only if {0:s} == 1] Overlap between adjacent patches, measured in '
    'number of pixels on the finest-resolution (2.5-km) grid.'
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
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INIT_TIME_ARG_NAME, type=str, required=True,
    help=INIT_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PATCHES_TO_FULL_GRID_ARG_NAME, type=int, required=False, default=0,
    help=PATCHES_TO_FULL_GRID_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SINGLE_PATCH_START_ROW_ARG_NAME, type=int, required=False, default=-1,
    help=SINGLE_PATCH_START_ROW_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SINGLE_PATCH_START_COLUMN_ARG_NAME, type=int, required=False,
    default=-1, help=SINGLE_PATCH_START_COLUMN_HELP_STRING
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
    '--' + USE_TRAPEZOIDAL_WEIGHTING_ARG_NAME, type=int, required=False,
    default=0, help=USE_TRAPEZOIDAL_WEIGHTING_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PATCH_OVERLAP_SIZE_ARG_NAME, type=int, required=False, default=-1,
    help=PATCH_OVERLAP_SIZE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(model_file_name, example_dir_name, init_time_string,
         patches_to_full_grid,
         single_patch_start_row_2pt5km, single_patch_start_column_2pt5km,
         use_ema, save_ensemble_mean_only,
         use_trapezoidal_weighting, patch_overlap_size_2pt5km_pixels,
         output_dir_name):
    """Applies trained neural network to pre-processed .npz files.

    This is effectively the main method.

    :param model_file_name: Same.
    :param example_dir_name: Same.
    :param init_time_string: Same.
    :param patches_to_full_grid: Same.
    :param single_patch_start_row_2pt5km: Same.
    :param single_patch_start_column_2pt5km: Same.
    :param use_ema: Same.
    :param save_ensemble_mean_only: Same.
    :param use_trapezoidal_weighting: Same.
    :param patch_overlap_size_2pt5km_pixels: Same.
    :param output_dir_name: Same.
    """

    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = nn_utils.read_model(
        hdf5_file_name=model_file_name, for_inference=use_ema
    )
    model_metafile_name = nn_utils.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = nn_utils.read_metafile(model_metafile_name)
    mmd = model_metadata_dict

    is_nn_multipatch = mmd[nn_utils.PATCH_OVERLAP_FOR_FAST_GEN_KEY] is not None
    if not is_nn_multipatch:
        patches_to_full_grid = False

    validation_option_dict = mmd[nn_utils.VALIDATION_OPTIONS_KEY]
    init_time_unix_sec = time_conversion.string_to_unix_sec(
        init_time_string, TIME_FORMAT
    )

    if patches_to_full_grid:
        data_dict = nn_training_simple.create_data_from_example_file(
            example_dir_name=example_dir_name,
            init_time_unix_sec=init_time_unix_sec,
            patch_start_row_2pt5km=None,
            patch_start_column_2pt5km=None
        )
    elif is_nn_multipatch:
        vod = validation_option_dict

        data_dict = nn_training_multipatch.create_data_from_example_files(
            example_dir_name=example_dir_name,
            init_time_unix_sec=init_time_unix_sec,
            patch_size_2pt5km_pixels=vod[nn_utils.PATCH_SIZE_KEY],
            patch_buffer_size_2pt5km_pixels=vod[nn_utils.PATCH_BUFFER_SIZE_KEY],
            patch_overlap_size_2pt5km_pixels=
            mmd[nn_utils.PATCH_OVERLAP_FOR_FAST_GEN_KEY]
        )
    else:
        if single_patch_start_row_2pt5km < 0:
            single_patch_start_row_2pt5km = None
        if single_patch_start_column_2pt5km < 0:
            single_patch_start_column_2pt5km = None

        data_dict = nn_training_simple.create_data_from_example_file(
            example_dir_name=example_dir_name,
            init_time_unix_sec=init_time_unix_sec,
            patch_start_row_2pt5km=single_patch_start_row_2pt5km,
            patch_start_column_2pt5km=single_patch_start_column_2pt5km
        )

    predictor_matrices = data_dict[
        nn_training_simple.PREDICTOR_MATRICES_KEY
    ]
    target_matrix_with_other_shit = data_dict[
        nn_training_simple.TARGET_MATRIX_KEY
    ]
    init_times_unix_sec = data_dict[nn_training_simple.INIT_TIMES_KEY]
    latitude_matrix_deg_n = data_dict[nn_training_simple.LATITUDE_MATRIX_KEY]
    longitude_matrix_deg_e = data_dict[nn_training_simple.LONGITUDE_MATRIX_KEY]

    num_target_fields = len(validation_option_dict[nn_utils.TARGET_FIELDS_KEY])
    target_matrix = target_matrix_with_other_shit[..., :num_target_fields]
    mask_matrix = (
        target_matrix_with_other_shit[..., -1] >= MASK_PIXEL_IF_WEIGHT_BELOW
    )

    vod = validation_option_dict

    if patches_to_full_grid:
        prediction_matrix = nn_training_multipatch.apply_model(
            model_object=model_object,
            full_predictor_matrices=predictor_matrices,
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
            model_metadata_dict=model_metadata_dict,
            verbose=True,
            patch_overlap_size_2pt5km_pixels=patch_overlap_size_2pt5km_pixels,
            use_trapezoidal_weighting=use_trapezoidal_weighting
        )

        if save_ensemble_mean_only:
            prediction_matrix = numpy.mean(
                prediction_matrix, axis=-1, keepdims=True
            )
    else:
        prediction_matrix = nn_training_simple.apply_model(
            model_object=model_object,
            predictor_matrices=predictor_matrices,
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
            verbose=True,
            target_field_names=vod[nn_utils.TARGET_FIELDS_KEY]
        )

        if save_ensemble_mean_only:
            prediction_matrix = numpy.mean(
                prediction_matrix, axis=-1, keepdims=True
            )

        while len(mask_matrix.shape) < len(prediction_matrix.shape):
            mask_matrix = numpy.expand_dims(mask_matrix, axis=-1)

        print(prediction_matrix.shape)
        print(mask_matrix.shape)

        prediction_matrix[mask_matrix == False] = numpy.nan

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
        model_file_name=model_file_name,
        isotonic_model_file_names=None,
        uncertainty_calib_model_file_names=None
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        init_time_string=getattr(INPUT_ARG_OBJECT, INIT_TIME_ARG_NAME),
        patches_to_full_grid=bool(
            getattr(INPUT_ARG_OBJECT, PATCHES_TO_FULL_GRID_ARG_NAME)
        ),
        single_patch_start_row_2pt5km=getattr(
            INPUT_ARG_OBJECT, SINGLE_PATCH_START_ROW_ARG_NAME
        ),
        single_patch_start_column_2pt5km=getattr(
            INPUT_ARG_OBJECT, SINGLE_PATCH_START_COLUMN_ARG_NAME
        ),
        use_ema=bool(getattr(INPUT_ARG_OBJECT, USE_EMA_ARG_NAME)),
        save_ensemble_mean_only=bool(
            getattr(INPUT_ARG_OBJECT, SAVE_MEAN_ONLY_ARG_NAME)
        ),
        use_trapezoidal_weighting=bool(
            getattr(INPUT_ARG_OBJECT, USE_TRAPEZOIDAL_WEIGHTING_ARG_NAME)
        ),
        patch_overlap_size_2pt5km_pixels=getattr(
            INPUT_ARG_OBJECT, PATCH_OVERLAP_SIZE_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
