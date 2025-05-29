"""Subsets prediction files to a smaller spatial domain."""

import argparse
import numpy
from ml_for_national_blend.outside_code import time_conversion
from ml_for_national_blend.outside_code import error_checking
from ml_for_national_blend.io import prediction_io
from ml_for_national_blend.io import region_mask_io
from ml_for_national_blend.utils import misc_utils

TIME_FORMAT = '%Y-%m-%d-%H'

INPUT_DIR_ARG_NAME = 'input_prediction_dir_name'
FIRST_INIT_TIME_ARG_NAME = 'first_init_time_string'
LAST_INIT_TIME_ARG_NAME = 'last_init_time_string'
PATCH_SIZE_ARG_NAME = 'patch_size_2pt5km_pixels'
PATCH_START_ROW_ARG_NAME = 'patch_start_row_2pt5km'
PATCH_START_COLUMN_ARG_NAME = 'patch_start_column_2pt5km'
MASK_FILE_ARG_NAME = 'region_mask_file_name'
OUTPUT_DIR_ARG_NAME = 'output_prediction_dir_name'

INPUT_DIR_HELP_STRING = (
    'Path to input directory, containing full-grid predictions.  Files therein '
    'will be found by `prediction_io.find_file` and read by '
    '`prediction_io.read_file`.'
)
FIRST_INIT_TIME_HELP_STRING = (
    'First initialization time (format "yyyy-mm-dd-HH").  This script will '
    'subset prediction files for every init time in the period '
    '`{0:s}`...`{1:s}`.'
).format(
    FIRST_INIT_TIME_ARG_NAME, LAST_INIT_TIME_ARG_NAME
)
LAST_INIT_TIME_HELP_STRING = 'See documentation for `{0:s}`.'.format(
    FIRST_INIT_TIME_ARG_NAME
)
PATCH_SIZE_HELP_STRING = (
    'Patch size (size of smaller domain), in number of pixels on the 2.5-km '
    'NBM grid.  If you would rather specify a region mask, leave this argument '
    'alone and use {0:s}.'
).format(
    MASK_FILE_ARG_NAME
)
PATCH_START_ROW_HELP_STRING = (
    '[used only if {0:s} is specified] Start row of patch (smaller domain).  '
    'If {1:s} == j, the first row of the patch is the [j]th row of the full '
    'NBM grid.'
).format(
    PATCH_SIZE_ARG_NAME, PATCH_START_ROW_ARG_NAME
)
PATCH_START_COLUMN_HELP_STRING = (
    '[used only if {0:s} is specified] Start column of patch (smaller domain).'
    '  If {1:s} == k, the first column of the patch is the [k]th column of the '
    'full NBM grid.'
).format(
    PATCH_SIZE_ARG_NAME, PATCH_START_COLUMN_ARG_NAME
)
MASK_FILE_HELP_STRING = (
    'Path to file with region mask (will be read by '
    '`region_mask_io.read_file`).  If you would rather specify a contiguous '
    'patch, leave this argument alone; use {0:s}, {1:s}, and {2:s}.'
).format(
    PATCH_SIZE_ARG_NAME, PATCH_START_ROW_ARG_NAME, PATCH_START_COLUMN_ARG_NAME
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory, where subset predictions (on the small patch '
    'domain) will be written.  Files will be written by '
    '`prediction_io.write_file`, to exact locations determined by '
    '`prediction_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_INIT_TIME_ARG_NAME, type=str, required=True,
    help=FIRST_INIT_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_INIT_TIME_ARG_NAME, type=str, required=True,
    help=LAST_INIT_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PATCH_SIZE_ARG_NAME, type=int, required=False, default=-1,
    help=PATCH_SIZE_HELP_STRING
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
    '--' + MASK_FILE_ARG_NAME, type=str, required=False, default='',
    help=MASK_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_prediction_dir_name, first_init_time_string,
         last_init_time_string, patch_size_2pt5km_pixels,
         patch_start_row_2pt5km, patch_start_column_2pt5km,
         region_mask_file_name, output_prediction_dir_name):
    """Subsets prediction files to a smaller spatial domain.

    This is effectively the main method.

    :param input_prediction_dir_name: See documentation at top of this script.
    :param first_init_time_string: Same.
    :param last_init_time_string: Same.
    :param patch_size_2pt5km_pixels: Same.
    :param patch_start_row_2pt5km: Same.
    :param patch_start_column_2pt5km: Same.
    :param region_mask_file_name: Same.
    :param output_prediction_dir_name: Same.
    """

    if patch_size_2pt5km_pixels < 1:
        patch_size_2pt5km_pixels = None

    if patch_size_2pt5km_pixels is None:
        print('Reading region mask from: "{0:s}"...'.format(
            region_mask_file_name
        ))
        region_mask_matrix = region_mask_io.read_file(region_mask_file_name)[
            region_mask_io.REGION_MASK_KEY
        ]
    else:
        error_checking.assert_is_greater(patch_size_2pt5km_pixels, 0)
        error_checking.assert_is_geq(patch_start_row_2pt5km, 0)
        error_checking.assert_is_geq(patch_start_column_2pt5km, 0)

        patch_rows = numpy.linspace(
            patch_start_row_2pt5km,
            patch_start_row_2pt5km + patch_size_2pt5km_pixels - 1,
            num=patch_size_2pt5km_pixels, dtype=int
        )
        patch_columns = numpy.linspace(
            patch_start_column_2pt5km,
            patch_start_column_2pt5km + patch_size_2pt5km_pixels - 1,
            num=patch_size_2pt5km_pixels, dtype=int
        )

    first_init_time_unix_sec = time_conversion.string_to_unix_sec(
        first_init_time_string, TIME_FORMAT
    )
    last_init_time_unix_sec = time_conversion.string_to_unix_sec(
        last_init_time_string, TIME_FORMAT
    )
    input_prediction_file_names = prediction_io.find_files_for_period(
        directory_name=input_prediction_dir_name,
        first_init_time_unix_sec=first_init_time_unix_sec,
        last_init_time_unix_sec=last_init_time_unix_sec,
        raise_error_if_all_missing=True,
        raise_error_if_any_missing=False
    )

    for this_input_file_name in input_prediction_file_names:
        print('Reading full-domain predictions from: "{0:s}"...'.format(
            this_input_file_name
        ))
        this_prediction_table_xarray = prediction_io.read_file(
            this_input_file_name
        )

        if patch_size_2pt5km_pixels is None:
            tptx = this_prediction_table_xarray
            this_target_matrix = tptx[prediction_io.TARGET_KEY].values
            this_prediction_matrix = tptx[prediction_io.PREDICTION_KEY].values

            for f in this_target_matrix.shape[-1]:
                this_target_matrix[..., f][
                    region_mask_matrix == False
                ] = numpy.nan

            for f in this_prediction_matrix.shape[-2]:
                for e in this_prediction_matrix.shape[-1]:
                    this_prediction_matrix[..., f, e][
                        region_mask_matrix == False
                    ] = numpy.nan

            tptx = tptx.assign({
                prediction_io.TARGET_KEY: (
                    tptx[prediction_io.TARGET_KEY].dims, this_target_matrix
                ),
                prediction_io.PREDICTION_KEY: (
                    tptx[prediction_io.PREDICTION_KEY].dims,
                    this_prediction_matrix
                )
            })
            this_prediction_table_xarray = tptx

            _, good_row_indices, good_column_indices = (
                misc_utils.trim_nans_from_2d_matrix(
                    numpy.nanmax(this_target_matrix, axis=-1)
                )
            )

            this_prediction_table_xarray = this_prediction_table_xarray.isel(
                {prediction_io.ROW_DIM: good_row_indices}
            )
            this_prediction_table_xarray = this_prediction_table_xarray.isel(
                {prediction_io.COLUMN_DIM: good_column_indices}
            )
        else:
            this_prediction_table_xarray = this_prediction_table_xarray.isel(
                {prediction_io.ROW_DIM: patch_rows}
            )
            this_prediction_table_xarray = this_prediction_table_xarray.isel(
                {prediction_io.COLUMN_DIM: patch_columns}
            )

        this_output_file_name = prediction_io.find_file(
            directory_name=output_prediction_dir_name,
            init_time_unix_sec=
            prediction_io.file_name_to_init_time(this_input_file_name),
            raise_error_if_missing=False
        )

        print((
            'Writing subset predictions (on small patch domain) to: "{0:s}"...'
        ).format(
            this_output_file_name
        ))

        tptx = this_prediction_table_xarray
        prediction_io.write_file(
            netcdf_file_name=this_output_file_name,
            target_matrix=tptx[prediction_io.TARGET_KEY].values,
            prediction_matrix=tptx[prediction_io.PREDICTION_KEY].values,
            latitude_matrix_deg_n=tptx[prediction_io.LATITUDE_KEY].values,
            longitude_matrix_deg_e=tptx[prediction_io.LONGITUDE_KEY].values,
            field_names=tptx[prediction_io.FIELD_NAME_KEY].values.tolist(),
            init_time_unix_sec=
            prediction_io.file_name_to_init_time(this_input_file_name),
            model_file_name=tptx.attrs[prediction_io.MODEL_FILE_KEY],
            isotonic_model_file_names=
            tptx.attrs[prediction_io.ISOTONIC_MODEL_FILES_KEY],
            uncertainty_calib_model_file_names=
            tptx.attrs[prediction_io.UNCERTAINTY_CALIB_MODEL_FILES_KEY]
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_prediction_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_init_time_string=getattr(
            INPUT_ARG_OBJECT, FIRST_INIT_TIME_ARG_NAME
        ),
        last_init_time_string=getattr(
            INPUT_ARG_OBJECT, LAST_INIT_TIME_ARG_NAME
        ),
        patch_size_2pt5km_pixels=getattr(INPUT_ARG_OBJECT, PATCH_SIZE_ARG_NAME),
        patch_start_row_2pt5km=getattr(
            INPUT_ARG_OBJECT, PATCH_START_ROW_ARG_NAME
        ),
        patch_start_column_2pt5km=getattr(
            INPUT_ARG_OBJECT, PATCH_START_COLUMN_ARG_NAME
        ),
        region_mask_file_name=getattr(INPUT_ARG_OBJECT, MASK_FILE_ARG_NAME),
        output_prediction_dir_name=getattr(
            INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME
        )
    )
