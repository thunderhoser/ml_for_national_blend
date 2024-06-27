"""Converts NWP-forecast files to prediction files.

An NWP-forecast file contains forecasts of all variables, for a given init time
and valid time.

A prediction file contains forecasts of only the target variables (the ones
we're trying to improve in the NBM) -- for a given init time and lead time --
along with the labels from URMA, which are considered ground truth for the
target variables.  A prediction file can be used directly as input to
evaluate_model.py.
"""

import os
import argparse
import numpy
from ml_for_national_blend.io import interp_nwp_model_io
from ml_for_national_blend.io import urma_io
from ml_for_national_blend.io import prediction_io
from ml_for_national_blend.utils import nwp_model_utils
from ml_for_national_blend.utils import urma_utils
from ml_for_national_blend.machine_learning import \
    neural_net_with_fancy_patches as neural_net
from ml_for_national_blend.outside_code import time_conversion
from ml_for_national_blend.outside_code import \
    longitude_conversion as lng_conversion
from ml_for_national_blend.outside_code import \
    temperature_conversions as temperature_conv
from ml_for_national_blend.outside_code import error_checking

# TODO(thunderhoser): The way I create NWP-ensemble files, "NaN"s in the
# wind-gust field are replaced by the sustained wind speed.  This will fuck up
# evaluation of the NWP ensembles, because these grid points should remain NaN
# for fair evaluation.

TOLERANCE_DEG = 1e-3
HOURS_TO_SECONDS = 3600
TIME_FORMAT = '%Y-%m-%d-%H'

URMA_FIELD_TO_NWP_FIELD = {
    urma_utils.TEMPERATURE_2METRE_NAME: nwp_model_utils.TEMPERATURE_2METRE_NAME,
    urma_utils.DEWPOINT_2METRE_NAME: nwp_model_utils.DEWPOINT_2METRE_NAME,
    urma_utils.U_WIND_10METRE_NAME: nwp_model_utils.U_WIND_10METRE_NAME,
    urma_utils.V_WIND_10METRE_NAME: nwp_model_utils.V_WIND_10METRE_NAME,
    urma_utils.WIND_GUST_10METRE_NAME: nwp_model_utils.WIND_GUST_10METRE_NAME
}

NWP_FORECAST_DIR_ARG_NAME = 'input_nwp_forecast_dir_name'
URMA_DIRECTORY_ARG_NAME = 'input_urma_directory_name'
NWP_MODEL_ARG_NAME = 'nwp_model_name'
FIRST_INIT_TIME_ARG_NAME = 'first_init_time_string'
LAST_INIT_TIME_ARG_NAME = 'last_init_time_string'
LEAD_TIME_ARG_NAME = 'lead_time_hours'
OUTPUT_DIR_ARG_NAME = 'output_prediction_dir_name'

NWP_FORECAST_DIR_HELP_STRING = (
    'Path to directory with NWP-forecast files.  Files therein '
    'will be found by `interp_nwp_model_io.find_file` and read by '
    '`interp_nwp_model_io.read_file`.'
)
URMA_DIRECTORY_HELP_STRING = (
    'Path to directory with URMA files (considered ground truth).  Files '
    'therein will be found by `urma_io.find_file` and read by '
    '`urma_io.read_file`.'
)
NWP_MODEL_HELP_STRING = (
    'Name of NWP model.  Must be accepted by '
    '`nwp_model_utils.check_model_name`.'
)
FIRST_INIT_TIME_HELP_STRING = (
    'First initialization time (format "yyyy-mm-dd-HH").  This script will '
    'convert NWP-forecast files to prediction files for every init time in the '
    'period `{0:s}`...`{1:s}`.'
).format(
    FIRST_INIT_TIME_ARG_NAME, LAST_INIT_TIME_ARG_NAME
)
LAST_INIT_TIME_HELP_STRING = 'See documentation for `{0:s}`.'.format(
    FIRST_INIT_TIME_ARG_NAME
)
LEAD_TIME_HELP_STRING = (
    'Will convert NWP-forecast files to prediction files for this one lead '
    'time.'
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Prediction files will be written here by '
    '`prediction_io.write_file`, to exact locations determined by '
    '`prediction_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + NWP_FORECAST_DIR_ARG_NAME, type=str, required=True,
    help=NWP_FORECAST_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + URMA_DIRECTORY_ARG_NAME, type=str, required=True,
    help=URMA_DIRECTORY_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NWP_MODEL_ARG_NAME, type=str, required=True,
    help=NWP_MODEL_HELP_STRING
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
    '--' + LEAD_TIME_ARG_NAME, type=int, required=True,
    help=LEAD_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _convert_nwp_forecasts_1init(
        nwp_forecast_file_name, urma_directory_name, lead_time_hours,
        prediction_dir_name):
    """Converts NWP forecasts from one model run (init time) to prediction file.

    :param nwp_forecast_file_name: Path to NWP-forecast file (will be read by
        `interp_nwp_model_io.read_file`).
    :param urma_directory_name: See documentation at top of this script.
    :param lead_time_hours: Same.
    :param prediction_dir_name: Same.
    """

    init_time_unix_sec = interp_nwp_model_io.file_name_to_init_time(
        nwp_forecast_file_name
    )
    valid_time_unix_sec = (
        init_time_unix_sec + lead_time_hours * HOURS_TO_SECONDS
    )
    urma_file_name = urma_io.find_file(
        directory_name=urma_directory_name,
        valid_date_string=time_conversion.unix_sec_to_string(
            valid_time_unix_sec, urma_io.DATE_FORMAT
        ),
        raise_error_if_missing=False
    )

    print('Reading NWP forecasts from: "{0:s}"...'.format(
        nwp_forecast_file_name
    ))
    nwp_forecast_table_xarray = interp_nwp_model_io.read_file(
        nwp_forecast_file_name
    )

    urma_field_names = list(URMA_FIELD_TO_NWP_FIELD.keys())
    urma_field_names.sort()

    these_matrices = [
        nwp_model_utils.get_field(
            nwp_forecast_table_xarray=nwp_forecast_table_xarray,
            field_name=URMA_FIELD_TO_NWP_FIELD[f]
        )[0, ...]
        for f in urma_field_names
    ]
    prediction_matrix = numpy.stack(these_matrices, axis=-1)

    print('Reading URMA labels from: "{0:s}"...'.format(urma_file_name))
    urma_table_xarray = urma_io.read_file(urma_file_name)

    urma_table_xarray = urma_utils.subset_by_time(
        urma_table_xarray=urma_table_xarray,
        desired_times_unix_sec=numpy.array([valid_time_unix_sec], dtype=int)
    )
    urma_table_xarray = urma_utils.subset_by_field(
        urma_table_xarray=urma_table_xarray,
        desired_field_names=urma_field_names
    )
    target_matrix = numpy.transpose(
        urma_table_xarray[urma_utils.DATA_KEY].values[0, ...],
        axes=(1, 0, 2)
    )

    k = urma_field_names.index(urma_utils.TEMPERATURE_2METRE_NAME)
    prediction_matrix[..., k] = temperature_conv.kelvins_to_celsius(
        prediction_matrix[..., k]
    )
    target_matrix[..., k] = temperature_conv.kelvins_to_celsius(
        target_matrix[..., k]
    )

    k = urma_field_names.index(urma_utils.DEWPOINT_2METRE_NAME)
    prediction_matrix[..., k] = temperature_conv.kelvins_to_celsius(
        prediction_matrix[..., k]
    )
    target_matrix[..., k] = temperature_conv.kelvins_to_celsius(
        target_matrix[..., k]
    )

    nwpft = nwp_forecast_table_xarray
    nwp_latitude_matrix_deg_n = nwpft[nwp_model_utils.LATITUDE_KEY].values
    nwp_longitude_matrix_deg_e = lng_conversion.convert_lng_positive_in_west(
        nwpft[nwp_model_utils.LONGITUDE_KEY].values
    )

    utx = urma_table_xarray
    urma_latitude_matrix_deg_n = numpy.transpose(
        utx[urma_utils.LATITUDE_KEY].values
    )
    urma_longitude_matrix_deg_e = lng_conversion.convert_lng_positive_in_west(
        numpy.transpose(utx[urma_utils.LONGITUDE_KEY].values)
    )

    assert numpy.allclose(
        nwp_latitude_matrix_deg_n, urma_latitude_matrix_deg_n,
        atol=TOLERANCE_DEG
    )
    assert numpy.allclose(
        nwp_longitude_matrix_deg_e, urma_longitude_matrix_deg_e,
        atol=TOLERANCE_DEG
    )

    output_file_name = prediction_io.find_file(
        directory_name=prediction_dir_name,
        init_time_unix_sec=init_time_unix_sec,
        raise_error_if_missing=False
    )

    dummy_training_option_dict = {
        neural_net.TARGET_LEAD_TIME_KEY: lead_time_hours,
        neural_net.TARGET_FIELDS_KEY: urma_field_names
    }
    dummy_model_file_name = '{0:s}/model.keras'.format(
        os.path.split(output_file_name)[0]
    )
    dummy_model_metafile_name = neural_net.find_metafile(
        model_file_name=dummy_model_file_name,
        raise_error_if_missing=False
    )

    print('Writing dummy metafile to: "{0:s}"...'.format(
        dummy_model_metafile_name
    ))
    neural_net.write_metafile(
        pickle_file_name=dummy_model_metafile_name,
        num_epochs=1000,
        num_training_batches_per_epoch=1000,
        training_option_dict=dummy_training_option_dict,
        num_validation_batches_per_epoch=1000,
        validation_option_dict=dummy_training_option_dict,
        loss_function_string='mse',
        optimizer_function_string='foo',
        metric_function_strings=['rmse'],
        plateau_patience_epochs=10,
        plateau_learning_rate_multiplier=0.9,
        early_stopping_patience_epochs=100,
        patch_overlap_fast_gen_2pt5km_pixels=None
    )

    print('Writing prediction file: "{0:s}"...'.format(output_file_name))
    prediction_io.write_file(
        netcdf_file_name=output_file_name,
        target_matrix=numpy.expand_dims(target_matrix, axis=0),
        prediction_matrix=numpy.expand_dims(prediction_matrix, axis=0),
        latitude_matrix_deg_n=numpy.expand_dims(
            nwp_latitude_matrix_deg_n, axis=0
        ),
        longitude_matrix_deg_e=numpy.expand_dims(
            nwp_longitude_matrix_deg_e, axis=0
        ),
        field_names=urma_field_names,
        init_times_unix_sec=numpy.array([init_time_unix_sec], dtype=int),
        model_file_name=dummy_model_file_name
    )


def _run(nwp_forecast_dir_name, urma_directory_name, nwp_model_name,
         first_init_time_string, last_init_time_string, lead_time_hours,
         prediction_dir_name):
    """Converts NWP-forecast files to prediction files.

    This is effectively the main method.

    :param nwp_forecast_dir_name: See documentation at top of this script.
    :param urma_directory_name: Same.
    :param nwp_model_name: Same.
    :param first_init_time_string: Same.
    :param last_init_time_string: Same.
    :param lead_time_hours: Same.
    :param prediction_dir_name: Same.
    """

    downsampling_factor = nwp_model_utils.model_to_nbm_downsampling_factor(
        nwp_model_name
    )
    error_checking.assert_equals(downsampling_factor, 1)

    first_init_time_unix_sec = time_conversion.string_to_unix_sec(
        first_init_time_string, TIME_FORMAT
    )
    last_init_time_unix_sec = time_conversion.string_to_unix_sec(
        last_init_time_string, TIME_FORMAT
    )
    nwp_forecast_file_names = interp_nwp_model_io.find_files_for_period(
        directory_name=nwp_forecast_dir_name,
        model_name=nwp_model_name,
        forecast_hour=lead_time_hours,
        first_init_time_unix_sec=first_init_time_unix_sec,
        last_init_time_unix_sec=last_init_time_unix_sec,
        raise_error_if_all_missing=True,
        raise_error_if_any_missing=False
    )

    for this_forecast_file_name in nwp_forecast_file_names:
        _convert_nwp_forecasts_1init(
            nwp_forecast_file_name=this_forecast_file_name,
            urma_directory_name=urma_directory_name,
            lead_time_hours=lead_time_hours,
            prediction_dir_name=prediction_dir_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        nwp_forecast_dir_name=getattr(
            INPUT_ARG_OBJECT, NWP_FORECAST_DIR_ARG_NAME
        ),
        urma_directory_name=getattr(INPUT_ARG_OBJECT, URMA_DIRECTORY_ARG_NAME),
        nwp_model_name=getattr(INPUT_ARG_OBJECT, NWP_MODEL_ARG_NAME),
        first_init_time_string=getattr(
            INPUT_ARG_OBJECT, FIRST_INIT_TIME_ARG_NAME
        ),
        last_init_time_string=getattr(
            INPUT_ARG_OBJECT, LAST_INIT_TIME_ARG_NAME
        ),
        lead_time_hours=getattr(INPUT_ARG_OBJECT, LEAD_TIME_ARG_NAME),
        prediction_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
