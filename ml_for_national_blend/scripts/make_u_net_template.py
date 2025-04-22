"""Makes simple U-net architecture."""

import numpy
import keras
from ml_for_national_blend.machine_learning import custom_losses
from ml_for_national_blend.machine_learning import custom_metrics
from ml_for_national_blend.machine_learning import neural_net_utils as nn_utils
from ml_for_national_blend.machine_learning import u_net_architecture
from ml_for_national_blend.outside_code import architecture_utils
from ml_for_national_blend.outside_code import file_system_utils

OUTPUT_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_national_blend_models/'
    'u_net_architecture/template'
)

# Weights for target variables.  The order is temperature, u-wind, v-wind,
# dewpoint, gust.
CHANNEL_WEIGHTS = numpy.array([
    0.02056336, 0.40520461, 0.33582517, 0.03271094, 0.20569591
])

# Create the loss function.
LOSS_FUNCTION = custom_losses.dual_weighted_crpss(
    channel_weights=CHANNEL_WEIGHTS,
    temperature_index=0, u_wind_index=1, v_wind_index=2,
    dewpoint_index=3, gust_index=4,
    function_name='loss_dwcrpss'
)

# Create a string version of the loss function.  This string is set up so that
# calling `eval(LOSS_FUNCTION_STRING)` yields `LOSS_FUNCTION`.  The string will
# be saved in a metafile.  Functions themselves can't be saved in metafiles,
# which is why this string is needed.
LOSS_FUNCTION_STRING = (
    'custom_losses.dual_weighted_crpss('
    'channel_weights=numpy.array([0.02056336, 0.40520461, 0.33582517, 0.03271094, 0.20569591]), '
    'temperature_index=0, u_wind_index=1, v_wind_index=2, '
    'dewpoint_index=3, gust_index=4, '
    'function_name="loss_dwcrpss"'
    ')'
)

# Create the metrics.
METRIC_FUNCTIONS = [
    custom_metrics.mean_squared_error(
        channel_index=0, temperature_index=0, u_wind_index=1, v_wind_index=2,
        dewpoint_index=3, gust_index=4, expect_ensemble=True,
        function_name='temp_mse_celsius2'
    ),
    custom_metrics.mean_squared_error(
        channel_index=1, temperature_index=0, u_wind_index=1, v_wind_index=2,
        dewpoint_index=3, gust_index=4, expect_ensemble=True,
        function_name='u_wind_mse_metres2_per_second2'
    ),
    custom_metrics.mean_squared_error(
        channel_index=2, temperature_index=0, u_wind_index=1, v_wind_index=2,
        dewpoint_index=3, gust_index=4, expect_ensemble=True,
        function_name='v_wind_mse_metres2_per_second2'
    ),
    custom_metrics.mean_squared_error(
        channel_index=3, temperature_index=0, u_wind_index=1, v_wind_index=2,
        dewpoint_index=3, gust_index=4, expect_ensemble=True,
        function_name='dewpoint_mse_celsius2'
    ),
    custom_metrics.mean_squared_error(
        channel_index=4, temperature_index=0, u_wind_index=1, v_wind_index=2,
        dewpoint_index=3, gust_index=4, expect_ensemble=True,
        function_name='gust_mse_metres2_per_second2'
    ),
]

# Create a string version of each metric.  Again, these are needed for the
# metafile.
METRIC_FUNCTION_STRINGS = [
    'custom_metrics.mean_squared_error('
        'channel_index=0, temperature_index=0, u_wind_index=1, v_wind_index=2, '
        'dewpoint_index=3, gust_index=4, expect_ensemble=True, '
        'function_name="temp_mse_celsius2"'
    ')',
    'custom_metrics.mean_squared_error('
        'channel_index=1, temperature_index=0, u_wind_index=1, v_wind_index=2, '
        'dewpoint_index=3, gust_index=4, expect_ensemble=True, '
        'function_name="u_wind_mse_metres2_per_second2"'
    ')',
    'custom_metrics.mean_squared_error('
        'channel_index=2, temperature_index=0, u_wind_index=1, v_wind_index=2, '
        'dewpoint_index=3, gust_index=4, expect_ensemble=True, '
        'function_name="v_wind_mse_metres2_per_second2"'
    ')',
    'custom_metrics.mean_squared_error('
        'channel_index=3, temperature_index=0, u_wind_index=1, v_wind_index=2, '
        'dewpoint_index=3, gust_index=4, expect_ensemble=True, '
        'function_name="dewpoint_mse_celsius2"'
    ')',
    'custom_metrics.mean_squared_error('
        'channel_index=4, temperature_index=0, u_wind_index=1, v_wind_index=2, '
        'dewpoint_index=3, gust_index=4, expect_ensemble=True, '
        'function_name="gust_mse_metres2_per_second2"'
    ')',
]

# Create the optimizer function.
OPTIMIZER_FUNCTION = keras.optimizers.AdamW(gradient_accumulation_steps=10)

# Create a string version of the optimizer function.  Again, this is for the
# metafile only.
OPTIMIZER_FUNCTION_STRING = (
    'keras.optimizers.AdamW(gradient_accumulation_steps=10)'
)

# Create some useful constants.
MODEL_DEPTH = 5
PATCH_SIZE_ONE_DIM = 208
NUM_CONV_BLOCKS_PER_LEVEL = 1

# Predictors will include 6 NWP models at 2.5-km resolution:
# - Raw ensemble with 8 variables (MSLP, surface pressure, 2-m temperature,
#   2-m dewpoint, 10-m u-wind, 10-m v-wind, accumulated precip, 10-m wind gust)
# - WRF-ARW with 7 variables (all of the above except gust)
# - NAM Nest with 7 variables (all of the above except gust)
# - HRRR with 7 variables (all of the above except gust)
# - Gridded LAMP with 5 variables (temp, dewp, u-wind, v-wind, gust)
# - GMOS with 5 variables (temp, dewp, u-wind, v-wind, gust)
#
# Thus, we have a total of 8 + 7 + 7 + 7 + 5 + 5 = 39 variables from these
# models.
INPUT_DIMS_2PT5KM = numpy.array(
    [PATCH_SIZE_ONE_DIM, PATCH_SIZE_ONE_DIM, 39], dtype=int
)

TARGET_FIELD_NAMES = [
    'temperature_2m_agl_kelvins', 'u_wind_10m_agl_m_s01',
    'v_wind_10m_agl_m_s01', 'dewpoint_2m_agl_kelvins', 'wind_gust_10m_agl_m_s01'
]


def _run():
    """Makes simple U-net architecture.

    This is effectively the main method.
    """

    # Put all architecture options into a dictionary.
    option_dict = {
        u_net_architecture.INPUT_DIMENSIONS_KEY: INPUT_DIMS_2PT5KM,
        u_net_architecture.NWP_ENCODER_NUM_CHANNELS_KEY:
            numpy.array([32, 48, 64, 96, 128, 192], dtype=int),
        u_net_architecture.NWP_POOLING_SIZE_KEY:
            numpy.array([2, 2, 2, 2, 2], dtype=int),
        u_net_architecture.NWP_ENCODER_NUM_CONV_LAYERS_KEY:
            numpy.array([1, 1, 1, 1, 1, 1], dtype=int),
        u_net_architecture.NWP_ENCODER_DROPOUT_RATES_KEY:
            numpy.array([0, 0, 0, 0, 0, 0], dtype=float),
        u_net_architecture.DECODER_NUM_CHANNELS_KEY:
            numpy.array([32, 48, 64, 96, 128], dtype=int),
        u_net_architecture.DECODER_NUM_CONV_LAYERS_KEY:
            numpy.array([1, 1, 1, 1, 1], dtype=int),
        u_net_architecture.UPSAMPLING_DROPOUT_RATES_KEY:
            numpy.array([0, 0, 0, 0, 0], dtype=float),
        u_net_architecture.SKIP_DROPOUT_RATES_KEY:
            numpy.array([0, 0, 0, 0, 0], dtype=float),
        u_net_architecture.INNER_ACTIV_FUNCTION_KEY:
            architecture_utils.RELU_FUNCTION_STRING,
        u_net_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
        u_net_architecture.L1_WEIGHT_KEY: 0.,
        u_net_architecture.L2_WEIGHT_KEY: 1e-7,
        u_net_architecture.USE_BATCH_NORM_KEY: True,
        u_net_architecture.ENSEMBLE_SIZE_KEY: 25,
        u_net_architecture.TARGET_FIELDS_KEY: TARGET_FIELD_NAMES,
        u_net_architecture.LOSS_FUNCTION_KEY: LOSS_FUNCTION,
        u_net_architecture.OPTIMIZER_FUNCTION_KEY: OPTIMIZER_FUNCTION,
        u_net_architecture.METRIC_FUNCTIONS_KEY: METRIC_FUNCTIONS
    }

    # Create the model architecture (or "template").  I call this a "template"
    # because, although it has the same architecture we ultimately want, it is a
    # completely untrained model.  All weights and biases in the model are
    # controlled by a random initialization done "under the hood" of Keras.
    model_object = u_net_architecture.create_model(option_dict)

    # Write the model architecture to a file.
    output_file_name = '{0:s}/model.keras'.format(OUTPUT_DIR_NAME)
    file_system_utils.mkdir_recursive_if_necessary(
        file_name=output_file_name
    )

    print('Writing model to: "{0:s}"...'.format(output_file_name))
    model_object.save(
        filepath=output_file_name, overwrite=True, include_optimizer=True
    )

    # Write the metafile.  All arguments to `neural_net_utils.write_metafile`,
    # except the first four, are dummy arguments.  You should never need to
    # change these dummy arguments.
    metafile_name = nn_utils.find_metafile(
        model_file_name=output_file_name,
        raise_error_if_missing=False
    )

    # Before writing the metafile, we must change functions to strings, since
    # functions cannot be "pickled" -- i.e., they cannot be saved to a Pickle
    # file.
    option_dict[u_net_architecture.LOSS_FUNCTION_KEY] = (
        LOSS_FUNCTION_STRING
    )
    option_dict[u_net_architecture.METRIC_FUNCTIONS_KEY] = (
        METRIC_FUNCTION_STRINGS
    )
    option_dict[u_net_architecture.OPTIMIZER_FUNCTION_KEY] = (
        OPTIMIZER_FUNCTION_STRING
    )

    nn_utils.write_metafile(
        pickle_file_name=metafile_name,
        loss_function_string=LOSS_FUNCTION_STRING,
        optimizer_function_string=OPTIMIZER_FUNCTION_STRING,
        metric_function_strings=METRIC_FUNCTION_STRINGS,
        num_epochs=100,
        use_exp_moving_average_with_decay=False,
        num_training_batches_per_epoch=32,
        training_option_dict={},
        num_validation_batches_per_epoch=16,
        validation_option_dict={},
        u_net_architecture_dict=option_dict,
        chiu_net_architecture_dict=None,
        chiu_net_pp_architecture_dict=None,
        chiu_next_pp_architecture_dict=None,
        chiu_next_ppp_architecture_dict=None,
        plateau_patience_epochs=10,
        plateau_learning_rate_multiplier=0.6,
        early_stopping_patience_epochs=50,
        patch_overlap_fast_gen_2pt5km_pixels=144,
        cosine_annealing_dict=None,
        cosine_annealing_with_restarts_dict=None
    )


if __name__ == '__main__':
    _run()
