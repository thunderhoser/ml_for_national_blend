"""Makes simple Chiu-net++ architecture."""

import numpy
import keras
from ml_for_national_blend.machine_learning import custom_losses
from ml_for_national_blend.machine_learning import custom_metrics
from ml_for_national_blend.machine_learning import neural_net_utils as nn_utils
from ml_for_national_blend.machine_learning import \
    chiu_net_pp_architecture as chiu_net_pp_arch
from ml_for_national_blend.outside_code import architecture_utils
from ml_for_national_blend.outside_code import file_system_utils

OUTPUT_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_national_blend_models/'
    'simple_architecture/template'
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
        function_name='u_wind_mse_celsius2'
    ),
    custom_metrics.mean_squared_error(
        channel_index=2, temperature_index=0, u_wind_index=1, v_wind_index=2,
        dewpoint_index=3, gust_index=4, expect_ensemble=True,
        function_name='v_wind_mse_celsius2'
    ),
    custom_metrics.mean_squared_error(
        channel_index=3, temperature_index=0, u_wind_index=1, v_wind_index=2,
        dewpoint_index=3, gust_index=4, expect_ensemble=True,
        function_name='dewpoint_mse_celsius2'
    ),
    custom_metrics.mean_squared_error(
        channel_index=4, temperature_index=0, u_wind_index=1, v_wind_index=2,
        dewpoint_index=3, gust_index=4, expect_ensemble=True,
        function_name='gust_mse_celsius2'
    )
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
    'function_name="u_wind_mse_celsius2"'
    ')',
    'custom_metrics.mean_squared_error('
    'channel_index=2, temperature_index=0, u_wind_index=1, v_wind_index=2, '
    'dewpoint_index=3, gust_index=4, expect_ensemble=True, '
    'function_name="v_wind_mse_celsius2"'
    ')',
    'custom_metrics.mean_squared_error('
    'channel_index=3, temperature_index=0, u_wind_index=1, v_wind_index=2, '
    'dewpoint_index=3, gust_index=4, expect_ensemble=True, '
    'function_name="dewpoint_mse_celsius2"'
    ')',
    'custom_metrics.mean_squared_error('
    'channel_index=4, temperature_index=0, u_wind_index=1, v_wind_index=2, '
    'dewpoint_index=3, gust_index=4, expect_ensemble=True, '
    'function_name="gust_mse_celsius2"'
    ')'
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
NUM_NWP_LEAD_TIMES = 5

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
    [PATCH_SIZE_ONE_DIM, PATCH_SIZE_ONE_DIM, NUM_NWP_LEAD_TIMES, 39], dtype=int
)

# Predictors will include two NWP models at 10-km resolution:
# - NAM with 7 variables (MSLP, surface pressure, 2-m temperature,
#   2-m dewpoint, 10-m u-wind, 10-m v-wind, accumulated precip)
# - RAP with the same 7 variables
INPUT_DIMS_10KM = numpy.array(
    [PATCH_SIZE_ONE_DIM // 4, PATCH_SIZE_ONE_DIM // 4, NUM_NWP_LEAD_TIMES, 14],
    dtype=int
)

# Predictors will include two NWP models at 20-km resolution:
# - GFS with 7 variables (MSLP, surface pressure, 2-m temperature,
#   2-m dewpoint, 10-m u-wind, 10-m v-wind, accumulated precip)
# - ECMWF with the same 7 variables
INPUT_DIMS_20KM = numpy.array(
    [PATCH_SIZE_ONE_DIM // 8, PATCH_SIZE_ONE_DIM // 8, NUM_NWP_LEAD_TIMES, 14],
    dtype=int
)

# Predictors will include one NWP model at 40-km resolution:
# - GEFS with 7 variables (MSLP, surface pressure, 2-m temperature,
#   2-m dewpoint, 10-m u-wind, 10-m v-wind, accumulated precip)
INPUT_DIMS_40KM = numpy.array(
    [PATCH_SIZE_ONE_DIM // 16, PATCH_SIZE_ONE_DIM // 16, NUM_NWP_LEAD_TIMES, 7],
    dtype=int
)

TARGET_FIELD_NAMES = [
    'temperature_2m_agl_kelvins', 'u_wind_10m_agl_m_s01',
    'v_wind_10m_agl_m_s01', 'dewpoint_2m_agl_kelvins', 'wind_gust_10m_agl_m_s01'
]


def _run():
    """Makes simple Chiu-net++ architecture.

    This is effectively the main method.
    """

    # Put all architecture options into a dictionary.
    option_dict = {
        chiu_net_pp_arch.INPUT_DIMENSIONS_CONST_KEY: None,
        chiu_net_pp_arch.INPUT_DIMENSIONS_2PT5KM_RES_KEY: INPUT_DIMS_2PT5KM,
        chiu_net_pp_arch.INPUT_DIMENSIONS_10KM_RES_KEY: INPUT_DIMS_10KM,
        chiu_net_pp_arch.INPUT_DIMENSIONS_20KM_RES_KEY: INPUT_DIMS_20KM,
        chiu_net_pp_arch.INPUT_DIMENSIONS_40KM_RES_KEY: INPUT_DIMS_40KM,
        chiu_net_pp_arch.INPUT_DIMENSIONS_LAGGED_TARGETS_KEY: None,
        chiu_net_pp_arch.INPUT_DIMENSIONS_2PT5KM_RCTBIAS_KEY: None,
        chiu_net_pp_arch.INPUT_DIMENSIONS_10KM_RCTBIAS_KEY: None,
        chiu_net_pp_arch.INPUT_DIMENSIONS_20KM_RCTBIAS_KEY: None,
        chiu_net_pp_arch.INPUT_DIMENSIONS_40KM_RCTBIAS_KEY: None,
        chiu_net_pp_arch.PREDN_BASELINE_DIMENSIONS_KEY: None,
        chiu_net_pp_arch.USE_RESIDUAL_BLOCKS_KEY: False,

        chiu_net_pp_arch.NWP_ENCODER_NUM_CHANNELS_KEY:
            numpy.array([32, 48, 64, 96, 128, 192], dtype=int),
        chiu_net_pp_arch.NWP_POOLING_SIZE_KEY:
            numpy.full(MODEL_DEPTH, 2, dtype=int),
        chiu_net_pp_arch.NWP_ENCODER_NUM_CONV_LAYERS_KEY:
            numpy.full(MODEL_DEPTH + 1, NUM_CONV_BLOCKS_PER_LEVEL, dtype=int),
        chiu_net_pp_arch.NWP_ENCODER_DROPOUT_RATES_KEY:
            numpy.full(MODEL_DEPTH + 1, 0.),
        chiu_net_pp_arch.NWP_FC_MODULE_NUM_CONV_LAYERS_KEY:
            NUM_CONV_BLOCKS_PER_LEVEL,
        chiu_net_pp_arch.NWP_FC_MODULE_DROPOUT_RATES_KEY: numpy.array([0.]),
        chiu_net_pp_arch.NWP_FC_MODULE_USE_3D_CONV: True,

        # The seven arguments below are all dummy arguments, because this NN
        # does not use lagged URMA (target) fields in the predictors, as
        # indicated by the argument
        # `chiu_net_pp_arch.INPUT_DIMENSIONS_LAGGED_TARGETS_KEY: None`.
        chiu_net_pp_arch.LAGTGT_ENCODER_NUM_CHANNELS_KEY:
            numpy.array([8, 12, 16, 24, 32, 48], dtype=int),
        chiu_net_pp_arch.LAGTGT_POOLING_SIZE_KEY:
            numpy.full(MODEL_DEPTH, 2, dtype=int),
        chiu_net_pp_arch.LAGTGT_ENCODER_NUM_CONV_LAYERS_KEY:
            numpy.full(MODEL_DEPTH + 1, NUM_CONV_BLOCKS_PER_LEVEL, dtype=int),
        chiu_net_pp_arch.LAGTGT_ENCODER_DROPOUT_RATES_KEY:
            numpy.full(MODEL_DEPTH + 1, 0.),
        chiu_net_pp_arch.LAGTGT_FC_MODULE_NUM_CONV_LAYERS_KEY:
            NUM_CONV_BLOCKS_PER_LEVEL,
        chiu_net_pp_arch.LAGTGT_FC_MODULE_DROPOUT_RATES_KEY: numpy.array([0.]),
        chiu_net_pp_arch.LAGTGT_FC_MODULE_USE_3D_CONV: True,

        # The seven arguments below are all dummy arguments, because this NN
        # does not use recent NWP biases in the predictors, as indicated by the
        # arguments `chiu_net_pp_arch.INPUT_DIMENSIONS_2PT5KM_RCTBIAS_KEY: None`
        # and analogous arguments for 10-, 20-, 40-km data.
        chiu_net_pp_arch.RCTBIAS_ENCODER_NUM_CHANNELS_KEY:
            numpy.array([8, 12, 16, 24, 32, 48], dtype=int),
        chiu_net_pp_arch.RCTBIAS_POOLING_SIZE_KEY:
            numpy.full(MODEL_DEPTH, 2, dtype=int),
        chiu_net_pp_arch.RCTBIAS_ENCODER_NUM_CONV_LAYERS_KEY:
            numpy.full(MODEL_DEPTH + 1, NUM_CONV_BLOCKS_PER_LEVEL, dtype=int),
        chiu_net_pp_arch.RCTBIAS_ENCODER_DROPOUT_RATES_KEY:
            numpy.full(MODEL_DEPTH + 1, 0.),
        chiu_net_pp_arch.RCTBIAS_FC_MODULE_NUM_CONV_LAYERS_KEY:
            NUM_CONV_BLOCKS_PER_LEVEL,
        chiu_net_pp_arch.RCTBIAS_FC_MODULE_DROPOUT_RATES_KEY: numpy.array([0.]),
        chiu_net_pp_arch.RCTBIAS_FC_MODULE_USE_3D_CONV: True,

        chiu_net_pp_arch.DECODER_NUM_CHANNELS_KEY:
            numpy.array([32, 48, 64, 96, 128], dtype=int),
        chiu_net_pp_arch.DECODER_NUM_CONV_LAYERS_KEY:
            numpy.full(MODEL_DEPTH, NUM_CONV_BLOCKS_PER_LEVEL, dtype=int),
        chiu_net_pp_arch.UPSAMPLING_DROPOUT_RATES_KEY: numpy.full(MODEL_DEPTH, 0.),
        chiu_net_pp_arch.SKIP_DROPOUT_RATES_KEY: numpy.full(MODEL_DEPTH, 0.),

        chiu_net_pp_arch.INCLUDE_PENULTIMATE_KEY: False,
        chiu_net_pp_arch.PENULTIMATE_DROPOUT_RATE_KEY: 0.,
        chiu_net_pp_arch.INNER_ACTIV_FUNCTION_KEY:
            architecture_utils.RELU_FUNCTION_STRING,
        chiu_net_pp_arch.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
        chiu_net_pp_arch.OUTPUT_ACTIV_FUNCTION_KEY: None,
        chiu_net_pp_arch.OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.,
        chiu_net_pp_arch.L1_WEIGHT_KEY: 0.,
        chiu_net_pp_arch.L2_WEIGHT_KEY: 1e-7,
        chiu_net_pp_arch.USE_BATCH_NORM_KEY: True,
        chiu_net_pp_arch.ENSEMBLE_SIZE_KEY: 25,
        chiu_net_pp_arch.TARGET_FIELDS_KEY: TARGET_FIELD_NAMES,
        chiu_net_pp_arch.LOSS_FUNCTION_KEY: LOSS_FUNCTION,
        chiu_net_pp_arch.OPTIMIZER_FUNCTION_KEY: OPTIMIZER_FUNCTION,
        chiu_net_pp_arch.METRIC_FUNCTIONS_KEY: METRIC_FUNCTIONS
    }

    # Create the model architecture (or "template").  I call this a "template"
    # because, although it has the same architecture we ultimately want, it is a
    # completely untrained model.  All weights and biases in the model are
    # controlled by a random initialization done "under the hood" of Keras.
    model_object = chiu_net_pp_arch.create_model(option_dict)

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
    option_dict[chiu_net_pp_arch.LOSS_FUNCTION_KEY] = (
        LOSS_FUNCTION_STRING
    )
    option_dict[chiu_net_pp_arch.METRIC_FUNCTIONS_KEY] = (
        METRIC_FUNCTION_STRINGS
    )
    option_dict[chiu_net_pp_arch.OPTIMIZER_FUNCTION_KEY] = (
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
        u_net_architecture_dict=None,
        chiu_net_architecture_dict=None,
        chiu_net_pp_architecture_dict=None,
        chiu_next_pp_architecture_dict=option_dict,
        plateau_patience_epochs=10,
        plateau_learning_rate_multiplier=0.6,
        early_stopping_patience_epochs=50,
        patch_overlap_fast_gen_2pt5km_pixels=144
    )


if __name__ == '__main__':
    _run()
