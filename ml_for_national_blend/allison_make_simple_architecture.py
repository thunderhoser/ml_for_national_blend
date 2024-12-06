"""Makes 'simple' Chiu-net++ architecture for Allison.

This architecture script goes with v1 of the documentation, written on Dec 5
2024.
"""

import os
import sys
import numpy
import keras

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import custom_losses
import chiu_net_pp_architecture as chiu_net_pp_arch
import file_system_utils

# TODO(Allison): You'll need to change this directory, since you don't have
# write access to my directories.
OUTPUT_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_national_blend_models/'
    'allison_simple_architecture'
)

# Define the loss function.
CHANNEL_WEIGHTS = numpy.array([
    0.02056336, 0.40520461, 0.33582517, 0.03271094, 0.20569591
])
LOSS_FUNCTION = custom_losses.dual_weighted_crpss(
    channel_weights=CHANNEL_WEIGHTS,
    temperature_index=0, u_wind_index=1, v_wind_index=2,
    dewpoint_index=3, gust_index=4,
    function_name='loss_dwcrpss'
)

# Define the optimizer.
OPTIMIZER_FUNCTION = keras.optimizers.AdamW(gradient_accumulation_steps=25)

# Define some other constants.
MODEL_DEPTH = 5
PATCH_SIZE_ONE_DIM = 208
NUM_CONV_LAYERS_PER_BLOCK = 1

OPTION_DICT = {
    chiu_net_pp_arch.INPUT_DIMENSIONS_CONST_KEY: None,  # Not using time-invariant fields in the predictors
    chiu_net_pp_arch.INPUT_DIMENSIONS_2PT5KM_RES_KEY: numpy.array(
        [PATCH_SIZE_ONE_DIM, PATCH_SIZE_ONE_DIM, 3, 15], dtype=int  # 15 variables at 2.5-km resolution
    ),
    chiu_net_pp_arch.INPUT_DIMENSIONS_10KM_RES_KEY: numpy.array(
        [PATCH_SIZE_ONE_DIM // 4, PATCH_SIZE_ONE_DIM // 4, 3, 6], dtype=int  # 6 variables at 10-km resolution
    ),
    chiu_net_pp_arch.INPUT_DIMENSIONS_20KM_RES_KEY: numpy.array(
        [PATCH_SIZE_ONE_DIM // 8, PATCH_SIZE_ONE_DIM // 8, 3, 6], dtype=int  # 6 variables at 20-km resolution
    ),
    chiu_net_pp_arch.INPUT_DIMENSIONS_40KM_RES_KEY: numpy.array(
        [PATCH_SIZE_ONE_DIM // 16, PATCH_SIZE_ONE_DIM // 16, 3, 8], dtype=int  # 8 variables at 40-km resolution
    ),
    chiu_net_pp_arch.INPUT_DIMENSIONS_2PT5KM_RCTBIAS_KEY: None,  # Not using recent NWP biases in the predictors
    chiu_net_pp_arch.INPUT_DIMENSIONS_10KM_RCTBIAS_KEY: None,
    chiu_net_pp_arch.INPUT_DIMENSIONS_20KM_RCTBIAS_KEY: None,
    chiu_net_pp_arch.INPUT_DIMENSIONS_40KM_RCTBIAS_KEY: None,
    chiu_net_pp_arch.PREDN_BASELINE_DIMENSIONS_KEY: None,  # Not doing residual prediction
    chiu_net_pp_arch.INPUT_DIMENSIONS_LAGGED_TARGETS_KEY: None,  # Not using lagged URMA truth in the predictors
    chiu_net_pp_arch.USE_RESIDUAL_BLOCKS_KEY: False,  # Using simple conv blocks, not residual blocks
    chiu_net_pp_arch.NWP_ENCODER_NUM_CHANNELS_KEY: numpy.array(
        [32, 48, 64, 96, 128, 192], dtype=int
    ),
    chiu_net_pp_arch.NWP_POOLING_SIZE_KEY: numpy.full(
        MODEL_DEPTH, 2, dtype=int
    ),
    chiu_net_pp_arch.NWP_ENCODER_NUM_CONV_LAYERS_KEY: numpy.full(
        MODEL_DEPTH + 1, NUM_CONV_LAYERS_PER_BLOCK, dtype=int
    ),
    chiu_net_pp_arch.NWP_ENCODER_DROPOUT_RATES_KEY: numpy.full(
        MODEL_DEPTH + 1, 0.
    ),
    chiu_net_pp_arch.NWP_FC_MODULE_NUM_CONV_LAYERS_KEY: 1,
    chiu_net_pp_arch.NWP_FC_MODULE_DROPOUT_RATES_KEY: numpy.array([0.]),
    chiu_net_pp_arch.NWP_FC_MODULE_USE_3D_CONV: True,
    chiu_net_pp_arch.DECODER_NUM_CHANNELS_KEY: numpy.array(
        [32, 48, 64, 96, 128], dtype=int
    ),
    chiu_net_pp_arch.DECODER_NUM_CONV_LAYERS_KEY: numpy.full(
        MODEL_DEPTH, NUM_CONV_LAYERS_PER_BLOCK, dtype=int
    ),
    chiu_net_pp_arch.UPSAMPLING_DROPOUT_RATES_KEY: numpy.full(MODEL_DEPTH, 0.),
    chiu_net_pp_arch.SKIP_DROPOUT_RATES_KEY: numpy.full(MODEL_DEPTH, 0.),
    chiu_net_pp_arch.INCLUDE_PENULTIMATE_KEY: False,
    chiu_net_pp_arch.PENULTIMATE_DROPOUT_RATE_KEY: 0.,
    chiu_net_pp_arch.INNER_ACTIV_FUNCTION_KEY: 'relu',
    chiu_net_pp_arch.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    chiu_net_pp_arch.OUTPUT_ACTIV_FUNCTION_KEY: None,
    chiu_net_pp_arch.OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.,
    chiu_net_pp_arch.L1_WEIGHT_KEY: 0.,
    chiu_net_pp_arch.L2_WEIGHT_KEY: 1e-7,
    chiu_net_pp_arch.USE_BATCH_NORM_KEY: True,
    chiu_net_pp_arch.ENSEMBLE_SIZE_KEY: 25,
    chiu_net_pp_arch.NUM_OUTPUT_CHANNELS_KEY: 5,
    chiu_net_pp_arch.PREDICT_GUST_EXCESS_KEY: False,
    chiu_net_pp_arch.PREDICT_DEWPOINT_DEPRESSION_KEY: False,
    chiu_net_pp_arch.LOSS_FUNCTION_KEY: LOSS_FUNCTION,
    chiu_net_pp_arch.OPTIMIZER_FUNCTION_KEY: OPTIMIZER_FUNCTION,
    chiu_net_pp_arch.METRIC_FUNCTIONS_KEY: [],

    # TODO(important): Any argument starting with "LAGTGT" won't be used,
    # because we don't have lagged URMA truth in the inputs.  Don't touch any of
    # these arguments.
    chiu_net_pp_arch.LAGTGT_ENCODER_NUM_CHANNELS_KEY: numpy.array([8, 12, 16, 24, 32, 48], dtype=int),
    chiu_net_pp_arch.LAGTGT_POOLING_SIZE_KEY: numpy.full(MODEL_DEPTH, 2, dtype=int),
    chiu_net_pp_arch.LAGTGT_ENCODER_NUM_CONV_LAYERS_KEY: numpy.full(MODEL_DEPTH + 1, NUM_CONV_LAYERS_PER_BLOCK, dtype=int),
    chiu_net_pp_arch.LAGTGT_ENCODER_DROPOUT_RATES_KEY: numpy.full(MODEL_DEPTH + 1, 0.),
    chiu_net_pp_arch.LAGTGT_FC_MODULE_NUM_CONV_LAYERS_KEY: 1,
    chiu_net_pp_arch.LAGTGT_FC_MODULE_DROPOUT_RATES_KEY: numpy.array([0.]),
    chiu_net_pp_arch.LAGTGT_FC_MODULE_USE_3D_CONV: True,

    # TODO(important): Any argument starting with "RCTBIAS" won't be used,
    # because we don't have recent NWP biases in the inputs.  Don't touch any of
    # these arguments.
    chiu_net_pp_arch.RCTBIAS_ENCODER_NUM_CHANNELS_KEY: numpy.array([8, 12, 16, 24, 32, 48], dtype=int),
    chiu_net_pp_arch.RCTBIAS_POOLING_SIZE_KEY: numpy.full(MODEL_DEPTH, 2, dtype=int),
    chiu_net_pp_arch.RCTBIAS_ENCODER_NUM_CONV_LAYERS_KEY: numpy.full(MODEL_DEPTH + 1, NUM_CONV_LAYERS_PER_BLOCK, dtype=int),
    chiu_net_pp_arch.RCTBIAS_ENCODER_DROPOUT_RATES_KEY: numpy.full(MODEL_DEPTH + 1, 0.),
    chiu_net_pp_arch.RCTBIAS_FC_MODULE_NUM_CONV_LAYERS_KEY: 1,
    chiu_net_pp_arch.RCTBIAS_FC_MODULE_DROPOUT_RATES_KEY: numpy.array([0.]),
    chiu_net_pp_arch.RCTBIAS_FC_MODULE_USE_3D_CONV: True
}


def _run():
    """Makes 'simple' Chiu-net++ architecture for Allison.

    This is effectively the main method.
    """

    model_object = chiu_net_pp_arch.create_model(OPTION_DICT)

    output_file_name = '{0:s}/model.keras'.format(OUTPUT_DIR_NAME)
    file_system_utils.mkdir_recursive_if_necessary(
        file_name=output_file_name
    )

    print('Writing model to: "{0:s}"...'.format(output_file_name))
    model_object.save(
        filepath=output_file_name, overwrite=True,
        include_optimizer=True
    )


if __name__ == '__main__':
    _run()
