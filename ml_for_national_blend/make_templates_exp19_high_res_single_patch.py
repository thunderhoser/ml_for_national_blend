"""Makes templates for Experiment 19 (high-res members on single patch)."""

import os
import sys
import copy
from itertools import combinations
import numpy
import keras

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import nwp_model_utils
import custom_losses
import custom_metrics
import neural_net_utils as nn_utils
import chiu_next_pp_architecture as chiu_next_pp_arch
import architecture_utils
import file_system_utils

OUTPUT_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_national_blend_models/'
    'experiment19_high_res_single_patch/templates'
)

CHANNEL_WEIGHTS = numpy.array([
    0.02056336, 0.40520461, 0.33582517, 0.03271094, 0.20569591
])

LOSS_FUNCTION = custom_losses.dual_weighted_crpss(
    channel_weights=CHANNEL_WEIGHTS,
    temperature_index=0, u_wind_index=1, v_wind_index=2,
    dewpoint_index=3, gust_index=4,
    function_name='loss_dwcrpss'
)

LOSS_FUNCTION_STRING = (
    'custom_losses.dual_weighted_crpss('
    'channel_weights=numpy.array([0.02056336, 0.40520461, 0.33582517, 0.03271094, 0.20569591]), '
    'temperature_index=0, u_wind_index=1, v_wind_index=2, '
    'dewpoint_index=3, gust_index=4, '
    'function_name="loss_dwcrpss"'
    ')'
)

METRIC_FUNCTIONS = [
    custom_metrics.max_prediction(channel_index=0, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name='temp_max_prediction_celsius'),
    custom_metrics.min_prediction(channel_index=0, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name='temp_min_prediction_celsius'),
    custom_metrics.spatial_max_bias(channel_index=0, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name='temp_spatial_max_bias'),
    custom_metrics.spatial_min_bias(channel_index=0, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name='temp_spatial_min_bias'),
    custom_metrics.mean_squared_error(channel_index=0, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name='temp_mse_celsius2'),
    custom_metrics.dual_weighted_mse(channel_index=0, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name='temp_dwmse_celsius3'),

    custom_metrics.max_prediction(channel_index=1, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name='u_wind_max_prediction_celsius'),
    custom_metrics.min_prediction(channel_index=1, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name='u_wind_min_prediction_celsius'),
    custom_metrics.spatial_max_bias(channel_index=1, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name='u_wind_spatial_max_bias'),
    custom_metrics.spatial_min_bias(channel_index=1, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name='u_wind_spatial_min_bias'),
    custom_metrics.mean_squared_error(channel_index=1, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name='u_wind_mse_celsius2'),
    custom_metrics.dual_weighted_mse(channel_index=1, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name='u_wind_dwmse_celsius3'),

    custom_metrics.max_prediction(channel_index=2, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name='v_wind_max_prediction_celsius'),
    custom_metrics.min_prediction(channel_index=2, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name='v_wind_min_prediction_celsius'),
    custom_metrics.spatial_max_bias(channel_index=2, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name='v_wind_spatial_max_bias'),
    custom_metrics.spatial_min_bias(channel_index=2, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name='v_wind_spatial_min_bias'),
    custom_metrics.mean_squared_error(channel_index=2, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name='v_wind_mse_celsius2'),
    custom_metrics.dual_weighted_mse(channel_index=2, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name='v_wind_dwmse_celsius3'),

    custom_metrics.max_prediction(channel_index=3, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name='dewpoint_max_prediction_celsius'),
    custom_metrics.min_prediction(channel_index=3, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name='dewpoint_min_prediction_celsius'),
    custom_metrics.spatial_max_bias(channel_index=3, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name='dewpoint_spatial_max_bias'),
    custom_metrics.spatial_min_bias(channel_index=3, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name='dewpoint_spatial_min_bias'),
    custom_metrics.mean_squared_error(channel_index=3, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name='dewpoint_mse_celsius2'),
    custom_metrics.dual_weighted_mse(channel_index=3, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name='dewpoint_dwmse_celsius3'),

    custom_metrics.max_prediction(channel_index=4, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name='gust_max_prediction_celsius'),
    custom_metrics.min_prediction(channel_index=4, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name='gust_min_prediction_celsius'),
    custom_metrics.spatial_max_bias(channel_index=4, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name='gust_spatial_max_bias'),
    custom_metrics.spatial_min_bias(channel_index=4, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name='gust_spatial_min_bias'),
    custom_metrics.mean_squared_error(channel_index=4, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name='gust_mse_celsius2'),
    custom_metrics.dual_weighted_mse(channel_index=4, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name='gust_dwmse_celsius3')
]

METRIC_FUNCTION_STRINGS = [
    'custom_metrics.max_prediction(channel_index=0, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name="temp_max_prediction_celsius")',
    'custom_metrics.min_prediction(channel_index=0, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name="temp_min_prediction_celsius")',
    'custom_metrics.spatial_max_bias(channel_index=0, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name="temp_spatial_max_bias")',
    'custom_metrics.spatial_min_bias(channel_index=0, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name="temp_spatial_min_bias")',
    'custom_metrics.mean_squared_error(channel_index=0, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name="temp_mse_celsius2")',
    'custom_metrics.dual_weighted_mse(channel_index=0, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name="temp_dwmse_celsius3")',

    'custom_metrics.max_prediction(channel_index=1, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name="u_wind_max_prediction_celsius")',
    'custom_metrics.min_prediction(channel_index=1, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name="u_wind_min_prediction_celsius")',
    'custom_metrics.spatial_max_bias(channel_index=1, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name="u_wind_spatial_max_bias")',
    'custom_metrics.spatial_min_bias(channel_index=1, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name="u_wind_spatial_min_bias")',
    'custom_metrics.mean_squared_error(channel_index=1, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name="u_wind_mse_celsius2")',
    'custom_metrics.dual_weighted_mse(channel_index=1, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name="u_wind_dwmse_celsius3")',

    'custom_metrics.max_prediction(channel_index=2, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name="v_wind_max_prediction_celsius")',
    'custom_metrics.min_prediction(channel_index=2, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name="v_wind_min_prediction_celsius")',
    'custom_metrics.spatial_max_bias(channel_index=2, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name="v_wind_spatial_max_bias")',
    'custom_metrics.spatial_min_bias(channel_index=2, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name="v_wind_spatial_min_bias")',
    'custom_metrics.mean_squared_error(channel_index=2, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name="v_wind_mse_celsius2")',
    'custom_metrics.dual_weighted_mse(channel_index=2, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name="v_wind_dwmse_celsius3")',

    'custom_metrics.max_prediction(channel_index=3, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name="dewpoint_max_prediction_celsius")',
    'custom_metrics.min_prediction(channel_index=3, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name="dewpoint_min_prediction_celsius")',
    'custom_metrics.spatial_max_bias(channel_index=3, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name="dewpoint_spatial_max_bias")',
    'custom_metrics.spatial_min_bias(channel_index=3, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name="dewpoint_spatial_min_bias")',
    'custom_metrics.mean_squared_error(channel_index=3, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name="dewpoint_mse_celsius2")',
    'custom_metrics.dual_weighted_mse(channel_index=3, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name="dewpoint_dwmse_celsius3")',

    'custom_metrics.max_prediction(channel_index=4, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name="gust_max_prediction_celsius")',
    'custom_metrics.min_prediction(channel_index=4, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name="gust_min_prediction_celsius")',
    'custom_metrics.spatial_max_bias(channel_index=4, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name="gust_spatial_max_bias")',
    'custom_metrics.spatial_min_bias(channel_index=4, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name="gust_spatial_min_bias")',
    'custom_metrics.mean_squared_error(channel_index=4, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name="gust_mse_celsius2")',
    'custom_metrics.dual_weighted_mse(channel_index=4, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=True, function_name="gust_dwmse_celsius3")'
]

OPTIMIZER_FUNCTION = keras.optimizers.AdamW(gradient_accumulation_steps=10)
OPTIMIZER_FUNCTION_STRING = 'keras.optimizers.AdamW(gradient_accumulation_steps=10)'

MODEL_DEPTH = 6
PATCH_SIZE_ONE_DIM = 400
NUM_CONV_BLOCKS_PER_LEVEL = 1

NUM_NWP_LEAD_TIMES = 5
NUM_LAG_TIMES = 4

ALL_MODEL_NAMES = [
    nwp_model_utils.ENSEMBLE_MODEL_NAME, nwp_model_utils.HRRR_MODEL_NAME,
    nwp_model_utils.WRF_ARW_MODEL_NAME, nwp_model_utils.NAM_NEST_MODEL_NAME,
    nwp_model_utils.GRIDDED_MOS_MODEL_NAME
]

THIS_LIST = [list(combinations(ALL_MODEL_NAMES, r)) for r in [3, 4, 5]]
MODEL_NAME_SETS = [list(comb) for sublist in THIS_LIST for comb in sublist]

MODEL_NAME_TO_NUM_PREDICTORS = {
    nwp_model_utils.ENSEMBLE_MODEL_NAME: 8,
    nwp_model_utils.HRRR_MODEL_NAME: 7,
    nwp_model_utils.WRF_ARW_MODEL_NAME: 7,
    nwp_model_utils.NAM_NEST_MODEL_NAME: 7,
    nwp_model_utils.GRIDDED_MOS_MODEL_NAME: 5
}

INPUT_DIMS_LAGGED_TARGETS = numpy.array(
    [PATCH_SIZE_ONE_DIM, PATCH_SIZE_ONE_DIM, NUM_LAG_TIMES, 5], dtype=int
)
TARGET_FIELD_NAMES = [
    'temperature_2m_agl_kelvins', 'u_wind_10m_agl_m_s01',
    'v_wind_10m_agl_m_s01', 'dewpoint_2m_agl_kelvins', 'wind_gust_10m_agl_m_s01'
]

DEFAULT_OPTION_DICT = {
    chiu_next_pp_arch.INPUT_DIMENSIONS_CONST_KEY: numpy.array(
        [PATCH_SIZE_ONE_DIM, PATCH_SIZE_ONE_DIM, 4], dtype=int),
    # chiu_next_pp_arch.INPUT_DIMENSIONS_2PT5KM_RES_KEY: INPUT_DIMS_2PT5KM,
    chiu_next_pp_arch.INPUT_DIMENSIONS_10KM_RES_KEY: None,
    chiu_next_pp_arch.INPUT_DIMENSIONS_20KM_RES_KEY: None,
    chiu_next_pp_arch.INPUT_DIMENSIONS_40KM_RES_KEY: None,
    chiu_next_pp_arch.INPUT_DIMENSIONS_LAGGED_TARGETS_KEY:
        INPUT_DIMS_LAGGED_TARGETS,
    chiu_next_pp_arch.INPUT_DIMENSIONS_2PT5KM_RCTBIAS_KEY: None,
    chiu_next_pp_arch.INPUT_DIMENSIONS_10KM_RCTBIAS_KEY: None,
    chiu_next_pp_arch.INPUT_DIMENSIONS_20KM_RCTBIAS_KEY: None,
    chiu_next_pp_arch.INPUT_DIMENSIONS_40KM_RCTBIAS_KEY: None,
    chiu_next_pp_arch.PREDN_BASELINE_DIMENSIONS_KEY: numpy.array([PATCH_SIZE_ONE_DIM, PATCH_SIZE_ONE_DIM, 5], dtype=int),

    # chiu_next_pp_arch.DO_CONVNEXT_V2_KEY: True,
    # chiu_next_pp_arch.USE_SPECTRAL_NORM_KEY: True,

    chiu_next_pp_arch.NWP_ENCODER_NUM_CHANNELS_KEY: numpy.array([24, 32, 48, 64, 96, 128, 192], dtype=int),
    chiu_next_pp_arch.NWP_POOLING_SIZE_KEY: numpy.full(MODEL_DEPTH, 2, dtype=int),
    chiu_next_pp_arch.NWP_ENCODER_NUM_CONV_BLOCKS_KEY: numpy.full(MODEL_DEPTH + 1, NUM_CONV_BLOCKS_PER_LEVEL, dtype=int),
    # chiu_next_pp_arch.NWP_ENCODER_DROPOUT_RATES_KEY: numpy.full(MODEL_DEPTH + 1, 0.1),
    # chiu_next_pp_arch.NWP_FC_MODULE_NUM_CONV_BLOCKS_KEY: numpy.concatenate([
    #     numpy.full(MODEL_DEPTH, NUM_CONV_BLOCKS_PER_LEVEL, dtype=int),
    #     numpy.array([3], dtype=int)
    # ]),
    # chiu_next_pp_arch.NWP_FC_MODULE_DROPOUT_RATES_KEY: numpy.full(MODEL_DEPTH + 1, 0.1),
    chiu_next_pp_arch.NWP_FC_MODULE_USE_3D_CONV: True,

    chiu_next_pp_arch.LAGTGT_ENCODER_NUM_CHANNELS_KEY: numpy.array([8, 12, 16, 24, 32, 48, 64], dtype=int),
    chiu_next_pp_arch.LAGTGT_POOLING_SIZE_KEY: numpy.full(MODEL_DEPTH, 2, dtype=int),
    chiu_next_pp_arch.LAGTGT_ENCODER_NUM_CONV_BLOCKS_KEY: numpy.full(MODEL_DEPTH + 1, NUM_CONV_BLOCKS_PER_LEVEL, dtype=int),
    # chiu_next_pp_arch.LAGTGT_ENCODER_DROPOUT_RATES_KEY: numpy.full(MODEL_DEPTH + 1, 0.1),
    # chiu_next_pp_arch.LAGTGT_FC_MODULE_NUM_CONV_BLOCKS_KEY: numpy.concatenate([
    #     numpy.full(MODEL_DEPTH, NUM_CONV_BLOCKS_PER_LEVEL, dtype=int),
    #     numpy.array([3], dtype=int)
    # ]),
    # chiu_next_pp_arch.LAGTGT_FC_MODULE_DROPOUT_RATES_KEY: numpy.full(MODEL_DEPTH + 1, 0.1),
    chiu_next_pp_arch.LAGTGT_FC_MODULE_USE_3D_CONV: True,

    chiu_next_pp_arch.RCTBIAS_ENCODER_NUM_CHANNELS_KEY: numpy.array([8, 12, 16, 24, 32, 48, 64], dtype=int),
    chiu_next_pp_arch.RCTBIAS_POOLING_SIZE_KEY: numpy.full(MODEL_DEPTH, 2, dtype=int),
    chiu_next_pp_arch.RCTBIAS_ENCODER_NUM_CONV_BLOCKS_KEY: numpy.full(MODEL_DEPTH + 1, NUM_CONV_BLOCKS_PER_LEVEL, dtype=int),
    # chiu_next_pp_arch.RCTBIAS_ENCODER_DROPOUT_RATES_KEY: numpy.full(MODEL_DEPTH + 1, 0.1),
    # chiu_next_pp_arch.RCTBIAS_FC_MODULE_NUM_CONV_BLOCKS_KEY: numpy.concatenate([
    #     numpy.full(MODEL_DEPTH, NUM_CONV_BLOCKS_PER_LEVEL, dtype=int),
    #     numpy.array([3], dtype=int)
    # ]),
    # chiu_next_pp_arch.RCTBIAS_FC_MODULE_DROPOUT_RATES_KEY: numpy.full(MODEL_DEPTH + 1, 0.1),
    chiu_next_pp_arch.RCTBIAS_FC_MODULE_USE_3D_CONV: True,

    chiu_next_pp_arch.DECODER_NUM_CHANNELS_KEY: numpy.array([32, 44, 64, 88, 128, 176], dtype=int),
    chiu_next_pp_arch.DECODER_NUM_CONV_BLOCKS_KEY: numpy.full(MODEL_DEPTH, NUM_CONV_BLOCKS_PER_LEVEL, dtype=int),
    # chiu_next_pp_arch.UPSAMPLING_DROPOUT_RATES_KEY: numpy.full(MODEL_DEPTH, 0.1),
    chiu_next_pp_arch.SKIP_DROPOUT_RATES_KEY: numpy.full(MODEL_DEPTH, 0.),

    chiu_next_pp_arch.INCLUDE_PENULTIMATE_KEY: False,
    chiu_next_pp_arch.PENULTIMATE_DROPOUT_RATE_KEY: 0.,
    chiu_next_pp_arch.OUTPUT_ACTIV_FUNCTION_KEY: None,
    chiu_next_pp_arch.OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.,
    chiu_next_pp_arch.L1_WEIGHT_KEY: 0.,
    chiu_next_pp_arch.L2_WEIGHT_KEY: 1e-7,
    chiu_next_pp_arch.ENSEMBLE_SIZE_KEY: 25,
    chiu_next_pp_arch.TARGET_FIELDS_KEY: TARGET_FIELD_NAMES,
    chiu_next_pp_arch.LOSS_FUNCTION_KEY: LOSS_FUNCTION,
    chiu_next_pp_arch.METRIC_FUNCTIONS_KEY: METRIC_FUNCTIONS,
    chiu_next_pp_arch.OPTIMIZER_FUNCTION_KEY: OPTIMIZER_FUNCTION
}


def _run():
    """Makes templates for Experiment 19 (high-res members on single patch).

    This is effectively the main method.
    """

    for i in range(len(MODEL_NAME_SETS)):
        model_names = MODEL_NAME_SETS[i]
        num_predictors = numpy.sum(numpy.array(
            [MODEL_NAME_TO_NUM_PREDICTORS[n] for n in model_names], dtype=int
        ))

        input_dims_2pt5km = numpy.array([
            PATCH_SIZE_ONE_DIM, PATCH_SIZE_ONE_DIM,
            NUM_NWP_LEAD_TIMES, num_predictors
        ], dtype=int)

        encoder_dropout_rates = numpy.full(MODEL_DEPTH + 1, 0.)
        decoder_dropout_rates = numpy.full(MODEL_DEPTH, 0.)

        fc_module_conv_block_counts = numpy.concatenate([
            numpy.full(MODEL_DEPTH, NUM_CONV_BLOCKS_PER_LEVEL, dtype=int),
            numpy.array([1], dtype=int)
        ])

        option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)

        option_dict.update({
            chiu_next_pp_arch.INPUT_DIMENSIONS_2PT5KM_RES_KEY: input_dims_2pt5km,

            chiu_next_pp_arch.DO_CONVNEXT_V2_KEY: False,
            chiu_next_pp_arch.USE_SPECTRAL_NORM_KEY: False,

            chiu_next_pp_arch.NWP_ENCODER_DROPOUT_RATES_KEY: encoder_dropout_rates,
            chiu_next_pp_arch.NWP_FC_MODULE_NUM_CONV_BLOCKS_KEY: fc_module_conv_block_counts,
            chiu_next_pp_arch.NWP_FC_MODULE_DROPOUT_RATES_KEY: encoder_dropout_rates,

            chiu_next_pp_arch.LAGTGT_ENCODER_DROPOUT_RATES_KEY: encoder_dropout_rates,
            chiu_next_pp_arch.LAGTGT_FC_MODULE_NUM_CONV_BLOCKS_KEY: fc_module_conv_block_counts,
            chiu_next_pp_arch.LAGTGT_FC_MODULE_DROPOUT_RATES_KEY: encoder_dropout_rates,

            chiu_next_pp_arch.RCTBIAS_ENCODER_DROPOUT_RATES_KEY: encoder_dropout_rates,
            chiu_next_pp_arch.RCTBIAS_FC_MODULE_NUM_CONV_BLOCKS_KEY: fc_module_conv_block_counts,
            chiu_next_pp_arch.RCTBIAS_FC_MODULE_DROPOUT_RATES_KEY: encoder_dropout_rates,

            chiu_next_pp_arch.UPSAMPLING_DROPOUT_RATES_KEY: decoder_dropout_rates
        })

        model_object = chiu_next_pp_arch.create_model(option_dict)

        model_names_one_string = '_'.join(
            [n.replace('_', '-') for n in model_names]
        )
        output_file_name = '{0:s}/nwp-inputs={1:s}/model.keras'.format(
            OUTPUT_DIR_NAME, model_names_one_string
        )
        file_system_utils.mkdir_recursive_if_necessary(
            file_name=output_file_name
        )

        print('Writing model to: "{0:s}"...'.format(output_file_name))
        model_object.save(
            filepath=output_file_name, overwrite=True, include_optimizer=True
        )

        metafile_name = nn_utils.find_metafile(
            model_file_name=output_file_name,
            raise_error_if_missing=False
        )
        option_dict[chiu_next_pp_arch.LOSS_FUNCTION_KEY] = (
            LOSS_FUNCTION_STRING
        )
        option_dict[chiu_next_pp_arch.METRIC_FUNCTIONS_KEY] = (
            METRIC_FUNCTION_STRINGS
        )
        option_dict[chiu_next_pp_arch.OPTIMIZER_FUNCTION_KEY] = (
            OPTIMIZER_FUNCTION_STRING
        )

        nn_utils.write_metafile(
            pickle_file_name=metafile_name,
            num_epochs=100,
            use_exp_moving_average_with_decay=False,
            num_training_batches_per_epoch=32,
            training_option_dict={},
            num_validation_batches_per_epoch=16,
            validation_option_dict={},
            loss_function_string=LOSS_FUNCTION_STRING,
            optimizer_function_string=OPTIMIZER_FUNCTION_STRING,
            metric_function_strings=METRIC_FUNCTION_STRINGS,
            chiu_net_architecture_dict=None,
            chiu_net_pp_architecture_dict=None,
            chiu_next_pp_architecture_dict=option_dict,
            plateau_patience_epochs=10,
            plateau_learning_rate_multiplier=0.6,
            early_stopping_patience_epochs=50,
            patch_overlap_fast_gen_2pt5km_pixels=144,
            temporary_predictor_dir_name=None
        )


if __name__ == '__main__':
    _run()
