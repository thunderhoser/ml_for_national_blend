"""Test for chiu_net_pp_architecture.py.

USE ONCE AND DESTROY.
"""

import os
import sys
import numpy
import keras

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import architecture_utils
import chiu_net_pp_architecture

INPUT_DIMENSIONS_2PT5KM_RES_KEY = 'input_dimensions_2pt5km_res'
INPUT_DIMENSIONS_10KM_RES_KEY = 'input_dimensions_10km_res'
INPUT_DIMENSIONS_20KM_RES_KEY = 'input_dimensions_20km_res'
INPUT_DIMENSIONS_40KM_RES_KEY = 'input_dimensions_40km_res'

NUM_CHANNELS_KEY = 'num_channels_by_level'
POOLING_SIZE_KEY = 'pooling_size_by_level_px'
ENCODER_NUM_CONV_LAYERS_KEY = 'encoder_num_conv_layers_by_level'
ENCODER_DROPOUT_RATES_KEY = 'encoder_dropout_rate_by_level'
DECODER_NUM_CONV_LAYERS_KEY = 'decoder_num_conv_layers_by_level'
UPSAMPLING_DROPOUT_RATES_KEY = 'upsampling_dropout_rate_by_level'
SKIP_DROPOUT_RATES_KEY = 'skip_dropout_rate_by_level'

FC_MODULE_NUM_CONV_LAYERS_KEY = 'forecast_module_num_conv_layers'
FC_MODULE_DROPOUT_RATES_KEY = 'forecast_module_dropout_rates'
FC_MODULE_USE_3D_CONV = 'forecast_module_use_3d_conv'

INCLUDE_PENULTIMATE_KEY = 'include_penultimate_conv'
PENULTIMATE_DROPOUT_RATE_KEY = 'penultimate_conv_dropout_rate'
INNER_ACTIV_FUNCTION_KEY = 'inner_activ_function_name'
INNER_ACTIV_FUNCTION_ALPHA_KEY = 'inner_activ_function_alpha'
OUTPUT_ACTIV_FUNCTION_KEY = 'output_activ_function_name'
OUTPUT_ACTIV_FUNCTION_ALPHA_KEY = 'output_activ_function_alpha'
L1_WEIGHT_KEY = 'l1_weight'
L2_WEIGHT_KEY = 'l2_weight'
USE_BATCH_NORM_KEY = 'use_batch_normalization'
ENSEMBLE_SIZE_KEY = 'ensemble_size'

NUM_OUTPUT_CHANNELS_KEY = 'num_output_channels'
LOSS_FUNCTION_KEY = 'loss_function'
OPTIMIZER_FUNCTION_KEY = 'optimizer_function'
METRIC_FUNCTIONS_KEY = 'metric_function_list'

option_dict = {
    INPUT_DIMENSIONS_2PT5KM_RES_KEY: numpy.array([1597, 2345, 6, 7], dtype=int),
    INPUT_DIMENSIONS_10KM_RES_KEY: numpy.array([399, 586, 6, 8], dtype=int),
    INPUT_DIMENSIONS_20KM_RES_KEY: numpy.array([199, 293, 6, 8], dtype=int),
    INPUT_DIMENSIONS_40KM_RES_KEY: numpy.array([99, 146, 6, 8], dtype=int),
    NUM_CHANNELS_KEY: numpy.array([8, 12, 16, 24, 32, 48, 64, 96, 96], dtype=int),
    POOLING_SIZE_KEY: numpy.full(8, 2, dtype=int),
    ENCODER_NUM_CONV_LAYERS_KEY: numpy.full(9, 2, dtype=int),
    ENCODER_DROPOUT_RATES_KEY: numpy.full(9, 0.),
    DECODER_NUM_CONV_LAYERS_KEY: numpy.full(8, 2, dtype=int),
    UPSAMPLING_DROPOUT_RATES_KEY: numpy.full(8, 0.),
    SKIP_DROPOUT_RATES_KEY: numpy.full(8, 0.),
    FC_MODULE_NUM_CONV_LAYERS_KEY: 1,
    FC_MODULE_DROPOUT_RATES_KEY: numpy.array([0.]),
    FC_MODULE_USE_3D_CONV: True,
    INCLUDE_PENULTIMATE_KEY: True,
    PENULTIMATE_DROPOUT_RATE_KEY: 0.,
    INNER_ACTIV_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    OUTPUT_ACTIV_FUNCTION_KEY: None,
    OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.,
    L1_WEIGHT_KEY: 0.,
    L2_WEIGHT_KEY: 1e-6,
    USE_BATCH_NORM_KEY: True,
    ENSEMBLE_SIZE_KEY: 10,
    NUM_OUTPUT_CHANNELS_KEY: 2,
    LOSS_FUNCTION_KEY: keras.losses.mean_squared_error,
    OPTIMIZER_FUNCTION_KEY: keras.optimizers.Adam(),
    METRIC_FUNCTIONS_KEY: []
}

chiu_net_pp_architecture.create_model(option_dict)