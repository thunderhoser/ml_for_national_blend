"""Creates Chiu-net++ templates for Experiment 10.

Same as Experiment 9 but with uncertainty quantification.
"""

import os
import sys
import copy
import numpy
import keras

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import custom_losses
import custom_metrics
import neural_net
import chiu_net_pp_architecture as chiu_net_pp_arch
import architecture_utils
import file_system_utils

OUTPUT_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_national_blend_models/'
    'experiment10_recent_biases_crps/templates'
)

LOSS_FUNCTION = custom_losses.dual_weighted_crpss(
    channel_weights=numpy.array([1.]),
    u_wind_index=-1, v_wind_index=-1, gust_index=-1,
    temperature_index=-1, dewpoint_index=-1,
    function_name='loss_dwmsess'
)

LOSS_FUNCTION_STRING = (
    'custom_losses.dual_weighted_crpss('
    'channel_weights=numpy.array([1.]), '
    'u_wind_index=-1, v_wind_index=-1, gust_index=-1, '
    'temperature_index=-1, dewpoint_index=-1, '
    'function_name="loss_dwmsess"'
    ')'
)

METRIC_FUNCTIONS = [
    custom_metrics.max_prediction(channel_index=0, temperature_index=0, u_wind_index=-1, v_wind_index=-1, dewpoint_index=-1, gust_index=-1, expect_ensemble=True, function_name='temp_max_prediction_celsius'),
    custom_metrics.min_prediction(channel_index=0, temperature_index=0, u_wind_index=-1, v_wind_index=-1, dewpoint_index=-1, gust_index=-1, expect_ensemble=True, function_name='temp_min_prediction_celsius'),
    custom_metrics.spatial_max_bias(channel_index=0, temperature_index=0, u_wind_index=-1, v_wind_index=-1, dewpoint_index=-1, gust_index=-1, expect_ensemble=True, function_name='temp_spatial_max_bias'),
    custom_metrics.spatial_min_bias(channel_index=0, temperature_index=0, u_wind_index=-1, v_wind_index=-1, dewpoint_index=-1, gust_index=-1, expect_ensemble=True, function_name='temp_spatial_min_bias'),
    custom_metrics.mean_squared_error(channel_index=0, temperature_index=0, u_wind_index=-1, v_wind_index=-1, dewpoint_index=-1, gust_index=-1, expect_ensemble=True, function_name='temp_mse_celsius2'),
    custom_metrics.dual_weighted_mse(channel_index=0, temperature_index=0, u_wind_index=-1, v_wind_index=-1, dewpoint_index=-1, gust_index=-1, expect_ensemble=True, function_name='temp_dwmse_celsius3')
]

METRIC_FUNCTION_STRINGS = [
    'custom_metrics.max_prediction(channel_index=0, temperature_index=0, u_wind_index=-1, v_wind_index=-1, dewpoint_index=-1, gust_index=-1, expect_ensemble=True, function_name="temp_max_prediction_celsius")',
    'custom_metrics.min_prediction(channel_index=0, temperature_index=0, u_wind_index=-1, v_wind_index=-1, dewpoint_index=-1, gust_index=-1, expect_ensemble=True, function_name="temp_min_prediction_celsius")',
    'custom_metrics.spatial_max_bias(channel_index=0, temperature_index=0, u_wind_index=-1, v_wind_index=-1, dewpoint_index=-1, gust_index=-1, expect_ensemble=True, function_name="temp_spatial_max_bias")',
    'custom_metrics.spatial_min_bias(channel_index=0, temperature_index=0, u_wind_index=-1, v_wind_index=-1, dewpoint_index=-1, gust_index=-1, expect_ensemble=True, function_name="temp_spatial_min_bias")',
    'custom_metrics.mean_squared_error(channel_index=0, temperature_index=0, u_wind_index=-1, v_wind_index=-1, dewpoint_index=-1, gust_index=-1, expect_ensemble=True, function_name="temp_mse_celsius2")',
    'custom_metrics.dual_weighted_mse(channel_index=0, temperature_index=0, u_wind_index=-1, v_wind_index=-1, dewpoint_index=-1, gust_index=-1, expect_ensemble=True, function_name="temp_dwmse_celsius3")'
]

OPTIMIZER_FUNCTION = keras.optimizers.Nadam(gradient_accumulation_steps=5)
OPTIMIZER_FUNCTION_STRING = 'keras.optimizers.Nadam(gradient_accumulation_steps=5)'

NUM_CONV_LAYERS_PER_BLOCK = 1

DEFAULT_OPTION_DICT = {
    chiu_net_pp_arch.INPUT_DIMENSIONS_CONST_KEY: numpy.array([432, 432, 4], dtype=int),
    # chiu_net_pp_arch.INPUT_DIMENSIONS_2PT5KM_RES_KEY: numpy.array([432, 432, 2, 22], dtype=int),
    chiu_net_pp_arch.INPUT_DIMENSIONS_10KM_RES_KEY: None,
    chiu_net_pp_arch.INPUT_DIMENSIONS_20KM_RES_KEY: None,
    chiu_net_pp_arch.INPUT_DIMENSIONS_40KM_RES_KEY: None,
    # chiu_net_pp_arch.INPUT_DIMENSIONS_2PT5KM_RCTBIAS_KEY: numpy.array([432, 432, 2, 3], dtype=int),
    chiu_net_pp_arch.INPUT_DIMENSIONS_10KM_RCTBIAS_KEY: None,
    chiu_net_pp_arch.INPUT_DIMENSIONS_20KM_RCTBIAS_KEY: None,
    chiu_net_pp_arch.INPUT_DIMENSIONS_40KM_RCTBIAS_KEY: None,
    chiu_net_pp_arch.PREDN_BASELINE_DIMENSIONS_KEY: numpy.array([432, 432, 1], dtype=int),
    # chiu_net_pp_arch.INPUT_DIMENSIONS_LAGGED_TARGETS_KEY: numpy.array([432, 432, 2, 1], dtype=int),
    chiu_net_pp_arch.USE_RESIDUAL_BLOCKS_KEY: False,
    chiu_net_pp_arch.NWP_ENCODER_NUM_CHANNELS_KEY: numpy.array([32, 48, 64, 96, 128, 192, 256], dtype=int),
    chiu_net_pp_arch.NWP_POOLING_SIZE_KEY: numpy.full(6, 2, dtype=int),
    chiu_net_pp_arch.NWP_ENCODER_NUM_CONV_LAYERS_KEY: numpy.full(7, NUM_CONV_LAYERS_PER_BLOCK, dtype=int),
    chiu_net_pp_arch.NWP_ENCODER_DROPOUT_RATES_KEY: numpy.full(7, 0.),
    chiu_net_pp_arch.NWP_FC_MODULE_NUM_CONV_LAYERS_KEY: 1,
    chiu_net_pp_arch.NWP_FC_MODULE_DROPOUT_RATES_KEY: numpy.array([0.]),
    chiu_net_pp_arch.NWP_FC_MODULE_USE_3D_CONV: True,
    chiu_net_pp_arch.LAGTGT_ENCODER_NUM_CHANNELS_KEY: numpy.array([8, 12, 16, 24, 32, 48, 64], dtype=int),
    chiu_net_pp_arch.LAGTGT_POOLING_SIZE_KEY: numpy.full(6, 2, dtype=int),
    chiu_net_pp_arch.LAGTGT_ENCODER_NUM_CONV_LAYERS_KEY: numpy.full(7, NUM_CONV_LAYERS_PER_BLOCK, dtype=int),
    chiu_net_pp_arch.LAGTGT_ENCODER_DROPOUT_RATES_KEY: numpy.full(7, 0.),
    chiu_net_pp_arch.LAGTGT_FC_MODULE_NUM_CONV_LAYERS_KEY: 1,
    chiu_net_pp_arch.LAGTGT_FC_MODULE_DROPOUT_RATES_KEY: numpy.array([0.]),
    chiu_net_pp_arch.LAGTGT_FC_MODULE_USE_3D_CONV: True,
    chiu_net_pp_arch.RCTBIAS_ENCODER_NUM_CHANNELS_KEY: numpy.array([8, 12, 16, 24, 32, 48, 64], dtype=int),
    chiu_net_pp_arch.RCTBIAS_POOLING_SIZE_KEY: numpy.full(6, 2, dtype=int),
    chiu_net_pp_arch.RCTBIAS_ENCODER_NUM_CONV_LAYERS_KEY: numpy.full(7, NUM_CONV_LAYERS_PER_BLOCK, dtype=int),
    chiu_net_pp_arch.RCTBIAS_ENCODER_DROPOUT_RATES_KEY: numpy.full(7, 0.),
    chiu_net_pp_arch.RCTBIAS_FC_MODULE_NUM_CONV_LAYERS_KEY: 1,
    chiu_net_pp_arch.RCTBIAS_FC_MODULE_DROPOUT_RATES_KEY: numpy.array([0.]),
    chiu_net_pp_arch.RCTBIAS_FC_MODULE_USE_3D_CONV: True,
    chiu_net_pp_arch.DECODER_NUM_CHANNELS_KEY: numpy.array([40, 60, 80, 120, 160, 240], dtype=int),
    chiu_net_pp_arch.DECODER_NUM_CONV_LAYERS_KEY: numpy.full(6, NUM_CONV_LAYERS_PER_BLOCK, dtype=int),
    chiu_net_pp_arch.UPSAMPLING_DROPOUT_RATES_KEY: numpy.full(6, 0.),
    chiu_net_pp_arch.SKIP_DROPOUT_RATES_KEY: numpy.full(6, 0.),
    chiu_net_pp_arch.INCLUDE_PENULTIMATE_KEY: False,
    chiu_net_pp_arch.PENULTIMATE_DROPOUT_RATE_KEY: 0.,
    chiu_net_pp_arch.INNER_ACTIV_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    chiu_net_pp_arch.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    chiu_net_pp_arch.OUTPUT_ACTIV_FUNCTION_KEY: None,
    chiu_net_pp_arch.OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.,
    chiu_net_pp_arch.L1_WEIGHT_KEY: 0.,
    chiu_net_pp_arch.L2_WEIGHT_KEY: 1e-7,
    chiu_net_pp_arch.USE_BATCH_NORM_KEY: True,
    chiu_net_pp_arch.ENSEMBLE_SIZE_KEY: 50,
    chiu_net_pp_arch.NUM_OUTPUT_CHANNELS_KEY: 1,
    chiu_net_pp_arch.PREDICT_GUST_FACTOR_KEY: False,
    chiu_net_pp_arch.PREDICT_DEWPOINT_DEPRESSION_KEY: False,
    # chiu_net_pp_arch.LOSS_FUNCTION_KEY: LOSS_FUNCTION,
    # chiu_net_pp_arch.OPTIMIZER_FUNCTION_KEY: OPTIMIZER_FUNCTION,
    # chiu_net_pp_arch.METRIC_FUNCTIONS_KEY: []
}

NWP_LEAD_TIME_COUNTS = numpy.array([1, 2, 3, 4], dtype=int)
LAG_TIME_COUNTS = numpy.array([1, 2, 3, 4], dtype=int)


def _run():
    """Creates Chiu-net++ templates for Experiment 10.

    This is effectively the main method.
    """

    for i in range(len(NWP_LEAD_TIME_COUNTS)):
        for j in range(len(LAG_TIME_COUNTS)):
            input_dims_2pt5km = numpy.array(
                [432, 432, NWP_LEAD_TIME_COUNTS[i], 22], dtype=int
            )
            input_dims_lagged_targets = numpy.array(
                [432, 432, LAG_TIME_COUNTS[j], 1], dtype=int
            )
            input_dims_2pt5km_rctbias = numpy.array(
                [432, 432, LAG_TIME_COUNTS[j], 3], dtype=int
            )

            option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)
            option_dict.update({
                chiu_net_pp_arch.LOSS_FUNCTION_KEY: LOSS_FUNCTION,
                chiu_net_pp_arch.METRIC_FUNCTIONS_KEY: METRIC_FUNCTIONS,
                chiu_net_pp_arch.OPTIMIZER_FUNCTION_KEY: OPTIMIZER_FUNCTION,
                chiu_net_pp_arch.INPUT_DIMENSIONS_2PT5KM_RES_KEY: input_dims_2pt5km,
                chiu_net_pp_arch.INPUT_DIMENSIONS_LAGGED_TARGETS_KEY: input_dims_lagged_targets,
                chiu_net_pp_arch.INPUT_DIMENSIONS_2PT5KM_RCTBIAS_KEY: input_dims_2pt5km_rctbias
            })
            model_object = chiu_net_pp_arch.create_model(option_dict)

            output_file_name = (
                '{0:s}/num-nwp-lead-times={1:d}_num-lag-times={2:d}/model.keras'
            ).format(
                OUTPUT_DIR_NAME, NWP_LEAD_TIME_COUNTS[i], LAG_TIME_COUNTS[j]
            )

            file_system_utils.mkdir_recursive_if_necessary(
                file_name=output_file_name
            )

            print('Writing model to: "{0:s}"...'.format(output_file_name))
            model_object.save(
                filepath=output_file_name, overwrite=True,
                include_optimizer=True
            )

            metafile_name = neural_net.find_metafile(
                model_file_name=output_file_name,
                raise_error_if_missing=False
            )
            option_dict[chiu_net_pp_arch.LOSS_FUNCTION_KEY] = (
                LOSS_FUNCTION_STRING
            )
            option_dict[chiu_net_pp_arch.METRIC_FUNCTIONS_KEY] = (
                METRIC_FUNCTION_STRINGS
            )
            option_dict[chiu_net_pp_arch.OPTIMIZER_FUNCTION_KEY] = (
                OPTIMIZER_FUNCTION_STRING
            )

            neural_net.write_metafile(
                pickle_file_name=metafile_name,
                num_epochs=100,
                num_training_batches_per_epoch=32,
                training_option_dict={},
                num_validation_batches_per_epoch=16,
                validation_option_dict={},
                loss_function_string=LOSS_FUNCTION_STRING,
                optimizer_function_string=OPTIMIZER_FUNCTION_STRING,
                metric_function_strings=METRIC_FUNCTION_STRINGS,
                chiu_net_architecture_dict=None,
                chiu_net_pp_architecture_dict=option_dict,
                plateau_patience_epochs=10,
                plateau_learning_rate_multiplier=0.6,
                early_stopping_patience_epochs=50,
                patch_overlap_fast_gen_2pt5km_pixels=144
            )


if __name__ == '__main__':
    _run()