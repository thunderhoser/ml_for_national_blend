"""Creates Chiu-net++ templates for Experiment 4."""

import os
import sys
import copy
import itertools
import numpy
import keras

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import nwp_model_utils
import custom_losses
import custom_metrics
import neural_net
import chiu_net_pp_architecture as chiu_net_pp_arch
import architecture_utils
import file_system_utils

numpy.random.seed(6695)

OUTPUT_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_national_blend_models/'
    'experiment04_temperature_only_more_models/templates'
)

LOSS_FUNCTION = custom_losses.dual_weighted_mse(
    channel_weights=numpy.array([1.]),
    expect_ensemble=False,
    function_name='loss_dwmse'
)

LOSS_FUNCTION_STRING = (
    'custom_losses.dual_weighted_mse('
    'channel_weights=numpy.array([1.]), '
    'expect_ensemble=False, '
    'function_name="loss_dwmse"'
    ')'
)

METRIC_FUNCTIONS = [
    custom_metrics.max_prediction(channel_index=0, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=False, function_name='temp_max_prediction_celsius'),
    custom_metrics.min_prediction(channel_index=0, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=False, function_name='temp_min_prediction_celsius'),
    custom_metrics.mean_squared_error(channel_index=0, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=False, function_name='temp_mse_celsius2'),
    custom_metrics.dual_weighted_mse(channel_index=0, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=False, function_name='temp_dwmse_celsius3')
]

METRIC_FUNCTION_STRINGS = [
    'custom_metrics.max_prediction(channel_index=0, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=False, function_name="temp_max_prediction_celsius")',
    'custom_metrics.min_prediction(channel_index=0, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=False, function_name="temp_min_prediction_celsius")',
    'custom_metrics.mean_squared_error(channel_index=0, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=False, function_name="temp_mse_celsius2")',
    'custom_metrics.dual_weighted_mse(channel_index=0, temperature_index=0, u_wind_index=1, v_wind_index=2, dewpoint_index=3, gust_index=4, expect_ensemble=False, function_name="temp_dwmse_celsius3")'
]

NUM_CONV_LAYERS_PER_BLOCK = 1
NUM_NWP_LEAD_TIMES = 4

OPTIMIZER_FUNCTION = keras.optimizers.Nadam(gradient_accumulation_steps=8)
OPTIMIZER_FUNCTION_STRING = (
    'keras.optimizers.Nadam(gradient_accumulation_steps=8)'
)

DEFAULT_OPTION_DICT = {
    chiu_net_pp_arch.INPUT_DIMENSIONS_CONST_KEY: numpy.array([449, 449, 2], dtype=int),
    chiu_net_pp_arch.INPUT_DIMENSIONS_2PT5KM_RES_KEY: None,
    chiu_net_pp_arch.INPUT_DIMENSIONS_10KM_RES_KEY: None,
    chiu_net_pp_arch.INPUT_DIMENSIONS_20KM_RES_KEY: None,
    chiu_net_pp_arch.INPUT_DIMENSIONS_40KM_RES_KEY: None,
    chiu_net_pp_arch.PREDN_BASELINE_DIMENSIONS_KEY: numpy.array([449, 449, 1], dtype=int),
    chiu_net_pp_arch.USE_RESIDUAL_BLOCKS_KEY: False,
    chiu_net_pp_arch.NUM_CHANNELS_KEY: numpy.array([32, 48, 64, 96, 128, 192, 256], dtype=int),
    chiu_net_pp_arch.POOLING_SIZE_KEY: numpy.full(6, 2, dtype=int),
    chiu_net_pp_arch.ENCODER_NUM_CONV_LAYERS_KEY: numpy.full(7, NUM_CONV_LAYERS_PER_BLOCK, dtype=int),
    chiu_net_pp_arch.ENCODER_DROPOUT_RATES_KEY: numpy.full(7, 0.),
    chiu_net_pp_arch.DECODER_NUM_CONV_LAYERS_KEY: numpy.full(6, NUM_CONV_LAYERS_PER_BLOCK, dtype=int),
    chiu_net_pp_arch.UPSAMPLING_DROPOUT_RATES_KEY: numpy.full(6, 0.),
    chiu_net_pp_arch.SKIP_DROPOUT_RATES_KEY: numpy.full(6, 0.),
    chiu_net_pp_arch.FC_MODULE_NUM_CONV_LAYERS_KEY: 1,
    chiu_net_pp_arch.FC_MODULE_DROPOUT_RATES_KEY: numpy.array([0.]),
    chiu_net_pp_arch.FC_MODULE_USE_3D_CONV: True,
    chiu_net_pp_arch.INCLUDE_PENULTIMATE_KEY: False,
    chiu_net_pp_arch.PENULTIMATE_DROPOUT_RATE_KEY: 0.,
    chiu_net_pp_arch.INNER_ACTIV_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    chiu_net_pp_arch.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    chiu_net_pp_arch.OUTPUT_ACTIV_FUNCTION_KEY: None,
    chiu_net_pp_arch.OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.,
    chiu_net_pp_arch.L1_WEIGHT_KEY: 0.,
    # chiu_net_pp_arch.L2_WEIGHT_KEY: 1e-6,
    chiu_net_pp_arch.L2_WEIGHT_KEY: 1e-7,
    chiu_net_pp_arch.USE_BATCH_NORM_KEY: True,
    chiu_net_pp_arch.ENSEMBLE_SIZE_KEY: 1,
    chiu_net_pp_arch.NUM_OUTPUT_CHANNELS_KEY: 1,
    chiu_net_pp_arch.PREDICT_GUST_FACTOR_KEY: False,
    chiu_net_pp_arch.PREDICT_DEWPOINT_DEPRESSION_KEY: False,
    # chiu_net_pp_arch.LOSS_FUNCTION_KEY: LOSS_FUNCTION,
    # chiu_net_pp_arch.OPTIMIZER_FUNCTION_KEY: OPTIMIZER_FUNCTION,
    # chiu_net_pp_arch.METRIC_FUNCTIONS_KEY: []
}

UNIQUE_PREDICTOR_SET_STRINGS = numpy.array([
    'pressure_mean_sea_level_pascals-pressure_surface_pascals-temperature_2m_agl_kelvins-dewpoint_2m_agl_kelvins-u_wind_10m_agl_m_s01-v_wind_10m_agl_m_s01-accumulated_precip_metres',
    'pressure_mean_sea_level_pascals-pressure_surface_pascals-temperature_2m_agl_kelvins-dewpoint_2m_agl_kelvins-u_wind_10m_agl_m_s01-v_wind_10m_agl_m_s01-accumulated_precip_metres-u_wind_1000mb_m_s01-v_wind_1000mb_m_s01-temperature_950mb_kelvins-relative_humidity_850mb-temperature_850mb_kelvins',
    'pressure_mean_sea_level_pascals-pressure_surface_pascals-temperature_2m_agl_kelvins-dewpoint_2m_agl_kelvins-u_wind_10m_agl_m_s01-v_wind_10m_agl_m_s01-accumulated_precip_metres-u_wind_1000mb_m_s01-v_wind_1000mb_m_s01-temperature_950mb_kelvins-relative_humidity_850mb-temperature_850mb_kelvins-geopotential_height_700mb_m_asl-relative_humidity_700mb-u_wind_700mb_m_s01-v_wind_700mb_m_s01',
    'pressure_mean_sea_level_pascals-pressure_surface_pascals-temperature_2m_agl_kelvins-dewpoint_2m_agl_kelvins-u_wind_10m_agl_m_s01-v_wind_10m_agl_m_s01-accumulated_precip_metres-u_wind_1000mb_m_s01-v_wind_1000mb_m_s01-temperature_950mb_kelvins-relative_humidity_850mb-temperature_850mb_kelvins-geopotential_height_700mb_m_asl-relative_humidity_700mb-u_wind_700mb_m_s01-v_wind_700mb_m_s01-geopotential_height_500mb_m_asl-relative_humidity_500mb-u_wind_500mb_m_s01-v_wind_500mb_m_s01',
])

UNIQUE_PREDICTOR_SET_DESCRIPTIONS = ['surface', 'surface-1000-850', 'surface-1000-850-700', 'surface-1000-850-700-500']


def _get_hyperparams():
    """Determines hyperparameters for each neural network.
    
    N = number of neural networks
    
    :return: nwp_model_set_strings: length-N list, indicating which NWP models
        are used to create predictors for each neural net.
    :return: nwp_model_to_field_dicts: length-N list, where the [i]th item is a
        dictionary specifying predictor fields for the [i]th neural net.
    :return: predictor_set_descriptions: length-N list of descriptions.
    """
    
    all_model_names = copy.deepcopy(nwp_model_utils.ALL_MODEL_NAMES)
    all_model_names.remove(nwp_model_utils.HRRR_MODEL_NAME)
    all_model_names.remove(nwp_model_utils.NAM_MODEL_NAME)
    
    unique_model_set_strings = []
    
    for subset_size in range(len(all_model_names) + 1):
        if subset_size == 0:
            continue
    
        for this_list in itertools.combinations(all_model_names, subset_size):
            this_list = list(this_list)
            this_list.append('hrrr')
            unique_model_set_strings.append(
                '-'.join([m for m in this_list])
            )
    
    num_model_sets = len(unique_model_set_strings)
    all_indices = numpy.linspace(
        0, num_model_sets - 1, num=num_model_sets, dtype=int
    )
    good_indices = numpy.random.choice(all_indices, size=25, replace=False)
    
    unique_model_set_strings = [
        unique_model_set_strings[k] for k in good_indices
    ]
    del num_model_sets
    
    nwp_model_set_strings = []
    nwp_model_to_field_dicts = []
    predictor_set_descriptions = []
    
    for this_model_set_string in unique_model_set_strings:
        for j in range(len(UNIQUE_PREDICTOR_SET_STRINGS)):
            these_model_names = this_model_set_string.split('-')
            these_predictor_names = UNIQUE_PREDICTOR_SET_STRINGS[j].split('-')
            new_dict = {}
    
            for this_model_name in these_model_names:
                new_dict[this_model_name] = []
    
                for this_predictor_name in these_predictor_names:
                    if (
                            this_predictor_name in
                            nwp_model_utils.model_to_maybe_missing_fields(this_model_name)
                    ):
                        continue
    
                    new_dict[this_model_name].append(this_predictor_name)
    
                if len(new_dict[this_model_name]) == 0:
                    raise ValueError
    
            nwp_model_set_strings.append(this_model_set_string)
            nwp_model_to_field_dicts.append(new_dict)
            predictor_set_descriptions.append(
                UNIQUE_PREDICTOR_SET_DESCRIPTIONS[j]
            )
    
    return (
        nwp_model_set_strings,
        nwp_model_to_field_dicts,
        predictor_set_descriptions
    )


def _determine_input_dims_1model(nwp_model_names, nwp_model_to_field_dict):
    """Determines input dimensions for one model.

    :param nwp_model_names: 1-D list with names of NWP models.
    :param nwp_model_to_field_dict: Dictionary, where each key is the name of an
        NWP model and the corresponding value is a list of predictor variables
        (fields).
    :return: option_dict: Dictionary with dimension options filled in.
    """

    nwp_downsampling_factors = numpy.array([
        nwp_model_utils.model_to_nbm_downsampling_factor(m)
        for m in nwp_model_names
    ], dtype=int)

    option_dict = {}

    for unique_ds_factor in numpy.unique(nwp_downsampling_factors):
        these_indices = numpy.where(
            nwp_downsampling_factors == unique_ds_factor
        )[0]

        these_lengths = numpy.array([
            len(nwp_model_to_field_dict[nwp_model_names[k]])
            for k in these_indices
        ], dtype=int)

        this_num_fields = numpy.sum(these_lengths)

        if unique_ds_factor == 1:
            these_dim = numpy.array(
                [449, 449, NUM_NWP_LEAD_TIMES, this_num_fields], dtype=int
            )
            option_dict[chiu_net_pp_arch.INPUT_DIMENSIONS_2PT5KM_RES_KEY] = (
                these_dim
            )
        elif unique_ds_factor == 4:
            these_dim = numpy.array(
                [113, 113, NUM_NWP_LEAD_TIMES, this_num_fields], dtype=int
            )
            option_dict[chiu_net_pp_arch.INPUT_DIMENSIONS_10KM_RES_KEY] = (
                these_dim
            )
        elif unique_ds_factor == 8:
            these_dim = numpy.array(
                [57, 57, NUM_NWP_LEAD_TIMES, this_num_fields], dtype=int
            )
            option_dict[chiu_net_pp_arch.INPUT_DIMENSIONS_20KM_RES_KEY] = (
                these_dim
            )
        else:
            these_dim = numpy.array(
                [29, 29, NUM_NWP_LEAD_TIMES, this_num_fields], dtype=int
            )
            option_dict[chiu_net_pp_arch.INPUT_DIMENSIONS_40KM_RES_KEY] = (
                these_dim
            )

    return option_dict


def _run():
    """Creates Chiu-net++ templates for Experiment 4.

    This is effectively the main method.
    """

    (
        nwp_model_set_strings,
        nwp_model_to_field_dicts,
        predictor_set_descriptions
    ) = _get_hyperparams()
    
    for i in range(len(nwp_model_set_strings)):
        these_nwp_model_names = nwp_model_set_strings[i].split('-')
        new_option_dict = _determine_input_dims_1model(
            nwp_model_names=these_nwp_model_names,
            nwp_model_to_field_dict=nwp_model_to_field_dicts[i]
        )

        option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)
        option_dict.update(new_option_dict)
        option_dict.update({
            chiu_net_pp_arch.LOSS_FUNCTION_KEY: LOSS_FUNCTION,
            chiu_net_pp_arch.OPTIMIZER_FUNCTION_KEY: OPTIMIZER_FUNCTION,
            chiu_net_pp_arch.METRIC_FUNCTIONS_KEY: METRIC_FUNCTIONS
        })
        model_object = chiu_net_pp_arch.create_model(option_dict)

        output_file_name = (
            '{0:s}/nwp-model-names={1:s}_nwp-fields={2:s}/model.keras'
        ).format(
            OUTPUT_DIR_NAME,
            nwp_model_set_strings[i],
            UNIQUE_PREDICTOR_SET_DESCRIPTIONS[i]
        )

        file_system_utils.mkdir_recursive_if_necessary(
            file_name=output_file_name
        )

        print('Writing model to: "{0:s}"...'.format(output_file_name))
        model_object.save(
            filepath=output_file_name, overwrite=True, include_optimizer=True
        )

        metafile_name = neural_net.find_metafile(
            model_file_name=output_file_name,
            raise_error_if_missing=False
        )
        option_dict[neural_net.LOSS_FUNCTION_KEY] = (
            LOSS_FUNCTION_STRING
        )
        option_dict[neural_net.METRIC_FUNCTIONS_KEY] = (
            METRIC_FUNCTION_STRINGS
        )
        option_dict[neural_net.OPTIMIZER_FUNCTION_KEY] = (
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
            plateau_patience_epochs=10,
            plateau_learning_rate_multiplier=0.6,
            early_stopping_patience_epochs=50
        )


if __name__ == '__main__':
    _run()
