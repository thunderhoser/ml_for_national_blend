"""Methods for creating Chiu-net architecture.

This is a U-net with TimeDistributed layers to handle NWP forecasts at different
lead times.

Based on Chiu et al. (2020): https://doi.org/10.1109/LRA.2020.2992184
"""

import numpy
import keras
from ml_for_national_blend.outside_code import error_checking
from ml_for_national_blend.outside_code import architecture_utils

INPUT_DIMENSIONS_CONST_KEY = 'input_dimensions_const'
INPUT_DIMENSIONS_2PT5KM_RES_KEY = 'input_dimensions_2pt5km_res'
INPUT_DIMENSIONS_10KM_RES_KEY = 'input_dimensions_10km_res'
INPUT_DIMENSIONS_20KM_RES_KEY = 'input_dimensions_20km_res'
INPUT_DIMENSIONS_40KM_RES_KEY = 'input_dimensions_40km_res'
PREDN_BASELINE_DIMENSIONS_KEY = 'input_dimensions_predn_baseline'
INPUT_DIMENSIONS_LAGGED_TARGETS_KEY = 'input_dimensions_lagged_targets'
USE_RESIDUAL_BLOCKS_KEY = 'use_residual_blocks'

NWP_ENCODER_NUM_CHANNELS_KEY = 'nwp_encoder_num_channels_by_level'
NWP_POOLING_SIZE_KEY = 'nwp_pooling_size_by_level_px'
NWP_ENCODER_NUM_CONV_LAYERS_KEY = 'nwp_encoder_num_conv_layers_by_level'
NWP_ENCODER_DROPOUT_RATES_KEY = 'nwp_encoder_dropout_rate_by_level'
NWP_FC_MODULE_NUM_CONV_LAYERS_KEY = 'nwp_forecast_module_num_conv_layers'
NWP_FC_MODULE_DROPOUT_RATES_KEY = 'nwp_forecast_module_dropout_rates'
NWP_FC_MODULE_USE_3D_CONV = 'nwp_forecast_module_use_3d_conv'

LAGTGT_ENCODER_NUM_CHANNELS_KEY = 'lagtgt_encoder_num_channels_by_level'
LAGTGT_POOLING_SIZE_KEY = 'lagtgt_pooling_size_by_level_px'
LAGTGT_ENCODER_NUM_CONV_LAYERS_KEY = 'lagtgt_encoder_num_conv_layers_by_level'
LAGTGT_ENCODER_DROPOUT_RATES_KEY = 'lagtgt_encoder_dropout_rate_by_level'
LAGTGT_FC_MODULE_NUM_CONV_LAYERS_KEY = 'lagtgt_forecast_module_num_conv_layers'
LAGTGT_FC_MODULE_DROPOUT_RATES_KEY = 'lagtgt_forecast_module_dropout_rates'
LAGTGT_FC_MODULE_USE_3D_CONV = 'lagtgt_forecast_module_use_3d_conv'

DECODER_NUM_CHANNELS_KEY = 'decoder_num_channels_by_level'
DECODER_NUM_CONV_LAYERS_KEY = 'decoder_num_conv_layers_by_level'
UPSAMPLING_DROPOUT_RATES_KEY = 'upsampling_dropout_rate_by_level'
SKIP_DROPOUT_RATES_KEY = 'skip_dropout_rate_by_level'

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
PREDICT_GUST_FACTOR_KEY = 'predict_gust_factor'
PREDICT_DEWPOINT_DEPRESSION_KEY = 'predict_dewpoint_depression'
LOSS_FUNCTION_KEY = 'loss_function'
OPTIMIZER_FUNCTION_KEY = 'optimizer_function'
METRIC_FUNCTIONS_KEY = 'metric_function_list'

DEFAULT_OPTION_DICT = {
    # ENCODER_NUM_CONV_LAYERS_KEY: numpy.full(9, 2, dtype=int),
    DECODER_NUM_CONV_LAYERS_KEY: numpy.full(8, 2, dtype=int),
    # POOLING_SIZE_KEY: numpy.full(8, 2, dtype=int),
    # NUM_CHANNELS_KEY: numpy.array([8, 12, 16, 24, 32, 48, 64, 96, 96], dtype=int),
    # ENCODER_DROPOUT_RATES_KEY: numpy.full(9, 0.),
    UPSAMPLING_DROPOUT_RATES_KEY: numpy.full(8, 0.),
    SKIP_DROPOUT_RATES_KEY: numpy.full(8, 0.),
    # FC_MODULE_NUM_CONV_LAYERS_KEY: 1,
    # FC_MODULE_DROPOUT_RATES_KEY: numpy.array([0.]),
    # FC_MODULE_USE_3D_CONV: True,
    INCLUDE_PENULTIMATE_KEY: True,
    PENULTIMATE_DROPOUT_RATE_KEY: 0.,
    INNER_ACTIV_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    OUTPUT_ACTIV_FUNCTION_KEY: None,
    OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.,
    L1_WEIGHT_KEY: 0.,
    USE_BATCH_NORM_KEY: True
}


def check_input_args(option_dict):
    """Error-checks input arguments.

    L = number of levels

    :param option_dict: Dictionary with the following keys.
    option_dict["input_dimensions_2pt5km_res"]:
        numpy array with input dimensions for 2.5-km NWP forecasts:
        [num_grid_rows, num_grid_columns, num_lead_times, num_channels].
        If you are not including 2.5-km data, make this None.
    option_dict["input_dimensions_const"]: Same but for 2.5-km constant fields.
    option_dict["input_dimensions_10km_res"]: Same but for 10-km NWP forecasts.
    option_dict["input_dimensions_20km_res"]: Same but for 20-km NWP forecasts.
    option_dict["input_dimensions_40km_res"]: Same but for 40-km NWP forecasts.
    option_dict["input_dimensions_predn_baseline"]: Same but for prediction
        baseline.
    option_dict["input_dimensions_lagged_targets"]: Same but for lagged targets.
    option_dict["use_residual_blocks"]: Boolean flag.  If True (False), the NN
        will use residual (simple conv) blocks.
    option_dict["nwp_encoder_num_channels_by_level"]: length-(L + 1) numpy array
        with number of channels (feature maps) at each level of NWP-encoder.
    option_dict["lagtgt_encoder_num_channels_by_level"]: Same but for lagged
        targets.  If you do not want to use lagged targets, make this None.
    option_dict["nwp_pooling_size_by_level_px"]: length-L numpy array with size
        of max-pooling window at each level of NWP-encoder.  For example, if you
        want 2-by-2 pooling at the [j]th level,
        make pooling_size_by_level_px[j] = 2.
    option_dict["lagtgt_pooling_size_by_level_px"]: Same but for lagged
        targets.  If you do not want to use lagged targets, make this None.
    option_dict["nwp_encoder_num_conv_layers_by_level"]: length-(L + 1) numpy
        array with number of conv layers at each level of NWP-encoder.
    option_dict["lagtgt_encoder_num_conv_layers_by_level"]: Same but for lagged
        targets.  If you do not want to use lagged targets, make this None.
    option_dict["nwp_encoder_dropout_rate_by_level"]: length-(L + 1) numpy array
        with dropout rate at each level of NWP-encoder.  Use numbers <= 0 to
        indicate no-dropout.
    option_dict["lagtgt_encoder_dropout_rate_by_level"]: Same but for lagged
        targets.  If you do not want to use lagged targets, make this None.
    option_dict["nwp_forecast_module_num_conv_layers"]: Number of conv layers in
        forecasting module at end of NWP-encoder.
    option_dict["lagtgt_forecast_module_num_conv_layers"]: Same but for lagged
        targets.  If you do not want to use lagged targets, make this None.
    option_dict["nwp_forecast_module_dropout_rates"]: length-F numpy array
        (where F = nwp_forecast_module_num_conv_layers) with dropout rate for
        each conv layer in NWP-forecasting module.  Use numbers <= 0 to indicate
        no-dropout.
    option_dict["lagtgt_forecast_module_dropout_rates"]: Same but for lagged
        targets.  If you do not want to use lagged targets, make this None.
    option_dict["nwp_forecast_module_use_3d_conv"]: Boolean flag.  Determines
        whether NWP-forecasting module will use 2-D or 3-D convolution.
    option_dict["lagtgt_forecast_module_use_3d_conv"]: Same but for lagged
        targets.  If you do not want to use lagged targets, make this None.
    option_dict["decoder_num_channels_by_level"]: length-L numpy array with
        number of channels (feature maps) at each level of decoder block.
    option_dict["decoder_num_conv_layers_by_level"]: length-L numpy array
        with number of conv layers at each level of decoder block.
    option_dict["upsampling_dropout_rate_by_level"]: length-L numpy array with
        dropout rate in upconv layer at each level of decoder block.  Use
        numbers <= 0 to indicate no-dropout.
    option_dict["skip_dropout_rate_by_level"]: length-L numpy array with dropout
        rate in skip connection at each level of decoder block.  Use
        numbers <= 0 to indicate no-dropout.
    option_dict["include_penultimate_conv"]: Boolean flag.  If True, will put in
        extra conv layer (with 3 x 3 filter) before final pixelwise conv.
    option_dict["penultimate_conv_dropout_rate"]: Dropout rate for penultimate
        conv layer.
    option_dict["inner_activ_function_name"]: Name of activation function for
        all non-output layers.  Must be accepted by
        `architecture_utils.check_activation_function`.
    option_dict["inner_activ_function_alpha"]: Alpha (slope parameter) for
        activation function for all non-output layers.  Applies only to ReLU and
        eLU.
    option_dict["output_activ_function_name"]: Name of activation function for
        output layer.  This can be None; otherwise, must be a string accepted by
        `architecture_utils.check_activation_function`.
    option_dict["output_activ_function_alpha"]: Alpha (slope parameter) for
        activation function for all output layer.  Applies only to ReLU and eLU.
    option_dict["l1_weight"]: Strength of L1 regularization (for conv layers
        only).
    option_dict["l2_weight"]: Strength of L2 regularization (for conv layers
        only).
    option_dict["use_batch_normalization"]: Boolean flag.  If True, will use
        batch normalization after each non-output layer.
    option_dict["ensemble_size"]: Number of ensemble members.
    option_dict["num_output_channels"]: Number of output channels.
    option_dict["predict_gust_factor"]: Boolean flag.  If True, the model needs
        to predict gust factor.
    option_dict["predict_dewpoint_depression"]: Boolean flag.  If True, the
        model needs to predict dewpoint depression.
    option_dict["loss_function"]: Loss function.
    option_dict["optimizer_function"]: Optimizer function.
    option_dict["metric_function_list"]: 1-D list of metric functions.

    :return: option_dict: Same as input but maybe with default values added.
    """

    orig_option_dict = option_dict.copy()
    option_dict = DEFAULT_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    error_checking.assert_is_numpy_array(
        option_dict[INPUT_DIMENSIONS_2PT5KM_RES_KEY],
        exact_dimensions=numpy.array([4], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(
        option_dict[INPUT_DIMENSIONS_2PT5KM_RES_KEY]
    )
    error_checking.assert_is_greater_numpy_array(
        option_dict[INPUT_DIMENSIONS_2PT5KM_RES_KEY], 0
    )

    if option_dict[PREDN_BASELINE_DIMENSIONS_KEY] is not None:
        num_rows = option_dict[INPUT_DIMENSIONS_2PT5KM_RES_KEY][0]
        num_columns = option_dict[INPUT_DIMENSIONS_2PT5KM_RES_KEY][1]
        num_channels = option_dict[NUM_OUTPUT_CHANNELS_KEY]
        expected_array = numpy.array(
            [num_rows, num_columns, num_channels], dtype=int
        )

        assert numpy.array_equal(
            option_dict[PREDN_BASELINE_DIMENSIONS_KEY],
            expected_array
        )

    error_checking.assert_is_boolean(option_dict[USE_RESIDUAL_BLOCKS_KEY])

    if option_dict[INPUT_DIMENSIONS_CONST_KEY] is not None:
        error_checking.assert_is_numpy_array(
            option_dict[INPUT_DIMENSIONS_CONST_KEY],
            exact_dimensions=numpy.array([3], dtype=int)
        )
        error_checking.assert_is_integer_numpy_array(
            option_dict[INPUT_DIMENSIONS_CONST_KEY]
        )
        error_checking.assert_is_greater_numpy_array(
            option_dict[INPUT_DIMENSIONS_CONST_KEY], 0
        )

    if option_dict[INPUT_DIMENSIONS_10KM_RES_KEY] is not None:
        error_checking.assert_is_numpy_array(
            option_dict[INPUT_DIMENSIONS_10KM_RES_KEY],
            exact_dimensions=numpy.array([4], dtype=int)
        )
        error_checking.assert_is_integer_numpy_array(
            option_dict[INPUT_DIMENSIONS_10KM_RES_KEY]
        )
        error_checking.assert_is_greater_numpy_array(
            option_dict[INPUT_DIMENSIONS_10KM_RES_KEY], 0
        )

    if option_dict[INPUT_DIMENSIONS_20KM_RES_KEY] is not None:
        error_checking.assert_is_numpy_array(
            option_dict[INPUT_DIMENSIONS_20KM_RES_KEY],
            exact_dimensions=numpy.array([4], dtype=int)
        )
        error_checking.assert_is_integer_numpy_array(
            option_dict[INPUT_DIMENSIONS_20KM_RES_KEY]
        )
        error_checking.assert_is_greater_numpy_array(
            option_dict[INPUT_DIMENSIONS_20KM_RES_KEY], 0
        )

    if option_dict[INPUT_DIMENSIONS_40KM_RES_KEY] is not None:
        error_checking.assert_is_numpy_array(
            option_dict[INPUT_DIMENSIONS_40KM_RES_KEY],
            exact_dimensions=numpy.array([4], dtype=int)
        )
        error_checking.assert_is_integer_numpy_array(
            option_dict[INPUT_DIMENSIONS_40KM_RES_KEY]
        )
        error_checking.assert_is_greater_numpy_array(
            option_dict[INPUT_DIMENSIONS_40KM_RES_KEY], 0
        )

    if option_dict[INPUT_DIMENSIONS_LAGGED_TARGETS_KEY] is not None:
        error_checking.assert_is_numpy_array(
            option_dict[INPUT_DIMENSIONS_LAGGED_TARGETS_KEY],
            exact_dimensions=numpy.array([4], dtype=int)
        )
        error_checking.assert_is_integer_numpy_array(
            option_dict[INPUT_DIMENSIONS_LAGGED_TARGETS_KEY]
        )
        error_checking.assert_is_greater_numpy_array(
            option_dict[INPUT_DIMENSIONS_LAGGED_TARGETS_KEY], 0
        )

    error_checking.assert_is_numpy_array(
        option_dict[NWP_ENCODER_NUM_CHANNELS_KEY], num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(
        option_dict[NWP_ENCODER_NUM_CHANNELS_KEY]
    )
    error_checking.assert_is_geq_numpy_array(
        option_dict[NWP_ENCODER_NUM_CHANNELS_KEY], 1
    )

    num_levels = len(option_dict[NWP_ENCODER_NUM_CHANNELS_KEY]) - 1

    error_checking.assert_is_numpy_array(
        option_dict[NWP_POOLING_SIZE_KEY],
        exact_dimensions=numpy.array([num_levels], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(
        option_dict[NWP_POOLING_SIZE_KEY]
    )
    error_checking.assert_is_geq_numpy_array(
        option_dict[NWP_POOLING_SIZE_KEY], 2
    )

    error_checking.assert_is_numpy_array(
        option_dict[NWP_ENCODER_NUM_CONV_LAYERS_KEY],
        exact_dimensions=numpy.array([num_levels + 1], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(
        option_dict[NWP_ENCODER_NUM_CONV_LAYERS_KEY]
    )
    error_checking.assert_is_geq_numpy_array(
        option_dict[NWP_ENCODER_NUM_CONV_LAYERS_KEY], 1
    )

    error_checking.assert_is_numpy_array(
        option_dict[NWP_ENCODER_DROPOUT_RATES_KEY],
        exact_dimensions=numpy.array([num_levels + 1], dtype=int)
    )
    error_checking.assert_is_leq_numpy_array(
        option_dict[NWP_ENCODER_DROPOUT_RATES_KEY], 1., allow_nan=True
    )

    nwp_fc_module_num_conv_layers = option_dict[
        NWP_FC_MODULE_NUM_CONV_LAYERS_KEY
    ]
    error_checking.assert_is_integer(nwp_fc_module_num_conv_layers)
    error_checking.assert_is_greater(nwp_fc_module_num_conv_layers, 0)

    expected_dim = numpy.array([nwp_fc_module_num_conv_layers], dtype=int)

    nwp_fc_module_dropout_rates = option_dict[NWP_FC_MODULE_DROPOUT_RATES_KEY]
    error_checking.assert_is_numpy_array(
        nwp_fc_module_dropout_rates, exact_dimensions=expected_dim
    )
    error_checking.assert_is_leq_numpy_array(
        nwp_fc_module_dropout_rates, 1., allow_nan=True
    )

    error_checking.assert_is_boolean(option_dict[NWP_FC_MODULE_USE_3D_CONV])

    if option_dict[INPUT_DIMENSIONS_LAGGED_TARGETS_KEY] is not None:
        error_checking.assert_is_numpy_array(
            option_dict[LAGTGT_ENCODER_NUM_CHANNELS_KEY],
            exact_dimensions=numpy.array([num_levels + 1], dtype=int)
        )
        error_checking.assert_is_integer_numpy_array(
            option_dict[LAGTGT_ENCODER_NUM_CHANNELS_KEY]
        )
        error_checking.assert_is_geq_numpy_array(
            option_dict[LAGTGT_ENCODER_NUM_CHANNELS_KEY], 1
        )

        error_checking.assert_is_numpy_array(
            option_dict[LAGTGT_POOLING_SIZE_KEY],
            exact_dimensions=numpy.array([num_levels], dtype=int)
        )
        error_checking.assert_is_integer_numpy_array(
            option_dict[LAGTGT_POOLING_SIZE_KEY]
        )
        error_checking.assert_is_geq_numpy_array(
            option_dict[LAGTGT_POOLING_SIZE_KEY], 2
        )

        error_checking.assert_is_numpy_array(
            option_dict[LAGTGT_ENCODER_NUM_CONV_LAYERS_KEY],
            exact_dimensions=numpy.array([num_levels + 1], dtype=int)
        )
        error_checking.assert_is_integer_numpy_array(
            option_dict[LAGTGT_ENCODER_NUM_CONV_LAYERS_KEY]
        )
        error_checking.assert_is_geq_numpy_array(
            option_dict[LAGTGT_ENCODER_NUM_CONV_LAYERS_KEY], 1
        )

        error_checking.assert_is_numpy_array(
            option_dict[LAGTGT_ENCODER_DROPOUT_RATES_KEY],
            exact_dimensions=numpy.array([num_levels + 1], dtype=int)
        )
        error_checking.assert_is_leq_numpy_array(
            option_dict[LAGTGT_ENCODER_DROPOUT_RATES_KEY], 1., allow_nan=True
        )

        lagtgt_fc_module_num_conv_layers = option_dict[
            LAGTGT_FC_MODULE_NUM_CONV_LAYERS_KEY
        ]
        error_checking.assert_is_integer(lagtgt_fc_module_num_conv_layers)
        error_checking.assert_is_greater(lagtgt_fc_module_num_conv_layers, 0)

        expected_dim = numpy.array([lagtgt_fc_module_num_conv_layers], dtype=int)

        lagtgt_fc_module_dropout_rates = option_dict[
            LAGTGT_FC_MODULE_DROPOUT_RATES_KEY
        ]
        error_checking.assert_is_numpy_array(
            lagtgt_fc_module_dropout_rates, exact_dimensions=expected_dim
        )
        error_checking.assert_is_leq_numpy_array(
            lagtgt_fc_module_dropout_rates, 1., allow_nan=True
        )

        error_checking.assert_is_boolean(
            option_dict[LAGTGT_FC_MODULE_USE_3D_CONV]
        )

    error_checking.assert_is_numpy_array(
        option_dict[DECODER_NUM_CHANNELS_KEY],
        exact_dimensions=numpy.array([num_levels], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(
        option_dict[DECODER_NUM_CHANNELS_KEY]
    )
    error_checking.assert_is_geq_numpy_array(
        option_dict[DECODER_NUM_CHANNELS_KEY], 1
    )

    error_checking.assert_is_numpy_array(
        option_dict[DECODER_NUM_CONV_LAYERS_KEY],
        exact_dimensions=numpy.array([num_levels], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(
        option_dict[DECODER_NUM_CONV_LAYERS_KEY]
    )
    error_checking.assert_is_geq_numpy_array(
        option_dict[DECODER_NUM_CONV_LAYERS_KEY], 1
    )

    error_checking.assert_is_numpy_array(
        option_dict[UPSAMPLING_DROPOUT_RATES_KEY],
        exact_dimensions=numpy.array([num_levels], dtype=int)
    )
    error_checking.assert_is_leq_numpy_array(
        option_dict[UPSAMPLING_DROPOUT_RATES_KEY], 1., allow_nan=True
    )

    error_checking.assert_is_numpy_array(
        option_dict[SKIP_DROPOUT_RATES_KEY],
        exact_dimensions=numpy.array([num_levels], dtype=int)
    )
    error_checking.assert_is_leq_numpy_array(
        option_dict[SKIP_DROPOUT_RATES_KEY], 1., allow_nan=True
    )

    error_checking.assert_is_boolean(option_dict[INCLUDE_PENULTIMATE_KEY])
    error_checking.assert_is_leq(
        option_dict[PENULTIMATE_DROPOUT_RATE_KEY], 1., allow_nan=True
    )

    error_checking.assert_is_geq(option_dict[L1_WEIGHT_KEY], 0.)
    error_checking.assert_is_geq(option_dict[L2_WEIGHT_KEY], 0.)
    error_checking.assert_is_boolean(option_dict[USE_BATCH_NORM_KEY])
    error_checking.assert_is_integer(option_dict[ENSEMBLE_SIZE_KEY])
    error_checking.assert_is_geq(option_dict[ENSEMBLE_SIZE_KEY], 1)
    error_checking.assert_is_integer(option_dict[NUM_OUTPUT_CHANNELS_KEY])
    error_checking.assert_is_geq(option_dict[NUM_OUTPUT_CHANNELS_KEY], 1)
    error_checking.assert_is_boolean(option_dict[PREDICT_GUST_FACTOR_KEY])
    error_checking.assert_is_boolean(
        option_dict[PREDICT_DEWPOINT_DEPRESSION_KEY]
    )

    error_checking.assert_is_list(option_dict[METRIC_FUNCTIONS_KEY])

    return option_dict


def _get_time_slicing_function(time_index):
    """Returns function that takes one time step from input tensor.

    :param time_index: Will take the [k]th time step, where k = `time_index`.
    :return: time_slicing_function: Function handle (see below).
    """

    def time_slicing_function(input_tensor_3d):
        """Takes one time step from the input tensor.

        :param input_tensor_3d: Input tensor with 3 spatiotemporal dimensions.
        :return: input_tensor_2d: Input tensor with 2 spatial dimensions.
        """

        return input_tensor_3d[:, time_index, ...]

    return time_slicing_function


def create_model(option_dict):
    """Creates CNN.

    :param option_dict: See documentation for `check_input_args`.
    :return: model_object: Untrained (but compiled) instance of
        `keras.models.Model`.
    """

    option_dict = check_input_args(option_dict)

    input_dimensions_2pt5km_res = option_dict[INPUT_DIMENSIONS_2PT5KM_RES_KEY]
    input_dimensions_const = option_dict[INPUT_DIMENSIONS_CONST_KEY]
    input_dimensions_10km_res = option_dict[INPUT_DIMENSIONS_10KM_RES_KEY]
    input_dimensions_20km_res = option_dict[INPUT_DIMENSIONS_20KM_RES_KEY]
    input_dimensions_40km_res = option_dict[INPUT_DIMENSIONS_40KM_RES_KEY]
    input_dimensions_predn_baseline = option_dict[PREDN_BASELINE_DIMENSIONS_KEY]
    use_residual_blocks = option_dict[USE_RESIDUAL_BLOCKS_KEY]

    assert input_dimensions_predn_baseline is None
    assert not use_residual_blocks

    num_channels_by_level = option_dict[NWP_ENCODER_NUM_CHANNELS_KEY]
    pooling_size_by_level_px = option_dict[NWP_POOLING_SIZE_KEY]
    num_encoder_conv_layers_by_level = option_dict[
        NWP_ENCODER_NUM_CONV_LAYERS_KEY
    ]
    encoder_dropout_rate_by_level = option_dict[NWP_ENCODER_DROPOUT_RATES_KEY]
    num_decoder_conv_layers_by_level = option_dict[DECODER_NUM_CONV_LAYERS_KEY]
    upsampling_dropout_rate_by_level = option_dict[UPSAMPLING_DROPOUT_RATES_KEY]
    skip_dropout_rate_by_level = option_dict[SKIP_DROPOUT_RATES_KEY]
    forecast_module_num_conv_layers = option_dict[
        NWP_FC_MODULE_NUM_CONV_LAYERS_KEY
    ]
    forecast_module_dropout_rates = option_dict[NWP_FC_MODULE_DROPOUT_RATES_KEY]
    forecast_module_use_3d_conv = option_dict[NWP_FC_MODULE_USE_3D_CONV]
    include_penultimate_conv = option_dict[INCLUDE_PENULTIMATE_KEY]
    penultimate_conv_dropout_rate = option_dict[PENULTIMATE_DROPOUT_RATE_KEY]
    inner_activ_function_name = option_dict[INNER_ACTIV_FUNCTION_KEY]
    inner_activ_function_alpha = option_dict[INNER_ACTIV_FUNCTION_ALPHA_KEY]
    output_activ_function_name = option_dict[OUTPUT_ACTIV_FUNCTION_KEY]
    output_activ_function_alpha = option_dict[OUTPUT_ACTIV_FUNCTION_ALPHA_KEY]
    l1_weight = option_dict[L1_WEIGHT_KEY]
    l2_weight = option_dict[L2_WEIGHT_KEY]
    use_batch_normalization = option_dict[USE_BATCH_NORM_KEY]
    loss_function = option_dict[LOSS_FUNCTION_KEY]
    optimizer_function = option_dict[OPTIMIZER_FUNCTION_KEY]
    metric_function_list = option_dict[METRIC_FUNCTIONS_KEY]
    ensemble_size = option_dict[ENSEMBLE_SIZE_KEY]
    num_output_channels = option_dict[NUM_OUTPUT_CHANNELS_KEY]
    predict_gust_factor = option_dict[PREDICT_GUST_FACTOR_KEY]
    predict_dewpoint_depression = option_dict[PREDICT_DEWPOINT_DEPRESSION_KEY]

    num_lead_times = input_dimensions_2pt5km_res[2]

    input_layer_object_2pt5km_res = keras.layers.Input(
        shape=tuple(input_dimensions_2pt5km_res.tolist()),
        name='2pt5km_inputs'
    )
    layer_object_2pt5km_res = keras.layers.Permute(
        dims=(3, 1, 2, 4), name='2pt5km_put_time_first'
    )(input_layer_object_2pt5km_res)

    if input_dimensions_const is None:
        input_layer_object_const = None
    else:
        input_layer_object_const = keras.layers.Input(
            shape=tuple(input_dimensions_const.tolist()),
            name='const_inputs'
        )

        new_dims = (1,) + tuple(input_dimensions_const.tolist())
        layer_object_const = keras.layers.Reshape(
            target_shape=new_dims, name='const_add-time-dim'
        )(input_layer_object_const)

        this_layer_object = keras.layers.Concatenate(
            axis=-4, name='const_add-2pt5km-times'
        )(
            num_lead_times * [layer_object_const]
        )

        layer_object_2pt5km_res = keras.layers.Concatenate(
            axis=-1, name='concat_with_constants'
        )(
            [layer_object_2pt5km_res, this_layer_object]
        )

    if input_dimensions_10km_res is None:
        input_layer_object_10km_res = None
        layer_object_10km_res = None
    else:
        input_layer_object_10km_res = keras.layers.Input(
            shape=tuple(input_dimensions_10km_res.tolist()),
            name='10km_inputs'
        )
        layer_object_10km_res = keras.layers.Permute(
            dims=(3, 1, 2, 4), name='10km_put_time_first'
        )(input_layer_object_10km_res)

    if input_dimensions_20km_res is None:
        input_layer_object_20km_res = None
        layer_object_20km_res = None
    else:
        input_layer_object_20km_res = keras.layers.Input(
            shape=tuple(input_dimensions_20km_res.tolist()),
            name='20km_inputs'
        )
        layer_object_20km_res = keras.layers.Permute(
            dims=(3, 1, 2, 4), name='20km_put_time_first'
        )(input_layer_object_20km_res)

    if input_dimensions_40km_res is None:
        input_layer_object_40km_res = None
        layer_object_40km_res = None
    else:
        input_layer_object_40km_res = keras.layers.Input(
            shape=tuple(input_dimensions_40km_res.tolist()),
            name='40km_inputs'
        )
        layer_object_40km_res = keras.layers.Permute(
            dims=(3, 1, 2, 4), name='40km_put_time_first'
        )(input_layer_object_40km_res)

    l2_function = architecture_utils.get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight
    )

    num_lead_times = input_dimensions_2pt5km_res[2]

    num_levels = len(pooling_size_by_level_px)
    conv_layer_by_level = [None] * (num_levels + 1)
    pooling_layer_by_level = [None] * num_levels

    if input_dimensions_10km_res is not None:
        num_levels_to_fill = 2
    elif input_dimensions_20km_res is not None:
        num_levels_to_fill = 3
    elif input_dimensions_40km_res is not None:
        num_levels_to_fill = 4
    else:
        num_levels_to_fill = 2

    for level_index in range(num_levels_to_fill):
        i = level_index

        for j in range(num_encoder_conv_layers_by_level[i]):
            if j == 0:
                if i == 0:
                    previous_layer_object = layer_object_2pt5km_res
                else:
                    previous_layer_object = pooling_layer_by_level[i - 1]
            else:
                previous_layer_object = conv_layer_by_level[i]

            this_name = 'encoder_level{0:d}_conv{1:d}'.format(i, j)
            this_conv_layer_object = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=num_channels_by_level[i],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=l2_function,
                layer_name=this_name
            )

            conv_layer_by_level[i] = keras.layers.TimeDistributed(
                this_conv_layer_object, name=this_name
            )(previous_layer_object)

            this_name = 'encoder_level{0:d}_activation{1:d}'.format(i, j)
            conv_layer_by_level[i] = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha,
                layer_name=this_name
            )(conv_layer_by_level[i])

            if encoder_dropout_rate_by_level[i] > 0:
                this_name = 'encoder_level{0:d}_dropout{1:d}'.format(i, j)
                conv_layer_by_level[i] = architecture_utils.get_dropout_layer(
                    dropout_fraction=encoder_dropout_rate_by_level[i],
                    layer_name=this_name
                )(conv_layer_by_level[i])

            if use_batch_normalization:
                this_name = 'encoder_level{0:d}_bn{1:d}'.format(i, j)
                conv_layer_by_level[i] = architecture_utils.get_batch_norm_layer(
                    layer_name=this_name
                )(conv_layer_by_level[i])

        this_name = 'encoder_level{0:d}_pooling'.format(i)
        this_pooling_layer_object = architecture_utils.get_2d_pooling_layer(
            num_rows_in_window=pooling_size_by_level_px[i],
            num_columns_in_window=pooling_size_by_level_px[i],
            num_rows_per_stride=pooling_size_by_level_px[i],
            num_columns_per_stride=pooling_size_by_level_px[i],
            pooling_type_string=architecture_utils.MAX_POOLING_STRING,
            layer_name=this_name
        )
        pooling_layer_by_level[i] = keras.layers.TimeDistributed(
            this_pooling_layer_object, name=this_name
        )(conv_layer_by_level[i])

    num_levels_filled = num_levels_to_fill + 0

    if input_dimensions_10km_res is not None:
        i = num_levels_filled - 1
        this_name = 'concat_with_10km'
        pooling_layer_by_level[i] = keras.layers.Concatenate(
            axis=-1, name=this_name
        )(
            [pooling_layer_by_level[i], layer_object_10km_res]
        )

        if input_dimensions_20km_res is not None:
            num_levels_to_fill = 1
        elif input_dimensions_40km_res is not None:
            num_levels_to_fill = 2
        else:
            num_levels_to_fill = 1

        for level_index in range(
                num_levels_filled, num_levels_filled + num_levels_to_fill
        ):
            i = level_index

            for j in range(num_encoder_conv_layers_by_level[i]):
                if j == 0:
                    if i == 0:
                        previous_layer_object = layer_object_10km_res
                    else:
                        previous_layer_object = pooling_layer_by_level[i - 1]
                else:
                    previous_layer_object = conv_layer_by_level[i]

                this_name = 'encoder_level{0:d}_conv{1:d}'.format(i, j)
                this_conv_layer_object = architecture_utils.get_2d_conv_layer(
                    num_kernel_rows=3, num_kernel_columns=3,
                    num_rows_per_stride=1, num_columns_per_stride=1,
                    num_filters=num_channels_by_level[i],
                    padding_type_string=architecture_utils.YES_PADDING_STRING,
                    weight_regularizer=l2_function,
                    layer_name=this_name
                )

                conv_layer_by_level[i] = keras.layers.TimeDistributed(
                    this_conv_layer_object, name=this_name
                )(previous_layer_object)

                this_name = 'encoder_level{0:d}_activation{1:d}'.format(i, j)
                conv_layer_by_level[i] = architecture_utils.get_activation_layer(
                    activation_function_string=inner_activ_function_name,
                    alpha_for_relu=inner_activ_function_alpha,
                    alpha_for_elu=inner_activ_function_alpha,
                    layer_name=this_name
                )(conv_layer_by_level[i])

                if encoder_dropout_rate_by_level[i] > 0:
                    this_name = 'encoder_level{0:d}_dropout{1:d}'.format(i, j)
                    conv_layer_by_level[i] = architecture_utils.get_dropout_layer(
                        dropout_fraction=encoder_dropout_rate_by_level[i],
                        layer_name=this_name
                    )(conv_layer_by_level[i])

                if use_batch_normalization:
                    this_name = 'encoder_level{0:d}_bn{1:d}'.format(i, j)
                    conv_layer_by_level[i] = architecture_utils.get_batch_norm_layer(
                        layer_name=this_name
                    )(conv_layer_by_level[i])

            this_name = 'encoder_level{0:d}_pooling'.format(i)
            this_pooling_layer_object = architecture_utils.get_2d_pooling_layer(
                num_rows_in_window=pooling_size_by_level_px[i],
                num_columns_in_window=pooling_size_by_level_px[i],
                num_rows_per_stride=pooling_size_by_level_px[i],
                num_columns_per_stride=pooling_size_by_level_px[i],
                pooling_type_string=architecture_utils.MAX_POOLING_STRING,
                layer_name=this_name
            )
            pooling_layer_by_level[i] = keras.layers.TimeDistributed(
                this_pooling_layer_object, name=this_name
            )(conv_layer_by_level[i])

        num_levels_filled += num_levels_to_fill

    if input_dimensions_20km_res is not None:
        i = num_levels_filled - 1

        this_name = 'concat_with_20km'
        pooling_layer_by_level[i] = keras.layers.Concatenate(
            axis=-1, name=this_name
        )(
            [pooling_layer_by_level[i], layer_object_20km_res]
        )

        i = num_levels_filled

        for j in range(num_encoder_conv_layers_by_level[i]):
            if j == 0:
                if i == 0:
                    previous_layer_object = layer_object_20km_res
                else:
                    previous_layer_object = pooling_layer_by_level[i - 1]
            else:
                previous_layer_object = conv_layer_by_level[i]

            this_name = 'encoder_level{0:d}_conv{1:d}'.format(i, j)
            this_conv_layer_object = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=num_channels_by_level[i],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=l2_function,
                layer_name=this_name
            )

            conv_layer_by_level[i] = keras.layers.TimeDistributed(
                this_conv_layer_object, name=this_name
            )(previous_layer_object)

            this_name = 'encoder_level{0:d}_activation{1:d}'.format(i, j)
            conv_layer_by_level[i] = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha,
                layer_name=this_name
            )(conv_layer_by_level[i])

            if encoder_dropout_rate_by_level[i] > 0:
                this_name = 'encoder_level{0:d}_dropout{1:d}'.format(i, j)
                conv_layer_by_level[i] = architecture_utils.get_dropout_layer(
                    dropout_fraction=encoder_dropout_rate_by_level[i],
                    layer_name=this_name
                )(conv_layer_by_level[i])

            if use_batch_normalization:
                this_name = 'encoder_level{0:d}_bn{1:d}'.format(i, j)
                conv_layer_by_level[i] = architecture_utils.get_batch_norm_layer(
                    layer_name=this_name
                )(conv_layer_by_level[i])

        this_name = 'encoder_level{0:d}_pooling'.format(i)
        this_pooling_layer_object = architecture_utils.get_2d_pooling_layer(
            num_rows_in_window=pooling_size_by_level_px[i],
            num_columns_in_window=pooling_size_by_level_px[i],
            num_rows_per_stride=pooling_size_by_level_px[i],
            num_columns_per_stride=pooling_size_by_level_px[i],
            pooling_type_string=architecture_utils.MAX_POOLING_STRING,
            layer_name=this_name
        )
        pooling_layer_by_level[i] = keras.layers.TimeDistributed(
            this_pooling_layer_object, name=this_name
        )(conv_layer_by_level[i])

        num_levels_filled += 1

    if input_dimensions_40km_res is not None:
        i = num_levels_filled - 1

        this_name = 'concat_with_40km'
        pooling_layer_by_level[i] = keras.layers.Concatenate(
            axis=-1, name=this_name
        )(
            [pooling_layer_by_level[i], layer_object_40km_res]
        )

        i = num_levels_filled

        for j in range(num_encoder_conv_layers_by_level[i]):
            if j == 0:
                if i == 0:
                    previous_layer_object = layer_object_40km_res
                else:
                    previous_layer_object = pooling_layer_by_level[i - 1]
            else:
                previous_layer_object = conv_layer_by_level[i]

            this_name = 'encoder_level{0:d}_conv{1:d}'.format(i, j)
            this_conv_layer_object = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=num_channels_by_level[i],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=l2_function,
                layer_name=this_name
            )

            conv_layer_by_level[i] = keras.layers.TimeDistributed(
                this_conv_layer_object, name=this_name
            )(previous_layer_object)

            this_name = 'encoder_level{0:d}_activation{1:d}'.format(i, j)
            conv_layer_by_level[i] = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha,
                layer_name=this_name
            )(conv_layer_by_level[i])

            if encoder_dropout_rate_by_level[i] > 0:
                this_name = 'encoder_level{0:d}_dropout{1:d}'.format(i, j)
                conv_layer_by_level[i] = architecture_utils.get_dropout_layer(
                    dropout_fraction=encoder_dropout_rate_by_level[i],
                    layer_name=this_name
                )(conv_layer_by_level[i])

            if use_batch_normalization:
                this_name = 'encoder_level{0:d}_bn{1:d}'.format(i, j)
                conv_layer_by_level[i] = architecture_utils.get_batch_norm_layer(
                    layer_name=this_name
                )(conv_layer_by_level[i])

        this_name = 'encoder_level{0:d}_pooling'.format(i)
        this_pooling_layer_object = architecture_utils.get_2d_pooling_layer(
            num_rows_in_window=pooling_size_by_level_px[i],
            num_columns_in_window=pooling_size_by_level_px[i],
            num_rows_per_stride=pooling_size_by_level_px[i],
            num_columns_per_stride=pooling_size_by_level_px[i],
            pooling_type_string=architecture_utils.MAX_POOLING_STRING,
            layer_name=this_name
        )
        pooling_layer_by_level[i] = keras.layers.TimeDistributed(
            this_pooling_layer_object, name=this_name
        )(conv_layer_by_level[i])

        num_levels_filled += 1

    for i in range(num_levels_filled, num_levels + 1):
        for j in range(num_encoder_conv_layers_by_level[i]):
            if j == 0:
                previous_layer_object = pooling_layer_by_level[i - 1]
            else:
                previous_layer_object = conv_layer_by_level[i]

            this_name = 'encoder_level{0:d}_conv{1:d}'.format(i, j)
            this_conv_layer_object = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=num_channels_by_level[i],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=l2_function,
                layer_name=this_name
            )

            conv_layer_by_level[i] = keras.layers.TimeDistributed(
                this_conv_layer_object, name=this_name
            )(previous_layer_object)

            this_name = 'encoder_level{0:d}_activation{1:d}'.format(i, j)
            conv_layer_by_level[i] = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha,
                layer_name=this_name
            )(conv_layer_by_level[i])

            if encoder_dropout_rate_by_level[i] > 0:
                this_name = 'encoder_level{0:d}_dropout{1:d}'.format(i, j)
                conv_layer_by_level[i] = architecture_utils.get_dropout_layer(
                    dropout_fraction=encoder_dropout_rate_by_level[i],
                    layer_name=this_name
                )(conv_layer_by_level[i])

            if use_batch_normalization:
                this_name = 'encoder_level{0:d}_bn{1:d}'.format(i, j)
                conv_layer_by_level[i] = architecture_utils.get_batch_norm_layer(
                    layer_name=this_name
                )(conv_layer_by_level[i])

        if i != num_levels:
            this_name = 'encoder_level{0:d}_pooling'.format(i)
            this_pooling_layer_object = architecture_utils.get_2d_pooling_layer(
                num_rows_in_window=pooling_size_by_level_px[i],
                num_columns_in_window=pooling_size_by_level_px[i],
                num_rows_per_stride=pooling_size_by_level_px[i],
                num_columns_per_stride=pooling_size_by_level_px[i],
                pooling_type_string=architecture_utils.MAX_POOLING_STRING,
                layer_name=this_name
            )
            pooling_layer_by_level[i] = keras.layers.TimeDistributed(
                this_pooling_layer_object, name=this_name
            )(conv_layer_by_level[i])

    forecast_module_layer_object = keras.layers.Permute(
        dims=(2, 3, 1, 4), name='fc_module_put_time_last'
    )(conv_layer_by_level[-1])

    if not forecast_module_use_3d_conv:
        orig_dims = forecast_module_layer_object.shape
        new_dims = orig_dims[1:-2] + [orig_dims[-2] * orig_dims[-1]]

        forecast_module_layer_object = keras.layers.Reshape(
            target_shape=new_dims, name='fc_module_remove_time_dim'
        )(forecast_module_layer_object)

    for j in range(forecast_module_num_conv_layers):
        if forecast_module_use_3d_conv:
            this_name = 'fc_module_conv3d_{0:d}'.format(j)

            if j == 0:
                forecast_module_layer_object = (
                    architecture_utils.get_3d_conv_layer(
                        num_kernel_rows=1, num_kernel_columns=1,
                        num_kernel_heights=num_lead_times,
                        num_rows_per_stride=1, num_columns_per_stride=1,
                        num_heights_per_stride=1,
                        num_filters=num_channels_by_level[-1],
                        padding_type_string=
                        architecture_utils.NO_PADDING_STRING,
                        weight_regularizer=l2_function,
                        layer_name=this_name
                    )(forecast_module_layer_object)
                )

                new_dims = (
                    forecast_module_layer_object.shape[1:-2] +
                    (forecast_module_layer_object.shape[-1],)
                )
                forecast_module_layer_object = keras.layers.Reshape(
                    target_shape=new_dims, name='fc_module_remove_time_dim'
                )(forecast_module_layer_object)
            else:
                forecast_module_layer_object = (
                    architecture_utils.get_2d_conv_layer(
                        num_kernel_rows=3, num_kernel_columns=3,
                        num_rows_per_stride=1, num_columns_per_stride=1,
                        num_filters=num_channels_by_level[-1],
                        padding_type_string=
                        architecture_utils.YES_PADDING_STRING,
                        weight_regularizer=l2_function,
                        layer_name=this_name
                    )(forecast_module_layer_object)
                )
        else:
            this_name = 'fc_module_conv2d_{0:d}'.format(j)
            forecast_module_layer_object = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=num_channels_by_level[-1],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=l2_function,
                layer_name=this_name
            )(forecast_module_layer_object)

        this_name = 'fc_module_activation{0:d}'.format(j)
        forecast_module_layer_object = architecture_utils.get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha,
            layer_name=this_name
        )(forecast_module_layer_object)

        if forecast_module_dropout_rates[j] > 0:
            this_name = 'fc_module_dropout{0:d}'.format(j)
            forecast_module_layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=forecast_module_dropout_rates[j],
                layer_name=this_name
            )(forecast_module_layer_object)

        if use_batch_normalization:
            this_name = 'fc_module_bn{0:d}'.format(j)
            forecast_module_layer_object = (
                architecture_utils.get_batch_norm_layer(layer_name=this_name)(
                    forecast_module_layer_object
                )
            )

    upconv_layer_by_level = [None] * num_levels
    skip_layer_by_level = [None] * num_levels
    merged_layer_by_level = [None] * num_levels

    this_name = 'upsampling_level{0:d}'.format(num_levels - 1)
    size_arg = (
        pooling_size_by_level_px[num_levels - 1],
        pooling_size_by_level_px[num_levels - 1]
    )

    try:
        this_layer_object = keras.layers.UpSampling2D(
            size=size_arg, interpolation='bilinear', name=this_name
        )(forecast_module_layer_object)
    except:
        this_layer_object = keras.layers.UpSampling2D(
            size=size_arg, name=this_name
        )(forecast_module_layer_object)

    this_name = 'upsampling_level{0:d}_conv'.format(num_levels - 1)
    i = num_levels - 1

    upconv_layer_by_level[i] = architecture_utils.get_2d_conv_layer(
        num_kernel_rows=2, num_kernel_columns=2,
        num_rows_per_stride=1, num_columns_per_stride=1,
        num_filters=num_channels_by_level[i],
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=l2_function,
        layer_name=this_name
    )(this_layer_object)

    this_name = 'upsampling_level{0:d}_activation'.format(num_levels - 1)
    upconv_layer_by_level[i] = architecture_utils.get_activation_layer(
        activation_function_string=inner_activ_function_name,
        alpha_for_relu=inner_activ_function_alpha,
        alpha_for_elu=inner_activ_function_alpha,
        layer_name=this_name
    )(upconv_layer_by_level[i])

    if upsampling_dropout_rate_by_level[i] > 0:
        this_name = 'upsampling_level{0:d}_dropout'.format(i)

        upconv_layer_by_level[i] = architecture_utils.get_dropout_layer(
            dropout_fraction=upsampling_dropout_rate_by_level[i],
            layer_name=this_name
        )(upconv_layer_by_level[i])

    if use_batch_normalization:
        this_name = 'upsampling_level{0:d}_bn'.format(i)
        upconv_layer_by_level[i] = architecture_utils.get_batch_norm_layer(
            layer_name=this_name
        )(upconv_layer_by_level[i])

    this_function = _get_time_slicing_function(-1)
    this_name = 'skip_level{0:d}_take_last_time'.format(i)
    conv_layer_by_level[i] = keras.layers.Lambda(
        this_function, name=this_name
    )(conv_layer_by_level[i])

    num_upconv_rows = upconv_layer_by_level[i].shape[1]
    num_desired_rows = conv_layer_by_level[i].shape[1]
    num_padding_rows = num_desired_rows - num_upconv_rows

    num_upconv_columns = upconv_layer_by_level[i].shape[2]
    num_desired_columns = conv_layer_by_level[i].shape[2]
    num_padding_columns = num_desired_columns - num_upconv_columns

    if num_padding_rows + num_padding_columns > 0:
        padding_arg = ((0, num_padding_rows), (0, num_padding_columns))
        this_name = 'padding_level{0:d}'.format(i)

        upconv_layer_by_level[i] = keras.layers.ZeroPadding2D(
            padding=padding_arg, name=this_name
        )(upconv_layer_by_level[i])

    this_name = 'skip_level{0:d}'.format(i)
    merged_layer_by_level[i] = keras.layers.Concatenate(
        axis=-1, name=this_name
    )(
        [conv_layer_by_level[i], upconv_layer_by_level[i]]
    )

    level_indices = numpy.linspace(
        0, num_levels - 1, num=num_levels, dtype=int
    )[::-1]

    for i in level_indices:
        for j in range(num_decoder_conv_layers_by_level[i]):
            if j == 0:
                this_input_layer_object = merged_layer_by_level[i]
            else:
                this_input_layer_object = skip_layer_by_level[i]

            this_name = 'skip_level{0:d}_conv{1:d}'.format(i, j)
            skip_layer_by_level[i] = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=num_channels_by_level[i],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=l2_function,
                layer_name=this_name
            )(this_input_layer_object)

            this_name = 'skip_level{0:d}_conv{1:d}_activation'.format(i, j)
            skip_layer_by_level[i] = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha,
                layer_name=this_name
            )(skip_layer_by_level[i])

            if skip_dropout_rate_by_level[i] > 0:
                this_name = 'skip_level{0:d}_conv{1:d}_dropout'.format(i, j)
                skip_layer_by_level[i] = architecture_utils.get_dropout_layer(
                    dropout_fraction=skip_dropout_rate_by_level[i],
                    layer_name=this_name
                )(skip_layer_by_level[i])

            if use_batch_normalization:
                this_name = 'skip_level{0:d}_conv{1:d}_bn'.format(i, j)
                skip_layer_by_level[i] = (
                    architecture_utils.get_batch_norm_layer(
                        layer_name=this_name
                    )(skip_layer_by_level[i])
                )

        if i == 0 and include_penultimate_conv:
            skip_layer_by_level[i] = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=2 * ensemble_size * num_output_channels,
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=l2_function,
                layer_name='penultimate_conv'
            )(skip_layer_by_level[i])

            skip_layer_by_level[i] = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha,
                layer_name='penultimate_conv_activation'
            )(skip_layer_by_level[i])

            if penultimate_conv_dropout_rate > 0:
                skip_layer_by_level[i] = architecture_utils.get_dropout_layer(
                    dropout_fraction=penultimate_conv_dropout_rate,
                    layer_name='penultimate_conv_dropout'
                )(skip_layer_by_level[i])

            if use_batch_normalization:
                skip_layer_by_level[i] = (
                    architecture_utils.get_batch_norm_layer(
                        layer_name='penultimate_conv_bn'
                    )(skip_layer_by_level[i])
                )

            break

        this_name = 'upsampling_level{0:d}'.format(i - 1)
        size_arg = (
            pooling_size_by_level_px[i - 1],
            pooling_size_by_level_px[i - 1]
        )

        try:
            this_layer_object = keras.layers.UpSampling2D(
                size=size_arg, interpolation='bilinear', name=this_name
            )(skip_layer_by_level[i])
        except:
            this_layer_object = keras.layers.UpSampling2D(
                size=size_arg, name=this_name
            )(skip_layer_by_level[i])

        this_name = 'upsampling_level{0:d}_conv'.format(i - 1)
        upconv_layer_by_level[i - 1] = architecture_utils.get_2d_conv_layer(
            num_kernel_rows=2, num_kernel_columns=2,
            num_rows_per_stride=1, num_columns_per_stride=1,
            num_filters=num_channels_by_level[i - 1],
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=l2_function,
            layer_name=this_name
        )(this_layer_object)

        this_name = 'upsampling_level{0:d}_activation'.format(i - 1)
        upconv_layer_by_level[i - 1] = architecture_utils.get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha,
            layer_name=this_name
        )(upconv_layer_by_level[i - 1])

        if upsampling_dropout_rate_by_level[i - 1] > 0:
            this_name = 'upsampling_level{0:d}_dropout'.format(i - 1)
            upconv_layer_by_level[i - 1] = architecture_utils.get_dropout_layer(
                dropout_fraction=upsampling_dropout_rate_by_level[i - 1],
                layer_name=this_name
            )(upconv_layer_by_level[i - 1])

        if use_batch_normalization:
            this_name = 'upsampling_level{0:d}_bn'.format(i - 1)
            upconv_layer_by_level[i - 1] = architecture_utils.get_batch_norm_layer(
                layer_name=this_name
            )(upconv_layer_by_level[i - 1])

        this_function = _get_time_slicing_function(-1)
        this_name = 'skip_level{0:d}_take_last_time'.format(i - 1)
        conv_layer_by_level[i - 1] = keras.layers.Lambda(
            this_function, name=this_name
        )(conv_layer_by_level[i - 1])

        num_upconv_rows = upconv_layer_by_level[i - 1].shape[1]
        num_desired_rows = conv_layer_by_level[i - 1].shape[1]
        num_padding_rows = num_desired_rows - num_upconv_rows

        num_upconv_columns = upconv_layer_by_level[i - 1].shape[2]
        num_desired_columns = conv_layer_by_level[i - 1].shape[2]
        num_padding_columns = num_desired_columns - num_upconv_columns

        if num_padding_rows + num_padding_columns > 0:
            padding_arg = ((0, num_padding_rows), (0, num_padding_columns))
            this_name = 'padding_level{0:d}'.format(i - 1)

            upconv_layer_by_level[i - 1] = keras.layers.ZeroPadding2D(
                padding=padding_arg, name=this_name
            )(upconv_layer_by_level[i - 1])

        this_name = 'skip_level{0:d}'.format(i - 1)
        merged_layer_by_level[i - 1] = keras.layers.Concatenate(
            axis=-1, name=this_name
        )(
            [conv_layer_by_level[i - 1], upconv_layer_by_level[i - 1]]
        )

    this_offset = int(predict_gust_factor) + int(predict_dewpoint_depression)

    simple_output_layer_object = architecture_utils.get_2d_conv_layer(
        num_kernel_rows=1, num_kernel_columns=1,
        num_rows_per_stride=1, num_columns_per_stride=1,
        num_filters=(num_output_channels - this_offset) * ensemble_size,
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=l2_function,
        layer_name='last_conv_simple'
    )(skip_layer_by_level[0])

    if output_activ_function_name is not None:
        simple_output_layer_object = architecture_utils.get_activation_layer(
            activation_function_string=output_activ_function_name,
            alpha_for_relu=output_activ_function_alpha,
            alpha_for_elu=output_activ_function_alpha,
            layer_name='last_conv_simple_activation'
        )(simple_output_layer_object)

    if predict_dewpoint_depression:
        dd_output_layer_object = architecture_utils.get_2d_conv_layer(
            num_kernel_rows=1, num_kernel_columns=1,
            num_rows_per_stride=1, num_columns_per_stride=1,
            num_filters=ensemble_size,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=l2_function,
            layer_name='last_conv_dd'
        )(skip_layer_by_level[0])

        dd_output_layer_object = architecture_utils.get_activation_layer(
            activation_function_string=architecture_utils.RELU_FUNCTION_STRING,
            alpha_for_relu=0.,
            alpha_for_elu=0.,
            layer_name='last_conv_dd_activation'
        )(dd_output_layer_object)
    else:
        dd_output_layer_object = None

    if predict_gust_factor:
        gf_output_layer_object = architecture_utils.get_2d_conv_layer(
            num_kernel_rows=1, num_kernel_columns=1,
            num_rows_per_stride=1, num_columns_per_stride=1,
            num_filters=ensemble_size,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=l2_function,
            layer_name='last_conv_gf'
        )(skip_layer_by_level[0])

        gf_output_layer_object = architecture_utils.get_activation_layer(
            activation_function_string=architecture_utils.RELU_FUNCTION_STRING,
            alpha_for_relu=0.,
            alpha_for_elu=0.,
            layer_name='last_conv_gf_activation'
        )(gf_output_layer_object)
    else:
        gf_output_layer_object = None

    these_layer_objects = [
        simple_output_layer_object, dd_output_layer_object,
        gf_output_layer_object
    ]
    these_layer_objects = [l for l in these_layer_objects if l is not None]

    if len(these_layer_objects) == 1:
        output_layer_object = simple_output_layer_object
    else:
        output_layer_object = keras.layers.Concatenate(
            axis=-1, name='last_conv_concat'
        )(these_layer_objects)

    # TODO(thunderhoser): For now, input_dimensions_2pt5km_res cannot actually
    # be None.  In other words, the model must take 2.5-km data as input.  I
    # will change this if need be.
    if ensemble_size > 1:
        new_dims = (
            input_dimensions_2pt5km_res[0], input_dimensions_2pt5km_res[1],
            num_output_channels, ensemble_size
        )
        output_layer_object = keras.layers.Reshape(
            target_shape=new_dims, name='reshape_predictions'
        )(output_layer_object)

    input_layer_objects = [
        l for l in [
            input_layer_object_2pt5km_res, input_layer_object_const,
            input_layer_object_10km_res, input_layer_object_20km_res,
            input_layer_object_40km_res
        ] if l is not None
    ]

    model_object = keras.models.Model(
        inputs=input_layer_objects, outputs=output_layer_object
    )

    model_object.compile(
        loss=loss_function, optimizer=optimizer_function,
        metrics=metric_function_list
    )

    model_object.summary()
    return model_object
