"""A simple architecture for complicated times."""

import numpy
import keras
from ml_for_national_blend.outside_code import architecture_utils
from ml_for_national_blend.outside_code import error_checking
from ml_for_national_blend.utils import urma_utils
from ml_for_national_blend.machine_learning import custom_losses
from ml_for_national_blend.machine_learning import custom_metrics

INPUT_DIMENSIONS_KEY = 'input_dimensions'

NWP_ENCODER_NUM_CHANNELS_KEY = 'nwp_encoder_num_channels_by_level'
NWP_POOLING_SIZE_KEY = 'nwp_pooling_size_by_level_px'
NWP_ENCODER_NUM_CONV_LAYERS_KEY = 'nwp_encoder_num_conv_layers_by_level'
NWP_ENCODER_DROPOUT_RATES_KEY = 'nwp_encoder_dropout_rate_by_level'

DECODER_NUM_CHANNELS_KEY = 'decoder_num_channels_by_level'
DECODER_NUM_CONV_LAYERS_KEY = 'decoder_num_conv_layers_by_level'
UPSAMPLING_DROPOUT_RATES_KEY = 'upsampling_dropout_rate_by_level'
SKIP_DROPOUT_RATES_KEY = 'skip_dropout_rate_by_level'

INNER_ACTIV_FUNCTION_KEY = 'inner_activ_function_name'
INNER_ACTIV_FUNCTION_ALPHA_KEY = 'inner_activ_function_alpha'

L1_WEIGHT_KEY = 'l1_weight'
L2_WEIGHT_KEY = 'l2_weight'
USE_BATCH_NORM_KEY = 'use_batch_normalization'
ENSEMBLE_SIZE_KEY = 'ensemble_size'

TARGET_FIELDS_KEY = 'target_field_names'
LOSS_FUNCTION_KEY = 'loss_function'
OPTIMIZER_FUNCTION_KEY = 'optimizer_function'
METRIC_FUNCTIONS_KEY = 'metric_function_list'

DEFAULT_TARGET_NAMES = [
    urma_utils.TEMPERATURE_2METRE_NAME, urma_utils.U_WIND_10METRE_NAME,
    urma_utils.V_WIND_10METRE_NAME, urma_utils.DEWPOINT_2METRE_NAME,
    urma_utils.WIND_GUST_10METRE_NAME
]
DEFAULT_CHANNEL_WEIGHTS_IN_LOSS = numpy.array([
    0.02056336, 0.40520461, 0.33582517, 0.03271094, 0.20569591
])
DEFAULT_LOSS_FUNCTION = custom_losses.dual_weighted_crpss(
    channel_weights=DEFAULT_CHANNEL_WEIGHTS_IN_LOSS,
    temperature_index=0, u_wind_index=1, v_wind_index=2,
    dewpoint_index=3, gust_index=4,
    function_name='loss_dwcrpss'
)

DEFAULT_METRIC_FUNCTIONS = [
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

DEFAULT_OPTION_DICT = {
    INPUT_DIMENSIONS_KEY: numpy.array([208, 208, 7], dtype=int),
    NWP_ENCODER_NUM_CHANNELS_KEY: numpy.array([32, 48, 64, 96, 128, 192], dtype=int),
    NWP_POOLING_SIZE_KEY: numpy.array([2, 2, 2, 2, 2], dtype=int),
    NWP_ENCODER_NUM_CONV_LAYERS_KEY: numpy.array([1, 1, 1, 1, 1, 1], dtype=int),
    NWP_ENCODER_DROPOUT_RATES_KEY: numpy.array([0, 0, 0, 0, 0, 0], dtype=float),
    DECODER_NUM_CHANNELS_KEY: numpy.array([32, 48, 64, 96, 128], dtype=int),
    DECODER_NUM_CONV_LAYERS_KEY: numpy.array([1, 1, 1, 1, 1], dtype=int),
    UPSAMPLING_DROPOUT_RATES_KEY: numpy.array([0, 0, 0, 0, 0], dtype=float),
    SKIP_DROPOUT_RATES_KEY: numpy.array([0, 0, 0, 0, 0], dtype=float),
    INNER_ACTIV_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    L1_WEIGHT_KEY: 0.,
    L2_WEIGHT_KEY: 1e-7,
    USE_BATCH_NORM_KEY: True,
    ENSEMBLE_SIZE_KEY: 25,
    TARGET_FIELDS_KEY: DEFAULT_TARGET_NAMES,
    LOSS_FUNCTION_KEY: DEFAULT_LOSS_FUNCTION,
    OPTIMIZER_FUNCTION_KEY: keras.optimizers.AdamW(),
    METRIC_FUNCTIONS_KEY: DEFAULT_METRIC_FUNCTIONS
}


def _check_input_args(option_dict):
    """Error-checks input arguments for U-net architecture.

    L = number of depth levels in the U-net, i.e., number of times we
        pool/upsample

    :param option_dict: Dictionary with the following keys.
    option_dict["input_dimensions"]: Dimensions of input data, which are only
        2.5-km NWP forecasts.  The simple U-net architecture does not handle
        lower-resolution NWP forecasts (at 10, 20, or 40 km), recent NWP biases,
        or lagged truth fields.  It also doesn't do residual prediction.
    option_dict["nwp_encoder_num_channels_by_level"]: numpy array, of length
        L + 1, with number of channels (feature maps) at each level of the NWP-
        encoder.  For this simple U-net architecture, the NWP-encoder is the
        only encoder, because NWP forecasts are the only input-data stream.
    option_dict["nwp_pooling_size_by_level_px"]: length-L numpy array with
        pooling size (in units of pixels) at each level of the NWP-encoder.
    option_dict["nwp_encoder_num_conv_layers_by_level"]: length-(L + 1) numpy
        array with number of convolutional layers at each level of the
        NWP-encoder.
    option_dict["nwp_encoder_dropout_rate_by_level"]: length-(L + 1) numpy array
        with dropout rate at each level of the NWP-encoder.  Dropout with a rate
        of nwp_encoder_dropout_rate_by_level[k] will be applied to every conv
        layer at the [k]th level of the encoder.  To omit dropout in the [k]th
        level, make nwp_encoder_dropout_rate_by_level[k] <= 0.
    option_dict["decoder_num_channels_by_level"]: length-L numpy array
        with number of channels (feature maps) at each level of the decoder.
    option_dict["decoder_num_conv_layers_by_level"]: length-L numpy array with
        number of convolutional layers at each level of the decoder.
    option_dict["upsampling_dropout_rate_by_level"]: length-L numpy array with
        dropout rate for upsampling at each level of the decoder.  Dropout with
        a rate of upsampling_dropout_rate_by_level[k] will be applied to the
        upsampling (convolutional) layer at the [k]th level of the decoder.  To
        omit dropout in the [k]th level, make
        upsampling_dropout_rate_by_level[k] <= 0.
    option_dict["skip_dropout_rate_by_level"]: Same as above but for skip
        connections in the decoder.
    option_dict["inner_activ_function_name"]: Name of activation function, used
        for all internal convolutional layers (i.e., NOT the output layer).
        To omit activation (not recommended), make this None.  Otherwise, this
        must be a string accepted by
        `architecture_utils.check_activation_function`.
    option_dict["inner_activ_function_alpha"]: Alpha (slope parameter) for the
        above activation function.  This applies only to the eLU and ReLU
        activation functions.  If your activation function is something else,
        just make this alpha 0.
    option_dict["l1_weight"]: Weight for L1 regularization.  I recommended just
        making this 0.
    option_dict["l2_weight"]: Weight for L2 regularization.
    option_dict["use_batch_normalization"]: Boolean flag.  If True, batch
        normalization will be used after every convolutional layer.  I recommend
        making this True.
    option_dict["ensemble_size"]: Ensemble size -- i.e., number of predictions
        produced by the NN for every pair of {grid point, target field}.  If you
        want a deterministic model, make this 1.
    option_dict["target_field_names"]: 1-D list with names of target fields.
        Each string in this list must be accepted by
        `urma_utils.check_field_name`.
    option_dict["loss_function"]: Loss function.  This must be a function
        handle.  You can find an example at the top of this file
        (DEFAULT_LOSS_FUNCTION).
    option_dict["optimizer_function"]: Optimizer -- this must also be a function
        handle.  You can find an example at the top of this file (AdamW).
    option_dict["metric_function_list"]: 1-D list of metric functions; each list
        item must be a function handle.  You can find an example at the top of
        this file (DEFAULT_METRIC_FUNCTIONS).  Note the difference between the
        loss and metrics.  The loss is used to guide the training (i.e., it is
        The Thing that training attempts to minimize), while the metrics are
        just reported (printed to the log file / terminal) during training for
        the user's information.  Metrics can be useful for debugging, so I
        highly recommend including at least those in the default list (mean
        squared error for every target field).

    :return: option_dict: Same as input, except that missing arguments have been
        replaced with defaults.
    """

    orig_option_dict = option_dict.copy()
    option_dict = DEFAULT_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    # Check input dimensions.  Only 2.5-km NWP forecasts are accepted by this
    # simple architecture.
    input_dimensions = option_dict[INPUT_DIMENSIONS_KEY]
    error_checking.assert_is_numpy_array(
        input_dimensions, exact_dimensions=numpy.array([3], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(
        option_dict[input_dimensions]
    )
    error_checking.assert_is_greater_numpy_array(
        option_dict[input_dimensions], 0
    )

    # Check target fields.  If target fields include dewpoint, the NN will
    # actually predict dewpoint depression, which means that temperature must be
    # included in the target fields.  If target fields include wind gust, the NN
    # will actually predict gust excess, which means that (sustained) u-wind and
    # v-wind speeds must be included in the target fields.
    target_field_names = option_dict[TARGET_FIELDS_KEY]

    error_checking.assert_is_string_list(target_field_names)
    for this_field_name in target_field_names:
        urma_utils.check_field_name(this_field_name)

    predict_dewpoint_depression = (
        urma_utils.DEWPOINT_2METRE_NAME in target_field_names
    )
    if predict_dewpoint_depression:
        assert urma_utils.TEMPERATURE_2METRE_NAME in target_field_names
        target_field_names.remove(urma_utils.DEWPOINT_2METRE_NAME)
        target_field_names.append(urma_utils.DEWPOINT_2METRE_NAME)

    predict_gust_excess = (
        urma_utils.WIND_GUST_10METRE_NAME in target_field_names
    )
    if predict_gust_excess:
        assert urma_utils.U_WIND_10METRE_NAME in target_field_names
        assert urma_utils.V_WIND_10METRE_NAME in target_field_names
        target_field_names.remove(urma_utils.WIND_GUST_10METRE_NAME)
        target_field_names.append(urma_utils.WIND_GUST_10METRE_NAME)

    # Check settings for NWP-encoder -- which, for this simple architecture, is
    # the only encoder.  In other words, we have an encoder only for NWP
    # forecasts, none for recent NWP biases or lagged truth.
    nwp_encoder_num_channels_by_level = option_dict[
        NWP_ENCODER_NUM_CHANNELS_KEY
    ]
    error_checking.assert_is_numpy_array(
        nwp_encoder_num_channels_by_level, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(
        nwp_encoder_num_channels_by_level
    )
    error_checking.assert_is_geq_numpy_array(
        nwp_encoder_num_channels_by_level, 1
    )
    num_levels = len(nwp_encoder_num_channels_by_level) - 1

    nwp_pooling_size_by_level_px = option_dict[NWP_POOLING_SIZE_KEY]
    error_checking.assert_is_numpy_array(
        nwp_pooling_size_by_level_px,
        exact_dimensions=numpy.array([num_levels], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(nwp_pooling_size_by_level_px)
    error_checking.assert_is_geq_numpy_array(nwp_pooling_size_by_level_px, 2)

    nwp_encoder_num_conv_layers_by_level = option_dict[
        NWP_ENCODER_NUM_CONV_LAYERS_KEY
    ]
    error_checking.assert_is_numpy_array(
        nwp_encoder_num_conv_layers_by_level,
        exact_dimensions=numpy.array([num_levels + 1], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(
        nwp_encoder_num_conv_layers_by_level
    )
    error_checking.assert_is_geq_numpy_array(
        nwp_encoder_num_conv_layers_by_level, 1
    )

    nwp_encoder_dropout_rate_by_level = option_dict[
        NWP_ENCODER_DROPOUT_RATES_KEY
    ]
    error_checking.assert_is_numpy_array(
        nwp_encoder_dropout_rate_by_level,
        exact_dimensions=numpy.array([num_levels + 1], dtype=int)
    )
    error_checking.assert_is_leq_numpy_array(
        nwp_encoder_dropout_rate_by_level, 1., allow_nan=True
    )

    # Check settings for decoder.
    decoder_num_channels_by_level = option_dict[DECODER_NUM_CHANNELS_KEY]
    error_checking.assert_is_numpy_array(
        decoder_num_channels_by_level,
        exact_dimensions=numpy.array([num_levels], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(decoder_num_channels_by_level)
    error_checking.assert_is_geq_numpy_array(decoder_num_channels_by_level, 1)

    decoder_num_conv_layers_by_level = option_dict[DECODER_NUM_CONV_LAYERS_KEY]
    error_checking.assert_is_numpy_array(
        decoder_num_conv_layers_by_level,
        exact_dimensions=numpy.array([num_levels], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(
        decoder_num_conv_layers_by_level
    )
    error_checking.assert_is_geq_numpy_array(
        decoder_num_conv_layers_by_level, 1
    )

    upsampling_dropout_rate_by_level = option_dict[UPSAMPLING_DROPOUT_RATES_KEY]
    error_checking.assert_is_numpy_array(
        upsampling_dropout_rate_by_level,
        exact_dimensions=numpy.array([num_levels], dtype=int)
    )
    error_checking.assert_is_leq_numpy_array(
        upsampling_dropout_rate_by_level, 1., allow_nan=True
    )

    skip_dropout_rate_by_level = option_dict[SKIP_DROPOUT_RATES_KEY]
    error_checking.assert_is_numpy_array(
        skip_dropout_rate_by_level,
        exact_dimensions=numpy.array([num_levels], dtype=int)
    )
    error_checking.assert_is_leq_numpy_array(
        skip_dropout_rate_by_level, 1., allow_nan=True
    )

    # Check activation function for internal (non-output) layers.
    architecture_utils.check_activation_function(
        activation_function_string=option_dict[INNER_ACTIV_FUNCTION_KEY],
        alpha_for_elu=option_dict[INNER_ACTIV_FUNCTION_ALPHA_KEY],
        alpha_for_relu=option_dict[INNER_ACTIV_FUNCTION_ALPHA_KEY],
        for_conv_layer=True
    )

    # Check regularization settings.
    error_checking.assert_is_geq(option_dict[L1_WEIGHT_KEY], 0.)
    error_checking.assert_is_geq(option_dict[L2_WEIGHT_KEY], 0.)

    # Check miscellaneous shit.
    error_checking.assert_is_boolean(option_dict[USE_BATCH_NORM_KEY])
    error_checking.assert_is_integer(option_dict[ENSEMBLE_SIZE_KEY])
    error_checking.assert_is_geq(option_dict[ENSEMBLE_SIZE_KEY], 1)
    error_checking.assert_is_list(option_dict[METRIC_FUNCTIONS_KEY])

    return option_dict


def create_model(option_dict):
    """Creates a simple U-net architecture.

    :param option_dict: See documentation for `_check_input_args`.
    :return: model_object: Untrained (but compiled) instance of
       `keras.models.Model`.
    """

    # Read input arguments.
    input_dimensions = option_dict[INPUT_DIMENSIONS_KEY]

    encoder_num_channels_by_level = option_dict[NWP_ENCODER_NUM_CHANNELS_KEY]
    pooling_size_by_level_px = option_dict[NWP_POOLING_SIZE_KEY]
    encoder_num_conv_layers_by_level = option_dict[
        NWP_ENCODER_NUM_CONV_LAYERS_KEY
    ]
    encoder_dropout_rate_by_level = option_dict[NWP_ENCODER_DROPOUT_RATES_KEY]

    decoder_num_channels_by_level = option_dict[DECODER_NUM_CHANNELS_KEY]
    decoder_num_conv_layers_by_level = option_dict[DECODER_NUM_CONV_LAYERS_KEY]
    upsampling_dropout_rate_by_level = option_dict[UPSAMPLING_DROPOUT_RATES_KEY]
    skip_dropout_rate_by_level = option_dict[SKIP_DROPOUT_RATES_KEY]

    inner_activ_function_name = option_dict[INNER_ACTIV_FUNCTION_KEY]
    inner_activ_function_alpha = option_dict[INNER_ACTIV_FUNCTION_ALPHA_KEY]

    l1_weight = option_dict[L1_WEIGHT_KEY]
    l2_weight = option_dict[L2_WEIGHT_KEY]
    use_batch_normalization = option_dict[USE_BATCH_NORM_KEY]
    ensemble_size = option_dict[ENSEMBLE_SIZE_KEY]

    target_field_names = option_dict[TARGET_FIELDS_KEY]
    loss_function = option_dict[LOSS_FUNCTION_KEY]
    optimizer_function = option_dict[OPTIMIZER_FUNCTION_KEY]
    metric_function_list = option_dict[METRIC_FUNCTIONS_KEY]
    num_output_channels = len(target_field_names)

    l2_function = architecture_utils.get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight
    )

    # Create input layer.
    input_layer_object = keras.layers.Input(
        shape=tuple(input_dimensions.tolist()),
        name='nwp_inputs_2pt5km'
    )

    # Create encoder.
    num_levels = len(pooling_size_by_level_px)
    conv_layer_by_level = [None] * (num_levels + 1)
    pooling_layer_by_level = [None] * num_levels

    for i in range(num_levels + 1):
        for j in range(encoder_num_conv_layers_by_level[i]):
            if i == 0 and j == 0:
                previous_layer_object = input_layer_object
            elif j == 0:
                previous_layer_object = pooling_layer_by_level[i - 1]
            else:
                previous_layer_object = conv_layer_by_level[i]

            this_name = 'encoder_level{0:d}_conv{1:d}'.format(i, j)
            conv_layer_by_level[i] = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=encoder_num_channels_by_level[i],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=l2_function,
                layer_name=this_name
            )(previous_layer_object)

            if inner_activ_function_name is not None:
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
            pooling_layer_by_level[i] = architecture_utils.get_2d_pooling_layer(
                num_rows_in_window=pooling_size_by_level_px[i],
                num_columns_in_window=pooling_size_by_level_px[i],
                num_rows_per_stride=pooling_size_by_level_px[i],
                num_columns_per_stride=pooling_size_by_level_px[i],
                pooling_type_string=architecture_utils.MAX_POOLING_STRING,
                layer_name=this_name
            )(conv_layer_by_level[i])

    # Start creating decoder -- i.e., upsample from coarsest resolution
    # (bottleneck level of U-net) to second-coarsest resolution.
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
        )(conv_layer_by_level[-1])
    except:
        this_layer_object = keras.layers.UpSampling2D(
            size=size_arg, name=this_name
        )(conv_layer_by_level[-1])

    this_name = 'upsampling_level{0:d}_conv'.format(num_levels - 1)
    i = num_levels - 1

    upconv_layer_by_level[i] = architecture_utils.get_2d_conv_layer(
        num_kernel_rows=2, num_kernel_columns=2,
        num_rows_per_stride=1, num_columns_per_stride=1,
        num_filters=decoder_num_channels_by_level[i],
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=l2_function,
        layer_name=this_name
    )(this_layer_object)

    if inner_activ_function_alpha is not None:
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

    # Do padding if necessary.  For example, suppose that data in the bottleneck
    # level (at the coarsest resolution) are 6 x 6 pixels, while data at the
    # next-coarsest resolution have 13 x 13 pixels.  The upsampling layer will
    # get you from 6 x 6 to 12 x 12, not to 13 x 13 -- so then you need an extra
    # row and an extra column.
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

    # Create the skip connection at this deepest level.
    this_name = 'skip_level{0:d}'.format(i)
    merged_layer_by_level[i] = keras.layers.Concatenate(
        axis=-1, name=this_name
    )(
        [conv_layer_by_level[i], upconv_layer_by_level[i]]
    )

    # Now create the rest of the decoder.
    level_indices = numpy.linspace(
        0, num_levels - 1, num=num_levels, dtype=int
    )[::-1]

    for i in level_indices:

        # Do convolutions after the skip connection.
        for j in range(decoder_num_conv_layers_by_level[i]):
            if j == 0:
                this_input_layer_object = merged_layer_by_level[i]
            else:
                this_input_layer_object = skip_layer_by_level[i]

            this_name = 'skip_level{0:d}_conv{1:d}'.format(i, j)
            skip_layer_by_level[i] = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=decoder_num_channels_by_level[i],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=l2_function,
                layer_name=this_name
            )(this_input_layer_object)

            if inner_activ_function_name is not None:
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

        # Upsample to the next-coarsest resolution.
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

        # This convolutional layer helps recover higher-resolution information
        # that cannot be recovered by the simple `UpSampling2D` layer, which
        # just does interpolation.  The skip connection and post-skip-connection
        # convolutional layers recover higher-resolution information even more
        # effectively.
        this_name = 'upsampling_level{0:d}_conv'.format(i - 1)
        upconv_layer_by_level[i - 1] = architecture_utils.get_2d_conv_layer(
            num_kernel_rows=2, num_kernel_columns=2,
            num_rows_per_stride=1, num_columns_per_stride=1,
            num_filters=decoder_num_channels_by_level[i - 1],
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=l2_function,
            layer_name=this_name
        )(this_layer_object)

        if inner_activ_function_name is not None:
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

        # Do padding if necessary.
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

        # Add skip connection.
        this_name = 'skip_level{0:d}'.format(i - 1)
        merged_layer_by_level[i - 1] = keras.layers.Concatenate(
            axis=-1, name=this_name
        )(
            [conv_layer_by_level[i - 1], upconv_layer_by_level[i - 1]]
        )

    # Create the "simple" output layer, excluding the two constrained target
    # fields: gust excess and dewpoint depression.
    predict_gust_excess = (
        urma_utils.WIND_GUST_10METRE_NAME in target_field_names
    )
    predict_dewpoint_depression = (
        urma_utils.DEWPOINT_2METRE_NAME in target_field_names
    )
    this_offset = int(predict_gust_excess) + int(predict_dewpoint_depression)

    simple_output_layer_object = architecture_utils.get_2d_conv_layer(
        num_kernel_rows=1, num_kernel_columns=1,
        num_rows_per_stride=1, num_columns_per_stride=1,
        num_filters=(num_output_channels - this_offset) * ensemble_size,
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=l2_function,
        layer_name='last_conv_simple'
    )(skip_layer_by_level[0])

    # If necessary, add one channel (dewpoint depression) to the output layer.
    if predict_dewpoint_depression:
        dd_output_layer_object = architecture_utils.get_2d_conv_layer(
            num_kernel_rows=1, num_kernel_columns=1,
            num_rows_per_stride=1, num_columns_per_stride=1,
            num_filters=ensemble_size,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=l2_function,
            layer_name='last_conv_dd'
        )(skip_layer_by_level[0])

        # Apply strict ReLU, because dewpoint depression must be non-negative.
        dd_output_layer_object = architecture_utils.get_activation_layer(
            activation_function_string=architecture_utils.RELU_FUNCTION_STRING,
            alpha_for_relu=0.,
            alpha_for_elu=0.,
            layer_name='last_conv_dd_activation'
        )(dd_output_layer_object)
    else:
        dd_output_layer_object = None

    # If necessary, add one channel (gust excess) to the output layer.
    if predict_gust_excess:
        gf_output_layer_object = architecture_utils.get_2d_conv_layer(
            num_kernel_rows=1, num_kernel_columns=1,
            num_rows_per_stride=1, num_columns_per_stride=1,
            num_filters=ensemble_size,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=l2_function,
            layer_name='last_conv_gex'
        )(skip_layer_by_level[0])

        # Apply strict ReLU, because gust excess must be non-negative.
        gf_output_layer_object = architecture_utils.get_activation_layer(
            activation_function_string=architecture_utils.RELU_FUNCTION_STRING,
            alpha_for_relu=0.,
            alpha_for_elu=0.,
            layer_name='last_conv_gex_activation'
        )(gf_output_layer_object)
    else:
        gf_output_layer_object = None

    # If necessary, concatenate all the output layers into one layer.
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

    # Current dimensions of output layer are data_example x grid_row x
    # grid_column x (num_output_channels * ensemble_size).  In other words,
    # output channels (target fields) and ensemble members are mingled together
    # along the final axis.  If this is an ensemble model, demingle the target
    # fields and ensemble members, so that they each have their own axis.
    if ensemble_size > 1:
        new_dims = (
            input_dimensions[0], input_dimensions[1],
            num_output_channels, ensemble_size
        )
        output_layer_object = keras.layers.Reshape(
            target_shape=new_dims, name='reshape_predictions'
        )(output_layer_object)

    # Finalize the model.
    model_object = keras.models.Model(
        inputs=input_layer_object, outputs=output_layer_object
    )

    # Compile the model.
    model_object.compile(
        loss=loss_function, optimizer=optimizer_function,
        metrics=metric_function_list
    )

    # Print the model summary and return the model.
    model_object.summary()
    return model_object
