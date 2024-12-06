"""Methods for building a Chiu-net++.

The Chiu-net++ is a hybrid between the Chiu net
(https://doi.org/10.1109/LRA.2020.2992184) and U-net++.
"""

import os
import sys
import numpy
import keras

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import architecture_utils
import chiu_net_architecture as chiu_net_arch

INPUT_DIMENSIONS_CONST_KEY = chiu_net_arch.INPUT_DIMENSIONS_CONST_KEY
INPUT_DIMENSIONS_2PT5KM_RES_KEY = chiu_net_arch.INPUT_DIMENSIONS_2PT5KM_RES_KEY
INPUT_DIMENSIONS_10KM_RES_KEY = chiu_net_arch.INPUT_DIMENSIONS_10KM_RES_KEY
INPUT_DIMENSIONS_20KM_RES_KEY = chiu_net_arch.INPUT_DIMENSIONS_20KM_RES_KEY
INPUT_DIMENSIONS_40KM_RES_KEY = chiu_net_arch.INPUT_DIMENSIONS_40KM_RES_KEY
INPUT_DIMENSIONS_2PT5KM_RCTBIAS_KEY = (
    chiu_net_arch.INPUT_DIMENSIONS_2PT5KM_RCTBIAS_KEY
)
INPUT_DIMENSIONS_10KM_RCTBIAS_KEY = (
    chiu_net_arch.INPUT_DIMENSIONS_10KM_RCTBIAS_KEY
)
INPUT_DIMENSIONS_20KM_RCTBIAS_KEY = (
    chiu_net_arch.INPUT_DIMENSIONS_20KM_RCTBIAS_KEY
)
INPUT_DIMENSIONS_40KM_RCTBIAS_KEY = (
    chiu_net_arch.INPUT_DIMENSIONS_40KM_RCTBIAS_KEY
)
PREDN_BASELINE_DIMENSIONS_KEY = chiu_net_arch.PREDN_BASELINE_DIMENSIONS_KEY
INPUT_DIMENSIONS_LAGGED_TARGETS_KEY = (
    chiu_net_arch.INPUT_DIMENSIONS_LAGGED_TARGETS_KEY
)
USE_RESIDUAL_BLOCKS_KEY = chiu_net_arch.USE_RESIDUAL_BLOCKS_KEY

NWP_ENCODER_NUM_CHANNELS_KEY = chiu_net_arch.NWP_ENCODER_NUM_CHANNELS_KEY
NWP_POOLING_SIZE_KEY = chiu_net_arch.NWP_POOLING_SIZE_KEY
NWP_ENCODER_NUM_CONV_LAYERS_KEY = chiu_net_arch.NWP_ENCODER_NUM_CONV_LAYERS_KEY
NWP_ENCODER_DROPOUT_RATES_KEY = chiu_net_arch.NWP_ENCODER_DROPOUT_RATES_KEY
NWP_FC_MODULE_NUM_CONV_LAYERS_KEY = (
    chiu_net_arch.NWP_FC_MODULE_NUM_CONV_LAYERS_KEY
)
NWP_FC_MODULE_DROPOUT_RATES_KEY = chiu_net_arch.NWP_FC_MODULE_DROPOUT_RATES_KEY
NWP_FC_MODULE_USE_3D_CONV = chiu_net_arch.NWP_FC_MODULE_USE_3D_CONV

LAGTGT_ENCODER_NUM_CHANNELS_KEY = chiu_net_arch.LAGTGT_ENCODER_NUM_CHANNELS_KEY
LAGTGT_POOLING_SIZE_KEY = chiu_net_arch.LAGTGT_POOLING_SIZE_KEY
LAGTGT_ENCODER_NUM_CONV_LAYERS_KEY = (
    chiu_net_arch.LAGTGT_ENCODER_NUM_CONV_LAYERS_KEY
)
LAGTGT_ENCODER_DROPOUT_RATES_KEY = (
    chiu_net_arch.LAGTGT_ENCODER_DROPOUT_RATES_KEY
)
LAGTGT_FC_MODULE_NUM_CONV_LAYERS_KEY = (
    chiu_net_arch.LAGTGT_FC_MODULE_NUM_CONV_LAYERS_KEY
)
LAGTGT_FC_MODULE_DROPOUT_RATES_KEY = (
    chiu_net_arch.LAGTGT_FC_MODULE_DROPOUT_RATES_KEY
)
LAGTGT_FC_MODULE_USE_3D_CONV = chiu_net_arch.LAGTGT_FC_MODULE_USE_3D_CONV

RCTBIAS_ENCODER_NUM_CHANNELS_KEY = (
    chiu_net_arch.RCTBIAS_ENCODER_NUM_CHANNELS_KEY
)
RCTBIAS_POOLING_SIZE_KEY = chiu_net_arch.RCTBIAS_POOLING_SIZE_KEY
RCTBIAS_ENCODER_NUM_CONV_LAYERS_KEY = (
    chiu_net_arch.RCTBIAS_ENCODER_NUM_CONV_LAYERS_KEY
)
RCTBIAS_ENCODER_DROPOUT_RATES_KEY = (
    chiu_net_arch.RCTBIAS_ENCODER_DROPOUT_RATES_KEY
)
RCTBIAS_FC_MODULE_NUM_CONV_LAYERS_KEY = (
    chiu_net_arch.RCTBIAS_FC_MODULE_NUM_CONV_LAYERS_KEY
)
RCTBIAS_FC_MODULE_DROPOUT_RATES_KEY = (
    chiu_net_arch.RCTBIAS_FC_MODULE_DROPOUT_RATES_KEY
)
RCTBIAS_FC_MODULE_USE_3D_CONV = chiu_net_arch.RCTBIAS_FC_MODULE_USE_3D_CONV

DECODER_NUM_CHANNELS_KEY = chiu_net_arch.DECODER_NUM_CHANNELS_KEY
DECODER_NUM_CONV_LAYERS_KEY = chiu_net_arch.DECODER_NUM_CONV_LAYERS_KEY
UPSAMPLING_DROPOUT_RATES_KEY = chiu_net_arch.UPSAMPLING_DROPOUT_RATES_KEY
SKIP_DROPOUT_RATES_KEY = chiu_net_arch.SKIP_DROPOUT_RATES_KEY

INCLUDE_PENULTIMATE_KEY = chiu_net_arch.INCLUDE_PENULTIMATE_KEY
PENULTIMATE_DROPOUT_RATE_KEY = chiu_net_arch.PENULTIMATE_DROPOUT_RATE_KEY
INNER_ACTIV_FUNCTION_KEY = chiu_net_arch.INNER_ACTIV_FUNCTION_KEY
INNER_ACTIV_FUNCTION_ALPHA_KEY = chiu_net_arch.INNER_ACTIV_FUNCTION_ALPHA_KEY
OUTPUT_ACTIV_FUNCTION_KEY = chiu_net_arch.OUTPUT_ACTIV_FUNCTION_KEY
OUTPUT_ACTIV_FUNCTION_ALPHA_KEY = chiu_net_arch.OUTPUT_ACTIV_FUNCTION_ALPHA_KEY
L1_WEIGHT_KEY = chiu_net_arch.L1_WEIGHT_KEY
L2_WEIGHT_KEY = chiu_net_arch.L2_WEIGHT_KEY
USE_BATCH_NORM_KEY = chiu_net_arch.USE_BATCH_NORM_KEY
BATCH_NORM_MOMENTUM_KEY = chiu_net_arch.BATCH_NORM_MOMENTUM_KEY
BATCH_NORM_SYNCH_FLAG_KEY = chiu_net_arch.BATCH_NORM_SYNCH_FLAG_KEY
ENSEMBLE_SIZE_KEY = chiu_net_arch.ENSEMBLE_SIZE_KEY

NUM_OUTPUT_CHANNELS_KEY = chiu_net_arch.NUM_OUTPUT_CHANNELS_KEY
PREDICT_GUST_EXCESS_KEY = chiu_net_arch.PREDICT_GUST_EXCESS_KEY
PREDICT_DEWPOINT_DEPRESSION_KEY = chiu_net_arch.PREDICT_DEWPOINT_DEPRESSION_KEY
LOSS_FUNCTION_KEY = chiu_net_arch.LOSS_FUNCTION_KEY
OPTIMIZER_FUNCTION_KEY = chiu_net_arch.OPTIMIZER_FUNCTION_KEY
METRIC_FUNCTIONS_KEY = chiu_net_arch.METRIC_FUNCTIONS_KEY


def _get_2d_conv_block(
        input_layer_object, do_residual, num_conv_layers, filter_size_px,
        num_filters, do_time_distributed_conv, regularizer_object,
        activation_function_name, activation_function_alpha,
        dropout_rates, use_batch_norm, batch_norm_momentum,
        batch_norm_synch_flag, basic_layer_name):
    """Creates convolutional block for data with 2 spatial dimensions.

    L = number of conv layers

    :param input_layer_object: Input layer to block.
    :param do_residual: Boolean flag.  If True, this will be a residual block.
    :param num_conv_layers: Number of conv layers in block.
    :param filter_size_px: Filter size for conv layers.  The same filter size
        will be used in both dimensions, and the same filter size will be used
        for every conv layer.
    :param num_filters: Number of filters -- same for every conv layer.
    :param do_time_distributed_conv: Boolean flag.  If True (False), will do
        time-distributed (basic) convolution.
    :param regularizer_object: Regularizer for conv layers (instance of
        `keras.regularizers.l1_l2` or similar).
    :param activation_function_name: Name of activation function -- same for
        every conv layer.  Must be accepted by
        `architecture_utils.check_activation_function`.
    :param activation_function_alpha: Alpha (slope parameter) for activation
        function -- same for every conv layer.  Applies only to ReLU and eLU.
    :param dropout_rates: Dropout rates for conv layers.  This can be a scalar
        (applied to every conv layer) or length-L numpy array.
    :param use_batch_norm: Boolean flag.  If True, will use batch normalization.
    :param batch_norm_momentum: Momentum for batch-normalization layers.  For
        more details, see documentation for `keras.layers.BatchNormalization`.
    :param batch_norm_synch_flag: Boolean flag for batch-normalization layers.
        For more details, see documentation for
        `keras.layers.BatchNormalization`.
    :param basic_layer_name: Basic layer name.  Each layer name will be made
        unique by adding a suffix.
    :return: output_layer_object: Output layer from block.
    """

    # Process input args.
    if do_residual:
        num_conv_layers = max([num_conv_layers, 2])

    try:
        _ = len(dropout_rates)
    except:
        dropout_rates = numpy.full(num_conv_layers, dropout_rates)

    if len(dropout_rates) < num_conv_layers:
        dropout_rates = numpy.concatenate([
            dropout_rates, dropout_rates[[-1]]
        ])

    assert len(dropout_rates) == num_conv_layers

    # Do actual stuff.
    current_layer_object = None

    for i in range(num_conv_layers):
        if i == 0:
            this_input_layer_object = input_layer_object
        else:
            this_input_layer_object = current_layer_object

        this_name = '{0:s}_conv{1:d}'.format(basic_layer_name, i)
        current_layer_object = architecture_utils.get_2d_conv_layer(
            num_kernel_rows=filter_size_px,
            num_kernel_columns=filter_size_px,
            num_rows_per_stride=1,
            num_columns_per_stride=1,
            num_filters=num_filters,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )

        if do_time_distributed_conv:
            current_layer_object = keras.layers.TimeDistributed(
                current_layer_object, name=this_name
            )(this_input_layer_object)
        else:
            current_layer_object = current_layer_object(this_input_layer_object)

        if i == num_conv_layers - 1 and do_residual:
            if input_layer_object.shape[-1] == num_filters:
                new_layer_object = input_layer_object
            else:
                this_name = '{0:s}_preresidual_conv'.format(basic_layer_name)
                new_layer_object = architecture_utils.get_2d_conv_layer(
                    num_kernel_rows=1,
                    num_kernel_columns=1,
                    num_rows_per_stride=1,
                    num_columns_per_stride=1,
                    num_filters=num_filters,
                    padding_type_string=architecture_utils.YES_PADDING_STRING,
                    weight_regularizer=regularizer_object,
                    layer_name=this_name
                )

                if do_time_distributed_conv:
                    new_layer_object = keras.layers.TimeDistributed(
                        new_layer_object, name=this_name
                    )(input_layer_object)
                else:
                    new_layer_object = new_layer_object(input_layer_object)

            this_name = '{0:s}_residual'.format(basic_layer_name)
            current_layer_object = keras.layers.Add(name=this_name)([
                current_layer_object, new_layer_object
            ])

        if activation_function_name is not None:
            this_name = '{0:s}_activ{1:d}'.format(basic_layer_name, i)
            current_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=activation_function_name,
                alpha_for_relu=activation_function_alpha,
                alpha_for_elu=activation_function_alpha,
                layer_name=this_name
            )(current_layer_object)

        if dropout_rates[i] > 0:
            this_name = '{0:s}_dropout{1:d}'.format(basic_layer_name, i)
            current_layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=dropout_rates[i], layer_name=this_name
            )(current_layer_object)

        if use_batch_norm:
            this_name = '{0:s}_bn{1:d}'.format(basic_layer_name, i)
            current_layer_object = architecture_utils.get_batch_norm_layer(
                momentum=batch_norm_momentum,
                synchronized=batch_norm_synch_flag,
                layer_name=this_name
            )(current_layer_object)

    return current_layer_object


def _get_3d_conv_block(
        input_layer_object, do_residual, num_conv_layers, filter_size_px,
        regularizer_object, activation_function_name, activation_function_alpha,
        dropout_rates, use_batch_norm, batch_norm_momentum,
        batch_norm_synch_flag, basic_layer_name):
    """Creates convolutional block for data with 3 spatial dimensions.

    :param input_layer_object: Input layer to block (with 3 spatial dims).
    :param do_residual: See documentation for `_get_2d_conv_block`.
    :param num_conv_layers: Same.
    :param filter_size_px: Same.
    :param regularizer_object: Same.
    :param activation_function_name: Same.
    :param activation_function_alpha: Same.
    :param dropout_rates: Same.
    :param use_batch_norm: Same.
    :param batch_norm_momentum: Same.
    :param batch_norm_synch_flag: Same.
    :param basic_layer_name: Same.
    :return: output_layer_object: Output layer from block (with 2 spatial dims).
    """

    # Process input args.
    if do_residual:
        num_conv_layers = max([num_conv_layers, 2])

    try:
        _ = len(dropout_rates)
    except:
        dropout_rates = numpy.full(num_conv_layers, dropout_rates)

    if len(dropout_rates) < num_conv_layers:
        dropout_rates = numpy.concatenate([
            dropout_rates, dropout_rates[[-1]]
        ])

    assert len(dropout_rates) == num_conv_layers

    # Do actual stuff.
    current_layer_object = None
    num_time_steps = input_layer_object.shape[-2]
    num_filters = input_layer_object.shape[-1]

    for i in range(num_conv_layers):
        this_name = '{0:s}_conv{1:d}'.format(basic_layer_name, i)

        if i == 0:
            current_layer_object = architecture_utils.get_3d_conv_layer(
                num_kernel_rows=filter_size_px,
                num_kernel_columns=filter_size_px,
                num_kernel_heights=num_time_steps,
                num_rows_per_stride=1,
                num_columns_per_stride=1,
                num_heights_per_stride=1,
                num_filters=num_filters,
                padding_type_string=architecture_utils.NO_PADDING_STRING,
                weight_regularizer=regularizer_object,
                layer_name=this_name
            )(input_layer_object)

            new_dims = (
                current_layer_object.shape[1:3] +
                (current_layer_object.shape[-1],)
            )

            this_name = '{0:s}_remove-time-dim'.format(basic_layer_name)
            current_layer_object = keras.layers.Reshape(
                target_shape=new_dims, name=this_name
            )(current_layer_object)
        else:
            current_layer_object = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=filter_size_px,
                num_kernel_columns=filter_size_px,
                num_rows_per_stride=1,
                num_columns_per_stride=1,
                num_filters=num_filters,
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object
            )(current_layer_object)

        if i == num_conv_layers - 1 and do_residual:
            this_name = '{0:s}_preresidual_avg'.format(basic_layer_name)
            this_layer_object = architecture_utils.get_3d_pooling_layer(
                num_rows_in_window=1,
                num_columns_in_window=1,
                num_heights_in_window=num_time_steps,
                num_rows_per_stride=1,
                num_columns_per_stride=1,
                num_heights_per_stride=num_time_steps,
                pooling_type_string=architecture_utils.MEAN_POOLING_STRING,
                layer_name=this_name
            )(input_layer_object)

            new_dims = (
                this_layer_object.shape[1:3] +
                (this_layer_object.shape[-1],)
            )

            this_name = '{0:s}_preresidual_squeeze'.format(basic_layer_name)
            this_layer_object = keras.layers.Reshape(
                target_shape=new_dims, name=this_name
            )(this_layer_object)

            this_name = '{0:s}_residual'.format(basic_layer_name)
            current_layer_object = keras.layers.Add(name=this_name)([
                current_layer_object, this_layer_object
            ])

        this_name = '{0:s}_activ{1:d}'.format(basic_layer_name, i)
        current_layer_object = architecture_utils.get_activation_layer(
            activation_function_string=activation_function_name,
            alpha_for_relu=activation_function_alpha,
            alpha_for_elu=activation_function_alpha,
            layer_name=this_name
        )(current_layer_object)

        if dropout_rates[i] > 0:
            this_name = '{0:s}_dropout{1:d}'.format(basic_layer_name, i)
            current_layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=dropout_rates[i], layer_name=this_name
            )(current_layer_object)

        if use_batch_norm:
            this_name = '{0:s}_bn{1:d}'.format(basic_layer_name, i)
            current_layer_object = architecture_utils.get_batch_norm_layer(
                momentum=batch_norm_momentum,
                synchronized=batch_norm_synch_flag,
                layer_name=this_name
            )(current_layer_object)

    return current_layer_object


def get_channel_counts_for_skip_cnxn(input_layer_objects, num_output_channels):
    """Determines number of channels for each input layer to skip connection.

    A = number of input layers.

    :param input_layer_objects: length-A list of input layers (instances of
        subclass of `keras.layers`).
    :param num_output_channels: Number of desired output channels (after
        concatenation).
    :return: desired_channel_counts: length-A numpy array with number of
        desired channels for each input layer.
    """

    error_checking.assert_is_list(input_layer_objects)
    error_checking.assert_is_integer(num_output_channels)
    error_checking.assert_is_geq(num_output_channels, 1)

    current_channel_counts = numpy.array(
        [l.shape[-1] for l in input_layer_objects], dtype=float
    )

    num_input_layers = len(input_layer_objects)
    desired_channel_counts = numpy.full(num_input_layers, -1, dtype=int)

    half_num_output_channels = int(numpy.round(0.5 * num_output_channels))
    desired_channel_counts[-1] = half_num_output_channels

    remaining_num_output_channels = (
        num_output_channels - half_num_output_channels
    )
    this_ratio = (
        float(remaining_num_output_channels) /
        numpy.sum(current_channel_counts[:-1])
    )
    desired_channel_counts[:-1] = numpy.round(
        current_channel_counts[:-1] * this_ratio
    ).astype(int)

    while numpy.sum(desired_channel_counts) > num_output_channels:
        desired_channel_counts[numpy.argmax(desired_channel_counts[:-1])] -= 1
    while numpy.sum(desired_channel_counts) < num_output_channels:
        desired_channel_counts[numpy.argmin(desired_channel_counts[:-1])] += 1

    assert numpy.sum(desired_channel_counts) == num_output_channels
    desired_channel_counts = numpy.maximum(desired_channel_counts, 1)

    return desired_channel_counts


def create_skip_connection(input_layer_objects, num_output_channels,
                           current_level_num, regularizer_object):
    """Creates skip connection.

    :param input_layer_objects: 1-D list of input layers (instances of subclass
        of `keras.layers`).
    :param num_output_channels: Desired number of output channels.
    :param current_level_num: Current level in Chiu-net++ architecture.  This
        should be a zero-based integer index.
    :param regularizer_object: Regularizer for conv layers (instance of
        `keras.regularizers.l1_l2` or similar).
    :return: concat_layer_object: Instance of `keras.layers.Concatenate`.
    """

    error_checking.assert_is_integer(current_level_num)
    error_checking.assert_is_geq(current_level_num, 0)

    desired_input_channel_counts = get_channel_counts_for_skip_cnxn(
        input_layer_objects=input_layer_objects,
        num_output_channels=num_output_channels
    )
    current_width = len(input_layer_objects) - 1

    for j in range(current_width):
        this_name = 'block{0:d}-{1:d}_preskipconv{2:d}'.format(
            current_level_num, current_width, j
        )

        input_layer_objects[j] = architecture_utils.get_2d_conv_layer(
            num_kernel_rows=1,
            num_kernel_columns=1,
            num_rows_per_stride=1,
            num_columns_per_stride=1,
            num_filters=desired_input_channel_counts[j],
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )(input_layer_objects[j])

    this_name = 'block{0:d}-{1:d}_skip'.format(current_level_num, current_width)
    return keras.layers.Concatenate(axis=-1, name=this_name)(
        input_layer_objects
    )


def pad_layer(source_layer_object, target_layer_object, padding_layer_name,
              num_spatiotemporal_dims):
    """Pads layer spatially.

    :param source_layer_object: Source layer.
    :param target_layer_object: Target layer.  The source layer will be padded,
        if necessary, to have the same dimensions as the target layer.
    :param padding_layer_name: Name of padding layer.
    :param num_spatiotemporal_dims: Number of dimensions.
    :return: source_layer_object: Same as input, except maybe with different
        spatial dimensions.
    """

    error_checking.assert_is_string(padding_layer_name)
    error_checking.assert_is_integer(num_spatiotemporal_dims)
    error_checking.assert_is_geq(num_spatiotemporal_dims, 2)
    error_checking.assert_is_leq(num_spatiotemporal_dims, 3)

    num_source_rows = source_layer_object.shape[1]
    num_target_rows = target_layer_object.shape[1]
    num_padding_rows = num_target_rows - num_source_rows

    num_source_columns = source_layer_object.shape[2]
    num_target_columns = target_layer_object.shape[2]
    num_padding_columns = num_target_columns - num_source_columns

    if num_spatiotemporal_dims == 2:
        if num_padding_rows + num_padding_columns > 0:
            padding_arg = ((0, num_padding_rows), (0, num_padding_columns))

            return keras.layers.ZeroPadding2D(
                padding=padding_arg, name=padding_layer_name
            )(source_layer_object)

        return source_layer_object

    num_source_heights = source_layer_object.shape[3]
    num_target_heights = target_layer_object.shape[3]
    num_padding_heights = num_target_heights - num_source_heights

    if num_padding_rows + num_padding_columns + num_padding_heights > 0:
        padding_arg = (
            (0, num_padding_rows),
            (0, num_padding_columns),
            (0, num_padding_heights)
        )

        return keras.layers.ZeroPadding3D(
            padding=padding_arg, name=padding_layer_name
        )(source_layer_object)

    return source_layer_object


def crop_layer(source_layer_object, target_layer_object, cropping_layer_name,
               num_spatiotemporal_dims):
    """Crops layer spatially.

    :param source_layer_object: Source layer.
    :param target_layer_object: Target layer.  The source layer will be cropped,
        if necessary, to have the same dimensions as the target layer.
    :param cropping_layer_name: Name of cropping layer.
    :param num_spatiotemporal_dims: Number of dimensions.
    :return: source_layer_object: Same as input, except maybe with different
        spatial dimensions.
    """

    error_checking.assert_is_string(cropping_layer_name)
    error_checking.assert_is_integer(num_spatiotemporal_dims)
    error_checking.assert_is_geq(num_spatiotemporal_dims, 2)
    error_checking.assert_is_leq(num_spatiotemporal_dims, 3)

    num_source_rows = source_layer_object.shape[1]
    num_target_rows = target_layer_object.shape[1]
    num_cropping_rows = num_source_rows - num_target_rows

    num_source_columns = source_layer_object.shape[2]
    num_target_columns = target_layer_object.shape[2]
    num_cropping_columns = num_source_columns - num_target_columns

    if num_spatiotemporal_dims == 2:
        if num_cropping_rows + num_cropping_columns > 0:
            cropping_arg = ((0, num_cropping_rows), (0, num_cropping_columns))

            return keras.layers.Cropping2D(
                cropping=cropping_arg, name=cropping_layer_name
            )(source_layer_object)

        return source_layer_object

    num_source_heights = source_layer_object.shape[3]
    num_target_heights = target_layer_object.shape[3]
    num_cropping_heights = num_source_heights - num_target_heights

    if num_cropping_rows + num_cropping_columns + num_cropping_heights > 0:
        cropping_arg = (
            (0, num_cropping_rows),
            (0, num_cropping_columns),
            (0, num_cropping_heights)
        )

        return keras.layers.Cropping3D(
            cropping=cropping_arg, name=cropping_layer_name
        )(source_layer_object)

    return source_layer_object


def create_model(option_dict):
    """Creates CNN.

    :param option_dict: See documentation for `check_input_args`.
    :return: model_object: Untrained (but compiled) instance of
        `keras.models.Model`.
    """

    option_dict = chiu_net_arch.check_input_args(option_dict)
    optd = option_dict

    input_dimensions_const = optd[INPUT_DIMENSIONS_CONST_KEY]
    input_dimensions_2pt5km_res = optd[INPUT_DIMENSIONS_2PT5KM_RES_KEY]
    input_dimensions_lagged_targets = optd[INPUT_DIMENSIONS_LAGGED_TARGETS_KEY]
    input_dimensions_10km_res = optd[INPUT_DIMENSIONS_10KM_RES_KEY]
    input_dimensions_20km_res = optd[INPUT_DIMENSIONS_20KM_RES_KEY]
    input_dimensions_40km_res = optd[INPUT_DIMENSIONS_40KM_RES_KEY]
    input_dimensions_2pt5km_rctbias = (
        option_dict[INPUT_DIMENSIONS_2PT5KM_RCTBIAS_KEY]
    )
    input_dimensions_10km_rctbias = (
        option_dict[INPUT_DIMENSIONS_10KM_RCTBIAS_KEY]
    )
    input_dimensions_20km_rctbias = (
        option_dict[INPUT_DIMENSIONS_20KM_RCTBIAS_KEY]
    )
    input_dimensions_40km_rctbias = (
        option_dict[INPUT_DIMENSIONS_40KM_RCTBIAS_KEY]
    )
    input_dimensions_predn_baseline = optd[PREDN_BASELINE_DIMENSIONS_KEY]
    use_residual_blocks = optd[USE_RESIDUAL_BLOCKS_KEY]

    nwp_encoder_num_channels_by_level = optd[NWP_ENCODER_NUM_CHANNELS_KEY]
    nwp_pooling_size_by_level_px = optd[NWP_POOLING_SIZE_KEY]
    nwp_encoder_num_conv_layers_by_level = optd[NWP_ENCODER_NUM_CONV_LAYERS_KEY]
    nwp_encoder_dropout_rate_by_level = optd[NWP_ENCODER_DROPOUT_RATES_KEY]
    nwp_forecast_module_num_conv_layers = optd[
        NWP_FC_MODULE_NUM_CONV_LAYERS_KEY
    ]
    nwp_forecast_module_dropout_rates = optd[NWP_FC_MODULE_DROPOUT_RATES_KEY]
    nwp_forecast_module_use_3d_conv = optd[NWP_FC_MODULE_USE_3D_CONV]

    lagtgt_encoder_num_channels_by_level = optd[LAGTGT_ENCODER_NUM_CHANNELS_KEY]
    lagtgt_pooling_size_by_level_px = optd[LAGTGT_POOLING_SIZE_KEY]
    lagtgt_encoder_num_conv_layers_by_level = optd[
        LAGTGT_ENCODER_NUM_CONV_LAYERS_KEY
    ]
    lagtgt_encoder_dropout_rate_by_level = optd[
        LAGTGT_ENCODER_DROPOUT_RATES_KEY
    ]
    lagtgt_forecast_module_num_conv_layers = optd[
        LAGTGT_FC_MODULE_NUM_CONV_LAYERS_KEY
    ]
    lagtgt_forecast_module_dropout_rates = optd[
        LAGTGT_FC_MODULE_DROPOUT_RATES_KEY
    ]
    lagtgt_forecast_module_use_3d_conv = optd[LAGTGT_FC_MODULE_USE_3D_CONV]

    rctbias_encoder_num_channels_by_level = optd[
        RCTBIAS_ENCODER_NUM_CHANNELS_KEY
    ]
    rctbias_pooling_size_by_level_px = optd[RCTBIAS_POOLING_SIZE_KEY]
    rctbias_encoder_num_conv_layers_by_level = optd[
        RCTBIAS_ENCODER_NUM_CONV_LAYERS_KEY
    ]
    rctbias_encoder_dropout_rate_by_level = optd[
        RCTBIAS_ENCODER_DROPOUT_RATES_KEY
    ]
    rctbias_forecast_module_num_conv_layers = optd[
        RCTBIAS_FC_MODULE_NUM_CONV_LAYERS_KEY
    ]
    rctbias_forecast_module_dropout_rates = optd[
        RCTBIAS_FC_MODULE_DROPOUT_RATES_KEY
    ]
    rctbias_forecast_module_use_3d_conv = optd[RCTBIAS_FC_MODULE_USE_3D_CONV]

    decoder_num_channels_by_level = optd[DECODER_NUM_CHANNELS_KEY]
    num_decoder_conv_layers_by_level = optd[DECODER_NUM_CONV_LAYERS_KEY]
    upsampling_dropout_rate_by_level = optd[UPSAMPLING_DROPOUT_RATES_KEY]
    skip_dropout_rate_by_level = optd[SKIP_DROPOUT_RATES_KEY]

    include_penultimate_conv = optd[INCLUDE_PENULTIMATE_KEY]
    penultimate_conv_dropout_rate = optd[PENULTIMATE_DROPOUT_RATE_KEY]
    inner_activ_function_name = optd[INNER_ACTIV_FUNCTION_KEY]
    inner_activ_function_alpha = optd[INNER_ACTIV_FUNCTION_ALPHA_KEY]
    output_activ_function_name = optd[OUTPUT_ACTIV_FUNCTION_KEY]
    output_activ_function_alpha = optd[OUTPUT_ACTIV_FUNCTION_ALPHA_KEY]
    l1_weight = optd[L1_WEIGHT_KEY]
    l2_weight = optd[L2_WEIGHT_KEY]
    use_batch_normalization = optd[USE_BATCH_NORM_KEY]
    batch_norm_momentum = optd[BATCH_NORM_MOMENTUM_KEY]
    batch_norm_synch_flag = optd[BATCH_NORM_SYNCH_FLAG_KEY]
    ensemble_size = optd[ENSEMBLE_SIZE_KEY]
    num_output_channels = optd[NUM_OUTPUT_CHANNELS_KEY]
    predict_gust_excess = optd[PREDICT_GUST_EXCESS_KEY]
    predict_dewpoint_depression = optd[PREDICT_DEWPOINT_DEPRESSION_KEY]

    loss_function = optd[LOSS_FUNCTION_KEY]
    optimizer_function = optd[OPTIMIZER_FUNCTION_KEY]
    metric_function_list = optd[METRIC_FUNCTIONS_KEY]

    use_recent_biases = input_dimensions_2pt5km_rctbias is not None
    num_lead_times = input_dimensions_2pt5km_res[2]

    input_layer_object_2pt5km_res = keras.layers.Input(
        shape=tuple(input_dimensions_2pt5km_res.tolist()),
        name='2pt5km_inputs'
    )
    layer_object_2pt5km_res = keras.layers.Permute(
        dims=(3, 1, 2, 4), name='2pt5km_put-time-first'
    )(input_layer_object_2pt5km_res)

    if input_dimensions_lagged_targets is None:
        input_layer_object_lagged_targets = None
        layer_object_lagged_targets = None
        num_lag_times = 0
    else:
        input_layer_object_lagged_targets = keras.layers.Input(
            shape=tuple(input_dimensions_lagged_targets.tolist()),
            name='lagtgt_inputs'
        )
        layer_object_lagged_targets = keras.layers.Permute(
            dims=(3, 1, 2, 4), name='lagtgt_put-time-first'
        )(input_layer_object_lagged_targets)

        num_lag_times = input_dimensions_lagged_targets[2]

    if input_dimensions_predn_baseline is None:
        input_layer_object_predn_baseline = None
    else:
        input_layer_object_predn_baseline = keras.layers.Input(
            shape=tuple(input_dimensions_predn_baseline.tolist()),
            name='resid_baseline_inputs'
        )

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
            axis=-1, name='const_concat-with-2pt5km'
        )(
            [layer_object_2pt5km_res, this_layer_object]
        )

        if num_lag_times > 0:
            this_layer_object = keras.layers.Concatenate(
                axis=-4, name='const_add-lag-times'
            )(
                num_lag_times * [layer_object_const]
            )

            layer_object_lagged_targets = keras.layers.Concatenate(
                axis=-1, name='const_concat-with-lagtgt'
            )(
                [layer_object_lagged_targets, this_layer_object]
            )

    if input_dimensions_2pt5km_rctbias is None:
        input_layer_object_2pt5km_rctbias = None
        layer_object_2pt5km_rctbias = None
    else:
        input_layer_object_2pt5km_rctbias = keras.layers.Input(
            shape=tuple(input_dimensions_2pt5km_rctbias.tolist()),
            name='2pt5km_rctbias'
        )
        layer_object_2pt5km_rctbias = keras.layers.Permute(
            dims=(3, 1, 2, 4), name='2pt5km_rctbias_put-time-first'
        )(input_layer_object_2pt5km_rctbias)

    if input_dimensions_10km_res is None:
        input_layer_object_10km_res = None
        layer_object_10km_res = None
    else:
        input_layer_object_10km_res = keras.layers.Input(
            shape=tuple(input_dimensions_10km_res.tolist()),
            name='10km_inputs'
        )
        layer_object_10km_res = keras.layers.Permute(
            dims=(3, 1, 2, 4), name='10km_put-time-first'
        )(input_layer_object_10km_res)

    if input_dimensions_10km_rctbias is None:
        input_layer_object_10km_rctbias = None
        layer_object_10km_rctbias = None
    else:
        input_layer_object_10km_rctbias = keras.layers.Input(
            shape=tuple(input_dimensions_10km_rctbias.tolist()),
            name='10km_rctbias'
        )
        layer_object_10km_rctbias = keras.layers.Permute(
            dims=(3, 1, 2, 4), name='10km_rctbias_put-time-first'
        )(input_layer_object_10km_rctbias)

    if input_dimensions_20km_res is None:
        input_layer_object_20km_res = None
        layer_object_20km_res = None
    else:
        input_layer_object_20km_res = keras.layers.Input(
            shape=tuple(input_dimensions_20km_res.tolist()),
            name='20km_inputs'
        )
        layer_object_20km_res = keras.layers.Permute(
            dims=(3, 1, 2, 4), name='20km_put-time-first'
        )(input_layer_object_20km_res)

    if input_dimensions_20km_rctbias is None:
        input_layer_object_20km_rctbias = None
        layer_object_20km_rctbias = None
    else:
        input_layer_object_20km_rctbias = keras.layers.Input(
            shape=tuple(input_dimensions_20km_rctbias.tolist()),
            name='20km_rctbias'
        )
        layer_object_20km_rctbias = keras.layers.Permute(
            dims=(3, 1, 2, 4), name='20km_rctbias_put-time-first'
        )(input_layer_object_20km_rctbias)

    if input_dimensions_40km_res is None:
        input_layer_object_40km_res = None
        layer_object_40km_res = None
    else:
        input_layer_object_40km_res = keras.layers.Input(
            shape=tuple(input_dimensions_40km_res.tolist()),
            name='40km_inputs'
        )
        layer_object_40km_res = keras.layers.Permute(
            dims=(3, 1, 2, 4), name='40km_put-time-first'
        )(input_layer_object_40km_res)

    if input_dimensions_40km_rctbias is None:
        input_layer_object_40km_rctbias = None
        layer_object_40km_rctbias = None
    else:
        input_layer_object_40km_rctbias = keras.layers.Input(
            shape=tuple(input_dimensions_40km_rctbias.tolist()),
            name='40km_rctbias'
        )
        layer_object_40km_rctbias = keras.layers.Permute(
            dims=(3, 1, 2, 4), name='40km_rctbias_put-time-first'
        )(input_layer_object_40km_rctbias)

    regularizer_object = architecture_utils.get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight
    )

    num_levels = len(nwp_pooling_size_by_level_px)
    nwp_encoder_conv_layer_objects = [None] * (num_levels + 1)
    nwp_fcst_module_layer_objects = [None] * (num_levels + 1)
    nwp_encoder_pooling_layer_objects = [None] * num_levels

    rctbias_encoder_conv_layer_objects = [None] * (num_levels + 1)
    rctbias_fcst_module_layer_objects = [None] * (num_levels + 1)
    rctbias_encoder_pooling_layer_objects = [None] * num_levels

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

        if i == 0:
            this_input_layer_object = layer_object_2pt5km_res
        else:
            this_input_layer_object = nwp_encoder_pooling_layer_objects[i - 1]

        nwp_encoder_conv_layer_objects[i] = _get_2d_conv_block(
            input_layer_object=this_input_layer_object,
            do_residual=use_residual_blocks,
            num_conv_layers=nwp_encoder_num_conv_layers_by_level[i],
            filter_size_px=3,
            num_filters=nwp_encoder_num_channels_by_level[i],
            do_time_distributed_conv=True,
            regularizer_object=regularizer_object,
            activation_function_name=inner_activ_function_name,
            activation_function_alpha=inner_activ_function_alpha,
            dropout_rates=nwp_encoder_dropout_rate_by_level[i],
            use_batch_norm=use_batch_normalization,
            batch_norm_momentum=batch_norm_momentum,
            batch_norm_synch_flag=batch_norm_synch_flag,
            basic_layer_name='nwp_encoder_level{0:d}'.format(i)
        )

        this_name = 'nwp_encoder_level{0:d}_pooling'.format(i)
        this_pooling_layer_object = architecture_utils.get_2d_pooling_layer(
            num_rows_in_window=nwp_pooling_size_by_level_px[i],
            num_columns_in_window=nwp_pooling_size_by_level_px[i],
            num_rows_per_stride=nwp_pooling_size_by_level_px[i],
            num_columns_per_stride=nwp_pooling_size_by_level_px[i],
            pooling_type_string=architecture_utils.MAX_POOLING_STRING,
            layer_name=this_name
        )
        nwp_encoder_pooling_layer_objects[i] = keras.layers.TimeDistributed(
            this_pooling_layer_object, name=this_name
        )(nwp_encoder_conv_layer_objects[i])

        if not use_recent_biases:
            continue

        if i == 0:
            this_input_layer_object = layer_object_2pt5km_rctbias
        else:
            this_input_layer_object = (
                rctbias_encoder_pooling_layer_objects[i - 1]
            )

        rctbias_encoder_conv_layer_objects[i] = _get_2d_conv_block(
            input_layer_object=this_input_layer_object,
            do_residual=use_residual_blocks,
            num_conv_layers=rctbias_encoder_num_conv_layers_by_level[i],
            filter_size_px=3,
            num_filters=rctbias_encoder_num_channels_by_level[i],
            do_time_distributed_conv=True,
            regularizer_object=regularizer_object,
            activation_function_name=inner_activ_function_name,
            activation_function_alpha=inner_activ_function_alpha,
            dropout_rates=rctbias_encoder_dropout_rate_by_level[i],
            use_batch_norm=use_batch_normalization,
            batch_norm_momentum=batch_norm_momentum,
            batch_norm_synch_flag=batch_norm_synch_flag,
            basic_layer_name='rctbias_encoder_level{0:d}'.format(i)
        )

        this_name = 'rctbias_encoder_level{0:d}_pooling'.format(i)
        this_pooling_layer_object = architecture_utils.get_2d_pooling_layer(
            num_rows_in_window=rctbias_pooling_size_by_level_px[i],
            num_columns_in_window=rctbias_pooling_size_by_level_px[i],
            num_rows_per_stride=rctbias_pooling_size_by_level_px[i],
            num_columns_per_stride=rctbias_pooling_size_by_level_px[i],
            pooling_type_string=architecture_utils.MAX_POOLING_STRING,
            layer_name=this_name
        )
        rctbias_encoder_pooling_layer_objects[i] = keras.layers.TimeDistributed(
            this_pooling_layer_object, name=this_name
        )(rctbias_encoder_conv_layer_objects[i])

    num_levels_filled = num_levels_to_fill + 0

    if input_dimensions_10km_res is not None:
        i = num_levels_filled - 1
        this_layer_object = crop_layer(
            target_layer_object=nwp_encoder_pooling_layer_objects[i],
            source_layer_object=layer_object_10km_res,
            cropping_layer_name='10km_concat-cropping',
            num_spatiotemporal_dims=3
        )

        this_name = '10km_concat-with-finer'
        nwp_encoder_pooling_layer_objects[i] = keras.layers.Concatenate(
            axis=-1, name=this_name
        )(
            [nwp_encoder_pooling_layer_objects[i], this_layer_object]
        )

        if use_recent_biases:
            this_layer_object = crop_layer(
                target_layer_object=rctbias_encoder_pooling_layer_objects[i],
                source_layer_object=layer_object_10km_rctbias,
                cropping_layer_name='10km_rctbias_concat-cropping',
                num_spatiotemporal_dims=3
            )

            this_name = '10km_rctbias_concat-with-finer'
            rctbias_encoder_pooling_layer_objects[i] = keras.layers.Concatenate(
                axis=-1, name=this_name
            )(
                [rctbias_encoder_pooling_layer_objects[i], this_layer_object]
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

            if i == 0:
                this_input_layer_object = layer_object_10km_res
            else:
                this_input_layer_object = nwp_encoder_pooling_layer_objects[
                    i - 1
                ]

            nwp_encoder_conv_layer_objects[i] = _get_2d_conv_block(
                input_layer_object=this_input_layer_object,
                do_residual=use_residual_blocks,
                num_conv_layers=nwp_encoder_num_conv_layers_by_level[i],
                filter_size_px=3,
                num_filters=nwp_encoder_num_channels_by_level[i],
                do_time_distributed_conv=True,
                regularizer_object=regularizer_object,
                activation_function_name=inner_activ_function_name,
                activation_function_alpha=inner_activ_function_alpha,
                dropout_rates=nwp_encoder_dropout_rate_by_level[i],
                use_batch_norm=use_batch_normalization,
                batch_norm_momentum=batch_norm_momentum,
                batch_norm_synch_flag=batch_norm_synch_flag,
                basic_layer_name='nwp_encoder_level{0:d}'.format(i)
            )

            this_name = 'nwp_encoder_level{0:d}_pooling'.format(i)
            this_pooling_layer_object = architecture_utils.get_2d_pooling_layer(
                num_rows_in_window=nwp_pooling_size_by_level_px[i],
                num_columns_in_window=nwp_pooling_size_by_level_px[i],
                num_rows_per_stride=nwp_pooling_size_by_level_px[i],
                num_columns_per_stride=nwp_pooling_size_by_level_px[i],
                pooling_type_string=architecture_utils.MAX_POOLING_STRING,
                layer_name=this_name
            )
            nwp_encoder_pooling_layer_objects[i] = keras.layers.TimeDistributed(
                this_pooling_layer_object, name=this_name
            )(nwp_encoder_conv_layer_objects[i])

            if not use_recent_biases:
                continue

            if i == 0:
                this_input_layer_object = layer_object_10km_rctbias
            else:
                this_input_layer_object = (
                    rctbias_encoder_pooling_layer_objects[i - 1]
                )

            rctbias_encoder_conv_layer_objects[i] = _get_2d_conv_block(
                input_layer_object=this_input_layer_object,
                do_residual=use_residual_blocks,
                num_conv_layers=rctbias_encoder_num_conv_layers_by_level[i],
                filter_size_px=3,
                num_filters=rctbias_encoder_num_channels_by_level[i],
                do_time_distributed_conv=True,
                regularizer_object=regularizer_object,
                activation_function_name=inner_activ_function_name,
                activation_function_alpha=inner_activ_function_alpha,
                dropout_rates=rctbias_encoder_dropout_rate_by_level[i],
                use_batch_norm=use_batch_normalization,
                batch_norm_momentum=batch_norm_momentum,
                batch_norm_synch_flag=batch_norm_synch_flag,
                basic_layer_name='rctbias_encoder_level{0:d}'.format(i)
            )

            this_name = 'rctbias_encoder_level{0:d}_pooling'.format(i)
            this_pooling_layer_object = architecture_utils.get_2d_pooling_layer(
                num_rows_in_window=rctbias_pooling_size_by_level_px[i],
                num_columns_in_window=rctbias_pooling_size_by_level_px[i],
                num_rows_per_stride=rctbias_pooling_size_by_level_px[i],
                num_columns_per_stride=rctbias_pooling_size_by_level_px[i],
                pooling_type_string=architecture_utils.MAX_POOLING_STRING,
                layer_name=this_name
            )
            rctbias_encoder_pooling_layer_objects[i] = keras.layers.TimeDistributed(
                this_pooling_layer_object, name=this_name
            )(rctbias_encoder_conv_layer_objects[i])

        num_levels_filled += num_levels_to_fill

    if input_dimensions_20km_res is not None:
        i = num_levels_filled - 1
        this_layer_object = crop_layer(
            target_layer_object=nwp_encoder_pooling_layer_objects[i],
            source_layer_object=layer_object_20km_res,
            cropping_layer_name='20km_concat-cropping',
            num_spatiotemporal_dims=3
        )

        this_name = '20km_concat-with-finer'
        nwp_encoder_pooling_layer_objects[i] = keras.layers.Concatenate(
            axis=-1, name=this_name
        )(
            [nwp_encoder_pooling_layer_objects[i], this_layer_object]
        )

        if use_recent_biases:
            this_layer_object = crop_layer(
                target_layer_object=rctbias_encoder_pooling_layer_objects[i],
                source_layer_object=layer_object_20km_rctbias,
                cropping_layer_name='20km_rctbias_concat-cropping',
                num_spatiotemporal_dims=3
            )

            this_name = '20km_rctbias_concat-with-finer'
            rctbias_encoder_pooling_layer_objects[i] = keras.layers.Concatenate(
                axis=-1, name=this_name
            )(
                [rctbias_encoder_pooling_layer_objects[i], this_layer_object]
            )

        i = num_levels_filled
        if i == 0:
            this_input_layer_object = layer_object_20km_res
        else:
            this_input_layer_object = nwp_encoder_pooling_layer_objects[i - 1]

        nwp_encoder_conv_layer_objects[i] = _get_2d_conv_block(
            input_layer_object=this_input_layer_object,
            do_residual=use_residual_blocks,
            num_conv_layers=nwp_encoder_num_conv_layers_by_level[i],
            filter_size_px=3,
            num_filters=nwp_encoder_num_channels_by_level[i],
            do_time_distributed_conv=True,
            regularizer_object=regularizer_object,
            activation_function_name=inner_activ_function_name,
            activation_function_alpha=inner_activ_function_alpha,
            dropout_rates=nwp_encoder_dropout_rate_by_level[i],
            use_batch_norm=use_batch_normalization,
            batch_norm_momentum=batch_norm_momentum,
            batch_norm_synch_flag=batch_norm_synch_flag,
            basic_layer_name='nwp_encoder_level{0:d}'.format(i)
        )

        this_name = 'nwp_encoder_level{0:d}_pooling'.format(i)
        this_pooling_layer_object = architecture_utils.get_2d_pooling_layer(
            num_rows_in_window=nwp_pooling_size_by_level_px[i],
            num_columns_in_window=nwp_pooling_size_by_level_px[i],
            num_rows_per_stride=nwp_pooling_size_by_level_px[i],
            num_columns_per_stride=nwp_pooling_size_by_level_px[i],
            pooling_type_string=architecture_utils.MAX_POOLING_STRING,
            layer_name=this_name
        )
        nwp_encoder_pooling_layer_objects[i] = keras.layers.TimeDistributed(
            this_pooling_layer_object, name=this_name
        )(nwp_encoder_conv_layer_objects[i])

        if use_recent_biases:
            if i == 0:
                this_input_layer_object = layer_object_20km_rctbias
            else:
                this_input_layer_object = (
                    rctbias_encoder_pooling_layer_objects[i - 1]
                )

            rctbias_encoder_conv_layer_objects[i] = _get_2d_conv_block(
                input_layer_object=this_input_layer_object,
                do_residual=use_residual_blocks,
                num_conv_layers=rctbias_encoder_num_conv_layers_by_level[i],
                filter_size_px=3,
                num_filters=rctbias_encoder_num_channels_by_level[i],
                do_time_distributed_conv=True,
                regularizer_object=regularizer_object,
                activation_function_name=inner_activ_function_name,
                activation_function_alpha=inner_activ_function_alpha,
                dropout_rates=rctbias_encoder_dropout_rate_by_level[i],
                use_batch_norm=use_batch_normalization,
                batch_norm_momentum=batch_norm_momentum,
                batch_norm_synch_flag=batch_norm_synch_flag,
                basic_layer_name='rctbias_encoder_level{0:d}'.format(i)
            )

            this_name = 'rctbias_encoder_level{0:d}_pooling'.format(i)
            this_pooling_layer_object = architecture_utils.get_2d_pooling_layer(
                num_rows_in_window=rctbias_pooling_size_by_level_px[i],
                num_columns_in_window=rctbias_pooling_size_by_level_px[i],
                num_rows_per_stride=rctbias_pooling_size_by_level_px[i],
                num_columns_per_stride=rctbias_pooling_size_by_level_px[i],
                pooling_type_string=architecture_utils.MAX_POOLING_STRING,
                layer_name=this_name
            )
            rctbias_encoder_pooling_layer_objects[i] = keras.layers.TimeDistributed(
                this_pooling_layer_object, name=this_name
            )(rctbias_encoder_conv_layer_objects[i])

        num_levels_filled += 1

    if input_dimensions_40km_res is not None:
        i = num_levels_filled - 1
        this_layer_object = crop_layer(
            target_layer_object=nwp_encoder_pooling_layer_objects[i],
            source_layer_object=layer_object_40km_res,
            cropping_layer_name='40km_concat-cropping',
            num_spatiotemporal_dims=3
        )

        this_name = '40km_concat-with-finer'
        nwp_encoder_pooling_layer_objects[i] = keras.layers.Concatenate(
            axis=-1, name=this_name
        )(
            [nwp_encoder_pooling_layer_objects[i], this_layer_object]
        )

        if use_recent_biases:
            this_layer_object = crop_layer(
                target_layer_object=rctbias_encoder_pooling_layer_objects[i],
                source_layer_object=layer_object_40km_rctbias,
                cropping_layer_name='40km_rctbias_concat-cropping',
                num_spatiotemporal_dims=3
            )

            this_name = '40km_rctbias_concat-with-finer'
            rctbias_encoder_pooling_layer_objects[i] = keras.layers.Concatenate(
                axis=-1, name=this_name
            )(
                [rctbias_encoder_pooling_layer_objects[i], this_layer_object]
            )

        i = num_levels_filled
        if i == 0:
            this_input_layer_object = layer_object_40km_res
        else:
            this_input_layer_object = nwp_encoder_pooling_layer_objects[i - 1]

        nwp_encoder_conv_layer_objects[i] = _get_2d_conv_block(
            input_layer_object=this_input_layer_object,
            do_residual=use_residual_blocks,
            num_conv_layers=nwp_encoder_num_conv_layers_by_level[i],
            filter_size_px=3,
            num_filters=nwp_encoder_num_channels_by_level[i],
            do_time_distributed_conv=True,
            regularizer_object=regularizer_object,
            activation_function_name=inner_activ_function_name,
            activation_function_alpha=inner_activ_function_alpha,
            dropout_rates=nwp_encoder_dropout_rate_by_level[i],
            use_batch_norm=use_batch_normalization,
            batch_norm_momentum=batch_norm_momentum,
            batch_norm_synch_flag=batch_norm_synch_flag,
            basic_layer_name='nwp_encoder_level{0:d}'.format(i)
        )

        this_name = 'nwp_encoder_level{0:d}_pooling'.format(i)
        this_pooling_layer_object = architecture_utils.get_2d_pooling_layer(
            num_rows_in_window=nwp_pooling_size_by_level_px[i],
            num_columns_in_window=nwp_pooling_size_by_level_px[i],
            num_rows_per_stride=nwp_pooling_size_by_level_px[i],
            num_columns_per_stride=nwp_pooling_size_by_level_px[i],
            pooling_type_string=architecture_utils.MAX_POOLING_STRING,
            layer_name=this_name
        )
        nwp_encoder_pooling_layer_objects[i] = keras.layers.TimeDistributed(
            this_pooling_layer_object, name=this_name
        )(nwp_encoder_conv_layer_objects[i])

        if use_recent_biases:
            if i == 0:
                this_input_layer_object = layer_object_40km_rctbias
            else:
                this_input_layer_object = rctbias_encoder_pooling_layer_objects[i - 1]

            rctbias_encoder_conv_layer_objects[i] = _get_2d_conv_block(
                input_layer_object=this_input_layer_object,
                do_residual=use_residual_blocks,
                num_conv_layers=rctbias_encoder_num_conv_layers_by_level[i],
                filter_size_px=3,
                num_filters=rctbias_encoder_num_channels_by_level[i],
                do_time_distributed_conv=True,
                regularizer_object=regularizer_object,
                activation_function_name=inner_activ_function_name,
                activation_function_alpha=inner_activ_function_alpha,
                dropout_rates=rctbias_encoder_dropout_rate_by_level[i],
                use_batch_norm=use_batch_normalization,
                batch_norm_momentum=batch_norm_momentum,
                batch_norm_synch_flag=batch_norm_synch_flag,
                basic_layer_name='rctbias_encoder_level{0:d}'.format(i)
            )

            this_name = 'rctbias_encoder_level{0:d}_pooling'.format(i)
            this_pooling_layer_object = architecture_utils.get_2d_pooling_layer(
                num_rows_in_window=rctbias_pooling_size_by_level_px[i],
                num_columns_in_window=rctbias_pooling_size_by_level_px[i],
                num_rows_per_stride=rctbias_pooling_size_by_level_px[i],
                num_columns_per_stride=rctbias_pooling_size_by_level_px[i],
                pooling_type_string=architecture_utils.MAX_POOLING_STRING,
                layer_name=this_name
            )
            rctbias_encoder_pooling_layer_objects[i] = keras.layers.TimeDistributed(
                this_pooling_layer_object, name=this_name
            )(rctbias_encoder_conv_layer_objects[i])

        num_levels_filled += 1

    for i in range(num_levels_filled, num_levels + 1):
        nwp_encoder_conv_layer_objects[i] = _get_2d_conv_block(
            input_layer_object=nwp_encoder_pooling_layer_objects[i - 1],
            do_residual=use_residual_blocks,
            num_conv_layers=nwp_encoder_num_conv_layers_by_level[i],
            filter_size_px=3,
            num_filters=nwp_encoder_num_channels_by_level[i],
            do_time_distributed_conv=True,
            regularizer_object=regularizer_object,
            activation_function_name=inner_activ_function_name,
            activation_function_alpha=inner_activ_function_alpha,
            dropout_rates=nwp_encoder_dropout_rate_by_level[i],
            use_batch_norm=use_batch_normalization,
            batch_norm_momentum=batch_norm_momentum,
            batch_norm_synch_flag=batch_norm_synch_flag,
            basic_layer_name='nwp_encoder_level{0:d}'.format(i)
        )

        if i != num_levels:
            this_name = 'nwp_encoder_level{0:d}_pooling'.format(i)
            this_pooling_layer_object = architecture_utils.get_2d_pooling_layer(
                num_rows_in_window=nwp_pooling_size_by_level_px[i],
                num_columns_in_window=nwp_pooling_size_by_level_px[i],
                num_rows_per_stride=nwp_pooling_size_by_level_px[i],
                num_columns_per_stride=nwp_pooling_size_by_level_px[i],
                pooling_type_string=architecture_utils.MAX_POOLING_STRING,
                layer_name=this_name
            )
            nwp_encoder_pooling_layer_objects[i] = keras.layers.TimeDistributed(
                this_pooling_layer_object, name=this_name
            )(nwp_encoder_conv_layer_objects[i])

        if not use_recent_biases:
            continue

        rctbias_encoder_conv_layer_objects[i] = _get_2d_conv_block(
            input_layer_object=rctbias_encoder_pooling_layer_objects[i - 1],
            do_residual=use_residual_blocks,
            num_conv_layers=rctbias_encoder_num_conv_layers_by_level[i],
            filter_size_px=3,
            num_filters=rctbias_encoder_num_channels_by_level[i],
            do_time_distributed_conv=True,
            regularizer_object=regularizer_object,
            activation_function_name=inner_activ_function_name,
            activation_function_alpha=inner_activ_function_alpha,
            dropout_rates=rctbias_encoder_dropout_rate_by_level[i],
            use_batch_norm=use_batch_normalization,
            batch_norm_momentum=batch_norm_momentum,
            batch_norm_synch_flag=batch_norm_synch_flag,
            basic_layer_name='rctbias_encoder_level{0:d}'.format(i)
        )

        if i != num_levels:
            this_name = 'rctbias_encoder_level{0:d}_pooling'.format(i)
            this_pooling_layer_object = architecture_utils.get_2d_pooling_layer(
                num_rows_in_window=rctbias_pooling_size_by_level_px[i],
                num_columns_in_window=rctbias_pooling_size_by_level_px[i],
                num_rows_per_stride=rctbias_pooling_size_by_level_px[i],
                num_columns_per_stride=rctbias_pooling_size_by_level_px[i],
                pooling_type_string=architecture_utils.MAX_POOLING_STRING,
                layer_name=this_name
            )
            rctbias_encoder_pooling_layer_objects[i] = keras.layers.TimeDistributed(
                this_pooling_layer_object, name=this_name
            )(rctbias_encoder_conv_layer_objects[i])

    for i in range(num_levels + 1):
        this_name = 'nwp_fcst_level{0:d}_put-time-last'.format(i)
        nwp_fcst_module_layer_objects[i] = keras.layers.Permute(
            dims=(2, 3, 1, 4), name=this_name
        )(nwp_encoder_conv_layer_objects[i])

        if nwp_forecast_module_use_3d_conv:
            nwp_fcst_module_layer_objects[i] = _get_3d_conv_block(
                input_layer_object=nwp_fcst_module_layer_objects[i],
                do_residual=use_residual_blocks,
                num_conv_layers=nwp_forecast_module_num_conv_layers,
                filter_size_px=1,
                regularizer_object=regularizer_object,
                activation_function_name=inner_activ_function_name,
                activation_function_alpha=inner_activ_function_alpha,
                dropout_rates=nwp_forecast_module_dropout_rates,
                use_batch_norm=use_batch_normalization,
                batch_norm_momentum=batch_norm_momentum,
                batch_norm_synch_flag=batch_norm_synch_flag,
                basic_layer_name='nwp_fcst_level{0:d}'.format(i)
            )
        else:
            orig_dims = nwp_fcst_module_layer_objects[i].shape
            new_dims = orig_dims[1:-2] + (orig_dims[-2] * orig_dims[-1],)

            this_name = 'nwp_fcst_level{0:d}_remove-time-dim'.format(i)
            nwp_fcst_module_layer_objects[i] = keras.layers.Reshape(
                target_shape=new_dims, name=this_name
            )(nwp_fcst_module_layer_objects[i])

            nwp_fcst_module_layer_objects[i] = _get_2d_conv_block(
                input_layer_object=nwp_fcst_module_layer_objects[i],
                do_residual=use_residual_blocks,
                num_conv_layers=nwp_forecast_module_num_conv_layers,
                filter_size_px=1,
                num_filters=nwp_encoder_num_channels_by_level[i],
                do_time_distributed_conv=False,
                regularizer_object=regularizer_object,
                activation_function_name=inner_activ_function_name,
                activation_function_alpha=inner_activ_function_alpha,
                dropout_rates=nwp_forecast_module_dropout_rates,
                use_batch_norm=use_batch_normalization,
                batch_norm_momentum=batch_norm_momentum,
                batch_norm_synch_flag=batch_norm_synch_flag,
                basic_layer_name='nwp_fcst_level{0:d}'.format(i)
            )

        if not use_recent_biases:
            continue

        this_name = 'rctbias_fcst_level{0:d}_put-time-last'.format(i)
        rctbias_fcst_module_layer_objects[i] = keras.layers.Permute(
            dims=(2, 3, 1, 4), name=this_name
        )(rctbias_encoder_conv_layer_objects[i])

        if rctbias_forecast_module_use_3d_conv:
            rctbias_fcst_module_layer_objects[i] = _get_3d_conv_block(
                input_layer_object=rctbias_fcst_module_layer_objects[i],
                do_residual=use_residual_blocks,
                num_conv_layers=rctbias_forecast_module_num_conv_layers,
                filter_size_px=1,
                regularizer_object=regularizer_object,
                activation_function_name=inner_activ_function_name,
                activation_function_alpha=inner_activ_function_alpha,
                dropout_rates=rctbias_forecast_module_dropout_rates,
                use_batch_norm=use_batch_normalization,
                batch_norm_momentum=batch_norm_momentum,
                batch_norm_synch_flag=batch_norm_synch_flag,
                basic_layer_name='rctbias_fcst_level{0:d}'.format(i)
            )
        else:
            orig_dims = rctbias_fcst_module_layer_objects[i].shape
            new_dims = orig_dims[1:-2] + (orig_dims[-2] * orig_dims[-1],)

            this_name = 'rctbias_fcst_level{0:d}_remove-time-dim'.format(i)
            rctbias_fcst_module_layer_objects[i] = keras.layers.Reshape(
                target_shape=new_dims, name=this_name
            )(rctbias_fcst_module_layer_objects[i])

            rctbias_fcst_module_layer_objects[i] = _get_2d_conv_block(
                input_layer_object=rctbias_fcst_module_layer_objects[i],
                do_residual=use_residual_blocks,
                num_conv_layers=rctbias_forecast_module_num_conv_layers,
                filter_size_px=1,
                num_filters=rctbias_encoder_num_channels_by_level[i],
                do_time_distributed_conv=False,
                regularizer_object=regularizer_object,
                activation_function_name=inner_activ_function_name,
                activation_function_alpha=inner_activ_function_alpha,
                dropout_rates=rctbias_forecast_module_dropout_rates,
                use_batch_norm=use_batch_normalization,
                batch_norm_momentum=batch_norm_momentum,
                batch_norm_synch_flag=batch_norm_synch_flag,
                basic_layer_name='rctbias_fcst_level{0:d}'.format(i)
            )

    lagtgt_encoder_conv_layer_objects = [None] * (num_levels + 1)
    lagtgt_fcst_module_layer_objects = [None] * (num_levels + 1)
    lagtgt_encoder_pooling_layer_objects = [None] * num_levels
    loop_max = num_levels + 1 if num_lag_times > 0 else 0

    for i in range(loop_max):
        if i == 0:
            this_input_layer_object = layer_object_lagged_targets
        else:
            this_input_layer_object = lagtgt_encoder_pooling_layer_objects[
                i - 1
            ]

        lagtgt_encoder_conv_layer_objects[i] = _get_2d_conv_block(
            input_layer_object=this_input_layer_object,
            do_residual=use_residual_blocks,
            num_conv_layers=lagtgt_encoder_num_conv_layers_by_level[i],
            filter_size_px=3,
            num_filters=lagtgt_encoder_num_channels_by_level[i],
            do_time_distributed_conv=True,
            regularizer_object=regularizer_object,
            activation_function_name=inner_activ_function_name,
            activation_function_alpha=inner_activ_function_alpha,
            dropout_rates=lagtgt_encoder_dropout_rate_by_level[i],
            use_batch_norm=use_batch_normalization,
            batch_norm_momentum=batch_norm_momentum,
            batch_norm_synch_flag=batch_norm_synch_flag,
            basic_layer_name='lagtgt_encoder_level{0:d}'.format(i)
        )

        this_name = 'lagtgt_fcst_level{0:d}_put-time-last'.format(i)
        lagtgt_fcst_module_layer_objects[i] = keras.layers.Permute(
            dims=(2, 3, 1, 4), name=this_name
        )(lagtgt_encoder_conv_layer_objects[i])

        if lagtgt_forecast_module_use_3d_conv:
            lagtgt_fcst_module_layer_objects[i] = _get_3d_conv_block(
                input_layer_object=lagtgt_fcst_module_layer_objects[i],
                do_residual=use_residual_blocks,
                num_conv_layers=lagtgt_forecast_module_num_conv_layers,
                filter_size_px=1,
                regularizer_object=regularizer_object,
                activation_function_name=inner_activ_function_name,
                activation_function_alpha=inner_activ_function_alpha,
                dropout_rates=lagtgt_forecast_module_dropout_rates,
                use_batch_norm=use_batch_normalization,
                batch_norm_momentum=batch_norm_momentum,
                batch_norm_synch_flag=batch_norm_synch_flag,
                basic_layer_name='lagtgt_fcst_level{0:d}'.format(i)
            )
        else:
            orig_dims = lagtgt_fcst_module_layer_objects[i].shape
            new_dims = orig_dims[1:-2] + (orig_dims[-2] * orig_dims[-1],)

            this_name = 'lagtgt_fcst_level{0:d}_remove-time-dim'.format(i)
            lagtgt_fcst_module_layer_objects[i] = keras.layers.Reshape(
                target_shape=new_dims, name=this_name
            )(lagtgt_fcst_module_layer_objects[i])

            lagtgt_fcst_module_layer_objects[i] = _get_2d_conv_block(
                input_layer_object=lagtgt_fcst_module_layer_objects[i],
                do_residual=use_residual_blocks,
                num_conv_layers=lagtgt_forecast_module_num_conv_layers,
                filter_size_px=1,
                num_filters=lagtgt_encoder_num_channels_by_level[i],
                do_time_distributed_conv=False,
                regularizer_object=regularizer_object,
                activation_function_name=inner_activ_function_name,
                activation_function_alpha=inner_activ_function_alpha,
                dropout_rates=lagtgt_forecast_module_dropout_rates,
                use_batch_norm=use_batch_normalization,
                batch_norm_momentum=batch_norm_momentum,
                batch_norm_synch_flag=batch_norm_synch_flag,
                basic_layer_name='lagtgt_fcst_level{0:d}'.format(i)
            )

        if i == num_levels:
            break

        this_name = 'lagtgt_encoder_level{0:d}_pooling'.format(i)
        this_pooling_layer_object = architecture_utils.get_2d_pooling_layer(
            num_rows_in_window=lagtgt_pooling_size_by_level_px[i],
            num_columns_in_window=lagtgt_pooling_size_by_level_px[i],
            num_rows_per_stride=lagtgt_pooling_size_by_level_px[i],
            num_columns_per_stride=lagtgt_pooling_size_by_level_px[i],
            pooling_type_string=architecture_utils.MAX_POOLING_STRING,
            layer_name=this_name
        )
        lagtgt_encoder_pooling_layer_objects[i] = keras.layers.TimeDistributed(
            this_pooling_layer_object, name=this_name
        )(lagtgt_encoder_conv_layer_objects[i])

    last_conv_layer_matrix = numpy.full(
        (num_levels + 1, num_levels + 1), '', dtype=object
    )

    for i in range(num_levels + 1):
        these_layer_objects = [nwp_fcst_module_layer_objects[i]]
        if use_recent_biases:
            these_layer_objects.append(rctbias_fcst_module_layer_objects[i])
        if num_lag_times > 0:
            these_layer_objects.append(lagtgt_fcst_module_layer_objects[i])

        if len(these_layer_objects) == 1:
            last_conv_layer_matrix[i, 0] = these_layer_objects[0]
        else:
            this_name = 'fcst_level{0:d}_concat'.format(i)

            last_conv_layer_matrix[i, 0] = keras.layers.Concatenate(
                axis=-1, name=this_name
            )(these_layer_objects)

        i_new = i + 0
        j = 0

        while i_new > 0:
            i_new -= 1
            j += 1

            this_name = 'block{0:d}-{1:d}_upsampling'.format(i_new, j)
            this_layer_object = keras.layers.UpSampling2D(
                size=(2, 2), name=this_name
            )(last_conv_layer_matrix[i_new + 1, j - 1])

            this_layer_object = pad_layer(
                source_layer_object=this_layer_object,
                target_layer_object=last_conv_layer_matrix[i_new, 0],
                padding_layer_name='block{0:d}-{1:d}_padding'.format(i_new, j),
                num_spatiotemporal_dims=2
            )

            this_num_channels = int(numpy.round(
                0.5 * decoder_num_channels_by_level[i_new]
            ))

            last_conv_layer_matrix[i_new, j] = _get_2d_conv_block(
                input_layer_object=this_layer_object,
                do_residual=use_residual_blocks,
                num_conv_layers=1,
                filter_size_px=3,
                num_filters=this_num_channels,
                do_time_distributed_conv=False,
                regularizer_object=regularizer_object,
                activation_function_name=inner_activ_function_name,
                activation_function_alpha=inner_activ_function_alpha,
                dropout_rates=upsampling_dropout_rate_by_level[i_new],
                use_batch_norm=use_batch_normalization,
                batch_norm_momentum=batch_norm_momentum,
                batch_norm_synch_flag=batch_norm_synch_flag,
                basic_layer_name='block{0:d}-{1:d}_up'.format(i_new, j)
            )

            last_conv_layer_matrix[i_new, j] = create_skip_connection(
                input_layer_objects=
                last_conv_layer_matrix[i_new, :(j + 1)].tolist(),
                num_output_channels=decoder_num_channels_by_level[i_new],
                current_level_num=i_new,
                regularizer_object=regularizer_object
            )

            last_conv_layer_matrix[i_new, j] = _get_2d_conv_block(
                input_layer_object=last_conv_layer_matrix[i_new, j],
                do_residual=use_residual_blocks,
                num_conv_layers=num_decoder_conv_layers_by_level[i_new],
                filter_size_px=3,
                num_filters=decoder_num_channels_by_level[i_new],
                do_time_distributed_conv=False,
                regularizer_object=regularizer_object,
                activation_function_name=inner_activ_function_name,
                activation_function_alpha=inner_activ_function_alpha,
                dropout_rates=skip_dropout_rate_by_level[i_new],
                use_batch_norm=use_batch_normalization,
                batch_norm_momentum=batch_norm_momentum,
                batch_norm_synch_flag=batch_norm_synch_flag,
                basic_layer_name='block{0:d}-{1:d}_skip'.format(i_new, j)
            )

    if include_penultimate_conv:
        last_conv_layer_matrix[0, -1] = _get_2d_conv_block(
            input_layer_object=last_conv_layer_matrix[0, -1],
            do_residual=use_residual_blocks,
            num_conv_layers=1,
            filter_size_px=3,
            num_filters=2 * num_output_channels * ensemble_size,
            do_time_distributed_conv=False,
            regularizer_object=regularizer_object,
            activation_function_name=inner_activ_function_name,
            activation_function_alpha=inner_activ_function_alpha,
            dropout_rates=penultimate_conv_dropout_rate,
            use_batch_norm=use_batch_normalization,
            batch_norm_momentum=batch_norm_momentum,
            batch_norm_synch_flag=batch_norm_synch_flag,
            basic_layer_name='penultimate'
        )

    num_constrained_output_channels = (
        int(predict_gust_excess) + int(predict_dewpoint_depression)
    )
    do_residual_prediction = input_dimensions_predn_baseline is not None

    simple_output_layer_object = _get_2d_conv_block(
        input_layer_object=last_conv_layer_matrix[0, -1],
        do_residual=use_residual_blocks,
        num_conv_layers=1,
        filter_size_px=1,
        num_filters=(
            (num_output_channels - num_constrained_output_channels) *
            ensemble_size
        ),
        do_time_distributed_conv=False,
        regularizer_object=regularizer_object,
        activation_function_name=None,
        activation_function_alpha=0.,
        dropout_rates=-1.,
        use_batch_norm=False,
        batch_norm_momentum=batch_norm_momentum,
        batch_norm_synch_flag=batch_norm_synch_flag,
        basic_layer_name='last_conv_simple'
    )

    if not do_residual_prediction and output_activ_function_name is not None:
        simple_output_layer_object = architecture_utils.get_activation_layer(
            activation_function_string=output_activ_function_name,
            alpha_for_relu=output_activ_function_alpha,
            alpha_for_elu=output_activ_function_alpha,
            layer_name='last_conv_simple_activ1'
        )(simple_output_layer_object)

    if predict_dewpoint_depression:
        dd_output_layer_object = _get_2d_conv_block(
            input_layer_object=last_conv_layer_matrix[0, -1],
            do_residual=use_residual_blocks,
            num_conv_layers=1,
            filter_size_px=1,
            num_filters=ensemble_size,
            do_time_distributed_conv=False,
            regularizer_object=regularizer_object,
            activation_function_name=None,
            activation_function_alpha=0.,
            dropout_rates=-1.,
            use_batch_norm=False,
            batch_norm_momentum=batch_norm_momentum,
            batch_norm_synch_flag=batch_norm_synch_flag,
            basic_layer_name='last_conv_dd'
        )

        if not do_residual_prediction and output_activ_function_name is not None:
            dd_output_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=
                architecture_utils.RELU_FUNCTION_STRING,
                alpha_for_relu=0.,
                alpha_for_elu=0.,
                layer_name='last_conv_dd_activ1'
            )(dd_output_layer_object)
    else:
        dd_output_layer_object = None

    if predict_gust_excess:
        gf_output_layer_object = _get_2d_conv_block(
            input_layer_object=last_conv_layer_matrix[0, -1],
            do_residual=use_residual_blocks,
            num_conv_layers=1,
            filter_size_px=1,
            num_filters=ensemble_size,
            do_time_distributed_conv=False,
            regularizer_object=regularizer_object,
            activation_function_name=None,
            activation_function_alpha=0.,
            dropout_rates=-1.,
            use_batch_norm=False,
            batch_norm_momentum=batch_norm_momentum,
            batch_norm_synch_flag=batch_norm_synch_flag,
            basic_layer_name='last_conv_gex'
        )

        if not do_residual_prediction and output_activ_function_name is not None:
            gf_output_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=
                architecture_utils.RELU_FUNCTION_STRING,
                alpha_for_relu=0.,
                alpha_for_elu=0.,
                layer_name='last_conv_gex_activ1'
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

    if ensemble_size > 1:
        new_dims = (
            input_dimensions_2pt5km_res[0],
            input_dimensions_2pt5km_res[1],
            num_output_channels,
            ensemble_size
        )
        output_layer_object = keras.layers.Reshape(
            target_shape=new_dims, name='reshape_output'
        )(output_layer_object)

    if do_residual_prediction:
        if ensemble_size > 1:
            new_dims = (
                input_dimensions_predn_baseline[0],
                input_dimensions_predn_baseline[1],
                input_dimensions_predn_baseline[2],
                1
            )

            layer_object_predn_baseline = keras.layers.Reshape(
                target_shape=new_dims,
                name='resid_baseline_reshape'
            )(input_layer_object_predn_baseline)

            # if use_evidential_nn:
            #     layer_object_predn_baseline = keras.layers.Permute(
            #         dims=(1, 2, 4, 3),
            #         name='resid_baseline_permute'
            #     )(layer_object_predn_baseline)
            #
            #     padding_arg = ((0, 0), (0, 0), (0, ensemble_size - 1))
            #     layer_object_predn_baseline = keras.layers.ZeroPadding3D(
            #         padding=padding_arg,
            #         name='resid_baseline_pad'
            #     )(layer_object_predn_baseline)
            #
            #     layer_object_predn_baseline = keras.layers.Permute(
            #         dims=(1, 2, 4, 3),
            #         name='resid_baseline_permute-back'
            #     )(layer_object_predn_baseline)
        else:
            layer_object_predn_baseline = input_layer_object_predn_baseline

        output_layer_object = keras.layers.Add(name='output_add_baseline')([
            output_layer_object, layer_object_predn_baseline
        ])

        if num_constrained_output_channels > 0:
            if ensemble_size == 1:
                new_dims = (
                    input_dimensions_predn_baseline[0],
                    input_dimensions_predn_baseline[1],
                    input_dimensions_predn_baseline[2],
                    1
                )

                output_layer_object = keras.layers.Reshape(
                    target_shape=new_dims,
                    name='output_expand_dims'
                )(output_layer_object)

            cropping_arg = (
                (0, 0),
                (0, 0),
                (num_output_channels - num_constrained_output_channels, 0)
            )

            constrained_output_layer_object = keras.layers.Cropping3D(
                cropping=cropping_arg,
                name='output_get_constrained'
            )(output_layer_object)

            constrained_output_layer_object = (
                architecture_utils.get_activation_layer(
                    activation_function_string=
                    architecture_utils.RELU_FUNCTION_STRING,
                    alpha_for_relu=0.,
                    alpha_for_elu=0.,
                    layer_name='output_activ_constrained'
                )(constrained_output_layer_object)
            )

            cropping_arg = (
                (0, 0),
                (0, 0),
                (0, num_constrained_output_channels)
            )

            basic_output_layer_object = keras.layers.Cropping3D(
                cropping=cropping_arg,
                name='output_get_basic'
            )(output_layer_object)

            if output_activ_function_name is not None:
                basic_output_layer_object = (
                    architecture_utils.get_activation_layer(
                        activation_function_string=output_activ_function_name,
                        alpha_for_relu=output_activ_function_alpha,
                        alpha_for_elu=output_activ_function_alpha,
                        layer_name='output_activ_basic'
                    )(basic_output_layer_object)
                )

            this_name = 'output' if ensemble_size > 1 else 'output_concat'
            output_layer_object = keras.layers.Concatenate(
                axis=3, name=this_name
            )(
                [basic_output_layer_object, constrained_output_layer_object]
            )

            if ensemble_size == 1:
                new_dims = (
                    input_dimensions_predn_baseline[0],
                    input_dimensions_predn_baseline[1],
                    input_dimensions_predn_baseline[2]
                )

                output_layer_object = keras.layers.Reshape(
                    target_shape=new_dims,
                    name='output'
                )(output_layer_object)
        else:
            if output_activ_function_name is not None:
                output_layer_object = architecture_utils.get_activation_layer(
                    activation_function_string=output_activ_function_name,
                    alpha_for_relu=output_activ_function_alpha,
                    alpha_for_elu=output_activ_function_alpha,
                    layer_name='output'
                )(output_layer_object)

    input_layer_objects = [
        l for l in [
            input_layer_object_2pt5km_res, input_layer_object_const,
            input_layer_object_lagged_targets, input_layer_object_10km_res,
            input_layer_object_20km_res, input_layer_object_40km_res,
            input_layer_object_2pt5km_rctbias, input_layer_object_10km_rctbias,
            input_layer_object_20km_rctbias, input_layer_object_40km_rctbias,
            input_layer_object_predn_baseline
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
