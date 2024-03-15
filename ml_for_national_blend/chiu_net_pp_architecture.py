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

import architecture_utils
import chiu_net_architecture as chiu_net_arch

INPUT_DIMENSIONS_2PT5KM_RES_KEY = chiu_net_arch.INPUT_DIMENSIONS_2PT5KM_RES_KEY
INPUT_DIMENSIONS_10KM_RES_KEY = chiu_net_arch.INPUT_DIMENSIONS_10KM_RES_KEY
INPUT_DIMENSIONS_20KM_RES_KEY = chiu_net_arch.INPUT_DIMENSIONS_20KM_RES_KEY
INPUT_DIMENSIONS_40KM_RES_KEY = chiu_net_arch.INPUT_DIMENSIONS_40KM_RES_KEY

NUM_CHANNELS_KEY = chiu_net_arch.NUM_CHANNELS_KEY
POOLING_SIZE_KEY = chiu_net_arch.POOLING_SIZE_KEY
ENCODER_NUM_CONV_LAYERS_KEY = chiu_net_arch.ENCODER_NUM_CONV_LAYERS_KEY
ENCODER_DROPOUT_RATES_KEY = chiu_net_arch.ENCODER_DROPOUT_RATES_KEY
DECODER_NUM_CONV_LAYERS_KEY = chiu_net_arch.DECODER_NUM_CONV_LAYERS_KEY
UPSAMPLING_DROPOUT_RATES_KEY = chiu_net_arch.UPSAMPLING_DROPOUT_RATES_KEY
SKIP_DROPOUT_RATES_KEY = chiu_net_arch.SKIP_DROPOUT_RATES_KEY

FC_MODULE_NUM_CONV_LAYERS_KEY = chiu_net_arch.FC_MODULE_NUM_CONV_LAYERS_KEY
FC_MODULE_DROPOUT_RATES_KEY = chiu_net_arch.FC_MODULE_DROPOUT_RATES_KEY
FC_MODULE_USE_3D_CONV = chiu_net_arch.FC_MODULE_USE_3D_CONV

INCLUDE_PENULTIMATE_KEY = chiu_net_arch.INCLUDE_PENULTIMATE_KEY
PENULTIMATE_DROPOUT_RATE_KEY = chiu_net_arch.PENULTIMATE_DROPOUT_RATE_KEY
INNER_ACTIV_FUNCTION_KEY = chiu_net_arch.INNER_ACTIV_FUNCTION_KEY
INNER_ACTIV_FUNCTION_ALPHA_KEY = chiu_net_arch.INNER_ACTIV_FUNCTION_ALPHA_KEY
OUTPUT_ACTIV_FUNCTION_KEY = chiu_net_arch.OUTPUT_ACTIV_FUNCTION_KEY
OUTPUT_ACTIV_FUNCTION_ALPHA_KEY = chiu_net_arch.OUTPUT_ACTIV_FUNCTION_ALPHA_KEY
L1_WEIGHT_KEY = chiu_net_arch.L1_WEIGHT_KEY
L2_WEIGHT_KEY = chiu_net_arch.L2_WEIGHT_KEY
USE_BATCH_NORM_KEY = chiu_net_arch.USE_BATCH_NORM_KEY
ENSEMBLE_SIZE_KEY = chiu_net_arch.ENSEMBLE_SIZE_KEY

NUM_OUTPUT_CHANNELS_KEY = chiu_net_arch.NUM_OUTPUT_CHANNELS_KEY
PREDICT_GUST_FACTOR_KEY = chiu_net_arch.PREDICT_GUST_FACTOR_KEY
PREDICT_DEWPOINT_DEPRESSION_KEY = chiu_net_arch.PREDICT_DEWPOINT_DEPRESSION_KEY
LOSS_FUNCTION_KEY = chiu_net_arch.LOSS_FUNCTION_KEY
OPTIMIZER_FUNCTION_KEY = chiu_net_arch.OPTIMIZER_FUNCTION_KEY
METRIC_FUNCTIONS_KEY = chiu_net_arch.METRIC_FUNCTIONS_KEY


def _get_channel_counts_for_skip_cnxn(input_layer_objects, num_output_channels):
    """Determines number of channels for each input layer to skip connection.

    A = number of input layers.

    :param input_layer_objects: length-A list of input layers (instances of
        subclass of `keras.layers`).
    :param num_output_channels: Number of desired output channels (after
        concatenation).
    :return: desired_channel_counts: length-A numpy array with number of
        desired channels for each input layer.
    """

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
        desired_channel_counts[numpy.argmax(desired_channel_counts)] -= 1
    while numpy.sum(desired_channel_counts) < num_output_channels:
        desired_channel_counts[numpy.argmin(desired_channel_counts)] += 1

    assert numpy.sum(desired_channel_counts) == num_output_channels
    desired_channel_counts = numpy.maximum(desired_channel_counts, 1)

    return desired_channel_counts


def _create_skip_connection(input_layer_objects, num_output_channels,
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

    desired_input_channel_counts = _get_channel_counts_for_skip_cnxn(
        input_layer_objects=input_layer_objects,
        num_output_channels=num_output_channels
    )
    current_width = len(input_layer_objects) - 1

    for j in range(current_width):
        this_name = 'block{0:d}-{1:d}_preskipconv{2:d}'.format(
            current_level_num, current_width, j
        )

        input_layer_objects[j] = architecture_utils.get_2d_conv_layer(
            num_kernel_rows=3, num_kernel_columns=3,
            num_rows_per_stride=1, num_columns_per_stride=1,
            num_filters=desired_input_channel_counts[j],
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )(input_layer_objects[j])

    this_name = 'block{0:d}-{1:d}_skip'.format(current_level_num, current_width)
    return keras.layers.Concatenate(axis=-1, name=this_name)(
        input_layer_objects
    )


def create_model(option_dict):
    """Creates CNN.

    :param option_dict: See documentation for `check_input_args`.
    :return: model_object: Untrained (but compiled) instance of
        `keras.models.Model`.
    """

    option_dict = chiu_net_arch.check_input_args(option_dict)

    input_dimensions_2pt5km_res = option_dict[INPUT_DIMENSIONS_2PT5KM_RES_KEY]
    input_dimensions_10km_res = option_dict[INPUT_DIMENSIONS_10KM_RES_KEY]
    input_dimensions_20km_res = option_dict[INPUT_DIMENSIONS_20KM_RES_KEY]
    input_dimensions_40km_res = option_dict[INPUT_DIMENSIONS_40KM_RES_KEY]

    num_channels_by_level = option_dict[NUM_CHANNELS_KEY]
    pooling_size_by_level_px = option_dict[POOLING_SIZE_KEY]
    num_encoder_conv_layers_by_level = option_dict[ENCODER_NUM_CONV_LAYERS_KEY]
    encoder_dropout_rate_by_level = option_dict[ENCODER_DROPOUT_RATES_KEY]
    num_decoder_conv_layers_by_level = option_dict[DECODER_NUM_CONV_LAYERS_KEY]
    upsampling_dropout_rate_by_level = option_dict[UPSAMPLING_DROPOUT_RATES_KEY]
    skip_dropout_rate_by_level = option_dict[SKIP_DROPOUT_RATES_KEY]

    forecast_module_num_conv_layers = option_dict[FC_MODULE_NUM_CONV_LAYERS_KEY]
    forecast_module_dropout_rates = option_dict[FC_MODULE_DROPOUT_RATES_KEY]
    forecast_module_use_3d_conv = option_dict[FC_MODULE_USE_3D_CONV]

    include_penultimate_conv = option_dict[INCLUDE_PENULTIMATE_KEY]
    penultimate_conv_dropout_rate = option_dict[PENULTIMATE_DROPOUT_RATE_KEY]
    inner_activ_function_name = option_dict[INNER_ACTIV_FUNCTION_KEY]
    inner_activ_function_alpha = option_dict[INNER_ACTIV_FUNCTION_ALPHA_KEY]
    output_activ_function_name = option_dict[OUTPUT_ACTIV_FUNCTION_KEY]
    output_activ_function_alpha = option_dict[OUTPUT_ACTIV_FUNCTION_ALPHA_KEY]
    l1_weight = option_dict[L1_WEIGHT_KEY]
    l2_weight = option_dict[L2_WEIGHT_KEY]
    use_batch_normalization = option_dict[USE_BATCH_NORM_KEY]
    ensemble_size = option_dict[ENSEMBLE_SIZE_KEY]
    num_output_channels = option_dict[NUM_OUTPUT_CHANNELS_KEY]
    predict_gust_factor = option_dict[PREDICT_GUST_FACTOR_KEY]
    predict_dewpoint_depression = option_dict[PREDICT_DEWPOINT_DEPRESSION_KEY]

    loss_function = option_dict[LOSS_FUNCTION_KEY]
    optimizer_function = option_dict[OPTIMIZER_FUNCTION_KEY]
    metric_function_list = option_dict[METRIC_FUNCTIONS_KEY]

    input_layer_object_2pt5km_res = keras.layers.Input(
        shape=tuple(input_dimensions_2pt5km_res.tolist())
    )
    layer_object_2pt5km_res = keras.layers.Permute(
        dims=(3, 1, 2, 4), name='2pt5km_put_time_first'
    )(input_layer_object_2pt5km_res)

    if input_dimensions_10km_res is None:
        input_layer_object_10km_res = None
        layer_object_10km_res = None
    else:
        input_layer_object_10km_res = keras.layers.Input(
            shape=tuple(input_dimensions_10km_res.tolist())
        )
        layer_object_10km_res = keras.layers.Permute(
            dims=(3, 1, 2, 4), name='10km_put_time_first'
        )(input_layer_object_10km_res)

    if input_dimensions_20km_res is None:
        input_layer_object_20km_res = None
        layer_object_20km_res = None
    else:
        input_layer_object_20km_res = keras.layers.Input(
            shape=tuple(input_dimensions_20km_res.tolist())
        )
        layer_object_20km_res = keras.layers.Permute(
            dims=(3, 1, 2, 4), name='20km_put_time_first'
        )(input_layer_object_20km_res)

    if input_dimensions_40km_res is None:
        input_layer_object_40km_res = None
        layer_object_40km_res = None
    else:
        input_layer_object_40km_res = keras.layers.Input(
            shape=tuple(input_dimensions_40km_res.tolist())
        )
        layer_object_40km_res = keras.layers.Permute(
            dims=(3, 1, 2, 4), name='40km_put_time_first'
        )(input_layer_object_40km_res)

    regularizer_object = architecture_utils.get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight
    )

    num_lead_times = input_dimensions_2pt5km_res[2]

    num_levels = len(pooling_size_by_level_px)
    encoder_conv_layer_objects = [None] * (num_levels + 1)
    encoder_pooling_layer_objects = [None] * num_levels

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
                    previous_layer_object = encoder_pooling_layer_objects[i - 1]
            else:
                previous_layer_object = encoder_conv_layer_objects[i]

            this_name = 'encoder_level{0:d}_conv{1:d}'.format(i, j)
            this_conv_layer_object = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=num_channels_by_level[i],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object,
                layer_name=this_name
            )

            encoder_conv_layer_objects[i] = keras.layers.TimeDistributed(
                this_conv_layer_object, name=this_name
            )(previous_layer_object)

            this_name = 'encoder_level{0:d}_activation{1:d}'.format(i, j)
            encoder_conv_layer_objects[i] = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha,
                layer_name=this_name
            )(encoder_conv_layer_objects[i])

            if encoder_dropout_rate_by_level[i] > 0:
                this_name = 'encoder_level{0:d}_dropout{1:d}'.format(i, j)
                encoder_conv_layer_objects[i] = architecture_utils.get_dropout_layer(
                    dropout_fraction=encoder_dropout_rate_by_level[i],
                    layer_name=this_name
                )(encoder_conv_layer_objects[i])

            if use_batch_normalization:
                this_name = 'encoder_level{0:d}_bn{1:d}'.format(i, j)
                encoder_conv_layer_objects[i] = architecture_utils.get_batch_norm_layer(
                    layer_name=this_name
                )(encoder_conv_layer_objects[i])

        this_name = 'encoder_level{0:d}_pooling'.format(i)
        this_pooling_layer_object = architecture_utils.get_2d_pooling_layer(
            num_rows_in_window=pooling_size_by_level_px[i],
            num_columns_in_window=pooling_size_by_level_px[i],
            num_rows_per_stride=pooling_size_by_level_px[i],
            num_columns_per_stride=pooling_size_by_level_px[i],
            pooling_type_string=architecture_utils.MAX_POOLING_STRING,
            layer_name=this_name
        )
        encoder_pooling_layer_objects[i] = keras.layers.TimeDistributed(
            this_pooling_layer_object, name=this_name
        )(encoder_conv_layer_objects[i])

    num_levels_filled = num_levels_to_fill + 0

    if input_dimensions_10km_res is not None:
        i = num_levels_filled - 1
        this_name = 'concat_with_10km'
        encoder_pooling_layer_objects[i] = keras.layers.Concatenate(
            axis=-1, name=this_name
        )(
            [encoder_pooling_layer_objects[i], layer_object_10km_res]
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
                        previous_layer_object = (
                            encoder_pooling_layer_objects[i - 1]
                        )
                else:
                    previous_layer_object = encoder_conv_layer_objects[i]

                this_name = 'encoder_level{0:d}_conv{1:d}'.format(i, j)
                this_conv_layer_object = architecture_utils.get_2d_conv_layer(
                    num_kernel_rows=3, num_kernel_columns=3,
                    num_rows_per_stride=1, num_columns_per_stride=1,
                    num_filters=num_channels_by_level[i],
                    padding_type_string=architecture_utils.YES_PADDING_STRING,
                    weight_regularizer=regularizer_object,
                    layer_name=this_name
                )

                encoder_conv_layer_objects[i] = keras.layers.TimeDistributed(
                    this_conv_layer_object, name=this_name
                )(previous_layer_object)

                this_name = 'encoder_level{0:d}_activation{1:d}'.format(i, j)
                encoder_conv_layer_objects[i] = architecture_utils.get_activation_layer(
                    activation_function_string=inner_activ_function_name,
                    alpha_for_relu=inner_activ_function_alpha,
                    alpha_for_elu=inner_activ_function_alpha,
                    layer_name=this_name
                )(encoder_conv_layer_objects[i])

                if encoder_dropout_rate_by_level[i] > 0:
                    this_name = 'encoder_level{0:d}_dropout{1:d}'.format(i, j)
                    encoder_conv_layer_objects[i] = architecture_utils.get_dropout_layer(
                        dropout_fraction=encoder_dropout_rate_by_level[i],
                        layer_name=this_name
                    )(encoder_conv_layer_objects[i])

                if use_batch_normalization:
                    this_name = 'encoder_level{0:d}_bn{1:d}'.format(i, j)
                    encoder_conv_layer_objects[i] = architecture_utils.get_batch_norm_layer(
                        layer_name=this_name
                    )(encoder_conv_layer_objects[i])

            this_name = 'encoder_level{0:d}_pooling'.format(i)
            this_pooling_layer_object = architecture_utils.get_2d_pooling_layer(
                num_rows_in_window=pooling_size_by_level_px[i],
                num_columns_in_window=pooling_size_by_level_px[i],
                num_rows_per_stride=pooling_size_by_level_px[i],
                num_columns_per_stride=pooling_size_by_level_px[i],
                pooling_type_string=architecture_utils.MAX_POOLING_STRING,
                layer_name=this_name
            )
            encoder_pooling_layer_objects[i] = keras.layers.TimeDistributed(
                this_pooling_layer_object, name=this_name
            )(encoder_conv_layer_objects[i])

        num_levels_filled += num_levels_to_fill

    if input_dimensions_20km_res is not None:
        i = num_levels_filled - 1

        this_name = 'concat_with_20km'
        encoder_pooling_layer_objects[i] = keras.layers.Concatenate(
            axis=-1, name=this_name
        )(
            [encoder_pooling_layer_objects[i], layer_object_20km_res]
        )

        i = num_levels_filled

        for j in range(num_encoder_conv_layers_by_level[i]):
            if j == 0:
                if i == 0:
                    previous_layer_object = layer_object_20km_res
                else:
                    previous_layer_object = encoder_pooling_layer_objects[i - 1]
            else:
                previous_layer_object = encoder_conv_layer_objects[i]

            this_name = 'encoder_level{0:d}_conv{1:d}'.format(i, j)
            this_conv_layer_object = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=num_channels_by_level[i],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object,
                layer_name=this_name
            )

            encoder_conv_layer_objects[i] = keras.layers.TimeDistributed(
                this_conv_layer_object, name=this_name
            )(previous_layer_object)

            this_name = 'encoder_level{0:d}_activation{1:d}'.format(i, j)
            encoder_conv_layer_objects[i] = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha,
                layer_name=this_name
            )(encoder_conv_layer_objects[i])

            if encoder_dropout_rate_by_level[i] > 0:
                this_name = 'encoder_level{0:d}_dropout{1:d}'.format(i, j)
                encoder_conv_layer_objects[i] = architecture_utils.get_dropout_layer(
                    dropout_fraction=encoder_dropout_rate_by_level[i],
                    layer_name=this_name
                )(encoder_conv_layer_objects[i])

            if use_batch_normalization:
                this_name = 'encoder_level{0:d}_bn{1:d}'.format(i, j)
                encoder_conv_layer_objects[i] = architecture_utils.get_batch_norm_layer(
                    layer_name=this_name
                )(encoder_conv_layer_objects[i])

        this_name = 'encoder_level{0:d}_pooling'.format(i)
        this_pooling_layer_object = architecture_utils.get_2d_pooling_layer(
            num_rows_in_window=pooling_size_by_level_px[i],
            num_columns_in_window=pooling_size_by_level_px[i],
            num_rows_per_stride=pooling_size_by_level_px[i],
            num_columns_per_stride=pooling_size_by_level_px[i],
            pooling_type_string=architecture_utils.MAX_POOLING_STRING,
            layer_name=this_name
        )
        encoder_pooling_layer_objects[i] = keras.layers.TimeDistributed(
            this_pooling_layer_object, name=this_name
        )(encoder_conv_layer_objects[i])

        num_levels_filled += 1

    if input_dimensions_40km_res is not None:
        i = num_levels_filled - 1

        this_name = 'concat_with_40km'
        encoder_pooling_layer_objects[i] = keras.layers.Concatenate(
            axis=-1, name=this_name
        )(
            [encoder_pooling_layer_objects[i], layer_object_40km_res]
        )

        i = num_levels_filled

        for j in range(num_encoder_conv_layers_by_level[i]):
            if j == 0:
                if i == 0:
                    previous_layer_object = layer_object_40km_res
                else:
                    previous_layer_object = encoder_pooling_layer_objects[i - 1]
            else:
                previous_layer_object = encoder_conv_layer_objects[i]

            this_name = 'encoder_level{0:d}_conv{1:d}'.format(i, j)
            this_conv_layer_object = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=num_channels_by_level[i],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object,
                layer_name=this_name
            )

            encoder_conv_layer_objects[i] = keras.layers.TimeDistributed(
                this_conv_layer_object, name=this_name
            )(previous_layer_object)

            this_name = 'encoder_level{0:d}_activation{1:d}'.format(i, j)
            encoder_conv_layer_objects[i] = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha,
                layer_name=this_name
            )(encoder_conv_layer_objects[i])

            if encoder_dropout_rate_by_level[i] > 0:
                this_name = 'encoder_level{0:d}_dropout{1:d}'.format(i, j)
                encoder_conv_layer_objects[i] = architecture_utils.get_dropout_layer(
                    dropout_fraction=encoder_dropout_rate_by_level[i],
                    layer_name=this_name
                )(encoder_conv_layer_objects[i])

            if use_batch_normalization:
                this_name = 'encoder_level{0:d}_bn{1:d}'.format(i, j)
                encoder_conv_layer_objects[i] = architecture_utils.get_batch_norm_layer(
                    layer_name=this_name
                )(encoder_conv_layer_objects[i])

        this_name = 'encoder_level{0:d}_pooling'.format(i)
        this_pooling_layer_object = architecture_utils.get_2d_pooling_layer(
            num_rows_in_window=pooling_size_by_level_px[i],
            num_columns_in_window=pooling_size_by_level_px[i],
            num_rows_per_stride=pooling_size_by_level_px[i],
            num_columns_per_stride=pooling_size_by_level_px[i],
            pooling_type_string=architecture_utils.MAX_POOLING_STRING,
            layer_name=this_name
        )
        encoder_pooling_layer_objects[i] = keras.layers.TimeDistributed(
            this_pooling_layer_object, name=this_name
        )(encoder_conv_layer_objects[i])

        num_levels_filled += 1

    for i in range(num_levels_filled, num_levels + 1):
        for j in range(num_encoder_conv_layers_by_level[i]):
            if j == 0:
                previous_layer_object = encoder_pooling_layer_objects[i - 1]
            else:
                previous_layer_object = encoder_conv_layer_objects[i]

            this_name = 'encoder_level{0:d}_conv{1:d}'.format(i, j)
            this_conv_layer_object = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=num_channels_by_level[i],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object,
                layer_name=this_name
            )

            encoder_conv_layer_objects[i] = keras.layers.TimeDistributed(
                this_conv_layer_object, name=this_name
            )(previous_layer_object)

            this_name = 'encoder_level{0:d}_activation{1:d}'.format(i, j)
            encoder_conv_layer_objects[i] = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha,
                layer_name=this_name
            )(encoder_conv_layer_objects[i])

            if encoder_dropout_rate_by_level[i] > 0:
                this_name = 'encoder_level{0:d}_dropout{1:d}'.format(i, j)
                encoder_conv_layer_objects[i] = architecture_utils.get_dropout_layer(
                    dropout_fraction=encoder_dropout_rate_by_level[i],
                    layer_name=this_name
                )(encoder_conv_layer_objects[i])

            if use_batch_normalization:
                this_name = 'encoder_level{0:d}_bn{1:d}'.format(i, j)
                encoder_conv_layer_objects[i] = architecture_utils.get_batch_norm_layer(
                    layer_name=this_name
                )(encoder_conv_layer_objects[i])

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
            encoder_pooling_layer_objects[i] = keras.layers.TimeDistributed(
                this_pooling_layer_object, name=this_name
            )(encoder_conv_layer_objects[i])

    fcst_module_layer_objects = [None] * (num_levels + 1)

    for i in range(num_levels + 1):
        this_name = 'fcst_level{0:d}_put-time-last'.format(i)
        fcst_module_layer_objects[i] = keras.layers.Permute(
            dims=(2, 3, 1, 4), name=this_name
        )(encoder_conv_layer_objects[i])

        if not forecast_module_use_3d_conv:
            orig_dims = fcst_module_layer_objects[i].shape
            new_dims = orig_dims[1:-2] + [orig_dims[-2] * orig_dims[-1]]

            this_name = 'fcst_level{0:d}_remove-time-dim'.format(i)
            fcst_module_layer_objects[i] = keras.layers.Reshape(
                target_shape=new_dims, name=this_name
            )(fcst_module_layer_objects[i])

        for j in range(forecast_module_num_conv_layers):
            this_name = 'fcst_level{0:d}_conv{1:d}'.format(i, j)

            if forecast_module_use_3d_conv:
                if j == 0:
                    fcst_module_layer_objects[i] = (
                        architecture_utils.get_3d_conv_layer(
                            num_kernel_rows=1, num_kernel_columns=1,
                            num_kernel_heights=num_lead_times,
                            num_rows_per_stride=1, num_columns_per_stride=1,
                            num_heights_per_stride=1,
                            num_filters=num_channels_by_level[i],
                            padding_type_string=
                            architecture_utils.NO_PADDING_STRING,
                            weight_regularizer=regularizer_object,
                            layer_name=this_name
                        )(fcst_module_layer_objects[i])
                    )

                    new_dims = (
                        fcst_module_layer_objects[i].shape[1:3] +
                        (fcst_module_layer_objects[i].shape[-1],)
                    )

                    this_name = 'fcst_level{0:d}_remove-time-dim'.format(i)
                    fcst_module_layer_objects[i] = keras.layers.Reshape(
                        target_shape=new_dims, name=this_name
                    )(fcst_module_layer_objects[i])
                else:
                    fcst_module_layer_objects[i] = (
                        architecture_utils.get_2d_conv_layer(
                            num_kernel_rows=3, num_kernel_columns=3,
                            num_rows_per_stride=1, num_columns_per_stride=1,
                            num_filters=num_channels_by_level[i],
                            padding_type_string=
                            architecture_utils.YES_PADDING_STRING,
                            weight_regularizer=regularizer_object,
                            layer_name=this_name
                        )(fcst_module_layer_objects[i])
                    )
            else:
                fcst_module_layer_objects[i] = architecture_utils.get_2d_conv_layer(
                    num_kernel_rows=3, num_kernel_columns=3,
                    num_rows_per_stride=1, num_columns_per_stride=1,
                    num_filters=num_channels_by_level[i],
                    padding_type_string=architecture_utils.YES_PADDING_STRING,
                    weight_regularizer=regularizer_object,
                    layer_name=this_name
                )(fcst_module_layer_objects[i])

            this_name = 'fcst_level{0:d}_conv{1:d}_activation'.format(i, j)
            fcst_module_layer_objects[i] = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha,
                layer_name=this_name
            )(fcst_module_layer_objects[i])

            if forecast_module_dropout_rates[j] > 0:
                this_name = 'fcst_level{0:d}_conv{1:d}_dropout'.format(i, j)
                fcst_module_layer_objects[i] = architecture_utils.get_dropout_layer(
                    dropout_fraction=forecast_module_dropout_rates[j],
                    layer_name=this_name
                )(fcst_module_layer_objects[i])

            if use_batch_normalization:
                this_name = 'fcst_level{0:d}_conv{1:d}_bn'.format(i, j)
                fcst_module_layer_objects[i] = (
                    architecture_utils.get_batch_norm_layer(
                        layer_name=this_name
                    )(fcst_module_layer_objects[i])
                )

    last_conv_layer_matrix = numpy.full(
        (num_levels + 1, num_levels + 1), '', dtype=object
    )

    for i in range(num_levels + 1):
        last_conv_layer_matrix[i, 0] = fcst_module_layer_objects[i]
        i_new = i + 0
        j = 0

        while i_new > 0:
            i_new -= 1
            j += 1

            this_num_channels = int(numpy.round(
                0.5 * num_channels_by_level[i_new]
            ))

            this_name = 'block{0:d}-{1:d}_upconv'.format(i_new, j)
            this_layer_object = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=this_num_channels,
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object, layer_name=this_name
            )(last_conv_layer_matrix[i_new + 1, j - 1])

            this_name = 'block{0:d}-{1:d}_upsampling'.format(i_new, j)
            this_layer_object = keras.layers.UpSampling2D(
                size=(2, 2), name=this_name
            )(this_layer_object)

            this_name = 'block{0:d}-{1:d}_upconv_activation'.format(i_new, j)
            this_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha,
                layer_name=this_name
            )(this_layer_object)

            if upsampling_dropout_rate_by_level[i_new] > 0:
                this_name = 'block{0:d}-{1:d}_upconv_dropout'.format(i_new, j)
                this_layer_object = architecture_utils.get_dropout_layer(
                    dropout_fraction=upsampling_dropout_rate_by_level[i_new],
                    layer_name=this_name
                )(this_layer_object)

            num_upconv_rows = this_layer_object.shape[1]
            num_desired_rows = last_conv_layer_matrix[i_new, 0].shape[1]
            num_padding_rows = num_desired_rows - num_upconv_rows

            num_upconv_columns = this_layer_object.shape[2]
            num_desired_columns = (
                last_conv_layer_matrix[i_new, 0].shape[2]
            )
            num_padding_columns = num_desired_columns - num_upconv_columns

            if num_padding_rows + num_padding_columns > 0:
                padding_arg = ((0, num_padding_rows), (0, num_padding_columns))

                this_layer_object = keras.layers.ZeroPadding2D(
                    padding=padding_arg
                )(this_layer_object)

            last_conv_layer_matrix[i_new, j] = this_layer_object
            last_conv_layer_matrix[i_new, j] = _create_skip_connection(
                input_layer_objects=
                last_conv_layer_matrix[i_new, :(j + 1)].tolist(),
                num_output_channels=num_channels_by_level[i_new],
                current_level_num=i_new,
                regularizer_object=regularizer_object
            )

            for k in range(num_decoder_conv_layers_by_level[i_new]):
                if k > 0:
                    this_name = 'block{0:d}-{1:d}_skipconv{2:d}'.format(
                        i_new, j, k
                    )

                    last_conv_layer_matrix[i_new, j] = (
                        architecture_utils.get_2d_conv_layer(
                            num_kernel_rows=3, num_kernel_columns=3,
                            num_rows_per_stride=1, num_columns_per_stride=1,
                            num_filters=num_channels_by_level[i_new],
                            padding_type_string=
                            architecture_utils.YES_PADDING_STRING,
                            weight_regularizer=regularizer_object,
                            layer_name=this_name
                        )(last_conv_layer_matrix[i_new, j])
                    )

                this_name = 'block{0:d}-{1:d}_skipconv{2:d}_activation'.format(
                    i_new, j, k
                )

                last_conv_layer_matrix[i_new, j] = (
                    architecture_utils.get_activation_layer(
                        activation_function_string=inner_activ_function_name,
                        alpha_for_relu=inner_activ_function_alpha,
                        alpha_for_elu=inner_activ_function_alpha,
                        layer_name=this_name
                    )(last_conv_layer_matrix[i_new, j])
                )

                if skip_dropout_rate_by_level[i_new] > 0:
                    this_name = 'block{0:d}-{1:d}_skipconv{2:d}_dropout'.format(
                        i_new, j, k
                    )

                    last_conv_layer_matrix[i_new, j] = (
                        architecture_utils.get_dropout_layer(
                            dropout_fraction=skip_dropout_rate_by_level[i_new],
                            layer_name=this_name
                        )(last_conv_layer_matrix[i_new, j])
                    )

                if use_batch_normalization:
                    this_name = 'block{0:d}-{1:d}_skipconv{2:d}_bn'.format(
                        i_new, j, k
                    )

                    last_conv_layer_matrix[i_new, j] = (
                        architecture_utils.get_batch_norm_layer(
                            layer_name=this_name
                        )(last_conv_layer_matrix[i_new, j])
                    )

    if include_penultimate_conv:
        last_conv_layer_matrix[0, -1] = architecture_utils.get_2d_conv_layer(
            num_kernel_rows=3, num_kernel_columns=3,
            num_rows_per_stride=1, num_columns_per_stride=1,
            num_filters=2 * num_output_channels * ensemble_size,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object, layer_name='penultimate_conv'
        )(last_conv_layer_matrix[0, -1])

        last_conv_layer_matrix[0, -1] = architecture_utils.get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha,
            layer_name='penultimate_conv_activation'
        )(last_conv_layer_matrix[0, -1])

        if penultimate_conv_dropout_rate > 0:
            last_conv_layer_matrix[0, -1] = (
                architecture_utils.get_dropout_layer(
                    dropout_fraction=penultimate_conv_dropout_rate,
                    layer_name='penultimate_conv_dropout'
                )(last_conv_layer_matrix[0, -1])
            )

        if use_batch_normalization:
            last_conv_layer_matrix[0, -1] = (
                architecture_utils.get_batch_norm_layer(
                    layer_name='penultimate_conv_bn'
                )(last_conv_layer_matrix[0, -1])
            )

    this_offset = int(predict_gust_factor) + int(predict_dewpoint_depression)

    simple_output_layer_object = architecture_utils.get_2d_conv_layer(
        num_kernel_rows=1, num_kernel_columns=1,
        num_rows_per_stride=1, num_columns_per_stride=1,
        num_filters=(num_output_channels - this_offset) * ensemble_size,
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=regularizer_object,
        layer_name='last_conv_simple'
    )(last_conv_layer_matrix[0, -1])

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
            weight_regularizer=regularizer_object,
            layer_name='last_conv_dd'
        )(last_conv_layer_matrix[0, -1])

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
            weight_regularizer=regularizer_object,
            layer_name='last_conv_gf'
        )(last_conv_layer_matrix[0, -1])

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

    if ensemble_size > 1:
        new_dims = (
            input_dimensions_2pt5km_res[0], input_dimensions_2pt5km_res[1],
            num_output_channels, ensemble_size
        )
        output_layer_object = keras.layers.Reshape(
            target_shape=new_dims, name='reshape_predictions'
        )(output_layer_object)

    input_layer_objects = [input_layer_object_2pt5km_res]
    if input_dimensions_10km_res is not None:
        input_layer_objects.append(input_layer_object_10km_res)
    if input_dimensions_20km_res is not None:
        input_layer_objects.append(input_layer_object_20km_res)
    if input_dimensions_40km_res is not None:
        input_layer_objects.append(input_layer_object_40km_res)

    model_object = keras.models.Model(
        inputs=input_layer_objects, outputs=output_layer_object
    )

    model_object.compile(
        loss=loss_function, optimizer=optimizer_function,
        metrics=metric_function_list
    )

    model_object.summary()
    return model_object
