"""Methods for creating Chiu-net architecture.

This is a U-net with TimeDistributed layers to handle NWP forecasts at different
lead times.

Based on Chiu et al. (2020): https://doi.org/10.1109/LRA.2020.2992184
"""

import os
import sys
import numpy
import keras
import keras.layers as layers

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import architecture_utils

try:
    _ = layers.Input(shape=(3, 4, 5))
except:
    import tensorflow.keras as keras
    import tensorflow.keras.layers as layers

INPUT_DIMENSIONS_2PT5KM_RES_KEY = 'input_dimensions_2pt5km_res'
INPUT_DIMENSIONS_10KM_RES_KEY = 'input_dimensions_10km_res'
INPUT_DIMENSIONS_20KM_RES_KEY = 'input_dimensions_20km_res'
INPUT_DIMENSIONS_40KM_RES_KEY = 'input_dimensions_40km_res'

NUM_CONV_LAYERS_KEY = 'num_conv_layers_by_level'
POOLING_SIZE_KEY = 'pooling_size_by_level_px'
NUM_CHANNELS_KEY = 'num_channels_by_level'
ENCODER_DROPOUT_RATES_KEY = 'encoder_dropout_rate_by_level'
DECODER_DROPOUT_RATES_KEY = 'decoder_dropout_rate_by_level'
SKIP_DROPOUT_RATES_KEY = 'skip_dropout_rate_by_level'

FC_MODULE_NUM_CONV_LAYERS_KEY = 'forecast_module_num_conv_layers'
FC_MODULE_DROPOUT_RATES_KEY = 'forecast_module_dropout_rates'
FC_MODULE_USE_3D_CONV = 'forecast_module_use_3d_conv'

INNER_ACTIV_FUNCTION_KEY = 'inner_activ_function_name'
INNER_ACTIV_FUNCTION_ALPHA_KEY = 'inner_activ_function_alpha'
OUTPUT_ACTIV_FUNCTION_KEY = 'output_activ_function_name'
OUTPUT_ACTIV_FUNCTION_ALPHA_KEY = 'output_activ_function_alpha'
L2_WEIGHT_KEY = 'l2_weight'
USE_BATCH_NORM_KEY = 'use_batch_normalization'
ENSEMBLE_SIZE_KEY = 'ensemble_size'
NUM_OUTPUT_CHANNELS_KEY = 'num_output_channels'
LOSS_FUNCTION_KEY = 'loss_function'
OPTIMIZER_FUNCTION_KEY = 'optimizer_function'

DEFAULT_OPTION_DICT = {
    NUM_CONV_LAYERS_KEY: numpy.full(9, 2, dtype=int),
    POOLING_SIZE_KEY: numpy.full(8, 2, dtype=int),
    NUM_CHANNELS_KEY: numpy.array([8, 12, 16, 24, 32, 48, 64, 96, 96], dtype=int),
    ENCODER_DROPOUT_RATES_KEY: numpy.full(9, 0.),
    DECODER_DROPOUT_RATES_KEY: numpy.full(8, 0.),
    SKIP_DROPOUT_RATES_KEY: numpy.full(8, 0.),
    FC_MODULE_NUM_CONV_LAYERS_KEY: 1,
    FC_MODULE_DROPOUT_RATES_KEY: numpy.array([0.]),
    FC_MODULE_USE_3D_CONV: True,
    INNER_ACTIV_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    OUTPUT_ACTIV_FUNCTION_KEY: None,
    OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.,
    USE_BATCH_NORM_KEY: True
}


def check_input_args(option_dict):
    """Error-checks input arguments.

    L = number of levels
    F = number of convolutional layers in forecasting module

    :param option_dict: Dictionary with the following keys.
    option_dict["input_dimensions_2pt5km_res"]:
        numpy array with input dimensions for 2.5-km NWP forecasts:
        [num_grid_rows, num_grid_columns, num_lead_times, num_channels].
        If you are not including 2.5-km data, make this None.
    option_dict["input_dimensions_10km_res"]: Same but for 10-km NWP forecasts.
    option_dict["input_dimensions_20km_res"]: Same but for 20-km NWP forecasts.
    option_dict["input_dimensions_40km_res"]: Same but for 40-km NWP forecasts.
    option_dict["num_conv_layers_by_level"]: length-B numpy array with number of
        conv layers for each block.
    option_dict["pooling_size_by_level_px"]: length-B numpy array with size of
        max-pooling window for each block.  For example, if you want 2-by-2
        pooling in the [j]th block, make pooling_size_by_level_px[j] = 2.
    option_dict["num_channels_by_level"]: length-(L + 1) numpy array with number
        of channels (feature maps) for each level.
    option_dict["encoder_dropout_rate_by_level"]: length-(L + 1) numpy array
        with dropout rate on encoder side for each level.  Use number <= 0 to
        indicate no-dropout.
    option_dict["decoder_dropout_rate_by_level"]: length-L numpy array with
        dropout rate on decoder side (in upconv layer) for each level.  Use
        number <= 0 to indicate no-dropout.
    option_dict["skip_dropout_rate_by_level"]: length-L numpy array with dropout
        rate in skip connection for each level.  Use number <= 0 to indicate
        no-dropout.
    option_dict["forecast_module_num_conv_layers"]: F in the above definitions.
    option_dict["forecast_module_dropout_rates"]: length-F numpy array with
        dropout rate for each conv layer in forecasting module.  Use number
        <= 0 to indicate no-dropout.
    option_dict["forecast_module_use_3d_conv"]: Boolean flag.  Determines
        whether forecasting module will use 2-D or 3-D convolution.
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
    option_dict["l2_weight"]: Strength of L2 regularization (for conv layers
        only).
    option_dict["use_batch_normalization"]: Boolean flag.  If True, will use
        batch normalization after each non-output layer.
    option_dict["ensemble_size"]: Number of ensemble members.
    option_dict["num_output_channels"]: Number of output channels.
    option_dict["loss_function"]: Loss function.
    option_dict["optimizer_function"]: Optimizer function.

    :return: option_dict: Same as input but maybe with default values added.
    """

    orig_option_dict = option_dict.copy()
    option_dict = DEFAULT_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    if option_dict[INPUT_DIMENSIONS_2PT5KM_RES_KEY] is not None:
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

    error_checking.assert_is_numpy_array(
        option_dict[NUM_CONV_LAYERS_KEY], num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(
        option_dict[NUM_CONV_LAYERS_KEY]
    )
    error_checking.assert_is_geq_numpy_array(
        option_dict[NUM_CONV_LAYERS_KEY], 1
    )

    num_levels = len(option_dict[NUM_CONV_LAYERS_KEY]) - 1

    error_checking.assert_is_numpy_array(
        option_dict[POOLING_SIZE_KEY],
        exact_dimensions=numpy.array([num_levels], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(
        option_dict[POOLING_SIZE_KEY]
    )
    error_checking.assert_is_geq_numpy_array(
        option_dict[POOLING_SIZE_KEY], 2
    )

    error_checking.assert_is_numpy_array(
        option_dict[NUM_CHANNELS_KEY],
        exact_dimensions=numpy.array([num_levels + 1], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(option_dict[NUM_CHANNELS_KEY])
    error_checking.assert_is_geq_numpy_array(option_dict[NUM_CHANNELS_KEY], 1)

    error_checking.assert_is_numpy_array(
        option_dict[ENCODER_DROPOUT_RATES_KEY],
        exact_dimensions=numpy.array([num_levels + 1], dtype=int)
    )
    error_checking.assert_is_leq_numpy_array(
        option_dict[ENCODER_DROPOUT_RATES_KEY], 1., allow_nan=True
    )

    error_checking.assert_is_numpy_array(
        option_dict[DECODER_DROPOUT_RATES_KEY],
        exact_dimensions=numpy.array([num_levels], dtype=int)
    )
    error_checking.assert_is_leq_numpy_array(
        option_dict[DECODER_DROPOUT_RATES_KEY], 1., allow_nan=True
    )

    error_checking.assert_is_numpy_array(
        option_dict[SKIP_DROPOUT_RATES_KEY],
        exact_dimensions=numpy.array([num_levels], dtype=int)
    )
    error_checking.assert_is_leq_numpy_array(
        option_dict[SKIP_DROPOUT_RATES_KEY], 1., allow_nan=True
    )

    fc_module_num_conv_layers = option_dict[FC_MODULE_NUM_CONV_LAYERS_KEY]
    error_checking.assert_is_integer(fc_module_num_conv_layers)
    error_checking.assert_is_greater(fc_module_num_conv_layers, 0)

    expected_dim = numpy.array([fc_module_num_conv_layers], dtype=int)

    fc_module_dropout_rates = option_dict[FC_MODULE_DROPOUT_RATES_KEY]
    error_checking.assert_is_numpy_array(
        fc_module_dropout_rates, exact_dimensions=expected_dim
    )
    error_checking.assert_is_leq_numpy_array(
        fc_module_dropout_rates, 1., allow_nan=True
    )

    error_checking.assert_is_boolean(option_dict[FC_MODULE_USE_3D_CONV])

    error_checking.assert_is_geq(option_dict[L2_WEIGHT_KEY], 0.)
    error_checking.assert_is_boolean(option_dict[USE_BATCH_NORM_KEY])
    error_checking.assert_is_integer(option_dict[ENSEMBLE_SIZE_KEY])
    error_checking.assert_is_geq(option_dict[ENSEMBLE_SIZE_KEY], 1)
    error_checking.assert_is_integer(option_dict[NUM_OUTPUT_CHANNELS_KEY])
    error_checking.assert_is_geq(option_dict[NUM_OUTPUT_CHANNELS_KEY], 1)

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

    # TODO(thunderhoser): metric_list should be an input arg.

    option_dict = check_input_args(option_dict)

    input_dimensions_2pt5km_res = option_dict[INPUT_DIMENSIONS_2PT5KM_RES_KEY]
    input_dimensions_10km_res = option_dict[INPUT_DIMENSIONS_10KM_RES_KEY]
    input_dimensions_20km_res = option_dict[INPUT_DIMENSIONS_20KM_RES_KEY]
    input_dimensions_40km_res = option_dict[INPUT_DIMENSIONS_40KM_RES_KEY]
    num_conv_layers_by_level = option_dict[NUM_CONV_LAYERS_KEY]
    pooling_size_by_level_px = option_dict[POOLING_SIZE_KEY]
    num_channels_by_level = option_dict[NUM_CHANNELS_KEY]
    encoder_dropout_rate_by_level = option_dict[ENCODER_DROPOUT_RATES_KEY]
    decoder_dropout_rate_by_level = option_dict[DECODER_DROPOUT_RATES_KEY]
    skip_dropout_rate_by_level = option_dict[SKIP_DROPOUT_RATES_KEY]
    forecast_module_num_conv_layers = option_dict[FC_MODULE_NUM_CONV_LAYERS_KEY]
    forecast_module_dropout_rates = option_dict[FC_MODULE_DROPOUT_RATES_KEY]
    forecast_module_use_3d_conv = option_dict[FC_MODULE_USE_3D_CONV]
    inner_activ_function_name = option_dict[INNER_ACTIV_FUNCTION_KEY]
    inner_activ_function_alpha = option_dict[INNER_ACTIV_FUNCTION_ALPHA_KEY]
    output_activ_function_name = option_dict[OUTPUT_ACTIV_FUNCTION_KEY]
    output_activ_function_alpha = option_dict[OUTPUT_ACTIV_FUNCTION_ALPHA_KEY]
    l2_weight = option_dict[L2_WEIGHT_KEY]
    use_batch_normalization = option_dict[USE_BATCH_NORM_KEY]
    loss_function = option_dict[LOSS_FUNCTION_KEY]
    optimizer_function = option_dict[OPTIMIZER_FUNCTION_KEY]
    ensemble_size = option_dict[ENSEMBLE_SIZE_KEY]
    num_output_channels = option_dict[NUM_OUTPUT_CHANNELS_KEY]

    if input_dimensions_2pt5km_res is None:
        input_layer_object_2pt5km_res = None
        layer_object_2pt5km_res = None
    else:
        input_layer_object_2pt5km_res = layers.Input(
            shape=tuple(input_dimensions_2pt5km_res.tolist())
        )
        layer_object_2pt5km_res = layers.Permute(
            dims=(3, 1, 2, 4), name='2pt5km_put_time_first'
        )(input_layer_object_2pt5km_res)

    if input_dimensions_10km_res is None:
        input_layer_object_10km_res = None
        layer_object_10km_res = None
    else:
        input_layer_object_10km_res = layers.Input(
            shape=tuple(input_dimensions_10km_res.tolist())
        )
        layer_object_10km_res = layers.Permute(
            dims=(3, 1, 2, 4), name='10km_put_time_first'
        )(input_layer_object_10km_res)

    if input_dimensions_20km_res is None:
        input_layer_object_20km_res = None
        layer_object_20km_res = None
    else:
        input_layer_object_20km_res = layers.Input(
            shape=tuple(input_dimensions_20km_res.tolist())
        )
        layer_object_20km_res = layers.Permute(
            dims=(3, 1, 2, 4), name='20km_put_time_first'
        )(input_layer_object_20km_res)

    if input_dimensions_40km_res is None:
        input_layer_object_40km_res = None
        layer_object_40km_res = None
    else:
        input_layer_object_40km_res = layers.Input(
            shape=tuple(input_dimensions_40km_res.tolist())
        )
        layer_object_40km_res = layers.Permute(
            dims=(3, 1, 2, 4), name='40km_put_time_first'
        )(input_layer_object_40km_res)

    l2_function = architecture_utils.get_weight_regularizer(l2_weight=l2_weight)

    num_lead_times = 0

    num_levels = len(pooling_size_by_level_px)
    conv_layer_by_level = [None] * (num_levels + 1)
    pooling_layer_by_level = [None] * num_levels

    if input_dimensions_2pt5km_res is not None:
        num_lead_times = input_dimensions_2pt5km_res[2]

        for level_index in range(2):
            i = level_index

            for j in range(num_conv_layers_by_level[i]):
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

                conv_layer_by_level[i] = layers.TimeDistributed(
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
            pooling_layer_by_level[i] = layers.TimeDistributed(
                this_pooling_layer_object, name=this_name
            )(conv_layer_by_level[i])

        if input_dimensions_10km_res is not None:
            i = 1
            this_name = 'concat_2pt5km_10km'
            pooling_layer_by_level[i] = layers.Concatenate(
                axis=-1, name=this_name
            )(
                [pooling_layer_by_level[i], layer_object_10km_res]
            )

    if input_dimensions_10km_res is not None:
        num_lead_times = input_dimensions_10km_res[2]
        i = 0 if input_dimensions_2pt5km_res is None else 2

        for j in range(num_conv_layers_by_level[i]):
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

            conv_layer_by_level[i] = layers.TimeDistributed(
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
        pooling_layer_by_level[i] = layers.TimeDistributed(
            this_pooling_layer_object, name=this_name
        )(conv_layer_by_level[i])

        if input_dimensions_20km_res is not None:
            i = 0 if input_dimensions_2pt5km_res is None else 2

            this_name = 'concat_10km_20km'
            pooling_layer_by_level[i] = layers.Concatenate(
                axis=-1, name=this_name
            )(
                [pooling_layer_by_level[i], layer_object_20km_res]
            )

    if input_dimensions_20km_res is not None:
        num_lead_times = input_dimensions_20km_res[2]
        i = 0 if input_dimensions_2pt5km_res is None else 2
        i += 0 if input_dimensions_10km_res is None else 1

        for j in range(num_conv_layers_by_level[i]):
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

            conv_layer_by_level[i] = layers.TimeDistributed(
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
        pooling_layer_by_level[i] = layers.TimeDistributed(
            this_pooling_layer_object, name=this_name
        )(conv_layer_by_level[i])

        if input_dimensions_40km_res is not None:
            i = 0 if input_dimensions_2pt5km_res is None else 2
            i += 0 if input_dimensions_10km_res is None else 1

            this_name = 'concat_20km_40km'
            pooling_layer_by_level[i] = layers.Concatenate(
                axis=-1, name=this_name
            )(
                [pooling_layer_by_level[i], layer_object_40km_res]
            )

    if input_dimensions_40km_res is not None:
        num_lead_times = input_dimensions_40km_res[2]
        i = 0 if input_dimensions_2pt5km_res is None else 2
        i += 0 if input_dimensions_10km_res is None else 1
        i += 0 if input_dimensions_20km_res is None else 1

        for j in range(num_conv_layers_by_level[i]):
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

            conv_layer_by_level[i] = layers.TimeDistributed(
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
        pooling_layer_by_level[i] = layers.TimeDistributed(
            this_pooling_layer_object, name=this_name
        )(conv_layer_by_level[i])

    start_index = 0 if input_dimensions_2pt5km_res is None else 2
    start_index += 0 if input_dimensions_10km_res is None else 1
    start_index += 0 if input_dimensions_20km_res is None else 1
    start_index += 0 if input_dimensions_40km_res is None else 1

    for i in range(start_index, num_levels + 1):
        for j in range(num_conv_layers_by_level[i]):
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

            conv_layer_by_level[i] = layers.TimeDistributed(
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
            pooling_layer_by_level[i] = layers.TimeDistributed(
                this_pooling_layer_object, name=this_name
            )(conv_layer_by_level[i])

    forecast_module_layer_object = layers.Permute(
        dims=(2, 3, 1, 4), name='fc_module_put_time_last'
    )(conv_layer_by_level[-1])

    if not forecast_module_use_3d_conv:
        orig_dims = forecast_module_layer_object.get_shape()
        new_dims = orig_dims[1:-2] + [orig_dims[-2] * orig_dims[-1]]

        forecast_module_layer_object = layers.Reshape(
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
                    [forecast_module_layer_object.shape[-1]]
                )
                forecast_module_layer_object = layers.Reshape(
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

    try:
        this_layer_object = keras.layers.UpSampling2D(
            size=(2, 2), interpolation='bilinear', name=this_name
        )(forecast_module_layer_object)
    except:
        this_layer_object = keras.layers.UpSampling2D(
            size=(2, 2), name=this_name
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

    if decoder_dropout_rate_by_level[i] > 0:
        this_name = 'upsampling_level{0:d}_dropout'.format(i)

        upconv_layer_by_level[i] = architecture_utils.get_dropout_layer(
            dropout_fraction=decoder_dropout_rate_by_level[i],
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

    num_upconv_rows = upconv_layer_by_level[i].get_shape()[1]
    num_desired_rows = conv_layer_by_level[i].get_shape()[1]
    num_padding_rows = num_desired_rows - num_upconv_rows

    num_upconv_columns = upconv_layer_by_level[i].get_shape()[2]
    num_desired_columns = conv_layer_by_level[i].get_shape()[2]
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
        for j in range(num_conv_layers_by_level[i]):
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

        if i == 0:
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

            if use_batch_normalization:
                skip_layer_by_level[i] = (
                    architecture_utils.get_batch_norm_layer(
                        layer_name='penultimate_conv_bn'
                    )(skip_layer_by_level[i])
                )

            break

        this_name = 'upsampling_level{0:d}'.format(i - 1)

        try:
            this_layer_object = keras.layers.UpSampling2D(
                size=(2, 2), interpolation='bilinear', name=this_name
            )(skip_layer_by_level[i])
        except:
            this_layer_object = keras.layers.UpSampling2D(
                size=(2, 2), name=this_name
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

        if decoder_dropout_rate_by_level[i - 1] > 0:
            this_name = 'upsampling_level{0:d}_dropout'.format(i - 1)
            upconv_layer_by_level[i - 1] = architecture_utils.get_dropout_layer(
                dropout_fraction=decoder_dropout_rate_by_level[i - 1],
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

        num_upconv_rows = upconv_layer_by_level[i - 1].get_shape()[1]
        num_desired_rows = conv_layer_by_level[i - 1].get_shape()[1]
        num_padding_rows = num_desired_rows - num_upconv_rows

        num_upconv_columns = upconv_layer_by_level[i - 1].get_shape()[2]
        num_desired_columns = conv_layer_by_level[i - 1].get_shape()[2]
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

    skip_layer_by_level[0] = architecture_utils.get_2d_conv_layer(
        num_kernel_rows=1, num_kernel_columns=1,
        num_rows_per_stride=1, num_columns_per_stride=1,
        num_filters=ensemble_size * num_output_channels,
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=l2_function,
        layer_name='last_conv'
    )(skip_layer_by_level[0])

    if output_activ_function_name is not None:
        skip_layer_by_level[0] = architecture_utils.get_activation_layer(
            activation_function_string=output_activ_function_name,
            alpha_for_relu=output_activ_function_alpha,
            alpha_for_elu=output_activ_function_alpha,
            layer_name='last_conv_activation'
        )(skip_layer_by_level[0])

    # TODO(thunderhoser): For now, input_dimensions_2pt5km_res cannot actually
    # be None.  In other words, the model must take 2.5-km data as input.  I
    # will change this if need be.
    new_dims = (
        input_dimensions_2pt5km_res[0], input_dimensions_2pt5km_res[1],
        num_output_channels, ensemble_size
    )
    skip_layer_by_level[0] = keras.layers.Reshape(
        target_shape=new_dims, name='reshape_predictions'
    )(skip_layer_by_level[0])

    input_layer_objects = []
    if input_dimensions_2pt5km_res is not None:
        input_layer_objects.append(input_layer_object_2pt5km_res)
    if input_dimensions_10km_res is not None:
        input_layer_objects.append(input_layer_object_10km_res)
    if input_dimensions_20km_res is not None:
        input_layer_objects.append(input_layer_object_20km_res)
    if input_dimensions_40km_res is not None:
        input_layer_objects.append(input_layer_object_40km_res)

    model_object = keras.models.Model(
        inputs=input_layer_objects, outputs=skip_layer_by_level[0]
    )

    model_object.compile(
        loss=loss_function, optimizer=optimizer_function
        # metrics=metric_function_list
    )

    model_object.summary()
    return model_object