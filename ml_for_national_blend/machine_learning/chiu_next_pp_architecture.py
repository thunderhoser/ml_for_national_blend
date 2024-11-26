"""Chiu net with U-net++ backbone and ConvNext blocks."""

import numpy
import keras
import tensorflow
import tensorflow.math
from gewittergefahr.gg_utils import error_checking
from ml_for_national_blend.outside_code import architecture_utils
from ml_for_national_blend.machine_learning import \
    chiu_net_pp_architecture as chiu_net_pp_arch

INPUT_DIMENSIONS_CONST_KEY = chiu_net_pp_arch.INPUT_DIMENSIONS_CONST_KEY
INPUT_DIMENSIONS_2PT5KM_RES_KEY = (
    chiu_net_pp_arch.INPUT_DIMENSIONS_2PT5KM_RES_KEY
)
INPUT_DIMENSIONS_10KM_RES_KEY = chiu_net_pp_arch.INPUT_DIMENSIONS_10KM_RES_KEY
INPUT_DIMENSIONS_20KM_RES_KEY = chiu_net_pp_arch.INPUT_DIMENSIONS_20KM_RES_KEY
INPUT_DIMENSIONS_40KM_RES_KEY = chiu_net_pp_arch.INPUT_DIMENSIONS_40KM_RES_KEY
INPUT_DIMENSIONS_2PT5KM_RCTBIAS_KEY = (
    chiu_net_pp_arch.INPUT_DIMENSIONS_2PT5KM_RCTBIAS_KEY
)
INPUT_DIMENSIONS_10KM_RCTBIAS_KEY = (
    chiu_net_pp_arch.INPUT_DIMENSIONS_10KM_RCTBIAS_KEY
)
INPUT_DIMENSIONS_20KM_RCTBIAS_KEY = (
    chiu_net_pp_arch.INPUT_DIMENSIONS_20KM_RCTBIAS_KEY
)
INPUT_DIMENSIONS_40KM_RCTBIAS_KEY = (
    chiu_net_pp_arch.INPUT_DIMENSIONS_40KM_RCTBIAS_KEY
)
PREDN_BASELINE_DIMENSIONS_KEY = chiu_net_pp_arch.PREDN_BASELINE_DIMENSIONS_KEY
INPUT_DIMENSIONS_LAGGED_TARGETS_KEY = (
    chiu_net_pp_arch.INPUT_DIMENSIONS_LAGGED_TARGETS_KEY
)
DO_CONVNEXT_V2_KEY = 'do_convnext_v2'
USE_SPECTRAL_NORM_KEY = 'use_spectral_norm'

NWP_ENCODER_NUM_CHANNELS_KEY = chiu_net_pp_arch.NWP_ENCODER_NUM_CHANNELS_KEY
NWP_POOLING_SIZE_KEY = chiu_net_pp_arch.NWP_POOLING_SIZE_KEY
NWP_ENCODER_NUM_CONV_BLOCKS_KEY = 'nwp_encoder_num_conv_blocks_by_level'
NWP_ENCODER_DROPOUT_RATES_KEY = chiu_net_pp_arch.NWP_ENCODER_DROPOUT_RATES_KEY
NWP_FC_MODULE_NUM_CONV_BLOCKS_KEY = 'nwp_forecast_num_conv_blocks_by_level'
NWP_FC_MODULE_DROPOUT_RATES_KEY = 'nwp_forecast_module_drop_rate_by_level'
NWP_FC_MODULE_USE_3D_CONV = chiu_net_pp_arch.NWP_FC_MODULE_USE_3D_CONV

LAGTGT_ENCODER_NUM_CHANNELS_KEY = (
    chiu_net_pp_arch.LAGTGT_ENCODER_NUM_CHANNELS_KEY
)
LAGTGT_POOLING_SIZE_KEY = chiu_net_pp_arch.LAGTGT_POOLING_SIZE_KEY
LAGTGT_ENCODER_NUM_CONV_BLOCKS_KEY = 'lagtgt_encoder_num_conv_blocks_by_level'
LAGTGT_ENCODER_DROPOUT_RATES_KEY = (
    chiu_net_pp_arch.LAGTGT_ENCODER_DROPOUT_RATES_KEY
)
LAGTGT_FC_MODULE_NUM_CONV_BLOCKS_KEY = (
    'lagtgt_forecast_num_conv_blocks_by_level'
)
LAGTGT_FC_MODULE_DROPOUT_RATES_KEY = 'lagtgt_forecast_module_drop_rate_by_level'
LAGTGT_FC_MODULE_USE_3D_CONV = chiu_net_pp_arch.LAGTGT_FC_MODULE_USE_3D_CONV

RCTBIAS_ENCODER_NUM_CHANNELS_KEY = (
    chiu_net_pp_arch.RCTBIAS_ENCODER_NUM_CHANNELS_KEY
)
RCTBIAS_POOLING_SIZE_KEY = chiu_net_pp_arch.RCTBIAS_POOLING_SIZE_KEY
RCTBIAS_ENCODER_NUM_CONV_BLOCKS_KEY = 'rctbias_encoder_num_conv_blocks_by_level'
RCTBIAS_ENCODER_DROPOUT_RATES_KEY = (
    chiu_net_pp_arch.RCTBIAS_ENCODER_DROPOUT_RATES_KEY
)
RCTBIAS_FC_MODULE_NUM_CONV_BLOCKS_KEY = (
    'rctbias_forecast_num_conv_blocks_by_level'
)
RCTBIAS_FC_MODULE_DROPOUT_RATES_KEY = (
    'rctbias_forecast_module_drop_rate_by_level'
)
RCTBIAS_FC_MODULE_USE_3D_CONV = chiu_net_pp_arch.RCTBIAS_FC_MODULE_USE_3D_CONV

DECODER_NUM_CHANNELS_KEY = chiu_net_pp_arch.DECODER_NUM_CHANNELS_KEY
DECODER_NUM_CONV_BLOCKS_KEY = 'decoder_num_conv_blocks_by_level'
UPSAMPLING_DROPOUT_RATES_KEY = chiu_net_pp_arch.UPSAMPLING_DROPOUT_RATES_KEY
SKIP_DROPOUT_RATES_KEY = chiu_net_pp_arch.SKIP_DROPOUT_RATES_KEY

INCLUDE_PENULTIMATE_KEY = chiu_net_pp_arch.INCLUDE_PENULTIMATE_KEY
PENULTIMATE_DROPOUT_RATE_KEY = chiu_net_pp_arch.PENULTIMATE_DROPOUT_RATE_KEY
OUTPUT_ACTIV_FUNCTION_KEY = chiu_net_pp_arch.OUTPUT_ACTIV_FUNCTION_KEY
OUTPUT_ACTIV_FUNCTION_ALPHA_KEY = (
    chiu_net_pp_arch.OUTPUT_ACTIV_FUNCTION_ALPHA_KEY
)
L1_WEIGHT_KEY = chiu_net_pp_arch.L1_WEIGHT_KEY
L2_WEIGHT_KEY = chiu_net_pp_arch.L2_WEIGHT_KEY
ENSEMBLE_SIZE_KEY = chiu_net_pp_arch.ENSEMBLE_SIZE_KEY

NUM_OUTPUT_CHANNELS_KEY = chiu_net_pp_arch.NUM_OUTPUT_CHANNELS_KEY
PREDICT_GUST_EXCESS_KEY = chiu_net_pp_arch.PREDICT_GUST_EXCESS_KEY
PREDICT_DEWPOINT_DEPRESSION_KEY = (
    chiu_net_pp_arch.PREDICT_DEWPOINT_DEPRESSION_KEY
)
LOSS_FUNCTION_KEY = chiu_net_pp_arch.LOSS_FUNCTION_KEY
OPTIMIZER_FUNCTION_KEY = chiu_net_pp_arch.OPTIMIZER_FUNCTION_KEY
METRIC_FUNCTIONS_KEY = chiu_net_pp_arch.METRIC_FUNCTIONS_KEY

EPSILON_FOR_LAYER_NORM = 1e-6
EXPANSION_FACTOR_FOR_CONVNEXT = 4
INIT_VALUE_FOR_LAYER_SCALE = 1e-6

DEFAULT_OPTION_DICT = {
    # ENCODER_NUM_CONV_BLOCKS_KEY: numpy.full(9, 2, dtype=int),
    DECODER_NUM_CONV_BLOCKS_KEY: numpy.full(8, 2, dtype=int),
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
    OUTPUT_ACTIV_FUNCTION_KEY: None,
    OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.,
    L1_WEIGHT_KEY: 0.
}


@keras.saving.register_keras_serializable()
class LayerScale(keras.layers.Layer):
    """Layer-scale module.

    Scavenged from: https://github.com/danielabdi-noaa/HRRRemulator/blob/
                    master/tfmodel/convnext.py
    """

    def __init__(self, init_values, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, _):
        self.gamma = self.add_weight(
            shape=(self.projection_dim,),
            initializer=keras.initializers.Constant(self.init_values),
            trainable=True,
            name="gamma",
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "init_values": self.init_values,
                "projection_dim": self.projection_dim,
            }
        )
        return config


@keras.saving.register_keras_serializable()
class GRN(keras.layers.Layer):
    """Global response normalization.

    Scavenged from: https://github.com/facebookresearch/ConvNeXt-V2/blob/
                    2553895753323c6fe0b2bf390683f5ea358a42b9/models/
                    utils.py#L105-L116
    """

    def __init__(self, init_values, projection_dim, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim
        self.epsilon = epsilon

    def build(self, _):

        # TODO(thunderhoser): Not sure what the initial values should be.
        # The linked webpage uses zeros; the ChatGPT suggestion
        # (https://chatgpt.com/c/6740daf4-b010-8013-86ee-b6329e8ea6e9)
        # uses gamma = 1 and beta = 0; meanwhile, LayerScale from ConvNext 1
        # uses gamma = 1e-6.
        self.gamma = self.add_weight(
            shape=(self.projection_dim,),
            initializer=keras.initializers.Constant(self.init_values),
            trainable=True,
            name="gamma",
        )
        self.beta = self.add_weight(
            shape=(self.projection_dim,),
            initializer=keras.initializers.Constant(self.init_values),
            trainable=True,
            name="beta",
        )

    def call(self, inputs):
        # gx = tensorflow.norm(
        #     inputs, ord=2, axis=(1, 2), keepdims=True
        # )
        gx = tensorflow.sqrt(tensorflow.reduce_sum(
            tensorflow.square(inputs), axis=(1, 2), keepdims=True
        ))
        denominator = self.epsilon + tensorflow.math.reduce_mean(
            gx, axis=-1, keepdims=True
        )
        nx = gx / denominator

        return (self.gamma * nx * inputs) + self.beta + inputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "init_values": self.init_values,
                "projection_dim": self.projection_dim,
                "epsilon": self.epsilon
            }
        )
        return config


@keras.saving.register_keras_serializable()
class StochasticDepth(keras.layers.Layer):
    def __init__(self, survival_prob=0.9, **kwargs):
        super().__init__(**kwargs)
        self.survival_prob = survival_prob

    def call(self, inputs, training=None):
        if not training:
            return inputs[0] + inputs[1]

        batch_size = tensorflow.shape(inputs[0])[0]
        random_tensor = self.survival_prob + tensorflow.random.uniform(
            [batch_size, 1, 1, 1]
        )
        binary_tensor = tensorflow.floor(random_tensor)
        output = inputs[0] + binary_tensor * inputs[1] / self.survival_prob
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "survival_prob": self.survival_prob
            }
        )
        return config


class SpectralNormalization(keras.layers.Layer):
    def __init__(self, layer):
        super(SpectralNormalization, self).__init__()
        self.layer = layer

    def build(self, input_shape):
        # Build the wrapped layer (Dense, Conv2D, Conv3D, DepthwiseConv2D)
        self.layer.build(input_shape)

        layer_type_string = str(type(self.layer)).lower()

        if 'depthwise' in layer_type_string:
            self.w = self.layer.depthwise_kernel
        elif 'conv3d' in layer_type_string:
            self.w = self.layer.kernel
        elif 'conv2d' in layer_type_string:
            self.w = self.layer.kernel
        elif 'dense' in layer_type_string:
            self.w = self.layer.kernel
        else:
            raise ValueError(f"Unsupported layer type: {type(self.layer)}")

        self.u = self.add_weight(
            shape=(1, self.w.shape[-1]),
            initializer='random_normal',
            trainable=False,
            name='u'
        )

    def call(self, inputs):
        layer_type_string = str(type(self.layer)).lower()

        if 'depthwise' in layer_type_string:
            w_reshaped = tensorflow.reshape(self.w, [-1, self.w.shape[-1]])
        elif 'conv3d' in layer_type_string:
            w_reshaped = tensorflow.reshape(self.w, [-1, self.w.shape[-1]])
        elif 'conv2d' in layer_type_string:
            w_reshaped = tensorflow.reshape(self.w, [-1, self.w.shape[-1]])
        elif 'dense' in layer_type_string:
            w_reshaped = self.w

        print('LAYER NAME = {0:s}'.format(self.layer.name))
        print(self.w.shape)
        print(self.u.shape)
        print(w_reshaped.shape)
        print('\n')

        v = tensorflow.linalg.matvec(
            tensorflow.transpose(w_reshaped), self.u, transpose_a=True
        )
        v = tensorflow.math.l2_normalize(v)

        u = tensorflow.linalg.matvec(w_reshaped, v)
        u = tensorflow.math.l2_normalize(u)

        sigma = tensorflow.linalg.matvec(
            u,
            tensorflow.linalg.matvec(w_reshaped, v)
        )
        self.u.assign(u)

        if 'depthwise' in layer_type_string:
            self.layer.depthwise_kernel.assign(self.w / sigma)
        else:
            self.layer.kernel.assign(self.w / sigma)

        return self.layer(inputs)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def get_config(self):
        # Serialize the wrapped layer using its config
        config = super(SpectralNormalization, self).get_config()
        config.update({
            "layer": {
                "class_name": self.layer.__class__.__name__,
                "config": self.layer.get_config(),
            }
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Deserialize the wrapped layer from its config
        layer_class = getattr(keras.layers, config["layer"]["class_name"])
        layer = layer_class.from_config(config["layer"]["config"])
        return cls(layer, **config)


def __get_2d_convnext_block(
        input_layer_object, num_conv_layers, filter_size_px, num_filters,
        do_time_distributed_conv, regularizer_object, use_spectral_norm,
        do_activation, dropout_rate, basic_layer_name):
    """Creates ConvNext block for data with 2 spatial dimensions.

    L = number of conv layers

    :param input_layer_object: See documentation for `_get_2d_conv_block`.
    :param num_conv_layers: Same.
    :param filter_size_px: Same.
    :param num_filters: Same.
    :param do_time_distributed_conv: Same.
    :param regularizer_object: Same.
    :param use_spectral_norm: Same.
    :param do_activation: Same.
    :param dropout_rate: Same.
    :param basic_layer_name: Same.
    :return: output_layer_object: Same.
    """

    # TODO(thunderhoser): HACK.
    if filter_size_px == 3:
        actual_filter_size_px = 7
    else:
        actual_filter_size_px = filter_size_px + 0

    current_layer_object = None

    for i in range(num_conv_layers):
        if i == 0:
            this_input_layer_object = input_layer_object
        else:
            this_input_layer_object = current_layer_object

        this_name = '{0:s}_conv{1:d}'.format(basic_layer_name, i)
        current_layer_object = architecture_utils.get_2d_depthwise_conv_layer(
            num_kernel_rows=actual_filter_size_px,
            num_kernel_columns=actual_filter_size_px,
            num_rows_per_stride=1,
            num_columns_per_stride=1,
            num_filters=num_filters,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )

        if use_spectral_norm:
            current_layer_object = SpectralNormalization(current_layer_object)

        if do_time_distributed_conv:
            current_layer_object = keras.layers.TimeDistributed(
                current_layer_object, name=this_name
            )(this_input_layer_object)
        else:
            current_layer_object = current_layer_object(this_input_layer_object)

        this_name = '{0:s}_lyrnorm{1:d}'.format(basic_layer_name, i)
        current_layer_object = keras.layers.LayerNormalization(
            epsilon=EPSILON_FOR_LAYER_NORM, name=this_name
        )(
            current_layer_object
        )

        this_name = '{0:s}_dense{1:d}a'.format(basic_layer_name, i)
        dense_layer_object = architecture_utils.get_dense_layer(
            num_output_units=EXPANSION_FACTOR_FOR_CONVNEXT * num_filters,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )

        if use_spectral_norm:
            dense_layer_object = SpectralNormalization(dense_layer_object)
        current_layer_object = dense_layer_object(current_layer_object)

        if do_activation:
            this_name = '{0:s}_gelu{1:d}'.format(basic_layer_name, i)
            current_layer_object = keras.layers.Activation(
                'gelu', name=this_name
            )(current_layer_object)

        this_name = '{0:s}_dense{1:d}b'.format(basic_layer_name, i)
        dense_layer_object = architecture_utils.get_dense_layer(
            num_output_units=num_filters,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )

        if use_spectral_norm:
            dense_layer_object = SpectralNormalization(dense_layer_object)
        current_layer_object = dense_layer_object(current_layer_object)

        this_name = '{0:s}_lyrscale{1:d}'.format(basic_layer_name, i)
        current_layer_object = LayerScale(
            INIT_VALUE_FOR_LAYER_SCALE, num_filters, name=this_name
        )(current_layer_object)

        if i != num_conv_layers - 1:
            continue

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

            if use_spectral_norm:
                new_layer_object = SpectralNormalization(new_layer_object)

            if do_time_distributed_conv:
                new_layer_object = keras.layers.TimeDistributed(
                    new_layer_object, name=this_name
                )(input_layer_object)
            else:
                new_layer_object = new_layer_object(input_layer_object)

        this_name = '{0:s}_residual'.format(basic_layer_name)

        if dropout_rate > 0:
            current_layer_object = StochasticDepth(
                survival_prob=1. - dropout_rate, name=this_name
            )([new_layer_object, current_layer_object])
        else:
            current_layer_object = keras.layers.Add(name=this_name)([
                new_layer_object, current_layer_object
            ])

    return current_layer_object


def __get_2d_convnext2_block(
        input_layer_object, num_conv_layers, filter_size_px, num_filters,
        do_time_distributed_conv, regularizer_object, use_spectral_norm,
        do_activation, dropout_rate, basic_layer_name):
    """Creates ConvNext-2 block for data with 2 spatial dimensions.

    L = number of conv layers

    :param input_layer_object: See documentation for `__get_2d_convnext_block`.
    :param num_conv_layers: Same.
    :param filter_size_px: Same.
    :param num_filters: Same.
    :param do_time_distributed_conv: Same.
    :param regularizer_object: Same.
    :param use_spectral_norm: Same.
    :param do_activation: Same.
    :param dropout_rate: Same.
    :param basic_layer_name: Same.
    :return: output_layer_object: Same.
    """

    # TODO(thunderhoser): HACK.
    if filter_size_px == 3:
        actual_filter_size_px = 7
    else:
        actual_filter_size_px = filter_size_px + 0

    current_layer_object = None

    for i in range(num_conv_layers):
        if i == 0:
            this_input_layer_object = input_layer_object
        else:
            this_input_layer_object = current_layer_object

        this_name = '{0:s}_conv{1:d}'.format(basic_layer_name, i)
        current_layer_object = architecture_utils.get_2d_depthwise_conv_layer(
            num_kernel_rows=actual_filter_size_px,
            num_kernel_columns=actual_filter_size_px,
            num_rows_per_stride=1,
            num_columns_per_stride=1,
            num_filters=num_filters,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )

        if use_spectral_norm:
            current_layer_object = SpectralNormalization(current_layer_object)

        if do_time_distributed_conv:
            current_layer_object = keras.layers.TimeDistributed(
                current_layer_object, name=this_name
            )(this_input_layer_object)
        else:
            current_layer_object = current_layer_object(this_input_layer_object)

        this_name = '{0:s}_lyrnorm{1:d}'.format(basic_layer_name, i)
        current_layer_object = keras.layers.LayerNormalization(
            epsilon=EPSILON_FOR_LAYER_NORM, name=this_name
        )(
            current_layer_object
        )

        this_name = '{0:s}_dense{1:d}a'.format(basic_layer_name, i)
        dense_layer_object = architecture_utils.get_dense_layer(
            num_output_units=EXPANSION_FACTOR_FOR_CONVNEXT * num_filters,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )

        if use_spectral_norm:
            dense_layer_object = SpectralNormalization(dense_layer_object)
        current_layer_object = dense_layer_object(current_layer_object)

        if do_activation:
            this_name = '{0:s}_gelu{1:d}'.format(basic_layer_name, i)
            current_layer_object = keras.layers.Activation(
                'gelu', name=this_name
            )(current_layer_object)

        this_name = '{0:s}_grn{1:d}'.format(basic_layer_name, i)
        current_layer_object = GRN(
            init_values=INIT_VALUE_FOR_LAYER_SCALE,
            projection_dim=EXPANSION_FACTOR_FOR_CONVNEXT * num_filters,
            name=this_name
        )(current_layer_object)

        this_name = '{0:s}_dense{1:d}b'.format(basic_layer_name, i)
        dense_layer_object = architecture_utils.get_dense_layer(
            num_output_units=num_filters,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )

        if use_spectral_norm:
            dense_layer_object = SpectralNormalization(dense_layer_object)
        current_layer_object = dense_layer_object(current_layer_object)

        if i != num_conv_layers - 1:
            continue

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

            if use_spectral_norm:
                new_layer_object = SpectralNormalization(new_layer_object)

            if do_time_distributed_conv:
                new_layer_object = keras.layers.TimeDistributed(
                    new_layer_object, name=this_name
                )(input_layer_object)
            else:
                new_layer_object = new_layer_object(input_layer_object)

        this_name = '{0:s}_residual'.format(basic_layer_name)

        if dropout_rate > 0:
            current_layer_object = StochasticDepth(
                survival_prob=1. - dropout_rate, name=this_name
            )([new_layer_object, current_layer_object])
        else:
            current_layer_object = keras.layers.Add(name=this_name)([
                new_layer_object, current_layer_object
            ])

    return current_layer_object


def __get_3d_convnext_block(
        input_layer_object, num_time_steps, num_conv_layers, filter_size_px,
        regularizer_object, use_spectral_norm, do_activation, dropout_rate,
        basic_layer_name):
    """Creates ConvNext block for data with 3 spatial dimensions.

    :param input_layer_object: See documentation for `_get_3d_conv_block`.
    :param num_time_steps: Same.
    :param num_conv_layers: Same.
    :param filter_size_px: Same.
    :param regularizer_object: Same.
    :param use_spectral_norm: Same.
    :param do_activation: Same.
    :param dropout_rate: Same.
    :param basic_layer_name: Same.
    :return: output_layer_object: Same.
    """

    current_layer_object = None
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
            )

            if use_spectral_norm:
                current_layer_object = SpectralNormalization(
                    current_layer_object
                )
            current_layer_object = current_layer_object(input_layer_object)

            new_dims = (
                current_layer_object.shape[1:3] +
                (current_layer_object.shape[-1],)
            )

            this_name = '{0:s}_remove-time-dim'.format(basic_layer_name)
            current_layer_object = keras.layers.Reshape(
                target_shape=new_dims, name=this_name
            )(current_layer_object)
        else:
            dwconv_layer_object = (
                architecture_utils.get_2d_depthwise_conv_layer(
                    num_kernel_rows=filter_size_px,
                    num_kernel_columns=filter_size_px,
                    num_rows_per_stride=1,
                    num_columns_per_stride=1,
                    num_filters=num_filters,
                    padding_type_string=architecture_utils.YES_PADDING_STRING,
                    weight_regularizer=regularizer_object,
                    layer_name=this_name
                )
            )

            if use_spectral_norm:
                dwconv_layer_object = SpectralNormalization(
                    dwconv_layer_object
                )
            current_layer_object = dwconv_layer_object(current_layer_object)

        this_name = '{0:s}_lyrnorm{1:d}'.format(basic_layer_name, i)
        current_layer_object = keras.layers.LayerNormalization(
            epsilon=EPSILON_FOR_LAYER_NORM, name=this_name
        )(
            current_layer_object
        )

        this_name = '{0:s}_dense{1:d}a'.format(basic_layer_name, i)
        dense_layer_object = architecture_utils.get_dense_layer(
            num_output_units=EXPANSION_FACTOR_FOR_CONVNEXT * num_filters,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )

        if use_spectral_norm:
            dense_layer_object = SpectralNormalization(dense_layer_object)
        current_layer_object = dense_layer_object(current_layer_object)

        if do_activation:
            this_name = '{0:s}_gelu{1:d}'.format(basic_layer_name, i)
            current_layer_object = keras.layers.Activation(
                'gelu', name=this_name
            )(current_layer_object)

        this_name = '{0:s}_dense{1:d}b'.format(basic_layer_name, i)
        dense_layer_object = architecture_utils.get_dense_layer(
            num_output_units=num_filters,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )

        if use_spectral_norm:
            dense_layer_object = SpectralNormalization(dense_layer_object)
        current_layer_object = dense_layer_object(current_layer_object)

        this_name = '{0:s}_lyrscale{1:d}'.format(basic_layer_name, i)
        current_layer_object = LayerScale(
            INIT_VALUE_FOR_LAYER_SCALE, num_filters, name=this_name
        )(current_layer_object)

        if i != num_conv_layers - 1:
            continue

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

        if dropout_rate > 0:
            current_layer_object = StochasticDepth(
                survival_prob=1. - dropout_rate, name=this_name
            )([this_layer_object, current_layer_object])
        else:
            current_layer_object = keras.layers.Add(name=this_name)([
                this_layer_object, current_layer_object
            ])

    return current_layer_object


def __get_3d_convnext2_block(
        input_layer_object, num_time_steps, num_conv_layers, filter_size_px,
        regularizer_object, use_spectral_norm, do_activation, dropout_rate,
        basic_layer_name):
    """Creates ConvNext-2 block for data with 3 spatial dimensions.

    :param input_layer_object: See documentation for `__get_3d_convnext_block`.
    :param num_time_steps: Same.
    :param num_conv_layers: Same.
    :param filter_size_px: Same.
    :param regularizer_object: Same.
    :param use_spectral_norm: Same.
    :param do_activation: Same.
    :param basic_layer_name: Same.
    :param dropout_rate: Same.
    :return: output_layer_object: Same.
    """

    current_layer_object = None
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
            )

            if use_spectral_norm:
                current_layer_object = SpectralNormalization(
                    current_layer_object
                )
            current_layer_object = current_layer_object(input_layer_object)

            new_dims = (
                current_layer_object.shape[1:3] +
                (current_layer_object.shape[-1],)
            )

            this_name = '{0:s}_remove-time-dim'.format(basic_layer_name)
            current_layer_object = keras.layers.Reshape(
                target_shape=new_dims, name=this_name
            )(current_layer_object)
        else:
            dwconv_layer_object = (
                architecture_utils.get_2d_depthwise_conv_layer(
                    num_kernel_rows=filter_size_px,
                    num_kernel_columns=filter_size_px,
                    num_rows_per_stride=1,
                    num_columns_per_stride=1,
                    num_filters=num_filters,
                    padding_type_string=architecture_utils.YES_PADDING_STRING,
                    weight_regularizer=regularizer_object,
                    layer_name=this_name
                )
            )

            if use_spectral_norm:
                dwconv_layer_object = SpectralNormalization(
                    dwconv_layer_object
                )
            current_layer_object = dwconv_layer_object(current_layer_object)

        this_name = '{0:s}_lyrnorm{1:d}'.format(basic_layer_name, i)
        current_layer_object = keras.layers.LayerNormalization(
            epsilon=EPSILON_FOR_LAYER_NORM, name=this_name
        )(
            current_layer_object
        )

        this_name = '{0:s}_dense{1:d}a'.format(basic_layer_name, i)
        dense_layer_object = architecture_utils.get_dense_layer(
            num_output_units=EXPANSION_FACTOR_FOR_CONVNEXT * num_filters,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )

        if use_spectral_norm:
            dense_layer_object = SpectralNormalization(dense_layer_object)
        current_layer_object = dense_layer_object(current_layer_object)

        if do_activation:
            this_name = '{0:s}_gelu{1:d}'.format(basic_layer_name, i)
            current_layer_object = keras.layers.Activation(
                'gelu', name=this_name
            )(current_layer_object)

        this_name = '{0:s}_grn{1:d}'.format(basic_layer_name, i)
        current_layer_object = GRN(
            init_values=INIT_VALUE_FOR_LAYER_SCALE,
            projection_dim=EXPANSION_FACTOR_FOR_CONVNEXT * num_filters,
            name=this_name
        )(current_layer_object)

        this_name = '{0:s}_dense{1:d}b'.format(basic_layer_name, i)
        dense_layer_object = architecture_utils.get_dense_layer(
            num_output_units=num_filters,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )

        if use_spectral_norm:
            dense_layer_object = SpectralNormalization(dense_layer_object)
        current_layer_object = dense_layer_object(current_layer_object)

        if i != num_conv_layers - 1:
            continue

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

        if dropout_rate > 0:
            current_layer_object = StochasticDepth(
                survival_prob=1. - dropout_rate, name=this_name
            )([this_layer_object, current_layer_object])
        else:
            current_layer_object = keras.layers.Add(name=this_name)([
                this_layer_object, current_layer_object
            ])

    return current_layer_object


def _check_input_args(option_dict):
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
    option_dict["input_dimensions_2pt5km_rctbias"]: Same but for recent bias of
        2.5-km NWP forecasts.
    option_dict["input_dimensions_10km_rctbias"]: Same but for recent bias of
        10-km NWP forecasts.
    option_dict["input_dimensions_20km_rctbias"]: Same but for recent bias of
        20-km NWP forecasts.
    option_dict["input_dimensions_40km_rctbias"]: Same but for recent bias of
        40-km NWP forecasts.
    option_dict["input_dimensions_predn_baseline"]: Same but for prediction
        baseline.
    option_dict["input_dimensions_lagged_targets"]: Same but for lagged targets.
    option_dict["do_convnext_v2"]: Boolean flag.  If True, will use version 2 of
        ConvNext.
    option_dict["use_spectral_norm"]: Boolean flag.  If True, will use spectral
        normalization to regularize every conv layer.
    option_dict["nwp_encoder_num_channels_by_level"]: length-(L + 1) numpy array
        with number of channels (feature maps) at each level of NWP-encoder.
    option_dict["lagtgt_encoder_num_channels_by_level"]: Same but for lagged
        targets.  If you do not want to use lagged targets, make this None.
    option_dict["rctbias_encoder_num_channels_by_level"]: Same but for recent
        NWP biases.  If you do not want to use recent biases, make this None.
    option_dict["nwp_pooling_size_by_level_px"]: length-L numpy array with size
        of max-pooling window at each level of NWP-encoder.  For example, if you
        want 2-by-2 pooling at the [j]th level,
        make pooling_size_by_level_px[j] = 2.
    option_dict["lagtgt_pooling_size_by_level_px"]: Same but for lagged
        targets.  If you do not want to use lagged targets, make this None.
    option_dict["rctbias_pooling_size_by_level_px"]: Same but for recent
        NWP biases.  If you do not want to use recent biases, make this None.
    option_dict["nwp_encoder_num_conv_blocks_by_level"]: length-(L + 1) numpy
        array with number of conv blocks at each level of NWP-encoder.
    option_dict["lagtgt_encoder_num_conv_blocks_by_level"]: Same but for lagged
        targets.  If you do not want to use lagged targets, make this None.
    option_dict["rctbias_encoder_num_conv_blocks_by_level"]: Same but for recent
        NWP biases.  If you do not want to use recent biases, make this None.
    option_dict["nwp_encoder_drop_rate_by_level"]: length-(L + 1) numpy array
        with dropout rate at each level of NWP-encoder.  Use numbers <= 0 to
        indicate no-dropout.
    option_dict["lagtgt_encoder_drop_rate_by_level"]: Same but for lagged
        targets.  If you do not want to use lagged targets, make this None.
    option_dict["rctbias_encoder_drop_rate_by_level"]: Same but for recent
        NWP biases.  If you do not want to use recent biases, make this None.
    option_dict["nwp_forecast_num_conv_blocks_by_level"]: length-(L + 1) numpy
        array with number of conv blocks, by level, in forecasting module after
        NWP-encoder.
    option_dict["lagtgt_forecast_num_conv_blocks_by_level"]: Same but for lagged
        targets.  If you do not want to use lagged targets, make this None.
    option_dict["rctbias_forecast_num_conv_blocks_by_level"]: Same but for
        recent NWP biases.  If you do not want to use recent biases, make this
        None.
    option_dict["nwp_forecast_module_drop_rate_by_level"]: length-(L + 1) numpy
        array with dropout rate in NWP-forecasting module for each level.  Use
        numbers <= 0 to indicate no-dropout.
    option_dict["lagtgt_forecast_module_drop_rate_by_level"]: Same but for
        lagged targets.  If you do not want to use lagged targets, make this
        None.
    option_dict["rctbias_forecast_module_drop_rate_by_level"]: Same but for
        recent NWP biases.  If you do not want to use recent biases, make this
        None.
    option_dict["nwp_forecast_module_use_3d_conv"]: Boolean flag.  Determines
        whether NWP-forecasting module will use 2-D or 3-D convolution.
    option_dict["lagtgt_forecast_module_use_3d_conv"]: Same but for lagged
        targets.  If you do not want to use lagged targets, make this None.
    option_dict["rctbias_forecast_module_use_3d_conv"]: Same but for recent
        NWP biases.  If you do not want to use recent biases, make this None.
    option_dict["decoder_num_channels_by_level"]: length-L numpy array with
        number of channels (feature maps) at each level of decoder.
    option_dict["decoder_num_conv_blocks_by_level"]: length-L numpy array
        with number of conv blocks at each level of decoder.
    option_dict["upsampling_drop_rate_by_level"]: length-L numpy array with
        dropout rate in upconv layer at each level of decoder.  Use
        numbers <= 0 to indicate no-dropout.
    option_dict["skip_drop_rate_by_level"]: length-L numpy array with dropout
        rate in skip connection at each level of decoder.  Use
        numbers <= 0 to indicate no-dropout.
    option_dict["include_penultimate_conv"]: Boolean flag.  If True, will put in
        extra conv layer (with 3 x 3 filter) before final pixelwise conv.
    option_dict["penultimate_conv_dropout_rate"]: Dropout rate for penultimate
        conv layer.
    option_dict["output_activ_function_name"]: Name of activation function for
        output layer.  This can be None; otherwise, must be a string accepted by
        `architecture_utils.check_activation_function`.
    option_dict["output_activ_function_alpha"]: Alpha (slope parameter) for
        activation function for all output layer.  Applies only to ReLU and eLU.
    option_dict["l1_weight"]: Strength of L1 regularization (for conv layers
        only).
    option_dict["l2_weight"]: Strength of L2 regularization (for conv layers
        only).
    option_dict["ensemble_size"]: Number of ensemble members.
    option_dict["num_output_channels"]: Number of output channels.
    option_dict["predict_gust_excess"]: Boolean flag.  If True, the model needs
        to predict gust excess.
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

    num_rows_2pt5km = option_dict[INPUT_DIMENSIONS_2PT5KM_RES_KEY][0]
    num_columns_2pt5km = option_dict[INPUT_DIMENSIONS_2PT5KM_RES_KEY][1]
    num_channels = option_dict[NUM_OUTPUT_CHANNELS_KEY]

    if option_dict[PREDN_BASELINE_DIMENSIONS_KEY] is not None:
        expected_dim = numpy.array(
            [num_rows_2pt5km, num_columns_2pt5km, num_channels], dtype=int
        )
        assert numpy.array_equal(
            option_dict[PREDN_BASELINE_DIMENSIONS_KEY], expected_dim
        )

    error_checking.assert_is_boolean(option_dict[DO_CONVNEXT_V2_KEY])
    error_checking.assert_is_boolean(option_dict[USE_SPECTRAL_NORM_KEY])

    if option_dict[INPUT_DIMENSIONS_CONST_KEY] is not None:
        expected_dim = numpy.array([
            num_rows_2pt5km, num_columns_2pt5km,
            option_dict[INPUT_DIMENSIONS_CONST_KEY][-1]
        ], dtype=int)

        assert numpy.array_equal(
            option_dict[INPUT_DIMENSIONS_CONST_KEY], expected_dim
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

    if option_dict[INPUT_DIMENSIONS_2PT5KM_RCTBIAS_KEY] is not None:
        expected_dim = numpy.array([
            num_rows_2pt5km, num_columns_2pt5km,
            option_dict[INPUT_DIMENSIONS_2PT5KM_RCTBIAS_KEY][-2],
            option_dict[INPUT_DIMENSIONS_2PT5KM_RCTBIAS_KEY][-1]
        ], dtype=int)

        assert numpy.array_equal(
            option_dict[INPUT_DIMENSIONS_2PT5KM_RCTBIAS_KEY], expected_dim
        )

    if option_dict[INPUT_DIMENSIONS_10KM_RCTBIAS_KEY] is not None:
        num_rows_10km = option_dict[INPUT_DIMENSIONS_10KM_RES_KEY][0]
        num_columns_10km = option_dict[INPUT_DIMENSIONS_10KM_RES_KEY][1]
        expected_dim = numpy.array([
            num_rows_10km, num_columns_10km,
            option_dict[INPUT_DIMENSIONS_10KM_RCTBIAS_KEY][-2],
            option_dict[INPUT_DIMENSIONS_10KM_RCTBIAS_KEY][-1]
        ], dtype=int)

        assert numpy.array_equal(
            option_dict[INPUT_DIMENSIONS_10KM_RCTBIAS_KEY], expected_dim
        )

    if option_dict[INPUT_DIMENSIONS_20KM_RCTBIAS_KEY] is not None:
        num_rows_20km = option_dict[INPUT_DIMENSIONS_20KM_RES_KEY][0]
        num_columns_20km = option_dict[INPUT_DIMENSIONS_20KM_RES_KEY][1]
        expected_dim = numpy.array([
            num_rows_20km, num_columns_20km,
            option_dict[INPUT_DIMENSIONS_20KM_RCTBIAS_KEY][-2],
            option_dict[INPUT_DIMENSIONS_20KM_RCTBIAS_KEY][-1]
        ], dtype=int)

        assert numpy.array_equal(
            option_dict[INPUT_DIMENSIONS_20KM_RCTBIAS_KEY], expected_dim
        )

    if option_dict[INPUT_DIMENSIONS_40KM_RCTBIAS_KEY] is not None:
        num_rows_40km = option_dict[INPUT_DIMENSIONS_40KM_RES_KEY][0]
        num_columns_40km = option_dict[INPUT_DIMENSIONS_40KM_RES_KEY][1]
        expected_dim = numpy.array([
            num_rows_40km, num_columns_40km,
            option_dict[INPUT_DIMENSIONS_40KM_RCTBIAS_KEY][-2],
            option_dict[INPUT_DIMENSIONS_40KM_RCTBIAS_KEY][-1]
        ], dtype=int)

        assert numpy.array_equal(
            option_dict[INPUT_DIMENSIONS_40KM_RCTBIAS_KEY], expected_dim
        )

    if option_dict[INPUT_DIMENSIONS_LAGGED_TARGETS_KEY] is not None:
        expected_dim = numpy.array([
            num_rows_2pt5km, num_columns_2pt5km,
            option_dict[INPUT_DIMENSIONS_LAGGED_TARGETS_KEY][-2], num_channels
        ], dtype=int)

        assert numpy.array_equal(
            option_dict[INPUT_DIMENSIONS_LAGGED_TARGETS_KEY], expected_dim
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
        option_dict[NWP_ENCODER_NUM_CONV_BLOCKS_KEY],
        exact_dimensions=numpy.array([num_levels + 1], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(
        option_dict[NWP_ENCODER_NUM_CONV_BLOCKS_KEY]
    )
    error_checking.assert_is_geq_numpy_array(
        option_dict[NWP_ENCODER_NUM_CONV_BLOCKS_KEY], 1
    )

    error_checking.assert_is_numpy_array(
        option_dict[NWP_ENCODER_DROPOUT_RATES_KEY],
        exact_dimensions=numpy.array([num_levels + 1], dtype=int)
    )
    error_checking.assert_is_leq_numpy_array(
        option_dict[NWP_ENCODER_DROPOUT_RATES_KEY], 1., allow_nan=True
    )

    nwp_forecast_num_conv_blocks_by_level = option_dict[
        NWP_FC_MODULE_NUM_CONV_BLOCKS_KEY
    ]
    error_checking.assert_is_numpy_array(
        nwp_forecast_num_conv_blocks_by_level,
        exact_dimensions=numpy.array([num_levels + 1], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(
        nwp_forecast_num_conv_blocks_by_level
    )
    error_checking.assert_is_greater_numpy_array(
        nwp_forecast_num_conv_blocks_by_level, 0
    )

    nwp_fc_module_drop_rate_by_level = option_dict[
        NWP_FC_MODULE_DROPOUT_RATES_KEY
    ]
    error_checking.assert_is_numpy_array(
        nwp_fc_module_drop_rate_by_level,
        exact_dimensions=numpy.array([num_levels + 1], dtype=int)
    )
    error_checking.assert_is_leq_numpy_array(
        nwp_fc_module_drop_rate_by_level, 1., allow_nan=True
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
            option_dict[LAGTGT_ENCODER_NUM_CONV_BLOCKS_KEY],
            exact_dimensions=numpy.array([num_levels + 1], dtype=int)
        )
        error_checking.assert_is_integer_numpy_array(
            option_dict[LAGTGT_ENCODER_NUM_CONV_BLOCKS_KEY]
        )
        error_checking.assert_is_geq_numpy_array(
            option_dict[LAGTGT_ENCODER_NUM_CONV_BLOCKS_KEY], 1
        )

        error_checking.assert_is_numpy_array(
            option_dict[LAGTGT_ENCODER_DROPOUT_RATES_KEY],
            exact_dimensions=numpy.array([num_levels + 1], dtype=int)
        )
        error_checking.assert_is_leq_numpy_array(
            option_dict[LAGTGT_ENCODER_DROPOUT_RATES_KEY], 1., allow_nan=True
        )

        lagtgt_forecast_num_conv_blocks_by_level = option_dict[
            LAGTGT_FC_MODULE_NUM_CONV_BLOCKS_KEY
        ]
        error_checking.assert_is_numpy_array(
            lagtgt_forecast_num_conv_blocks_by_level,
            exact_dimensions=numpy.array([num_levels + 1], dtype=int)
        )
        error_checking.assert_is_integer_numpy_array(
            lagtgt_forecast_num_conv_blocks_by_level
        )
        error_checking.assert_is_greater_numpy_array(
            lagtgt_forecast_num_conv_blocks_by_level, 0
        )

        lagtgt_fc_module_drop_rate_by_level = option_dict[
            LAGTGT_FC_MODULE_DROPOUT_RATES_KEY
        ]
        error_checking.assert_is_numpy_array(
            lagtgt_fc_module_drop_rate_by_level,
            exact_dimensions=numpy.array([num_levels + 1], dtype=int)
        )
        error_checking.assert_is_leq_numpy_array(
            lagtgt_fc_module_drop_rate_by_level, 1., allow_nan=True
        )

        error_checking.assert_is_boolean(
            option_dict[LAGTGT_FC_MODULE_USE_3D_CONV]
        )

    use_recent_biases = (
        option_dict[INPUT_DIMENSIONS_2PT5KM_RCTBIAS_KEY] is not None
    )

    if use_recent_biases:
        error_checking.assert_is_numpy_array(
            option_dict[RCTBIAS_ENCODER_NUM_CHANNELS_KEY],
            exact_dimensions=numpy.array([num_levels + 1], dtype=int)
        )
        error_checking.assert_is_integer_numpy_array(
            option_dict[RCTBIAS_ENCODER_NUM_CHANNELS_KEY]
        )
        error_checking.assert_is_geq_numpy_array(
            option_dict[RCTBIAS_ENCODER_NUM_CHANNELS_KEY], 1
        )

        error_checking.assert_is_numpy_array(
            option_dict[RCTBIAS_POOLING_SIZE_KEY],
            exact_dimensions=numpy.array([num_levels], dtype=int)
        )
        error_checking.assert_is_integer_numpy_array(
            option_dict[RCTBIAS_POOLING_SIZE_KEY]
        )
        error_checking.assert_is_geq_numpy_array(
            option_dict[RCTBIAS_POOLING_SIZE_KEY], 2
        )

        error_checking.assert_is_numpy_array(
            option_dict[RCTBIAS_ENCODER_NUM_CONV_BLOCKS_KEY],
            exact_dimensions=numpy.array([num_levels + 1], dtype=int)
        )
        error_checking.assert_is_integer_numpy_array(
            option_dict[RCTBIAS_ENCODER_NUM_CONV_BLOCKS_KEY]
        )
        error_checking.assert_is_geq_numpy_array(
            option_dict[RCTBIAS_ENCODER_NUM_CONV_BLOCKS_KEY], 1
        )

        error_checking.assert_is_numpy_array(
            option_dict[RCTBIAS_ENCODER_DROPOUT_RATES_KEY],
            exact_dimensions=numpy.array([num_levels + 1], dtype=int)
        )
        error_checking.assert_is_leq_numpy_array(
            option_dict[RCTBIAS_ENCODER_DROPOUT_RATES_KEY], 1., allow_nan=True
        )

        rctbias_forecast_num_conv_blocks_by_level = option_dict[
            RCTBIAS_FC_MODULE_NUM_CONV_BLOCKS_KEY
        ]
        error_checking.assert_is_numpy_array(
            rctbias_forecast_num_conv_blocks_by_level,
            exact_dimensions=numpy.array([num_levels + 1], dtype=int)
        )
        error_checking.assert_is_integer_numpy_array(
            rctbias_forecast_num_conv_blocks_by_level
        )
        error_checking.assert_is_greater_numpy_array(
            rctbias_forecast_num_conv_blocks_by_level, 0
        )

        rctbias_fc_module_drop_rate_by_level = option_dict[
            RCTBIAS_FC_MODULE_DROPOUT_RATES_KEY
        ]
        error_checking.assert_is_numpy_array(
            rctbias_fc_module_drop_rate_by_level,
            exact_dimensions=numpy.array([num_levels + 1], dtype=int)
        )
        error_checking.assert_is_leq_numpy_array(
            rctbias_fc_module_drop_rate_by_level, 1., allow_nan=True
        )

        error_checking.assert_is_boolean(
            option_dict[RCTBIAS_FC_MODULE_USE_3D_CONV]
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
        option_dict[DECODER_NUM_CONV_BLOCKS_KEY],
        exact_dimensions=numpy.array([num_levels], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(
        option_dict[DECODER_NUM_CONV_BLOCKS_KEY]
    )
    error_checking.assert_is_geq_numpy_array(
        option_dict[DECODER_NUM_CONV_BLOCKS_KEY], 1
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
    error_checking.assert_is_integer(option_dict[ENSEMBLE_SIZE_KEY])
    error_checking.assert_is_geq(option_dict[ENSEMBLE_SIZE_KEY], 1)
    error_checking.assert_is_integer(option_dict[NUM_OUTPUT_CHANNELS_KEY])
    error_checking.assert_is_geq(option_dict[NUM_OUTPUT_CHANNELS_KEY], 1)
    error_checking.assert_is_boolean(option_dict[PREDICT_GUST_EXCESS_KEY])
    error_checking.assert_is_boolean(
        option_dict[PREDICT_DEWPOINT_DEPRESSION_KEY]
    )

    error_checking.assert_is_list(option_dict[METRIC_FUNCTIONS_KEY])

    return option_dict


def _get_2d_conv_block(
        input_layer_object, do_convnext_v2, use_spectral_norm, num_conv_layers,
        filter_size_px, num_filters,
        do_time_distributed_conv, regularizer_object, do_activation,
        dropout_rate, basic_layer_name):
    """Creates conv block for data with 2 spatial dimensions.

    L = number of conv layers

    :param input_layer_object: Input layer to block.
    :param do_convnext_v2: Boolean flag.  If True, will use version 2 of
        ConvNext.
    :param use_spectral_norm: Boolean flag.  If True, will use spectral
        normalization for every conv layer.
    :param num_conv_layers: Number of conv layers in block.
    :param filter_size_px: Filter size for conv layers.  The same filter size
        will be used in both dimensions, and the same filter size will be used
        for every conv layer.
    :param num_filters: Number of filters -- same for every conv layer.
    :param do_time_distributed_conv: Boolean flag.  If True (False), will do
        time-distributed (basic) convolution.
    :param regularizer_object: Regularizer for conv layers (instance of
        `keras.regularizers.l1_l2` or similar).
    :param do_activation: Boolean flag.  If True (False), will apply GeLU
        activation function.
    :param dropout_rate: Dropout rate for block.
    :param basic_layer_name: Basic layer name.  Each layer name will be made
        unique by adding a suffix.
    :return: output_layer_object: Output layer from block.
    """

    # Do actual stuff.
    if do_convnext_v2:
        current_layer_object = __get_2d_convnext2_block(
            input_layer_object=input_layer_object,
            num_conv_layers=1,
            filter_size_px=filter_size_px,
            num_filters=num_filters,
            do_time_distributed_conv=do_time_distributed_conv,
            regularizer_object=regularizer_object,
            use_spectral_norm=use_spectral_norm,
            do_activation=do_activation,
            dropout_rate=dropout_rate,
            basic_layer_name=(
                basic_layer_name if num_conv_layers == 1
                else basic_layer_name + '_0'
            )
        )
    else:
        current_layer_object = __get_2d_convnext_block(
            input_layer_object=input_layer_object,
            num_conv_layers=1,
            filter_size_px=filter_size_px,
            num_filters=num_filters,
            do_time_distributed_conv=do_time_distributed_conv,
            regularizer_object=regularizer_object,
            use_spectral_norm=use_spectral_norm,
            do_activation=do_activation,
            dropout_rate=dropout_rate,
            basic_layer_name=(
                basic_layer_name if num_conv_layers == 1
                else basic_layer_name + '_0'
            )
        )

    for i in range(num_conv_layers):
        if i == 0:
            continue

        if do_convnext_v2:
            current_layer_object = __get_2d_convnext2_block(
                input_layer_object=current_layer_object,
                num_conv_layers=1,
                filter_size_px=filter_size_px,
                num_filters=num_filters,
                do_time_distributed_conv=do_time_distributed_conv,
                regularizer_object=regularizer_object,
                use_spectral_norm=use_spectral_norm,
                do_activation=do_activation,
                dropout_rate=dropout_rate,
                basic_layer_name='{0:s}_{1:d}'.format(basic_layer_name, i)
            )
        else:
            current_layer_object = __get_2d_convnext_block(
                input_layer_object=current_layer_object,
                num_conv_layers=1,
                filter_size_px=filter_size_px,
                num_filters=num_filters,
                do_time_distributed_conv=do_time_distributed_conv,
                regularizer_object=regularizer_object,
                use_spectral_norm=use_spectral_norm,
                do_activation=do_activation,
                dropout_rate=dropout_rate,
                basic_layer_name='{0:s}_{1:d}'.format(basic_layer_name, i)
            )

    return current_layer_object


def _get_3d_conv_block(
        input_layer_object, num_time_steps, do_convnext_v2, use_spectral_norm,
        num_conv_layers, filter_size_px, regularizer_object, do_activation,
        dropout_rate, basic_layer_name):
    """Creates conv block for data with 2 spatial dimensions.

    :param input_layer_object: Input layer to block (with 3 spatial dims).
    :param num_time_steps: Number of time steps expected in input.
    :param do_convnext_v2: See documentation for `_get_2d_conv_block`.
    :param use_spectral_norm: Same.
    :param num_conv_layers: Same.
    :param filter_size_px: Same.
    :param regularizer_object: Same.
    :param do_activation: Same.
    :param dropout_rate: Dropout rate for block.
    :param basic_layer_name: Same.
    :return: output_layer_object: Output layer from block (with 2 spatial dims).
    """

    if do_convnext_v2:
        current_layer_object = __get_3d_convnext2_block(
            input_layer_object=input_layer_object,
            num_time_steps=num_time_steps,
            num_conv_layers=1,
            filter_size_px=filter_size_px,
            regularizer_object=regularizer_object,
            use_spectral_norm=use_spectral_norm,
            do_activation=do_activation,
            dropout_rate=dropout_rate,
            basic_layer_name=(
                basic_layer_name if num_conv_layers == 1
                else basic_layer_name + '_0'
            )
        )
    else:
        current_layer_object = __get_3d_convnext_block(
            input_layer_object=input_layer_object,
            num_time_steps=num_time_steps,
            num_conv_layers=1,
            filter_size_px=filter_size_px,
            regularizer_object=regularizer_object,
            use_spectral_norm=use_spectral_norm,
            do_activation=do_activation,
            dropout_rate=dropout_rate,
            basic_layer_name=(
                basic_layer_name if num_conv_layers == 1
                else basic_layer_name + '_0'
            )
        )

    for i in range(num_conv_layers):
        if i == 0:
            continue

        if do_convnext_v2:
            current_layer_object = __get_2d_convnext2_block(
                input_layer_object=current_layer_object,
                num_conv_layers=1,
                filter_size_px=filter_size_px,
                num_filters=current_layer_object.shape[-1],
                do_time_distributed_conv=False,
                regularizer_object=regularizer_object,
                use_spectral_norm=use_spectral_norm,
                do_activation=do_activation,
                dropout_rate=dropout_rate,
                basic_layer_name='{0:s}_{1:d}'.format(basic_layer_name, i)
            )
        else:
            current_layer_object = __get_2d_convnext_block(
                input_layer_object=current_layer_object,
                num_conv_layers=1,
                filter_size_px=filter_size_px,
                num_filters=current_layer_object.shape[-1],
                do_time_distributed_conv=False,
                regularizer_object=regularizer_object,
                use_spectral_norm=use_spectral_norm,
                do_activation=do_activation,
                dropout_rate=dropout_rate,
                basic_layer_name='{0:s}_{1:d}'.format(basic_layer_name, i)
            )

    return current_layer_object


def create_model(option_dict):
    """Creates CNN.

    :param option_dict: See documentation for `check_input_args`.
    :return: model_object: Untrained (but compiled) instance of
        `keras.models.Model`.
    """

    option_dict = _check_input_args(option_dict)
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
    do_convnext_v2 = optd[DO_CONVNEXT_V2_KEY]
    use_spectral_norm = optd[USE_SPECTRAL_NORM_KEY]

    nwp_encoder_num_channels_by_level = optd[NWP_ENCODER_NUM_CHANNELS_KEY]
    nwp_pooling_size_by_level_px = optd[NWP_POOLING_SIZE_KEY]
    nwp_encoder_num_conv_blocks_by_level = optd[NWP_ENCODER_NUM_CONV_BLOCKS_KEY]
    nwp_encoder_drop_rate_by_level = optd[NWP_ENCODER_DROPOUT_RATES_KEY]
    nwp_forecast_num_conv_blocks_by_level = optd[
        NWP_FC_MODULE_NUM_CONV_BLOCKS_KEY
    ]
    nwp_forecast_module_drop_rate_by_level = optd[
        NWP_FC_MODULE_DROPOUT_RATES_KEY
    ]
    nwp_forecast_module_use_3d_conv = optd[NWP_FC_MODULE_USE_3D_CONV]

    lagtgt_encoder_num_channels_by_level = optd[LAGTGT_ENCODER_NUM_CHANNELS_KEY]
    lagtgt_pooling_size_by_level_px = optd[LAGTGT_POOLING_SIZE_KEY]
    lagtgt_encoder_num_conv_blocks_by_level = optd[
        LAGTGT_ENCODER_NUM_CONV_BLOCKS_KEY
    ]
    lagtgt_encoder_drop_rate_by_level = optd[
        LAGTGT_ENCODER_DROPOUT_RATES_KEY
    ]
    lagtgt_forecast_num_conv_blocks_by_level = optd[
        LAGTGT_FC_MODULE_NUM_CONV_BLOCKS_KEY
    ]
    lagtgt_forecast_module_drop_rate_by_level = optd[
        LAGTGT_FC_MODULE_DROPOUT_RATES_KEY
    ]
    lagtgt_forecast_module_use_3d_conv = optd[LAGTGT_FC_MODULE_USE_3D_CONV]

    rctbias_encoder_num_channels_by_level = optd[
        RCTBIAS_ENCODER_NUM_CHANNELS_KEY
    ]
    rctbias_pooling_size_by_level_px = optd[RCTBIAS_POOLING_SIZE_KEY]
    rctbias_encoder_num_conv_blocks_by_level = optd[
        RCTBIAS_ENCODER_NUM_CONV_BLOCKS_KEY
    ]
    rctbias_encoder_drop_rate_by_level = optd[
        RCTBIAS_ENCODER_DROPOUT_RATES_KEY
    ]
    rctbias_forecast_num_conv_blocks_by_level = optd[
        RCTBIAS_FC_MODULE_NUM_CONV_BLOCKS_KEY
    ]
    rctbias_forecast_module_drop_rate_by_level = optd[
        RCTBIAS_FC_MODULE_DROPOUT_RATES_KEY
    ]
    rctbias_forecast_module_use_3d_conv = optd[RCTBIAS_FC_MODULE_USE_3D_CONV]

    decoder_num_channels_by_level = optd[DECODER_NUM_CHANNELS_KEY]
    num_decoder_conv_blocks_by_level = optd[DECODER_NUM_CONV_BLOCKS_KEY]
    upsampling_drop_rate_by_level = optd[UPSAMPLING_DROPOUT_RATES_KEY]
    skip_drop_rate_by_level = optd[SKIP_DROPOUT_RATES_KEY]

    include_penultimate_conv = optd[INCLUDE_PENULTIMATE_KEY]
    penultimate_conv_dropout_rate = optd[PENULTIMATE_DROPOUT_RATE_KEY]
    output_activ_function_name = optd[OUTPUT_ACTIV_FUNCTION_KEY]
    output_activ_function_alpha = optd[OUTPUT_ACTIV_FUNCTION_ALPHA_KEY]
    l1_weight = optd[L1_WEIGHT_KEY]
    l2_weight = optd[L2_WEIGHT_KEY]
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

        for j in range(nwp_encoder_num_conv_blocks_by_level[i]):
            if j == 0:
                nwp_encoder_conv_layer_objects[i] = _get_2d_conv_block(
                    input_layer_object=this_input_layer_object,
                    do_convnext_v2=do_convnext_v2,
                    use_spectral_norm=use_spectral_norm,
                    num_conv_layers=1,
                    filter_size_px=7,
                    num_filters=nwp_encoder_num_channels_by_level[i],
                    do_time_distributed_conv=True,
                    regularizer_object=regularizer_object,
                    do_activation=True,
                    dropout_rate=nwp_encoder_drop_rate_by_level[i],
                    basic_layer_name='nwp_encoder_level{0:d}-{1:d}'.format(i, j)
                )
            else:
                nwp_encoder_conv_layer_objects[i] = _get_2d_conv_block(
                    input_layer_object=nwp_encoder_conv_layer_objects[i],
                    do_convnext_v2=do_convnext_v2,
                    use_spectral_norm=use_spectral_norm,
                    num_conv_layers=1,
                    filter_size_px=7,
                    num_filters=nwp_encoder_num_channels_by_level[i],
                    do_time_distributed_conv=True,
                    regularizer_object=regularizer_object,
                    do_activation=True,
                    dropout_rate=nwp_encoder_drop_rate_by_level[i],
                    basic_layer_name='nwp_encoder_level{0:d}-{1:d}'.format(i, j)
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

        for j in range(rctbias_encoder_num_conv_blocks_by_level[i]):
            if j == 0:
                rctbias_encoder_conv_layer_objects[i] = _get_2d_conv_block(
                    input_layer_object=this_input_layer_object,
                    do_convnext_v2=do_convnext_v2,
                    use_spectral_norm=use_spectral_norm,
                    num_conv_layers=1,
                    filter_size_px=7,
                    num_filters=rctbias_encoder_num_channels_by_level[i],
                    do_time_distributed_conv=True,
                    regularizer_object=regularizer_object,
                    do_activation=True,
                    dropout_rate=rctbias_encoder_drop_rate_by_level[i],
                    basic_layer_name=
                    'rctbias_encoder_level{0:d}-{1:d}'.format(i, j)
                )
            else:
                rctbias_encoder_conv_layer_objects[i] = _get_2d_conv_block(
                    input_layer_object=rctbias_encoder_conv_layer_objects[i],
                    do_convnext_v2=do_convnext_v2,
                    use_spectral_norm=use_spectral_norm,
                    num_conv_layers=1,
                    filter_size_px=7,
                    num_filters=rctbias_encoder_num_channels_by_level[i],
                    do_time_distributed_conv=True,
                    regularizer_object=regularizer_object,
                    do_activation=True,
                    dropout_rate=rctbias_encoder_drop_rate_by_level[i],
                    basic_layer_name=
                    'rctbias_encoder_level{0:d}-{1:d}'.format(i, j)
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
        this_layer_object = chiu_net_pp_arch.crop_layer(
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
            this_layer_object = chiu_net_pp_arch.crop_layer(
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

            for j in range(nwp_encoder_num_conv_blocks_by_level[i]):
                if j == 0:
                    nwp_encoder_conv_layer_objects[i] = _get_2d_conv_block(
                        input_layer_object=this_input_layer_object,
                        do_convnext_v2=do_convnext_v2,
                        use_spectral_norm=use_spectral_norm,
                        num_conv_layers=1,
                        filter_size_px=7,
                        num_filters=nwp_encoder_num_channels_by_level[i],
                        do_time_distributed_conv=True,
                        regularizer_object=regularizer_object,
                        do_activation=True,
                        dropout_rate=nwp_encoder_drop_rate_by_level[i],
                        basic_layer_name=
                        'nwp_encoder_level{0:d}-{1:d}'.format(i, j)
                    )
                else:
                    nwp_encoder_conv_layer_objects[i] = _get_2d_conv_block(
                        input_layer_object=nwp_encoder_conv_layer_objects[i],
                        do_convnext_v2=do_convnext_v2,
                        use_spectral_norm=use_spectral_norm,
                        num_conv_layers=1,
                        filter_size_px=7,
                        num_filters=nwp_encoder_num_channels_by_level[i],
                        do_time_distributed_conv=True,
                        regularizer_object=regularizer_object,
                        do_activation=True,
                        dropout_rate=nwp_encoder_drop_rate_by_level[i],
                        basic_layer_name=
                        'nwp_encoder_level{0:d}-{1:d}'.format(i, j)
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

            for j in range(rctbias_encoder_num_conv_blocks_by_level[i]):
                if j == 0:
                    rctbias_encoder_conv_layer_objects[i] = _get_2d_conv_block(
                        input_layer_object=this_input_layer_object,
                        do_convnext_v2=do_convnext_v2,
                        use_spectral_norm=use_spectral_norm,
                        num_conv_layers=1,
                        filter_size_px=7,
                        num_filters=rctbias_encoder_num_channels_by_level[i],
                        do_time_distributed_conv=True,
                        regularizer_object=regularizer_object,
                        do_activation=True,
                        dropout_rate=rctbias_encoder_drop_rate_by_level[i],
                        basic_layer_name=
                        'rctbias_encoder_level{0:d}-{1:d}'.format(i, j)
                    )
                else:
                    rctbias_encoder_conv_layer_objects[i] = _get_2d_conv_block(
                        input_layer_object=
                        rctbias_encoder_conv_layer_objects[i],
                        do_convnext_v2=do_convnext_v2,
                        use_spectral_norm=use_spectral_norm,
                        num_conv_layers=1,
                        filter_size_px=7,
                        num_filters=rctbias_encoder_num_channels_by_level[i],
                        do_time_distributed_conv=True,
                        regularizer_object=regularizer_object,
                        do_activation=True,
                        dropout_rate=rctbias_encoder_drop_rate_by_level[i],
                        basic_layer_name=
                        'rctbias_encoder_level{0:d}-{1:d}'.format(i, j)
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
        this_layer_object = chiu_net_pp_arch.crop_layer(
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
            this_layer_object = chiu_net_pp_arch.crop_layer(
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

        for j in range(nwp_encoder_num_conv_blocks_by_level[i]):
            if j == 0:
                nwp_encoder_conv_layer_objects[i] = _get_2d_conv_block(
                    input_layer_object=this_input_layer_object,
                    do_convnext_v2=do_convnext_v2,
                    use_spectral_norm=use_spectral_norm,
                    num_conv_layers=1,
                    filter_size_px=7,
                    num_filters=nwp_encoder_num_channels_by_level[i],
                    do_time_distributed_conv=True,
                    regularizer_object=regularizer_object,
                    do_activation=True,
                    dropout_rate=nwp_encoder_drop_rate_by_level[i],
                    basic_layer_name=
                    'nwp_encoder_level{0:d}-{1:d}'.format(i, j)
                )
            else:
                nwp_encoder_conv_layer_objects[i] = _get_2d_conv_block(
                    input_layer_object=nwp_encoder_conv_layer_objects[i],
                    do_convnext_v2=do_convnext_v2,
                    use_spectral_norm=use_spectral_norm,
                    num_conv_layers=1,
                    filter_size_px=7,
                    num_filters=nwp_encoder_num_channels_by_level[i],
                    do_time_distributed_conv=True,
                    regularizer_object=regularizer_object,
                    do_activation=True,
                    dropout_rate=nwp_encoder_drop_rate_by_level[i],
                    basic_layer_name=
                    'nwp_encoder_level{0:d}-{1:d}'.format(i, j)
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

            for j in range(rctbias_encoder_num_conv_blocks_by_level[i]):
                if j == 0:
                    rctbias_encoder_conv_layer_objects[i] = _get_2d_conv_block(
                        input_layer_object=this_input_layer_object,
                        do_convnext_v2=do_convnext_v2,
                        use_spectral_norm=use_spectral_norm,
                        num_conv_layers=1,
                        filter_size_px=7,
                        num_filters=rctbias_encoder_num_channels_by_level[i],
                        do_time_distributed_conv=True,
                        regularizer_object=regularizer_object,
                        do_activation=True,
                        dropout_rate=rctbias_encoder_drop_rate_by_level[i],
                        basic_layer_name=
                        'rctbias_encoder_level{0:d}-{1:d}'.format(i, j)
                    )
                else:
                    rctbias_encoder_conv_layer_objects[i] = _get_2d_conv_block(
                        input_layer_object=
                        rctbias_encoder_conv_layer_objects[i],
                        do_convnext_v2=do_convnext_v2,
                        use_spectral_norm=use_spectral_norm,
                        num_conv_layers=1,
                        filter_size_px=7,
                        num_filters=rctbias_encoder_num_channels_by_level[i],
                        do_time_distributed_conv=True,
                        regularizer_object=regularizer_object,
                        do_activation=True,
                        dropout_rate=rctbias_encoder_drop_rate_by_level[i],
                        basic_layer_name=
                        'rctbias_encoder_level{0:d}-{1:d}'.format(i, j)
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
        this_layer_object = chiu_net_pp_arch.crop_layer(
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
            this_layer_object = chiu_net_pp_arch.crop_layer(
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

        for j in range(nwp_encoder_num_conv_blocks_by_level[i]):
            if j == 0:
                nwp_encoder_conv_layer_objects[i] = _get_2d_conv_block(
                    input_layer_object=this_input_layer_object,
                    do_convnext_v2=do_convnext_v2,
                    use_spectral_norm=use_spectral_norm,
                    num_conv_layers=1,
                    filter_size_px=7,
                    num_filters=nwp_encoder_num_channels_by_level[i],
                    do_time_distributed_conv=True,
                    regularizer_object=regularizer_object,
                    do_activation=True,
                    dropout_rate=nwp_encoder_drop_rate_by_level[i],
                    basic_layer_name=
                    'nwp_encoder_level{0:d}-{1:d}'.format(i, j)
                )
            else:
                nwp_encoder_conv_layer_objects[i] = _get_2d_conv_block(
                    input_layer_object=nwp_encoder_conv_layer_objects[i],
                    do_convnext_v2=do_convnext_v2,
                    use_spectral_norm=use_spectral_norm,
                    num_conv_layers=1,
                    filter_size_px=7,
                    num_filters=nwp_encoder_num_channels_by_level[i],
                    do_time_distributed_conv=True,
                    regularizer_object=regularizer_object,
                    do_activation=True,
                    dropout_rate=nwp_encoder_drop_rate_by_level[i],
                    basic_layer_name=
                    'nwp_encoder_level{0:d}-{1:d}'.format(i, j)
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

            for j in range(rctbias_encoder_num_conv_blocks_by_level[i]):
                if j == 0:
                    rctbias_encoder_conv_layer_objects[i] = _get_2d_conv_block(
                        input_layer_object=this_input_layer_object,
                        do_convnext_v2=do_convnext_v2,
                        use_spectral_norm=use_spectral_norm,
                        num_conv_layers=1,
                        filter_size_px=7,
                        num_filters=rctbias_encoder_num_channels_by_level[i],
                        do_time_distributed_conv=True,
                        regularizer_object=regularizer_object,
                        do_activation=True,
                        dropout_rate=rctbias_encoder_drop_rate_by_level[i],
                        basic_layer_name=
                        'rctbias_encoder_level{0:d}-{1:d}'.format(i, j)
                    )
                else:
                    rctbias_encoder_conv_layer_objects[i] = _get_2d_conv_block(
                        input_layer_object=
                        rctbias_encoder_conv_layer_objects[i],
                        do_convnext_v2=do_convnext_v2,
                        use_spectral_norm=use_spectral_norm,
                        num_conv_layers=1,
                        filter_size_px=7,
                        num_filters=rctbias_encoder_num_channels_by_level[i],
                        do_time_distributed_conv=True,
                        regularizer_object=regularizer_object,
                        do_activation=True,
                        dropout_rate=rctbias_encoder_drop_rate_by_level[i],
                        basic_layer_name=
                        'rctbias_encoder_level{0:d}-{1:d}'.format(i, j)
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
        for j in range(nwp_encoder_num_conv_blocks_by_level[i]):
            if j == 0:
                nwp_encoder_conv_layer_objects[i] = _get_2d_conv_block(
                    input_layer_object=nwp_encoder_pooling_layer_objects[i - 1],
                    do_convnext_v2=do_convnext_v2,
                    use_spectral_norm=use_spectral_norm,
                    num_conv_layers=1,
                    filter_size_px=7,
                    num_filters=nwp_encoder_num_channels_by_level[i],
                    do_time_distributed_conv=True,
                    regularizer_object=regularizer_object,
                    do_activation=True,
                    dropout_rate=nwp_encoder_drop_rate_by_level[i],
                    basic_layer_name=
                    'nwp_encoder_level{0:d}-{1:d}'.format(i, j)
                )
            else:
                nwp_encoder_conv_layer_objects[i] = _get_2d_conv_block(
                    input_layer_object=nwp_encoder_conv_layer_objects[i],
                    do_convnext_v2=do_convnext_v2,
                    use_spectral_norm=use_spectral_norm,
                    num_conv_layers=1,
                    filter_size_px=7,
                    num_filters=nwp_encoder_num_channels_by_level[i],
                    do_time_distributed_conv=True,
                    regularizer_object=regularizer_object,
                    do_activation=True,
                    dropout_rate=nwp_encoder_drop_rate_by_level[i],
                    basic_layer_name=
                    'nwp_encoder_level{0:d}-{1:d}'.format(i, j)
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

        for j in range(rctbias_encoder_num_conv_blocks_by_level[i]):
            if j == 0:
                rctbias_encoder_conv_layer_objects[i] = _get_2d_conv_block(
                    input_layer_object=
                    rctbias_encoder_pooling_layer_objects[i - 1],
                    do_convnext_v2=do_convnext_v2,
                    use_spectral_norm=use_spectral_norm,
                    num_conv_layers=1,
                    filter_size_px=7,
                    num_filters=rctbias_encoder_num_channels_by_level[i],
                    do_time_distributed_conv=True,
                    regularizer_object=regularizer_object,
                    do_activation=True,
                    dropout_rate=rctbias_encoder_drop_rate_by_level[i],
                    basic_layer_name=
                    'rctbias_encoder_level{0:d}-{1:d}'.format(i, j)
                )
            else:
                rctbias_encoder_conv_layer_objects[i] = _get_2d_conv_block(
                    input_layer_object=
                    rctbias_encoder_conv_layer_objects[i],
                    do_convnext_v2=do_convnext_v2,
                    use_spectral_norm=use_spectral_norm,
                    num_conv_layers=1,
                    filter_size_px=7,
                    num_filters=rctbias_encoder_num_channels_by_level[i],
                    do_time_distributed_conv=True,
                    regularizer_object=regularizer_object,
                    do_activation=True,
                    dropout_rate=rctbias_encoder_drop_rate_by_level[i],
                    basic_layer_name=
                    'rctbias_encoder_level{0:d}-{1:d}'.format(i, j)
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
            for j in range(nwp_forecast_num_conv_blocks_by_level[i]):
                if j == 0:
                    nwp_fcst_module_layer_objects[i] = _get_3d_conv_block(
                        input_layer_object=nwp_fcst_module_layer_objects[i],
                        num_time_steps=input_dimensions_2pt5km_res[-2],
                        do_convnext_v2=do_convnext_v2,
                        use_spectral_norm=use_spectral_norm,
                        num_conv_layers=2,
                        filter_size_px=1,
                        regularizer_object=regularizer_object,
                        do_activation=True,
                        dropout_rate=nwp_forecast_module_drop_rate_by_level[i],
                        basic_layer_name=
                        'nwp_fcst_level{0:d}-{1:d}'.format(i, j)
                    )
                else:
                    nwp_fcst_module_layer_objects[i] = _get_2d_conv_block(
                        input_layer_object=nwp_fcst_module_layer_objects[i],
                        do_convnext_v2=do_convnext_v2,
                        use_spectral_norm=use_spectral_norm,
                        num_conv_layers=1,
                        filter_size_px=1,
                        num_filters=nwp_encoder_num_channels_by_level[i],
                        do_time_distributed_conv=False,
                        regularizer_object=regularizer_object,
                        do_activation=True,
                        dropout_rate=nwp_forecast_module_drop_rate_by_level[i],
                        basic_layer_name=
                        'nwp_fcst_level{0:d}-{1:d}'.format(i, j)
                    )
        else:
            orig_dims = nwp_fcst_module_layer_objects[i].shape
            new_dims = orig_dims[1:-2] + (orig_dims[-2] * orig_dims[-1],)

            this_name = 'nwp_fcst_level{0:d}_remove-time-dim'.format(i)
            nwp_fcst_module_layer_objects[i] = keras.layers.Reshape(
                target_shape=new_dims, name=this_name
            )(nwp_fcst_module_layer_objects[i])

            for j in range(nwp_forecast_num_conv_blocks_by_level[i]):
                nwp_fcst_module_layer_objects[i] = _get_2d_conv_block(
                    input_layer_object=nwp_fcst_module_layer_objects[i],
                    do_convnext_v2=do_convnext_v2,
                    use_spectral_norm=use_spectral_norm,
                    num_conv_layers=1,
                    filter_size_px=1,
                    num_filters=nwp_encoder_num_channels_by_level[i],
                    do_time_distributed_conv=False,
                    regularizer_object=regularizer_object,
                    do_activation=True,
                    dropout_rate=nwp_forecast_module_drop_rate_by_level[i],
                    basic_layer_name=
                    'nwp_fcst_level{0:d}-{1:d}'.format(i, j)
                )

        if not use_recent_biases:
            continue

        this_name = 'rctbias_fcst_level{0:d}_put-time-last'.format(i)
        rctbias_fcst_module_layer_objects[i] = keras.layers.Permute(
            dims=(2, 3, 1, 4), name=this_name
        )(rctbias_encoder_conv_layer_objects[i])

        if rctbias_forecast_module_use_3d_conv:
            for j in range(rctbias_forecast_num_conv_blocks_by_level[i]):
                if j == 0:
                    rctbias_fcst_module_layer_objects[i] = _get_3d_conv_block(
                        input_layer_object=rctbias_fcst_module_layer_objects[i],
                        num_time_steps=input_dimensions_2pt5km_rctbias[-2],
                        do_convnext_v2=do_convnext_v2,
                        use_spectral_norm=use_spectral_norm,
                        num_conv_layers=2,
                        filter_size_px=1,
                        regularizer_object=regularizer_object,
                        do_activation=True,
                        dropout_rate=
                        rctbias_forecast_module_drop_rate_by_level[i],
                        basic_layer_name=
                        'rctbias_fcst_level{0:d}-{1:d}'.format(i, j)
                    )
                else:
                    rctbias_fcst_module_layer_objects[i] = _get_2d_conv_block(
                        input_layer_object=rctbias_fcst_module_layer_objects[i],
                        do_convnext_v2=do_convnext_v2,
                        use_spectral_norm=use_spectral_norm,
                        num_conv_layers=1,
                        filter_size_px=1,
                        num_filters=rctbias_encoder_num_channels_by_level[i],
                        do_time_distributed_conv=False,
                        regularizer_object=regularizer_object,
                        do_activation=True,
                        dropout_rate=
                        rctbias_forecast_module_drop_rate_by_level[i],
                        basic_layer_name=
                        'rctbias_fcst_level{0:d}-{1:d}'.format(i, j)
                    )
        else:
            orig_dims = rctbias_fcst_module_layer_objects[i].shape
            new_dims = orig_dims[1:-2] + (orig_dims[-2] * orig_dims[-1],)

            this_name = 'rctbias_fcst_level{0:d}_remove-time-dim'.format(i)
            rctbias_fcst_module_layer_objects[i] = keras.layers.Reshape(
                target_shape=new_dims, name=this_name
            )(rctbias_fcst_module_layer_objects[i])

            for j in range(rctbias_forecast_num_conv_blocks_by_level[i]):
                rctbias_fcst_module_layer_objects[i] = _get_2d_conv_block(
                    input_layer_object=rctbias_fcst_module_layer_objects[i],
                    do_convnext_v2=do_convnext_v2,
                    use_spectral_norm=use_spectral_norm,
                    num_conv_layers=1,
                    filter_size_px=1,
                    num_filters=rctbias_encoder_num_channels_by_level[i],
                    do_time_distributed_conv=False,
                    regularizer_object=regularizer_object,
                    do_activation=True,
                    dropout_rate=rctbias_forecast_module_drop_rate_by_level[i],
                    basic_layer_name=
                    'rctbias_fcst_level{0:d}-{1:d}'.format(i, j)
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

        for j in range(lagtgt_encoder_num_conv_blocks_by_level[i]):
            if j == 0:
                lagtgt_encoder_conv_layer_objects[i] = _get_2d_conv_block(
                    input_layer_object=this_input_layer_object,
                    do_convnext_v2=do_convnext_v2,
                    use_spectral_norm=use_spectral_norm,
                    num_conv_layers=1,
                    filter_size_px=7,
                    num_filters=lagtgt_encoder_num_channels_by_level[i],
                    do_time_distributed_conv=True,
                    regularizer_object=regularizer_object,
                    do_activation=True,
                    dropout_rate=lagtgt_encoder_drop_rate_by_level[i],
                    basic_layer_name=
                    'lagtgt_encoder_level{0:d}-{1:d}'.format(i, j)
                )
            else:
                lagtgt_encoder_conv_layer_objects[i] = _get_2d_conv_block(
                    input_layer_object=lagtgt_encoder_conv_layer_objects[i],
                    do_convnext_v2=do_convnext_v2,
                    use_spectral_norm=use_spectral_norm,
                    num_conv_layers=1,
                    filter_size_px=7,
                    num_filters=lagtgt_encoder_num_channels_by_level[i],
                    do_time_distributed_conv=True,
                    regularizer_object=regularizer_object,
                    do_activation=True,
                    dropout_rate=lagtgt_encoder_drop_rate_by_level[i],
                    basic_layer_name=
                    'lagtgt_encoder_level{0:d}-{1:d}'.format(i, j)
                )

        this_name = 'lagtgt_fcst_level{0:d}_put-time-last'.format(i)
        lagtgt_fcst_module_layer_objects[i] = keras.layers.Permute(
            dims=(2, 3, 1, 4), name=this_name
        )(lagtgt_encoder_conv_layer_objects[i])

        if lagtgt_forecast_module_use_3d_conv:
            for j in range(lagtgt_forecast_num_conv_blocks_by_level[i]):
                if j == 0:
                    lagtgt_fcst_module_layer_objects[i] = _get_3d_conv_block(
                        input_layer_object=lagtgt_fcst_module_layer_objects[i],
                        num_time_steps=input_dimensions_lagged_targets[-2],
                        do_convnext_v2=do_convnext_v2,
                        use_spectral_norm=use_spectral_norm,
                        num_conv_layers=2,
                        filter_size_px=1,
                        regularizer_object=regularizer_object,
                        do_activation=True,
                        dropout_rate=
                        lagtgt_forecast_module_drop_rate_by_level[i],
                        basic_layer_name=
                        'lagtgt_fcst_level{0:d}-{1:d}'.format(i, j)
                    )
                else:
                    lagtgt_fcst_module_layer_objects[i] = _get_2d_conv_block(
                        input_layer_object=lagtgt_fcst_module_layer_objects[i],
                        do_convnext_v2=do_convnext_v2,
                        use_spectral_norm=use_spectral_norm,
                        num_conv_layers=1,
                        filter_size_px=1,
                        num_filters=lagtgt_encoder_num_channels_by_level[i],
                        do_time_distributed_conv=False,
                        regularizer_object=regularizer_object,
                        do_activation=True,
                        dropout_rate=
                        lagtgt_forecast_module_drop_rate_by_level[i],
                        basic_layer_name=
                        'lagtgt_fcst_level{0:d}-{1:d}'.format(i, j)
                    )
        else:
            orig_dims = lagtgt_fcst_module_layer_objects[i].shape
            new_dims = orig_dims[1:-2] + (orig_dims[-2] * orig_dims[-1],)

            this_name = 'lagtgt_fcst_level{0:d}_remove-time-dim'.format(i)
            lagtgt_fcst_module_layer_objects[i] = keras.layers.Reshape(
                target_shape=new_dims, name=this_name
            )(lagtgt_fcst_module_layer_objects[i])

            for j in range(lagtgt_forecast_num_conv_blocks_by_level[i]):
                lagtgt_fcst_module_layer_objects[i] = _get_2d_conv_block(
                    input_layer_object=lagtgt_fcst_module_layer_objects[i],
                    do_convnext_v2=do_convnext_v2,
                    use_spectral_norm=use_spectral_norm,
                    num_conv_layers=1,
                    filter_size_px=1,
                    num_filters=lagtgt_encoder_num_channels_by_level[i],
                    do_time_distributed_conv=False,
                    regularizer_object=regularizer_object,
                    do_activation=True,
                    dropout_rate=lagtgt_forecast_module_drop_rate_by_level[i],
                    basic_layer_name=
                    'lagtgt_fcst_level{0:d}-{1:d}'.format(i, j)
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

            this_layer_object = chiu_net_pp_arch.pad_layer(
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
                do_convnext_v2=do_convnext_v2,
                use_spectral_norm=use_spectral_norm,
                num_conv_layers=1,
                filter_size_px=7,
                num_filters=this_num_channels,
                do_time_distributed_conv=False,
                regularizer_object=regularizer_object,
                do_activation=True,
                dropout_rate=upsampling_drop_rate_by_level[i_new],
                basic_layer_name='block{0:d}-{1:d}_up'.format(i_new, j)
            )

            last_conv_layer_matrix[i_new, j] = (
                chiu_net_pp_arch.create_skip_connection(
                    input_layer_objects=
                    last_conv_layer_matrix[i_new, :(j + 1)].tolist(),
                    num_output_channels=decoder_num_channels_by_level[i_new],
                    current_level_num=i_new,
                    regularizer_object=regularizer_object
                )
            )

            for k in range(num_decoder_conv_blocks_by_level[i_new]):
                last_conv_layer_matrix[i_new, j] = _get_2d_conv_block(
                    input_layer_object=last_conv_layer_matrix[i_new, j],
                    do_convnext_v2=do_convnext_v2,
                    use_spectral_norm=use_spectral_norm,
                    num_conv_layers=1,
                    filter_size_px=7,
                    num_filters=decoder_num_channels_by_level[i_new],
                    do_time_distributed_conv=False,
                    regularizer_object=regularizer_object,
                    do_activation=True,
                    dropout_rate=skip_drop_rate_by_level[i_new],
                    basic_layer_name=
                    'block{0:d}-{1:d}_skip_{2:d}'.format(i_new, j, k)
                )

    if include_penultimate_conv:
        last_conv_layer_matrix[0, -1] = _get_2d_conv_block(
            input_layer_object=last_conv_layer_matrix[0, -1],
            do_convnext_v2=do_convnext_v2,
            use_spectral_norm=use_spectral_norm,
            num_conv_layers=1,
            filter_size_px=7,
            num_filters=2 * num_output_channels * ensemble_size,
            do_time_distributed_conv=False,
            regularizer_object=regularizer_object,
            do_activation=True,
            dropout_rate=penultimate_conv_dropout_rate,
            basic_layer_name='penultimate'
        )

    num_constrained_output_channels = (
        int(predict_gust_excess) + int(predict_dewpoint_depression)
    )
    do_residual_prediction = input_dimensions_predn_baseline is not None

    simple_output_layer_object = _get_2d_conv_block(
        input_layer_object=last_conv_layer_matrix[0, -1],
        do_convnext_v2=do_convnext_v2,
        use_spectral_norm=use_spectral_norm,
        num_conv_layers=1,
        filter_size_px=1,
        num_filters=(
            (num_output_channels - num_constrained_output_channels) *
            ensemble_size
        ),
        do_time_distributed_conv=False,
        regularizer_object=regularizer_object,
        do_activation=False,
        dropout_rate=-1.,
        basic_layer_name='last_conv_simple'
    )

    if not do_residual_prediction:
        simple_output_layer_object = architecture_utils.get_activation_layer(
            activation_function_string=output_activ_function_name,
            alpha_for_relu=output_activ_function_alpha,
            alpha_for_elu=output_activ_function_alpha,
            layer_name='last_conv_simple_activ1'
        )(simple_output_layer_object)

    if predict_dewpoint_depression:
        dd_output_layer_object = _get_2d_conv_block(
            input_layer_object=last_conv_layer_matrix[0, -1],
            do_convnext_v2=do_convnext_v2,
            use_spectral_norm=use_spectral_norm,
            num_conv_layers=1,
            filter_size_px=1,
            num_filters=ensemble_size,
            do_time_distributed_conv=False,
            regularizer_object=regularizer_object,
            do_activation=False,
            dropout_rate=-1.,
            basic_layer_name='last_conv_dd'
        )

        if not do_residual_prediction:
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
            do_convnext_v2=do_convnext_v2,
            use_spectral_norm=use_spectral_norm,
            num_conv_layers=1,
            filter_size_px=1,
            num_filters=ensemble_size,
            do_time_distributed_conv=False,
            regularizer_object=regularizer_object,
            do_activation=False,
            dropout_rate=-1.,
            basic_layer_name='last_conv_gex'
        )

        if not do_residual_prediction:
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
