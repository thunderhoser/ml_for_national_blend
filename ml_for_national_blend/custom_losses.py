"""Custom loss functions."""

import os
import sys
import numpy
import tensorflow
from tensorflow.keras import backend as K

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking


def _log2(input_tensor):
    """Computes logarithm in base 2.

    :param input_tensor: Keras tensor.
    :return: logarithm_tensor: Keras tensor with the same shape as
        `input_tensor`.
    """

    return (
        K.log(K.maximum(input_tensor, 1e-6)) /
        K.log(tensorflow.Variable(2., dtype=tensorflow.float64))
    )


def mean_squared_error(function_name, expect_ensemble=True, test_mode=False):
    """Creates mean squared error (MSE) loss function.

    :param function_name: Function name (string).
    :param expect_ensemble: Boolean flag.  If True, will expect
        prediction_tensor to have dimensions E x M x N x T x S.  If False, will
        expect prediction_tensor to have dimensions E x M x N x T.
    :param test_mode: Leave this alone.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(test_mode)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (mean squared error).

        E = number of examples
        M = number of grid rows
        N = number of grid columns
        T = number of target variables (channels)
        S = ensemble size

        :param target_tensor: E-by-M-by-N-by-(T + 1) tensor, where
            target_tensor[..., :-1] contains the actual target values and
            target_tensor[..., -1] contains weights.
        :param prediction_tensor: Tensor of predicted values.  If
            expect_ensemble == True, will expect dimensions E x M x N x T x S.
            Otherwise, will expect E x M x N x T.
        :return: loss: Mean squared error.
        """

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)

        if expect_ensemble:
            relevant_target_tensor = K.expand_dims(target_tensor, axis=-1)
        else:
            relevant_target_tensor = target_tensor

        return K.mean((relevant_target_tensor - prediction_tensor) ** 2)

    loss.__name__ = function_name
    return loss


def dual_weighted_mse(
        channel_weights, function_name, expect_ensemble=True, test_mode=False):
    """Creates dual-weighted mean squared error (DWMSE) loss function.

    K = number of output channels (target variables)

    :param channel_weights: length-K numpy array of channel weights.
    :param function_name: See doc for `mean_squared_error`.
    :param expect_ensemble: Same.
    :param test_mode: Same.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_numpy_array(channel_weights, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(channel_weights, 0.)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(test_mode)

    error_checking.assert_is_numpy_array(
        channel_weights,
        exact_dimensions=numpy.array([len(channel_weights)], dtype=int)
    )
    error_checking.assert_is_greater_numpy_array(channel_weights, 0.)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (DWMSE).

        :param target_tensor: See doc for `mean_squared_error`.
        :param prediction_tensor: Same.
        :return: loss: Mean squared error.
        """

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)

        if expect_ensemble:
            relevant_target_tensor = K.expand_dims(target_tensor, axis=-1)
        else:
            relevant_target_tensor = target_tensor

        dual_weight_tensor = K.maximum(
            K.abs(relevant_target_tensor),
            K.abs(prediction_tensor)
        )
        dual_weight_tensor = K.maximum(dual_weight_tensor, 1.)

        channel_weight_tensor = K.cast(
            K.constant(channel_weights), dual_weight_tensor.dtype
        )
        for _ in range(3):
            channel_weight_tensor = K.expand_dims(channel_weight_tensor, axis=0)
        if expect_ensemble:
            channel_weight_tensor = K.expand_dims(
                channel_weight_tensor, axis=-1
            )

        error_tensor = (
            channel_weight_tensor * dual_weight_tensor *
            (relevant_target_tensor - prediction_tensor) ** 2
        )

        return K.mean(error_tensor)

    loss.__name__ = function_name
    return loss


def dual_weighted_mse_1channel(
        channel_weight, channel_index, function_name,
        expect_ensemble=True, test_mode=False):
    """Creates DWMSE loss function for one channel (target variable).

    :param channel_weight: Channel weight.
    :param channel_index: Channel index.
    :param function_name: See doc for `mean_squared_error`.
    :param expect_ensemble: Same.
    :param test_mode: Same.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_greater(channel_weight, 0.)
    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(test_mode)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (one-channel DWMSE).

        :param target_tensor: See doc for `mean_squared_error`.
        :param prediction_tensor: Same.
        :return: loss: One-channel DWMSE.
        """

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)

        if expect_ensemble:
            relevant_target_tensor = K.expand_dims(
                target_tensor[..., channel_index], axis=-1
            )
            relevant_prediction_tensor = (
                prediction_tensor[:, :, :, channel_index, :]
            )
        else:
            relevant_target_tensor = target_tensor[..., channel_index]
            relevant_prediction_tensor = (
                prediction_tensor[:, :, :, channel_index]
            )

        dual_weight_tensor = K.maximum(
            K.abs(relevant_target_tensor),
            K.abs(relevant_prediction_tensor)
        )
        dual_weight_tensor = K.maximum(dual_weight_tensor, 1.)

        error_tensor = (
            channel_weight * dual_weight_tensor *
            (relevant_target_tensor - relevant_prediction_tensor) ** 2
        )

        return K.mean(error_tensor)

    loss.__name__ = function_name
    return loss


def dual_weighted_mse_constrained_gust(
        channel_weights, u_wind_index, v_wind_index, gust_factor_index,
        function_name, expect_ensemble=True, test_mode=False):
    """Creates DWMSE loss function with constrained gust speed.

    "Constrained gust speed" means that gust speed is computed from 3
    quantities: sustained u-wind, sustained v-wind, and gust factor.

    :param channel_weights: length-K numpy array of channel weights.
    :param u_wind_index: Array index for sustained u-wind.  This tells the
        method that u-wind predictions and targets can be found in
        prediction_tensor[:, :, :, u_wind_index, ...] and
        target_tensor[:, :, :, u_wind_index, ...], respectively.
    :param v_wind_index: Same but for v-wind.
    :param gust_factor_index: Same but for gust factor.
    :param function_name: See doc for `mean_squared_error`.
    :param expect_ensemble: Same.
    :param test_mode: Same.
    :return: loss: Loss function (defined below).
    """

    # TODO(thunderhoser): This method assumes that the strict ReLU activation
    # function is used, meaning that gust-factor predictions are contrained only
    # to be non-negative.  Thus, we add 1 to the gust-factor prediction here.
    # In other words, we assume that the NN model outputs (gust_factor - 1).

    error_checking.assert_is_numpy_array(channel_weights, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(channel_weights, 0.)
    error_checking.assert_is_integer(u_wind_index)
    error_checking.assert_is_geq(u_wind_index, 0)
    error_checking.assert_is_integer(v_wind_index)
    error_checking.assert_is_geq(v_wind_index, 0)
    error_checking.assert_is_integer(gust_factor_index)
    error_checking.assert_is_geq(gust_factor_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(test_mode)

    assert u_wind_index != v_wind_index
    assert u_wind_index != gust_factor_index
    assert v_wind_index != gust_factor_index

    error_checking.assert_is_numpy_array(
        channel_weights,
        exact_dimensions=numpy.array([len(channel_weights)], dtype=int)
    )
    error_checking.assert_is_greater_numpy_array(channel_weights, 0.)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (DWMSE).

        :param target_tensor: See doc for `mean_squared_error`.
        :param prediction_tensor: Same.
        :return: loss: Mean squared error.
        """

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)

        gust_factor_prediction_tensor = (
            prediction_tensor[:, :, :, gust_factor_index, ...] + 1.
        )
        gust_speed_prediction_tensor = gust_factor_prediction_tensor * K.sqrt(
            prediction_tensor[:, :, :, u_wind_index, ...] ** 2 +
            prediction_tensor[:, :, :, v_wind_index, ...] ** 2
        )
        prediction_tensor[:, :, :, gust_factor_index, ...] = (
            gust_speed_prediction_tensor
        )

        if expect_ensemble:
            relevant_target_tensor = K.expand_dims(target_tensor, axis=-1)
        else:
            relevant_target_tensor = target_tensor

        dual_weight_tensor = K.maximum(
            K.abs(relevant_target_tensor),
            K.abs(prediction_tensor)
        )
        dual_weight_tensor = K.maximum(dual_weight_tensor, 1.)

        channel_weight_tensor = K.cast(
            K.constant(channel_weights), dual_weight_tensor.dtype
        )
        for _ in range(3):
            channel_weight_tensor = K.expand_dims(channel_weight_tensor, axis=0)
        if expect_ensemble:
            channel_weight_tensor = K.expand_dims(
                channel_weight_tensor, axis=-1
            )

        error_tensor = (
            channel_weight_tensor * dual_weight_tensor *
            (relevant_target_tensor - prediction_tensor) ** 2
        )
        return K.mean(error_tensor)

    loss.__name__ = function_name
    return loss
