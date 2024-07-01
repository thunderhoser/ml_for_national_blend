"""Custom loss functions."""

import numpy
import tensorflow
from tensorflow.keras import backend as K
from ml_for_national_blend.outside_code import error_checking


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


def process_dewpoint_predictions(prediction_tensor, temperature_index,
                                 dewpoint_index):
    """Processes dewpoint predictions.

    Specifically, this method assumes that raw dewpoint predictions are actually
    dewpoint-depression predictions -- and then converts them to actual
    dewpoint.

    E = number of examples
    M = number of grid rows
    N = number of grid columns
    T = number of target variables (channels)
    S = ensemble size

    :param prediction_tensor: Tensor of predicted values.  If
        expect_ensemble == True, will expect dimensions E x M x N x T x S.
        Otherwise, will expect E x M x N x T.
    :param temperature_index: Array index for temperature.  This tells the
        method that temperature predictions can be found in
        prediction_tensor[:, :, :, temperature_index, ...].
    :param dewpoint_index: Same but for dewpoint.
    :return: prediction_tensor: Same as input, except that
        prediction_tensor[:, :, :, dewpoint_index, ...] now contains dewpoints
        and not dewpoint depressions.
    """

    error_checking.assert_is_integer(temperature_index)
    error_checking.assert_is_geq(temperature_index, 0)
    error_checking.assert_is_integer(dewpoint_index)
    error_checking.assert_is_geq(dewpoint_index, 0)
    assert temperature_index != dewpoint_index

    new_dewpoint_tensor = (
        prediction_tensor[:, :, :, temperature_index, ...] -
        prediction_tensor[:, :, :, dewpoint_index, ...]
    )

    prediction_tensor = K.concatenate([
        prediction_tensor[:, :, :, :dewpoint_index, ...],
        K.expand_dims(new_dewpoint_tensor, axis=3),
        prediction_tensor[:, :, :, (dewpoint_index + 1):, ...]
    ], axis=3)

    return prediction_tensor


def process_gust_predictions(prediction_tensor, u_wind_index, v_wind_index,
                             gust_index):
    """Processes wind-gust predictions.

    Specifically, this method assumes that raw gust predictions are actually
    (gust factor - 1) -- and then converts them to actual gust speeds.

    :param prediction_tensor: See doc for `process_dewpoint_predictions`.
    :param u_wind_index: Array index for u-wind.  This tells the method that
        u-wind predictions can be found in
        prediction_tensor[:, :, :, u_wind_index, ...].
    :param v_wind_index: Same but for v-wind.
    :param gust_index: Same but for gust (factor).
    :return: prediction_tensor: Same as input, except that
        prediction_tensor[:, :, :, gust_index, ...] now contains gust speeds
        and not gust factors.
    """

    error_checking.assert_is_integer(u_wind_index)
    error_checking.assert_is_geq(u_wind_index, 0)
    error_checking.assert_is_integer(v_wind_index)
    error_checking.assert_is_geq(v_wind_index, 0)
    error_checking.assert_is_integer(gust_index)
    error_checking.assert_is_geq(gust_index, 0)

    assert u_wind_index != v_wind_index
    assert u_wind_index != gust_index
    assert v_wind_index != gust_index

    gust_factor_prediction_tensor = (
        prediction_tensor[:, :, :, gust_index, ...] + 1.
    )
    gust_speed_prediction_tensor = gust_factor_prediction_tensor * K.sqrt(
        prediction_tensor[:, :, :, u_wind_index, ...] ** 2 +
        prediction_tensor[:, :, :, v_wind_index, ...] ** 2
    )

    prediction_tensor = K.concatenate([
        prediction_tensor[:, :, :, :gust_index, ...],
        K.expand_dims(gust_speed_prediction_tensor, axis=3),
        prediction_tensor[:, :, :, (gust_index + 1):, ...]
    ], axis=3)

    return prediction_tensor


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

        mask_weight_tensor = K.expand_dims(target_tensor, axis=-1)
        relevant_target_tensor = target_tensor[..., :-1]

        if expect_ensemble:
            relevant_target_tensor = K.expand_dims(
                relevant_target_tensor, axis=-1
            )

        squared_error_tensor = (relevant_target_tensor - prediction_tensor) ** 2
        return (
            K.sum(mask_weight_tensor * squared_error_tensor) /
            K.sum(mask_weight_tensor * K.ones_like(squared_error_tensor))
        )

    loss.__name__ = function_name
    return loss


def dual_weighted_mse(
        channel_weights, function_name, dual_weight_exponent=1.,
        expect_ensemble=True, test_mode=False):
    """Creates dual-weighted mean squared error (DWMSE) loss function.

    K = number of output channels (target variables)

    :param channel_weights: length-K numpy array of channel weights.
    :param function_name: See doc for `mean_squared_error`.
    :param dual_weight_exponent: Exponent for dual weight.  If 1, the weight for
        every data point will be max(abs(target), abs(prediction)).  If the
        exponent is E, this weight will be
        max(abs(target), abs(prediction)) ** E.
    :param expect_ensemble: Same.
    :param test_mode: Same.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_numpy_array(channel_weights, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(channel_weights, 0.)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_geq(dual_weight_exponent, 1.)
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

        mask_weight_tensor = K.expand_dims(target_tensor, axis=-1)
        relevant_target_tensor = target_tensor[..., :-1]

        if expect_ensemble:
            relevant_target_tensor = K.expand_dims(
                relevant_target_tensor, axis=-1
            )

        dual_weight_tensor = dual_weight_exponent * K.maximum(
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
        return (
            K.sum(mask_weight_tensor * error_tensor) /
            K.sum(mask_weight_tensor * K.ones_like(error_tensor))
        )

    loss.__name__ = function_name
    return loss


def dual_weighted_mse_1channel(
        channel_weight, channel_index,
        u_wind_index, v_wind_index, gust_index,
        temperature_index, dewpoint_index,
        function_name, dual_weight_exponent=1.,
        expect_ensemble=True, test_mode=False):
    """Creates DWMSE loss function for one channel (target variable).

    For the following input args -- u_wind_index, v_wind_index, gust_index,
    temperature_index, dewpoint_index -- if said quantity is not a target
    variable, just make the argument negative!

    :param channel_weight: Channel weight.
    :param channel_index: Channel index.
    :param u_wind_index: Array index for sustained u-wind.  This tells the
        method that u-wind predictions and targets can be found in
        prediction_tensor[:, :, :, u_wind_index, ...] and
        target_tensor[:, :, :, u_wind_index, ...], respectively.
    :param v_wind_index: Same but for v-wind.
    :param gust_index: Same but for wind gust.
    :param temperature_index: Same but for temperature.
    :param dewpoint_index: Same but for dewpoint.
    :param function_name: See doc for `mean_squared_error`.
    :param dual_weight_exponent: See doc for `dual_weighted_mse`.
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

    error_checking.assert_is_integer(u_wind_index)
    error_checking.assert_is_integer(v_wind_index)
    error_checking.assert_is_integer(gust_index)
    error_checking.assert_is_integer(temperature_index)
    error_checking.assert_is_integer(dewpoint_index)
    error_checking.assert_is_geq(dual_weight_exponent, 1.)

    all_indices = [
        u_wind_index, v_wind_index, gust_index,
        temperature_index, dewpoint_index
    ]
    all_indices = [i for i in all_indices if i >= 0]
    all_indices = numpy.array(all_indices, dtype=int)

    assert len(all_indices) == len(numpy.unique(all_indices))

    def loss(target_tensor, prediction_tensor):
        """Computes loss (one-channel DWMSE).

        :param target_tensor: See doc for `mean_squared_error`.
        :param prediction_tensor: Same.
        :return: loss: One-channel DWMSE.
        """

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)

        mask_weight_tensor = K.expand_dims(target_tensor, axis=-1)
        relevant_target_tensor = target_tensor[..., :-1]

        if channel_index == dewpoint_index:
            prediction_tensor = process_dewpoint_predictions(
                prediction_tensor=prediction_tensor,
                temperature_index=temperature_index,
                dewpoint_index=dewpoint_index
            )

        if channel_index == gust_index:
            prediction_tensor = process_gust_predictions(
                prediction_tensor=prediction_tensor,
                u_wind_index=u_wind_index,
                v_wind_index=v_wind_index,
                gust_index=gust_index
            )

        if expect_ensemble:
            relevant_target_tensor = K.expand_dims(
                relevant_target_tensor[..., channel_index], axis=-1
            )
            relevant_prediction_tensor = (
                prediction_tensor[:, :, :, channel_index, :]
            )
        else:
            relevant_target_tensor = relevant_target_tensor[..., channel_index]
            relevant_prediction_tensor = (
                prediction_tensor[:, :, :, channel_index]
            )
            mask_weight_tensor = mask_weight_tensor[..., 0]

        dual_weight_tensor = dual_weight_exponent * K.maximum(
            K.abs(relevant_target_tensor),
            K.abs(relevant_prediction_tensor)
        )
        dual_weight_tensor = K.maximum(dual_weight_tensor, 1.)

        error_tensor = (
            channel_weight * dual_weight_tensor *
            (relevant_target_tensor - relevant_prediction_tensor) ** 2
        )
        return (
            K.sum(mask_weight_tensor * error_tensor) /
            K.sum(mask_weight_tensor * K.ones_like(error_tensor))
        )

    loss.__name__ = function_name
    return loss


def dual_weighted_mse_with_constraints(
        channel_weights, u_wind_index, v_wind_index, gust_index,
        temperature_index, dewpoint_index, function_name,
        dual_weight_exponent=1., expect_ensemble=True, test_mode=False):
    """Creates DWMSE loss function with constrained dewpoint and wind gust.

    "Constrained dewpoint": We assume that the model's raw dewpoint prediction
    is actually dewpoint depression, then use temperature to convert this to
    real dewpoint.

    "Constrained wind gust": We assume that the model's raw gust prediction is
    actually gust factor, then use sustained wind speed to convert this to real
    gust speed.

    :param channel_weights: length-K numpy array of channel weights.
    :param u_wind_index: Array index for sustained u-wind.  This tells the
        method that u-wind predictions and targets can be found in
        prediction_tensor[:, :, :, u_wind_index, ...] and
        target_tensor[:, :, :, u_wind_index, ...], respectively.
    :param v_wind_index: Same but for v-wind.
    :param gust_index: Same but for wind gust.
    :param temperature_index: Same but for temperature.
    :param dewpoint_index: Same but for dewpoint.
    :param function_name: See doc for `mean_squared_error`.
    :param dual_weight_exponent: See doc for `dual_weighted_mse`.
    :param expect_ensemble: Same.
    :param test_mode: Same.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_numpy_array(channel_weights, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(channel_weights, 0.)

    error_checking.assert_is_integer(u_wind_index)
    error_checking.assert_is_geq(u_wind_index, 0)
    error_checking.assert_is_integer(v_wind_index)
    error_checking.assert_is_geq(v_wind_index, 0)
    error_checking.assert_is_integer(gust_index)
    error_checking.assert_is_geq(gust_index, 0)
    error_checking.assert_is_integer(temperature_index)
    error_checking.assert_is_geq(temperature_index, 0)
    error_checking.assert_is_integer(dewpoint_index)
    error_checking.assert_is_geq(dewpoint_index, 0)
    error_checking.assert_is_geq(dual_weight_exponent, 1.)

    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(test_mode)

    all_indices = numpy.array([
        u_wind_index, v_wind_index, gust_index,
        temperature_index, dewpoint_index
    ], dtype=int)

    assert len(all_indices) == len(numpy.unique(all_indices))

    def loss(target_tensor, prediction_tensor):
        """Computes loss (DWMSE).

        :param target_tensor: See doc for `mean_squared_error`.
        :param prediction_tensor: Same.
        :return: loss: DWMSE.
        """

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)

        mask_weight_tensor = K.expand_dims(target_tensor, axis=-1)
        relevant_target_tensor = target_tensor[..., :-1]

        prediction_tensor = process_dewpoint_predictions(
            prediction_tensor=prediction_tensor,
            temperature_index=temperature_index,
            dewpoint_index=dewpoint_index
        )
        prediction_tensor = process_gust_predictions(
            prediction_tensor=prediction_tensor,
            u_wind_index=u_wind_index,
            v_wind_index=v_wind_index,
            gust_index=gust_index
        )

        if expect_ensemble:
            relevant_target_tensor = K.expand_dims(
                relevant_target_tensor, axis=-1
            )

        dual_weight_tensor = dual_weight_exponent * K.maximum(
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
        return (
            K.sum(mask_weight_tensor * error_tensor) /
            K.sum(mask_weight_tensor * K.ones_like(error_tensor))
        )

    loss.__name__ = function_name
    return loss
