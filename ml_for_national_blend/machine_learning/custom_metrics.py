"""Custom metrics."""

import numpy
from tensorflow.keras import backend as K
from gewittergefahr.gg_utils import error_checking
from ml_for_national_blend.machine_learning import custom_losses


def _check_index_args(u_wind_index, v_wind_index, gust_index,
                      temperature_index, dewpoint_index):
    """Error-checks index arguments.

    :param u_wind_index: See doc for `max_prediction`.
    :param v_wind_index: Same.
    :param gust_index: Same.
    :param temperature_index: Same.
    :param dewpoint_index: Same.
    """

    error_checking.assert_is_integer(u_wind_index)
    error_checking.assert_is_integer(v_wind_index)
    error_checking.assert_is_integer(gust_index)
    error_checking.assert_is_integer(temperature_index)
    error_checking.assert_is_integer(dewpoint_index)

    all_indices = [
        u_wind_index, v_wind_index, gust_index,
        temperature_index, dewpoint_index
    ]
    all_indices = [i for i in all_indices if i >= 0]
    all_indices = numpy.array(all_indices, dtype=int)

    assert len(all_indices) == len(numpy.unique(all_indices))


def max_prediction(
        channel_index, u_wind_index, v_wind_index, gust_index,
        temperature_index, dewpoint_index,
        function_name, expect_ensemble=True, test_mode=False):
    """Creates metric to return max prediction.

    For the following input args -- u_wind_index, v_wind_index, gust_index,
    temperature_index, dewpoint_index -- if said quantity is not a target
    variable, just make the argument negative!

    :param channel_index: Will compute metric for the [k]th channel, where
        k = `channel_index`.
    :param u_wind_index: Array index for sustained u-wind.  This tells the
        method that u-wind predictions and targets can be found in
        prediction_tensor[:, :, :, u_wind_index, ...] and
        target_tensor[:, :, :, u_wind_index, ...], respectively.
    :param v_wind_index: Same but for v-wind.
    :param gust_index: Same but for wind gust.
    :param temperature_index: Same but for temperature.
    :param dewpoint_index: Same but for dewpoint.
    :param function_name: Function name (string).
    :param expect_ensemble: Boolean flag.  If True, will expect
        prediction_tensor to have dimensions E x M x N x T x S.  If False, will
        expect prediction_tensor to have dimensions E x M x N x T.
    :param test_mode: Leave this alone.
    :return: metric: Metric function (defined below).
    """

    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(test_mode)

    _check_index_args(
        u_wind_index=u_wind_index,
        v_wind_index=v_wind_index,
        gust_index=gust_index,
        temperature_index=temperature_index,
        dewpoint_index=dewpoint_index
    )

    def metric(target_tensor, prediction_tensor):
        """Computes metric (max prediction).

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
        :return: metric: Max prediction.
        """

        if channel_index == dewpoint_index:
            prediction_tensor = custom_losses.process_dewpoint_predictions(
                prediction_tensor=prediction_tensor,
                temperature_index=temperature_index,
                dewpoint_index=dewpoint_index
            )

        if channel_index == gust_index:
            prediction_tensor = custom_losses.process_gust_predictions(
                prediction_tensor=prediction_tensor,
                u_wind_index=u_wind_index,
                v_wind_index=v_wind_index,
                gust_index=gust_index
            )

        return K.max(prediction_tensor[:, :, :, channel_index, ...])

    metric.__name__ = function_name
    return metric


def spatial_max_bias(
        channel_index, u_wind_index, v_wind_index, gust_index,
        temperature_index, dewpoint_index,
        function_name, expect_ensemble=True, test_mode=False):
    """Creates metric to return bias in spatial maximum.

    :param channel_index: See doc for `max_prediction`.
    :param u_wind_index: Same.
    :param v_wind_index: Same.
    :param gust_index: Same.
    :param temperature_index: Same.
    :param dewpoint_index: Same.
    :param function_name: Same.
    :param expect_ensemble: Same.
    :param test_mode: Same.
    :return: metric: Metric function (defined below).
    """

    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(test_mode)

    _check_index_args(
        u_wind_index=u_wind_index,
        v_wind_index=v_wind_index,
        gust_index=gust_index,
        temperature_index=temperature_index,
        dewpoint_index=dewpoint_index
    )

    def metric(target_tensor, prediction_tensor):
        """Computes metric (bias in spatial maximum).

        :param target_tensor: See doc for `max_prediction`.
        :param prediction_tensor: See doc for `max_prediction`.
        :return: metric: Bias in spatial max.
        """

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)

        if channel_index == dewpoint_index:
            prediction_tensor = custom_losses.process_dewpoint_predictions(
                prediction_tensor=prediction_tensor,
                temperature_index=temperature_index,
                dewpoint_index=dewpoint_index
            )

        if channel_index == gust_index:
            prediction_tensor = custom_losses.process_gust_predictions(
                prediction_tensor=prediction_tensor,
                u_wind_index=u_wind_index,
                v_wind_index=v_wind_index,
                gust_index=gust_index
            )

        if expect_ensemble:
            relevant_prediction_tensor = K.mean(
                prediction_tensor[..., channel_index, :], axis=-1
            )
        else:
            relevant_prediction_tensor = prediction_tensor[..., channel_index]

        relevant_target_tensor = target_tensor[..., channel_index]

        max_predictions = K.max(relevant_prediction_tensor, axis=(1, 2))
        max_targets = K.max(relevant_target_tensor, axis=(1, 2))
        return K.mean(max_predictions - max_targets)

    metric.__name__ = function_name
    return metric


def min_prediction(
        channel_index, u_wind_index, v_wind_index, gust_index,
        temperature_index, dewpoint_index,
        function_name, expect_ensemble=True, test_mode=False):
    """Creates metric to return minimum prediction.

    :param channel_index: See doc for `max_prediction`.
    :param u_wind_index: Same.
    :param v_wind_index: Same.
    :param gust_index: Same.
    :param temperature_index: Same.
    :param dewpoint_index: Same.
    :param function_name: Same.
    :param expect_ensemble: Same.
    :param test_mode: Same.
    :return: metric: Metric function (defined below).
    """

    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(test_mode)

    _check_index_args(
        u_wind_index=u_wind_index,
        v_wind_index=v_wind_index,
        gust_index=gust_index,
        temperature_index=temperature_index,
        dewpoint_index=dewpoint_index
    )

    def metric(target_tensor, prediction_tensor):
        """Computes metric (minimum prediction).

        :param target_tensor: See doc for `max_prediction`.
        :param prediction_tensor: See doc for `max_prediction`.
        :return: metric: Minimum prediction.
        """

        if channel_index == dewpoint_index:
            prediction_tensor = custom_losses.process_dewpoint_predictions(
                prediction_tensor=prediction_tensor,
                temperature_index=temperature_index,
                dewpoint_index=dewpoint_index
            )

        if channel_index == gust_index:
            prediction_tensor = custom_losses.process_gust_predictions(
                prediction_tensor=prediction_tensor,
                u_wind_index=u_wind_index,
                v_wind_index=v_wind_index,
                gust_index=gust_index
            )

        return K.min(prediction_tensor[:, :, :, channel_index, ...])

    metric.__name__ = function_name
    return metric


def spatial_min_bias(
        channel_index, u_wind_index, v_wind_index, gust_index,
        temperature_index, dewpoint_index,
        function_name, expect_ensemble=True, test_mode=False):
    """Creates metric to return bias in spatial minimum.

    :param channel_index: See doc for `max_prediction`.
    :param u_wind_index: Same.
    :param v_wind_index: Same.
    :param gust_index: Same.
    :param temperature_index: Same.
    :param dewpoint_index: Same.
    :param function_name: Same.
    :param expect_ensemble: Same.
    :param test_mode: Same.
    :return: metric: Metric function (defined below).
    """

    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(test_mode)

    _check_index_args(
        u_wind_index=u_wind_index,
        v_wind_index=v_wind_index,
        gust_index=gust_index,
        temperature_index=temperature_index,
        dewpoint_index=dewpoint_index
    )

    def metric(target_tensor, prediction_tensor):
        """Computes metric (bias in spatial minimum).

        :param target_tensor: See doc for `max_prediction`.
        :param prediction_tensor: See doc for `max_prediction`.
        :return: metric: Bias in spatial minimum.
        """

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)

        if channel_index == dewpoint_index:
            prediction_tensor = custom_losses.process_dewpoint_predictions(
                prediction_tensor=prediction_tensor,
                temperature_index=temperature_index,
                dewpoint_index=dewpoint_index
            )

        if channel_index == gust_index:
            prediction_tensor = custom_losses.process_gust_predictions(
                prediction_tensor=prediction_tensor,
                u_wind_index=u_wind_index,
                v_wind_index=v_wind_index,
                gust_index=gust_index
            )

        if expect_ensemble:
            relevant_prediction_tensor = K.mean(
                prediction_tensor[..., channel_index, :], axis=-1
            )
        else:
            relevant_prediction_tensor = prediction_tensor[..., channel_index]

        relevant_target_tensor = target_tensor[..., channel_index]

        min_predictions = K.min(relevant_prediction_tensor, axis=(1, 2))
        min_targets = K.min(relevant_target_tensor, axis=(1, 2))
        return K.mean(min_predictions - min_targets)

    metric.__name__ = function_name
    return metric


def mean_squared_error(
        channel_index, u_wind_index, v_wind_index, gust_index,
        temperature_index, dewpoint_index,
        function_name, expect_ensemble=True, test_mode=False):
    """Creates function to return mean squared error (MSE).

    :param channel_index: See doc for `max_prediction`.
    :param u_wind_index: Same.
    :param v_wind_index: Same.
    :param gust_index: Same.
    :param temperature_index: Same.
    :param dewpoint_index: Same.
    :param function_name: Same.
    :param expect_ensemble: Same.
    :param test_mode: Same.
    :return: metric: Metric function (defined below).
    """

    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(test_mode)

    _check_index_args(
        u_wind_index=u_wind_index,
        v_wind_index=v_wind_index,
        gust_index=gust_index,
        temperature_index=temperature_index,
        dewpoint_index=dewpoint_index
    )

    def metric(target_tensor, prediction_tensor):
        """Computes metric (MSE).

        :param target_tensor: See doc for `max_prediction`.
        :param prediction_tensor: Same.
        :return: metric: MSE.
        """

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)

        if channel_index == dewpoint_index:
            prediction_tensor = custom_losses.process_dewpoint_predictions(
                prediction_tensor=prediction_tensor,
                temperature_index=temperature_index,
                dewpoint_index=dewpoint_index
            )

        if channel_index == gust_index:
            prediction_tensor = custom_losses.process_gust_predictions(
                prediction_tensor=prediction_tensor,
                u_wind_index=u_wind_index,
                v_wind_index=v_wind_index,
                gust_index=gust_index
            )

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

        squared_error_tensor = (
            (relevant_target_tensor - relevant_prediction_tensor) ** 2
        )
        return K.mean(squared_error_tensor)

    metric.__name__ = function_name
    return metric


def dual_weighted_mse(
        channel_index, u_wind_index, v_wind_index, gust_index,
        temperature_index, dewpoint_index,
        function_name, expect_ensemble=True, test_mode=False):
    """Creates function to return dual-weighted MSE (DWMSE).

    :param channel_index: See doc for `max_prediction`.
    :param u_wind_index: Same.
    :param v_wind_index: Same.
    :param gust_index: Same.
    :param temperature_index: Same.
    :param dewpoint_index: Same.
    :param function_name: Same.
    :param expect_ensemble: Same.
    :param test_mode: Same.
    :return: metric: Metric function (defined below).
    """

    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(test_mode)

    _check_index_args(
        u_wind_index=u_wind_index,
        v_wind_index=v_wind_index,
        gust_index=gust_index,
        temperature_index=temperature_index,
        dewpoint_index=dewpoint_index
    )

    def metric(target_tensor, prediction_tensor):
        """Computes metric (DWMSE).

        :param target_tensor: See doc for `max_prediction`.
        :param prediction_tensor: Same.
        :return: metric: DWMSE.
        """

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)

        if channel_index == dewpoint_index:
            prediction_tensor = custom_losses.process_dewpoint_predictions(
                prediction_tensor=prediction_tensor,
                temperature_index=temperature_index,
                dewpoint_index=dewpoint_index
            )

        if channel_index == gust_index:
            prediction_tensor = custom_losses.process_gust_predictions(
                prediction_tensor=prediction_tensor,
                u_wind_index=u_wind_index,
                v_wind_index=v_wind_index,
                gust_index=gust_index
            )

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
        error_tensor = (
            dual_weight_tensor *
            (relevant_target_tensor - relevant_prediction_tensor) ** 2
        )

        return K.mean(error_tensor)

    metric.__name__ = function_name
    return metric
