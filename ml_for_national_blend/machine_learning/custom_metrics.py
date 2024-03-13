"""Custom metrics."""

from tensorflow.keras import backend as K
from gewittergefahr.gg_utils import error_checking


def max_prediction(channel_index, function_name, expect_ensemble=True,
                   test_mode=False):
    """Creates metric to return max prediction.

    :param channel_index: Will compute metric for the [k]th channel, where
        k = `channel_index`.
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

        return K.max(prediction_tensor[:, :, :, channel_index, ...])

    metric.__name__ = function_name
    return metric


def min_prediction(channel_index, function_name, expect_ensemble=True,
                   test_mode=False):
    """Creates metric to return minimum prediction.

    :param channel_index: See doc for `max_prediction`.
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

    def metric(target_tensor, prediction_tensor):
        """Computes metric (minimum prediction).

        :param target_tensor: See doc for `max_prediction`.
        :param prediction_tensor: See doc for `max_prediction`.
        :return: metric: Minimum prediction.
        """

        return K.min(prediction_tensor[:, :, :, channel_index, ...])

    metric.__name__ = function_name
    return metric


def mean_squared_error(channel_index, function_name,
                       expect_ensemble=True, test_mode=False):
    """Creates function to return mean squared error (MSE).

    :param channel_index: See doc for `max_prediction`.
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

    def metric(target_tensor, prediction_tensor):
        """Computes metric (MSE).

        :param target_tensor: See doc for `max_prediction`.
        :param prediction_tensor: Same.
        :return: metric: MSE.
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

        squared_error_tensor = (
            (relevant_target_tensor - relevant_prediction_tensor) ** 2
        )
        return K.mean(squared_error_tensor)

    metric.__name__ = function_name
    return metric


def dual_weighted_mse(channel_index, function_name,
                      expect_ensemble=True, test_mode=False):
    """Creates function to return dual-weighted MSE (DWMSE).

    :param channel_index: See doc for `max_prediction`.
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

    def metric(target_tensor, prediction_tensor):
        """Computes metric (DWMSE).

        :param target_tensor: See doc for `max_prediction`.
        :param prediction_tensor: Same.
        :return: metric: DWMSE.
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
        error_tensor = (
            dual_weight_tensor *
            (relevant_target_tensor - relevant_prediction_tensor) ** 2
        )

        return K.mean(error_tensor)

    metric.__name__ = function_name
    return metric
