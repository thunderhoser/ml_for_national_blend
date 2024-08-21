"""Custom loss functions."""

import os
import sys
import numpy
import tensorflow
import tensorflow.math as tf_math
from tensorflow.keras import backend as K

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking


def __get_num_target_fields(prediction_tensor, expect_ensemble):
    """Determines number of target fields.

    :param prediction_tensor: See documentation for `mean_squared_error`.
    :param expect_ensemble: Same.
    :return: num_target_fields: Integer.
    """

    if expect_ensemble:
        return prediction_tensor.shape[-2]

    return prediction_tensor.shape[-1]


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


def check_index_args(u_wind_index, v_wind_index, gust_index, temperature_index,
                     dewpoint_index):
    """Error-checks index arguments.

    :param u_wind_index: See doc for `mean_squared_error`.
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


def mean_squared_error(
        u_wind_index, v_wind_index, gust_index, temperature_index,
        dewpoint_index, function_name, expect_ensemble=True, test_mode=False):
    """Creates mean squared error (MSE) loss function.

    E = number of examples
    M = number of grid rows
    N = number of grid columns
    T = number of target variables (channels)
    S = ensemble size

    For the following input args -- u_wind_index, v_wind_index, gust_index,
    temperature_index, dewpoint_index -- if said quantity is not a target
    variable, just make the argument negative!

    :param function_name: Function name (string).
    :param u_wind_index: Array index for sustained u-wind.  This tells the
        method that u-wind predictions and targets can be found in
        prediction_tensor[:, :, :, u_wind_index, ...] and
        target_tensor[:, :, :, u_wind_index, ...], respectively.
    :param v_wind_index: Same but for v-wind.
    :param gust_index: Same but for wind gust.
    :param temperature_index: Same but for temperature.
    :param dewpoint_index: Same but for dewpoint.
    :param expect_ensemble: Boolean flag.  If True, will expect
        prediction_tensor to have dimensions E x M x N x T x S.  If False, will
        expect prediction_tensor to have dimensions E x M x N x T.
    :param test_mode: Leave this alone.
    :return: loss: Loss function (defined below).
    """

    check_index_args(
        u_wind_index=u_wind_index,
        v_wind_index=v_wind_index,
        gust_index=gust_index,
        temperature_index=temperature_index,
        dewpoint_index=dewpoint_index
    )

    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(test_mode)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (mean squared error).

        :param target_tensor: E-by-M-by-N-by-(T + 1) tensor, where
            target_tensor[..., :-1] contains the actual target values and
            target_tensor[..., -1] contains weights.
        :param prediction_tensor: Tensor of predicted values.  If
            expect_ensemble == True, will expect dimensions E x M x N x T x S.
            Otherwise, will expect E x M x N x T.
        :return: scalar_mse: MSE (a scalar value).
        """

        if dewpoint_index >= 0 and temperature_index >= 0:
            prediction_tensor = process_dewpoint_predictions(
                prediction_tensor=prediction_tensor,
                temperature_index=temperature_index,
                dewpoint_index=dewpoint_index
            )

        if u_wind_index >= 0 and v_wind_index >= 0 and gust_index >= 0:
            prediction_tensor = process_gust_predictions(
                prediction_tensor=prediction_tensor,
                u_wind_index=u_wind_index,
                v_wind_index=v_wind_index,
                gust_index=gust_index
            )

        num_target_fields = __get_num_target_fields(
            prediction_tensor=prediction_tensor,
            expect_ensemble=expect_ensemble
        )

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)
        relevant_target_tensor = target_tensor[..., :num_target_fields]
        mask_weight_tensor = K.expand_dims(target_tensor[..., -1], axis=-1)

        if expect_ensemble:
            relevant_target_tensor = K.expand_dims(
                relevant_target_tensor, axis=-1
            )
            mask_weight_tensor = K.expand_dims(
                mask_weight_tensor, axis=-1
            )

        squared_error_tensor = (relevant_target_tensor - prediction_tensor) ** 2
        return (
            K.sum(mask_weight_tensor * squared_error_tensor) /
            K.sum(mask_weight_tensor * K.ones_like(squared_error_tensor))
        )

    loss.__name__ = function_name
    return loss


def dual_weighted_mse(
        channel_weights, u_wind_index, v_wind_index, gust_index,
        temperature_index, dewpoint_index, function_name,
        dual_weight_exponent=1., expect_ensemble=True, test_mode=False):
    """Creates dual-weighted mean squared error (DWMSE) loss function.

    T = number of target variables (channels)

    :param channel_weights: length-T numpy array of channel weights.
    :param u_wind_index: See documentation for `mean_squared_error`.
    :param v_wind_index: Same.
    :param gust_index: Same.
    :param temperature_index: Same.
    :param dewpoint_index: Same.
    :param function_name: Same.
    :param dual_weight_exponent: Exponent for dual weight.  If 1, the weight for
        every data point will be max(abs(target), abs(prediction)).  If the
        exponent is E, this weight will be
        max(abs(target), abs(prediction)) ** E.
    :param expect_ensemble: Same.
    :param test_mode: Same.
    :return: loss: Loss function (defined below).
    """

    check_index_args(
        u_wind_index=u_wind_index,
        v_wind_index=v_wind_index,
        gust_index=gust_index,
        temperature_index=temperature_index,
        dewpoint_index=dewpoint_index
    )

    error_checking.assert_is_numpy_array(channel_weights, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(channel_weights, 0.)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_geq(dual_weight_exponent, 1.)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(test_mode)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (DWMSE).

        :param target_tensor: See doc for `mean_squared_error`.
        :param prediction_tensor: Same.
        :return: scalar_dwmse: DWMSE (a scalar value).
        """

        if dewpoint_index >= 0 and temperature_index >= 0:
            prediction_tensor = process_dewpoint_predictions(
                prediction_tensor=prediction_tensor,
                temperature_index=temperature_index,
                dewpoint_index=dewpoint_index
            )

        if u_wind_index >= 0 and v_wind_index >= 0 and gust_index >= 0:
            prediction_tensor = process_gust_predictions(
                prediction_tensor=prediction_tensor,
                u_wind_index=u_wind_index,
                v_wind_index=v_wind_index,
                gust_index=gust_index
            )

        num_target_fields = __get_num_target_fields(
            prediction_tensor=prediction_tensor,
            expect_ensemble=expect_ensemble
        )

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)
        relevant_target_tensor = target_tensor[..., :num_target_fields]
        mask_weight_tensor = K.expand_dims(target_tensor[..., -1], axis=-1)

        if expect_ensemble:
            relevant_target_tensor = K.expand_dims(
                relevant_target_tensor, axis=-1
            )
            mask_weight_tensor = K.expand_dims(
                mask_weight_tensor, axis=-1
            )

        dual_weight_tensor = K.pow(
            K.maximum(K.abs(relevant_target_tensor), K.abs(prediction_tensor)),
            dual_weight_exponent
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


def dual_weighted_msess(
        channel_weights, u_wind_index, v_wind_index, gust_index,
        temperature_index, dewpoint_index, function_name,
        dual_weight_exponent=1., expect_ensemble=True, test_mode=False):
    """Creates DWMSE-skill-score loss function.

    :param channel_weights: See documentation for `dual_weighted_mse`.
    :param u_wind_index: Same.
    :param v_wind_index: Same.
    :param gust_index: Same.
    :param temperature_index: Same.
    :param dewpoint_index: Same.
    :param function_name: Same.
    :param dual_weight_exponent: Same.
    :param expect_ensemble: Same.
    :param test_mode: Same.
    :return: loss: Loss function (defined below).
    """

    check_index_args(
        u_wind_index=u_wind_index,
        v_wind_index=v_wind_index,
        gust_index=gust_index,
        temperature_index=temperature_index,
        dewpoint_index=dewpoint_index
    )

    error_checking.assert_is_numpy_array(channel_weights, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(channel_weights, 0.)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_geq(dual_weight_exponent, 1.)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(test_mode)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (DWMSE skill score).

        :param target_tensor: See doc for `mean_squared_error`.
        :param prediction_tensor: Same.
        :return: scalar_dwmsess: DWMSE skill score (a scalar value).
        """

        if dewpoint_index >= 0 and temperature_index >= 0:
            prediction_tensor = process_dewpoint_predictions(
                prediction_tensor=prediction_tensor,
                temperature_index=temperature_index,
                dewpoint_index=dewpoint_index
            )

        if u_wind_index >= 0 and v_wind_index >= 0 and gust_index >= 0:
            prediction_tensor = process_gust_predictions(
                prediction_tensor=prediction_tensor,
                u_wind_index=u_wind_index,
                v_wind_index=v_wind_index,
                gust_index=gust_index
            )

        num_target_fields = __get_num_target_fields(
            prediction_tensor=prediction_tensor,
            expect_ensemble=expect_ensemble
        )

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)
        relevant_target_tensor = target_tensor[..., :num_target_fields]
        relevant_baseline_prediction_tensor = (
            target_tensor[..., num_target_fields:-1]
        )
        mask_weight_tensor = K.expand_dims(target_tensor[..., -1], axis=-1)

        # Ensure compatible tensor shapes.
        if expect_ensemble:
            relevant_target_tensor = K.expand_dims(
                relevant_target_tensor, axis=-1
            )
            relevant_baseline_prediction_tensor = K.expand_dims(
                relevant_baseline_prediction_tensor, axis=-1
            )
            mask_weight_tensor = K.expand_dims(
                mask_weight_tensor, axis=-1
            )

        # Create dual-weight tensor.
        dual_weight_tensor = K.pow(
            K.maximum(K.abs(relevant_target_tensor), K.abs(prediction_tensor)),
            dual_weight_exponent
        )
        dual_weight_tensor = K.maximum(dual_weight_tensor, 1.)

        # Create channel-weight tensor.
        channel_weight_tensor = K.cast(
            K.constant(channel_weights), dual_weight_tensor.dtype
        )
        for _ in range(3):
            channel_weight_tensor = K.expand_dims(channel_weight_tensor, axis=0)
        if expect_ensemble:
            channel_weight_tensor = K.expand_dims(
                channel_weight_tensor, axis=-1
            )

        # Compute dual-weighted MSE.
        error_tensor = (
            channel_weight_tensor * dual_weight_tensor *
            (relevant_target_tensor - prediction_tensor) ** 2
        )
        actual_dwmse = (
            K.sum(mask_weight_tensor * error_tensor) /
            K.sum(mask_weight_tensor * K.ones_like(error_tensor))
        )

        # Create dual-weight tensor for baseline.
        dual_weight_tensor = K.pow(
            K.maximum(
                K.abs(relevant_target_tensor),
                K.abs(relevant_baseline_prediction_tensor)
            ),
            dual_weight_exponent
        )
        dual_weight_tensor = K.maximum(dual_weight_tensor, 1.)

        # Compute dual-weighted MSE for baseline.
        error_tensor = (
            channel_weight_tensor * dual_weight_tensor *
            (relevant_target_tensor - relevant_baseline_prediction_tensor) ** 2
        )

        nan_mask_tensor = tf_math.is_finite(error_tensor)
        error_tensor = tensorflow.where(
            nan_mask_tensor, error_tensor, tensorflow.zeros_like(error_tensor)
        )

        mask_weight_tensor = mask_weight_tensor * K.ones_like(error_tensor)
        mask_weight_tensor = tensorflow.where(
            nan_mask_tensor,
            mask_weight_tensor, tensorflow.zeros_like(mask_weight_tensor)
        )

        baseline_dwmse = (
            K.sum(mask_weight_tensor * error_tensor) /
            K.sum(mask_weight_tensor * K.ones_like(error_tensor))
        )

        # Return negative skill score.
        return (actual_dwmse - baseline_dwmse) / baseline_dwmse

    loss.__name__ = function_name
    return loss


def dual_weighted_crpss(
        channel_weights, u_wind_index, v_wind_index, gust_index,
        temperature_index, dewpoint_index, function_name,
        dual_weight_exponent=1., test_mode=False):
    """Creates dual-weighted-CRPSS loss function.

    :param channel_weights: See documentation for `dual_weighted_mse`.
    :param u_wind_index: Same.
    :param v_wind_index: Same.
    :param gust_index: Same.
    :param temperature_index: Same.
    :param dewpoint_index: Same.
    :param function_name: Same.
    :param dual_weight_exponent: Same.
    :param test_mode: Same.
    :return: loss: Loss function (defined below).
    """

    check_index_args(
        u_wind_index=u_wind_index,
        v_wind_index=v_wind_index,
        gust_index=gust_index,
        temperature_index=temperature_index,
        dewpoint_index=dewpoint_index
    )

    error_checking.assert_is_numpy_array(channel_weights, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(channel_weights, 0.)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_geq(dual_weight_exponent, 1.)
    error_checking.assert_is_boolean(test_mode)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (dual-weighted CRPSS).

        :param target_tensor: See doc for `mean_squared_error`.
        :param prediction_tensor: Same.
        :return: scalar_dual_weighted_crpss: Dual-weighted CRPSS (a scalar
            value).
        """

        if dewpoint_index >= 0 and temperature_index >= 0:
            prediction_tensor = process_dewpoint_predictions(
                prediction_tensor=prediction_tensor,
                temperature_index=temperature_index,
                dewpoint_index=dewpoint_index
            )

        if u_wind_index >= 0 and v_wind_index >= 0 and gust_index >= 0:
            prediction_tensor = process_gust_predictions(
                prediction_tensor=prediction_tensor,
                u_wind_index=u_wind_index,
                v_wind_index=v_wind_index,
                gust_index=gust_index
            )

        num_target_fields = __get_num_target_fields(
            prediction_tensor=prediction_tensor,
            expect_ensemble=True
        )

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)
        relevant_target_tensor = target_tensor[..., :num_target_fields]
        relevant_baseline_prediction_tensor = (
            target_tensor[..., num_target_fields:-1]
        )
        mask_weight_tensor = K.expand_dims(target_tensor[..., -1], axis=-1)

        # Ensure compatible tensor shapes.
        relevant_target_tensor = K.expand_dims(relevant_target_tensor, axis=-1)
        relevant_baseline_prediction_tensor = K.expand_dims(
            relevant_baseline_prediction_tensor, axis=-1
        )
        mask_weight_tensor = K.expand_dims(mask_weight_tensor, axis=-1)

        # Create dual-weight tensor.
        dual_weight_tensor = K.pow(
            K.maximum(K.abs(relevant_target_tensor), K.abs(prediction_tensor)),
            dual_weight_exponent
        )
        dual_weight_tensor = K.maximum(dual_weight_tensor, 1.)

        # Create channel-weight tensor.
        channel_weight_tensor = K.cast(
            K.constant(channel_weights), dual_weight_tensor.dtype
        )
        for _ in range(3):
            channel_weight_tensor = K.expand_dims(channel_weight_tensor, axis=0)

        # Compute dual-weighted CRPS.
        absolute_error_tensor = K.abs(
            prediction_tensor - relevant_target_tensor
        )
        mean_prediction_error_tensor = K.mean(
            dual_weight_tensor * absolute_error_tensor, axis=-1
        )

        prediction_tensor = tensorflow.transpose(
            prediction_tensor, perm=[1, 0, 2, 3, 4]
        )

        def compute_mapd_1row(prediction_tensor_1row, dual_weight_exponent):
            """Computes MAPD for one grid row.

            MAPD = mean absolute pairwise difference

            :param prediction_tensor_1row: E-by-N-by-T-by-S tensor of
                predictions.
            :param dual_weight_exponent: See documentation for
                `dual_weighted_mse`.
            :return: mapd_tensor_1row: E-by-N-by-T tensor of mean absolute
                pairwise differences.
            """

            pt1row = prediction_tensor_1row

            return K.mean(
                K.pow(
                    K.maximum(
                        K.abs(K.expand_dims(pt1row, axis=-1)),
                        K.abs(K.expand_dims(pt1row, axis=-2))
                    ),
                    dual_weight_exponent
                ) *
                K.abs(
                    K.expand_dims(pt1row, axis=-1) -
                    K.expand_dims(pt1row, axis=-2)
                ),
                axis=(-2, -1)
            )

        def loop_body(i, mapd_tensor):
            """Body of while-loop for computing MAPD.

            This method is run once for every iteration through the while-loop,
            i.e., once for every grid row.

            :param i: Index of current grid row.
            :param mapd_tensor: M-by-E-by-N-by-T tensor of MAPD values, which
                this method will update.
            :return: i_new: Index of next grid row.
            :return: mapd_tensor: Updated version of input.
            """

            this_mapd_tensor = compute_mapd_1row(
                prediction_tensor_1row=prediction_tensor[i, ...],
                dual_weight_exponent=dual_weight_exponent
            )

            mapd_tensor = mapd_tensor.write(i, this_mapd_tensor)
            return i + 1, mapd_tensor

        mapd_tensor = tensorflow.TensorArray(
            size=prediction_tensor.shape[0],
            dtype=tensorflow.float32
        )

        i = tensorflow.constant(0)
        condition = lambda i, mapd_tensor: tensorflow.less(
            i, prediction_tensor.shape[0]
        )

        _, mapd_tensor = tensorflow.while_loop(
            cond=condition,
            body=loop_body,
            loop_vars=[i, mapd_tensor],
            maximum_iterations=prediction_tensor.shape[0],
            # parallel_iterations=1,
            # swap_memory=True
        )
        mapd_tensor = mapd_tensor.stack()

        mapd_tensor = tensorflow.transpose(
            mapd_tensor, perm=[1, 0, 2, 3]
        )
        error_tensor = channel_weight_tensor * (
            mean_prediction_error_tensor -
            0.5 * mapd_tensor
        )
        actual_dwcrps = (
            K.sum(mask_weight_tensor * error_tensor) /
            K.sum(mask_weight_tensor * K.ones_like(error_tensor))
        )

        # Create dual-weight tensor for baseline.
        dual_weight_tensor = K.pow(
            K.maximum(
                K.abs(relevant_target_tensor),
                K.abs(relevant_baseline_prediction_tensor)
            ),
            dual_weight_exponent
        )
        dual_weight_tensor = K.maximum(dual_weight_tensor, 1.)

        # Compute dual-weighted CRPSS for baseline.
        absolute_error_tensor = K.abs(
            relevant_baseline_prediction_tensor - relevant_target_tensor
        )
        mean_prediction_error_tensor = K.mean(
            dual_weight_tensor * absolute_error_tensor, axis=-1
        )
        error_tensor = channel_weight_tensor * mean_prediction_error_tensor

        nan_mask_tensor = tf_math.is_finite(error_tensor)
        error_tensor = tensorflow.where(
            nan_mask_tensor, error_tensor, tensorflow.zeros_like(error_tensor)
        )
        mask_weight_tensor = mask_weight_tensor * K.ones_like(error_tensor)
        mask_weight_tensor = tensorflow.where(
            nan_mask_tensor,
            mask_weight_tensor, tensorflow.zeros_like(mask_weight_tensor)
        )

        baseline_dwcrps = (
            K.sum(mask_weight_tensor * error_tensor) /
            K.sum(mask_weight_tensor * K.ones_like(error_tensor))
        )

        # Return negative skill score.
        return (actual_dwcrps - baseline_dwcrps) / baseline_dwcrps

    loss.__name__ = function_name
    return loss


def dual_weighted_mse_1channel(
        channel_weight, channel_index,
        u_wind_index, v_wind_index, gust_index,
        temperature_index, dewpoint_index,
        function_name, dual_weight_exponent=1.,
        expect_ensemble=True, test_mode=False):
    """Creates DWMSE loss function for one channel (target variable).

    :param channel_weight: Channel weight.
    :param channel_index: Channel index.
    :param u_wind_index: Same.
    :param v_wind_index: Same.
    :param gust_index: Same.
    :param temperature_index: Same.
    :param dewpoint_index: Same.
    :param function_name: See doc for `mean_squared_error`.
    :param dual_weight_exponent: See doc for `dual_weighted_mse`.
    :param expect_ensemble: Same.
    :param test_mode: Same.
    :return: loss: Loss function (defined below).
    """

    check_index_args(
        u_wind_index=u_wind_index,
        v_wind_index=v_wind_index,
        gust_index=gust_index,
        temperature_index=temperature_index,
        dewpoint_index=dewpoint_index
    )

    error_checking.assert_is_greater(channel_weight, 0.)
    error_checking.assert_is_integer(channel_index)
    error_checking.assert_is_geq(channel_index, 0)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(test_mode)
    error_checking.assert_is_geq(dual_weight_exponent, 1.)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (one-channel DWMSE).

        :param target_tensor: See doc for `mean_squared_error`.
        :param prediction_tensor: Same.
        :return: scalar_dwmse: DWMSE (a scalar value).
        """

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)
        relevant_target_tensor = target_tensor[..., :-1]
        mask_weight_tensor = K.expand_dims(target_tensor[..., -1], axis=-1)

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

        dual_weight_tensor = K.pow(
            K.maximum(
                K.abs(relevant_target_tensor),
                K.abs(relevant_prediction_tensor)
            ),
            dual_weight_exponent
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
