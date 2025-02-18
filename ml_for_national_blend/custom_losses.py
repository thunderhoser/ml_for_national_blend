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

PASCALS_TO_MB = 0.01
CELSIUS_TO_KELVINS_ADDEND = 273.15

BASE_VAPOUR_PRESSURE_PASCALS = 610.78
MAGNUS_NUMERATOR_COEFF_WATER = 17.08085
MAGNUS_NUMERATOR_COEFF_ICE = 17.84362
MAGNUS_DENOMINATOR_COEFF_WATER = 234.175
MAGNUS_DENOMINATOR_COEFF_ICE = 245.425


def __get_num_target_fields(prediction_tensor, expect_ensemble):
    """Determines number of target fields.

    :param prediction_tensor: See documentation for `dual_weighted_mse`.
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


def _dewpoint_to_vapour_pressure(dewpoint_tensor_kelvins,
                                 temperature_tensor_kelvins):
    """Converts dewpoint to vapour pressure.

    Source:
    https://content.meteoblue.com/hu/specifications/weather-variables/humidity

    :param dewpoint_tensor_kelvins: Tensor of dewpoints.
    :param temperature_tensor_kelvins: Tensor of temperatures (must have same
        shape as `dewpoint_tensor_kelvins`).
    :return: vapour_pressure_tensor_pascals: Tensor of vapour pressures (with
        same shape as both input tensors).
    """

    temperature_tensor_kelvins = K.maximum(temperature_tensor_kelvins, 0.)
    dewpoint_tensor_kelvins = K.maximum(dewpoint_tensor_kelvins, 0.)
    dewpoint_tensor_kelvins = K.minimum(
        dewpoint_tensor_kelvins, temperature_tensor_kelvins
    )

    temperature_tensor_deg_c = (
        temperature_tensor_kelvins - CELSIUS_TO_KELVINS_ADDEND
    )
    dewpoint_tensor_deg_c = (
        dewpoint_tensor_kelvins - CELSIUS_TO_KELVINS_ADDEND
    )

    numerator_coeff_tensor = tensorflow.where(
        temperature_tensor_deg_c >= 0,
        MAGNUS_NUMERATOR_COEFF_WATER,
        MAGNUS_NUMERATOR_COEFF_ICE
    )
    denominator_coeff_tensor = tensorflow.where(
        temperature_tensor_deg_c >= 0,
        MAGNUS_DENOMINATOR_COEFF_WATER,
        MAGNUS_DENOMINATOR_COEFF_ICE
    )
    denominator_tensor = denominator_coeff_tensor + dewpoint_tensor_deg_c
    exponential_arg_tensor = (
        numerator_coeff_tensor * dewpoint_tensor_deg_c / denominator_tensor
    )

    vapour_pressure_tensor_pascals = (
        BASE_VAPOUR_PRESSURE_PASCALS * K.exp(exponential_arg_tensor)
    )
    vapour_pressure_tensor_pascals = tensorflow.where(
        denominator_tensor <= 0.,
        0.,
        vapour_pressure_tensor_pascals
    )
    vapour_pressure_tensor_pascals = tensorflow.where(
        tf_math.is_finite(vapour_pressure_tensor_pascals),
        vapour_pressure_tensor_pascals,
        0.
    )

    return K.minimum(vapour_pressure_tensor_pascals, 10000.)


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

    :param prediction_tensor: Tensor of predicted values.  For an ensemble
        model, dimensions should be E x M x N x T x S.
        Otherwise, dimensions should be E x M x N x T.
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
    gust minus sustained -- and then converts them to actual gust speeds.

    :param prediction_tensor: See doc for `process_dewpoint_predictions`.
    :param u_wind_index: Array index for u-wind.  This tells the method that
        u-wind predictions can be found in
        prediction_tensor[:, :, :, u_wind_index, ...].
    :param v_wind_index: Same but for v-wind.
    :param gust_index: Same but for gust (excess).
    :return: prediction_tensor: Same as input, except that
        prediction_tensor[:, :, :, gust_index, ...] now contains gust speeds
        and not gust excesses.
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

    gust_excess_prediction_tensor = (
        prediction_tensor[:, :, :, gust_index, ...]
    )
    gust_speed_prediction_tensor = gust_excess_prediction_tensor + K.sqrt(
        prediction_tensor[:, :, :, u_wind_index, ...] ** 2 +
        prediction_tensor[:, :, :, v_wind_index, ...] ** 2
    )

    prediction_tensor = K.concatenate([
        prediction_tensor[:, :, :, :gust_index, ...],
        K.expand_dims(gust_speed_prediction_tensor, axis=3),
        prediction_tensor[:, :, :, (gust_index + 1):, ...]
    ], axis=3)

    return prediction_tensor


def compute_hdwi(prediction_tensor, target_tensor, u_wind_index, v_wind_index,
                 temperature_index, dewpoint_index):
    """Computes hot-dry-windy index (HDWI) in both predictions and targets.

    E = number of examples
    M = number of grid rows
    N = number of grid columns
    T = number of target variables (channels)
    S = ensemble size

    WARNING: This method assumes that you have already run
    `process_dewpoint_predictions` and `process_gust_predictions`, to convert
    dewpoint depressions into actual dewpoints and gust excesses into actual
    gust speeds.  Furthermore, this method assumes that `target_tensor`
    contains only the target fields and nothing else.

    :param prediction_tensor: Tensor of predicted values.  For an ensemble
        model, dimensions should be E x M x N x T x S.
        Otherwise, dimensions should be E x M x N x T.
    :param target_tensor: Tensor of target values, with dimensions
        E x M x N x T.
    :param u_wind_index: Array index for u-wind.  This tells the method that
        u-wind predictions can be found in
        prediction_tensor[:, :, :, u_wind_index, ...] and u-wind targets can be
        found in target_tensor[:, :, :, u_wind_index].
    :param v_wind_index: Same as above but for v-wind.
    :param temperature_index: Same as above but for temperature.
    :param dewpoint_index: Same as above but for dewpoint.
    :return: prediction_tensor: Same as input but with dimensions
        E x M x N x (T + 1) x S or E x M x N x (T + 1).  The extra slice along
        the channel axis contains HDWI values.
    :return: target_tensor: Same as input but with dimensions
        E x M x N x (T + 1).  The extra slice along the channel axis contains
        HDWI values.
    """

    error_checking.assert_is_integer(u_wind_index)
    error_checking.assert_is_geq(u_wind_index, 0)
    error_checking.assert_is_integer(v_wind_index)
    error_checking.assert_is_geq(v_wind_index, 0)
    error_checking.assert_is_integer(temperature_index)
    error_checking.assert_is_geq(temperature_index, 0)
    error_checking.assert_is_integer(dewpoint_index)
    error_checking.assert_is_geq(dewpoint_index, 0)

    unique_indices = set([
        u_wind_index, v_wind_index, temperature_index, dewpoint_index
    ])
    assert len(unique_indices) == 4

    pred_vapour_press_tensor_pascals = _dewpoint_to_vapour_pressure(
        dewpoint_tensor_kelvins=prediction_tensor[:, :, :, dewpoint_index, ...],
        temperature_tensor_kelvins=
        prediction_tensor[:, :, :, temperature_index, ...]
    )
    pred_sat_vapour_press_tensor_pascals = _dewpoint_to_vapour_pressure(
        dewpoint_tensor_kelvins=
        prediction_tensor[:, :, :, temperature_index, ...],
        temperature_tensor_kelvins=
        prediction_tensor[:, :, :, temperature_index, ...]
    )

    pred_vapour_press_depr_tensor_mb = PASCALS_TO_MB * (
        pred_sat_vapour_press_tensor_pascals - pred_vapour_press_tensor_pascals
    )
    pred_wind_speed_tensor_m_s01 = K.sqrt(
        prediction_tensor[:, :, :, u_wind_index, ...] ** 2 +
        prediction_tensor[:, :, :, v_wind_index, ...] ** 2
    )
    pred_hdwi_tensor = (
        pred_wind_speed_tensor_m_s01 * pred_vapour_press_depr_tensor_mb
    )

    prediction_tensor = K.concatenate([
        prediction_tensor,
        K.expand_dims(pred_hdwi_tensor, axis=3)
    ], axis=3)

    target_vapour_press_tensor_pascals = _dewpoint_to_vapour_pressure(
        dewpoint_tensor_kelvins=target_tensor[..., dewpoint_index],
        temperature_tensor_kelvins=target_tensor[..., temperature_index]
    )
    target_sat_vapour_press_tensor_pascals = _dewpoint_to_vapour_pressure(
        dewpoint_tensor_kelvins=target_tensor[..., temperature_index],
        temperature_tensor_kelvins=target_tensor[..., temperature_index]
    )

    target_vapour_press_depr_tensor_mb = PASCALS_TO_MB * (
        target_sat_vapour_press_tensor_pascals -
        target_vapour_press_tensor_pascals
    )
    target_wind_speed_tensor_m_s01 = K.sqrt(
        target_tensor[..., u_wind_index] ** 2 +
        target_tensor[..., v_wind_index] ** 2
    )
    target_hdwi_tensor = (
        target_wind_speed_tensor_m_s01 * target_vapour_press_depr_tensor_mb
    )

    target_tensor = K.concatenate([
        target_tensor,
        K.expand_dims(target_hdwi_tensor, axis=3)
    ], axis=3)

    return prediction_tensor, target_tensor


def check_index_args(u_wind_index, v_wind_index, gust_index, temperature_index,
                     dewpoint_index):
    """Error-checks index arguments.

    :param u_wind_index: Array index for u-wind.  This tells the method that
        u-wind predictions can be found in
        prediction_tensor[:, :, :, u_wind_index, ...] and u-wind targets can be
        found in target_tensor[:, :, :, u_wind_index].
    :param v_wind_index: Same as above but for v-wind.
    :param gust_index: Same as above but for wind gust.
    :param temperature_index: Same as above but for temperature.
    :param dewpoint_index: Same as above but for dewpoint.
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


def dual_weighted_mse(
        channel_weights, u_wind_index, v_wind_index, gust_index,
        temperature_index, dewpoint_index, function_name,
        dual_weight_exponent=1., expect_ensemble=True, include_hdwi=False,
        test_mode=False):
    """Creates dual-weighted mean squared error (DWMSE) loss function.

    T = number of target variables (channels).  If `include_hdwi == True`,
    this number includes HDWI.

    :param channel_weights: length-T numpy array of channel weights.
    :param u_wind_index: Array index for u-wind.  This tells the method that
        u-wind predictions can be found in
        prediction_tensor[:, :, :, u_wind_index, ...] and u-wind targets can be
        found in target_tensor[:, :, :, u_wind_index].
    :param v_wind_index: Same as above but for v-wind.
    :param gust_index: Same as above but for wind gust.
    :param temperature_index: Same as above but for temperature.
    :param dewpoint_index: Same as above but for dewpoint.
    :param function_name: Name of function (string).
    :param dual_weight_exponent: Exponent for dual weight.  If 1, the weight for
        every data point will be max(abs(target), abs(prediction)).  If the
        exponent is E, this weight will be
        max(abs(target), abs(prediction)) ** E.
    :param expect_ensemble: Boolean flag.  If True, will assume that
        `prediction_tensor` contains an ensemble for every grid point and
        variable.  If False, will assume that `prediction_tensor` contains one
        deterministic forecast for every grid point and variable.
    :param include_hdwi: Boolean flag.  If True, will include hot-dry-windy
        index (HDWI) in the target variables.
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

    error_checking.assert_is_numpy_array(channel_weights, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(channel_weights, 0.)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_geq(dual_weight_exponent, 1.)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(include_hdwi)
    error_checking.assert_is_boolean(test_mode)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (DWMSE).

        E = number of examples
        M = number of grid rows
        N = number of grid columns
        T = number of target variables (channels)
        S = ensemble size

        :param prediction_tensor: Tensor of predicted values.  For an ensemble
            model, dimensions should be E x M x N x T x S.
            Otherwise, dimensions should be E x M x N x T.
        :param target_tensor: Tensor of target values, with dimensions
            E x M x N x (T + 1).  target_tensor[..., :-1] contains actual target
            values, and target_tensor[..., -1] contains a binary mask for
            evaluation.
        :return: scalar_dwmse: DWMSE (a scalar value).
        """

        if include_hdwi or dewpoint_index >= 0:
            prediction_tensor = process_dewpoint_predictions(
                prediction_tensor=prediction_tensor,
                temperature_index=temperature_index,
                dewpoint_index=dewpoint_index
            )

        if gust_index >= 0:
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

        if include_hdwi:
            prediction_tensor, relevant_target_tensor = compute_hdwi(
                prediction_tensor=prediction_tensor,
                target_tensor=relevant_target_tensor,
                u_wind_index=u_wind_index,
                v_wind_index=v_wind_index,
                temperature_index=temperature_index,
                dewpoint_index=dewpoint_index
            )

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
        dual_weight_exponent=1., expect_ensemble=True, include_hdwi=False,
        test_mode=False):
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
    :param include_hdwi: Same.
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
    error_checking.assert_is_boolean(include_hdwi)
    error_checking.assert_is_boolean(test_mode)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (DWMSE skill score).

        :param target_tensor: See doc for `dual_weighted_mse`.
        :param prediction_tensor: Same.
        :return: scalar_dwmsess: DWMSE skill score (a scalar value).
        """

        if include_hdwi or dewpoint_index >= 0:
            prediction_tensor = process_dewpoint_predictions(
                prediction_tensor=prediction_tensor,
                temperature_index=temperature_index,
                dewpoint_index=dewpoint_index
            )

        if gust_index >= 0:
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

        if include_hdwi:
            prediction_tensor, relevant_target_tensor = compute_hdwi(
                prediction_tensor=prediction_tensor,
                target_tensor=relevant_target_tensor,
                u_wind_index=u_wind_index,
                v_wind_index=v_wind_index,
                temperature_index=temperature_index,
                dewpoint_index=dewpoint_index
            )
            prediction_tensor, relevant_baseline_prediction_tensor = (
                compute_hdwi(
                    prediction_tensor=prediction_tensor,
                    target_tensor=relevant_baseline_prediction_tensor,
                    u_wind_index=u_wind_index,
                    v_wind_index=v_wind_index,
                    temperature_index=temperature_index,
                    dewpoint_index=dewpoint_index
                )
            )
            prediction_tensor = prediction_tensor[:, :, :, :-1, ...]

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
        dual_weight_exponent=1., include_hdwi=False, test_mode=False):
    """Creates dual-weighted-CRPSS loss function.

    :param channel_weights: See documentation for `dual_weighted_mse`.
    :param u_wind_index: Same.
    :param v_wind_index: Same.
    :param gust_index: Same.
    :param temperature_index: Same.
    :param dewpoint_index: Same.
    :param function_name: Same.
    :param dual_weight_exponent: Same.
    :param include_hdwi: Same.
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
    error_checking.assert_is_boolean(include_hdwi)
    error_checking.assert_is_boolean(test_mode)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (dual-weighted CRPSS).

        :param target_tensor: See doc for `dual_weighted_mse`.
        :param prediction_tensor: Same.
        :return: scalar_dual_weighted_crpss: Dual-weighted CRPSS (a scalar
            value).
        """

        # E x M x N x T x S
        if include_hdwi or dewpoint_index >= 0:
            prediction_tensor = process_dewpoint_predictions(
                prediction_tensor=prediction_tensor,
                temperature_index=temperature_index,
                dewpoint_index=dewpoint_index
            )

        # E x M x N x T x S
        if gust_index >= 0:
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
        relevant_target_tensor = target_tensor[..., :num_target_fields]  # E x M x N x T
        relevant_baseline_prediction_tensor = (
            target_tensor[..., num_target_fields:-1]  # E x M x N x T
        )
        mask_weight_tensor = K.expand_dims(target_tensor[..., -1], axis=-1)  # E x M x N x 1

        if include_hdwi:
            prediction_tensor, relevant_target_tensor = compute_hdwi(
                prediction_tensor=prediction_tensor,
                target_tensor=relevant_target_tensor,
                u_wind_index=u_wind_index,
                v_wind_index=v_wind_index,
                temperature_index=temperature_index,
                dewpoint_index=dewpoint_index
            )
            prediction_tensor, relevant_baseline_prediction_tensor = (
                compute_hdwi(
                    prediction_tensor=prediction_tensor,
                    target_tensor=relevant_baseline_prediction_tensor,
                    u_wind_index=u_wind_index,
                    v_wind_index=v_wind_index,
                    temperature_index=temperature_index,
                    dewpoint_index=dewpoint_index
                )
            )
            prediction_tensor = prediction_tensor[:, :, :, :-1, ...]

        # Ensure compatible tensor shapes.
        relevant_target_tensor = K.expand_dims(relevant_target_tensor, axis=-1)  # E x M x N x T x 1
        relevant_baseline_prediction_tensor = K.expand_dims(
            relevant_baseline_prediction_tensor, axis=-1  # E x M x N x T x 1
        )

        # Create dual-weight tensor.
        dual_weight_tensor = K.pow(
            K.maximum(K.abs(relevant_target_tensor), K.abs(prediction_tensor)),
            dual_weight_exponent
        )
        dual_weight_tensor = K.maximum(dual_weight_tensor, 1.)  # E x M x N x T x S

        # Create channel-weight tensor.
        channel_weight_tensor = K.cast(
            K.constant(channel_weights), dual_weight_tensor.dtype
        )
        for _ in range(3):
            channel_weight_tensor = K.expand_dims(channel_weight_tensor, axis=0)  # 1 x 1 x 1 x T

        # Compute dual-weighted CRPS.
        absolute_error_tensor = K.abs(
            prediction_tensor - relevant_target_tensor  # E x M x N x T x S
        )
        mean_prediction_error_tensor = K.mean(
            dual_weight_tensor * absolute_error_tensor, axis=-1  # E x M x N x T
        )

        # M x E x N x T x S
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
        mapd_tensor = mapd_tensor.stack()  # M x E x N x T

        mapd_tensor = tensorflow.transpose(
            mapd_tensor, perm=[1, 0, 2, 3]  # E x M x N x T
        )
        error_tensor = channel_weight_tensor * (  # E x M x N x T
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
        dual_weight_tensor = K.maximum(dual_weight_tensor, 1.)  # E x M x N x T x 1

        # Compute dual-weighted CRPSS for baseline.
        absolute_error_tensor = K.abs(
            relevant_baseline_prediction_tensor - relevant_target_tensor  # E x M x N x T x 1
        )
        mean_prediction_error_tensor = K.mean(
            dual_weight_tensor * absolute_error_tensor, axis=-1  # E x M x N x T
        )
        error_tensor = channel_weight_tensor * mean_prediction_error_tensor  # E x M x N x T

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
    :param channel_index: Channel index or "hdwi".
    :param u_wind_index: See doc for `dual_weighted_mse`.
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

    error_checking.assert_is_greater(channel_weight, 0.)
    error_checking.assert_is_string(function_name)
    error_checking.assert_is_boolean(expect_ensemble)
    error_checking.assert_is_boolean(test_mode)
    error_checking.assert_is_geq(dual_weight_exponent, 1.)

    if channel_index != 'hdwi':
        error_checking.assert_is_integer(channel_index)
        error_checking.assert_is_geq(channel_index, 0)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (one-channel DWMSE).

        :param target_tensor: See doc for `dual_weighted_mse`.
        :param prediction_tensor: Same.
        :return: scalar_dwmse: DWMSE (a scalar value).
        """

        target_tensor = K.cast(target_tensor, prediction_tensor.dtype)
        relevant_target_tensor = target_tensor[..., :-1]
        mask_weight_tensor = K.expand_dims(target_tensor[..., -1], axis=-1)

        if channel_index in ['hdwi', dewpoint_index]:
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

        if channel_index == 'hdwi':
            prediction_tensor, relevant_target_tensor = compute_hdwi(
                prediction_tensor=prediction_tensor,
                target_tensor=relevant_target_tensor,
                u_wind_index=u_wind_index,
                v_wind_index=v_wind_index,
                temperature_index=temperature_index,
                dewpoint_index=dewpoint_index
            )
            new_channel_index = -1
        else:
            new_channel_index = channel_index + 0

        if expect_ensemble:
            relevant_target_tensor = K.expand_dims(
                relevant_target_tensor[..., new_channel_index], axis=-1
            )
            relevant_prediction_tensor = (
                prediction_tensor[:, :, :, new_channel_index, :]
            )
        else:
            relevant_target_tensor = (
                relevant_target_tensor[..., new_channel_index]
            )
            relevant_prediction_tensor = (
                prediction_tensor[:, :, :, new_channel_index]
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
