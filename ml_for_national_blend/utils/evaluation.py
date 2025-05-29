"""Model-evaluation methods."""

import copy
import numpy
import xarray
from numba import njit
from scipy.stats import ks_2samp
from ml_for_national_blend.outside_code import temperature_conversions as temperature_conv
from ml_for_national_blend.outside_code import file_system_utils
from ml_for_national_blend.outside_code import error_checking
from ml_for_national_blend.io import urma_io
from ml_for_national_blend.io import prediction_io
from ml_for_national_blend.utils import urma_utils

# TODO(thunderhoser): Allow multiple lead times.

TOLERANCE = 1e-6
NUM_BINS_FOR_SSREL = 100

RELIABILITY_BIN_DIM = 'reliability_bin'
BOOTSTRAP_REP_DIM = 'bootstrap_replicate'
ROW_DIM = 'grid_row'
COLUMN_DIM = 'grid_column'
FIELD_DIM = 'field'

LATITUDE_KEY = 'latitude_deg_n'
LONGITUDE_KEY = 'longitude_deg_e'

TARGET_STDEV_KEY = 'target_standard_deviation'
PREDICTION_STDEV_KEY = 'prediction_standard_deviation'
TARGET_MEAN_KEY = 'target_mean'
PREDICTION_MEAN_KEY = 'prediction_mean'

MSE_KEY = 'mean_squared_error'
MSE_BIAS_KEY = 'mse_bias'
MSE_VARIANCE_KEY = 'mse_variance'
MSE_SKILL_SCORE_KEY = 'mse_skill_score'
DWMSE_KEY = 'dual_weighted_mean_squared_error'
DWMSE_SKILL_SCORE_KEY = 'dwmse_skill_score'

KS_STATISTIC_KEY = 'kolmogorov_smirnov_statistic'
KS_P_VALUE_KEY = 'kolmogorov_smirnov_p_value'

MAE_KEY = 'mean_absolute_error'
MAE_SKILL_SCORE_KEY = 'mae_skill_score'
BIAS_KEY = 'bias'
SSRAT_KEY = 'spread_skill_ratio'
SSDIFF_KEY = 'spread_skill_difference'
SSREL_KEY = 'spread_skill_reliability'
SPATIAL_MIN_BIAS_KEY = 'spatial_min_bias'
SPATIAL_MAX_BIAS_KEY = 'spatial_max_bias'
CORRELATION_KEY = 'correlation'
KGE_KEY = 'kling_gupta_efficiency'
RELIABILITY_KEY = 'reliability'
RELIABILITY_X_KEY = 'reliability_x'
RELIABILITY_Y_KEY = 'reliability_y'
RELIABILITY_BIN_CENTER_KEY = 'reliability_bin_center'
RELIABILITY_COUNT_KEY = 'reliability_count'
INV_RELIABILITY_BIN_CENTER_KEY = 'inv_reliability_bin_center'
INV_RELIABILITY_COUNT_KEY = 'inv_reliability_count'

MODEL_FILE_KEY = 'model_file_name'
PREDICTION_FILES_KEY = 'prediction_file_names'


@njit
def __find_examples_in_each_bin(example_to_bin_indices, num_bins):
    """Creates list of data examples in each bin.

    E = number of examples
    B = number of bins

    :param example_to_bin_indices: length-E numpy array indexing examples to
        bins.  example_to_bin_indices[i] = j indicates that the (i)th example is
        a member of the (j)th bin.
    :param num_bins: Total number of bins (B in the above definitions).
    :return: example_indices_by_bin: length-B list, where the (j)th element is a
        1-D numpy array with the indices of examples belonging to the (j)th bin.
    """

    bin_sizes = numpy.zeros(num_bins, dtype=numpy.int64)
    for j in example_to_bin_indices:
        bin_sizes[j] += 1

    example_indices_by_bin = [
        numpy.empty(bs, dtype=numpy.int64) for bs in bin_sizes
    ]

    bin_counters = numpy.zeros(num_bins, dtype=numpy.int64)
    for i, j in enumerate(example_to_bin_indices):
        example_indices_by_bin[j][bin_counters[j]] = i
        bin_counters[j] += 1

    return example_indices_by_bin


def _get_mse_one_scalar(target_values, predicted_values, per_grid_cell):
    """Computes mean squared error (MSE) for one scalar target variable.

    :param target_values: numpy array of target (actual) values.
    :param predicted_values: numpy array of predicted values with the same shape
        as `target_values`.
    :param per_grid_cell: Boolean flag.  If True, will compute a separate set of
        scores at each grid cell.  If False, will compute one set of scores for
        the whole domain.
    :return: mse_total: Total MSE.
    :return: mse_bias: Bias component.
    :return: mse_variance: Variance component.
    """

    if per_grid_cell:
        mse_total = numpy.nanmean(
            (target_values - predicted_values) ** 2, axis=0
        )
        mse_bias = numpy.nanmean(target_values - predicted_values, axis=0) ** 2
    else:
        mse_total = numpy.nanmean((target_values - predicted_values) ** 2)
        mse_bias = numpy.nanmean(target_values - predicted_values) ** 2

    return mse_total, mse_bias, mse_total - mse_bias


def _get_mse_ss_one_scalar(target_values, predicted_values, per_grid_cell,
                           mean_training_target_value):
    """Computes MSE skill score for one scalar target variable.

    :param target_values: See doc for `_get_mse_one_scalar`.
    :param predicted_values: Same.
    :param per_grid_cell: Same.
    :param mean_training_target_value: Mean target value over all training
        examples.
    :return: mse_skill_score: Self-explanatory.
    """

    mse_actual = _get_mse_one_scalar(
        target_values=target_values,
        predicted_values=predicted_values,
        per_grid_cell=per_grid_cell
    )[0]
    mse_climo = _get_mse_one_scalar(
        target_values=target_values,
        predicted_values=mean_training_target_value,
        per_grid_cell=per_grid_cell
    )[0]

    return (mse_climo - mse_actual) / mse_climo


def _get_dwmse_one_scalar(target_values, predicted_values, per_grid_cell):
    """Computes dual-weighted MSE (DWMSE) for one scalar target variable.

    :param target_values: See doc for `_get_mse_one_scalar`.
    :param predicted_values: Same.
    :param per_grid_cell: Same.
    :return: dwmse: Self-explanatory.
    """

    dual_weights = numpy.maximum(
        numpy.absolute(target_values),
        numpy.absolute(predicted_values)
    )

    if per_grid_cell:
        return numpy.nanmean(
            dual_weights * (target_values - predicted_values) ** 2,
            axis=0
        )

    return numpy.nanmean(
        dual_weights * (target_values - predicted_values) ** 2
    )


def _get_dwmse_ss_one_scalar(target_values, predicted_values, per_grid_cell,
                             mean_training_target_value):
    """Computes DWMSE skill score for one scalar target variable.

    :param target_values: See doc for `_get_dwmse_one_scalar`.
    :param predicted_values: Same.
    :param per_grid_cell: Same.
    :param mean_training_target_value: Mean target value over all training
        examples.
    :return: dwmse_skill_score: Self-explanatory.
    """

    dwmse_actual = _get_dwmse_one_scalar(
        target_values=target_values,
        predicted_values=predicted_values,
        per_grid_cell=per_grid_cell
    )
    dwmse_climo = _get_dwmse_one_scalar(
        target_values=target_values,
        predicted_values=numpy.array([mean_training_target_value]),
        per_grid_cell=per_grid_cell
    )

    return (dwmse_climo - dwmse_actual) / dwmse_climo


def _get_mae_one_scalar(target_values, predicted_values, per_grid_cell):
    """Computes mean absolute error (MAE) for one scalar target variable.

    :param target_values: See doc for `_get_mse_one_scalar`.
    :param predicted_values: Same.
    :param per_grid_cell: Same.
    :return: mean_absolute_error: Self-explanatory.
    """

    if per_grid_cell:
        return numpy.nanmean(
            numpy.absolute(target_values - predicted_values),
            axis=0
        )

    return numpy.nanmean(
        numpy.absolute(target_values - predicted_values)
    )


def _get_mae_ss_one_scalar(target_values, predicted_values, per_grid_cell,
                           mean_training_target_value):
    """Computes MAE skill score for one scalar target variable.

    :param target_values: See doc for `_get_mse_one_scalar`.
    :param predicted_values: Same.
    :param per_grid_cell: Same.
    :param mean_training_target_value: See doc for `_get_mse_ss_one_scalar`.
    :return: mae_skill_score: Self-explanatory.
    """

    mae_actual = _get_mae_one_scalar(
        target_values=target_values,
        predicted_values=predicted_values,
        per_grid_cell=per_grid_cell
    )
    mae_climo = _get_mae_one_scalar(
        target_values=target_values,
        predicted_values=mean_training_target_value,
        per_grid_cell=per_grid_cell
    )

    return (mae_climo - mae_actual) / mae_climo


def _get_bias_one_scalar(target_values, predicted_values, per_grid_cell):
    """Computes bias (mean signed error) for one scalar target variable.

    :param target_values: See doc for `_get_mse_one_scalar`.
    :param predicted_values: Same.
    :param per_grid_cell: Same.
    :return: bias: Self-explanatory.
    """

    if per_grid_cell:
        return numpy.nanmean(predicted_values - target_values, axis=0)

    return numpy.nanmean(predicted_values - target_values)


def _get_spatial_min_bias_one_field(target_matrix, prediction_matrix):
    """Computes bias in spatial minimum for one target field.

    E = number of examples (time steps)
    M = number of rows in grid
    N = number of columns in grid

    :param target_matrix: E-by-M-by-N numpy array of actual values.
    :param prediction_matrix: E-by-M-by-N numpy array of predicted values.
    :return: spatial_min_bias: Self-explanatory.
    """

    min_target_values = numpy.nanmin(target_matrix, axis=(1, 2))
    min_predicted_values = numpy.nanmin(prediction_matrix, axis=(1, 2))
    return numpy.mean(min_predicted_values - min_target_values)


def _get_spatial_max_bias_one_field(target_matrix, prediction_matrix):
    """Computes bias in spatial maximum for one target field.

    :param target_matrix: See doc for `_get_spatial_min_bias_one_field`.
    :param prediction_matrix: Same.
    :return: spatial_max_bias: Self-explanatory.
    """

    max_target_values = numpy.nanmax(target_matrix, axis=(1, 2))
    max_predicted_values = numpy.nanmax(prediction_matrix, axis=(1, 2))
    return numpy.mean(max_predicted_values - max_target_values)


def _get_correlation_one_scalar(target_values, predicted_values, per_grid_cell):
    """Computes Pearson correlation for one scalar target variable.

    :param target_values: See doc for `_get_mse_one_scalar`.
    :param predicted_values: Same.
    :param per_grid_cell: Same.
    :return: correlation: Self-explanatory.
    """

    if per_grid_cell:
        numerator = numpy.nansum(
            (target_values - numpy.nanmean(target_values)) *
            (predicted_values - numpy.nanmean(predicted_values)),
            axis=0
        )
        sum_squared_target_diffs = numpy.nansum(
            (target_values - numpy.nanmean(target_values)) ** 2,
            axis=0
        )
        sum_squared_prediction_diffs = numpy.nansum(
            (predicted_values - numpy.nanmean(predicted_values)) ** 2,
            axis=0
        )
    else:
        numerator = numpy.nansum(
            (target_values - numpy.nanmean(target_values)) *
            (predicted_values - numpy.nanmean(predicted_values))
        )
        sum_squared_target_diffs = numpy.nansum(
            (target_values - numpy.nanmean(target_values)) ** 2
        )
        sum_squared_prediction_diffs = numpy.nansum(
            (predicted_values - numpy.nanmean(predicted_values)) ** 2
        )

    correlation = (
        numerator /
        numpy.sqrt(sum_squared_target_diffs * sum_squared_prediction_diffs)
    )

    return correlation


def _get_kge_one_scalar(target_values, predicted_values, per_grid_cell):
    """Computes KGE (Kling-Gupta efficiency) for one scalar target variable.

    :param target_values: See doc for `_get_mse_one_scalar`.
    :param predicted_values: Same.
    :param per_grid_cell: Same.
    :return: kge: Self-explanatory.
    """

    correlation = _get_correlation_one_scalar(
        target_values=target_values,
        predicted_values=predicted_values,
        per_grid_cell=per_grid_cell
    )

    if per_grid_cell:
        mean_target_value = numpy.nanmean(target_values, axis=0)
        mean_predicted_value = numpy.nanmean(predicted_values, axis=0)
        stdev_target_value = numpy.nanstd(target_values, ddof=1, axis=0)
        stdev_predicted_value = numpy.nanstd(predicted_values, ddof=1, axis=0)
    else:
        mean_target_value = numpy.nanmean(target_values)
        mean_predicted_value = numpy.nanmean(predicted_values)
        stdev_target_value = numpy.nanstd(target_values, ddof=1)
        stdev_predicted_value = numpy.nanstd(predicted_values, ddof=1)

    variance_bias = (
        (stdev_predicted_value / mean_predicted_value) *
        (stdev_target_value / mean_target_value) ** -1
    )
    mean_bias = mean_predicted_value / mean_target_value

    kge = 1. - numpy.sqrt(
        (correlation - 1.) ** 2 +
        (variance_bias - 1.) ** 2 +
        (mean_bias - 1.) ** 2
    )

    return kge


def _get_rel_curve_one_scalar(
        target_values, predicted_values,
        num_bins, min_bin_edge, max_bin_edge, invert=False):
    """Computes reliability curve for one scalar target variable.

    B = number of bins

    :param target_values: See doc for `_get_mse_one_scalar`.
    :param predicted_values: Same.
    :param num_bins: Number of bins (points in curve).
    :param min_bin_edge: Value at lower edge of first bin.
    :param max_bin_edge: Value at upper edge of last bin.
    :param invert: Boolean flag.  If True, will return inverted reliability
        curve, which bins by target value and relates target value to
        conditional mean prediction.  If False, will return normal reliability
        curve, which bins by predicted value and relates predicted value to
        conditional mean observation (target).
    :return: mean_predictions: length-B numpy array of x-coordinates.
    :return: mean_observations: length-B numpy array of y-coordinates.
    :return: example_counts: length-B numpy array with num examples in each bin.
    """

    # max_bin_edge = max([max_bin_edge, numpy.finfo(float).eps])
    # min_bin_edge = min([min_bin_edge, 0.])

    real_indices = numpy.where(numpy.invert(numpy.logical_or(
        numpy.isnan(target_values),
        numpy.isnan(predicted_values)
    )))[0]
    real_target_values = target_values[real_indices]
    real_predicted_values = predicted_values[real_indices]

    mean_predictions = numpy.full(num_bins, numpy.nan)
    mean_observations = numpy.full(num_bins, numpy.nan)
    example_counts = numpy.full(num_bins, numpy.nan)

    bin_edges = numpy.linspace(min_bin_edge, max_bin_edge, num=num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    if len(real_indices) == 0:
        return bin_centers, mean_observations, example_counts

    example_to_bin_indices = numpy.digitize(
        real_target_values if invert else real_predicted_values,
        bin_edges,
        right=False
    ) - 1
    example_to_bin_indices = numpy.clip(example_to_bin_indices, 0, num_bins - 1)

    example_indices_by_bin = __find_examples_in_each_bin(
        example_to_bin_indices=example_to_bin_indices,
        num_bins=num_bins
    )

    for j in range(num_bins):
        if len(example_indices_by_bin[j]) == 0:
            continue

        example_counts[j] = len(example_indices_by_bin[j])
        mean_predictions[j] = numpy.mean(
            real_predicted_values[example_indices_by_bin[j]]
        )
        mean_observations[j] = numpy.mean(
            real_target_values[example_indices_by_bin[j]]
        )

    mean_predictions[numpy.isnan(mean_predictions)] = bin_centers[
        numpy.isnan(mean_predictions)
    ]

    return mean_predictions, mean_observations, example_counts


def _get_ss_plot_one_scalar(
        squared_errors, ensemble_variances,
        num_bins, min_bin_edge_stdev, max_bin_edge_stdev):
    """Computes spread-skill plot for one scalar target variable.

    B = number of bins
    E = number of examples

    :param squared_errors: length-E numpy array with squared error of ensemble
        mean for every example.
    :param ensemble_variances: length-E numpy array with ensemble variance for
        every example.
    :param num_bins: Number of bins (points in curve).
    :param min_bin_edge_stdev: Value at lower edge of first bin.  This is an
        ensemble standard deviation, not a variance.
    :param max_bin_edge_stdev: Value at upper edge of last bin.  This is an
        ensemble standard deviation, not a variance.
    :return: mean_ensemble_variances: length-B numpy array of x-coordinates.
    :return: mean_squared_errors: length-B numpy array of y-coordinates.
    :return: example_counts: length-B numpy array with num examples in each bin.
    """

    real_indices = numpy.where(numpy.invert(numpy.logical_or(
        numpy.isnan(squared_errors),
        numpy.isnan(ensemble_variances)
    )))[0]
    real_squared_errors = squared_errors[real_indices]
    real_ensemble_variances = ensemble_variances[real_indices]

    mean_ensemble_variances = numpy.full(num_bins, numpy.nan)
    mean_squared_errors = numpy.full(num_bins, numpy.nan)
    example_counts = numpy.full(num_bins, numpy.nan)

    if len(real_indices) == 0:
        return mean_ensemble_variances, mean_squared_errors, example_counts

    bin_cutoffs = numpy.linspace(
        min_bin_edge_stdev, max_bin_edge_stdev, num=num_bins + 1
    )
    example_to_bin_indices = numpy.digitize(
        numpy.sqrt(real_ensemble_variances), bin_cutoffs, right=False
    ) - 1
    example_to_bin_indices = numpy.clip(example_to_bin_indices, 0, num_bins - 1)

    example_indices_by_bin = __find_examples_in_each_bin(
        example_to_bin_indices=example_to_bin_indices,
        num_bins=num_bins
    )

    for j in range(num_bins):
        if len(example_indices_by_bin[j]) == 0:
            continue

        example_counts[j] = len(example_indices_by_bin[j])
        mean_ensemble_variances[j] = numpy.mean(
            real_ensemble_variances[example_indices_by_bin[j]]
        )
        mean_squared_errors[j] = numpy.mean(
            real_squared_errors[example_indices_by_bin[j]]
        )

    return mean_ensemble_variances, mean_squared_errors, example_counts


def _get_ssrat_one_replicate(
        full_squared_error_matrix, full_prediction_variance_matrix,
        example_indices_in_replicate, per_grid_cell):
    """Computes SSRAT for one bootstrap replicate.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid
    T = number of target fields

    :param full_squared_error_matrix: E-by-M-by-N-by-T numpy array of squared
        errors.
    :param full_prediction_variance_matrix: E-by-M-by-N-by-T numpy array of
        ensemble variances.
    :param example_indices_in_replicate: See documentation for
        `_get_scores_one_replicate`.
    :param per_grid_cell: Same.
    :return: ssrat_matrix: If `per_grid_cell == True`, this is an M-by-N-by-T
        numpy array of spread-skill ratios.  Otherwise, this is a length-T numpy
        array of spread-skill ratios.
    """

    squared_error_matrix = full_squared_error_matrix[
        example_indices_in_replicate, ...
    ]
    prediction_variance_matrix = full_prediction_variance_matrix[
        example_indices_in_replicate, ...
    ]
    num_target_fields = squared_error_matrix.shape[3]

    if per_grid_cell:
        num_grid_rows = squared_error_matrix.shape[1]
        num_grid_columns = squared_error_matrix.shape[2]
        these_dim = (num_grid_rows, num_grid_columns, num_target_fields)
    else:
        these_dim = (num_target_fields,)

    ssrat_matrix = numpy.full(these_dim, numpy.nan)

    for k in range(num_target_fields):
        if per_grid_cell:
            numerator = numpy.sqrt(
                numpy.nanmean(prediction_variance_matrix[..., k], axis=0)
            )
            denominator = numpy.sqrt(
                numpy.nanmean(squared_error_matrix[..., k], axis=0)
            )
        else:
            numerator = numpy.sqrt(
                numpy.nanmean(prediction_variance_matrix[..., k])
            )
            denominator = numpy.sqrt(
                numpy.nanmean(squared_error_matrix[..., k])
            )

        ssrat_matrix[..., k] = numerator / denominator

    return ssrat_matrix


def _get_ssdiff_one_replicate(
        full_squared_error_matrix, full_prediction_variance_matrix,
        example_indices_in_replicate, per_grid_cell):
    """Computes SSDIFF for one bootstrap replicate.

    SSDIFF = spread-skill difference = spread minus skill

    M = number of rows in grid
    N = number of columns in grid
    T = number of target fields

    :param full_squared_error_matrix: See documentation for
        `_get_ssrat_one_replicate`.
    :param full_prediction_variance_matrix: Same.
    :param example_indices_in_replicate: Same.
    :param per_grid_cell: Same.
    :return: ssdiff_matrix: If `per_grid_cell == True`, this is an M-by-N-by-T
        numpy array of spread-skill differences.  Otherwise, this is a length-T
        numpy array of spread-skill differences.
    """

    squared_error_matrix = full_squared_error_matrix[
        example_indices_in_replicate, ...
    ]
    prediction_variance_matrix = full_prediction_variance_matrix[
        example_indices_in_replicate, ...
    ]
    num_target_fields = squared_error_matrix.shape[3]

    if per_grid_cell:
        num_grid_rows = squared_error_matrix.shape[1]
        num_grid_columns = squared_error_matrix.shape[2]
        these_dim = (num_grid_rows, num_grid_columns, num_target_fields)
    else:
        these_dim = (num_target_fields,)

    ssdiff_matrix = numpy.full(these_dim, numpy.nan)

    for k in range(num_target_fields):
        if per_grid_cell:
            numerator = numpy.sqrt(
                numpy.nanmean(prediction_variance_matrix[..., k], axis=0)
            )
            denominator = numpy.sqrt(
                numpy.nanmean(squared_error_matrix[..., k], axis=0)
            )
        else:
            numerator = numpy.sqrt(
                numpy.nanmean(prediction_variance_matrix[..., k])
            )
            denominator = numpy.sqrt(
                numpy.nanmean(squared_error_matrix[..., k])
            )

        ssdiff_matrix[..., k] = numerator - denominator

    return ssdiff_matrix


def _get_ssrel_one_replicate(
        full_squared_error_matrix, full_prediction_variance_matrix,
        example_indices_in_replicate, per_grid_cell):
    """Computes SSREL for one bootstrap replicate.

    SSREL = spread-skill reliability

    M = number of rows in grid
    N = number of columns in grid
    T = number of target fields

    :param full_squared_error_matrix: See documentation for
        `_get_ssrat_one_replicate`.
    :param full_prediction_variance_matrix: Same.
    :param example_indices_in_replicate: Same.
    :param per_grid_cell: Same.
    :return: ssrel_matrix: If `per_grid_cell == True`, this is an M-by-N-by-T
        numpy array of spread-skill reliabilities.  Otherwise, this is a
        length-T numpy array of spread-skill reliabilities.
    """

    squared_error_matrix = full_squared_error_matrix[
        example_indices_in_replicate, ...
    ]
    prediction_variance_matrix = full_prediction_variance_matrix[
        example_indices_in_replicate, ...
    ]
    num_target_fields = squared_error_matrix.shape[3]

    if per_grid_cell:
        num_grid_rows = squared_error_matrix.shape[1]
        num_grid_columns = squared_error_matrix.shape[2]
        these_dim = (num_grid_rows, num_grid_columns, num_target_fields)
    else:
        num_grid_rows = 0
        num_grid_columns = 0
        these_dim = (num_target_fields,)

    ssrel_matrix = numpy.full(these_dim, numpy.nan)

    for k in range(num_target_fields):
        if per_grid_cell:
            for i in range(num_grid_rows):
                print((
                    'Have computed SSREL for {0:d} of {1:d} grid rows...'
                ).format(
                    i, num_grid_rows
                ))

                for j in range(num_grid_columns):
                    min_bin_edge_stdev = numpy.min(
                        numpy.sqrt(prediction_variance_matrix[:, i, j, k])
                    )
                    max_bin_edge_stdev = numpy.max(
                        numpy.sqrt(prediction_variance_matrix[:, i, j, k])
                    )
                    max_bin_edge_stdev = max([
                        max_bin_edge_stdev,
                        min_bin_edge_stdev + TOLERANCE
                    ])

                    (
                        mean_ensemble_variances,
                        mean_squared_errors,
                        example_counts
                    ) = _get_ss_plot_one_scalar(
                        squared_errors=squared_error_matrix[:, i, j, k],
                        ensemble_variances=
                        prediction_variance_matrix[:, i, j, k],
                        num_bins=NUM_BINS_FOR_SSREL,
                        min_bin_edge_stdev=min_bin_edge_stdev,
                        max_bin_edge_stdev=max_bin_edge_stdev
                    )

                    these_diffs = numpy.absolute(
                        mean_ensemble_variances - mean_squared_errors
                    )
                    ssrel_matrix[i, j, k] = (
                        numpy.nansum(example_counts * these_diffs) /
                        numpy.nansum(example_counts)
                    )
        else:
            min_bin_edge_stdev = numpy.min(
                numpy.sqrt(prediction_variance_matrix[..., k])
            )
            max_bin_edge_stdev = numpy.max(
                numpy.sqrt(prediction_variance_matrix[..., k])
            )
            max_bin_edge_stdev = max([
                max_bin_edge_stdev,
                min_bin_edge_stdev + TOLERANCE
            ])

            (
                mean_ensemble_variances, mean_squared_errors, example_counts
            ) = _get_ss_plot_one_scalar(
                squared_errors=numpy.ravel(squared_error_matrix[..., k]),
                ensemble_variances=
                numpy.ravel(prediction_variance_matrix[..., k]),
                num_bins=NUM_BINS_FOR_SSREL,
                min_bin_edge_stdev=min_bin_edge_stdev,
                max_bin_edge_stdev=max_bin_edge_stdev
            )

            these_diffs = numpy.absolute(
                mean_ensemble_variances - mean_squared_errors
            )
            ssrel_matrix[k] = (
                numpy.nansum(example_counts * these_diffs) /
                numpy.nansum(example_counts)
            )

    return ssrel_matrix


def _get_scores_one_replicate(
        result_table_xarray,
        full_target_matrix, full_prediction_matrix,
        replicate_index, example_indices_in_replicate,
        mean_training_target_values, num_relia_bins_by_target,
        min_relia_bin_edge_by_target, max_relia_bin_edge_by_target,
        min_relia_bin_edge_prctile_by_target,
        max_relia_bin_edge_prctile_by_target,
        per_grid_cell, keep_it_simple=False):
    """Computes scores for one bootstrap replicate.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid
    T = number of target fields

    :param result_table_xarray: See doc for `get_scores_with_bootstrapping`.
    :param full_target_matrix: E-by-M-by-N-by-T numpy array of correct values.
    :param full_prediction_matrix: E-by-M-by-N-by-T numpy array of predicted
        values.
    :param replicate_index: Index of current bootstrap replicate.
    :param example_indices_in_replicate: 1-D numpy array with indices of
        examples in this bootstrap replicate.
    :param mean_training_target_values: length-T numpy array with mean target
        values in training data (i.e., "climatology").
    :param num_relia_bins_by_target: See doc for `get_scores_with_bootstrapping`.
    :param min_relia_bin_edge_by_target: Same.
    :param max_relia_bin_edge_by_target: Same.
    :param min_relia_bin_edge_prctile_by_target: Same.
    :param max_relia_bin_edge_prctile_by_target: Same.
    :param per_grid_cell: Same.
    :param keep_it_simple: Same.
    :return: result_table_xarray: Same as input but with values filled for [i]th
        bootstrap replicate, where i = `replicate_index`.
    """

    t = result_table_xarray
    rep_idx = replicate_index + 0
    num_examples = len(example_indices_in_replicate)

    target_matrix = full_target_matrix[example_indices_in_replicate, ...]
    prediction_matrix = full_prediction_matrix[
        example_indices_in_replicate, ...
    ]

    num_target_fields = len(mean_training_target_values)

    if per_grid_cell:
        t[TARGET_STDEV_KEY].values[..., rep_idx] = numpy.nanstd(
            target_matrix, ddof=1, axis=0
        )
        t[PREDICTION_STDEV_KEY].values[..., rep_idx] = numpy.nanstd(
            prediction_matrix, ddof=1, axis=0
        )
        t[TARGET_MEAN_KEY].values[..., rep_idx] = numpy.nanmean(
            target_matrix, axis=0
        )
        t[PREDICTION_MEAN_KEY].values[..., rep_idx] = numpy.nanmean(
            prediction_matrix, axis=0
        )
    else:
        # matchy_target_matrix = target_matrix + 0.
        # matchy_prediction_matrix = prediction_matrix + 0.
        # matchy_target_matrix[numpy.isnan(matchy_prediction_matrix)] = numpy.nan
        # matchy_prediction_matrix[numpy.isnan(matchy_target_matrix)] = numpy.nan
        #
        # t[TARGET_STDEV_KEY].values[:, rep_idx] = numpy.nanstd(
        #     matchy_target_matrix, ddof=1, axis=(0, 1, 2)
        # )
        # t[PREDICTION_STDEV_KEY].values[:, rep_idx] = numpy.nanstd(
        #     matchy_prediction_matrix, ddof=1, axis=(0, 1, 2)
        # )
        # t[TARGET_MEAN_KEY].values[:, rep_idx] = numpy.nanmean(
        #     matchy_target_matrix, axis=(0, 1, 2)
        # )
        # t[PREDICTION_MEAN_KEY].values[:, rep_idx] = numpy.nanmean(
        #     matchy_prediction_matrix, axis=(0, 1, 2)
        # )

        t[TARGET_STDEV_KEY].values[:, rep_idx] = numpy.nanstd(
            target_matrix, ddof=1, axis=(0, 1, 2)
        )
        t[PREDICTION_STDEV_KEY].values[:, rep_idx] = numpy.nanstd(
            prediction_matrix, ddof=1, axis=(0, 1, 2)
        )
        t[TARGET_MEAN_KEY].values[:, rep_idx] = numpy.nanmean(
            target_matrix, axis=(0, 1, 2)
        )
        t[PREDICTION_MEAN_KEY].values[:, rep_idx] = numpy.nanmean(
            prediction_matrix, axis=(0, 1, 2)
        )

    for k in range(num_target_fields):
        t[MAE_KEY].values[..., k, rep_idx] = _get_mae_one_scalar(
            target_values=target_matrix[..., k],
            predicted_values=prediction_matrix[..., k],
            per_grid_cell=per_grid_cell
        )
        t[MAE_SKILL_SCORE_KEY].values[..., k, rep_idx] = _get_mae_ss_one_scalar(
            target_values=target_matrix[..., k],
            predicted_values=prediction_matrix[..., k],
            mean_training_target_value=mean_training_target_values[k],
            per_grid_cell=per_grid_cell
        )

        (
            t[MSE_KEY].values[..., k, rep_idx],
            t[MSE_BIAS_KEY].values[..., k, rep_idx],
            t[MSE_VARIANCE_KEY].values[..., k, rep_idx]
        ) = _get_mse_one_scalar(
            target_values=target_matrix[..., k],
            predicted_values=prediction_matrix[..., k],
            per_grid_cell=per_grid_cell
        )

        t[MSE_SKILL_SCORE_KEY].values[..., k, rep_idx] = _get_mse_ss_one_scalar(
            target_values=target_matrix[..., k],
            predicted_values=prediction_matrix[..., k],
            mean_training_target_value=mean_training_target_values[k],
            per_grid_cell=per_grid_cell
        )
        t[DWMSE_KEY].values[..., k, rep_idx] = _get_dwmse_one_scalar(
            target_values=target_matrix[..., k],
            predicted_values=prediction_matrix[..., k],
            per_grid_cell=per_grid_cell
        )
        t[DWMSE_SKILL_SCORE_KEY].values[..., k, rep_idx] = (
            _get_dwmse_ss_one_scalar(
                target_values=target_matrix[..., k],
                predicted_values=prediction_matrix[..., k],
                mean_training_target_value=mean_training_target_values[k],
                per_grid_cell=per_grid_cell
            )
        )
        t[BIAS_KEY].values[..., k, rep_idx] = _get_bias_one_scalar(
            target_values=target_matrix[..., k],
            predicted_values=prediction_matrix[..., k],
            per_grid_cell=per_grid_cell
        )
        t[SPATIAL_MIN_BIAS_KEY].values[k, rep_idx] = (
            _get_spatial_min_bias_one_field(
                target_matrix=target_matrix[..., k],
                prediction_matrix=prediction_matrix[..., k]
            )
        )
        t[SPATIAL_MAX_BIAS_KEY].values[k, rep_idx] = (
            _get_spatial_max_bias_one_field(
                target_matrix=target_matrix[..., k],
                prediction_matrix=prediction_matrix[..., k]
            )
        )
        t[CORRELATION_KEY].values[..., k, rep_idx] = (
            _get_correlation_one_scalar(
                target_values=target_matrix[..., k],
                predicted_values=prediction_matrix[..., k],
                per_grid_cell=per_grid_cell
            )
        )
        t[KGE_KEY].values[..., k, rep_idx] = _get_kge_one_scalar(
            target_values=target_matrix[..., k],
            predicted_values=prediction_matrix[..., k],
            per_grid_cell=per_grid_cell
        )

        if keep_it_simple:
            continue

        if num_examples == 0:
            min_bin_edge = 0.
            max_bin_edge = 1.
        elif min_relia_bin_edge_by_target is not None:
            min_bin_edge = min_relia_bin_edge_by_target[k] + 0.
            max_bin_edge = max_relia_bin_edge_by_target[k] + 0.
        else:
            min_bin_edge = numpy.nanpercentile(
                prediction_matrix[..., k],
                min_relia_bin_edge_prctile_by_target[k]
            )
            max_bin_edge = numpy.nanpercentile(
                prediction_matrix[..., k],
                max_relia_bin_edge_prctile_by_target[k]
            )

        num_bins = num_relia_bins_by_target[k]

        if per_grid_cell:
            num_grid_rows = len(t.coords[ROW_DIM].values)
            num_grid_columns = len(t.coords[COLUMN_DIM].values)

            for i in range(num_grid_rows):
                print((
                    'Have computed reliability curve for {0:d} of {1:d} grid '
                    'rows...'
                ).format(
                    i, num_grid_rows
                ))

                for j in range(num_grid_columns):
                    (
                        t[RELIABILITY_X_KEY].values[i, j, k, :num_bins, rep_idx],
                        t[RELIABILITY_Y_KEY].values[i, j, k, :num_bins, rep_idx],
                        these_counts
                    ) = _get_rel_curve_one_scalar(
                        target_values=target_matrix[:, i, j, k],
                        predicted_values=prediction_matrix[:, i, j, k],
                        num_bins=num_bins,
                        min_bin_edge=min_bin_edge,
                        max_bin_edge=max_bin_edge,
                        invert=False
                    )

                    these_squared_diffs = (
                        t[RELIABILITY_X_KEY].values[i, j, k, :num_bins, rep_idx] -
                        t[RELIABILITY_Y_KEY].values[i, j, k, :num_bins, rep_idx]
                    ) ** 2

                    t[RELIABILITY_KEY].values[i, j, k, rep_idx] = (
                        numpy.nansum(these_counts * these_squared_diffs) /
                        numpy.nansum(these_counts)
                    )

                    if rep_idx == 0:
                        (
                            t[RELIABILITY_BIN_CENTER_KEY].values[i, j, k, :num_bins],
                            _,
                            t[RELIABILITY_COUNT_KEY].values[i, j, k, :num_bins]
                        ) = _get_rel_curve_one_scalar(
                            target_values=full_target_matrix[:, i, j, k],
                            predicted_values=full_prediction_matrix[:, i, j, k],
                            num_bins=num_bins,
                            min_bin_edge=min_bin_edge,
                            max_bin_edge=max_bin_edge,
                            invert=False
                        )

                        (
                            t[INV_RELIABILITY_BIN_CENTER_KEY].values[i, j, k, :num_bins],
                            _,
                            t[INV_RELIABILITY_COUNT_KEY].values[i, j, k, :num_bins]
                        ) = _get_rel_curve_one_scalar(
                            target_values=full_target_matrix[:, i, j, k],
                            predicted_values=full_prediction_matrix[:, i, j, k],
                            num_bins=num_bins,
                            min_bin_edge=min_bin_edge,
                            max_bin_edge=max_bin_edge,
                            invert=True
                        )

                    if rep_idx == 0 and full_target_matrix.size > 0:
                        real_indices = numpy.where(numpy.invert(numpy.logical_or(
                            numpy.isnan(full_target_matrix[:, i, j, k]),
                            numpy.isnan(full_prediction_matrix[:, i, j, k])
                        )))[0]

                        if len(real_indices) > 0:
                            (
                                t[KS_STATISTIC_KEY].values[i, j, k],
                                t[KS_P_VALUE_KEY].values[i, j, k]
                            ) = ks_2samp(
                                full_target_matrix[real_indices, i, j, k],
                                full_prediction_matrix[real_indices, i, j, k],
                                alternative='two-sided',
                                mode='auto'
                            )
        else:
            (
                t[RELIABILITY_X_KEY].values[k, :num_bins, rep_idx],
                t[RELIABILITY_Y_KEY].values[k, :num_bins, rep_idx],
                these_counts
            ) = _get_rel_curve_one_scalar(
                target_values=numpy.ravel(target_matrix[..., k]),
                predicted_values=numpy.ravel(prediction_matrix[..., k]),
                num_bins=num_bins,
                min_bin_edge=min_bin_edge,
                max_bin_edge=max_bin_edge,
                invert=False
            )

            these_squared_diffs = (
                t[RELIABILITY_X_KEY].values[k, :num_bins, rep_idx] -
                t[RELIABILITY_Y_KEY].values[k, :num_bins, rep_idx]
            ) ** 2

            t[RELIABILITY_KEY].values[k, rep_idx] = (
                numpy.nansum(these_counts * these_squared_diffs) /
                numpy.nansum(these_counts)
            )

            if rep_idx == 0:
                (
                    t[RELIABILITY_BIN_CENTER_KEY].values[k, :num_bins],
                    _,
                    t[RELIABILITY_COUNT_KEY].values[k, :num_bins]
                ) = _get_rel_curve_one_scalar(
                    target_values=numpy.ravel(full_target_matrix[..., k]),
                    predicted_values=numpy.ravel(full_prediction_matrix[..., k]),
                    num_bins=num_bins,
                    min_bin_edge=min_bin_edge,
                    max_bin_edge=max_bin_edge,
                    invert=False
                )

                (
                    t[INV_RELIABILITY_BIN_CENTER_KEY].values[k, :num_bins],
                    _,
                    t[INV_RELIABILITY_COUNT_KEY].values[k, :num_bins]
                ) = _get_rel_curve_one_scalar(
                    target_values=numpy.ravel(full_target_matrix[..., k]),
                    predicted_values=numpy.ravel(full_prediction_matrix[..., k]),
                    num_bins=num_bins,
                    min_bin_edge=min_bin_edge,
                    max_bin_edge=max_bin_edge,
                    invert=True
                )

            if rep_idx == 0 and full_target_matrix.size > 0:
                real_indices = numpy.where(numpy.invert(numpy.logical_or(
                    numpy.isnan(full_target_matrix[..., k]),
                    numpy.isnan(full_prediction_matrix[..., k])
                )))

                if len(real_indices) > 0:
                    (
                        t[KS_STATISTIC_KEY].values[k],
                        t[KS_P_VALUE_KEY].values[k]
                    ) = ks_2samp(
                        numpy.ravel(full_target_matrix[..., k][real_indices]),
                        numpy.ravel(full_prediction_matrix[..., k][real_indices]),
                        alternative='two-sided',
                        mode='auto'
                    )

    return t


def confidence_interval_to_polygon(
        x_value_matrix, y_value_matrix, confidence_level, same_order):
    """Turns confidence interval into polygon.

    P = number of points
    B = number of bootstrap replicates
    V = number of vertices in resulting polygon = 2 * P + 1

    :param x_value_matrix: P-by-B numpy array of x-values.
    :param y_value_matrix: P-by-B numpy array of y-values.
    :param confidence_level: Confidence level (in range 0...1).
    :param same_order: Boolean flag.  If True (False), minimum x-values will be
        matched with minimum (maximum) y-values.
    :return: polygon_coord_matrix: V-by-2 numpy array of coordinates
        (x-coordinates in first column, y-coords in second).
    """

    error_checking.assert_is_numpy_array(x_value_matrix, num_dimensions=2)
    error_checking.assert_is_numpy_array(y_value_matrix, num_dimensions=2)

    expected_dim = numpy.array([
        x_value_matrix.shape[0], y_value_matrix.shape[1]
    ], dtype=int)

    error_checking.assert_is_numpy_array(
        y_value_matrix, exact_dimensions=expected_dim
    )

    error_checking.assert_is_geq(confidence_level, 0.9)
    error_checking.assert_is_leq(confidence_level, 1.)
    error_checking.assert_is_boolean(same_order)

    min_percentile = 50 * (1. - confidence_level)
    max_percentile = 50 * (1. + confidence_level)

    x_values_bottom = numpy.nanpercentile(
        x_value_matrix, min_percentile, axis=1, interpolation='linear'
    )
    x_values_top = numpy.nanpercentile(
        x_value_matrix, max_percentile, axis=1, interpolation='linear'
    )
    y_values_bottom = numpy.nanpercentile(
        y_value_matrix, min_percentile, axis=1, interpolation='linear'
    )
    y_values_top = numpy.nanpercentile(
        y_value_matrix, max_percentile, axis=1, interpolation='linear'
    )

    real_indices = numpy.where(numpy.invert(numpy.logical_or(
        numpy.isnan(x_values_bottom),
        numpy.isnan(y_values_bottom)
    )))[0]

    if len(real_indices) == 0:
        return None

    x_values_bottom = x_values_bottom[real_indices]
    x_values_top = x_values_top[real_indices]
    y_values_bottom = y_values_bottom[real_indices]
    y_values_top = y_values_top[real_indices]

    x_vertices = numpy.concatenate((
        x_values_top, x_values_bottom[::-1], x_values_top[[0]]
    ))

    if same_order:
        y_vertices = numpy.concatenate((
            y_values_top, y_values_bottom[::-1], y_values_top[[0]]
        ))
    else:
        y_vertices = numpy.concatenate((
            y_values_bottom, y_values_top[::-1], y_values_bottom[[0]]
        ))

    return numpy.transpose(numpy.vstack((
        x_vertices, y_vertices
    )))


def read_inputs(prediction_file_names, target_field_names, take_ensemble_mean):
    """Reads inputs (predictions and targets) from many files.

    P = number of files

    :param prediction_file_names: length-F list of paths to prediction files.
        Each file will be read by `prediction_io.read_file`.
    :param target_field_names: length-T list of field names desired.
    :param take_ensemble_mean: Boolean flag.
    :return: prediction_tables_xarray: length-F list of xarray tables in format
        returned by `prediction_io.read_file`.
    """

    # TODO(thunderhoser): Put this in prediction_io.py.

    error_checking.assert_is_string_list(prediction_file_names)
    error_checking.assert_is_string_list(target_field_names)
    error_checking.assert_is_boolean(take_ensemble_mean)

    num_times = len(prediction_file_names)
    prediction_tables_xarray = [xarray.Dataset()] * num_times
    model_file_name = None

    for i in range(num_times):
        print('Reading data from: "{0:s}"...'.format(prediction_file_names[i]))
        prediction_tables_xarray[i] = prediction_io.read_file(
            prediction_file_names[i]
        )
        if take_ensemble_mean:
            prediction_tables_xarray[i] = prediction_io.take_ensemble_mean(
                prediction_tables_xarray[i]
            )

        pt_i = prediction_tables_xarray[i]

        # TODO(thunderhoser): This is a HACK to allow for prediction
        # files without the model attribute.
        if prediction_io.MODEL_FILE_KEY not in pt_i:
            pt_i.attrs[prediction_io.MODEL_FILE_KEY] = 'foo'

        if model_file_name is None:
            model_file_name = copy.deepcopy(
                pt_i.attrs[prediction_io.MODEL_FILE_KEY]
            )

        assert model_file_name == pt_i.attrs[prediction_io.MODEL_FILE_KEY]

        these_indices = numpy.array([
            numpy.where(pt_i[prediction_io.FIELD_NAME_KEY].values == f)[0][0]
            for f in target_field_names
        ], dtype=int)

        pt_i = pt_i.isel({prediction_io.FIELD_DIM: these_indices})
        prediction_tables_xarray[i] = pt_i

    return prediction_tables_xarray


def _read_spread_skill_inputs(prediction_file_names, target_field_names):
    """Reads inputs for calculation of spread-skill stuff.

    :param prediction_file_names: See doc for `read_inputs`.
    :param target_field_names: Same.
    :return: prediction_tables_xarray: length-F list of xarray tables in format
        returned by `prediction_io.read_file`.
    """

    # TODO(thunderhoser): This is a HACK.

    error_checking.assert_is_string_list(prediction_file_names)
    error_checking.assert_is_string_list(target_field_names)

    num_times = len(prediction_file_names)
    prediction_tables_xarray = [xarray.Dataset()] * num_times
    model_file_name = None

    for i in range(num_times):
        print('Reading data from: "{0:s}"...'.format(prediction_file_names[i]))
        prediction_tables_xarray[i] = prediction_io.read_file(
            prediction_file_names[i]
        )
        prediction_tables_xarray[i] = (
            prediction_io.prep_for_uncertainty_calib_training(
                prediction_tables_xarray[i]
            )
        )

        pt_i = prediction_tables_xarray[i]

        # TODO(thunderhoser): This is a HACK to allow for prediction
        # files without the model attribute.
        if prediction_io.MODEL_FILE_KEY not in pt_i:
            pt_i.attrs[prediction_io.MODEL_FILE_KEY] = 'foo'

        if model_file_name is None:
            model_file_name = copy.deepcopy(
                pt_i.attrs[prediction_io.MODEL_FILE_KEY]
            )

        assert model_file_name == pt_i.attrs[prediction_io.MODEL_FILE_KEY]

        these_indices = numpy.array([
            numpy.where(pt_i[prediction_io.FIELD_NAME_KEY].values == f)[0][0]
            for f in target_field_names
        ], dtype=int)

        pt_i = pt_i.isel({prediction_io.FIELD_DIM: these_indices})
        prediction_tables_xarray[i] = pt_i

    return prediction_tables_xarray


def get_scores_with_bootstrapping(
        prediction_file_names, num_bootstrap_reps,
        target_field_names, target_normalization_file_name,
        num_relia_bins_by_target,
        min_relia_bin_edge_by_target, max_relia_bin_edge_by_target,
        min_relia_bin_edge_prctile_by_target,
        max_relia_bin_edge_prctile_by_target,
        per_grid_cell, keep_it_simple=False,
        compute_ssrat=False, compute_ssrel=False):
    """Computes all scores with bootstrapping.

    T = number of target fields

    :param prediction_file_names: 1-D list of paths to prediction files.  Each
        file will be read by `prediction_io.read_file`.
    :param num_bootstrap_reps: Number of bootstrap replicates.
    :param target_field_names: length-T list of field names.
    :param target_normalization_file_name: Path to file with normalization
        params for target fields.  Mean (climo) values will be read from this
        file.
    :param num_relia_bins_by_target: length-T numpy array with number of bins in
        reliability curve for each target.
    :param min_relia_bin_edge_by_target: length-T numpy array with minimum
        target/predicted value in reliability curve for each target.  If you
        instead want minimum values to be percentiles over the data, make this
        argument None and use `min_relia_bin_edge_prctile_by_target`.
    :param max_relia_bin_edge_by_target: Same as above but for max.
    :param min_relia_bin_edge_prctile_by_target: length-T numpy array with
        percentile level used to determine minimum target/predicted value in
        reliability curve for each target.  If you instead want to specify raw
        values, make this argument None and use `min_relia_bin_edge_by_target`.
    :param max_relia_bin_edge_prctile_by_target: Same as above but for max.
    :param per_grid_cell: Boolean flag.  If True, will compute a separate set of
        scores at each grid cell.  If False, will compute one set of scores for
        the whole domain.
    :param keep_it_simple: Boolean flag.  If True, will avoid Kolmogorov-Smirnov
        test and attributes diagram.
    :param compute_ssrat: Boolean flag.  If True, will compute spread-skill
        ratio (SSRAT) and spread-skill difference (SSDIFF).
    :param compute_ssrel: Boolean flag.  If True, will compute spread-skill
        reliability (SSREL).
    :return: result_table_xarray: xarray table with results (variable and
        dimension names should make the table self-explanatory).
    """

    error_checking.assert_is_string_list(prediction_file_names)
    error_checking.assert_is_integer(num_bootstrap_reps)
    error_checking.assert_is_greater(num_bootstrap_reps, 0)
    error_checking.assert_is_string_list(target_field_names)
    error_checking.assert_is_boolean(per_grid_cell)
    error_checking.assert_is_boolean(keep_it_simple)
    error_checking.assert_is_boolean(compute_ssrat)
    error_checking.assert_is_boolean(compute_ssrel)

    num_target_fields = len(target_field_names)
    expected_dim = numpy.array([num_target_fields], dtype=int)

    error_checking.assert_is_numpy_array(
        num_relia_bins_by_target, exact_dimensions=expected_dim
    )
    error_checking.assert_is_integer_numpy_array(num_relia_bins_by_target)
    error_checking.assert_is_geq_numpy_array(num_relia_bins_by_target, 10)
    error_checking.assert_is_leq_numpy_array(num_relia_bins_by_target, 1000)

    if (
            min_relia_bin_edge_by_target is None or
            max_relia_bin_edge_by_target is None
    ):
        error_checking.assert_is_numpy_array(
            min_relia_bin_edge_prctile_by_target, exact_dimensions=expected_dim
        )
        error_checking.assert_is_geq_numpy_array(
            min_relia_bin_edge_prctile_by_target, 0.
        )
        error_checking.assert_is_leq_numpy_array(
            min_relia_bin_edge_prctile_by_target, 10.
        )

        error_checking.assert_is_numpy_array(
            max_relia_bin_edge_prctile_by_target, exact_dimensions=expected_dim
        )
        error_checking.assert_is_geq_numpy_array(
            max_relia_bin_edge_prctile_by_target, 90.
        )
        error_checking.assert_is_leq_numpy_array(
            max_relia_bin_edge_prctile_by_target, 100.
        )
    else:
        error_checking.assert_is_numpy_array(
            min_relia_bin_edge_by_target, exact_dimensions=expected_dim
        )
        error_checking.assert_is_numpy_array(
            max_relia_bin_edge_by_target, exact_dimensions=expected_dim
        )

        for j in range(num_target_fields):
            error_checking.assert_is_greater(
                max_relia_bin_edge_by_target[j],
                min_relia_bin_edge_by_target[j]
            )

    if compute_ssrat or compute_ssrel:
        prediction_tables_xarray = _read_spread_skill_inputs(
            prediction_file_names=prediction_file_names,
            target_field_names=target_field_names
        )

        prediction_variance_matrix = numpy.stack([
            ptx[prediction_io.PREDICTION_KEY].values[..., 0]
            for ptx in prediction_tables_xarray
        ], axis=0)

        squared_error_matrix = numpy.stack([
            ptx[prediction_io.TARGET_KEY].values
            for ptx in prediction_tables_xarray
        ], axis=0)

        if per_grid_cell:
            num_grid_rows = squared_error_matrix.shape[1]
            num_grid_columns = squared_error_matrix.shape[2]
            these_dim = (
                num_grid_rows, num_grid_columns,
                num_target_fields, num_bootstrap_reps
            )
        else:
            these_dim = (num_target_fields, num_bootstrap_reps)

        if compute_ssrat:
            ssrat_matrix = numpy.full(these_dim, numpy.nan)
            ssdiff_matrix = numpy.full(these_dim, numpy.nan)
        else:
            ssrat_matrix = None
            ssdiff_matrix = None

        if compute_ssrel:
            ssrel_matrix = numpy.full(these_dim, numpy.nan)
        else:
            ssrel_matrix = None

        num_examples = squared_error_matrix.shape[0]
        example_indices = numpy.linspace(
            0, num_examples - 1, num=num_examples, dtype=int
        )

        for i in range(num_bootstrap_reps):
            if num_bootstrap_reps == 1:
                these_indices = example_indices
            else:
                these_indices = numpy.random.choice(
                    example_indices, size=num_examples, replace=True
                )

            if compute_ssrat:
                print((
                    'Computing SSRAT and SSDIFF for {0:d}th of {1:d} bootstrap '
                    'replicates...'
                ).format(
                    i + 1, num_bootstrap_reps
                ))

                ssrat_matrix[..., i] = _get_ssrat_one_replicate(
                    full_squared_error_matrix=squared_error_matrix,
                    full_prediction_variance_matrix=prediction_variance_matrix,
                    example_indices_in_replicate=these_indices,
                    per_grid_cell=per_grid_cell
                )

                ssdiff_matrix[..., i] = _get_ssdiff_one_replicate(
                    full_squared_error_matrix=squared_error_matrix,
                    full_prediction_variance_matrix=prediction_variance_matrix,
                    example_indices_in_replicate=these_indices,
                    per_grid_cell=per_grid_cell
                )

            if compute_ssrel:
                print((
                    'Computing SSREL for {0:d}th of {1:d} bootstrap '
                    'replicates...'
                ).format(
                    i + 1, num_bootstrap_reps
                ))

                ssrel_matrix[..., i] = _get_ssrel_one_replicate(
                    full_squared_error_matrix=squared_error_matrix,
                    full_prediction_variance_matrix=prediction_variance_matrix,
                    example_indices_in_replicate=these_indices,
                    per_grid_cell=per_grid_cell
                )
    else:
        ssrat_matrix = None
        ssdiff_matrix = None
        ssrel_matrix = None

    prediction_tables_xarray = read_inputs(
        prediction_file_names=prediction_file_names,
        target_field_names=target_field_names,
        take_ensemble_mean=True
    )

    prediction_matrix = numpy.stack([
        ptx[prediction_io.PREDICTION_KEY].values[..., 0]
        for ptx in prediction_tables_xarray
    ], axis=0)

    target_matrix = numpy.stack([
        ptx[prediction_io.TARGET_KEY].values
        for ptx in prediction_tables_xarray
    ], axis=0)

    # TODO(thunderhoser): This is a HACK to allow for prediction
    # files without the model attribute.
    if prediction_io.MODEL_FILE_KEY not in prediction_tables_xarray[0]:
        prediction_tables_xarray[0].attrs[prediction_io.MODEL_FILE_KEY] = 'foo'

    model_file_name = (
        prediction_tables_xarray[0].attrs[prediction_io.MODEL_FILE_KEY]
    )

    print('Reading mean (climo) values from: "{0:s}"...'.format(
        target_normalization_file_name
    ))
    target_norm_param_table_xarray = urma_io.read_normalization_file(
        target_normalization_file_name
    )
    tnpt = target_norm_param_table_xarray

    these_indices = numpy.array([
        numpy.where(tnpt.coords[urma_utils.FIELD_DIM].values == f)[0][0]
        for f in target_field_names
    ], dtype=int)

    mean_training_target_values = (
        tnpt[urma_utils.MEAN_VALUE_KEY].values[these_indices]
    )

    # TODO(thunderhoser): Modularize this temperature-conversion shit.
    for j in range(len(target_field_names)):
        if target_field_names[j] not in [
                urma_utils.TEMPERATURE_2METRE_NAME,
                urma_utils.DEWPOINT_2METRE_NAME
        ]:
            continue

        mean_training_target_values[j] = temperature_conv.kelvins_to_celsius(
            numpy.array([mean_training_target_values[j]], dtype=float)
        )[0]

    for j in range(num_target_fields):
        print('Climo-mean {0:s} = {1:.4f}'.format(
            target_field_names[j], mean_training_target_values[j]
        ))

    num_grid_rows = target_matrix.shape[1]
    num_grid_columns = target_matrix.shape[2]

    if per_grid_cell:
        these_dimensions = (
            num_grid_rows, num_grid_columns, num_target_fields,
            num_bootstrap_reps
        )
        these_dim_keys = (
            ROW_DIM, COLUMN_DIM, FIELD_DIM, BOOTSTRAP_REP_DIM
        )
    else:
        these_dimensions = (num_target_fields, num_bootstrap_reps)
        these_dim_keys = (FIELD_DIM, BOOTSTRAP_REP_DIM)

    main_data_dict = {
        TARGET_STDEV_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        PREDICTION_STDEV_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        TARGET_MEAN_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        PREDICTION_MEAN_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        MAE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        MAE_SKILL_SCORE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        MSE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        MSE_BIAS_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        MSE_VARIANCE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        MSE_SKILL_SCORE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        DWMSE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        DWMSE_SKILL_SCORE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        BIAS_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        CORRELATION_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        KGE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        RELIABILITY_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        )
    }

    if compute_ssrat:
        new_dict = {
            SSRAT_KEY: (these_dim_keys, ssrat_matrix),
            SSDIFF_KEY: (these_dim_keys, ssdiff_matrix)
        }
        main_data_dict.update(new_dict)

    if compute_ssrel:
        new_dict = {
            SSREL_KEY: (these_dim_keys, ssrel_matrix)
        }
        main_data_dict.update(new_dict)

    these_dimensions = (num_target_fields, num_bootstrap_reps)
    these_dim_keys = (FIELD_DIM, BOOTSTRAP_REP_DIM)
    new_dict = {
        SPATIAL_MIN_BIAS_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        SPATIAL_MAX_BIAS_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        )
    }
    main_data_dict.update(new_dict)

    if per_grid_cell:
        these_dimensions = (
            num_grid_rows, num_grid_columns, num_target_fields,
            numpy.max(num_relia_bins_by_target), num_bootstrap_reps
        )
        these_dim_keys = (
            ROW_DIM, COLUMN_DIM, FIELD_DIM,
            RELIABILITY_BIN_DIM, BOOTSTRAP_REP_DIM
        )
    else:
        these_dimensions = (
            num_target_fields, numpy.max(num_relia_bins_by_target),
            num_bootstrap_reps
        )
        these_dim_keys = (FIELD_DIM, RELIABILITY_BIN_DIM, BOOTSTRAP_REP_DIM)

    new_dict = {
        RELIABILITY_X_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        RELIABILITY_Y_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        )
    }
    main_data_dict.update(new_dict)

    if per_grid_cell:
        these_dimensions = (
            num_grid_rows, num_grid_columns, num_target_fields,
            numpy.max(num_relia_bins_by_target)
        )
        these_dim_keys = (
            ROW_DIM, COLUMN_DIM, FIELD_DIM, RELIABILITY_BIN_DIM
        )
    else:
        these_dimensions = (
            num_target_fields, numpy.max(num_relia_bins_by_target)
        )
        these_dim_keys = (FIELD_DIM, RELIABILITY_BIN_DIM)

    new_dict = {
        RELIABILITY_BIN_CENTER_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        RELIABILITY_COUNT_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        INV_RELIABILITY_BIN_CENTER_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        INV_RELIABILITY_COUNT_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        )
    }
    main_data_dict.update(new_dict)

    if per_grid_cell:
        these_dimensions = (num_grid_rows, num_grid_columns, num_target_fields)
        these_dim_keys = (ROW_DIM, COLUMN_DIM, FIELD_DIM)
    else:
        these_dimensions = (num_target_fields,)
        these_dim_keys = (FIELD_DIM,)

    new_dict = {
        KS_STATISTIC_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        KS_P_VALUE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        )
    }
    main_data_dict.update(new_dict)

    if per_grid_cell:
        latitude_matrix_deg_n = numpy.stack([
            ptx[prediction_io.LATITUDE_KEY].values
            for ptx in prediction_tables_xarray
        ], axis=0)

        longitude_matrix_deg_e = numpy.stack([
            ptx[prediction_io.LONGITUDE_KEY].values
            for ptx in prediction_tables_xarray
        ], axis=0)

        latitude_diff_matrix_deg = (
            numpy.max(latitude_matrix_deg_n, axis=0) -
            numpy.min(latitude_matrix_deg_n, axis=0)
        )
        longitude_diff_matrix_deg = (
            numpy.max(longitude_matrix_deg_e, axis=0) -
            numpy.min(longitude_matrix_deg_e, axis=0)
        )

        if (
                numpy.all(latitude_diff_matrix_deg < TOLERANCE) and
                numpy.all(longitude_diff_matrix_deg < TOLERANCE)
        ):
            latitude_matrix_deg_n = latitude_matrix_deg_n[0, ...]
            longitude_matrix_deg_e = longitude_matrix_deg_e[0, ...]
        else:
            latitude_matrix_deg_n = numpy.full(
                latitude_matrix_deg_n[0, ...].shape, numpy.nan
            )
            longitude_matrix_deg_e = numpy.full(
                longitude_matrix_deg_e[0, ...].shape, numpy.nan
            )

        these_dim_keys = (ROW_DIM, COLUMN_DIM)

        new_dict = {
            LATITUDE_KEY: (these_dim_keys, latitude_matrix_deg_n),
            LONGITUDE_KEY: (these_dim_keys, longitude_matrix_deg_e)
        }
        main_data_dict.update(new_dict)

    reliability_bin_indices = numpy.linspace(
        0, numpy.max(num_relia_bins_by_target) - 1,
        num=numpy.max(num_relia_bins_by_target), dtype=int
    )
    bootstrap_indices = numpy.linspace(
        0, num_bootstrap_reps - 1, num=num_bootstrap_reps, dtype=int
    )
    metadata_dict = {
        FIELD_DIM: target_field_names,
        RELIABILITY_BIN_DIM: reliability_bin_indices,
        BOOTSTRAP_REP_DIM: bootstrap_indices
    }

    if per_grid_cell:
        metadata_dict.update({
            ROW_DIM: numpy.linspace(
                0, num_grid_rows - 1, num=num_grid_rows, dtype=int
            ),
            COLUMN_DIM: numpy.linspace(
                0, num_grid_columns - 1, num=num_grid_columns, dtype=int
            )
        })

    result_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )
    result_table_xarray.attrs[MODEL_FILE_KEY] = model_file_name
    result_table_xarray.attrs[PREDICTION_FILES_KEY] = ' '.join([
        '{0:s}'.format(f) for f in prediction_file_names
    ])

    num_examples = target_matrix.shape[0]
    example_indices = numpy.linspace(
        0, num_examples - 1, num=num_examples, dtype=int
    )

    for i in range(num_bootstrap_reps):
        if num_bootstrap_reps == 1:
            these_indices = example_indices
        else:
            these_indices = numpy.random.choice(
                example_indices, size=num_examples, replace=True
            )

        print((
            'Computing scores for {0:d}th of {1:d} bootstrap replicates...'
        ).format(
            i + 1, num_bootstrap_reps
        ))

        result_table_xarray = _get_scores_one_replicate(
            result_table_xarray=result_table_xarray,
            full_target_matrix=target_matrix,
            full_prediction_matrix=prediction_matrix,
            replicate_index=i,
            example_indices_in_replicate=these_indices,
            mean_training_target_values=mean_training_target_values,
            num_relia_bins_by_target=num_relia_bins_by_target,
            min_relia_bin_edge_by_target=min_relia_bin_edge_by_target,
            max_relia_bin_edge_by_target=max_relia_bin_edge_by_target,
            min_relia_bin_edge_prctile_by_target=
            min_relia_bin_edge_prctile_by_target,
            max_relia_bin_edge_prctile_by_target=
            max_relia_bin_edge_prctile_by_target,
            per_grid_cell=per_grid_cell,
            keep_it_simple=keep_it_simple
        )

    return result_table_xarray


def write_file(result_table_xarray, netcdf_file_name):
    """Writes evaluation results to NetCDF file.

    :param result_table_xarray: xarray table produced by
        `get_scores_with_bootstrapping`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    # result_table_xarray.to_netcdf(
    #     path=netcdf_file_name, mode='w', format='NETCDF4_CLASSIC'
    # )

    result_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF4_CLASSIC'
    )


def read_file(netcdf_file_name):
    """Reads evaluation results from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: result_table_xarray: xarray table produced by
        `get_scores_with_bootstrapping`.
    """

    result_table_xarray = xarray.open_dataset(netcdf_file_name)
    result_table_xarray.attrs[PREDICTION_FILES_KEY] = (
        result_table_xarray.attrs[PREDICTION_FILES_KEY].split(' ')
    )

    return result_table_xarray
