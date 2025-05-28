"""Helper methods for training and applying neural networks."""

import os
import copy
import pickle
import warnings
import numpy
import keras
import tensorflow
from tensorflow.keras.saving import load_model
from ml_for_national_blend.io import urma_io
from ml_for_national_blend.utils import nbm_utils
from ml_for_national_blend.utils import misc_utils
from ml_for_national_blend.utils import nwp_model_utils
from ml_for_national_blend.utils import urma_utils
from ml_for_national_blend.utils import nbm_constant_utils
from ml_for_national_blend.utils import normalization as non_resid_normalization
from ml_for_national_blend.utils import \
    residual_normalization as resid_normalization
from ml_for_national_blend.outside_code import time_conversion
from ml_for_national_blend.outside_code import time_periods
from ml_for_national_blend.outside_code import \
    temperature_conversions as temperature_conv
from ml_for_national_blend.outside_code import file_system_utils
from ml_for_national_blend.outside_code import error_checking
from ml_for_national_blend.machine_learning import nwp_input
from ml_for_national_blend.machine_learning import \
    neural_net_training_simple as nn_training_simple
from ml_for_national_blend.machine_learning import custom_losses
from ml_for_national_blend.machine_learning import custom_metrics

TOLERANCE = 1e-6
HOURS_TO_SECONDS = 3600
POSSIBLE_DOWNSAMPLING_FACTORS = numpy.array([1, 4, 8, 16], dtype=int)

FIRST_INIT_TIMES_KEY = 'first_init_times_unix_sec'
LAST_INIT_TIMES_KEY = 'last_init_times_unix_sec'
NWP_LEAD_TIMES_KEY = 'nwp_lead_times_hours'
NWP_MODEL_TO_DIR_KEY = 'nwp_model_to_dir_name'
NWP_MODEL_TO_FIELDS_KEY = 'nwp_model_to_field_names'
NWP_NORM_FILE_KEY = 'nwp_normalization_file_name'
NWP_RESID_NORM_FILE_KEY = 'nwp_resid_norm_file_name'
NWP_USE_QUANTILE_NORM_KEY = 'nwp_use_quantile_norm'
BACKUP_NWP_MODEL_KEY = 'backup_nwp_model_name'
BACKUP_NWP_DIR_KEY = 'backup_nwp_directory_name'
TARGET_LEAD_TIME_KEY = 'target_lead_time_hours'
TARGET_FIELDS_KEY = 'target_field_names'
TARGET_LAG_TIMES_KEY = 'target_lag_times_hours'
TARGET_DIR_KEY = 'target_dir_name'
TARGET_NORM_FILE_KEY = 'target_normalization_file_name'
TARGET_RESID_NORM_FILE_KEY = 'target_resid_norm_file_name'
TARGETS_USE_QUANTILE_NORM_KEY = 'targets_use_quantile_norm'
RECENT_BIAS_LAG_TIMES_KEY = 'recent_bias_init_time_lags_hours'
RECENT_BIAS_LEAD_TIMES_KEY = 'recent_bias_lead_times_hours'
NBM_CONSTANT_FIELDS_KEY = 'nbm_constant_field_names'
NBM_CONSTANT_FILE_KEY = 'nbm_constant_file_name'
COMPARE_TO_BASELINE_IN_LOSS_KEY = 'compare_to_baseline_in_loss'
BATCH_SIZE_KEY = 'num_examples_per_batch'
SENTINEL_VALUE_KEY = 'sentinel_value'
PATCH_SIZE_KEY = 'patch_size_2pt5km_pixels'
PATCH_BUFFER_SIZE_KEY = 'patch_buffer_size_2pt5km_pixels'
PATCH_START_ROW_KEY = 'patch_start_row_2pt5km'
PATCH_START_COLUMN_KEY = 'patch_start_column_2pt5km'
REQUIRE_ALL_PREDICTORS_KEY = 'require_all_predictors'

DO_RESIDUAL_PREDICTION_KEY = 'do_residual_prediction'
RESID_BASELINE_MODEL_KEY = 'resid_baseline_model_name'
RESID_BASELINE_LEAD_TIME_KEY = 'resid_baseline_lead_time_hours'
RESID_BASELINE_MODEL_DIR_KEY = 'resid_baseline_model_dir_name'

DEFAULT_GENERATOR_OPTION_DICT = {
    SENTINEL_VALUE_KEY: -10.,
    PATCH_START_ROW_KEY: None,
    PATCH_START_COLUMN_KEY: None
}

NUM_EPOCHS_KEY = 'num_epochs'
EMA_DECAY_KEY = 'use_exp_moving_average_with_decay'
NUM_TRAINING_BATCHES_KEY = 'num_training_batches_per_epoch'
TRAINING_OPTIONS_KEY = 'training_option_dict'
NUM_VALIDATION_BATCHES_KEY = 'num_validation_batches_per_epoch'
VALIDATION_OPTIONS_KEY = 'validation_option_dict'
LOSS_FUNCTION_KEY = 'loss_function_string'
OPTIMIZER_FUNCTION_KEY = 'optimizer_function_string'
METRIC_FUNCTIONS_KEY = 'metric_function_strings'
U_NET_ARCHITECTURE_KEY = 'u_net_architecture_dict'
CHIU_NET_ARCHITECTURE_KEY = 'chiu_net_architecture_dict'
CHIU_NET_PP_ARCHITECTURE_KEY = 'chiu_net_pp_architecture_dict'
CHIU_NEXT_PP_ARCHITECTURE_KEY = 'chiu_next_pp_architecture_dict'
CHIU_NEXT_PPP_ARCHITECTURE_KEY = 'chiu_next_ppp_architecture_dict'
PLATEAU_PATIENCE_KEY = 'plateau_patience_epochs'
PLATEAU_LR_MUTIPLIER_KEY = 'plateau_learning_rate_multiplier'
EARLY_STOPPING_PATIENCE_KEY = 'early_stopping_patience_epochs'
PATCH_OVERLAP_FOR_FAST_GEN_KEY = 'patch_overlap_fast_gen_2pt5km_pixels'
COSINE_ANNEALING_KEY = 'cosine_annealing_dict'
COSINE_ANNEALING_WITH_RESTARTS_KEY = 'cosine_annealing_with_restarts_dict'

METADATA_KEYS = [
    NUM_EPOCHS_KEY, EMA_DECAY_KEY,
    NUM_TRAINING_BATCHES_KEY, TRAINING_OPTIONS_KEY,
    NUM_VALIDATION_BATCHES_KEY, VALIDATION_OPTIONS_KEY, LOSS_FUNCTION_KEY,
    OPTIMIZER_FUNCTION_KEY, METRIC_FUNCTIONS_KEY,
    U_NET_ARCHITECTURE_KEY, CHIU_NET_ARCHITECTURE_KEY,
    CHIU_NET_PP_ARCHITECTURE_KEY, CHIU_NEXT_PP_ARCHITECTURE_KEY,
    CHIU_NEXT_PPP_ARCHITECTURE_KEY,
    PLATEAU_PATIENCE_KEY, PLATEAU_LR_MUTIPLIER_KEY,
    EARLY_STOPPING_PATIENCE_KEY, PATCH_OVERLAP_FOR_FAST_GEN_KEY,
    COSINE_ANNEALING_KEY, COSINE_ANNEALING_WITH_RESTARTS_KEY
]

PREDICTOR_MATRIX_2PT5KM_KEY = 'predictor_matrix_2pt5km'
PREDICTOR_MATRIX_10KM_KEY = 'predictor_matrix_10km'
PREDICTOR_MATRIX_20KM_KEY = 'predictor_matrix_20km'
PREDICTOR_MATRIX_40KM_KEY = 'predictor_matrix_40km'
RECENT_BIAS_MATRIX_2PT5KM_KEY = 'recent_bias_matrix_2pt5km'
RECENT_BIAS_MATRIX_10KM_KEY = 'recent_bias_matrix_10km'
RECENT_BIAS_MATRIX_20KM_KEY = 'recent_bias_matrix_20km'
RECENT_BIAS_MATRIX_40KM_KEY = 'recent_bias_matrix_40km'
PREDICTOR_MATRIX_BASELINE_KEY = 'predictor_matrix_resid_baseline'
PREDICTOR_MATRIX_LAGTGT_KEY = 'predictor_matrix_lagged_targets'
TARGET_MATRIX_KEY = 'target_matrix'


class CosineAnnealingScheduler(keras.callbacks.Callback):
    """Implements cosine-annealing schedule for learning rate."""

    def __init__(self, num_epochs_in_cycle, min_learning_rate,
                 max_learning_rate):
        """Initializer.

        :param num_epochs_in_cycle: Length of annealing cycle.
        :param max_learning_rate: Max learning rate.
        :param min_learning_rate: Minimum learning rate.
        """

        error_checking.assert_is_integer(num_epochs_in_cycle)
        error_checking.assert_is_geq(num_epochs_in_cycle, 10)
        error_checking.assert_is_greater(max_learning_rate, min_learning_rate)
        error_checking.assert_is_greater(min_learning_rate, 0.)

        super().__init__()
        self.num_epochs_in_cycle = num_epochs_in_cycle
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate

    def on_epoch_begin(self, epoch, logs=None):
        """This method triggers at the beginning of each epoch.

        :param epoch: Epoch number.
        :param logs: I don't know why this argument is here.  Just leave it
            alone.
        """

        multiplier = 0.5 * (
            1 + numpy.cos(numpy.pi * epoch / self.num_epochs_in_cycle)
        )
        current_learning_rate = (
            self.min_learning_rate +
            multiplier * (self.max_learning_rate - self.min_learning_rate)
        )

        self.model.optimizer.learning_rate.assign(current_learning_rate)
        print((
            'Epoch {0:d}: CosineAnnealingScheduler setting learning rate to '
            '{1:.4g}!'
        ).format(
            epoch, current_learning_rate
        ))


class CosineAnnealingWarmRestarts(keras.callbacks.Callback):
    """Implements cosine-annealing schedule with warm restarts."""

    def __init__(self, min_learning_rate, max_learning_rate,
                 num_epochs_in_first_cycle, cycle_length_multiplier):
        """Initializer.

        :param max_learning_rate: Max learning rate.
        :param min_learning_rate: Minimum learning rate.
        :param num_epochs_in_first_cycle: Length of first annealing cycle.
        :param cycle_length_multiplier: Multiplier for length of subsequent
            annealing cycles.
        """

        error_checking.assert_is_greater(max_learning_rate, min_learning_rate)
        error_checking.assert_is_greater(min_learning_rate, 0.)
        error_checking.assert_is_integer(num_epochs_in_first_cycle)
        error_checking.assert_is_geq(num_epochs_in_first_cycle, 10)
        error_checking.assert_is_integer(cycle_length_multiplier)
        error_checking.assert_is_greater(cycle_length_multiplier, 0)

        super().__init__()
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.num_epochs_in_first_cycle = num_epochs_in_first_cycle
        self.cycle_length_multiplier = cycle_length_multiplier

    def on_epoch_begin(self, epoch, logs=None):
        """This method triggers at the beginning of each epoch.

        :param epoch: Epoch number.
        :param logs: I don't know why this argument is here.  Just leave it
            alone.
        """

        num_epochs_in_this_cycle = self.num_epochs_in_first_cycle + 0
        epochs_into_this_cycle = epoch + 0
        cycle_index = 0

        while epochs_into_this_cycle >= num_epochs_in_this_cycle:
            epochs_into_this_cycle -= num_epochs_in_this_cycle
            num_epochs_in_this_cycle *= self.cycle_length_multiplier
            cycle_index += 1

        multiplier = 0.5 * (1 + numpy.cos(
            numpy.pi * epochs_into_this_cycle / num_epochs_in_this_cycle
        ))
        current_learning_rate = (
            self.min_learning_rate +
            multiplier * (self.max_learning_rate - self.min_learning_rate)
        )

        self.model.optimizer.learning_rate.assign(current_learning_rate)
        print((
            'Epoch {0:d}: CosineAnnealingWarmRestarts setting learning rate to '
            '{1:.4g}!'
        ).format(
            epoch, current_learning_rate
        ))


class EMAHelper:
    def __init__(self, model, decay=0.99):
        self.model = model
        self.decay = decay
        self.shadow_weights = [
            tensorflow.Variable(w, trainable=False) for w in model.weights
        ]
        self.original_weights = [
            tensorflow.Variable(w, trainable=False) for w in model.weights
        ]

    def apply_ema(self):  # Updates only the shadow weights.
        for sw, w in zip(self.shadow_weights, self.model.weights):
            sw.assign(self.decay * sw + (1 - self.decay) * w)

    def set_ema_weights(self):
        for orig_w, w, sw in zip(
                self.original_weights, self.model.weights, self.shadow_weights
        ):
            orig_w.assign(w)  # Save the current (non-EMA) weights
            w.assign(sw)      # Set the model weights to EMA weights

    def restore_original_weights(self):
        for orig_w, w in zip(self.original_weights, self.model.weights):
            w.assign(orig_w)

    def save_optimizer_state(self, checkpoint_dir, epoch):
        checkpoint_object = tensorflow.train.Checkpoint(
            model=self.model,
            ema_shadow_weights={
                str(i): sw for i, sw in enumerate(self.shadow_weights)
            }
        )
        output_path = '{0:s}/checkpoint_epoch_{1:d}'.format(
            checkpoint_dir, epoch
        )

        print('Saving model and optimizer state to: "{0:s}"...'.format(
            output_path
        ))
        checkpoint_object.save(output_path)

    def restore_optimizer_state(self, checkpoint_dir, raise_error_if_missing):
        checkpoint_object = tensorflow.train.Checkpoint(
            model=self.model,
            ema_shadow_weights={
                str(i): sw for i, sw in enumerate(self.shadow_weights)
            }
        )

        print('Restoring optimizer state from: "{0:s}"...'.format(
            checkpoint_dir
        ))

        try:
            status = checkpoint_object.restore(
                tensorflow.train.latest_checkpoint(checkpoint_dir)
            )
        except:
            if raise_error_if_missing:
                raise

            warning_string = (
                'POTENTIAL ERROR: Cannot find EMA checkpoint at "{0:s}"'
            ).format(checkpoint_dir)

            warnings.warn(warning_string)
            return

        # if raise_error_if_missing:
        #     status.assert_consumed()
        status.expect_partial()

        found_any_diff = False

        for i, sw in enumerate(self.shadow_weights):
            if not found_any_diff:
                found_any_diff = not numpy.allclose(
                    sw,
                    checkpoint_object.ema_shadow_weights[str(i)],
                    atol=TOLERANCE
                )

            sw.assign(checkpoint_object.ema_shadow_weights[str(i)])

        if raise_error_if_missing:
            assert found_any_diff


def create_data_dict_or_tuple(
        predictor_matrix_2pt5km, nbm_constant_matrix,
        predictor_matrix_lagged_targets, predictor_matrix_10km,
        predictor_matrix_20km, predictor_matrix_40km,
        predictor_matrix_resid_baseline,
        recent_bias_matrix_2pt5km, recent_bias_matrix_10km,
        recent_bias_matrix_20km, recent_bias_matrix_40km,
        target_matrix, sentinel_value, return_predictors_as_dict=False,
        allow_nan=False):
    """Finalizes data-processing by creating dictionary or tuple.

    The dictionary or tuple can be directly fed into a neural network.

    :param predictor_matrix_2pt5km: See output doc for
        `neural_net_training_simple.data_generator`.
    :param nbm_constant_matrix: Same.
    :param predictor_matrix_lagged_targets: Same.
    :param predictor_matrix_10km: Same.
    :param predictor_matrix_20km: Same.
    :param predictor_matrix_40km: Same.
    :param predictor_matrix_resid_baseline: Same.
    :param recent_bias_matrix_2pt5km: Same.
    :param recent_bias_matrix_10km: Same.
    :param recent_bias_matrix_20km: Same.
    :param recent_bias_matrix_40km: Same.
    :param target_matrix: Same.
    :param sentinel_value: See input doc for
        `neural_net_training_simple.data_generator`.
    :param return_predictors_as_dict: Boolean flag.  If True (False), will
        return predictor matrices as dictionary (tuple).
    :param allow_nan: Leave this alone for now.
    :return: predictor_matrices: Dictionary or tuple with 32-bit predictor
        matrices.
    """

    error_checking.assert_is_boolean(return_predictors_as_dict)
    if not allow_nan:
        error_checking.assert_is_numpy_array_without_nan(target_matrix)

    print((
        'Shape of target matrix = {0:s} ... NaN fraction = {1:.4f}'
    ).format(
        str(target_matrix.shape),
        numpy.mean(numpy.isnan(target_matrix))
    ))

    these_min = numpy.nanmin(
        target_matrix, axis=(0, 1, 2)
    )
    these_max = numpy.nanmax(
        target_matrix, axis=(0, 1, 2)
    )

    print('Min values in target matrix: {0:s}'.format(
        str(these_min)
    ))
    print('Max values in target matrix: {0:s}'.format(
        str(these_max)
    ))

    # TODO(thunderhoser): This HACK removes the lead-time axis if there is only
    # one lead time.  Needed to make my simple U-net architecture work.
    if predictor_matrix_2pt5km is not None and predictor_matrix_2pt5km.shape[-2] == 1:
        predictor_matrix_2pt5km = predictor_matrix_2pt5km[..., 0, :]

    predictor_matrices = (
        predictor_matrix_2pt5km, nbm_constant_matrix,
        predictor_matrix_lagged_targets, predictor_matrix_10km,
        predictor_matrix_20km, predictor_matrix_40km,
        recent_bias_matrix_2pt5km, recent_bias_matrix_10km,
        recent_bias_matrix_20km, recent_bias_matrix_40km,
        predictor_matrix_resid_baseline
    )
    pred_matrix_descriptions = [
        '2.5-km predictor matrix', 'NBM-constant predictor matrix',
        'lagged-target predictor matrix', '10-km predictor matrix',
        '20-km predictor matrix', '40-km predictor matrix',
        '2.5-km recent-bias matrix', '10-km recent-bias matrix',
        '20-km recent-bias matrix', '40-km recent-bias matrix',
        'residual baseline matrix'
    ]
    predictor_keys = [
        '2pt5km_inputs', 'const_inputs',
        'lagtgt_inputs', '10km_inputs',
        '20km_inputs', '40km_inputs',
        '2pt5km_rctbias', '10km_rctbias',
        '20km_rctbias', '40km_rctbias',
        'resid_baseline_inputs'
    ]
    allow_nan_flags = [
        True, False,
        True, True,
        True, True,
        True, True,
        True, True,
        allow_nan
    ]
    replace_nan_flags = [
        True, False,
        True, True,
        True, True,
        True, True,
        True, True,
        False
    ]

    for k in range(len(predictor_matrices)):
        if predictor_matrices[k] is None:
            continue

        print((
            'Shape of {0:s}: {1:s} ... NaN fraction = {2:.4f} ... '
            'min/max = {3:.4f}/{4:.4f}'
        ).format(
            pred_matrix_descriptions[k],
            str(predictor_matrices[k].shape),
            numpy.mean(numpy.isnan(predictor_matrices[k])),
            numpy.nanmin(predictor_matrices[k]),
            numpy.nanmax(predictor_matrices[k])
        ))

        if replace_nan_flags[k]:
            predictor_matrices[k][numpy.isnan(predictor_matrices[k])] = (
                sentinel_value
            )

        if not allow_nan_flags[k]:
            error_checking.assert_is_numpy_array_without_nan(
                predictor_matrices[k]
            )

    if predictor_matrix_resid_baseline is not None:
        these_min = numpy.nanmin(
            predictor_matrix_resid_baseline, axis=(0, 1, 2)
        )
        these_max = numpy.nanmax(
            predictor_matrix_resid_baseline, axis=(0, 1, 2)
        )

        print('Min values in residual baseline matrix: {0:s}'.format(
            str(these_min)
        ))
        print('Max values in residual baseline matrix: {0:s}'.format(
            str(these_max)
        ))

    if not return_predictors_as_dict:
        return tuple(
            pm.astype('float32') for pm in predictor_matrices if pm is not None
        )

    predictor_dict = dict(zip(predictor_keys, predictor_matrices))
    return {
        key: value.astype('float32')
        for key, value in predictor_dict.items()
        if value is not None
    }


def predicted_depression_to_dewpoint(prediction_matrix, target_field_names):
    """Converts predicted 2-metre dewpoint depression to dewpoint temperature.

    :param prediction_matrix: See documentation for
        `neural_net_training_simple.apply_model`.
    :param target_field_names: Same.
    :return: prediction_matrix: Same as input, except that dewpoint depressions
        have been replaced with dewpoint temperatures.
    """

    error_checking.assert_is_numpy_array_without_nan(prediction_matrix)
    num_target_fields = prediction_matrix.shape[-2]
    target_field_names = numpy.array(target_field_names)

    error_checking.assert_is_numpy_array(
        target_field_names,
        exact_dimensions=numpy.array([num_target_fields], dtype=int)
    )

    dewp_index = numpy.where(
        target_field_names == urma_utils.DEWPOINT_2METRE_NAME
    )[0][0]
    temp_index = numpy.where(
        target_field_names == urma_utils.TEMPERATURE_2METRE_NAME
    )[0][0]
    prediction_matrix[..., dewp_index, :] = (
        prediction_matrix[..., temp_index, :] -
        prediction_matrix[..., dewp_index, :]
    )

    return prediction_matrix


def predicted_gust_excess_to_speed(prediction_matrix, target_field_names):
    """Converts predicted 10-metre gust excess to gust speed.

    :param prediction_matrix: See documentation for
        `neural_net_training_simple.apply_model`.
    :param target_field_names: Same.
    :return: prediction_matrix: Same as input, except that gust excesses have
        been replaced with gust speeds.
    """

    error_checking.assert_is_numpy_array_without_nan(prediction_matrix)
    num_target_fields = prediction_matrix.shape[-2]
    target_field_names = numpy.array(target_field_names)

    error_checking.assert_is_numpy_array(
        target_field_names,
        exact_dimensions=numpy.array([num_target_fields], dtype=int)
    )

    u_index = numpy.where(
        target_field_names == urma_utils.U_WIND_10METRE_NAME
    )[0][0]
    v_index = numpy.where(
        target_field_names == urma_utils.V_WIND_10METRE_NAME
    )[0][0]
    gust_index = numpy.where(
        target_field_names == urma_utils.WIND_GUST_10METRE_NAME
    )[0][0]

    gust_excess_matrix = prediction_matrix[..., gust_index, :]
    sustained_speed_matrix = numpy.sqrt(
        prediction_matrix[..., u_index, :] ** 2 +
        prediction_matrix[..., v_index, :] ** 2
    )
    prediction_matrix[..., gust_index, :] = (
        gust_excess_matrix + sustained_speed_matrix
    )

    return prediction_matrix


def increment_init_time(current_index, init_times_unix_sec):
    """Increments initialization time for generator.

    This allows the generator to read the next init time.

    :param current_index: Current index.  If current_index == k, this means the
        last init time read is init_times_unix_sec[k].
    :param init_times_unix_sec: 1-D numpy array of init times.
    :return: current_index: Updated version of input.
    :return: init_times_unix_sec: Possibly shuffled version of input.
    """

    error_checking.assert_is_integer_numpy_array(init_times_unix_sec)
    error_checking.assert_is_numpy_array(init_times_unix_sec, num_dimensions=1)
    error_checking.assert_is_integer(current_index)
    error_checking.assert_is_geq(current_index, 0)
    error_checking.assert_is_less_than(current_index, len(init_times_unix_sec))

    if current_index == len(init_times_unix_sec) - 1:
        numpy.random.shuffle(init_times_unix_sec)
        current_index = 0
    else:
        current_index += 1

    return current_index, init_times_unix_sec


def patch_buffer_to_mask(patch_size_2pt5km_pixels,
                         patch_buffer_size_2pt5km_pixels):
    """Converts patch size and patch buffer size to mask.

    M = number of rows in outer patch (used for predictors)
    m = number of rows in inner patch (used for loss function)

    :param patch_size_2pt5km_pixels: M in the above discussion.
    :param patch_buffer_size_2pt5km_pixels: m in the above discussion.
    :return: mask_matrix: M-by-M numpy array of integers, where 1 (0) means that
        the pixel will (not) be considered in the loss function.
    """

    error_checking.assert_is_integer(patch_size_2pt5km_pixels)
    error_checking.assert_is_greater(patch_size_2pt5km_pixels, 0)
    error_checking.assert_is_integer(patch_buffer_size_2pt5km_pixels)
    error_checking.assert_is_geq(patch_buffer_size_2pt5km_pixels, 0)
    error_checking.assert_is_less_than(
        patch_buffer_size_2pt5km_pixels,
        patch_size_2pt5km_pixels // 2
    )

    mask_matrix = numpy.full(
        (patch_size_2pt5km_pixels, patch_size_2pt5km_pixels), 1, dtype=int
    )
    mask_matrix[:patch_buffer_size_2pt5km_pixels, :] = 0
    mask_matrix[-patch_buffer_size_2pt5km_pixels:, :] = 0
    mask_matrix[:, :patch_buffer_size_2pt5km_pixels] = 0
    mask_matrix[:, -patch_buffer_size_2pt5km_pixels:] = 0

    return mask_matrix


def set_model_weights_to_ema(model_object, metafile_name):
    """Sets model weights to exponential moving average.

    :param model_object: Trained instance of `keras.models.Model` or
        `keras.models.Sequential`.
    :param metafile_name: Path to metafile.
    """

    metadata_dict = read_metafile(metafile_name)
    ema_object = EMAHelper(
        model=model_object, decay=metadata_dict[EMA_DECAY_KEY]
    )

    ema_backup_dir_name = '{0:s}/exponential_moving_average'.format(
        os.path.split(metafile_name)[0]
    )
    ema_object.restore_optimizer_state(
        checkpoint_dir=ema_backup_dir_name, raise_error_if_missing=True
    )

    for layer_object in model_object.layers:
        if 'conv' not in layer_object.name.lower():
            continue

        weight_matrix = numpy.array(layer_object.get_weights()[0])
        print('Weights for {0:s} before EMA:\n{1:s}'.format(
            layer_object.name, str(weight_matrix)
        ))
        break

    ema_object.set_ema_weights()

    for layer_object in model_object.layers:
        if 'conv' not in layer_object.name.lower():
            continue

        weight_matrix = numpy.array(layer_object.get_weights()[0])
        print('Weights for {0:s} after EMA:\n{1:s}'.format(
            layer_object.name, str(weight_matrix)
        ))
        break


def check_generator_args(option_dict):
    """Checks input arguments for generator.

    :param option_dict: See doc for `neural_net_training_simple.data_generator`.
    :return: option_dict: Same as input, except defaults may have been added.
    """

    orig_option_dict = option_dict.copy()
    option_dict = DEFAULT_GENERATOR_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    error_checking.assert_is_numpy_array(
        option_dict[FIRST_INIT_TIMES_KEY], num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(
        option_dict[FIRST_INIT_TIMES_KEY]
    )

    expected_dim = numpy.array(
        [len(option_dict[FIRST_INIT_TIMES_KEY])], dtype=int
    )
    error_checking.assert_is_numpy_array(
        option_dict[LAST_INIT_TIMES_KEY], exact_dimensions=expected_dim
    )
    error_checking.assert_is_integer_numpy_array(
        option_dict[LAST_INIT_TIMES_KEY]
    )

    error_checking.assert_is_geq_numpy_array(
        option_dict[LAST_INIT_TIMES_KEY] - option_dict[FIRST_INIT_TIMES_KEY],
        0
    )

    error_checking.assert_is_numpy_array(
        option_dict[NWP_LEAD_TIMES_KEY], num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(
        option_dict[NWP_LEAD_TIMES_KEY]
    )
    error_checking.assert_is_greater_numpy_array(
        option_dict[NWP_LEAD_TIMES_KEY], 0
    )
    option_dict[NWP_LEAD_TIMES_KEY] = numpy.unique(
        option_dict[NWP_LEAD_TIMES_KEY]
    )

    nwp_model_to_dir_name = option_dict[NWP_MODEL_TO_DIR_KEY]
    nwp_model_to_field_names = option_dict[NWP_MODEL_TO_FIELDS_KEY]

    first_nwp_model_names = list(nwp_model_to_dir_name.keys())
    second_nwp_model_names = list(nwp_model_to_field_names.keys())
    assert set(first_nwp_model_names) == set(second_nwp_model_names)

    nwp_model_names = second_nwp_model_names
    nwp_model_names.sort()

    for this_model_name in nwp_model_names:
        error_checking.assert_is_string(nwp_model_to_dir_name[this_model_name])
        error_checking.assert_is_string_list(
            nwp_model_to_field_names[this_model_name]
        )

        for this_field_name in nwp_model_to_field_names[this_model_name]:
            nwp_model_utils.check_field_name(this_field_name)

    error_checking.assert_is_boolean(option_dict[NWP_USE_QUANTILE_NORM_KEY])
    if option_dict[NWP_NORM_FILE_KEY] is not None:
        error_checking.assert_file_exists(option_dict[NWP_NORM_FILE_KEY])
    if option_dict[NWP_RESID_NORM_FILE_KEY] is not None:
        error_checking.assert_file_exists(option_dict[NWP_RESID_NORM_FILE_KEY])

    error_checking.assert_is_string(option_dict[BACKUP_NWP_MODEL_KEY])
    error_checking.assert_is_string(option_dict[BACKUP_NWP_DIR_KEY])

    error_checking.assert_is_integer(option_dict[TARGET_LEAD_TIME_KEY])
    error_checking.assert_is_greater(option_dict[TARGET_LEAD_TIME_KEY], 0)
    error_checking.assert_is_string_list(option_dict[TARGET_FIELDS_KEY])
    for this_field_name in option_dict[TARGET_FIELDS_KEY]:
        urma_utils.check_field_name(this_field_name)

    if option_dict[TARGET_LAG_TIMES_KEY] is not None:
        error_checking.assert_is_numpy_array(
            option_dict[TARGET_LAG_TIMES_KEY], num_dimensions=1
        )
        error_checking.assert_is_integer_numpy_array(
            option_dict[TARGET_LAG_TIMES_KEY]
        )
        error_checking.assert_is_geq_numpy_array(
            option_dict[TARGET_LAG_TIMES_KEY], 0
        )
        option_dict[TARGET_LAG_TIMES_KEY] = numpy.unique(
            option_dict[TARGET_LAG_TIMES_KEY]
        )[::-1]

    error_checking.assert_is_string(option_dict[TARGET_DIR_KEY])
    error_checking.assert_is_boolean(option_dict[TARGETS_USE_QUANTILE_NORM_KEY])
    if option_dict[TARGET_NORM_FILE_KEY] is not None:
        error_checking.assert_file_exists(option_dict[TARGET_NORM_FILE_KEY])
    if option_dict[TARGET_RESID_NORM_FILE_KEY] is not None:
        error_checking.assert_file_exists(
            option_dict[TARGET_RESID_NORM_FILE_KEY]
        )

    use_recent_biases = not (
        option_dict[RECENT_BIAS_LAG_TIMES_KEY] is None
        or option_dict[RECENT_BIAS_LEAD_TIMES_KEY] is None
    )

    if use_recent_biases:
        error_checking.assert_is_numpy_array(
            option_dict[RECENT_BIAS_LAG_TIMES_KEY], num_dimensions=1
        )
        error_checking.assert_is_integer_numpy_array(
            option_dict[RECENT_BIAS_LAG_TIMES_KEY]
        )
        error_checking.assert_is_greater_numpy_array(
            option_dict[RECENT_BIAS_LAG_TIMES_KEY], 0
        )

        num_recent_bias_times = len(option_dict[RECENT_BIAS_LAG_TIMES_KEY])
        expected_dim = numpy.array([num_recent_bias_times], dtype=int)

        error_checking.assert_is_numpy_array(
            option_dict[RECENT_BIAS_LEAD_TIMES_KEY],
            exact_dimensions=expected_dim
        )
        error_checking.assert_is_integer_numpy_array(
            option_dict[RECENT_BIAS_LEAD_TIMES_KEY]
        )
        error_checking.assert_is_greater_numpy_array(
            option_dict[RECENT_BIAS_LEAD_TIMES_KEY], 0
        )

        lookahead_times_hours = (
            option_dict[RECENT_BIAS_LEAD_TIMES_KEY] -
            option_dict[RECENT_BIAS_LAG_TIMES_KEY]
        )
        error_checking.assert_is_leq_numpy_array(lookahead_times_hours, 0)

    if (
            option_dict[NBM_CONSTANT_FILE_KEY] is None
            or len(option_dict[NBM_CONSTANT_FIELDS_KEY]) == 0
    ):
        option_dict[NBM_CONSTANT_FILE_KEY] = None
        option_dict[NBM_CONSTANT_FIELDS_KEY] = []

    if option_dict[NBM_CONSTANT_FILE_KEY] is not None:
        error_checking.assert_file_exists(option_dict[NBM_CONSTANT_FILE_KEY])
    for this_field_name in option_dict[NBM_CONSTANT_FIELDS_KEY]:
        nbm_constant_utils.check_field_name(this_field_name)

    error_checking.assert_is_integer(option_dict[BATCH_SIZE_KEY])
    error_checking.assert_is_geq(option_dict[BATCH_SIZE_KEY], 1)
    error_checking.assert_is_not_nan(option_dict[SENTINEL_VALUE_KEY])

    if option_dict[PATCH_SIZE_KEY] is None:
        option_dict[PATCH_BUFFER_SIZE_KEY] = None
        option_dict[PATCH_START_ROW_KEY] = None
        option_dict[PATCH_START_COLUMN_KEY] = None

    if option_dict[PATCH_SIZE_KEY] is not None:
        error_checking.assert_is_integer(option_dict[PATCH_SIZE_KEY])
        error_checking.assert_is_greater(option_dict[PATCH_SIZE_KEY], 0)

        error_checking.assert_is_integer(option_dict[PATCH_BUFFER_SIZE_KEY])
        error_checking.assert_is_geq(option_dict[PATCH_BUFFER_SIZE_KEY], 0)
        error_checking.assert_is_less_than(
            option_dict[PATCH_BUFFER_SIZE_KEY],
            option_dict[PATCH_SIZE_KEY] // 2
        )

    if option_dict[PATCH_START_ROW_KEY] is not None:
        assert option_dict[PATCH_START_COLUMN_KEY] is not None
        error_checking.assert_is_integer(option_dict[PATCH_START_ROW_KEY])
        error_checking.assert_is_geq(option_dict[PATCH_START_ROW_KEY], 0)

    if option_dict[PATCH_START_COLUMN_KEY] is not None:
        assert option_dict[PATCH_START_ROW_KEY] is not None
        error_checking.assert_is_integer(option_dict[PATCH_START_COLUMN_KEY])
        error_checking.assert_is_geq(option_dict[PATCH_START_COLUMN_KEY], 0)

    predict_dewpoint_depression = (
        urma_utils.DEWPOINT_2METRE_NAME in option_dict[TARGET_FIELDS_KEY]
    )
    if predict_dewpoint_depression:
        assert (
            urma_utils.TEMPERATURE_2METRE_NAME in option_dict[TARGET_FIELDS_KEY]
        )
        option_dict[TARGET_FIELDS_KEY].remove(urma_utils.DEWPOINT_2METRE_NAME)
        option_dict[TARGET_FIELDS_KEY].append(urma_utils.DEWPOINT_2METRE_NAME)

    predict_gust_excess = (
        urma_utils.WIND_GUST_10METRE_NAME in option_dict[TARGET_FIELDS_KEY]
    )
    if predict_gust_excess:
        assert (
            urma_utils.U_WIND_10METRE_NAME in option_dict[TARGET_FIELDS_KEY]
        )
        assert (
            urma_utils.V_WIND_10METRE_NAME in option_dict[TARGET_FIELDS_KEY]
        )
        option_dict[TARGET_FIELDS_KEY].remove(urma_utils.WIND_GUST_10METRE_NAME)
        option_dict[TARGET_FIELDS_KEY].append(urma_utils.WIND_GUST_10METRE_NAME)

    error_checking.assert_is_boolean(option_dict[DO_RESIDUAL_PREDICTION_KEY])
    error_checking.assert_is_boolean(
        option_dict[COMPARE_TO_BASELINE_IN_LOSS_KEY]
    )

    if not (
            option_dict[DO_RESIDUAL_PREDICTION_KEY] or
            option_dict[COMPARE_TO_BASELINE_IN_LOSS_KEY]
    ):
        option_dict[RESID_BASELINE_MODEL_KEY] = None
        option_dict[RESID_BASELINE_MODEL_DIR_KEY] = None
        option_dict[RESID_BASELINE_LEAD_TIME_KEY] = -1
        return option_dict

    error_checking.assert_is_string(option_dict[RESID_BASELINE_MODEL_KEY])
    error_checking.assert_is_string(option_dict[RESID_BASELINE_MODEL_DIR_KEY])
    error_checking.assert_is_integer(option_dict[RESID_BASELINE_LEAD_TIME_KEY])
    error_checking.assert_is_greater(
        option_dict[RESID_BASELINE_LEAD_TIME_KEY], 0
    )

    resid_baseline_ds_factor = nwp_model_utils.model_to_nbm_downsampling_factor(
        option_dict[RESID_BASELINE_MODEL_KEY]
    )
    error_checking.assert_equals(resid_baseline_ds_factor, 1)

    return option_dict


def check_u_net_generator_args(option_dict):
    """Checks input arguments for generator.

    :param option_dict: See doc for
        `neural_net_training_multipatch.data_generator_for_u_net`.
    :return: option_dict: Same as input, except defaults may have been added.
    """

    orig_option_dict = option_dict.copy()
    option_dict = DEFAULT_GENERATOR_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    first_init_time_unix_sec = option_dict['first_init_time_unix_sec']
    last_init_time_unix_sec = option_dict['last_init_time_unix_sec']
    error_checking.assert_is_integer(first_init_time_unix_sec)
    error_checking.assert_is_integer(last_init_time_unix_sec)
    error_checking.assert_is_greater(
        last_init_time_unix_sec, first_init_time_unix_sec
    )

    nwp_lead_time_hours = option_dict['nwp_lead_time_hours']
    error_checking.assert_is_integer(nwp_lead_time_hours)
    error_checking.assert_is_greater(nwp_lead_time_hours, 0)

    nwp_model_to_dir_name = option_dict[NWP_MODEL_TO_DIR_KEY]
    nwp_model_to_field_names = option_dict[NWP_MODEL_TO_FIELDS_KEY]

    first_nwp_model_names = list(nwp_model_to_dir_name.keys())
    second_nwp_model_names = list(nwp_model_to_field_names.keys())
    assert set(first_nwp_model_names) == set(second_nwp_model_names)

    nwp_model_names = second_nwp_model_names
    nwp_model_names.sort()

    for this_model_name in nwp_model_names:
        nwp_model_utils.check_model_name(
            model_name=this_model_name, allow_ensemble=True
        )

        # All NWP data going into the simple U-net must have 2.5-km resolution.
        this_ds_factor = nwp_model_utils.model_to_nbm_downsampling_factor(
            this_model_name
        )
        error_checking.assert_equals(this_ds_factor, 1)

        error_checking.assert_directory_exists(
            nwp_model_to_dir_name[this_model_name]
        )

        error_checking.assert_is_string_list(
            nwp_model_to_field_names[this_model_name]
        )
        for this_field_name in nwp_model_to_field_names[this_model_name]:
            nwp_model_utils.check_field_name(this_field_name)

    nwp_normalization_file_name = option_dict[NWP_NORM_FILE_KEY]
    error_checking.assert_file_exists(nwp_normalization_file_name)

    backup_nwp_model_name = option_dict[BACKUP_NWP_MODEL_KEY]
    backup_nwp_directory_name = option_dict[BACKUP_NWP_DIR_KEY]

    nwp_model_utils.check_model_name(
        model_name=backup_nwp_model_name, allow_ensemble=True
    )
    error_checking.assert_directory_exists(backup_nwp_directory_name)

    # All NWP data going into the simple U-net must have 2.5-km resolution.
    backup_model_ds_factor = nwp_model_utils.model_to_nbm_downsampling_factor(
        backup_nwp_model_name
    )
    error_checking.assert_equals(backup_model_ds_factor, 1)

    target_lead_time_hours = option_dict[TARGET_LEAD_TIME_KEY]
    target_field_names = option_dict[TARGET_FIELDS_KEY]
    target_dir_name = option_dict[TARGET_DIR_KEY]

    error_checking.assert_is_integer(target_lead_time_hours)
    error_checking.assert_is_greater(target_lead_time_hours, 0)
    error_checking.assert_directory_exists(target_dir_name)
    error_checking.assert_is_string_list(target_field_names)
    for this_field_name in target_field_names:
        urma_utils.check_field_name(this_field_name)

    compare_to_baseline_in_loss = option_dict[COMPARE_TO_BASELINE_IN_LOSS_KEY]
    num_examples_per_batch = option_dict[BATCH_SIZE_KEY]
    sentinel_value = option_dict[SENTINEL_VALUE_KEY]

    error_checking.assert_is_boolean(compare_to_baseline_in_loss)
    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_geq(num_examples_per_batch, 1)
    error_checking.assert_is_not_nan(sentinel_value)

    patch_size_2pt5km_pixels = option_dict[PATCH_SIZE_KEY]
    patch_buffer_size_2pt5km_pixels = option_dict[PATCH_BUFFER_SIZE_KEY]

    error_checking.assert_is_integer(patch_size_2pt5km_pixels)
    error_checking.assert_is_greater(patch_size_2pt5km_pixels, 0)
    error_checking.assert_is_integer(patch_buffer_size_2pt5km_pixels)
    error_checking.assert_is_geq(patch_buffer_size_2pt5km_pixels, 0)
    error_checking.assert_is_less_than(
        patch_buffer_size_2pt5km_pixels,
        patch_size_2pt5km_pixels // 2
    )

    # If predicting dewpoint depression, make this the last target field in the
    # list.
    predict_dewpoint_depression = (
        urma_utils.DEWPOINT_2METRE_NAME in target_field_names
    )
    if predict_dewpoint_depression:
        assert urma_utils.TEMPERATURE_2METRE_NAME in target_field_names
        target_field_names.remove(urma_utils.DEWPOINT_2METRE_NAME)
        target_field_names.append(urma_utils.DEWPOINT_2METRE_NAME)

    # If predicting gust excess, make this the last target field in the list.
    predict_gust_excess = (
        urma_utils.WIND_GUST_10METRE_NAME in target_field_names
    )
    if predict_gust_excess:
        assert urma_utils.U_WIND_10METRE_NAME in target_field_names
        assert urma_utils.V_WIND_10METRE_NAME in target_field_names
        target_field_names.remove(urma_utils.WIND_GUST_10METRE_NAME)
        target_field_names.append(urma_utils.WIND_GUST_10METRE_NAME)

    option_dict[TARGET_FIELDS_KEY] = target_field_names

    if not compare_to_baseline_in_loss:
        option_dict[RESID_BASELINE_MODEL_KEY] = None
        option_dict[RESID_BASELINE_MODEL_DIR_KEY] = None
        option_dict[RESID_BASELINE_LEAD_TIME_KEY] = -1
        return option_dict

    resid_baseline_model_name = option_dict[RESID_BASELINE_MODEL_KEY]
    resid_baseline_model_dir_name = option_dict[RESID_BASELINE_MODEL_DIR_KEY]
    resid_baseline_lead_time_hours = option_dict[RESID_BASELINE_LEAD_TIME_KEY]

    nwp_model_utils.check_model_name(
        model_name=resid_baseline_model_name, allow_ensemble=True
    )
    error_checking.assert_directory_exists(resid_baseline_model_dir_name)
    error_checking.assert_is_integer(resid_baseline_lead_time_hours)
    error_checking.assert_is_greater(resid_baseline_lead_time_hours, 0)

    # All NWP data going into the simple U-net must have 2.5-km resolution.
    resid_baseline_ds_factor = nwp_model_utils.model_to_nbm_downsampling_factor(
        resid_baseline_model_name
    )
    error_checking.assert_equals(resid_baseline_ds_factor, 1)

    return option_dict


def init_matrices_1batch(
        nwp_model_names, nwp_model_to_field_names, num_nwp_lead_times,
        target_field_names, num_target_lag_times, num_recent_bias_times,
        num_examples_per_batch, do_residual_prediction, patch_location_dict):
    """Initializes predictor and target matrices for one batch.

    :param nwp_model_names: 1-D list with names of NWP models.
    :param nwp_model_to_field_names: Dictionary.  For details, see documentation
        for `neural_net_training_simple.data_generator`.
    :param num_nwp_lead_times: Number of lead times.
    :param target_field_names: 1-D list with names of target fields.
    :param num_target_lag_times: Number of lag times for targets, to be used in
        the predictor variables.  This can be 0.
    :param num_recent_bias_times: Number of recent-bias times, to be used in
        the predictor variables.  This can be 0.
    :param num_examples_per_batch: Batch size.
    :param do_residual_prediction: Boolean flag.  If True, the NN is predicting
        difference between a given NWP forecast and the URMA truth.  If True,
        the NN is predicting the URMA truth directly.
    :param patch_location_dict: Dictionary produced by
        `misc_utils.determine_patch_locations`.  If you are training with
        full-grid data, make this None.

    :return: matrix_dict: Dictionary with the following keys.
    matrix_dict["predictor_matrix_2pt5km"]: numpy array for NWP data with 2.5-km
        resolution.  If there are no 2.5-km models, this is None instead of an
        array.
    matrix_dict["predictor_matrix_10km"]: Same but for 10-km models.
    matrix_dict["predictor_matrix_20km"]: Same but for 20-km models.
    matrix_dict["predictor_matrix_40km"]: Same but for 40-km models.
    matrix_dict["recent_bias_matrix_2pt5km"]: numpy array with recent biases for
        2.5-km NWP models.  If there are no 2.5-km models OR if
        `num_recent_bias_times == 0`, this is None.
    matrix_dict["recent_bias_matrix_10km"]: Same but for 10-km models.
    matrix_dict["recent_bias_matrix_20km"]: Same but for 20-km models.
    matrix_dict["recent_bias_matrix_40km"]: Same but for 40-km models.
    matrix_dict["predictor_matrix_resid_baseline"]: Same but for residual
        baseline.
    matrix_dict["predictor_matrix_lagged_targets"]: Same but for lagged targets.
    matrix_dict["target_matrix"]: Same but for target fields.
    """

    num_target_fields = len(target_field_names)

    if num_recent_bias_times > 0:
        nwp_model_to_target_names = nwp_input.nwp_models_to_target_fields(
            nwp_model_names=nwp_model_names,
            target_field_names=target_field_names
        )
    else:
        nwp_model_to_target_names = dict()

    downsampling_factors = numpy.array([
        nwp_model_utils.model_to_nbm_downsampling_factor(m)
        for m in nwp_model_names
    ], dtype=int)

    num_rows, num_columns = nwp_input.get_grid_dimensions(
        grid_spacing_km=2.5,
        patch_location_dict=patch_location_dict
    )

    target_matrix = numpy.full(
        (num_examples_per_batch, num_rows, num_columns, num_target_fields),
        numpy.nan
    )

    if do_residual_prediction:
        predictor_matrix_resid_baseline = numpy.full(
            (num_examples_per_batch, num_rows, num_columns, num_target_fields),
            numpy.nan
        )
    else:
        predictor_matrix_resid_baseline = None

    if num_target_lag_times == 0:
        predictor_matrix_lagged_targets = None
    else:
        predictor_matrix_lagged_targets = numpy.full(
            (num_examples_per_batch, num_rows, num_columns,
             num_target_lag_times, num_target_fields),
            numpy.nan
        )

    nwp_to_fields = nwp_model_to_field_names
    nwp_to_targets = nwp_model_to_target_names

    model_indices = numpy.where(downsampling_factors == 1)[0]
    recent_bias_matrix_2pt5km = None

    if len(model_indices) == 0:
        predictor_matrix_2pt5km = None
    else:
        first_dim = (
            num_examples_per_batch, num_rows, num_columns, num_nwp_lead_times
        )

        predictor_matrix_2pt5km = numpy.concatenate([
            numpy.full(
                first_dim + (len(nwp_to_fields[nwp_model_names[k]]),),
                numpy.nan
            )
            for k in model_indices
        ], axis=-1)

        if num_recent_bias_times > 0:
            first_dim = (
                num_examples_per_batch, num_rows, num_columns,
                num_recent_bias_times
            )

            recent_bias_matrix_2pt5km = numpy.concatenate([
                numpy.full(
                    first_dim + (len(nwp_to_targets[nwp_model_names[k]]),),
                    numpy.nan
                )
                for k in model_indices
            ], axis=-1)

    model_indices = numpy.where(downsampling_factors == 4)[0]
    num_rows, num_columns = nwp_input.get_grid_dimensions(
        grid_spacing_km=10.,
        patch_location_dict=patch_location_dict
    )
    recent_bias_matrix_10km = None

    if len(model_indices) == 0:
        predictor_matrix_10km = None
    else:
        first_dim = (
            num_examples_per_batch, num_rows, num_columns, num_nwp_lead_times
        )

        predictor_matrix_10km = numpy.concatenate([
            numpy.full(
                first_dim + (len(nwp_to_fields[nwp_model_names[k]]),),
                numpy.nan
            )
            for k in model_indices
        ], axis=-1)

        if num_recent_bias_times > 0:
            first_dim = (
                num_examples_per_batch, num_rows, num_columns,
                num_recent_bias_times
            )

            recent_bias_matrix_10km = numpy.concatenate([
                numpy.full(
                    first_dim + (len(nwp_to_targets[nwp_model_names[k]]),),
                    numpy.nan
                )
                for k in model_indices
            ], axis=-1)

    model_indices = numpy.where(downsampling_factors == 8)[0]
    num_rows, num_columns = nwp_input.get_grid_dimensions(
        grid_spacing_km=20.,
        patch_location_dict=patch_location_dict
    )
    recent_bias_matrix_20km = None

    if len(model_indices) == 0:
        predictor_matrix_20km = None
    else:
        first_dim = (
            num_examples_per_batch, num_rows, num_columns, num_nwp_lead_times
        )

        predictor_matrix_20km = numpy.concatenate([
            numpy.full(
                first_dim + (len(nwp_to_fields[nwp_model_names[k]]),),
                numpy.nan
            )
            for k in model_indices
        ], axis=-1)

        if num_recent_bias_times > 0:
            first_dim = (
                num_examples_per_batch, num_rows, num_columns,
                num_recent_bias_times
            )

            recent_bias_matrix_20km = numpy.concatenate([
                numpy.full(
                    first_dim + (len(nwp_to_targets[nwp_model_names[k]]),),
                    numpy.nan
                )
                for k in model_indices
            ], axis=-1)

    model_indices = numpy.where(downsampling_factors == 16)[0]
    num_rows, num_columns = nwp_input.get_grid_dimensions(
        grid_spacing_km=40.,
        patch_location_dict=patch_location_dict
    )
    recent_bias_matrix_40km = None

    if len(model_indices) == 0:
        predictor_matrix_40km = None
    else:
        first_dim = (
            num_examples_per_batch, num_rows, num_columns, num_nwp_lead_times
        )

        predictor_matrix_40km = numpy.concatenate([
            numpy.full(
                first_dim + (len(nwp_to_fields[nwp_model_names[k]]),),
                numpy.nan
            )
            for k in model_indices
        ], axis=-1)

        if num_recent_bias_times > 0:
            first_dim = (
                num_examples_per_batch, num_rows, num_columns,
                num_recent_bias_times
            )

            recent_bias_matrix_40km = numpy.concatenate([
                numpy.full(
                    first_dim + (len(nwp_to_targets[nwp_model_names[k]]),),
                    numpy.nan
                )
                for k in model_indices
            ], axis=-1)

    return {
        PREDICTOR_MATRIX_2PT5KM_KEY: predictor_matrix_2pt5km,
        PREDICTOR_MATRIX_10KM_KEY: predictor_matrix_10km,
        PREDICTOR_MATRIX_20KM_KEY: predictor_matrix_20km,
        PREDICTOR_MATRIX_40KM_KEY: predictor_matrix_40km,
        RECENT_BIAS_MATRIX_2PT5KM_KEY: recent_bias_matrix_2pt5km,
        RECENT_BIAS_MATRIX_10KM_KEY: recent_bias_matrix_10km,
        RECENT_BIAS_MATRIX_20KM_KEY: recent_bias_matrix_20km,
        RECENT_BIAS_MATRIX_40KM_KEY: recent_bias_matrix_40km,
        PREDICTOR_MATRIX_BASELINE_KEY: predictor_matrix_resid_baseline,
        PREDICTOR_MATRIX_LAGTGT_KEY: predictor_matrix_lagged_targets,
        TARGET_MATRIX_KEY: target_matrix
    }


def read_targets_one_example(
        init_time_unix_sec, target_lead_time_hours,
        target_field_names, target_dir_name,
        target_norm_param_table_xarray, use_quantile_norm,
        target_resid_norm_param_table_xarray, patch_location_dict):
    """Reads target fields for one example.

    M = number of rows in NBM grid (2.5-km target grid)
    N = number of columns in NBM grid (2.5-km target grid)
    F = number of target fields

    :param init_time_unix_sec: Forecast-initialization time.
    :param target_lead_time_hours: See documentation for
        `neural_net_training_simple.data_generator`.
    :param target_field_names: Same.
    :param target_dir_name: Same.
    :param target_norm_param_table_xarray: xarray table with normalization
        parameters for target variables.
    :param use_quantile_norm: See documentation for
        `neural_net_training_simple.data_generator`.
    :param target_resid_norm_param_table_xarray: xarray table with
        residual-normalization parameters for target variables.
    :param patch_location_dict: Dictionary produced by
        `misc_utils.determine_patch_locations`.  If you are training with
        full-grid data, make this None.
    :return: target_matrix: M-by-N-by-F numpy array of target values.
    """

    if target_lead_time_hours > 0:
        target_norm_param_table_xarray = None
    else:
        pass
        # assert target_norm_param_table_xarray is not None

    target_valid_time_unix_sec = (
        init_time_unix_sec + HOURS_TO_SECONDS * target_lead_time_hours
    )
    target_valid_date_string = time_conversion.unix_sec_to_string(
        target_valid_time_unix_sec, urma_io.DATE_FORMAT
    )

    urma_file_name = urma_io.find_file(
        directory_name=target_dir_name,
        valid_date_string=target_valid_date_string,
        raise_error_if_missing=False
    )

    if not os.path.isfile(urma_file_name):
        warning_string = (
            'POTENTIAL ERROR: Could not find file expected at: "{0:s}"'
        ).format(urma_file_name)

        warnings.warn(warning_string)
        return None

    print('Reading data from: "{0:s}"...'.format(urma_file_name))
    urma_table_xarray = urma_io.read_file(urma_file_name)
    urma_table_xarray = urma_utils.subset_by_time(
        urma_table_xarray=urma_table_xarray,
        desired_times_unix_sec=
        numpy.array([target_valid_time_unix_sec], dtype=int)
    )
    urma_table_xarray = urma_utils.subset_by_field(
        urma_table_xarray=urma_table_xarray,
        desired_field_names=target_field_names
    )

    if target_norm_param_table_xarray is None:
        data_matrix = urma_table_xarray[urma_utils.DATA_KEY].values

        if urma_utils.TEMPERATURE_2METRE_NAME in target_field_names:
            k = numpy.where(
                urma_table_xarray.coords[urma_utils.FIELD_DIM].values ==
                urma_utils.TEMPERATURE_2METRE_NAME
            )[0][0]

            data_matrix[..., k] = temperature_conv.kelvins_to_celsius(
                data_matrix[..., k]
            )

        if urma_utils.DEWPOINT_2METRE_NAME in target_field_names:
            k = numpy.where(
                urma_table_xarray.coords[urma_utils.FIELD_DIM].values ==
                urma_utils.DEWPOINT_2METRE_NAME
            )[0][0]

            data_matrix[..., k] = temperature_conv.kelvins_to_celsius(
                data_matrix[..., k]
            )

        urma_table_xarray = urma_table_xarray.assign({
            urma_utils.DATA_KEY: (
                urma_table_xarray[urma_utils.DATA_KEY].dims, data_matrix
            )
        })
    else:
        print('Normalizing target variables to z-scores...')
        urma_table_xarray = non_resid_normalization.normalize_targets(
            urma_table_xarray=urma_table_xarray,
            norm_param_table_xarray=target_norm_param_table_xarray,
            use_quantile_norm=use_quantile_norm
        )

        if target_resid_norm_param_table_xarray is not None:
            print('Normalizing target variables to residual scores...')
            urma_table_xarray = resid_normalization.normalize_targets(
                urma_table_xarray=urma_table_xarray,
                norm_param_table_xarray=target_resid_norm_param_table_xarray
            )

    target_matrix = numpy.transpose(
        urma_table_xarray[urma_utils.DATA_KEY].values[0, ...],
        axes=(1, 0, 2)
    )

    if patch_location_dict is not None:
        i_start = patch_location_dict[misc_utils.ROW_LIMITS_2PT5KM_KEY][0]
        i_end = patch_location_dict[misc_utils.ROW_LIMITS_2PT5KM_KEY][1] + 1
        j_start = patch_location_dict[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][0]
        j_end = patch_location_dict[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][1] + 1

        target_matrix = target_matrix[i_start:i_end, j_start:j_end]

    return target_matrix


def find_relevant_init_times(first_time_by_period_unix_sec,
                             last_time_by_period_unix_sec, nwp_model_names):
    """Finds relevant model-initialization times.

    P = number of continuous periods

    :param first_time_by_period_unix_sec: length-P numpy array with start time
        of each period.
    :param last_time_by_period_unix_sec: length-P numpy array with end time
        of each period.
    :param nwp_model_names: 1-D list with names of NWP models used for predictors
        (each list item must be accepted by `nwp_model_utils.check_model_name`).
    :return: relevant_init_times_unix_sec: 1-D numpy array of relevant init
        times.
    """

    error_checking.assert_is_integer_numpy_array(first_time_by_period_unix_sec)
    error_checking.assert_is_numpy_array(
        first_time_by_period_unix_sec, num_dimensions=1
    )
    num_periods = len(first_time_by_period_unix_sec)

    error_checking.assert_is_integer_numpy_array(last_time_by_period_unix_sec)
    error_checking.assert_is_numpy_array(
        last_time_by_period_unix_sec,
        exact_dimensions=numpy.array([num_periods], dtype=int)
    )
    error_checking.assert_is_geq_numpy_array(
        last_time_by_period_unix_sec - first_time_by_period_unix_sec, 0
    )

    error_checking.assert_is_string_list(nwp_model_names)

    nwp_init_time_intervals_sec = numpy.array([
        nwp_model_utils.model_to_init_time_interval(m) for m in nwp_model_names
    ], dtype=int)

    slow_refresh_flags = nwp_init_time_intervals_sec > 6 * HOURS_TO_SECONDS
    if numpy.mean(slow_refresh_flags) > 0.5:
        nn_init_time_interval_sec = 12 * HOURS_TO_SECONDS
    else:
        nn_init_time_interval_sec = 6 * HOURS_TO_SECONDS

    num_periods = len(first_time_by_period_unix_sec)
    relevant_init_times_unix_sec = numpy.array([], dtype=int)

    for i in range(num_periods):
        these_init_times_unix_sec = time_periods.range_and_interval_to_list(
            start_time_unix_sec=first_time_by_period_unix_sec[i],
            end_time_unix_sec=last_time_by_period_unix_sec[i],
            time_interval_sec=nn_init_time_interval_sec,
            include_endpoint=True
        )

        # TODO(thunderhoser): This HACKY if-condition is designed to deal with a
        # situation where nn_init_time_interval_sec = 43200 (12 hours) and
        # first_init_time = last_init_time != a multiple of 12 hours.
        if (
                first_time_by_period_unix_sec[i] ==
                last_time_by_period_unix_sec[i]
                and first_time_by_period_unix_sec[i]
                not in these_init_times_unix_sec
        ):
            continue

        relevant_init_times_unix_sec = numpy.concatenate([
            relevant_init_times_unix_sec, these_init_times_unix_sec
        ])

    return misc_utils.remove_unused_days(relevant_init_times_unix_sec)


def find_metafile(model_file_name, raise_error_if_missing=True):
    """Finds metafile for neural net.

    :param model_file_name: Path to model file.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: metafile_name: Path to metafile.
    """

    error_checking.assert_is_string(model_file_name)
    metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0]
    )

    if raise_error_if_missing and not os.path.isfile(metafile_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            metafile_name
        )
        raise ValueError(error_string)

    return metafile_name


def write_metafile(
        pickle_file_name, num_epochs, use_exp_moving_average_with_decay,
        num_training_batches_per_epoch,
        training_option_dict, num_validation_batches_per_epoch,
        validation_option_dict, loss_function_string, optimizer_function_string,
        metric_function_strings,
        u_net_architecture_dict, chiu_net_architecture_dict,
        chiu_net_pp_architecture_dict, chiu_next_pp_architecture_dict,
        chiu_next_ppp_architecture_dict,
        plateau_patience_epochs, plateau_learning_rate_multiplier,
        early_stopping_patience_epochs, patch_overlap_fast_gen_2pt5km_pixels,
        cosine_annealing_dict, cosine_annealing_with_restarts_dict):
    """Writes metadata to Pickle file.

    :param pickle_file_name: Path to output file.
    :param num_epochs: See doc for `neural_net_training_multipatch.train_model`.
    :param use_exp_moving_average_with_decay: Same.
    :param num_training_batches_per_epoch: Same.
    :param training_option_dict: Same.
    :param num_validation_batches_per_epoch: Same.
    :param validation_option_dict: Same.
    :param loss_function_string: Same.
    :param optimizer_function_string: Same.
    :param metric_function_strings: Same.
    :param u_net_architecture_dict: Same.
    :param chiu_net_architecture_dict: Same.
    :param chiu_net_pp_architecture_dict: Same.
    :param chiu_next_pp_architecture_dict: Same.
    :param chiu_next_ppp_architecture_dict: Same.
    :param plateau_patience_epochs: Same.
    :param plateau_learning_rate_multiplier: Same.
    :param early_stopping_patience_epochs: Same.
    :param patch_overlap_fast_gen_2pt5km_pixels: Same.
    :param cosine_annealing_dict: Same.
    :param cosine_annealing_with_restarts_dict: Same.
    """

    metadata_dict = {
        NUM_EPOCHS_KEY: num_epochs,
        EMA_DECAY_KEY: use_exp_moving_average_with_decay,
        NUM_TRAINING_BATCHES_KEY: num_training_batches_per_epoch,
        TRAINING_OPTIONS_KEY: training_option_dict,
        NUM_VALIDATION_BATCHES_KEY: num_validation_batches_per_epoch,
        VALIDATION_OPTIONS_KEY: validation_option_dict,
        LOSS_FUNCTION_KEY: loss_function_string,
        OPTIMIZER_FUNCTION_KEY: optimizer_function_string,
        METRIC_FUNCTIONS_KEY: metric_function_strings,
        U_NET_ARCHITECTURE_KEY: u_net_architecture_dict,
        CHIU_NET_ARCHITECTURE_KEY: chiu_net_architecture_dict,
        CHIU_NET_PP_ARCHITECTURE_KEY: chiu_net_pp_architecture_dict,
        CHIU_NEXT_PP_ARCHITECTURE_KEY: chiu_next_pp_architecture_dict,
        CHIU_NEXT_PPP_ARCHITECTURE_KEY: chiu_next_ppp_architecture_dict,
        PLATEAU_PATIENCE_KEY: plateau_patience_epochs,
        PLATEAU_LR_MUTIPLIER_KEY: plateau_learning_rate_multiplier,
        EARLY_STOPPING_PATIENCE_KEY: early_stopping_patience_epochs,
        PATCH_OVERLAP_FOR_FAST_GEN_KEY: patch_overlap_fast_gen_2pt5km_pixels,
        COSINE_ANNEALING_KEY: cosine_annealing_dict,
        COSINE_ANNEALING_WITH_RESTARTS_KEY: cosine_annealing_with_restarts_dict
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(metadata_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_metafile(pickle_file_name):
    """Reads metadata from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict["num_epochs"]: See doc for
        `neural_net_training_multipatch.train_model`.
    metadata_dict["use_exp_moving_average_with_decay"]: Same.
    metadata_dict["num_training_batches_per_epoch"]: Same.
    metadata_dict["training_option_dict"]: Same.
    metadata_dict["num_validation_batches_per_epoch"]: Same.
    metadata_dict["validation_option_dict"]: Same.
    metadata_dict["loss_function_string"]: Same.
    metadata_dict["optimizer_function_string"]: Same.
    metadata_dict["metric_function_strings"]: Same.
    metadata_dict["u_net_architecture_dict"]: Same.
    metadata_dict["chiu_net_architecture_dict"]: Same.
    metadata_dict["chiu_net_pp_architecture_dict"]: Same.
    metadata_dict["chiu_next_pp_architecture_dict"]: Same.
    metadata_dict["chiu_next_ppp_architecture_dict"]: Same.
    metadata_dict["plateau_patience_epochs"]: Same.
    metadata_dict["plateau_learning_rate_multiplier"]: Same.
    metadata_dict["early_stopping_patience_epochs"]: Same.
    metadata_dict["patch_overlap_fast_gen_2pt5km_pixels"]: Same.
    metadata_dict["cosine_annealing_dict"]: Same.
    metadata_dict["cosine_annealing_with_restarts_dict"]: Same.

    :raises: ValueError: if any expected key is not found in dictionary.
    """

    error_checking.assert_file_exists(pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'rb')
    metadata_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    if PATCH_OVERLAP_FOR_FAST_GEN_KEY not in metadata_dict:
        metadata_dict[PATCH_OVERLAP_FOR_FAST_GEN_KEY] = None
    if COSINE_ANNEALING_KEY not in metadata_dict:
        metadata_dict[COSINE_ANNEALING_KEY] = None
    if COSINE_ANNEALING_WITH_RESTARTS_KEY not in metadata_dict:
        metadata_dict[COSINE_ANNEALING_WITH_RESTARTS_KEY] = None
    if U_NET_ARCHITECTURE_KEY not in metadata_dict:
        metadata_dict[U_NET_ARCHITECTURE_KEY] = None
    if CHIU_NEXT_PP_ARCHITECTURE_KEY not in metadata_dict:
        metadata_dict[CHIU_NEXT_PP_ARCHITECTURE_KEY] = None
    if CHIU_NEXT_PPP_ARCHITECTURE_KEY not in metadata_dict:
        metadata_dict[CHIU_NEXT_PPP_ARCHITECTURE_KEY] = None
    if EMA_DECAY_KEY not in metadata_dict:
        metadata_dict[EMA_DECAY_KEY] = None

    training_option_dict = metadata_dict[TRAINING_OPTIONS_KEY]
    validation_option_dict = metadata_dict[VALIDATION_OPTIONS_KEY]

    if PATCH_BUFFER_SIZE_KEY not in training_option_dict:
        training_option_dict[PATCH_BUFFER_SIZE_KEY] = 0
        validation_option_dict[PATCH_BUFFER_SIZE_KEY] = 0
    if REQUIRE_ALL_PREDICTORS_KEY not in training_option_dict:
        training_option_dict[REQUIRE_ALL_PREDICTORS_KEY] = False
        validation_option_dict[REQUIRE_ALL_PREDICTORS_KEY] = False
    if COMPARE_TO_BASELINE_IN_LOSS_KEY not in training_option_dict:
        training_option_dict[COMPARE_TO_BASELINE_IN_LOSS_KEY] = False
        validation_option_dict[COMPARE_TO_BASELINE_IN_LOSS_KEY] = False
    if TARGET_LAG_TIMES_KEY not in training_option_dict:
        training_option_dict[TARGET_LAG_TIMES_KEY] = None
        validation_option_dict[TARGET_LAG_TIMES_KEY] = None
    if RECENT_BIAS_LAG_TIMES_KEY not in training_option_dict:
        training_option_dict[RECENT_BIAS_LAG_TIMES_KEY] = None
        validation_option_dict[RECENT_BIAS_LAG_TIMES_KEY] = None
        training_option_dict[RECENT_BIAS_LEAD_TIMES_KEY] = None
        validation_option_dict[RECENT_BIAS_LEAD_TIMES_KEY] = None
    if NWP_RESID_NORM_FILE_KEY not in training_option_dict:
        training_option_dict[NWP_RESID_NORM_FILE_KEY] = None
        validation_option_dict[NWP_RESID_NORM_FILE_KEY] = None
    if TARGET_RESID_NORM_FILE_KEY not in training_option_dict:
        training_option_dict[TARGET_RESID_NORM_FILE_KEY] = None
        validation_option_dict[TARGET_RESID_NORM_FILE_KEY] = None

    metadata_dict[TRAINING_OPTIONS_KEY] = training_option_dict
    metadata_dict[VALIDATION_OPTIONS_KEY] = validation_option_dict

    missing_keys = list(set(METADATA_KEYS) - set(metadata_dict.keys()))
    if len(missing_keys) == 0:
        return metadata_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), pickle_file_name)

    raise ValueError(error_string)


def read_model(hdf5_file_name, for_inference):
    """Reads model from HDF5 file.

    :param hdf5_file_name: Path to input file.
    :param for_inference: Boolean flag.  If True (False), reading the model for
        inference (further training).
    :return: model_object: Instance of `keras.models.Model`.
    """

    error_checking.assert_file_exists(hdf5_file_name)
    error_checking.assert_is_boolean(for_inference)

    metafile_name = find_metafile(
        model_file_name=hdf5_file_name, raise_error_if_missing=True
    )
    metadata_dict = read_metafile(metafile_name)
    print(metadata_dict[LOSS_FUNCTION_KEY])

    u_net_architecture_dict = metadata_dict[U_NET_ARCHITECTURE_KEY]
    if u_net_architecture_dict is not None:
        from ml_for_national_blend.machine_learning import u_net_architecture

        arch_dict = u_net_architecture_dict

        for this_key in [
                u_net_architecture.LOSS_FUNCTION_KEY,
                u_net_architecture.OPTIMIZER_FUNCTION_KEY
        ]:
            arch_dict[this_key] = eval(arch_dict[this_key])

        for this_key in [u_net_architecture.METRIC_FUNCTIONS_KEY]:
            for k in range(len(arch_dict[this_key])):
                arch_dict[this_key][k] = eval(arch_dict[this_key][k])

        model_object = u_net_architecture.create_model(arch_dict)
        model_object.load_weights(hdf5_file_name)

        if for_inference and metadata_dict[EMA_DECAY_KEY] is not None:
            set_model_weights_to_ema(
                model_object=model_object, metafile_name=metafile_name
            )

        return model_object

    chiu_net_architecture_dict = metadata_dict[CHIU_NET_ARCHITECTURE_KEY]
    if chiu_net_architecture_dict is not None:
        from ml_for_national_blend.machine_learning import chiu_net_architecture

        arch_dict = chiu_net_architecture_dict

        for this_key in [
                chiu_net_architecture.LOSS_FUNCTION_KEY,
                chiu_net_architecture.OPTIMIZER_FUNCTION_KEY
        ]:
            arch_dict[this_key] = eval(arch_dict[this_key])

        for this_key in [chiu_net_architecture.METRIC_FUNCTIONS_KEY]:
            for k in range(len(arch_dict[this_key])):
                arch_dict[this_key][k] = eval(arch_dict[this_key][k])

        model_object = chiu_net_architecture.create_model(arch_dict)
        model_object.load_weights(hdf5_file_name)

        if for_inference and metadata_dict[EMA_DECAY_KEY] is not None:
            set_model_weights_to_ema(
                model_object=model_object, metafile_name=metafile_name
            )

        return model_object

    chiu_net_pp_architecture_dict = metadata_dict[CHIU_NET_PP_ARCHITECTURE_KEY]
    if chiu_net_pp_architecture_dict is not None:
        from ml_for_national_blend.machine_learning import \
            chiu_net_pp_architecture

        arch_dict = chiu_net_pp_architecture_dict

        for this_key in [
                chiu_net_pp_architecture.LOSS_FUNCTION_KEY,
                chiu_net_pp_architecture.OPTIMIZER_FUNCTION_KEY
        ]:
            arch_dict[this_key] = eval(arch_dict[this_key])

        for this_key in [chiu_net_pp_architecture.METRIC_FUNCTIONS_KEY]:
            for k in range(len(arch_dict[this_key])):
                arch_dict[this_key][k] = eval(arch_dict[this_key][k])

        model_object = chiu_net_pp_architecture.create_model(arch_dict)
        model_object.load_weights(hdf5_file_name)

        if for_inference and metadata_dict[EMA_DECAY_KEY] is not None:
            set_model_weights_to_ema(
                model_object=model_object, metafile_name=metafile_name
            )

        return model_object

    chiu_next_pp_architecture_dict = metadata_dict[
        CHIU_NEXT_PP_ARCHITECTURE_KEY
    ]
    if chiu_next_pp_architecture_dict is not None:
        from ml_for_national_blend.machine_learning import \
            chiu_next_pp_architecture

        arch_dict = chiu_next_pp_architecture_dict

        for this_key in [
                chiu_next_pp_architecture.LOSS_FUNCTION_KEY,
                chiu_next_pp_architecture.OPTIMIZER_FUNCTION_KEY
        ]:
            arch_dict[this_key] = eval(arch_dict[this_key])

        for this_key in [chiu_next_pp_architecture.METRIC_FUNCTIONS_KEY]:
            for k in range(len(arch_dict[this_key])):
                arch_dict[this_key][k] = eval(arch_dict[this_key][k])

        model_object = chiu_next_pp_architecture.create_model(arch_dict)
        model_object.load_weights(hdf5_file_name)

        if for_inference and metadata_dict[EMA_DECAY_KEY] is not None:
            set_model_weights_to_ema(
                model_object=model_object, metafile_name=metafile_name
            )

        return model_object

    chiu_next_ppp_architecture_dict = metadata_dict[
        CHIU_NEXT_PPP_ARCHITECTURE_KEY
    ]
    if chiu_next_ppp_architecture_dict is not None:
        from ml_for_national_blend.machine_learning import \
            chiu_next_ppp_architecture

        arch_dict = chiu_next_ppp_architecture_dict

        for this_key in [
                chiu_next_ppp_architecture.LOSS_FUNCTION_KEY,
                chiu_next_ppp_architecture.OPTIMIZER_FUNCTION_KEY
        ]:
            arch_dict[this_key] = eval(arch_dict[this_key])

        for this_key in [chiu_next_ppp_architecture.METRIC_FUNCTIONS_KEY]:
            for k in range(len(arch_dict[this_key])):
                arch_dict[this_key][k] = eval(arch_dict[this_key][k])

        model_object = chiu_next_ppp_architecture.create_model(arch_dict)
        model_object.load_weights(hdf5_file_name)

        if for_inference and metadata_dict[EMA_DECAY_KEY] is not None:
            set_model_weights_to_ema(
                model_object=model_object, metafile_name=metafile_name
            )

        return model_object

    custom_object_dict = {
        'loss': eval(metadata_dict[LOSS_FUNCTION_KEY])
    }
    model_object = load_model(
        hdf5_file_name, custom_objects=custom_object_dict, compile=False
    )

    metric_function_list = [
        eval(m) for m in metadata_dict[METRIC_FUNCTIONS_KEY]
    ]
    model_object.compile(
        loss=custom_object_dict['loss'],
        optimizer=eval(metadata_dict[OPTIMIZER_FUNCTION_KEY]),
        metrics=metric_function_list
    )

    return model_object


def apply_many_single_patch_models(
        model_objects, model_metadata_dicts, full_predictor_matrices,
        num_examples_per_batch, patch_buffer_size_px,
        patch_start_row_by_model, patch_start_column_by_model,
        verbose=True):
    """Stitches together predictions from many single-patch (subdomain) models.

    A = number of models

    E = number of examples
    M = number of rows in full grid
    N = number of columns in full grid
    F = number of target fields
    S = ensemble size

    :param model_objects: length-A list of trained neural nets (instances of
        `keras.models.Model`).
    :param model_metadata_dicts: length-A list of dictionaries returned by
        `neural_net_utils.read_metafile`.
    :param full_predictor_matrices: See output doc for
        `neural_net_training_simple.data_generator`.
    :param num_examples_per_batch: Batch size.
    :param patch_buffer_size_px: Size of buffer zone between inner (target) and
        outer (predictor) domains.  Units are number of pixels on the 2.5-km NBM
        grid.
    :param patch_start_row_by_model: length-A numpy array, where
        patch_start_row_by_model[k] is the first row in the subdomain covered by
        the [k]th model.  This should be a row index on the 2.5-km NBM grid, and
        it should be the first row in the outer domain, not the inner domain.
    :param patch_start_column_by_model: Same as above but for columns.
    :param verbose: Boolean flag.  If True, will print progress messages.
    :return: prediction_matrix: E-by-M-by-N-by-F-by-S numpy array of predicted
        values.
    """

    # Check input args.
    error_checking.assert_is_list(model_objects)
    num_models = len(model_objects)

    error_checking.assert_is_list(model_metadata_dicts)
    assert len(model_metadata_dicts) == num_models

    error_checking.assert_is_integer(patch_buffer_size_px)
    error_checking.assert_is_greater(patch_buffer_size_px, 0)

    error_checking.assert_is_integer_numpy_array(patch_start_row_by_model)
    error_checking.assert_is_geq_numpy_array(patch_start_row_by_model, 0)
    error_checking.assert_is_numpy_array(
        patch_start_row_by_model,
        exact_dimensions=numpy.array([num_models], dtype=int)
    )

    error_checking.assert_is_integer_numpy_array(patch_start_column_by_model)
    error_checking.assert_is_geq_numpy_array(patch_start_column_by_model, 0)
    error_checking.assert_is_numpy_array(
        patch_start_column_by_model,
        exact_dimensions=numpy.array([num_models], dtype=int)
    )

    patch_start_rowcol_matrix = numpy.transpose(numpy.vstack(
        [patch_start_row_by_model, patch_start_column_by_model]
    ))
    error_checking.assert_equals(
        patch_start_rowcol_matrix.shape[0],
        numpy.unique(patch_start_rowcol_matrix, axis=0).shape[0]
    )

    error_checking.assert_is_boolean(verbose)

    # Do actual stuff.
    num_rows_2pt5km = len(nbm_utils.NBM_Y_COORDS_METRES)
    num_columns_2pt5km = len(nbm_utils.NBM_X_COORDS_METRES)

    target_field_names = []
    patch_size_by_model_px = numpy.full(num_models, -1, dtype=int)

    for k in range(num_models):
        this_vod = model_metadata_dicts[k][VALIDATION_OPTIONS_KEY]
        these_field_names = this_vod[TARGET_FIELDS_KEY]
        if k == 0:
            target_field_names = copy.deepcopy(these_field_names)

        assert these_field_names == target_field_names

        patch_size_by_model_px[k] = model_objects[k].layers[-1].output.shape[1]

    num_target_fields = len(target_field_names)
    num_examples = full_predictor_matrices[0].shape[0]
    these_dim = (
        num_examples, num_rows_2pt5km, num_columns_2pt5km, num_target_fields, 1
    )

    # TODO(thunderhoser): I don't know if this is the best way to handle varying
    # ensemble size by grid point.
    prediction_count_matrix = numpy.full(these_dim, 0, dtype=float)
    summed_prediction_matrix = None

    for k in range(num_models):
        if verbose:
            i_start = patch_start_row_by_model[k] + 0
            i_end = i_start + patch_size_by_model_px[k] - 1
            j_start = patch_start_column_by_model[k] + 0
            j_end = j_start + patch_size_by_model_px[k] - 1

            print((
                'Applying {0:d}th of {1:d} models '
                'to rows {2:d}-{3:d} of {4:d}, '
                'and columns {5:d}-{6:d} of {7:d}, in finest-resolution grid...'
            ).format(
                k + 1, num_models,
                i_start, i_end, num_rows_2pt5km,
                j_start, j_end, num_columns_2pt5km
            ))

        patch_predictor_matrices = []

        for this_full_pred_matrix in full_predictor_matrices:
            this_downsampling_factor = int(numpy.round(
                float(num_rows_2pt5km) /
                this_full_pred_matrix.shape[1]
            ))
            assert this_downsampling_factor in POSSIBLE_DOWNSAMPLING_FACTORS

            i_start = patch_start_row_by_model[k] + 0
            i_end = i_start + patch_size_by_model_px[k]
            j_start = patch_start_column_by_model[k] + 0
            j_end = j_start + patch_size_by_model_px[k]

            i_start = i_start // this_downsampling_factor
            i_end = i_end // this_downsampling_factor
            j_start = j_start // this_downsampling_factor
            j_end = j_end // this_downsampling_factor

            patch_predictor_matrices.append(
                this_full_pred_matrix[:, i_start:i_end, j_start:j_end, ...]
            )

        found_nan = any([
            numpy.any(numpy.isnan(ppm)) for ppm in patch_predictor_matrices
        ])
        if found_nan:
            warning_string = (
                'POTENTIAL MAJOR ERROR: Found NaN values in predictor matrices '
                'for {0:d}th of {1:d} models!'
            ).format(
                k + 1, num_models
            )

            warnings.warn(warning_string)
            continue

        patch_prediction_matrix = nn_training_simple.apply_model(
            model_object=model_objects[k],
            predictor_matrices=patch_predictor_matrices,
            num_examples_per_batch=num_examples_per_batch,
            target_field_names=target_field_names,
            verbose=False
        )

        if summed_prediction_matrix is None:
            ensemble_size = patch_prediction_matrix.shape[-1]
            these_dim = (
                num_examples, num_rows_2pt5km, num_columns_2pt5km,
                num_target_fields, ensemble_size
            )
            summed_prediction_matrix = numpy.full(these_dim, 0.)

        i_start = patch_start_row_by_model[k] + 0
        i_end = i_start + patch_size_by_model_px[k]
        j_start = patch_start_column_by_model[k] + 0
        j_end = j_start + patch_size_by_model_px[k]

        weight_matrix = patch_buffer_to_mask(
            patch_size_2pt5km_pixels=patch_size_by_model_px[k],
            patch_buffer_size_2pt5km_pixels=patch_buffer_size_px
        )
        weight_matrix = numpy.expand_dims(weight_matrix, axis=0)
        weight_matrix = numpy.expand_dims(weight_matrix, axis=-1)
        weight_matrix = numpy.expand_dims(weight_matrix, axis=-1)

        summed_prediction_matrix[:, i_start:i_end, j_start:j_end, ...] += (
            weight_matrix * patch_prediction_matrix
        )
        prediction_count_matrix[:, i_start:i_end, j_start:j_end, ...] += (
            weight_matrix
        )

    if verbose:
        print('Have applied model everywhere in full grid!')

    prediction_count_matrix = prediction_count_matrix.astype(float)
    prediction_count_matrix[prediction_count_matrix < TOLERANCE] = numpy.nan
    return summed_prediction_matrix / prediction_count_matrix
