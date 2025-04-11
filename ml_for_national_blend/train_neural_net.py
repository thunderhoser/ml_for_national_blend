"""Trains neural net."""

import os
import sys
import copy
import json
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import nwp_model_utils
import neural_net_utils as nn_utils
import neural_net_training_simple as nn_training_simple
import neural_net_training_multipatch as nn_training_multipatch
import training_args

TIME_FORMAT = '%Y-%m-%d-%H'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER = training_args.add_input_args(parser_object=INPUT_ARG_PARSER)


def _process_nwp_directories(nwp_directory_names, nwp_model_names):
    """Processes NWP directories for either training or validation data.

    :param nwp_directory_names: See documentation for input arg
        "nwp_dir_names_for_training" to this script.
    :param nwp_model_names: See documentation for input arg to this script.
    :return: nwp_model_to_dir_name: Dictionary, where each key is the name of an
        NWP model and the corresponding value is the input directory.
    """

    # assert len(nwp_model_names) == len(nwp_directory_names)
    nwp_directory_names = nwp_directory_names[:len(nwp_model_names)]

    if len(nwp_directory_names) == 1:
        found_any_model_name_in_dir_name = any([
            m in nwp_directory_names[0]
            for m in nwp_model_utils.ALL_MODEL_NAMES_WITH_ENSEMBLE
        ])
        infer_directories = (
            len(nwp_model_names) > 1 or
            (len(nwp_model_names) == 1 and not found_any_model_name_in_dir_name)
        )
    else:
        infer_directories = False

    if infer_directories:
        top_directory_name = copy.deepcopy(nwp_directory_names[0])
        nwp_directory_names = [
            '{0:s}/{1:s}/processed/interp_to_nbm_grid'.format(
                top_directory_name, m
            ) for m in nwp_model_names
        ]

    return dict(zip(nwp_model_names, nwp_directory_names))


def _run(template_file_name, output_dir_name,
         nwp_lead_times_hours, nwp_model_names, nwp_model_to_field_names,
         nwp_normalization_file_name, nwp_use_quantile_norm,
         nwp_resid_norm_file_name,
         backup_nwp_model_name, backup_nwp_dir_name,
         target_lead_time_hours, target_field_names, target_lag_times_hours,
         target_normalization_file_name, targets_use_quantile_norm,
         target_resid_norm_file_name,
         recent_bias_init_time_lags_hours, recent_bias_lead_times_hours,
         nbm_constant_field_names, nbm_constant_file_name,
         compare_to_baseline_in_loss, num_examples_per_batch, sentinel_value,
         patch_size_2pt5km_pixels, patch_buffer_size_2pt5km_pixels,
         patch_start_row_2pt5km, patch_start_column_2pt5km,
         use_fast_patch_generator, patch_overlap_size_2pt5km_pixels,
         require_all_predictors,
         do_residual_prediction, resid_baseline_model_name,
         resid_baseline_lead_time_hours, resid_baseline_model_dir_name,
         first_init_time_strings_for_training,
         last_init_time_strings_for_training,
         nwp_dir_names_for_training, target_dir_name_for_training,
         first_init_time_strings_for_validation,
         last_init_time_strings_for_validation,
         nwp_dir_names_for_validation, target_dir_name_for_validation,
         num_epochs, use_exp_moving_average_with_decay,
         num_training_batches_per_epoch, num_validation_batches_per_epoch,
         plateau_patience_epochs, plateau_learning_rate_multiplier,
         early_stopping_patience_epochs):
    """Trains neural net.

    This is effectively the main method.

    :param template_file_name: See documentation at top of this script.
    :param output_dir_name: Same.
    :param nwp_lead_times_hours: Same.
    :param nwp_model_names: Same.
    :param nwp_model_to_field_names: Same.
    :param nwp_normalization_file_name: Same.
    :param nwp_resid_norm_file_name: Same.
    :param nwp_use_quantile_norm: Same.
    :param backup_nwp_model_name: Same.
    :param backup_nwp_dir_name: Same.
    :param target_lead_time_hours: Same.
    :param target_field_names: Same.
    :param target_lag_times_hours: Same.
    :param target_normalization_file_name: Same.
    :param target_resid_norm_file_name: Same.
    :param targets_use_quantile_norm: Same.
    :param recent_bias_init_time_lags_hours: Same.
    :param recent_bias_lead_times_hours: Same.
    :param nbm_constant_field_names: Same.
    :param nbm_constant_file_name: Same.
    :param compare_to_baseline_in_loss: Same.
    :param num_examples_per_batch: Same.
    :param sentinel_value: Same.
    :param patch_size_2pt5km_pixels: Same.
    :param patch_buffer_size_2pt5km_pixels: Same.
    :param patch_start_row_2pt5km: Same.
    :param patch_start_column_2pt5km: Same.
    :param use_fast_patch_generator: Same.
    :param patch_overlap_size_2pt5km_pixels: Same.
    :param require_all_predictors: Same.
    :param do_residual_prediction: Same.
    :param resid_baseline_model_name: Same.
    :param resid_baseline_lead_time_hours: Same.
    :param resid_baseline_model_dir_name: Same.
    :param first_init_time_strings_for_training: Same.
    :param last_init_time_strings_for_training: Same.
    :param nwp_dir_names_for_training: Same.
    :param target_dir_name_for_training: Same.
    :param first_init_time_strings_for_validation: Same.
    :param last_init_time_strings_for_validation: Same.
    :param nwp_dir_names_for_validation: Same.
    :param target_dir_name_for_validation: Same.
    :param num_epochs: Same.
    :param use_exp_moving_average_with_decay: Same.
    :param num_training_batches_per_epoch: Same.
    :param num_validation_batches_per_epoch: Same.
    :param plateau_patience_epochs: Same.
    :param plateau_learning_rate_multiplier: Same.
    :param early_stopping_patience_epochs: Same.
    """

    if nwp_resid_norm_file_name == '':
        nwp_resid_norm_file_name = None
    if target_normalization_file_name == '':
        target_normalization_file_name = None
    if target_resid_norm_file_name == '':
        target_resid_norm_file_name = None
    if resid_baseline_model_name == '':
        resid_baseline_model_name = None
    if resid_baseline_model_dir_name == '':
        resid_baseline_model_dir_name = None
    if resid_baseline_lead_time_hours <= 0:
        resid_baseline_lead_time_hours = None
    if not use_fast_patch_generator:
        patch_overlap_size_2pt5km_pixels = None
    if patch_size_2pt5km_pixels < 0:
        patch_size_2pt5km_pixels = None
    if patch_start_row_2pt5km < 0:
        patch_start_row_2pt5km = None
    if patch_start_column_2pt5km < 0:
        patch_start_column_2pt5km = None
    if use_exp_moving_average_with_decay < 0:
        use_exp_moving_average_with_decay = None

    if nbm_constant_file_name == '':
        nbm_constant_file_name = None
        nbm_constant_field_names = []
    if len(nbm_constant_field_names) == 1 and nbm_constant_field_names[0] == '':
        nbm_constant_file_name = None
        nbm_constant_field_names = []
    if len(target_lag_times_hours) == 1 and target_lag_times_hours[0] < 0:
        target_lag_times_hours = None

    if (
            len(recent_bias_init_time_lags_hours) == 1 and
            recent_bias_init_time_lags_hours[0] < 0
    ):
        recent_bias_init_time_lags_hours = None

    if (
            len(recent_bias_lead_times_hours) == 1 and
            recent_bias_lead_times_hours[0] < 0
    ):
        recent_bias_lead_times_hours = None

    nwp_model_to_training_dir_name = _process_nwp_directories(
        nwp_directory_names=nwp_dir_names_for_training,
        nwp_model_names=nwp_model_names
    )
    nwp_model_to_validation_dir_name = _process_nwp_directories(
        nwp_directory_names=nwp_dir_names_for_validation,
        nwp_model_names=nwp_model_names
    )

    first_init_times_for_training_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, TIME_FORMAT)
        for t in first_init_time_strings_for_training
    ], dtype=int)
    last_init_times_for_training_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, TIME_FORMAT)
        for t in last_init_time_strings_for_training
    ], dtype=int)
    first_init_times_for_validation_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, TIME_FORMAT)
        for t in first_init_time_strings_for_validation
    ], dtype=int)
    last_init_times_for_validation_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, TIME_FORMAT)
        for t in last_init_time_strings_for_validation
    ], dtype=int)

    training_option_dict = {
        nn_utils.FIRST_INIT_TIMES_KEY: first_init_times_for_training_unix_sec,
        nn_utils.LAST_INIT_TIMES_KEY: last_init_times_for_training_unix_sec,
        nn_utils.NWP_LEAD_TIMES_KEY: nwp_lead_times_hours,
        nn_utils.NWP_MODEL_TO_DIR_KEY: nwp_model_to_training_dir_name,
        nn_utils.NWP_MODEL_TO_FIELDS_KEY: nwp_model_to_field_names,
        nn_utils.NWP_NORM_FILE_KEY: nwp_normalization_file_name,
        nn_utils.NWP_RESID_NORM_FILE_KEY: nwp_resid_norm_file_name,
        nn_utils.NWP_USE_QUANTILE_NORM_KEY: nwp_use_quantile_norm,
        nn_utils.BACKUP_NWP_MODEL_KEY: backup_nwp_model_name,
        nn_utils.BACKUP_NWP_DIR_KEY: backup_nwp_dir_name,
        nn_utils.TARGET_LEAD_TIME_KEY: target_lead_time_hours,
        nn_utils.TARGET_FIELDS_KEY: target_field_names,
        nn_utils.TARGET_LAG_TIMES_KEY: target_lag_times_hours,
        nn_utils.TARGET_DIR_KEY: target_dir_name_for_training,
        nn_utils.TARGET_NORM_FILE_KEY: target_normalization_file_name,
        nn_utils.TARGET_RESID_NORM_FILE_KEY: target_resid_norm_file_name,
        nn_utils.TARGETS_USE_QUANTILE_NORM_KEY: targets_use_quantile_norm,
        nn_utils.RECENT_BIAS_LAG_TIMES_KEY: recent_bias_init_time_lags_hours,
        nn_utils.RECENT_BIAS_LEAD_TIMES_KEY: recent_bias_lead_times_hours,
        nn_utils.NBM_CONSTANT_FIELDS_KEY: nbm_constant_field_names,
        nn_utils.NBM_CONSTANT_FILE_KEY: nbm_constant_file_name,
        nn_utils.COMPARE_TO_BASELINE_IN_LOSS_KEY: compare_to_baseline_in_loss,
        nn_utils.BATCH_SIZE_KEY: num_examples_per_batch,
        nn_utils.SENTINEL_VALUE_KEY: sentinel_value,
        nn_utils.DO_RESIDUAL_PREDICTION_KEY: do_residual_prediction,
        nn_utils.RESID_BASELINE_MODEL_KEY: resid_baseline_model_name,
        nn_utils.RESID_BASELINE_LEAD_TIME_KEY: resid_baseline_lead_time_hours,
        nn_utils.RESID_BASELINE_MODEL_DIR_KEY: resid_baseline_model_dir_name,
        nn_utils.PATCH_SIZE_KEY: patch_size_2pt5km_pixels,
        nn_utils.PATCH_BUFFER_SIZE_KEY: patch_buffer_size_2pt5km_pixels,
        nn_utils.PATCH_START_ROW_KEY: patch_start_row_2pt5km,
        nn_utils.PATCH_START_COLUMN_KEY: patch_start_column_2pt5km,
        nn_utils.REQUIRE_ALL_PREDICTORS_KEY: require_all_predictors
    }

    validation_option_dict = {
        nn_utils.FIRST_INIT_TIMES_KEY: first_init_times_for_validation_unix_sec,
        nn_utils.LAST_INIT_TIMES_KEY: last_init_times_for_validation_unix_sec,
        nn_utils.NWP_MODEL_TO_DIR_KEY: nwp_model_to_validation_dir_name,
        nn_utils.TARGET_DIR_KEY: target_dir_name_for_validation
    }

    print('Reading model template from: "{0:s}"...'.format(template_file_name))
    model_object = nn_utils.read_model(
        hdf5_file_name=template_file_name, for_inference=False
    )

    model_metafile_name = nn_utils.find_metafile(
        model_file_name=template_file_name, raise_error_if_missing=True
    )
    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = nn_utils.read_metafile(model_metafile_name)
    mmd = model_metadata_dict

    if use_fast_patch_generator:
        nn_training_multipatch.train_model(
            model_object=model_object,
            num_epochs=num_epochs,
            use_exp_moving_average_with_decay=use_exp_moving_average_with_decay,
            num_training_batches_per_epoch=num_training_batches_per_epoch,
            training_option_dict=training_option_dict,
            num_validation_batches_per_epoch=num_validation_batches_per_epoch,
            validation_option_dict=validation_option_dict,
            loss_function_string=mmd[nn_utils.LOSS_FUNCTION_KEY],
            optimizer_function_string=mmd[nn_utils.OPTIMIZER_FUNCTION_KEY],
            metric_function_strings=mmd[nn_utils.METRIC_FUNCTIONS_KEY],
            u_net_architecture_dict=mmd[nn_utils.U_NET_ARCHITECTURE_KEY],
            chiu_net_architecture_dict=mmd[nn_utils.CHIU_NET_ARCHITECTURE_KEY],
            chiu_net_pp_architecture_dict=
            mmd[nn_utils.CHIU_NET_PP_ARCHITECTURE_KEY],
            chiu_next_pp_architecture_dict=
            mmd[nn_utils.CHIU_NEXT_PP_ARCHITECTURE_KEY],
            plateau_patience_epochs=plateau_patience_epochs,
            plateau_learning_rate_multiplier=plateau_learning_rate_multiplier,
            early_stopping_patience_epochs=early_stopping_patience_epochs,
            patch_overlap_fast_gen_2pt5km_pixels=
            patch_overlap_size_2pt5km_pixels,
            output_dir_name=output_dir_name
        )
    else:
        nn_training_simple.train_model(
            model_object=model_object,
            num_epochs=num_epochs,
            use_exp_moving_average_with_decay=use_exp_moving_average_with_decay,
            num_training_batches_per_epoch=num_training_batches_per_epoch,
            training_option_dict=training_option_dict,
            num_validation_batches_per_epoch=num_validation_batches_per_epoch,
            validation_option_dict=validation_option_dict,
            loss_function_string=mmd[nn_utils.LOSS_FUNCTION_KEY],
            optimizer_function_string=mmd[nn_utils.OPTIMIZER_FUNCTION_KEY],
            metric_function_strings=mmd[nn_utils.METRIC_FUNCTIONS_KEY],
            u_net_architecture_dict=mmd[nn_utils.U_NET_ARCHITECTURE_KEY],
            chiu_net_architecture_dict=mmd[nn_utils.CHIU_NET_ARCHITECTURE_KEY],
            chiu_net_pp_architecture_dict=
            mmd[nn_utils.CHIU_NET_PP_ARCHITECTURE_KEY],
            chiu_next_pp_architecture_dict=
            mmd[nn_utils.CHIU_NEXT_PP_ARCHITECTURE_KEY],
            plateau_patience_epochs=plateau_patience_epochs,
            plateau_learning_rate_multiplier=plateau_learning_rate_multiplier,
            early_stopping_patience_epochs=early_stopping_patience_epochs,
            output_dir_name=output_dir_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        template_file_name=getattr(
            INPUT_ARG_OBJECT, training_args.TEMPLATE_FILE_ARG_NAME
        ),
        output_dir_name=getattr(
            INPUT_ARG_OBJECT, training_args.OUTPUT_DIR_ARG_NAME
        ),
        nwp_lead_times_hours=numpy.array(
            getattr(INPUT_ARG_OBJECT, training_args.NWP_LEAD_TIMES_ARG_NAME),
            dtype=int
        ),
        nwp_model_names=getattr(
            INPUT_ARG_OBJECT, training_args.NWP_MODELS_ARG_NAME
        ),
        nwp_model_to_field_names=json.loads(getattr(
            INPUT_ARG_OBJECT, training_args.NWP_MODEL_TO_FIELDS_ARG_NAME
        )),
        nwp_normalization_file_name=getattr(
            INPUT_ARG_OBJECT, training_args.NWP_NORMALIZATION_FILE_ARG_NAME
        ),
        nwp_resid_norm_file_name=getattr(
            INPUT_ARG_OBJECT, training_args.NWP_RESID_NORM_FILE_ARG_NAME
        ),
        nwp_use_quantile_norm=bool(getattr(
            INPUT_ARG_OBJECT, training_args.NWP_USE_QUANTILE_NORM_ARG_NAME
        )),
        backup_nwp_model_name=getattr(
            INPUT_ARG_OBJECT, training_args.BACKUP_NWP_MODEL_ARG_NAME
        ),
        backup_nwp_dir_name=getattr(
            INPUT_ARG_OBJECT, training_args.BACKUP_NWP_DIR_ARG_NAME
        ),
        target_lead_time_hours=getattr(
            INPUT_ARG_OBJECT, training_args.TARGET_LEAD_TIME_ARG_NAME
        ),
        target_field_names=getattr(
            INPUT_ARG_OBJECT, training_args.TARGET_FIELDS_ARG_NAME
        ),
        target_lag_times_hours=numpy.array(
            getattr(INPUT_ARG_OBJECT, training_args.TARGET_LAG_TIMES_ARG_NAME),
            dtype=int
        ),
        target_normalization_file_name=getattr(
            INPUT_ARG_OBJECT, training_args.TARGET_NORMALIZATION_FILE_ARG_NAME
        ),
        target_resid_norm_file_name=getattr(
            INPUT_ARG_OBJECT, training_args.TARGET_RESID_NORM_FILE_ARG_NAME
        ),
        targets_use_quantile_norm=bool(getattr(
            INPUT_ARG_OBJECT, training_args.TARGETS_USE_QUANTILE_NORM_ARG_NAME
        )),
        recent_bias_init_time_lags_hours=numpy.array(
            getattr(
                INPUT_ARG_OBJECT,
                training_args.RECENT_BIAS_LAG_TIMES_ARG_NAME
            ),
            dtype=int
        ),
        recent_bias_lead_times_hours=numpy.array(
            getattr(
                INPUT_ARG_OBJECT,
                training_args.RECENT_BIAS_LEAD_TIMES_ARG_NAME
            ),
            dtype=int
        ),
        nbm_constant_field_names=getattr(
            INPUT_ARG_OBJECT, training_args.NBM_CONSTANT_FIELDS_ARG_NAME
        ),
        nbm_constant_file_name=getattr(
            INPUT_ARG_OBJECT, training_args.NBM_CONSTANT_FILE_ARG_NAME
        ),
        compare_to_baseline_in_loss=bool(getattr(
            INPUT_ARG_OBJECT, training_args.COMPARE_TO_BASELINE_ARG_NAME
        )),
        num_examples_per_batch=getattr(
            INPUT_ARG_OBJECT, training_args.BATCH_SIZE_ARG_NAME
        ),
        sentinel_value=getattr(
            INPUT_ARG_OBJECT, training_args.SENTINEL_VALUE_ARG_NAME
        ),
        patch_size_2pt5km_pixels=getattr(
            INPUT_ARG_OBJECT, training_args.PATCH_SIZE_ARG_NAME
        ),
        patch_buffer_size_2pt5km_pixels=getattr(
            INPUT_ARG_OBJECT, training_args.PATCH_BUFFER_SIZE_ARG_NAME
        ),
        patch_start_row_2pt5km=getattr(
            INPUT_ARG_OBJECT, training_args.PATCH_START_ROW_ARG_NAME
        ),
        patch_start_column_2pt5km=getattr(
            INPUT_ARG_OBJECT, training_args.PATCH_START_COLUMN_ARG_NAME
        ),
        use_fast_patch_generator=bool(getattr(
            INPUT_ARG_OBJECT, training_args.USE_FAST_PATCH_GENERATOR_ARG_NAME
        )),
        patch_overlap_size_2pt5km_pixels=getattr(
            INPUT_ARG_OBJECT, training_args.PATCH_OVERLAP_SIZE_ARG_NAME
        ),
        require_all_predictors=bool(getattr(
            INPUT_ARG_OBJECT, training_args.REQUIRE_ALL_PREDICTORS_ARG_NAME
        )),
        do_residual_prediction=bool(getattr(
            INPUT_ARG_OBJECT, training_args.DO_RESIDUAL_PREDICTION_ARG_NAME
        )),
        resid_baseline_model_name=getattr(
            INPUT_ARG_OBJECT, training_args.RESID_BASELINE_MODEL_ARG_NAME
        ),
        resid_baseline_lead_time_hours=getattr(
            INPUT_ARG_OBJECT, training_args.RESID_BASELINE_LEAD_TIME_ARG_NAME
        ),
        resid_baseline_model_dir_name=getattr(
            INPUT_ARG_OBJECT, training_args.RESID_BASELINE_MODEL_DIR_ARG_NAME
        ),
        first_init_time_strings_for_training=getattr(
            INPUT_ARG_OBJECT, training_args.FIRST_TRAINING_TIMES_ARG_NAME
        ),
        last_init_time_strings_for_training=getattr(
            INPUT_ARG_OBJECT, training_args.LAST_TRAINING_TIMES_ARG_NAME
        ),
        nwp_dir_names_for_training=getattr(
            INPUT_ARG_OBJECT, training_args.TRAINING_NWP_DIRS_ARG_NAME
        ),
        target_dir_name_for_training=getattr(
            INPUT_ARG_OBJECT, training_args.TRAINING_TARGET_DIR_ARG_NAME
        ),
        first_init_time_strings_for_validation=getattr(
            INPUT_ARG_OBJECT, training_args.FIRST_VALIDATION_TIMES_ARG_NAME
        ),
        last_init_time_strings_for_validation=getattr(
            INPUT_ARG_OBJECT, training_args.LAST_VALIDATION_TIMES_ARG_NAME
        ),
        nwp_dir_names_for_validation=getattr(
            INPUT_ARG_OBJECT, training_args.VALIDATION_NWP_DIRS_ARG_NAME
        ),
        target_dir_name_for_validation=getattr(
            INPUT_ARG_OBJECT, training_args.VALIDATION_TARGET_DIR_ARG_NAME
        ),
        num_epochs=getattr(INPUT_ARG_OBJECT, training_args.NUM_EPOCHS_ARG_NAME),
        use_exp_moving_average_with_decay=getattr(
            INPUT_ARG_OBJECT, training_args.EMA_DECAY_ARG_NAME
        ),
        num_training_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, training_args.NUM_TRAINING_BATCHES_ARG_NAME
        ),
        num_validation_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, training_args.NUM_VALIDATION_BATCHES_ARG_NAME
        ),
        plateau_patience_epochs=getattr(
            INPUT_ARG_OBJECT, training_args.PLATEAU_PATIENCE_ARG_NAME
        ),
        plateau_learning_rate_multiplier=getattr(
            INPUT_ARG_OBJECT, training_args.PLATEAU_MULTIPLIER_ARG_NAME
        ),
        early_stopping_patience_epochs=getattr(
            INPUT_ARG_OBJECT, training_args.EARLY_STOPPING_PATIENCE_ARG_NAME
        )
    )
