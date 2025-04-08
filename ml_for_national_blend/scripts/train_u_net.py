"""Trains a simple U-net."""

import json
import argparse
import numpy
from ml_for_national_blend.outside_code import time_conversion
from ml_for_national_blend.machine_learning import neural_net
from ml_for_national_blend.scripts import training_args_u_net as training_args

TIME_FORMAT = '%Y-%m-%d-%H'
NONE_STRINGS = ['', 'none', 'None']

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER = training_args.add_input_args(parser_object=INPUT_ARG_PARSER)


def _run(template_file_name, output_dir_name, nwp_lead_time_hours,
         nwp_model_names, nwp_model_to_field_names, nwp_normalization_file_name,
         backup_nwp_model_name, backup_nwp_dir_name,
         target_lead_time_hours, target_field_names,
         compare_to_baseline_in_loss, num_examples_per_batch, sentinel_value,
         patch_size_2pt5km_pixels, patch_buffer_size_2pt5km_pixels,
         patch_start_row_2pt5km, patch_start_column_2pt5km,
         patch_overlap_size_2pt5km_pixels, resid_baseline_model_name,
         resid_baseline_lead_time_hours, resid_baseline_model_dir_name,
         nwp_directory_names, target_dir_name,
         training_init_time_limit_strings, validation_init_time_limit_strings,
         num_epochs, num_training_batches_per_epoch,
         num_validation_batches_per_epoch,
         plateau_patience_epochs, plateau_learning_rate_multiplier,
         early_stopping_patience_epochs):
    """Trains a simple U-net.

    This is effectively the main method.

    :param template_file_name: See documentation at top of this script.
    :param output_dir_name: Same.
    :param nwp_lead_time_hours: Same.
    :param nwp_model_names: Same.
    :param nwp_model_to_field_names: Same.
    :param nwp_normalization_file_name: Same.
    :param backup_nwp_model_name: Same.
    :param backup_nwp_dir_name: Same.
    :param target_lead_time_hours: Same.
    :param target_field_names: Same.
    :param compare_to_baseline_in_loss: Same.
    :param num_examples_per_batch: Same.
    :param sentinel_value: Same.
    :param patch_size_2pt5km_pixels: Same.
    :param patch_buffer_size_2pt5km_pixels: Same.
    :param patch_start_row_2pt5km: Same.
    :param patch_start_column_2pt5km: Same.
    :param patch_overlap_size_2pt5km_pixels: Same.
    :param resid_baseline_model_name: Same.
    :param resid_baseline_lead_time_hours: Same.
    :param resid_baseline_model_dir_name: Same.
    :param nwp_directory_names: Same.
    :param target_dir_name: Same.
    :param training_init_time_limit_strings: Same.
    :param validation_init_time_limit_strings: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param num_validation_batches_per_epoch: Same.
    :param plateau_patience_epochs: Same.
    :param plateau_learning_rate_multiplier: Same.
    :param early_stopping_patience_epochs: Same.
    """

    if resid_baseline_model_name in NONE_STRINGS:
        resid_baseline_model_name = None
    if resid_baseline_model_dir_name in NONE_STRINGS:
        resid_baseline_model_dir_name = None
    if resid_baseline_lead_time_hours <= 0:
        resid_baseline_lead_time_hours = None
    if patch_start_row_2pt5km < 0:
        patch_start_row_2pt5km = None
    if patch_start_column_2pt5km < 0:
        patch_start_column_2pt5km = None

    nwp_directory_names = nwp_directory_names[:len(nwp_model_names)]
    nwp_model_to_dir_name = dict(zip(nwp_model_names, nwp_directory_names))

    training_init_time_limits_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, TIME_FORMAT)
        for t in training_init_time_limit_strings
    ], dtype=int)

    validation_init_time_limits_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, TIME_FORMAT)
        for t in validation_init_time_limit_strings
    ], dtype=int)

    training_option_dict = {
        'first_init_time_unix_sec': training_init_time_limits_unix_sec[0],
        'last_init_time_unix_sec': training_init_time_limits_unix_sec[1],
        'nwp_lead_time_hours': nwp_lead_time_hours,
        neural_net.NWP_MODEL_TO_DIR_KEY: nwp_model_to_dir_name,
        neural_net.NWP_MODEL_TO_FIELDS_KEY: nwp_model_to_field_names,
        neural_net.NWP_NORM_FILE_KEY: nwp_normalization_file_name,
        neural_net.BACKUP_NWP_MODEL_KEY: backup_nwp_model_name,
        neural_net.BACKUP_NWP_DIR_KEY: backup_nwp_dir_name,
        neural_net.TARGET_LEAD_TIME_KEY: target_lead_time_hours,
        neural_net.TARGET_FIELDS_KEY: target_field_names,
        neural_net.TARGET_DIR_KEY: target_dir_name,
        neural_net.COMPARE_TO_BASELINE_IN_LOSS_KEY: compare_to_baseline_in_loss,
        neural_net.BATCH_SIZE_KEY: num_examples_per_batch,
        neural_net.SENTINEL_VALUE_KEY: sentinel_value,
        neural_net.PATCH_SIZE_KEY: patch_size_2pt5km_pixels,
        neural_net.PATCH_BUFFER_SIZE_KEY: patch_buffer_size_2pt5km_pixels,
        neural_net.PATCH_START_ROW_KEY: patch_start_row_2pt5km,
        neural_net.PATCH_START_COLUMN_KEY: patch_start_column_2pt5km,
        neural_net.RESID_BASELINE_MODEL_KEY: resid_baseline_model_name,
        neural_net.RESID_BASELINE_LEAD_TIME_KEY: resid_baseline_lead_time_hours,
        neural_net.RESID_BASELINE_MODEL_DIR_KEY: resid_baseline_model_dir_name
    }

    validation_option_dict = {
        'first_init_time_unix_sec': validation_init_time_limits_unix_sec[0],
        'last_init_time_unix_sec': validation_init_time_limits_unix_sec[1]
    }

    print('Reading model template from: "{0:s}"...'.format(template_file_name))
    model_object = neural_net.read_model(
        hdf5_file_name=template_file_name, for_inference=False
    )

    model_metafile_name = neural_net.find_metafile(
        model_file_name=template_file_name, raise_error_if_missing=True
    )
    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    mmd = model_metadata_dict

    neural_net.train_u_net(
        model_object=model_object,
        num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        training_option_dict=training_option_dict,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_option_dict=validation_option_dict,
        loss_function_string=mmd[neural_net.LOSS_FUNCTION_KEY],
        optimizer_function_string=mmd[neural_net.OPTIMIZER_FUNCTION_KEY],
        metric_function_strings=mmd[neural_net.METRIC_FUNCTIONS_KEY],
        u_net_architecture_dict=mmd[neural_net.U_NET_ARCHITECTURE_KEY],
        plateau_patience_epochs=plateau_patience_epochs,
        plateau_learning_rate_multiplier=plateau_learning_rate_multiplier,
        early_stopping_patience_epochs=early_stopping_patience_epochs,
        patch_overlap_size_2pt5km_pixels=patch_overlap_size_2pt5km_pixels,
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
        nwp_lead_time_hours=getattr(
            INPUT_ARG_OBJECT, training_args.NWP_LEAD_TIME_ARG_NAME
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
        patch_overlap_size_2pt5km_pixels=getattr(
            INPUT_ARG_OBJECT, training_args.PATCH_OVERLAP_SIZE_ARG_NAME
        ),
        resid_baseline_model_name=getattr(
            INPUT_ARG_OBJECT, training_args.RESID_BASELINE_MODEL_ARG_NAME
        ),
        resid_baseline_lead_time_hours=getattr(
            INPUT_ARG_OBJECT, training_args.RESID_BASELINE_LEAD_TIME_ARG_NAME
        ),
        resid_baseline_model_dir_name=getattr(
            INPUT_ARG_OBJECT, training_args.RESID_BASELINE_MODEL_DIR_ARG_NAME
        ),
        nwp_directory_names=getattr(
            INPUT_ARG_OBJECT, training_args.NWP_DIRS_ARG_NAME
        ),
        target_dir_name=getattr(
            INPUT_ARG_OBJECT, training_args.TARGET_DIR_ARG_NAME
        ),
        training_init_time_limit_strings=getattr(
            INPUT_ARG_OBJECT, training_args.TRAINING_TIME_LIMITS_ARG_NAME
        ),
        validation_init_time_limit_strings=getattr(
            INPUT_ARG_OBJECT, training_args.VALIDATION_TIME_LIMITS_ARG_NAME
        ),
        num_epochs=getattr(INPUT_ARG_OBJECT, training_args.NUM_EPOCHS_ARG_NAME),
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
