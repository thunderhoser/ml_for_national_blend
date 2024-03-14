"""Trains neural net."""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import neural_net
import training_args

TIME_FORMAT = '%Y-%m-%d-%H'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER = training_args.add_input_args(parser_object=INPUT_ARG_PARSER)


def _run(template_file_name, output_dir_name, nwp_lead_times_hours,
         nwp_model_names, nwp_field_names, nwp_normalization_file_name,
         target_lead_time_hours, target_field_names,
         target_normalization_file_name, num_examples_per_batch, sentinel_value,
         init_time_limit_strings_for_training, nwp_dir_names_for_training,
         target_dir_name_for_training, init_time_limit_strings_for_validation,
         nwp_dir_names_for_validation, target_dir_name_for_validation,
         num_epochs, num_training_batches_per_epoch,
         num_validation_batches_per_epoch,
         plateau_patience_epochs, plateau_learning_rate_multiplier,
         early_stopping_patience_epochs):
    """Trains neural net.

    This is effectively the main method.

    :param template_file_name: See documentation at top of this script.
    :param output_dir_name: Same.
    :param nwp_lead_times_hours: Same.
    :param nwp_model_names: Same.
    :param nwp_field_names: Same.
    :param nwp_normalization_file_name: Same.
    :param target_lead_time_hours: Same.
    :param target_field_names: Same.
    :param target_normalization_file_name: Same.
    :param num_examples_per_batch: Same.
    :param sentinel_value: Same.
    :param init_time_limit_strings_for_training: Same.
    :param nwp_dir_names_for_training: Same.
    :param target_dir_name_for_training: Same.
    :param init_time_limit_strings_for_validation: Same.
    :param nwp_dir_names_for_validation: Same.
    :param target_dir_name_for_validation: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param num_validation_batches_per_epoch: Same.
    :param plateau_patience_epochs: Same.
    :param plateau_learning_rate_multiplier: Same.
    :param early_stopping_patience_epochs: Same.
    """

    if nwp_normalization_file_name == '':
        nwp_normalization_file_name = None
    if target_normalization_file_name == '':
        target_normalization_file_name = None

    # assert len(nwp_model_names) == len(nwp_dir_names_for_training)
    # assert len(nwp_model_names) == len(nwp_dir_names_for_validation)

    nwp_dir_names_for_training = nwp_dir_names_for_training[:len(nwp_model_names)]
    nwp_dir_names_for_validation = nwp_dir_names_for_validation[:len(nwp_model_names)]

    nwp_model_to_training_dir_name = dict(
        zip(nwp_model_names, nwp_dir_names_for_training)
    )
    nwp_model_to_validation_dir_name = dict(
        zip(nwp_model_names, nwp_dir_names_for_validation)
    )

    nwp_model_to_field_names = dict()
    for this_model in nwp_model_names:
        nwp_model_to_field_names[this_model] = nwp_field_names

    init_time_limits_for_training_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, TIME_FORMAT)
        for t in init_time_limit_strings_for_training
    ], dtype=int)

    init_time_limits_for_validation_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, TIME_FORMAT)
        for t in init_time_limit_strings_for_validation
    ], dtype=int)

    training_option_dict = {
        neural_net.INIT_TIME_LIMITS_KEY: init_time_limits_for_training_unix_sec,
        neural_net.NWP_LEAD_TIMES_KEY: nwp_lead_times_hours,
        neural_net.NWP_MODEL_TO_DIR_KEY: nwp_model_to_training_dir_name,
        neural_net.NWP_MODEL_TO_FIELDS_KEY: nwp_model_to_field_names,
        neural_net.NWP_NORM_FILE_KEY: nwp_normalization_file_name,
        neural_net.TARGET_LEAD_TIME_KEY: target_lead_time_hours,
        neural_net.TARGET_FIELDS_KEY: target_field_names,
        neural_net.TARGET_DIR_KEY: target_dir_name_for_training,
        neural_net.TARGET_NORM_FILE_KEY: target_normalization_file_name,
        neural_net.BATCH_SIZE_KEY: num_examples_per_batch,
        neural_net.SENTINEL_VALUE_KEY: sentinel_value
    }

    validation_option_dict = {
        neural_net.INIT_TIME_LIMITS_KEY:
            init_time_limits_for_validation_unix_sec,
        neural_net.NWP_MODEL_TO_DIR_KEY: nwp_model_to_validation_dir_name,
        neural_net.TARGET_DIR_KEY: target_dir_name_for_validation
    }

    print('Reading model template from: "{0:s}"...'.format(template_file_name))
    model_object = neural_net.read_model(hdf5_file_name=template_file_name)

    model_metafile_name = neural_net.find_metafile(
        model_file_name=template_file_name, raise_error_if_missing=True
    )
    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)

    neural_net.train_model(
        model_object=model_object,
        num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        training_option_dict=training_option_dict,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_option_dict=validation_option_dict,
        loss_function_string=model_metadata_dict[neural_net.LOSS_FUNCTION_KEY],
        optimizer_function_string=
        model_metadata_dict[neural_net.OPTIMIZER_FUNCTION_KEY],
        metric_function_strings=
        model_metadata_dict[neural_net.METRIC_FUNCTIONS_KEY],
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
        nwp_field_names=getattr(
            INPUT_ARG_OBJECT, training_args.NWP_FIELDS_ARG_NAME
        ),
        nwp_normalization_file_name=getattr(
            INPUT_ARG_OBJECT, training_args.NWP_NORMALIZATION_FILE_ARG_NAME
        ),
        target_lead_time_hours=getattr(
            INPUT_ARG_OBJECT, training_args.TARGET_LEAD_TIME_ARG_NAME
        ),
        target_field_names=getattr(
            INPUT_ARG_OBJECT, training_args.TARGET_FIELDS_ARG_NAME
        ),
        target_normalization_file_name=getattr(
            INPUT_ARG_OBJECT, training_args.TARGET_NORMALIZATION_FILE_ARG_NAME
        ),
        num_examples_per_batch=getattr(
            INPUT_ARG_OBJECT, training_args.BATCH_SIZE_ARG_NAME
        ),
        sentinel_value=getattr(
            INPUT_ARG_OBJECT, training_args.SENTINEL_VALUE_ARG_NAME
        ),
        init_time_limit_strings_for_training=getattr(
            INPUT_ARG_OBJECT, training_args.TRAINING_TIME_LIMITS_ARG_NAME
        ),
        nwp_dir_names_for_training=getattr(
            INPUT_ARG_OBJECT, training_args.TRAINING_NWP_DIRS_ARG_NAME
        ),
        target_dir_name_for_training=getattr(
            INPUT_ARG_OBJECT, training_args.TRAINING_TARGET_DIR_ARG_NAME
        ),
        init_time_limit_strings_for_validation=getattr(
            INPUT_ARG_OBJECT, training_args.VALIDATION_TIME_LIMITS_ARG_NAME
        ),
        nwp_dir_names_for_validation=getattr(
            INPUT_ARG_OBJECT, training_args.VALIDATION_NWP_DIRS_ARG_NAME
        ),
        target_dir_name_for_validation=getattr(
            INPUT_ARG_OBJECT, training_args.VALIDATION_TARGET_DIR_ARG_NAME
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
