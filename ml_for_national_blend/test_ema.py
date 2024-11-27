"""Test script for EMA (exponential moving average) training method."""

import os
import sys
import argparse
import numpy
import keras
import pandas
import tensorflow

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import architecture_utils
import file_system_utils

OUTPUT_DIR_ARG_NAME = 'output_dir_name'
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  The model, checkpoints, etc. will all be saved '
    'here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


class EMAHelper:
    def __init__(self, model, optimizer, decay=0.99):
        self.model = model
        self.optimizer = optimizer
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
            optimizer=self.optimizer,
            ema_shadow_weights={str(i): sw for i, sw in enumerate(self.shadow_weights)}  # String keys
        )
        output_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}')

        print(f'Saving model and optimizer state to: "{output_path}"...')
        checkpoint_object.save(output_path)

    def restore_optimizer_state(self, checkpoint_dir, raise_error_if_missing):
        checkpoint_object = tensorflow.train.Checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            ema_shadow_weights={str(i): sw for i, sw in enumerate(self.shadow_weights)}  # String keys
        )

        print(f'Restoring optimizer state from: "{checkpoint_dir}"...')

        status = checkpoint_object.restore(
            tensorflow.train.latest_checkpoint(checkpoint_dir)
        )

        # Ensure optimizer variables are initialized
        dummy_input = tensorflow.zeros((1,) + self.model.input_shape[1:])
        _ = self.model(dummy_input, training=False)  # Forward pass to initialize optimizer variables

        if raise_error_if_missing:
            status.assert_consumed()  # Ensure everything was restored

        status.expect_partial()  # Suppress warnings if partial restore is OK

        # Ensure shadow weights are reassigned correctly after restore
        for i, sw in enumerate(self.shadow_weights):
            sw.assign(checkpoint_object.ema_shadow_weights[str(i)])  # Access with string keys


def _run(output_dir_name):
    """Test script for EMA (exponential moving average) training method.

    This is effectively the main method.

    :param output_dir_name: See documentation at top of this script.
    """

    # Create training and validation data.
    predictor_matrix = numpy.random.uniform(low=0., high=1., size=(20000, 2))
    target_values = numpy.logical_xor(
        predictor_matrix[:, 0] >= 0.5,
        predictor_matrix[:, 1] >= 0.5
    ).astype(int)

    training_indices = numpy.linspace(0, 9999, num=10000, dtype=int)
    validation_indices = training_indices + 10000

    # Create model architecture.
    num_predictors = predictor_matrix.shape[1]

    input_layer_object = keras.layers.Input(shape=(num_predictors,))
    layer_object = architecture_utils.get_dense_layer(num_output_units=10)(
        input_layer_object
    )
    layer_object = architecture_utils.get_activation_layer(
        activation_function_string=architecture_utils.RELU_FUNCTION_STRING,
        alpha_for_relu=0.2
    )(layer_object)

    layer_object = architecture_utils.get_dense_layer(num_output_units=100)(
        layer_object
    )
    layer_object = architecture_utils.get_activation_layer(
        activation_function_string=architecture_utils.RELU_FUNCTION_STRING,
        alpha_for_relu=0.2
    )(layer_object)

    layer_object = architecture_utils.get_dense_layer(num_output_units=1)(
        layer_object
    )
    layer_object = architecture_utils.get_activation_layer(
        activation_function_string=architecture_utils.SIGMOID_FUNCTION_STRING
    )(layer_object)

    model_object = keras.models.Model(
        inputs=input_layer_object, outputs=layer_object
    )
    model_object.compile(
        loss='binary_crossentropy', optimizer=keras.optimizers.AdamW(),
        metrics=[]
    )
    model_object.summary()

    # Create checkpoints.
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    backup_dir_name = '{0:s}/backup_and_restore'.format(output_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=backup_dir_name
    )

    model_file_name = '{0:s}/model.weights.h5'.format(output_dir_name)
    history_file_name = '{0:s}/history.csv'.format(output_dir_name)

    try:
        history_table_pandas = pandas.read_csv(history_file_name)
        initial_epoch = history_table_pandas['epoch'].max() + 1
        best_validation_loss = history_table_pandas['val_loss'].min()
    except:
        initial_epoch = 0
        best_validation_loss = numpy.inf

    history_object = keras.callbacks.CSVLogger(
        filename=history_file_name, separator=',', append=True
    )
    checkpoint_object = keras.callbacks.ModelCheckpoint(
        filepath=model_file_name, monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=True, mode='min',
        save_freq='epoch'
    )
    checkpoint_object.best = best_validation_loss

    early_stopping_object = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0.,
        patience=100, verbose=1, mode='min'
    )
    plateau_object = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.95,
        patience=10, verbose=1, mode='min',
        min_delta=0., cooldown=0
    )
    backup_object = keras.callbacks.BackupAndRestore(
        backup_dir_name, save_freq='epoch', delete_checkpoint=False
    )

    list_of_callback_objects = [
        history_object, checkpoint_object,
        early_stopping_object, plateau_object,
        backup_object
    ]

    ema_object = EMAHelper(
        model=model_object,
        optimizer=model_object.optimizer,
        decay=0.9
    )

    ema_backup_dir_name = '{0:s}/exponential_moving_average'.format(
        output_dir_name
    )
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=ema_backup_dir_name
    )

    # TODO(thunderhoser): I don't know what happens here if the directory
    # doesn't exist.
    ema_object.restore_optimizer_state(
        checkpoint_dir=ema_backup_dir_name,
        raise_error_if_missing=initial_epoch > 0
    )

    for this_epoch in range(initial_epoch, 1000):
        model_object.fit(
            x=predictor_matrix[training_indices, :],
            y=target_values[training_indices],
            batch_size=100,
            steps_per_epoch=100,
            epochs=this_epoch + 1,
            initial_epoch=this_epoch,
            verbose=1,
            callbacks=list_of_callback_objects,
            validation_data=(
                predictor_matrix[validation_indices, :],
                target_values[validation_indices]
            ),
            validation_steps=100
        )

        ema_object.apply_ema()
        ema_object.save_optimizer_state(
            checkpoint_dir=ema_backup_dir_name, epoch=this_epoch
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
