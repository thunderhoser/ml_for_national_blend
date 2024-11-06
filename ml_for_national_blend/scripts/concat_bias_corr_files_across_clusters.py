"""Concatenates bias-correction files across clusters.

Each input file should contain bias-correction models for a subset of clusters.
"""

import copy
import glob
import argparse
import numpy
from ml_for_national_blend.machine_learning import bias_correction

INPUT_FILE_PATTERN_ARG_NAME = 'input_file_pattern'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILE_PATTERN_HELP_STRING = (
    'Glob pattern for input files.  Each input file will be read by '
    '`bias_correction.read_file`.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Will be written in Dill format by '
    '`bias_correction.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_PATTERN_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_PATTERN_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_file_pattern, output_file_name):
    """Concatenates bias-correction files across clusters.

    This is effectively the main method.

    :param input_file_pattern: See documentation at top of this script.
    :param output_file_name: Same.
    """

    input_file_names = glob.glob(input_file_pattern)
    assert len(input_file_names) > 0
    input_file_names.sort()

    cluster_id_matrix = numpy.array([], dtype=int)
    cluster_id_to_model_object = dict()
    target_field_name = None
    do_uncertainty_calibration = None
    do_iso_reg_before_uncertainty_calib = None

    for i in range(len(input_file_names)):
        print('Reading data from: "{0:s}"...'.format(input_file_names[i]))
        this_model_dict = bias_correction.read_file(input_file_names[i])

        assert this_model_dict[bias_correction.MODEL_KEY] is None

        if i == 0:
            assert len(this_model_dict[bias_correction.FIELD_NAMES_KEY]) == 1
            target_field_name = this_model_dict[
                bias_correction.FIELD_NAMES_KEY
            ][0]
            do_uncertainty_calibration = this_model_dict[
                bias_correction.DO_UNCERTAINTY_CALIB_KEY
            ]
            do_iso_reg_before_uncertainty_calib = this_model_dict[
                bias_correction.DO_IR_BEFORE_UC_KEY
            ]

        assert [target_field_name] == this_model_dict[
            bias_correction.FIELD_NAMES_KEY
        ]
        assert do_uncertainty_calibration == this_model_dict[
            bias_correction.DO_UNCERTAINTY_CALIB_KEY
        ]
        assert do_iso_reg_before_uncertainty_calib == this_model_dict[
            bias_correction.DO_IR_BEFORE_UC_KEY
        ]

        if i == 0:
            cluster_id_matrix = (
                this_model_dict[bias_correction.CLUSTER_IDS_KEY] + 0
            )
            cluster_id_to_model_object = copy.deepcopy(
                this_model_dict[bias_correction.CLUSTER_TO_MODEL_KEY]
            )
            continue

        this_cluster_id_matrix = this_model_dict[bias_correction.CLUSTER_IDS_KEY]
        assert not numpy.any(numpy.logical_and(
            cluster_id_matrix > 0, this_cluster_id_matrix > 0
        ))

        cluster_id_matrix[this_cluster_id_matrix > 0] = this_cluster_id_matrix[
            this_cluster_id_matrix > 0
        ]

        this_cluster_id_to_model_object = this_model_dict[
            bias_correction.CLUSTER_TO_MODEL_KEY
        ]
        common_keys = (
            set(cluster_id_to_model_object.keys()) &
            set(this_cluster_id_to_model_object.keys())
        )
        assert len(common_keys) == 0

        cluster_id_to_model_object.update(this_cluster_id_to_model_object)

    model_dict = {
        bias_correction.MODEL_KEY: None,
        bias_correction.CLUSTER_TO_MODEL_KEY: cluster_id_to_model_object,
        bias_correction.CLUSTER_IDS_KEY: cluster_id_matrix,
        bias_correction.FIELD_NAMES_KEY: [target_field_name],
        bias_correction.DO_UNCERTAINTY_CALIB_KEY: do_uncertainty_calibration,
        bias_correction.DO_IR_BEFORE_UC_KEY: do_iso_reg_before_uncertainty_calib
    }

    print('Writing all models to: "{0:s}"...'.format(output_file_name))
    bias_correction.write_file(
        model_dict=model_dict, dill_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_pattern=getattr(
            INPUT_ARG_OBJECT, INPUT_FILE_PATTERN_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
