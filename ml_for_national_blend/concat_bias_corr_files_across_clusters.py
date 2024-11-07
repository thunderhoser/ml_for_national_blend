"""Concatenates bias-correction files across clusters.

Each input file should contain bias-correction models for a subset of clusters.
"""

import os
import sys
import copy
import glob
import argparse
import warnings
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import bias_clustering
import bias_correction

INPUT_FILE_PATTERN_ARG_NAME = 'input_model_file_pattern'
CLUSTER_FILE_ARG_NAME = 'input_cluster_file_name'
OUTPUT_FILE_ARG_NAME = 'output_model_file_name'

INPUT_FILE_PATTERN_HELP_STRING = (
    'Glob pattern for input files.  Each input file will be read by '
    '`bias_correction.read_file`.'
)
CLUSTER_FILE_HELP_STRING = (
    'Path to file with full cluster table.  Will be read by '
    '`bias_clustering.read_file`.'
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
    '--' + CLUSTER_FILE_ARG_NAME, type=str, required=True,
    help=CLUSTER_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_model_file_pattern, cluster_file_name, output_model_file_name):
    """Concatenates bias-correction files across clusters.

    This is effectively the main method.

    :param input_model_file_pattern: See documentation at top of this script.
    :param cluster_file_name: Same.
    :param output_model_file_name: Same.
    """

    input_model_file_names = glob.glob(input_model_file_pattern)
    assert len(input_model_file_names) > 0
    input_model_file_names.sort()

    print('Reading full cluster table from: "{0:s}"...'.format(
        cluster_file_name
    ))
    full_cluster_id_matrix = bias_clustering.read_file(cluster_file_name)[
        bias_clustering.CLUSTER_ID_KEY
    ].values

    cluster_id_matrix = numpy.array([], dtype=int)
    cluster_id_to_model_object = dict()
    target_field_name = None
    do_uncertainty_calibration = None
    do_iso_reg_before_uncertainty_calib = None

    for i in range(len(input_model_file_names)):
        print('Reading data from: "{0:s}"...'.format(input_model_file_names[i]))
        this_model_dict = bias_correction.read_file(input_model_file_names[i])

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

        this_cluster_id_matrix = this_model_dict[
            bias_correction.CLUSTER_IDS_KEY
        ]
        overlap_cluster_ids = numpy.unique(
            this_cluster_id_matrix[cluster_id_matrix > 0]
        )
        overlap_cluster_ids = overlap_cluster_ids[overlap_cluster_ids > 0]

        if len(overlap_cluster_ids) > 0:
            warning_string = (
                'POTENTIAL ERROR: Found {0:d} cluster IDs in more than one '
                'file:\n{1:s}'
            ).format(
                len(overlap_cluster_ids),
                str(overlap_cluster_ids)
            )

            warnings.warn(warning_string)

            this_cluster_id_matrix[
                numpy.isin(this_cluster_id_matrix, overlap_cluster_ids)
            ] = -1

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

    found_all_clusters = numpy.array_equal(
        cluster_id_matrix[cluster_id_matrix > 0],
        full_cluster_id_matrix[full_cluster_id_matrix > 0]
    )

    if not found_all_clusters:
        unique_cluster_ids_all = numpy.unique(
            full_cluster_id_matrix[full_cluster_id_matrix > 0]
        )
        unique_cluster_ids_found = numpy.unique(
            cluster_id_matrix[cluster_id_matrix > 0]
        )

        missing_cluster_ids = (
            set(unique_cluster_ids_all.tolist()) -
            set(unique_cluster_ids_found.tolist())
        )
        missing_cluster_ids = numpy.array(list(missing_cluster_ids), dtype=int)

        error_string = (
            'Could not find bias-correction models for {0:d} cluster IDs:'
        ).format(len(missing_cluster_ids))

        for this_id in missing_cluster_ids:
            error_string += '\n{0:d}'.format(this_id)

        raise ValueError(error_string)

    model_dict = {
        bias_correction.MODEL_KEY: None,
        bias_correction.CLUSTER_TO_MODEL_KEY: cluster_id_to_model_object,
        bias_correction.CLUSTER_IDS_KEY: cluster_id_matrix,
        bias_correction.FIELD_NAMES_KEY: [target_field_name],
        bias_correction.DO_UNCERTAINTY_CALIB_KEY: do_uncertainty_calibration,
        bias_correction.DO_IR_BEFORE_UC_KEY: do_iso_reg_before_uncertainty_calib
    }

    print('Writing all models to: "{0:s}"...'.format(output_model_file_name))
    bias_correction.write_file(
        model_dict=model_dict, dill_file_name=output_model_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_model_file_pattern=getattr(
            INPUT_ARG_OBJECT, INPUT_FILE_PATTERN_ARG_NAME
        ),
        cluster_file_name=getattr(INPUT_ARG_OBJECT, CLUSTER_FILE_ARG_NAME),
        output_model_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
