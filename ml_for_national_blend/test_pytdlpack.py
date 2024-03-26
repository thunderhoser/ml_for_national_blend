"""Tests the pytdlpack library.

USE ONCE AND DESTROY.
"""

from datetime import datetime
import numpy
import pytdlpack

SENTINEL_VALUE = 9999.

INPUT_FILE_NAME = (
    '/scratch1/NCEPDEV/mdl/Geoff.Wagner/firewx_test_data/gmos/tdlpack/'
    'gfs00gmos_co.trimmed.202211'
)


def _run():
    """Tests the pytdlpack library.

    This is effectively the main method.
    """

    print('Reading data from: "{0:s}"...'.format(INPUT_FILE_NAME))
    file_object = pytdlpack.open(INPUT_FILE_NAME, 'r')
    print(file_object)

    variable_id_word1 = 222030008
    variable_id_word2 = 000000000
    # variable_id_word3 = 000000006
    variable_id_word3 = 6

    file_object.rewind()
    start_time_object = datetime.now()
    print(start_time_object)

    desired_record_object = None

    while not file_object.eof:
        this_record_object = file_object.read(unpack=True)

        try:
            _ = this_record_object.id
            _ = this_record_object.plain
        except:
            pass
        else:
            if (
                    this_record_object.id[0] == variable_id_word1 and
                    this_record_object.id[1] == variable_id_word2 and
                    this_record_object.id[2] == variable_id_word3
            ):
                print(this_record_object.is1[7])

                desired_record_object = this_record_object
                desired_record_object.unpack(data=True)
                continue

    if desired_record_object is None:
        error_string = (
            'Cannot find desired record (words = {0:09d}, {1:09d}, {2:09d}) '
            'in file: "{3:s}"'
        ).format(
            variable_id_word1,
            variable_id_word2,
            variable_id_word3,
            INPUT_FILE_NAME
        )

        raise ValueError(error_string)

    data_matrix = desired_record_object.data
    data_matrix[data_matrix >= SENTINEL_VALUE - 1] = numpy.nan

    print(data_matrix.shape)
    print(numpy.nanmin(data_matrix))
    print(numpy.nanmax(data_matrix))


if __name__ == '__main__':
    _run()
