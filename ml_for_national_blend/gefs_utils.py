"""Helper methods for GEFS data."""

import os
import sys
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import number_rounding
import longitude_conversion as lng_conversion
import error_checking

# TODO(thunderhoser): Basically all the code in this module is duplicated from
# gfs_utils.py.  I will eventually generalize the code to work for both GFS and
# GEFS, rather than having two copies that are nearly identical.

TOLERANCE = 1e-6

LATITUDE_SPACING_DEG = 0.5
LONGITUDE_SPACING_DEG = 0.5

GRID_LATITUDES_DEG_N = numpy.linspace(-90, 90, num=361, dtype=float)
GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E = numpy.linspace(
    0, 359.5, num=720, dtype=float
)
GRID_LONGITUDES_NEGATIVE_IN_WEST_DEG_E = (
    lng_conversion.convert_lng_negative_in_west(
        GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E
    )
)


def desired_longitudes_to_columns(start_longitude_deg_e, end_longitude_deg_e):
    """Converts desired longitudes to desired grid columns.

    :param start_longitude_deg_e: Longitude at start of desired range.  This may
        be in either format (positive or negative values in western hemisphere).
    :param end_longitude_deg_e: Longitude at end of desired range.  This may
        be in either format.
    :return: desired_column_indices: 1-D numpy array with indices of desired
        columns.
    """

    start_longitude_deg_e = number_rounding.floor_to_nearest(
        start_longitude_deg_e, LONGITUDE_SPACING_DEG
    )
    end_longitude_deg_e = number_rounding.ceiling_to_nearest(
        end_longitude_deg_e, LONGITUDE_SPACING_DEG
    )
    error_checking.assert_is_greater(
        numpy.absolute(start_longitude_deg_e - end_longitude_deg_e),
        TOLERANCE
    )

    start_longitude_deg_e = lng_conversion.convert_lng_positive_in_west(
        start_longitude_deg_e, allow_nan=False
    )
    end_longitude_deg_e = lng_conversion.convert_lng_positive_in_west(
        end_longitude_deg_e, allow_nan=False
    )

    if end_longitude_deg_e > start_longitude_deg_e:
        are_longitudes_positive_in_west = True
    else:
        start_longitude_deg_e = lng_conversion.convert_lng_negative_in_west(
            start_longitude_deg_e, allow_nan=False
        )
        end_longitude_deg_e = lng_conversion.convert_lng_negative_in_west(
            end_longitude_deg_e, allow_nan=False
        )
        are_longitudes_positive_in_west = False

    if are_longitudes_positive_in_west:
        grid_longitudes_deg_e = GRID_LONGITUDES_POSITIVE_IN_WEST_DEG_E
    else:
        grid_longitudes_deg_e = GRID_LONGITUDES_NEGATIVE_IN_WEST_DEG_E

    num_longitudes = 1 + int(numpy.absolute(numpy.round(
        (end_longitude_deg_e - start_longitude_deg_e) / LONGITUDE_SPACING_DEG
    )))

    if end_longitude_deg_e > start_longitude_deg_e:
        desired_longitudes_deg_e = numpy.linspace(
            start_longitude_deg_e, end_longitude_deg_e, num=num_longitudes
        )

        desired_column_indices = numpy.array([
            numpy.where(
                numpy.absolute(grid_longitudes_deg_e - d) <= TOLERANCE
            )[0][0]
            for d in desired_longitudes_deg_e
        ], dtype=int)
    else:
        undesired_longitudes_deg_e = numpy.linspace(
            end_longitude_deg_e, start_longitude_deg_e, num=num_longitudes
        )[1:-1]

        undesired_column_indices = numpy.array([
            numpy.where(
                numpy.absolute(grid_longitudes_deg_e - d) <= TOLERANCE
            )[0][0]
            for d in undesired_longitudes_deg_e
        ], dtype=int)

        all_column_indices = numpy.linspace(
            0, len(grid_longitudes_deg_e) - 1, num=len(grid_longitudes_deg_e),
            dtype=int
        )
        desired_column_indices = (
            set(all_column_indices.tolist()) -
            set(undesired_column_indices.tolist())
        )
        desired_column_indices = numpy.array(
            list(desired_column_indices), dtype=int
        )

        break_index = 1 + numpy.where(
            numpy.diff(desired_column_indices) > 1
        )[0][0]

        desired_column_indices = numpy.concatenate((
            desired_column_indices[break_index:],
            desired_column_indices[:break_index]
        ))

    return desired_column_indices


def desired_latitudes_to_rows(start_latitude_deg_n, end_latitude_deg_n):
    """Converts desired latitudes to desired grid rows.

    :param start_latitude_deg_n: Latitude at start of desired range (deg north).
    :param end_latitude_deg_n: Latitude at end of desired range (deg north).
    :return: desired_row_indices: 1-D numpy array with indices of desired rows.
    """

    start_latitude_deg_n = number_rounding.floor_to_nearest(
        start_latitude_deg_n, LONGITUDE_SPACING_DEG
    )
    end_latitude_deg_n = number_rounding.ceiling_to_nearest(
        end_latitude_deg_n, LONGITUDE_SPACING_DEG
    )
    error_checking.assert_is_greater(
        numpy.absolute(start_latitude_deg_n - end_latitude_deg_n),
        TOLERANCE
    )

    error_checking.assert_is_valid_latitude(start_latitude_deg_n)
    error_checking.assert_is_valid_latitude(end_latitude_deg_n)

    num_latitudes = 1 + int(numpy.absolute(numpy.round(
        (end_latitude_deg_n - start_latitude_deg_n) / LATITUDE_SPACING_DEG
    )))
    desired_latitudes_deg_n = numpy.linspace(
        start_latitude_deg_n, end_latitude_deg_n, num=num_latitudes
    )
    desired_row_indices = numpy.array([
        numpy.where(numpy.absolute(GRID_LATITUDES_DEG_N - d) <= TOLERANCE)[0][0]
        for d in desired_latitudes_deg_n
    ], dtype=int)

    return desired_row_indices
