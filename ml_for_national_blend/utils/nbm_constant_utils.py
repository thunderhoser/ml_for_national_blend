"""Helper methods for NBM constant fields."""

from ml_for_national_blend.outside_code import error_checking

ROW_DIM = 'row'
COLUMN_DIM = 'column'
FIELD_DIM = 'field_name'
QUANTILE_LEVEL_DIM = 'quantile_level'

LATITUDE_KEY = 'latitude_deg_n'
LONGITUDE_KEY = 'longitude_deg_e'
DATA_KEY = 'data_matrix'

MEAN_VALUE_KEY = 'mean_value'
MEAN_SQUARED_VALUE_KEY = 'mean_squared_value'
STDEV_KEY = 'standard_deviation'
QUANTILE_KEY = 'quantile'

LAND_SEA_MASK_NAME = 'land_sea_mask_land1'
OROGRAPHIC_HEIGHT_NAME = 'orographic_height_m_asl'
LATITUDE_NAME = 'latitude_matrix_deg_n'
LONGITUDE_NAME = 'longitude_matrix_deg_e'
ALL_FIELD_NAMES = [
    LAND_SEA_MASK_NAME, OROGRAPHIC_HEIGHT_NAME, LATITUDE_NAME, LONGITUDE_NAME
]


def check_field_name(field_name):
    """Ensures that field name is valid.

    :param field_name: String (must be in list `ALL_FIELD_NAMES`).
    :raises: ValueError: if `field_name not in ALL_FIELD_NAMES`.
    """

    error_checking.assert_is_string(field_name)
    if field_name in ALL_FIELD_NAMES:
        return

    error_string = (
        'Field name "{0:s}" is not in the list of accepted field names '
        '(below):\n{1:s}'
    ).format(
        field_name, str(ALL_FIELD_NAMES)
    )

    raise ValueError(error_string)
