"""Input/output methods for raw NBM constants.

NBM = National Blend of Models

Each raw file should be a GRIB2 file with one constant variable on the NBM grid.
"""

import numpy
import xarray
from ml_for_national_blend.outside_code import grib_io
from ml_for_national_blend.outside_code import error_checking
from ml_for_national_blend.utils import nbm_utils
from ml_for_national_blend.utils import nbm_constant_utils

FIELD_NAME_TO_GRIB_NAME = {
    nbm_constant_utils.LAND_SEA_MASK_NAME: 'LAND:surface',
    nbm_constant_utils.OROGRAPHIC_HEIGHT_NAME: 'DIST:surface'
}


def read_file(grib2_file_name, wgrib2_exe_name, temporary_dir_name, field_name):
    """Reads NBM constant field from GRIB2 file into xarray table.

    :param grib2_file_name: Path to input file.
    :param wgrib2_exe_name: Path to wgrib2 executable.
    :param temporary_dir_name: Path to temporary directory for text files
        created by wgrib2.
    :param field_name: Name of field to read.  Must be accepted by
        `nbm_constant_utils.check_field_name`.
    :return: nbm_constant_table_xarray: xarray table with all data.  Metadata
        and variable names should make this table self-explanatory.
    """

    # Check input args.
    error_checking.assert_file_exists(grib2_file_name)
    nbm_constant_utils.check_field_name(field_name)

    # Do actual stuff.
    latitude_matrix_deg_n, longitude_matrix_deg_e = nbm_utils.read_coords()
    num_grid_rows = latitude_matrix_deg_n.shape[0]
    num_grid_columns = latitude_matrix_deg_n.shape[1]

    grib_search_string = FIELD_NAME_TO_GRIB_NAME[field_name]

    print('Reading line "{0:s}" from GRIB2 file: "{1:s}"...'.format(
        grib_search_string, grib2_file_name
    ))
    data_matrix = grib_io.read_field_from_grib_file(
        grib_file_name=grib2_file_name,
        field_name_grib1=grib_search_string,
        num_grid_rows=num_grid_rows,
        num_grid_columns=num_grid_columns,
        wgrib_exe_name=wgrib2_exe_name,
        wgrib2_exe_name=wgrib2_exe_name,
        temporary_dir_name=temporary_dir_name,
        sentinel_value=None,
        raise_error_if_fails=True
    )[0]

    # orig_dimensions = data_matrix.shape
    # data_matrix = numpy.reshape(
    #     numpy.ravel(data_matrix), orig_dimensions, order='F'
    # )
    assert not numpy.any(numpy.isnan(data_matrix))

    coord_dict = {
        nbm_constant_utils.ROW_DIM: numpy.linspace(
            0, num_grid_rows - 1, num=num_grid_rows, dtype=int
        ),
        nbm_constant_utils.COLUMN_DIM: numpy.linspace(
            0, num_grid_columns - 1, num=num_grid_columns, dtype=int
        ),
        nbm_constant_utils.FIELD_DIM: [field_name]
    }

    these_dim = (
        nbm_constant_utils.ROW_DIM, nbm_constant_utils.COLUMN_DIM,
        nbm_constant_utils.FIELD_DIM
    )
    main_data_dict = {
        nbm_constant_utils.DATA_KEY: (
            these_dim, numpy.expand_dims(data_matrix, axis=-1)
        )
    }

    these_dim = (nbm_constant_utils.ROW_DIM, nbm_constant_utils.COLUMN_DIM)
    main_data_dict.update({
        nbm_constant_utils.LATITUDE_KEY: (these_dim, latitude_matrix_deg_n),
        nbm_constant_utils.LONGITUDE_KEY: (these_dim, longitude_matrix_deg_e)
    })

    return xarray.Dataset(data_vars=main_data_dict, coords=coord_dict)
