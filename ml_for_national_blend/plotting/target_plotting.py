"""Methods for plotting target variables."""

import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from scipy.interpolate import RegularGridInterpolator
from ml_for_national_blend.outside_code import grids
from ml_for_national_blend.outside_code import longitude_conversion as lng_conversion
from ml_for_national_blend.outside_code import error_checking
from ml_for_national_blend.outside_code import gg_plotting_utils
from ml_for_national_blend.utils import urma_utils

# TODO(thunderhoser): I might want code in this module to convert temperature
# and dewpoint to plotting units (from K to deg C).  But for now, all the
# prediction files have deg C anyways.

TOLERANCE = 1e-6
NAN_COLOUR = numpy.full(3, 152. / 255)

FIELD_NAME_TO_FANCY = {
    urma_utils.TEMPERATURE_2METRE_NAME: r'2-m temperature ($^{\circ}$C)',
    urma_utils.DEWPOINT_2METRE_NAME: r'2-m dewpoint ($^{\circ}$C)',
    urma_utils.U_WIND_10METRE_NAME: r'10-m zonal wind (m s$^{-1}$)',
    urma_utils.V_WIND_10METRE_NAME: r'10-m meridional wind (m s$^{-1}$)',
    urma_utils.WIND_GUST_10METRE_NAME: r'10-m wind gust (m s$^{-1}$)'
}


def _grid_points_to_edges_1d(grid_point_coords):
    """Converts grid points (i.e., cell centers) to cell edges for 1-D grid.

    P = number of grid points

    :param grid_point_coords: length-P numpy array of grid-point coordinates, in
        increasing order.
    :return: grid_cell_edge_coords: length-(P + 1) numpy array of grid-cell-edge
        coordinates, also in increasing order.
    """

    # TODO(thunderhoser): Maybe make this method public.

    grid_cell_edge_coords = (grid_point_coords[:-1] + grid_point_coords[1:]) / 2
    first_edge_coords = (
        grid_point_coords[0] - numpy.diff(grid_point_coords[:2]) / 2
    )
    last_edge_coords = (
        grid_point_coords[-1] + numpy.diff(grid_point_coords[-2:]) / 2
    )

    return numpy.concatenate((
        first_edge_coords, grid_cell_edge_coords, last_edge_coords
    ))


def _grid_points_to_edges_2d(grid_point_coord_matrix):
    """Converts grid points (i.e., cell centers) to cell edges for 2-D grid.

    M = number of rows
    N = number of columns

    :param grid_point_coord_matrix: M-by-N numpy array of grid-point
        coordinates.
    :return: grid_cell_edge_coord_matrix: (M + 1)-by-(N + 1) numpy array of
        grid-cell-edge coordinates.
    """

    # TODO(thunderhoser): Make this method public.

    num_rows = grid_point_coord_matrix.shape[0]
    num_columns = grid_point_coord_matrix.shape[1]

    row_indices_orig = numpy.linspace(
        0, num_rows - 1, num=num_rows, dtype=float
    )
    column_indices_orig = numpy.linspace(
        0, num_columns - 1, num=num_columns, dtype=float
    )

    row_indices_new = _grid_points_to_edges_1d(row_indices_orig)
    column_indices_new = _grid_points_to_edges_1d(column_indices_orig)

    interp_object = RegularGridInterpolator(
        points=(row_indices_orig, column_indices_orig),
        values=grid_point_coord_matrix,
        method='linear', bounds_error=False, fill_value=None
    )

    column_index_matrix_new, row_index_matrix_new = numpy.meshgrid(
        column_indices_new, row_indices_new
    )
    rowcol_index_matrix_nw = numpy.transpose(numpy.vstack((
        numpy.ravel(row_index_matrix_new),
        numpy.ravel(column_index_matrix_new)
    )))

    grid_cell_edge_coords = interp_object(rowcol_index_matrix_nw)

    return numpy.reshape(
        grid_cell_edge_coords,
        (len(row_indices_new), len(column_indices_new))
    )


def field_to_colour_scheme(field_name, min_value, max_value):
    """Returns colour scheme for one target field.

    :param field_name: Field name.  Must be accepted by
        `urma_utils.check_field_name`.
    :param min_value: Minimum value in colour scheme.
    :param max_value: Max value in colour scheme.
    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    """

    urma_utils.check_field_name(field_name)

    if field_name == urma_utils.WIND_GUST_10METRE_NAME:
        min_value = max([min_value, 0.])

    if field_name in [
            urma_utils.TEMPERATURE_2METRE_NAME,
            urma_utils.DEWPOINT_2METRE_NAME,
            urma_utils.WIND_GUST_10METRE_NAME
    ]:
        colour_map_object = pyplot.get_cmap('viridis')
        colour_map_object.set_bad(NAN_COLOUR)

        max_value = max([max_value, min_value + TOLERANCE])
        colour_norm_object = pyplot.Normalize(vmin=min_value, vmax=max_value)

        return colour_map_object, colour_norm_object

    max_absolute_value = max([
        numpy.absolute(min_value),
        numpy.absolute(max_value)
    ])
    max_absolute_value = max([max_absolute_value, TOLERANCE])

    colour_map_object = pyplot.get_cmap('seismic')
    colour_map_object.set_bad(NAN_COLOUR)
    colour_norm_object = pyplot.Normalize(
        vmin=-1 * max_absolute_value, vmax=max_absolute_value
    )

    return colour_map_object, colour_norm_object


def plot_field(data_matrix, latitude_matrix_deg_n, longitude_matrix_deg_e,
               colour_map_object, colour_norm_object, axes_object,
               plot_colour_bar):
    """Plots one field on a lat/long grid.

    M = number of rows in grid
    N = number of columns in grid

    :param data_matrix: M-by-N numpy array of data values.
    :param latitude_matrix_deg_n: M-by-N numpy array of latitudes (deg north).
    :param longitude_matrix_deg_e: M-by-N numpy array of longitudes (deg east).
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.colors.ListedColormap` or similar).
    :param colour_norm_object: Colour-normalizer, used to map from physical
        values to colours (instance of `matplotlib.colors.BoundaryNorm` or
        similar).
    :param axes_object: Will plot on this set of axes (instance of
        `matplotlib.axes._subplots.AxesSubplot` or similar).
    :param plot_colour_bar: Boolean flag.
    :return: colour_bar_object: If `plot_colour_bar == True`, this is the
        handle for the colour bar.  Otherwise, this is None.
    """

    # Check input args.
    error_checking.assert_is_numpy_array(data_matrix, num_dimensions=2)

    error_checking.assert_is_numpy_array(
        latitude_matrix_deg_n,
        exact_dimensions=numpy.array(data_matrix.shape, dtype=int)
    )
    error_checking.assert_is_valid_lat_numpy_array(
        latitude_matrix_deg_n, allow_nan=False
    )

    error_checking.assert_is_numpy_array(
        longitude_matrix_deg_e,
        exact_dimensions=numpy.array(data_matrix.shape, dtype=int)
    )
    longitude_matrix_deg_e = lng_conversion.convert_lng_positive_in_west(
        longitude_matrix_deg_e, allow_nan=False
    )

    error_checking.assert_is_boolean(plot_colour_bar)

    # Do actual stuff.
    edge_latitude_matrix_deg_n = _grid_points_to_edges_2d(
        latitude_matrix_deg_n
    )
    edge_longitude_matrix_deg_e = _grid_points_to_edges_2d(
        longitude_matrix_deg_e
    )
    edge_longitude_matrix_deg_e = lng_conversion.convert_lng_positive_in_west(
        edge_longitude_matrix_deg_e
    )
    data_matrix_to_plot = grids.latlng_field_grid_points_to_edges(
        field_matrix=data_matrix,
        min_latitude_deg=1., min_longitude_deg=1.,
        lat_spacing_deg=1e-6, lng_spacing_deg=1e-6
    )[0]

    data_matrix_to_plot = numpy.ma.masked_where(
        numpy.isnan(data_matrix_to_plot), data_matrix_to_plot
    )

    axes_object.pcolor(
        edge_longitude_matrix_deg_e, edge_latitude_matrix_deg_n,
        data_matrix_to_plot,
        cmap=colour_map_object, norm=colour_norm_object,
        edgecolors='None', zorder=-1e11
    )

    if plot_colour_bar:
        colour_bar_object = gg_plotting_utils.plot_colour_bar(
            axes_object_or_matrix=axes_object,
            data_matrix=data_matrix,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            orientation_string='vertical',
            extend_min=True, extend_max=True
        )
    else:
        colour_bar_object = None

    return colour_bar_object
