"""Scratch space."""

# import numpy
# import xarray
#
# coord_matrix = numpy.loadtxt('/home/ralager/Downloads/wrf_arw_coords.txt', delimiter=',')
# print(coord_matrix[:5, :])
#
# row_indices = numpy.round(coord_matrix[:, 0]).astype(int)
# column_indices = numpy.round(coord_matrix[:, 1]).astype(int)
# latitudes_deg_n = coord_matrix[:, 2]
# longitudes_deg_e = coord_matrix[:, 3]
#
# num_rows = numpy.max(row_indices)
# num_columns = numpy.max(column_indices)
#
# # row_index_matrix = numpy.reshape(row_indices, (num_rows, num_columns), order='F')
# latitude_matrix_deg_n = numpy.reshape(latitudes_deg_n, (num_rows, num_columns), order='F')
# longitude_matrix_deg_n = numpy.reshape(longitudes_deg_e, (num_rows, num_columns), order='F')
#
# ROW_DIM = 'row'
# COLUMN_DIM = 'column'
#
# LATITUDE_KEY = 'latitude_deg_n'
# LONGITUDE_KEY = 'longitude_deg_e'
#
# data_dict = {
#     LATITUDE_KEY: ((ROW_DIM, COLUMN_DIM), latitude_matrix_deg_n),
#     LONGITUDE_KEY: ((ROW_DIM, COLUMN_DIM), longitude_matrix_deg_n)
# }
#
# coord_table_xarray = xarray.Dataset(data_vars=data_dict)
# print(coord_table_xarray)
#
# coord_table_xarray.to_netcdf(
#     path='/home/ralager/ml_for_national_blend/ml_for_national_blend/utils/wrf_arw_coords.nc',
#     mode='w', format='NETCDF3_64BIT'
# )









# import numpy
# from gewittergefahr.gg_utils import time_conversion
# from gewittergefahr.gg_utils import time_periods
#
# init_times_unix_sec = time_periods.range_and_interval_to_list(
#     start_time_unix_sec=time_conversion.string_to_unix_sec('2022-11-01-00', '%Y-%m-%d-%H'),
#     end_time_unix_sec=time_conversion.string_to_unix_sec('2023-10-27-00', '%Y-%m-%d-%H'),
#     time_interval_sec=5 * 86400,
#     include_endpoint=True
# )
#
# init_times_unix_sec = init_times_unix_sec[:-1] + 2 * 86400
# init_times_unix_sec = numpy.concatenate((init_times_unix_sec, init_times_unix_sec + 43200))
# init_times_unix_sec = numpy.sort(init_times_unix_sec)
#
# init_time_strings = [
#     time_conversion.unix_sec_to_string(t, '%Y-%m-%d-%H')
#     for t in init_times_unix_sec
# ]
#
# print(len(init_time_strings))
# print(' '.join(['"{0:s}"'.format(t) for t in init_time_strings]))








import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from scipy.interpolate import RegularGridInterpolator
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from ml_for_wildfire_wpo.io import border_io
from ml_for_wildfire_wpo.plotting import plotting_utils
from ml_for_national_blend.io import wrf_arw_io
from ml_for_national_blend.utils import wrf_arw_utils

TOLERANCE = 1e-6

LEAD_TIMES_HOURS = numpy.array([2, 24, 48], dtype=int)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

SEQUENTIAL_COLOUR_MAP_OBJECT = pyplot.get_cmap('viridis')
DIVERGING_COLOUR_MAP_OBJECT = pyplot.get_cmap('seismic')

OUTPUT_DIR_NAME = (
    '/home/ralager/condo/swatwork/ralager/scratch1/RDARCH/rda-ghpcs/'
    'Ryan.Lagerquist/ml_for_national_blend_project/nwp_model_data/wrf_arw/'
    'processed_test/wrf_arw_2022-11-21-12'
)


def _grid_points_to_edges_1d(grid_point_coords):
    """Converts grid points (i.e., cell centers) to cell edges for 1-D grid.

    P = number of grid points

    :param grid_point_coords: length-P numpy array of grid-point coordinates, in
        increasing order.
    :return: grid_cell_edge_coords: length-(P + 1) numpy array of grid-cell-edge
        coordinates, also in increasing order.
    """

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


file_system_utils.mkdir_recursive_if_necessary(directory_name=OUTPUT_DIR_NAME)

wrf_arw_table_xarray = wrf_arw_io.read_file(
    '/home/ralager/condo/swatwork/ralager/scratch1/RDARCH/rda-ghpcs/'
    'Ryan.Lagerquist/ml_for_national_blend_project/nwp_model_data/wrf_arw/'
    'processed_test/wrf_arw_2022-11-21-12.zarr'
)

print(wrf_arw_table_xarray)

border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()
border_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
    border_longitudes_deg_e
)

field_names = wrf_arw_table_xarray.coords['field_name'].values

for i in range(len(LEAD_TIMES_HOURS)):
    i_other = numpy.where(
        wrf_arw_table_xarray.coords['forecast_hour'].values
        == LEAD_TIMES_HOURS[i]
    )[0][0]

    for j in range(len(field_names)):
        figure_object, axes_object = pyplot.subplots(
            1, 1,
            figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

        data_matrix = wrf_arw_table_xarray[wrf_arw_utils.DATA_KEY].values[
            i_other, ..., j
        ]
        # orig_dimensions = data_matrix.shape
        # data_matrix = numpy.ravel(data_matrix)
        # data_matrix = numpy.reshape(data_matrix, orig_dimensions, order='F')

        if '_wind' in field_names[j]:
            colour_map_object = DIVERGING_COLOUR_MAP_OBJECT
            max_colour_value = numpy.nanpercentile(
                numpy.absolute(data_matrix), 99.9
            )
            if numpy.isnan(max_colour_value):
                max_colour_value = TOLERANCE

            max_colour_value = max([max_colour_value, TOLERANCE])
            min_colour_value = -1 * max_colour_value
        else:
            colour_map_object = SEQUENTIAL_COLOUR_MAP_OBJECT
            min_colour_value = numpy.nanpercentile(
                data_matrix, 0.1
            )
            max_colour_value = numpy.nanpercentile(
                data_matrix, 99.9
            )

            if numpy.isnan(min_colour_value):
                min_colour_value = 0.
                max_colour_value = TOLERANCE

        colour_norm_object = pyplot.Normalize(
            vmin=min_colour_value, vmax=max_colour_value
        )
        plot_in_log2_scale = field_names[j] in [wrf_arw_utils.PRECIP_NAME]

        if plot_in_log2_scale:
            data_matrix_to_plot = numpy.log2(data_matrix + 1.)
            colour_norm_object = pyplot.Normalize(
                vmin=numpy.log2(colour_norm_object.vmin + 1),
                vmax=numpy.log2(colour_norm_object.vmax + 1)
            )
        else:
            data_matrix_to_plot = data_matrix + 0.

        edge_latitude_matrix_deg_n = _grid_points_to_edges_2d(
            wrf_arw_table_xarray[wrf_arw_utils.LATITUDE_KEY].values
        )
        edge_longitude_matrix_deg_e = _grid_points_to_edges_2d(
            wrf_arw_table_xarray[wrf_arw_utils.LONGITUDE_KEY].values
        )
        data_matrix_to_plot = grids.latlng_field_grid_points_to_edges(
            field_matrix=data_matrix_to_plot,
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

        colour_bar_object = gg_plotting_utils.plot_colour_bar(
            axes_object_or_matrix=axes_object,
            data_matrix=data_matrix,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            orientation_string='vertical',
            extend_min=True, extend_max=True
        )

        if plot_in_log2_scale:
            tick_values = colour_bar_object.get_ticks()
            tick_strings = [
                '{0:.0f}'.format(numpy.power(2., v) - 1) for v in tick_values
            ]

            colour_bar_object.set_ticks(tick_values)
            colour_bar_object.set_ticklabels(tick_strings)

        plotting_utils.plot_borders(
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            axes_object=axes_object,
            line_colour=numpy.full(3, 0.)
        )
        plotting_utils.plot_grid_lines(
            plot_latitudes_deg_n=numpy.ravel(edge_latitude_matrix_deg_n),
            plot_longitudes_deg_e=numpy.ravel(edge_longitude_matrix_deg_e),
            axes_object=axes_object,
            meridian_spacing_deg=10.,
            parallel_spacing_deg=5.
        )

        axes_object.set_xlim(
            numpy.min(wrf_arw_table_xarray[wrf_arw_utils.LONGITUDE_KEY].values),
            numpy.max(wrf_arw_table_xarray[wrf_arw_utils.LONGITUDE_KEY].values)
        )
        axes_object.set_ylim(
            numpy.min(wrf_arw_table_xarray[wrf_arw_utils.LATITUDE_KEY].values),
            numpy.max(wrf_arw_table_xarray[wrf_arw_utils.LATITUDE_KEY].values)
        )

        axes_object.set_title('{0:s} at {1:d}-hour lead time'.format(
            field_names[j], LEAD_TIMES_HOURS[i]
        ))

        output_file_name = '{0:s}/{1:s}_lead={2:02d}hours.jpg'.format(
            OUTPUT_DIR_NAME, field_names[j], LEAD_TIMES_HOURS[i]
        )

        print(output_file_name)
        figure_object.savefig(
            output_file_name, dpi=300, pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)
