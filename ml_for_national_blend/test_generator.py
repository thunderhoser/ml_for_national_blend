"""Test for data-generator.

USE ONCE AND DESTROY.
"""

import os
import sys
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from scipy.interpolate import RegularGridInterpolator

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import grids
import longitude_conversion as lng_conversion
import time_conversion
import file_system_utils
import gg_plotting_utils
import nbm_utils
import urma_utils
import nwp_model_utils
import neural_net

TOLERANCE = 1e-6

SEQUENTIAL_COLOUR_MAP_OBJECT = pyplot.get_cmap('viridis')
DIVERGING_COLOUR_MAP_OBJECT = pyplot.get_cmap('seismic')

FIGURE_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_national_blend_models/'
    'test_generator'
)

DEFAULT_FONT_SIZE = 30
pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

INIT_TIME_LIMITS_KEY = 'init_time_limits_unix_sec'
NWP_LEAD_TIMES_KEY = 'nwp_lead_times_hours'
NWP_MODEL_TO_DIR_KEY = 'nwp_model_to_dir_name'
NWP_MODEL_TO_FIELDS_KEY = 'nwp_model_to_field_names'
TARGET_LEAD_TIME_KEY = 'target_lead_time_hours'
TARGET_FIELDS_KEY = 'target_field_names'
TARGET_DIR_KEY = 'target_dir_name'
BATCH_SIZE_KEY = 'num_examples_per_batch'
SENTINEL_VALUE_KEY = 'sentinel_value'

# init_time_limits_unix_sec = numpy.array([
#     time_conversion.string_to_unix_sec('2022-11-01-00', '%Y-%m-%d-%H'),
#     time_conversion.string_to_unix_sec('2023-02-01-00', '%Y-%m-%d-%H')
# ], dtype=int)

init_time_limits_unix_sec = numpy.array([
    time_conversion.string_to_unix_sec('2022-11-01-00', '%Y-%m-%d-%H'),
    time_conversion.string_to_unix_sec('2022-11-01-12', '%Y-%m-%d-%H')
], dtype=int)

nwp_model_to_dir_name = {
    nwp_model_utils.GFS_MODEL_NAME:
        '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_national_blend_project/nwp_model_data/gfs/processed/interp_to_nbm_grid',
    nwp_model_utils.GEFS_MODEL_NAME:
        '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_national_blend_project/nwp_model_data/gefs/processed/interp_to_nbm_grid',
    nwp_model_utils.RAP_MODEL_NAME:
        '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_national_blend_project/nwp_model_data/rap/processed/interp_to_nbm_grid',
    nwp_model_utils.GRIDDED_LAMP_MODEL_NAME:
        '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_national_blend_project/nwp_model_data/gridded_lamp/processed/interp_to_nbm_grid'
}

FIELD_NAMES = [
    nwp_model_utils.TEMPERATURE_2METRE_NAME,
    nwp_model_utils.DEWPOINT_2METRE_NAME,
    nwp_model_utils.U_WIND_10METRE_NAME, nwp_model_utils.V_WIND_10METRE_NAME,
    nwp_model_utils.PRECIP_NAME,
    nwp_model_utils.HEIGHT_700MB_NAME,
    nwp_model_utils.RELATIVE_HUMIDITY_850MB_NAME,
    nwp_model_utils.U_WIND_1000MB_NAME, nwp_model_utils.V_WIND_1000MB_NAME,
    nwp_model_utils.TEMPERATURE_950MB_NAME
]

TARGET_FIELD_NAMES = [
    urma_utils.TEMPERATURE_2METRE_NAME, urma_utils.DEWPOINT_2METRE_NAME,
    urma_utils.U_WIND_10METRE_NAME, urma_utils.V_WIND_10METRE_NAME,
    urma_utils.WIND_GUST_10METRE_NAME
]

nwp_model_to_field_names = {
    nwp_model_utils.GFS_MODEL_NAME: FIELD_NAMES,
    nwp_model_utils.GEFS_MODEL_NAME: FIELD_NAMES[:-1],
    nwp_model_utils.RAP_MODEL_NAME: FIELD_NAMES[:-2],
    nwp_model_utils.GRIDDED_LAMP_MODEL_NAME: FIELD_NAMES[:-3]
}

option_dict = {
    INIT_TIME_LIMITS_KEY: init_time_limits_unix_sec,
    NWP_LEAD_TIMES_KEY: numpy.array([6, 12, 18], dtype=int),
    NWP_MODEL_TO_DIR_KEY: nwp_model_to_dir_name,
    NWP_MODEL_TO_FIELDS_KEY: nwp_model_to_field_names,
    TARGET_LEAD_TIME_KEY: 24,
    TARGET_FIELDS_KEY: TARGET_FIELD_NAMES,
    TARGET_DIR_KEY: '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_national_blend_project/urma_data/processed',
    BATCH_SIZE_KEY: 4,
    SENTINEL_VALUE_KEY: -10.
}


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


generator_object = neural_net.data_generator(option_dict)
predictor_matrices, target_matrix = next(generator_object)

file_system_utils.mkdir_recursive_if_necessary(directory_name=FIGURE_DIR_NAME)

predictor_matrix_2pt5km = predictor_matrices[0][0, ...]
predictor_matrix_10km = predictor_matrices[1][0, ...]
predictor_matrix_20km = predictor_matrices[2][0, ...]
predictor_matrix_40km = predictor_matrices[3][0, ...]
target_matrix = target_matrix[0, ...]

nwp_lead_times_hours = numpy.array([6, 12, 18], dtype=int)

full_latitude_matrix_deg_n, full_longitude_matrix_deg_e = nbm_utils.read_coords()

for i in range(len(nwp_lead_times_hours)):
    if i == 1:
        continue

    field_names = nwp_model_to_field_names[nwp_model_utils.GRIDDED_LAMP_MODEL_NAME]

    for j in range(len(field_names)):
        this_data_matrix = predictor_matrix_2pt5km[..., i, j]
        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(15, 15)
        )

        if '_wind' in field_names[j]:
            colour_map_object = DIVERGING_COLOUR_MAP_OBJECT
            max_colour_value = numpy.nanpercentile(
                numpy.absolute(this_data_matrix), 99.9
            )
            if numpy.isnan(max_colour_value):
                max_colour_value = TOLERANCE

            max_colour_value = max([max_colour_value, TOLERANCE])
            min_colour_value = -1 * max_colour_value
        else:
            colour_map_object = SEQUENTIAL_COLOUR_MAP_OBJECT
            min_colour_value = numpy.nanpercentile(
                this_data_matrix, 0.1
            )
            max_colour_value = numpy.nanpercentile(
                this_data_matrix, 99.9
            )

            if numpy.isnan(min_colour_value):
                min_colour_value = 0.
                max_colour_value = TOLERANCE

        colour_norm_object = pyplot.Normalize(
            vmin=min_colour_value, vmax=max_colour_value
        )
        data_matrix_to_plot = this_data_matrix + 0.

        edge_latitude_matrix_deg_n = _grid_points_to_edges_2d(
            full_latitude_matrix_deg_n + 0.
        )
        edge_longitude_matrix_deg_e = _grid_points_to_edges_2d(
            full_longitude_matrix_deg_e + 0.
        )
        edge_longitude_matrix_deg_e = lng_conversion.convert_lng_positive_in_west(
            edge_longitude_matrix_deg_e
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
            data_matrix=this_data_matrix,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            orientation_string='vertical',
            extend_min=True, extend_max=True
        )

        axes_object.set_xlim(
            numpy.min(edge_longitude_matrix_deg_e),
            numpy.max(edge_longitude_matrix_deg_e)
        )
        axes_object.set_ylim(
            numpy.min(edge_latitude_matrix_deg_n),
            numpy.max(edge_latitude_matrix_deg_n)
        )

        axes_object.set_title('GLAMP {0:s} at {1:d}-hour lead time'.format(
            field_names[j], nwp_lead_times_hours[i]
        ))

        output_file_name = '{0:s}/{1:s}_{2:02d}hours_glamp.jpg'.format(
            FIGURE_DIR_NAME, field_names[j], nwp_lead_times_hours[i]
        )

        print(output_file_name)
        figure_object.savefig(
            output_file_name, dpi=300, pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)


for i in range(len(nwp_lead_times_hours)):
    if i == 1:
        continue

    field_names = nwp_model_to_field_names[nwp_model_utils.RAP_MODEL_NAME]

    for j in range(len(field_names)):
        this_data_matrix = predictor_matrix_10km[..., i, j]
        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(15, 15)
        )

        if '_wind' in field_names[j]:
            colour_map_object = DIVERGING_COLOUR_MAP_OBJECT
            max_colour_value = numpy.nanpercentile(
                numpy.absolute(this_data_matrix), 99.9
            )
            if numpy.isnan(max_colour_value):
                max_colour_value = TOLERANCE

            max_colour_value = max([max_colour_value, TOLERANCE])
            min_colour_value = -1 * max_colour_value
        else:
            colour_map_object = SEQUENTIAL_COLOUR_MAP_OBJECT
            min_colour_value = numpy.nanpercentile(
                this_data_matrix, 0.1
            )
            max_colour_value = numpy.nanpercentile(
                this_data_matrix, 99.9
            )

            if numpy.isnan(min_colour_value):
                min_colour_value = 0.
                max_colour_value = TOLERANCE

        colour_norm_object = pyplot.Normalize(
            vmin=min_colour_value, vmax=max_colour_value
        )
        data_matrix_to_plot = this_data_matrix + 0.

        edge_latitude_matrix_deg_n = _grid_points_to_edges_2d(
            full_latitude_matrix_deg_n[::4, ::4] + 0.
        )
        edge_longitude_matrix_deg_e = _grid_points_to_edges_2d(
            full_longitude_matrix_deg_e[::4, ::4] + 0.
        )
        edge_longitude_matrix_deg_e = lng_conversion.convert_lng_positive_in_west(
            edge_longitude_matrix_deg_e
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
            data_matrix=this_data_matrix,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            orientation_string='vertical',
            extend_min=True, extend_max=True
        )

        axes_object.set_xlim(
            numpy.min(edge_longitude_matrix_deg_e),
            numpy.max(edge_longitude_matrix_deg_e)
        )
        axes_object.set_ylim(
            numpy.min(edge_latitude_matrix_deg_n),
            numpy.max(edge_latitude_matrix_deg_n)
        )

        axes_object.set_title('RAP {0:s} at {1:d}-hour lead time'.format(
            field_names[j], nwp_lead_times_hours[i]
        ))

        output_file_name = '{0:s}/{1:s}_{2:02d}hours_rap.jpg'.format(
            FIGURE_DIR_NAME, field_names[j], nwp_lead_times_hours[i]
        )

        print(output_file_name)
        figure_object.savefig(
            output_file_name, dpi=300, pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)


for i in range(len(nwp_lead_times_hours)):
    if i == 1:
        continue

    field_names = nwp_model_to_field_names[nwp_model_utils.GFS_MODEL_NAME]

    for j in range(len(field_names)):
        this_data_matrix = predictor_matrix_20km[..., i, j]
        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(15, 15)
        )

        if '_wind' in field_names[j]:
            colour_map_object = DIVERGING_COLOUR_MAP_OBJECT
            max_colour_value = numpy.nanpercentile(
                numpy.absolute(this_data_matrix), 99.9
            )
            if numpy.isnan(max_colour_value):
                max_colour_value = TOLERANCE

            max_colour_value = max([max_colour_value, TOLERANCE])
            min_colour_value = -1 * max_colour_value
        else:
            colour_map_object = SEQUENTIAL_COLOUR_MAP_OBJECT
            min_colour_value = numpy.nanpercentile(
                this_data_matrix, 0.1
            )
            max_colour_value = numpy.nanpercentile(
                this_data_matrix, 99.9
            )

            if numpy.isnan(min_colour_value):
                min_colour_value = 0.
                max_colour_value = TOLERANCE

        colour_norm_object = pyplot.Normalize(
            vmin=min_colour_value, vmax=max_colour_value
        )
        data_matrix_to_plot = this_data_matrix + 0.

        edge_latitude_matrix_deg_n = _grid_points_to_edges_2d(
            full_latitude_matrix_deg_n[::8, ::8] + 0.
        )
        edge_longitude_matrix_deg_e = _grid_points_to_edges_2d(
            full_longitude_matrix_deg_e[::8, ::8] + 0.
        )
        edge_longitude_matrix_deg_e = lng_conversion.convert_lng_positive_in_west(
            edge_longitude_matrix_deg_e
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
            data_matrix=this_data_matrix,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            orientation_string='vertical',
            extend_min=True, extend_max=True
        )

        axes_object.set_xlim(
            numpy.min(edge_longitude_matrix_deg_e),
            numpy.max(edge_longitude_matrix_deg_e)
        )
        axes_object.set_ylim(
            numpy.min(edge_latitude_matrix_deg_n),
            numpy.max(edge_latitude_matrix_deg_n)
        )

        axes_object.set_title('GFS {0:s} at {1:d}-hour lead time'.format(
            field_names[j], nwp_lead_times_hours[i]
        ))

        output_file_name = '{0:s}/{1:s}_{2:02d}hours_gfs.jpg'.format(
            FIGURE_DIR_NAME, field_names[j], nwp_lead_times_hours[i]
        )

        print(output_file_name)
        figure_object.savefig(
            output_file_name, dpi=300, pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)


for i in range(len(nwp_lead_times_hours)):
    if i == 1:
        continue

    field_names = nwp_model_to_field_names[nwp_model_utils.GEFS_MODEL_NAME]

    for j in range(len(field_names)):
        this_data_matrix = predictor_matrix_40km[..., i, j]
        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(15, 15)
        )

        if '_wind' in field_names[j]:
            colour_map_object = DIVERGING_COLOUR_MAP_OBJECT
            max_colour_value = numpy.nanpercentile(
                numpy.absolute(this_data_matrix), 99.9
            )
            if numpy.isnan(max_colour_value):
                max_colour_value = TOLERANCE

            max_colour_value = max([max_colour_value, TOLERANCE])
            min_colour_value = -1 * max_colour_value
        else:
            colour_map_object = SEQUENTIAL_COLOUR_MAP_OBJECT
            min_colour_value = numpy.nanpercentile(
                this_data_matrix, 0.1
            )
            max_colour_value = numpy.nanpercentile(
                this_data_matrix, 99.9
            )

            if numpy.isnan(min_colour_value):
                min_colour_value = 0.
                max_colour_value = TOLERANCE

        colour_norm_object = pyplot.Normalize(
            vmin=min_colour_value, vmax=max_colour_value
        )
        data_matrix_to_plot = this_data_matrix + 0.

        edge_latitude_matrix_deg_n = _grid_points_to_edges_2d(
            full_latitude_matrix_deg_n[::16, ::16] + 0.
        )
        edge_longitude_matrix_deg_e = _grid_points_to_edges_2d(
            full_longitude_matrix_deg_e[::16, ::16] + 0.
        )
        edge_longitude_matrix_deg_e = lng_conversion.convert_lng_positive_in_west(
            edge_longitude_matrix_deg_e
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
            data_matrix=this_data_matrix,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            orientation_string='vertical',
            extend_min=True, extend_max=True
        )

        axes_object.set_xlim(
            numpy.min(edge_longitude_matrix_deg_e),
            numpy.max(edge_longitude_matrix_deg_e)
        )
        axes_object.set_ylim(
            numpy.min(edge_latitude_matrix_deg_n),
            numpy.max(edge_latitude_matrix_deg_n)
        )

        axes_object.set_title('GEFS {0:s} at {1:d}-hour lead time'.format(
            field_names[j], nwp_lead_times_hours[i]
        ))

        output_file_name = '{0:s}/{1:s}_{2:02d}hours_gefs.jpg'.format(
            FIGURE_DIR_NAME, field_names[j], nwp_lead_times_hours[i]
        )

        print(output_file_name)
        figure_object.savefig(
            output_file_name, dpi=300, pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)


for j in range(len(TARGET_FIELD_NAMES)):
    this_data_matrix = target_matrix[..., j]
    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(15, 15)
    )

    if '_wind' in TARGET_FIELD_NAMES[j]:
        colour_map_object = DIVERGING_COLOUR_MAP_OBJECT
        max_colour_value = numpy.nanpercentile(
            numpy.absolute(this_data_matrix), 99.9
        )
        if numpy.isnan(max_colour_value):
            max_colour_value = TOLERANCE

        max_colour_value = max([max_colour_value, TOLERANCE])
        min_colour_value = -1 * max_colour_value
    else:
        colour_map_object = SEQUENTIAL_COLOUR_MAP_OBJECT
        min_colour_value = numpy.nanpercentile(
            this_data_matrix, 0.1
        )
        max_colour_value = numpy.nanpercentile(
            this_data_matrix, 99.9
        )

        if numpy.isnan(min_colour_value):
            min_colour_value = 0.
            max_colour_value = TOLERANCE

    colour_norm_object = pyplot.Normalize(
        vmin=min_colour_value, vmax=max_colour_value
    )
    data_matrix_to_plot = this_data_matrix + 0.

    edge_latitude_matrix_deg_n = _grid_points_to_edges_2d(
        full_latitude_matrix_deg_n + 0.
    )
    edge_longitude_matrix_deg_e = _grid_points_to_edges_2d(
        full_longitude_matrix_deg_e + 0.
    )
    edge_longitude_matrix_deg_e = lng_conversion.convert_lng_positive_in_west(
        edge_longitude_matrix_deg_e
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
        data_matrix=this_data_matrix,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical',
        extend_min=True, extend_max=True
    )

    axes_object.set_xlim(
        numpy.min(edge_longitude_matrix_deg_e),
        numpy.max(edge_longitude_matrix_deg_e)
    )
    axes_object.set_ylim(
        numpy.min(edge_latitude_matrix_deg_n),
        numpy.max(edge_latitude_matrix_deg_n)
    )

    axes_object.set_title('URMA {0:s} at 24-hour lead time'.format(
        TARGET_FIELD_NAMES[j]
    ))

    output_file_name = '{0:s}/{1:s}_24hours_urma.jpg'.format(
        FIGURE_DIR_NAME, TARGET_FIELD_NAMES[j]
    )

    print(output_file_name)
    figure_object.savefig(
        output_file_name, dpi=300, pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)
