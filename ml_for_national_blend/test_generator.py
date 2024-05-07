"""Test for data-generator.

USE ONCE AND DESTROY.
"""

import os
import sys
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import file_system_utils
import gg_plotting_utils
import nbm_utils
import urma_utils
import nwp_model_utils
import nbm_constant_utils
import neural_net

TOLERANCE = 1e-6

SEQUENTIAL_COLOUR_MAP_OBJECT = pyplot.get_cmap('viridis')
DIVERGING_COLOUR_MAP_OBJECT = pyplot.get_cmap('seismic')

FIGURE_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_national_blend_models/'
    'test_generator/unnormalized'
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

NWP_NORM_FILE_KEY = 'nwp_normalization_file_name'
NWP_USE_QUANTILE_NORM_KEY = 'nwp_use_quantile_norm'
TARGET_NORM_FILE_KEY = 'target_normalization_file_name'
TARGETS_USE_QUANTILE_NORM_KEY = 'targets_use_quantile_norm'
NBM_CONSTANT_FIELDS_KEY = 'nbm_constant_field_names'
NBM_CONSTANT_FILE_KEY = 'nbm_constant_file_name'
SUBSET_GRID_KEY = 'subset_grid'

# init_time_limits_unix_sec = numpy.array([
#     time_conversion.string_to_unix_sec('2022-11-01-00', '%Y-%m-%d-%H'),
#     time_conversion.string_to_unix_sec('2023-02-01-00', '%Y-%m-%d-%H')
# ], dtype=int)

init_time_limits_unix_sec = numpy.array([
    time_conversion.string_to_unix_sec('2022-11-01-00', '%Y-%m-%d-%H'),
    time_conversion.string_to_unix_sec('2022-11-01-12', '%Y-%m-%d-%H')
], dtype=int)

nwp_model_to_dir_name = {
    nwp_model_utils.HRRR_MODEL_NAME:
        '/scratch2/BMC/gsd-hpcs/Ryan.Lagerquist/ml_for_national_blend_project/'
        'nwp_model_data/hrrr/processed/interp_to_nbm_grid'
}

FIELD_NAMES = [
    nwp_model_utils.MSL_PRESSURE_NAME,
    nwp_model_utils.SURFACE_PRESSURE_NAME,
    nwp_model_utils.TEMPERATURE_2METRE_NAME,
    nwp_model_utils.DEWPOINT_2METRE_NAME,
    nwp_model_utils.U_WIND_10METRE_NAME,
    nwp_model_utils.V_WIND_10METRE_NAME,
    nwp_model_utils.PRECIP_NAME,
    nwp_model_utils.U_WIND_1000MB_NAME,
    nwp_model_utils.V_WIND_1000MB_NAME,
    nwp_model_utils.TEMPERATURE_950MB_NAME,
    nwp_model_utils.RELATIVE_HUMIDITY_850MB_NAME,
    nwp_model_utils.TEMPERATURE_850MB_NAME,
    nwp_model_utils.HEIGHT_700MB_NAME,
    nwp_model_utils.RELATIVE_HUMIDITY_700MB_NAME,
    nwp_model_utils.U_WIND_700MB_NAME,
    nwp_model_utils.V_WIND_700MB_NAME,
    nwp_model_utils.HEIGHT_500MB_NAME,
    nwp_model_utils.RELATIVE_HUMIDITY_500MB_NAME,
    nwp_model_utils.U_WIND_500MB_NAME,
    nwp_model_utils.V_WIND_500MB_NAME
]

TARGET_FIELD_NAMES = [
    urma_utils.TEMPERATURE_2METRE_NAME, urma_utils.DEWPOINT_2METRE_NAME,
    urma_utils.U_WIND_10METRE_NAME, urma_utils.V_WIND_10METRE_NAME,
    urma_utils.WIND_GUST_10METRE_NAME
]

nwp_model_to_field_names = {
    nwp_model_utils.HRRR_MODEL_NAME: FIELD_NAMES
}

option_dict = {
    INIT_TIME_LIMITS_KEY: init_time_limits_unix_sec,
    NWP_LEAD_TIMES_KEY: numpy.array([6, 48], dtype=int),
    NWP_MODEL_TO_DIR_KEY: nwp_model_to_dir_name,
    NWP_MODEL_TO_FIELDS_KEY: nwp_model_to_field_names,
    NWP_NORM_FILE_KEY: None,
        # '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/'
        # 'ml_for_national_blend_project/nwp_model_data/'
        # 'normalization_params_20221101-20230531.nc',
    NWP_USE_QUANTILE_NORM_KEY: True,
    TARGET_LEAD_TIME_KEY: 24,
    TARGET_FIELDS_KEY: TARGET_FIELD_NAMES,
    TARGET_DIR_KEY:
        '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/'
        'ml_for_national_blend_project/urma_data/processed',
    TARGET_NORM_FILE_KEY: None,
    TARGETS_USE_QUANTILE_NORM_KEY: False,
    NBM_CONSTANT_FIELDS_KEY: [
        nbm_constant_utils.LAND_SEA_MASK_NAME,
        nbm_constant_utils.OROGRAPHIC_HEIGHT_NAME
    ],
    NBM_CONSTANT_FILE_KEY:
        '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_national_blend_project/nbm_constants/nbm_constants.nc',
        # '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/'
        # 'ml_for_national_blend_project/nbm_constants/'
        # 'nbm_constants_quantile_normalized.nc',
    BATCH_SIZE_KEY: 4,
    SENTINEL_VALUE_KEY: -9999.,
    SUBSET_GRID_KEY: True
}


generator_object = neural_net.data_generator(option_dict)
predictor_matrices, target_matrix = next(generator_object)

file_system_utils.mkdir_recursive_if_necessary(directory_name=FIGURE_DIR_NAME)

predictor_matrix_2pt5km = predictor_matrices['2pt5km_inputs'][0, ...].astype(numpy.float64)
predictor_matrix_constant = predictor_matrices['constant_inputs'][0, ...].astype(numpy.float64)
target_matrix = target_matrix[0, ...].astype(numpy.float64)

predictor_matrix_2pt5km[predictor_matrix_2pt5km < -9000] = numpy.nan
predictor_matrix_constant[predictor_matrix_constant < -9000] = numpy.nan
target_matrix[target_matrix < -9000] = numpy.nan

nwp_lead_times_hours = numpy.array([6, 48], dtype=int)

full_latitude_matrix_deg_n, full_longitude_matrix_deg_e = nbm_utils.read_coords()
full_latitude_matrix_deg_n = full_latitude_matrix_deg_n[544:993, 752:1201]
full_longitude_matrix_deg_e = full_longitude_matrix_deg_e[544:993, 752:1201]

for i in range(len(nwp_lead_times_hours)):
    field_names = nwp_model_to_field_names[nwp_model_utils.HRRR_MODEL_NAME]

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
        data_matrix_to_plot = numpy.ma.masked_where(
            numpy.isnan(data_matrix_to_plot), data_matrix_to_plot
        )

        axes_object.pcolor(
            full_longitude_matrix_deg_e, full_latitude_matrix_deg_n,
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
            numpy.min(full_longitude_matrix_deg_e),
            numpy.max(full_longitude_matrix_deg_e)
        )
        axes_object.set_ylim(
            numpy.min(full_latitude_matrix_deg_n),
            numpy.max(full_latitude_matrix_deg_n)
        )

        axes_object.set_title('HRRR {0:s} at {1:d}-hour lead time'.format(
            field_names[j], nwp_lead_times_hours[i]
        ))

        output_file_name = '{0:s}/{1:s}_{2:02d}hours_hrrr.jpg'.format(
            FIGURE_DIR_NAME, field_names[j], nwp_lead_times_hours[i]
        )

        print(output_file_name)
        figure_object.savefig(
            output_file_name, dpi=300, pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

field_names = [
    nbm_constant_utils.LAND_SEA_MASK_NAME,
    nbm_constant_utils.OROGRAPHIC_HEIGHT_NAME
]

for j in range(len(field_names)):
    this_data_matrix = predictor_matrix_constant[..., j]
    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(15, 15)
    )
    
    colour_map_object = SEQUENTIAL_COLOUR_MAP_OBJECT
    min_colour_value = numpy.nanpercentile(this_data_matrix, 0.1)
    max_colour_value = numpy.nanpercentile(this_data_matrix, 99.9)

    if numpy.isnan(min_colour_value):
        min_colour_value = 0.
        max_colour_value = TOLERANCE

    colour_norm_object = pyplot.Normalize(
        vmin=min_colour_value, vmax=max_colour_value
    )
    data_matrix_to_plot = this_data_matrix + 0.
    data_matrix_to_plot = numpy.ma.masked_where(
        numpy.isnan(data_matrix_to_plot), data_matrix_to_plot
    )

    axes_object.pcolor(
        full_longitude_matrix_deg_e, full_latitude_matrix_deg_n,
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
        numpy.min(full_longitude_matrix_deg_e),
        numpy.max(full_longitude_matrix_deg_e)
    )
    axes_object.set_ylim(
        numpy.min(full_latitude_matrix_deg_n),
        numpy.max(full_latitude_matrix_deg_n)
    )

    axes_object.set_title('NBM-constant {0:s}'.format(field_names[j]))
    output_file_name = '{0:s}/{1:s}_nbm_constant.jpg'.format(
        FIGURE_DIR_NAME, field_names[j]
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
    data_matrix_to_plot = numpy.ma.masked_where(
        numpy.isnan(data_matrix_to_plot), data_matrix_to_plot
    )

    axes_object.pcolor(
        full_longitude_matrix_deg_e, full_latitude_matrix_deg_n,
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
        numpy.min(full_longitude_matrix_deg_e),
        numpy.max(full_longitude_matrix_deg_e)
    )
    axes_object.set_ylim(
        numpy.min(full_latitude_matrix_deg_n),
        numpy.max(full_latitude_matrix_deg_n)
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
