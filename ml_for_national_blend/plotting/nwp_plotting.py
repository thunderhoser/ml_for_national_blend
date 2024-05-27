"""Methods for plotting NWP output."""

import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import temperature_conversions as temperature_conv
from ml_for_national_blend.utils import nwp_model_utils
from ml_for_national_blend.plotting import target_plotting

TOLERANCE = 1e-6
NAN_COLOUR = numpy.full(3, 152. / 255)

PASCALS_TO_MB = 0.01
FRACTION_TO_PERCENT = 100.
METRES_TO_MM = 1000.
METRES_TO_DEKAMETRES = 0.1

FIELD_NAME_TO_FANCY = {
    nwp_model_utils.MSL_PRESSURE_NAME: 'MSL pressure (hPa)',
    nwp_model_utils.SURFACE_PRESSURE_NAME: 'surface pressure (hPa)',
    nwp_model_utils.TEMPERATURE_2METRE_NAME: r'2-m temperature ($^{\circ}$C)',
    nwp_model_utils.DEWPOINT_2METRE_NAME: r'2-m dewpoint ($^{\circ}$C)',
    nwp_model_utils.RELATIVE_HUMIDITY_2METRE_NAME: r'2-m relative humidity (%)',
    nwp_model_utils.U_WIND_10METRE_NAME: r'10-m zonal wind (m s$^{-1}$)',
    nwp_model_utils.V_WIND_10METRE_NAME: r'10-m meridional wind (m s$^{-1}$)',
    nwp_model_utils.WIND_GUST_10METRE_NAME: r'10-m wind gust (m s$^{-1}$)',
    nwp_model_utils.PRECIP_NAME: 'accumulated precip (mm)',
    nwp_model_utils.HEIGHT_500MB_NAME: '500-hPa height (dam)',
    nwp_model_utils.HEIGHT_700MB_NAME: '700-hPa height (dam)',
    nwp_model_utils.RELATIVE_HUMIDITY_500MB_NAME:
        '500-hPa relative humidity (%)',
    nwp_model_utils.RELATIVE_HUMIDITY_700MB_NAME:
        '700-hPa relative humidity (%)',
    nwp_model_utils.RELATIVE_HUMIDITY_850MB_NAME:
        '850-hPa relative humidity (%)',
    nwp_model_utils.U_WIND_500MB_NAME: r'500-hPa zonal wind (m s$^{-1}$)',
    nwp_model_utils.U_WIND_700MB_NAME: r'700-hPa zonal wind (m s$^{-1}$)',
    nwp_model_utils.U_WIND_1000MB_NAME: r'1000-hPa zonal wind (m s$^{-1}$)',
    nwp_model_utils.V_WIND_500MB_NAME: r'500-hPa meridional wind (m s$^{-1}$)',
    nwp_model_utils.V_WIND_700MB_NAME: r'700-hPa meridional wind (m s$^{-1}$)',
    nwp_model_utils.V_WIND_1000MB_NAME:
        r'1000-hPa meridional wind (m s$^{-1}$)',
    nwp_model_utils.TEMPERATURE_850MB_NAME:
        r'850-hPa temperature ($^{\circ}$C)',
    nwp_model_utils.TEMPERATURE_950MB_NAME:
        r'950-hPa temperature ($^{\circ}$C)',
    nwp_model_utils.MIN_RELATIVE_HUMIDITY_2METRE_NAME: 'minimum 2-m RH (%)',
    nwp_model_utils.MAX_RELATIVE_HUMIDITY_2METRE_NAME: 'maximum 2-m RH (%)'
}

FIELD_NAME_TO_CONV_FACTOR = {
    nwp_model_utils.MSL_PRESSURE_NAME: PASCALS_TO_MB,
    nwp_model_utils.SURFACE_PRESSURE_NAME: PASCALS_TO_MB,
    nwp_model_utils.TEMPERATURE_2METRE_NAME:
        temperature_conv.kelvins_to_celsius,
    nwp_model_utils.DEWPOINT_2METRE_NAME: temperature_conv.kelvins_to_celsius,
    nwp_model_utils.RELATIVE_HUMIDITY_2METRE_NAME: FRACTION_TO_PERCENT,
    nwp_model_utils.U_WIND_10METRE_NAME: 1.,
    nwp_model_utils.V_WIND_10METRE_NAME: 1.,
    nwp_model_utils.WIND_GUST_10METRE_NAME: 1.,
    nwp_model_utils.PRECIP_NAME: METRES_TO_MM,
    nwp_model_utils.HEIGHT_500MB_NAME: METRES_TO_DEKAMETRES,
    nwp_model_utils.HEIGHT_700MB_NAME: METRES_TO_DEKAMETRES,
    nwp_model_utils.RELATIVE_HUMIDITY_500MB_NAME: FRACTION_TO_PERCENT,
    nwp_model_utils.RELATIVE_HUMIDITY_700MB_NAME: FRACTION_TO_PERCENT,
    nwp_model_utils.RELATIVE_HUMIDITY_850MB_NAME: FRACTION_TO_PERCENT,
    nwp_model_utils.U_WIND_500MB_NAME: 1.,
    nwp_model_utils.U_WIND_700MB_NAME: 1.,
    nwp_model_utils.U_WIND_1000MB_NAME: 1.,
    nwp_model_utils.V_WIND_500MB_NAME: 1.,
    nwp_model_utils.V_WIND_700MB_NAME: 1.,
    nwp_model_utils.V_WIND_1000MB_NAME: 1.,
    nwp_model_utils.TEMPERATURE_850MB_NAME: temperature_conv.kelvins_to_celsius,
    nwp_model_utils.TEMPERATURE_950MB_NAME: temperature_conv.kelvins_to_celsius,
    nwp_model_utils.MIN_RELATIVE_HUMIDITY_2METRE_NAME: FRACTION_TO_PERCENT,
    nwp_model_utils.MAX_RELATIVE_HUMIDITY_2METRE_NAME: FRACTION_TO_PERCENT
}

MODEL_NAME_TO_FANCY = {
    nwp_model_utils.WRF_ARW_MODEL_NAME: 'WRF-ARW',
    nwp_model_utils.NAM_MODEL_NAME: 'NAM',
    nwp_model_utils.NAM_NEST_MODEL_NAME: 'NAM Nest',
    nwp_model_utils.RAP_MODEL_NAME: 'RAP',
    nwp_model_utils.GFS_MODEL_NAME: 'GFS',
    nwp_model_utils.HRRR_MODEL_NAME: 'HRRR',
    nwp_model_utils.GEFS_MODEL_NAME: 'GEFS',
    nwp_model_utils.GRIDDED_LAMP_MODEL_NAME: 'GLAMP',
    nwp_model_utils.ECMWF_MODEL_NAME: 'ECMWF',
    nwp_model_utils.GRIDDED_MOS_MODEL_NAME: 'GMOS'
}


def field_to_colour_scheme(field_name, min_value, max_value):
    """Returns colour scheme for one target field.

    :param field_name: Field name.  Must be accepted by
        `urma_utils.check_field_name`.
    :param min_value: Minimum value in colour scheme.
    :param max_value: Max value in colour scheme.
    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    """

    nwp_model_utils.check_field_name(field_name)
    conversion_factor = FIELD_NAME_TO_CONV_FACTOR[field_name]

    if callable(conversion_factor):
        min_value_converted = conversion_factor(min_value)
        max_value_converted = conversion_factor(max_value)
    else:
        min_value_converted = min_value * conversion_factor
        max_value_converted = max_value * conversion_factor

    if field_name not in [
            nwp_model_utils.U_WIND_10METRE_NAME,
            nwp_model_utils.U_WIND_500MB_NAME,
            nwp_model_utils.U_WIND_700MB_NAME,
            nwp_model_utils.U_WIND_1000MB_NAME,
            nwp_model_utils.V_WIND_10METRE_NAME,
            nwp_model_utils.V_WIND_500MB_NAME,
            nwp_model_utils.V_WIND_700MB_NAME,
            nwp_model_utils.V_WIND_1000MB_NAME
    ]:
        colour_map_object = pyplot.get_cmap('viridis')
        colour_map_object.set_bad(NAN_COLOUR)

        max_value_converted = max([
            max_value_converted, min_value_converted + TOLERANCE
        ])
        colour_norm_object = pyplot.Normalize(
            vmin=min_value_converted, vmax=max_value_converted
        )

        return colour_map_object, colour_norm_object

    max_absolute_value = max([
        numpy.absolute(min_value_converted),
        numpy.absolute(max_value_converted)
    ])
    max_absolute_value = max([max_absolute_value, TOLERANCE])

    colour_map_object = pyplot.get_cmap('seismic')
    colour_map_object.set_bad(NAN_COLOUR)
    colour_norm_object = pyplot.Normalize(
        vmin=-1 * max_absolute_value, vmax=max_absolute_value
    )

    return colour_map_object, colour_norm_object


def plot_field(data_matrix, field_name,
               latitude_matrix_deg_n, longitude_matrix_deg_e,
               colour_map_object, colour_norm_object, axes_object,
               plot_colour_bar):
    """Plots one field on a lat/long grid.

    This method is a lightweight wrapper for `target_plotting.plot_field`.

    :param data_matrix: See doc for `target_plotting.plot_field`.
    :param field_name: Field name.  Must be accepted by
        `urma_utils.check_field_name`.
    :param latitude_matrix_deg_n: See doc for `target_plotting.plot_field`.
    :param longitude_matrix_deg_e: Same.
    :param colour_map_object: Same.
    :param colour_norm_object: Same.
    :param axes_object: Same.
    :param plot_colour_bar: Same.
    :return: colour_bar_object: Same.
    """

    nwp_model_utils.check_field_name(field_name)
    conversion_factor = FIELD_NAME_TO_CONV_FACTOR[field_name]

    return target_plotting.plot_field(
        data_matrix=(
            conversion_factor(data_matrix) if callable(conversion_factor)
            else data_matrix * conversion_factor
        ),
        latitude_matrix_deg_n=latitude_matrix_deg_n,
        longitude_matrix_deg_e=longitude_matrix_deg_e,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        axes_object=axes_object,
        plot_colour_bar=plot_colour_bar
    )
