"""Test for data-generator.

USE ONCE AND DESTROY.
"""

import os
import sys
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import urma_utils
import nwp_model_utils
import neural_net

INIT_TIME_LIMITS_KEY = 'init_time_limits_unix_sec'
NWP_LEAD_TIMES_KEY = 'nwp_lead_times_hours'
NWP_MODEL_TO_DIR_KEY = 'nwp_model_to_dir_name'
NWP_MODEL_TO_FIELDS_KEY = 'nwp_model_to_field_names'
TARGET_LEAD_TIME_KEY = 'target_lead_time_hours'
TARGET_FIELDS_KEY = 'target_field_names'
TARGET_DIR_KEY = 'target_dir_name'
BATCH_SIZE_KEY = 'num_examples_per_batch'
SENTINEL_VALUE_KEY = 'sentinel_value'

init_time_limits_unix_sec = numpy.array([
    time_conversion.string_to_unix_sec('2022-11-01-00', '%Y-%m-%d-%H'),
    time_conversion.string_to_unix_sec('2023-02-01-00', '%Y-%m-%d-%H')
], dtype=int)

nwp_model_to_dir_name = {
    nwp_model_utils.GFS_MODEL_NAME:
        '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_national_blend_project/nwp_model_data/gfs/processed/interp_to_nbm_grid',
    nwp_model_utils.GEFS_MODEL_NAME:
        '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_national_blend_project/nwp_model_data/gefs/processed/interp_to_nbm_grid',
    nwp_model_utils.RAP_MODEL_NAME:
        '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml_for_national_blend_project/nwp_model_data/rap/processed/interp_to_nbm_grid'
}

FIELD_NAMES = [
    nwp_model_utils.TEMPERATURE_2METRE_NAME,
    nwp_model_utils.DEWPOINT_2METRE_NAME,
    nwp_model_utils.U_WIND_10METRE_NAME, nwp_model_utils.V_WIND_10METRE_NAME,
    nwp_model_utils.WIND_GUST_10METRE_NAME, nwp_model_utils.PRECIP_NAME,
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
    nwp_model_utils.RAP_MODEL_NAME: FIELD_NAMES[:-2]
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

generator_object = neural_net.data_generator(option_dict)
next(generator_object)
