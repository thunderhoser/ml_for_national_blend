"""Helper methods for inputting NWP data to a neural network."""

import os
import sys
import warnings
import numpy
from scipy.interpolate import interp1d

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import temperature_conversions as temperature_conv
import interp_nwp_model_io
import nbm_utils
import misc_utils
import nwp_model_utils
import urma_utils
import normalization

HOURS_TO_SECONDS = 3600
MAX_INTERPOLATION_TIME_HOURS = 24

POSSIBLE_GRID_SPACINGS_KM = numpy.array([2.5, 10, 20, 40])


def _convert_2m_dewp_to_depression(nwp_forecast_table_xarray):
    """Converts 2-metre NWP dewpoints to dewpoint depressions.

    :param nwp_forecast_table_xarray: xarray table in format returned by
        `interp_nwp_model_io.read_file`.
    :return: nwp_forecast_table_xarray: Same, except that dewpoints have been
        replaced with dewpoint depressions.
    """

    dewpoint_matrix_kelvins = nwp_model_utils.get_field(
        nwp_forecast_table_xarray=nwp_forecast_table_xarray,
        field_name=nwp_model_utils.DEWPOINT_2METRE_NAME
    )
    temperature_matrix_kelvins = nwp_model_utils.get_field(
        nwp_forecast_table_xarray=nwp_forecast_table_xarray,
        field_name=nwp_model_utils.TEMPERATURE_2METRE_NAME
    )
    dewp_depression_matrix_kelvins = (
        temperature_matrix_kelvins - dewpoint_matrix_kelvins
    )

    k = numpy.where(
        nwp_forecast_table_xarray.coords[nwp_model_utils.FIELD_DIM].values ==
        nwp_model_utils.DEWPOINT_2METRE_NAME
    )[0][0]

    data_matrix = nwp_forecast_table_xarray[nwp_model_utils.DATA_KEY].values
    data_matrix[..., k] = dewp_depression_matrix_kelvins

    return nwp_forecast_table_xarray.assign({
        nwp_model_utils.DATA_KEY: (
            nwp_forecast_table_xarray[nwp_model_utils.DATA_KEY].dims,
            data_matrix
        )
    })


def _convert_2m_temp_to_celsius(nwp_forecast_table_xarray):
    """Converts 2-metre NWP temperatures from Kelvins to Celsius.

    :param nwp_forecast_table_xarray: xarray table in format returned by
        `interp_nwp_model_io.read_file`.
    :return: nwp_forecast_table_xarray: Same, except that temperatures have been
        converted to Celsius.
    """

    k = numpy.where(
        nwp_forecast_table_xarray.coords[nwp_model_utils.FIELD_DIM].values ==
        nwp_model_utils.TEMPERATURE_2METRE_NAME
    )[0][0]

    data_matrix = nwp_forecast_table_xarray[nwp_model_utils.DATA_KEY].values
    data_matrix[..., k] = temperature_conv.kelvins_to_celsius(
        data_matrix[..., k]
    )

    return nwp_forecast_table_xarray.assign({
        nwp_model_utils.DATA_KEY: (
            nwp_forecast_table_xarray[nwp_model_utils.DATA_KEY].dims,
            data_matrix
        )
    })


def _convert_10m_gust_speed_to_factor(nwp_forecast_table_xarray):
    """Converts 10-metre NWP gust speeds to gust factors.

    *** Actually, gust factors minus one. ***

    :param nwp_forecast_table_xarray: xarray table in format returned by
        `interp_nwp_model_io.read_file`.
    :return: nwp_forecast_table_xarray: Same, except that gust speeds have been
        replaced with gust factors.
    """

    k_inds = numpy.where(
        nwp_forecast_table_xarray.coords[nwp_model_utils.FIELD_DIM].values ==
        nwp_model_utils.WIND_GUST_10METRE_NAME
    )[0]

    if len(k_inds) == 0:
        return nwp_forecast_table_xarray

    u_wind_matrix_m_s01 = nwp_model_utils.get_field(
        nwp_forecast_table_xarray=nwp_forecast_table_xarray,
        field_name=nwp_model_utils.U_WIND_10METRE_NAME
    )
    v_wind_matrix_m_s01 = nwp_model_utils.get_field(
        nwp_forecast_table_xarray=nwp_forecast_table_xarray,
        field_name=nwp_model_utils.V_WIND_10METRE_NAME
    )
    sustained_speed_matrix_m_s01 = numpy.sqrt(
        u_wind_matrix_m_s01 ** 2 + v_wind_matrix_m_s01 ** 2
    )
    gust_speed_matrix_m_s01 = nwp_model_utils.get_field(
        nwp_forecast_table_xarray=nwp_forecast_table_xarray,
        field_name=nwp_model_utils.WIND_GUST_10METRE_NAME
    )

    data_matrix = nwp_forecast_table_xarray[nwp_model_utils.DATA_KEY].values
    k = k_inds[0]
    data_matrix[..., k] = (
        gust_speed_matrix_m_s01 / sustained_speed_matrix_m_s01 - 1.
    )

    return nwp_forecast_table_xarray.assign({
        nwp_model_utils.DATA_KEY: (
            nwp_forecast_table_xarray[nwp_model_utils.DATA_KEY].dims,
            data_matrix
        )
    })


def _find_best_lead_times(nwp_forecast_file_names, desired_lead_time_hours):
    """Finds best lead times -- i.e., those closest to the desired lead time.

    :param nwp_forecast_file_names: 1-D list of paths to NWP-forecast files,
        readable by `interp_nwp_model_io.read_file`.  Every file should come
        from the same NWP model and init time -- just different lead times.
    :param desired_lead_time_hours: Desired lead time, i.e., target lead time
        for interpolation.
    :return: nwp_forecast_file_names: If desired lead time was found, this is a
        length-1 list of files to use.  If only nearby lead times were found,
        this is a length-2 list of paths to files for use in interpolation.
        Otherwise, this is an empty list.
    """

    lead_times_hours = numpy.array([
        interp_nwp_model_io.file_name_to_forecast_hour(f)
        for f in nwp_forecast_file_names
    ], dtype=int)

    good_indices = numpy.where(lead_times_hours == desired_lead_time_hours)[0]
    if len(good_indices) > 0:
        return [nwp_forecast_file_names[good_indices[0]]]

    earlier_lead_times_hours = lead_times_hours[
        lead_times_hours < desired_lead_time_hours
    ]
    earlier_lead_times_hours = earlier_lead_times_hours[
        numpy.absolute(earlier_lead_times_hours - desired_lead_time_hours) <=
        MAX_INTERPOLATION_TIME_HOURS
    ]

    if len(earlier_lead_times_hours) > 0:
        earlier_lead_time_hours = earlier_lead_times_hours[
            numpy.argmin(numpy.absolute(
                earlier_lead_times_hours - desired_lead_time_hours
            ))
        ]
    else:
        earlier_lead_time_hours = None

    if earlier_lead_time_hours is None:
        return []

    later_lead_times_hours = lead_times_hours[
        lead_times_hours > desired_lead_time_hours
    ]
    later_lead_times_hours = later_lead_times_hours[
        numpy.absolute(later_lead_times_hours - desired_lead_time_hours) <=
        MAX_INTERPOLATION_TIME_HOURS
    ]

    if len(later_lead_times_hours) > 0:
        later_lead_time_hours = later_lead_times_hours[
            numpy.argmin(numpy.absolute(
                later_lead_times_hours - desired_lead_time_hours
            ))
        ]
    else:
        later_lead_time_hours = None

    if later_lead_time_hours is None:
        return []

    first_idx = numpy.where(lead_times_hours == earlier_lead_time_hours)[0][0]
    second_idx = numpy.where(lead_times_hours == later_lead_time_hours)[0][0]

    return [
        nwp_forecast_file_names[first_idx],
        nwp_forecast_file_names[second_idx]
    ]


def _interp_predictors_by_lead_time(predictor_matrix, source_lead_times_hours,
                                    target_lead_times_hours):
    """Interpolates predictors to fill missing lead times.

    M = number of rows in grid
    N = number of columns in grid
    S = number of source lead times
    T = number of target lead times
    P = number of predictor variables

    :param predictor_matrix: M-by-N-by-S-by-P numpy array of predictor values.
    :param source_lead_times_hours: length-S numpy array of source lead times.
    :param target_lead_times_hours: length-T numpy array of target lead times.
    :return: predictor_matrix: M-by-N-by-T-by-P numpy array of predictor values.
    """

    num_target_lead_times = len(target_lead_times_hours)
    target_to_source_index = numpy.full(num_target_lead_times, -1, dtype=int)

    for t in range(num_target_lead_times):
        if target_lead_times_hours[t] not in source_lead_times_hours:
            continue

        target_to_source_index[t] = numpy.where(
            source_lead_times_hours == target_lead_times_hours[t]
        )[0][0]

    num_predictors = predictor_matrix.shape[-1]
    these_dims = (
        predictor_matrix.shape[0], predictor_matrix.shape[1],
        num_target_lead_times, num_predictors
    )
    new_predictor_matrix = numpy.full(these_dims, numpy.nan)

    for p in range(num_predictors):
        missing_target_time_flags = numpy.full(
            num_target_lead_times, True, dtype=bool
        )

        for t in range(num_target_lead_times):
            if target_to_source_index[t] == -1:
                continue

            this_predictor_matrix = predictor_matrix[
                ..., target_to_source_index[t], p
            ]
            missing_target_time_flags[t] = numpy.all(numpy.isnan(
                this_predictor_matrix
            ))

            if missing_target_time_flags[t]:
                continue

            new_predictor_matrix[..., t, p] = this_predictor_matrix

        missing_target_time_indices = numpy.where(missing_target_time_flags)[0]
        missing_source_time_flags = numpy.all(
            numpy.isnan(predictor_matrix[..., p]),
            axis=(0, 1)
        )
        filled_source_time_indices = numpy.where(
            numpy.invert(missing_source_time_flags)
        )[0]

        if len(filled_source_time_indices) == 0:
            continue

        missing_target_time_indices = [
            t for t in missing_target_time_indices
            if target_lead_times_hours[t] >= numpy.min(
                source_lead_times_hours[filled_source_time_indices]
            )
        ]
        missing_target_time_indices = [
            t for t in missing_target_time_indices
            if target_lead_times_hours[t] <= numpy.max(
                source_lead_times_hours[filled_source_time_indices]
            )
        ]
        missing_target_time_indices = numpy.array(
            missing_target_time_indices, dtype=int
        )

        if len(missing_target_time_indices) == 0:
            continue

        interp_object = interp1d(
            x=source_lead_times_hours[filled_source_time_indices],
            y=predictor_matrix[..., filled_source_time_indices, p],
            axis=2,
            kind='linear',
            bounds_error=True,
            assume_sorted=True
        )

        new_predictor_matrix[
            ..., missing_target_time_indices, p
        ] = interp_object(target_lead_times_hours[missing_target_time_indices])

    return new_predictor_matrix


def _init_predictor_matrices_1example(
        nwp_model_names, nwp_model_to_field_names, num_nwp_lead_times,
        patch_location_dict):
    """Initializes predictor matrices for one example.

    :param nwp_model_names: 1-D list with names of NWP models.
    :param nwp_model_to_field_names: Dictionary.  For details, see documentation
        for `data_generator`.
    :param num_nwp_lead_times: Number of lead times.
    :param patch_location_dict: Dictionary produced by
        `misc_utils.determine_patch_locations`.  If you are training with the
        full grid (not the patchwise approach), make this None.
    :return: predictor_matrices_2pt5km: 1-D list of numpy arrays for 2.5-km
        resolution.  One array per 2.5-km model.  If there are no 2.5-km models,
        this is None instead of a list.
    :return: predictor_matrices_10km: Same but for 10-km models.
    :return: predictor_matrices_20km: Same but for 20-km models.
    :return: predictor_matrices_40km: Same but for 40-km models.
    """

    downsampling_factors = numpy.array([
        nwp_model_utils.model_to_nbm_downsampling_factor(m)
        for m in nwp_model_names
    ], dtype=int)

    model_indices = numpy.where(downsampling_factors == 1)[0]
    num_rows, num_columns = get_grid_dimensions(
        grid_spacing_km=2.5,
        patch_location_dict=patch_location_dict
    )

    if len(model_indices) == 0:
        predictor_matrices_2pt5km = None
    else:
        first_dim = (num_rows, num_columns, num_nwp_lead_times)

        predictor_matrices_2pt5km = [
            numpy.full(
                first_dim + (len(nwp_model_to_field_names[nwp_model_names[k]]),),
                numpy.nan
            )
            for k in model_indices
        ]

    model_indices = numpy.where(downsampling_factors == 4)[0]
    num_rows, num_columns = get_grid_dimensions(
        grid_spacing_km=10.,
        patch_location_dict=patch_location_dict
    )

    if len(model_indices) == 0:
        predictor_matrices_10km = None
    else:
        first_dim = (num_rows, num_columns, num_nwp_lead_times)

        predictor_matrices_10km = [
            numpy.full(
                first_dim + (len(nwp_model_to_field_names[nwp_model_names[k]]),),
                numpy.nan
            )
            for k in model_indices
        ]

    model_indices = numpy.where(downsampling_factors == 8)[0]
    num_rows, num_columns = get_grid_dimensions(
        grid_spacing_km=20.,
        patch_location_dict=patch_location_dict
    )

    if len(model_indices) == 0:
        predictor_matrices_20km = None
    else:
        first_dim = (num_rows, num_columns, num_nwp_lead_times)

        predictor_matrices_20km = [
            numpy.full(
                first_dim + (len(nwp_model_to_field_names[nwp_model_names[k]]),),
                numpy.nan
            )
            for k in model_indices
        ]

    model_indices = numpy.where(downsampling_factors == 16)[0]
    num_rows, num_columns = get_grid_dimensions(
        grid_spacing_km=40.,
        patch_location_dict=patch_location_dict
    )

    if len(model_indices) == 0:
        predictor_matrices_40km = None
    else:
        first_dim = (num_rows, num_columns, num_nwp_lead_times)

        predictor_matrices_40km = [
            numpy.full(
                first_dim + (len(nwp_model_to_field_names[nwp_model_names[k]]),),
                numpy.nan
            )
            for k in model_indices
        ]

    return (
        predictor_matrices_2pt5km, predictor_matrices_10km,
        predictor_matrices_20km, predictor_matrices_40km
    )


def _find_predictors_1example_1model_rigid(
        init_time_unix_sec, nwp_model_name, nwp_lead_times_hours,
        nwp_directory_name):
    """Finds predictor files for one example and one NWP model.

    :param init_time_unix_sec: Forecast-initialization time.
    :param nwp_model_name: Name of desired NWP model.
    :param nwp_lead_times_hours: 1-D numpy array of desired lead times from NWP
        model.
    :param nwp_directory_name: Path to directory with NWP data.  Files therein
        will be found by `interp_nwp_model_io.find_file` and read by
        `interp_nwp_model_io.read_file`.
    :return: nwp_forecast_file_names: 1-D list of file paths, one for each lead
        time.  These are readable by `interp_nwp_model_io.read_file`.  This may
        also be None.
    """

    nwp_forecast_file_names = []
    num_lead_times = len(nwp_lead_times_hours)

    for j in range(num_lead_times):
        if nwp_model_name == nwp_model_utils.RAP_MODEL_NAME:
            adjusted_init_time_unix_sec = (
                init_time_unix_sec + 3 * HOURS_TO_SECONDS
            )
            adjusted_lead_time_hours = nwp_lead_times_hours[j] - 3
        else:
            # TODO(thunderhoser): This HACK deals with the fact that, although
            # some models have an init-time interval of 12 hours, I use an
            # init-time interval of 6 hours for training.
            try:
                nwp_model_utils.check_init_time(
                    init_time_unix_sec=init_time_unix_sec,
                    model_name=nwp_model_name
                )
            except:
                return None

            adjusted_init_time_unix_sec = init_time_unix_sec + 0
            adjusted_lead_time_hours = nwp_lead_times_hours[j] + 0

        all_lead_times_hours = nwp_model_utils.model_to_forecast_hours(
            model_name=nwp_model_name,
            init_time_unix_sec=adjusted_init_time_unix_sec
        )

        these_file_names = [
            interp_nwp_model_io.find_file(
                directory_name=nwp_directory_name,
                init_time_unix_sec=adjusted_init_time_unix_sec,
                forecast_hour=l,
                model_name=nwp_model_name,
                raise_error_if_missing=False
            )
            for l in all_lead_times_hours
        ]
        these_file_names = [f for f in these_file_names if os.path.isfile(f)]

        nwp_forecast_file_names += _find_best_lead_times(
            nwp_forecast_file_names=these_file_names,
            desired_lead_time_hours=adjusted_lead_time_hours
        )

    if len(nwp_forecast_file_names) > 0:
        pathless_forecast_file_names = [
            os.path.split(f)[1] for f in nwp_forecast_file_names
        ]

        print((
            'Informational message (NOT AN ERROR):\n'
            'Desired NWP lead times: {0:s}\n'
            'NWP files to be used: {1:s}'
        ).format(
            str(nwp_lead_times_hours),
            str(pathless_forecast_file_names)
        ))

        return nwp_forecast_file_names

    return None


def _find_predictors_1example_1model(
        init_time_unix_sec, nwp_model_name, nwp_lead_times_hours,
        nwp_directory_name, backup_nwp_model_name,
        backup_nwp_directory_name, rigid_flag):
    """Flexible version of `_find_predictors_1example_1model_rigid`.

    :param init_time_unix_sec: See documentation for
        `_find_predictors_1example_1model_rigid`.
    :param nwp_model_name: Same.
    :param nwp_lead_times_hours: Same.
    :param nwp_directory_name: Same.
    :param backup_nwp_model_name: Same as `nwp_model_name` but for backup model.
    :param backup_nwp_directory_name: Same as `nwp_directory_name` but for
        backup model.
    :param rigid_flag: See documentation for `read_predictors_one_example`.
    :return: nwp_forecast_file_names: See documentation for
        `_find_predictors_1example_1model_rigid`.
    """

    try_init_time_offsets_hours = numpy.array(
        [0, 6, 0,
         12, 6,
         18, 12,
         24, 18, 24],
        dtype=int
    )

    try_init_times_unix_sec = (
        init_time_unix_sec - HOURS_TO_SECONDS * try_init_time_offsets_hours
    )

    try_nwp_model_names = [
        nwp_model_name, nwp_model_name, backup_nwp_model_name,
        nwp_model_name, backup_nwp_model_name,
        nwp_model_name, backup_nwp_model_name,
        nwp_model_name, backup_nwp_model_name, backup_nwp_model_name
    ]
    try_nwp_directory_names = [
        nwp_directory_name, nwp_directory_name, backup_nwp_directory_name,
        nwp_directory_name, backup_nwp_directory_name,
        nwp_directory_name, backup_nwp_directory_name,
        nwp_directory_name, backup_nwp_directory_name, backup_nwp_directory_name
    ]

    num_tries = len(try_init_times_unix_sec)
    nwp_forecast_file_names = None

    for i in range(num_tries):
        if i == 1 and rigid_flag:
            break

        nwp_forecast_file_names = _find_predictors_1example_1model_rigid(
            init_time_unix_sec=try_init_times_unix_sec[i],
            nwp_model_name=try_nwp_model_names[i],
            nwp_lead_times_hours=
            nwp_lead_times_hours + try_init_time_offsets_hours[i],
            nwp_directory_name=try_nwp_directory_names[i]
        )

        if nwp_forecast_file_names is None:
            pass

            # print((
            #     'Could NOT find predictors for init time {0:s} '
            #     '({1:d} hours before desired) and NWP model "{2:s}"'
            # ).format(
            #     time_conversion.unix_sec_to_string(
            #         try_init_times_unix_sec[i], '%Y-%m-%d-%H'
            #     ),
            #     try_init_time_offsets_hours[i],
            #     try_nwp_model_names[i]
            # ))
        else:
            found_lead_times_hours = numpy.array([
                interp_nwp_model_io.file_name_to_forecast_hour(f)
                for f in nwp_forecast_file_names
            ], dtype=int)

            print((
                'Wanted predictors for NWP model "{0:s}" at init time {1:s} '
                'and lead times {2:s} hours.  FOUND predictors for model '
                '"{3:s}" at init time {4:s} and lead times {5:s} hours.'
            ).format(
                nwp_model_name,
                time_conversion.unix_sec_to_string(
                    init_time_unix_sec, '%Y-%m-%d-%H'
                ),
                str(nwp_lead_times_hours),
                try_nwp_model_names[i],
                time_conversion.unix_sec_to_string(
                    try_init_times_unix_sec[i], '%Y-%m-%d-%H'
                ),
                str(found_lead_times_hours)
            ))

            break

    return nwp_forecast_file_names


def _read_predictors_1example_1model(
        nwp_forecast_file_names, desired_nwp_model_name,
        desired_valid_times_unix_sec, field_names, patch_location_dict,
        nwp_norm_param_table_xarray, use_quantile_norm):
    """Reads predictors for one example and one NWP model.

    M = number of rows in grid for desired NWP model
    N = number of columns in grid for desired NWP model
    V = number of desired valid times
    F = number of fields

    :param nwp_forecast_file_names: 1-D list of file paths, one for each lead
        time.  These are readable by `interp_nwp_model_io.read_file`.
    :param desired_nwp_model_name: Name of desired NWP model.
    :param desired_valid_times_unix_sec: length-V numpy array of desired valid
        times.
    :param field_names: length-F list of field names.
    :param patch_location_dict: See documentation for
        `read_predictors_one_example`.
    :param nwp_norm_param_table_xarray: Same.
    :param use_quantile_norm: Same.
    :return: predictor_matrix: M-by-N-by-V-by-F numpy array of predictor values.
    """

    # Extract metadata from input files.
    nwp_model_names = [
        interp_nwp_model_io.file_name_to_model_name(f)
        for f in nwp_forecast_file_names
    ]

    assert len(set(nwp_model_names)) == 1
    nwp_model_name = nwp_model_names[0]

    init_times_unix_sec = numpy.array([
        interp_nwp_model_io.file_name_to_init_time(f)
        for f in nwp_forecast_file_names
    ], dtype=int)

    assert len(numpy.unique(init_times_unix_sec)) == 1
    init_time_unix_sec = init_times_unix_sec[0]

    lead_times_hours = numpy.array([
        interp_nwp_model_io.file_name_to_forecast_hour(f)
        for f in nwp_forecast_file_names
    ], dtype=int)

    _, sort_indices = numpy.unique(lead_times_hours, return_index=True)
    lead_times_hours = lead_times_hours[sort_indices]
    nwp_forecast_file_names = [nwp_forecast_file_names[k] for k in sort_indices]

    valid_times_unix_sec = (
        init_time_unix_sec + HOURS_TO_SECONDS * lead_times_hours
    )

    # Determine whether spatial interpolation will be needed.
    downsampling_factor = (
        nwp_model_utils.model_to_nbm_downsampling_factor(nwp_model_name)
    )
    desired_downsampling_factor = (
        nwp_model_utils.model_to_nbm_downsampling_factor(desired_nwp_model_name)
    )

    # Read data.
    num_lead_times = len(lead_times_hours)
    predictor_matrix = None

    for j in range(num_lead_times):
        print('Reading data from: "{0:s}"...'.format(
            nwp_forecast_file_names[j]
        ))
        nwp_forecast_table_xarray = interp_nwp_model_io.read_file(
            nwp_forecast_file_names[j]
        )
        nwp_forecast_table_xarray = nwp_model_utils.subset_by_field(
            nwp_forecast_table_xarray=nwp_forecast_table_xarray,
            desired_field_names=field_names
        )

        if patch_location_dict is not None:
            pld = patch_location_dict

            if downsampling_factor == 1:
                i_start = pld[misc_utils.ROW_LIMITS_2PT5KM_KEY][0]
                i_end = pld[misc_utils.ROW_LIMITS_2PT5KM_KEY][1]
                j_start = pld[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][0]
                j_end = pld[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][1]
            elif downsampling_factor == 4:
                i_start = pld[misc_utils.ROW_LIMITS_10KM_KEY][0]
                i_end = pld[misc_utils.ROW_LIMITS_10KM_KEY][1]
                j_start = pld[misc_utils.COLUMN_LIMITS_10KM_KEY][0]
                j_end = pld[misc_utils.COLUMN_LIMITS_10KM_KEY][1]
            elif downsampling_factor == 8:
                i_start = pld[misc_utils.ROW_LIMITS_20KM_KEY][0]
                i_end = pld[misc_utils.ROW_LIMITS_20KM_KEY][1]
                j_start = pld[misc_utils.COLUMN_LIMITS_20KM_KEY][0]
                j_end = pld[misc_utils.COLUMN_LIMITS_20KM_KEY][1]
            else:
                i_start = pld[misc_utils.ROW_LIMITS_40KM_KEY][0]
                i_end = pld[misc_utils.ROW_LIMITS_40KM_KEY][1]
                j_start = pld[misc_utils.COLUMN_LIMITS_40KM_KEY][0]
                j_end = pld[misc_utils.COLUMN_LIMITS_40KM_KEY][1]

            nwp_forecast_table_xarray = nwp_model_utils.subset_by_row(
                nwp_forecast_table_xarray=nwp_forecast_table_xarray,
                desired_row_indices=numpy.linspace(
                    i_start, i_end, num=i_end - i_start + 1, dtype=int
                )
            )

            nwp_forecast_table_xarray = nwp_model_utils.subset_by_column(
                nwp_forecast_table_xarray=nwp_forecast_table_xarray,
                desired_column_indices=numpy.linspace(
                    j_start, j_end, num=j_end - j_start + 1, dtype=int
                )
            )

        if nwp_norm_param_table_xarray is not None:
            print('Normalizing predictor variables to z-scores...')
            nwp_forecast_table_xarray = (
                normalization.normalize_nwp_data(
                    nwp_forecast_table_xarray=nwp_forecast_table_xarray,
                    norm_param_table_xarray=nwp_norm_param_table_xarray,
                    use_quantile_norm=use_quantile_norm
                )
            )

        if downsampling_factor != desired_downsampling_factor:
            print('Downsampling factor = {0:d}'.format(downsampling_factor))
            print('Desired downsampling factor = {0:d}'.format(desired_downsampling_factor))

            nwp_forecast_table_xarray = nwp_model_utils.interp_data_to_nbm_grid(
                nwp_forecast_table_xarray=nwp_forecast_table_xarray,
                model_name=desired_nwp_model_name,  # TODO: change in master branch.
                use_nearest_neigh=False,
                interp_to_full_resolution=False,
                proj_object=nbm_utils.NBM_PROJECTION_OBJECT
            )

            print(nwp_forecast_table_xarray)

        nwpft = nwp_forecast_table_xarray
        this_predictor_matrix = nwpft[nwp_model_utils.DATA_KEY].values[0, ...]

        if predictor_matrix is None:
            these_dims = (
                this_predictor_matrix.shape[0], this_predictor_matrix.shape[1],
                num_lead_times, this_predictor_matrix.shape[2]
            )
            predictor_matrix = numpy.full(these_dims, numpy.nan)

        predictor_matrix[..., j, :] = this_predictor_matrix

    return _interp_predictors_by_lead_time(
        predictor_matrix=predictor_matrix,
        source_lead_times_hours=valid_times_unix_sec - init_time_unix_sec,
        target_lead_times_hours=
        desired_valid_times_unix_sec - init_time_unix_sec
    )


def get_grid_dimensions(grid_spacing_km, patch_location_dict):
    """Determines grid dimensions for neural network.

    :param grid_spacing_km: Grid spacing (must be 2.5, 10, 20, or 40 km).
    :param patch_location_dict: Dictionary produced by
        `misc_utils.determine_patch_locations`.  If you are training with the
        full grid (not the patchwise approach), make this None.
    :return: num_rows: Number of rows in grid.
    :return: num_columns: Number of columns in grid.
    """

    k = numpy.argmin(
        numpy.absolute(grid_spacing_km - POSSIBLE_GRID_SPACINGS_KM)
    )

    if k == 0:
        if patch_location_dict is None:
            num_rows, num_columns = nwp_model_utils.model_to_nbm_grid_size(
                nwp_model_utils.HRRR_MODEL_NAME
            )
        else:
            num_rows = numpy.diff(
                patch_location_dict[misc_utils.ROW_LIMITS_2PT5KM_KEY]
            )[0] + 1

            num_columns = numpy.diff(
                patch_location_dict[misc_utils.COLUMN_LIMITS_2PT5KM_KEY]
            )[0] + 1

        return num_rows, num_columns

    if k == 1:
        if patch_location_dict is None:
            num_rows, num_columns = nwp_model_utils.model_to_nbm_grid_size(
                nwp_model_utils.RAP_MODEL_NAME
            )
        else:
            num_rows = numpy.diff(
                patch_location_dict[misc_utils.ROW_LIMITS_10KM_KEY]
            )[0] + 1

            num_columns = numpy.diff(
                patch_location_dict[misc_utils.COLUMN_LIMITS_10KM_KEY]
            )[0] + 1

        return num_rows, num_columns

    if k == 2:
        if patch_location_dict is None:
            num_rows, num_columns = nwp_model_utils.model_to_nbm_grid_size(
                nwp_model_utils.GFS_MODEL_NAME
            )
        else:
            num_rows = numpy.diff(
                patch_location_dict[misc_utils.ROW_LIMITS_20KM_KEY]
            )[0] + 1

            num_columns = numpy.diff(
                patch_location_dict[misc_utils.COLUMN_LIMITS_20KM_KEY]
            )[0] + 1

        return num_rows, num_columns

    if patch_location_dict is None:
        num_rows, num_columns = nwp_model_utils.model_to_nbm_grid_size(
            nwp_model_utils.GEFS_MODEL_NAME
        )
    else:
        num_rows = numpy.diff(
            patch_location_dict[misc_utils.ROW_LIMITS_40KM_KEY]
        )[0] + 1

        num_columns = numpy.diff(
            patch_location_dict[misc_utils.COLUMN_LIMITS_40KM_KEY]
        )[0] + 1

    return num_rows, num_columns


def read_residual_baseline_one_example(
        init_time_unix_sec, nwp_model_name, nwp_lead_time_hours,
        nwp_directory_name, target_field_names, patch_location_dict,
        predict_dewpoint_depression, predict_gust_factor):
    """Reads residual baseline for one example.

    M = number of rows in NBM grid (2.5-km target grid)
    N = number of columns in NBM grid (2.5-km target grid)
    F = number of target fields

    Should be used only if the NN is doing residual prediction, i.e., predicting
    the departure between the URMA truth and an NWP forecast.

    :param init_time_unix_sec: Forecast-initialization time.
    :param nwp_model_name: Name of NWP model used to create residual baseline.
    :param nwp_lead_time_hours: NWP lead time used to create residual baseline.
    :param nwp_directory_name: Path to NWP data.  Relevant files therein will be
        found by `interp_nwp_model_io.find_file`.
    :param target_field_names: length-F list with names of target fields.  Each
        must be accepted by `urma_utils.check_field_name`.
    :param patch_location_dict: Dictionary produced by
        `misc_utils.determine_patch_locations`.  If you are training with the
        full grid (not the patchwise approach), make this None.
    :param predict_dewpoint_depression: Boolean flag.  If True, the NN is
        predicting dewpoint depression instead of dewpoint.
    :param predict_gust_factor: Boolean flag.  If True, the NN is predicting
        gust factor instead of gust speed.
    :return: residual_baseline_matrix: M-by-N-by-F numpy array of baseline
        predictions.  These are all unnormalized.
    """

    target_field_to_baseline_nwp_field = {
        urma_utils.TEMPERATURE_2METRE_NAME:
            nwp_model_utils.TEMPERATURE_2METRE_NAME,
        urma_utils.DEWPOINT_2METRE_NAME: nwp_model_utils.DEWPOINT_2METRE_NAME,
        urma_utils.U_WIND_10METRE_NAME: nwp_model_utils.U_WIND_10METRE_NAME,
        urma_utils.V_WIND_10METRE_NAME: nwp_model_utils.V_WIND_10METRE_NAME,
        urma_utils.WIND_GUST_10METRE_NAME:
            nwp_model_utils.WIND_GUST_10METRE_NAME
    }

    input_file_name = interp_nwp_model_io.find_file(
        directory_name=nwp_directory_name,
        init_time_unix_sec=init_time_unix_sec,
        forecast_hour=nwp_lead_time_hours,
        model_name=nwp_model_name,
        raise_error_if_missing=False
    )

    if not os.path.isfile(input_file_name):
        warning_string = (
            'POTENTIAL ERROR: Could not find file expected at: "{0:s}".  This '
            'is needed for residual baseline.'
        ).format(input_file_name)

        warnings.warn(warning_string)
        return None

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    nwp_forecast_table_xarray = interp_nwp_model_io.read_file(input_file_name)

    if (
            nwp_model_utils.WIND_GUST_10METRE_NAME not in
            nwp_forecast_table_xarray.coords[nwp_model_utils.FIELD_DIM].values
    ):
        nwpft = nwp_forecast_table_xarray

        u = numpy.where(
            nwpft.coords[nwp_model_utils.FIELD_DIM].values ==
            nwp_model_utils.U_WIND_10METRE_NAME
        )[0][0]
        v = numpy.where(
            nwpft.coords[nwp_model_utils.FIELD_DIM].values ==
            nwp_model_utils.V_WIND_10METRE_NAME
        )[0][0]

        gust_matrix = numpy.sqrt(
            nwpft[nwp_model_utils.DATA_KEY].values[..., u] ** 2 +
            nwpft[nwp_model_utils.DATA_KEY].values[..., v] ** 2
        )
        data_matrix = numpy.concatenate([
            nwpft[nwp_model_utils.DATA_KEY].values,
            numpy.expand_dims(gust_matrix, axis=-1)
        ], axis=-1)

        field_names = (
            nwpft.coords[nwp_model_utils.FIELD_DIM].values.tolist() +
            [nwp_model_utils.WIND_GUST_10METRE_NAME]
        )

        nwpft.drop_vars(names=[nwp_model_utils.DATA_KEY])
        nwpft = nwpft.assign_coords({
            nwp_model_utils.FIELD_DIM: field_names
        })

        these_dims = (
            nwp_model_utils.FORECAST_HOUR_DIM, nwp_model_utils.ROW_DIM,
            nwp_model_utils.COLUMN_DIM, nwp_model_utils.FIELD_DIM
        )
        nwpft = nwpft.assign({
            nwp_model_utils.DATA_KEY: (these_dims, data_matrix)
        })

        nwp_forecast_table_xarray = nwpft

    if patch_location_dict is not None:
        i_start = patch_location_dict[misc_utils.ROW_LIMITS_2PT5KM_KEY][0]
        i_end = patch_location_dict[misc_utils.ROW_LIMITS_2PT5KM_KEY][1]
        j_start = patch_location_dict[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][0]
        j_end = patch_location_dict[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][1]

        nwp_forecast_table_xarray = nwp_model_utils.subset_by_row(
            nwp_forecast_table_xarray=nwp_forecast_table_xarray,
            desired_row_indices=numpy.linspace(
                i_start, i_end, num=i_end - i_start + 1, dtype=int
            )
        )

        nwp_forecast_table_xarray = nwp_model_utils.subset_by_column(
            nwp_forecast_table_xarray=nwp_forecast_table_xarray,
            desired_column_indices=numpy.linspace(
                j_start, j_end, num=j_end - j_start + 1, dtype=int
            )
        )

    if predict_dewpoint_depression:
        nwp_forecast_table_xarray = _convert_2m_dewp_to_depression(
            nwp_forecast_table_xarray
        )

    if predict_gust_factor:
        nwp_forecast_table_xarray = _convert_10m_gust_speed_to_factor(
            nwp_forecast_table_xarray
        )

    nwp_forecast_table_xarray = _convert_2m_temp_to_celsius(
        nwp_forecast_table_xarray
    )

    these_matrices = [
        nwp_model_utils.get_field(
            nwp_forecast_table_xarray=nwp_forecast_table_xarray,
            field_name=target_field_to_baseline_nwp_field[f]
        )[0, ...]
        for f in target_field_names
    ]
    residual_baseline_matrix = numpy.stack(these_matrices, axis=-1)

    num_fields = residual_baseline_matrix.shape[-1]
    for k in range(num_fields):
        residual_baseline_matrix[..., k] = misc_utils.fill_nans_by_nn_interp(
            residual_baseline_matrix[..., k]
        )

    return residual_baseline_matrix


def read_predictors_one_example(
        init_time_unix_sec, nwp_model_names, nwp_lead_times_hours,
        nwp_model_to_field_names, nwp_model_to_dir_name,
        nwp_norm_param_table_xarray, use_quantile_norm,
        backup_nwp_model_name, backup_nwp_directory_name, patch_location_dict,
        rigid_flag=False):
    """Reads predictor fields for one example.

    :param init_time_unix_sec: Forecast-initialization time.
    :param nwp_model_names: 1-D list with names of NWP models used to create
        predictors.
    :param nwp_lead_times_hours: See documentation for
        `neural_net.data_generator`.
    :param nwp_model_to_field_names: Same.
    :param nwp_model_to_dir_name: Same.
    :param nwp_norm_param_table_xarray: xarray table with normalization
        parameters for predictor variables.  If you do not want to normalize (or
        if the input directory already contains normalized data), this should be
        None.
    :param use_quantile_norm: See documentation for
        `neural_net.data_generator`.
    :param backup_nwp_model_name: Same.
    :param backup_nwp_directory_name: Same.
    :param patch_location_dict: Dictionary produced by
        `misc_utils.determine_patch_locations`.  If you are training with the
        full grid (not the patchwise approach), make this None.
    :param rigid_flag: Boolean flag.  If True, will not try to fill missing data
        with alternative model runs (i.e., alternative init times).
    :return: predictor_matrix_2pt5km: Same as output from `data_generator` but
        without first axis.
    :return: predictor_matrix_10km: Same as output from `data_generator` but
        without first axis.
    :return: predictor_matrix_20km: Same as output from `data_generator` but
        without first axis.
    :return: predictor_matrix_40km: Same as output from `data_generator` but
        without first axis.
    :return: found_any_predictors: Boolean flag.  If True, at least one output
        matrix contains a real value.  If False, the output matrices are all NaN
        or None.
    :return: found_all_predictors: Boolean flag.  If True, there are no "NaN"s
        in the predictor matrices.
    """

    num_nwp_models = len(nwp_model_names)
    num_nwp_lead_times = len(nwp_lead_times_hours)

    nwp_downsampling_factors = numpy.array([
        nwp_model_utils.model_to_nbm_downsampling_factor(m)
        for m in nwp_model_names
    ], dtype=int)

    (
        predictor_matrices_2pt5km, predictor_matrices_10km,
        predictor_matrices_20km, predictor_matrices_40km
    ) = _init_predictor_matrices_1example(
        nwp_model_names=nwp_model_names,
        nwp_model_to_field_names=nwp_model_to_field_names,
        num_nwp_lead_times=num_nwp_lead_times,
        patch_location_dict=patch_location_dict
    )

    for i in range(num_nwp_models):
        these_forecast_file_names = _find_predictors_1example_1model(
            init_time_unix_sec=init_time_unix_sec,
            nwp_model_name=nwp_model_names[i],
            nwp_lead_times_hours=nwp_lead_times_hours,
            nwp_directory_name=nwp_model_to_dir_name[nwp_model_names[i]],
            backup_nwp_model_name=(
                None if rigid_flag else backup_nwp_model_name
            ),
            backup_nwp_directory_name=(
                None if rigid_flag else backup_nwp_directory_name
            ),
            rigid_flag=rigid_flag
        )

        if these_forecast_file_names is None:
            this_predictor_matrix = None

            missing_lead_time_indices = numpy.linspace(
                0, len(nwp_lead_times_hours) - 1,
                num=len(nwp_lead_times_hours), dtype=int
            )
        else:
            this_predictor_matrix = _read_predictors_1example_1model(
                nwp_forecast_file_names=these_forecast_file_names,
                desired_nwp_model_name=nwp_model_names[i],
                desired_valid_times_unix_sec=
                init_time_unix_sec + HOURS_TO_SECONDS * nwp_lead_times_hours,
                field_names=nwp_model_to_field_names[nwp_model_names[i]],
                patch_location_dict=patch_location_dict,
                nwp_norm_param_table_xarray=nwp_norm_param_table_xarray,
                use_quantile_norm=use_quantile_norm
            )

            missing_lead_time_flags = numpy.all(
                numpy.isnan(this_predictor_matrix), axis=(0, 1, 3)
            )
            missing_lead_time_indices = numpy.where(missing_lead_time_flags)[0]

        if len(missing_lead_time_indices) > 0 and not rigid_flag:
            these_forecast_file_names = (
                _find_predictors_1example_1model(
                    init_time_unix_sec=init_time_unix_sec,
                    nwp_model_name=backup_nwp_model_name,
                    nwp_lead_times_hours=
                    nwp_lead_times_hours[missing_lead_time_indices],
                    nwp_directory_name=backup_nwp_directory_name,
                    backup_nwp_model_name=backup_nwp_model_name,
                    backup_nwp_directory_name=backup_nwp_directory_name,
                    rigid_flag=rigid_flag
                )
            )

            if these_forecast_file_names is not None:
                new_predictor_matrix = _read_predictors_1example_1model(
                    nwp_forecast_file_names=these_forecast_file_names,
                    desired_nwp_model_name=nwp_model_names[i],
                    desired_valid_times_unix_sec=(
                        init_time_unix_sec + HOURS_TO_SECONDS *
                        nwp_lead_times_hours[missing_lead_time_indices]
                    ),
                    field_names=nwp_model_to_field_names[nwp_model_names[i]],
                    patch_location_dict=patch_location_dict,
                    nwp_norm_param_table_xarray=nwp_norm_param_table_xarray,
                    use_quantile_norm=use_quantile_norm
                )

                print('Backup model = {0:s}'.format(backup_nwp_model_name))
                print('Desired model = {0:s}'.format(nwp_model_names[i]))
                if this_predictor_matrix is not None:
                    print('Shape of this_predictor_matrix = {0:s}'.format(str(this_predictor_matrix.shape)))
                print('Shape of new_predictor_matrix = {0:s}'.format(str(new_predictor_matrix.shape)))

                if this_predictor_matrix is None:
                    this_predictor_matrix = new_predictor_matrix
                else:
                    this_predictor_matrix[..., missing_lead_time_indices, :] = (
                        new_predictor_matrix
                    )

        if this_predictor_matrix is None:
            warning_string = (
                'POTENTIAL ERROR: Cannot find any data for NWP model "{0:s}" '
                '(or backup model "{1:s}") at init time {2:s}.  Filling '
                'predictor values with NaN.'
            ).format(
                nwp_model_names[i],
                'None' if backup_nwp_model_name is None
                else backup_nwp_model_name,
                time_conversion.unix_sec_to_string(
                    init_time_unix_sec, '%Y-%m-%d-%H'
                )
            )

            warnings.warn(warning_string)
            continue

        matrix_index = numpy.sum(
            nwp_downsampling_factors[:i] == nwp_downsampling_factors[i]
        )

        if nwp_downsampling_factors[i] == 1:
            predictor_matrices_2pt5km[matrix_index] = this_predictor_matrix + 0.
        elif nwp_downsampling_factors[i] == 4:
            predictor_matrices_10km[matrix_index] = this_predictor_matrix + 0.
        elif nwp_downsampling_factors[i] == 8:
            predictor_matrices_20km[matrix_index] = this_predictor_matrix + 0.
        else:
            predictor_matrices_40km[matrix_index] = this_predictor_matrix + 0.

    found_any_predictors = False
    found_all_predictors = True

    if predictor_matrices_2pt5km is None:
        predictor_matrix_2pt5km = None
    else:
        predictor_matrix_2pt5km = numpy.concatenate(
            predictor_matrices_2pt5km, axis=-1
        )
        nan_matrix = numpy.all(
            numpy.isnan(predictor_matrix_2pt5km), axis=(0, 1)
        )
        found_all_predictors &= not numpy.any(nan_matrix)
        found_any_predictors |= not numpy.all(
            numpy.isnan(predictor_matrix_2pt5km)
        )

    if predictor_matrices_10km is None:
        predictor_matrix_10km = None
    else:
        predictor_matrix_10km = numpy.concatenate(
            predictor_matrices_10km, axis=-1
        )
        nan_matrix = numpy.all(numpy.isnan(predictor_matrix_10km), axis=(0, 1))
        found_all_predictors &= not numpy.any(nan_matrix)
        found_any_predictors |= not numpy.all(
            numpy.isnan(predictor_matrix_10km)
        )

    if predictor_matrices_20km is None:
        predictor_matrix_20km = None
    else:
        predictor_matrix_20km = numpy.concatenate(
            predictor_matrices_20km, axis=-1
        )
        nan_matrix = numpy.all(numpy.isnan(predictor_matrix_20km), axis=(0, 1))
        found_all_predictors &= not numpy.any(nan_matrix)
        found_any_predictors |= not numpy.all(
            numpy.isnan(predictor_matrix_20km)
        )

    if predictor_matrices_40km is None:
        predictor_matrix_40km = None
    else:
        predictor_matrix_40km = numpy.concatenate(
            predictor_matrices_40km, axis=-1
        )
        nan_matrix = numpy.all(numpy.isnan(predictor_matrix_40km), axis=(0, 1))
        found_all_predictors &= not numpy.any(nan_matrix)
        found_any_predictors |= not numpy.all(
            numpy.isnan(predictor_matrix_40km)
        )

    return (
        predictor_matrix_2pt5km, predictor_matrix_10km,
        predictor_matrix_20km, predictor_matrix_40km,
        found_any_predictors, found_all_predictors
    )
