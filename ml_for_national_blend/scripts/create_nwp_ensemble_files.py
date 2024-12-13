"""Creates NWP-ensemble files.

Each output file has the following properties:

- One init time
- One lead time
- Interpolated to the full-resolution (2.5-km) NBM grid
- Contains forecasts of either [1] all predictor variables or [2] the five
  target variables
"""

import copy
import argparse
import warnings
import numpy
import xarray
from ml_for_national_blend.outside_code import time_conversion
from ml_for_national_blend.outside_code import time_periods
from ml_for_national_blend.outside_code import error_checking
from ml_for_national_blend.io import interp_nwp_model_io
from ml_for_national_blend.utils import nbm_utils
from ml_for_national_blend.utils import misc_utils
from ml_for_national_blend.utils import nwp_model_utils
from ml_for_national_blend.machine_learning import nwp_input

# TODO(thunderhoser): When this script calls
# `nwp_input.read_predictors_one_file`, it interpolates between missing lead
# times -- so it is not completely "rigid".  Maybe I want to add the option to
# disallow this interpolation?

# TODO(thunderhoser): Another thing: this script uses data that have already
# been interpolated to the NBM grid (or a downsampled version thereof), but
# in the future I might want to use raw model data.  This raises a storage
# problem, though, because storing both raw and interpolated model data requires
# a lot of disk space.  I have deleted some of the raw data after interpolating
# to the NBM grid, so I would also need to recreate some raw data.

HOURS_TO_SECONDS = 3600
TIME_FORMAT = '%Y-%m-%d-%H'

TARGET_FIELD_NAMES = [
    nwp_model_utils.TEMPERATURE_2METRE_NAME,
    nwp_model_utils.DEWPOINT_2METRE_NAME,
    nwp_model_utils.U_WIND_10METRE_NAME,
    nwp_model_utils.V_WIND_10METRE_NAME,
    nwp_model_utils.WIND_GUST_10METRE_NAME
]

INPUT_DIRS_ARG_NAME = 'input_nwp_dir_names'
NWP_MODELS_ARG_NAME = 'nwp_model_names'
LEAD_TIMES_ARG_NAME = 'lead_times_hours'
TARGETS_ONLY_ARG_NAME = 'targets_only'
FIRST_INIT_TIME_ARG_NAME = 'first_init_time_string'
LAST_INIT_TIME_ARG_NAME = 'last_init_time_string'
OUTPUT_DIR_ARG_NAME = 'output_ensemble_dir_name'

INPUT_DIRS_HELP_STRING = (
    'List of paths to input directories, one per NWP model to be included in '
    'the ensemble.  Within each directory, files will be found by '
    '`interp_nwp_model_io.find_file` and read by '
    '`interp_nwp_model_io.read_file`.'
)
NWP_MODELS_HELP_STRING = (
    'List of model names.  This list must have the same length as {0:s}, and '
    'each model name must be accepted by `nwp_model_utils.check_model_name`.'
).format(
    INPUT_DIRS_HELP_STRING
)
LEAD_TIMES_HELP_STRING = (
    'List of lead times.  This script will process forecasts at every lead '
    'time in the list.'
)
TARGETS_ONLY_HELP_STRING = (
    'Boolean flag.  If 1, output files will contain only the five target '
    'variables.  If 0, the output files will contain all variables.'
)
FIRST_INIT_TIME_HELP_STRING = (
    'This script will process model runs initialized at all times in the '
    'period `{0:s}`...`{1:s}`.  Use the time format "yyyy-mm-dd-HH".'
).format(
    FIRST_INIT_TIME_ARG_NAME, LAST_INIT_TIME_ARG_NAME
)
LAST_INIT_TIME_HELP_STRING = FIRST_INIT_TIME_HELP_STRING
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Files will be written here by '
    '`interp_nwp_model_io.write_file`, to exact locations determined by '
    '`interp_nwp_model_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIRS_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_DIRS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NWP_MODELS_ARG_NAME, type=str, nargs='+', required=True,
    help=NWP_MODELS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LEAD_TIMES_ARG_NAME, type=int, nargs='+', required=True,
    help=LEAD_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGETS_ONLY_ARG_NAME, type=int, required=True,
    help=TARGETS_ONLY_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_INIT_TIME_ARG_NAME, type=str, required=True,
    help=FIRST_INIT_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_INIT_TIME_ARG_NAME, type=str, required=True,
    help=LAST_INIT_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _extract_1model_from_matrices(
        nwp_model_to_field_names, desired_nwp_model_name, targets_only,
        data_matrix_2pt5km, data_matrix_10km, data_matrix_20km,
        data_matrix_40km):
    """Extracts data for one model from set of data matrices.

    M = number of rows in desired model's grid
    N = number of columns in desired model's grid
    F = number of target fields

    :param nwp_model_to_field_names: Dictionary, where each key is the name of
        an NWP model and the corresponding value is a 1-D list, containing
        fields from said model in the dataset.  NWP-model names must be accepted
        by `nwp_model_utils.check_model_name`, and field names must be accepted
        by `nwp_model_utils.check_field_name`.
    :param desired_nwp_model_name: Name of desired model.  Will extract data for
        this model only.
    :param targets_only: Boolean flag.  If True, will ensemble only the five
        target variables.  If False, will ensemble all variables.
    :param data_matrix_2pt5km: See documentation for
        `nwp_input.read_predictors_one_example`.
    :param data_matrix_10km: Same.
    :param data_matrix_20km: Same.
    :param data_matrix_40km: Same.
    :return: desired_model_data_matrix: M-by-N-by-F numpy array of data values.
    """

    nwp_model_names = list(nwp_model_to_field_names.keys())
    nwp_model_names.sort()

    nwp_downsampling_factors = numpy.array([
        nwp_model_utils.model_to_nbm_downsampling_factor(m)
        for m in nwp_model_names
    ], dtype=int)

    i = nwp_model_names.index(desired_nwp_model_name)

    current_field_names = copy.deepcopy(
        nwp_model_to_field_names[desired_nwp_model_name]
    )
    same_ds_factor_indices = numpy.where(
        nwp_downsampling_factors[:i] == nwp_downsampling_factors[i]
    )[0]

    k_start = sum([
        len(nwp_model_to_field_names[nwp_model_names[s]])
        for s in same_ds_factor_indices
    ])
    k_end = k_start + len(current_field_names)

    if nwp_downsampling_factors[i] == 1:
        print((
            'Found {0:d} fields from model {1:s}, at indices {2:d}-{3:d} of '
            'data_matrix_2pt5km!'
        ).format(
            len(current_field_names),
            desired_nwp_model_name,
            k_start,
            k_end - 1
        ))

        desired_model_data_matrix = data_matrix_2pt5km[..., 0, k_start:k_end]
    elif nwp_downsampling_factors[i] == 4:
        print((
            'Found {0:d} fields from model {1:s}, at indices {2:d}-{3:d} of '
            'data_matrix_10km!'
        ).format(
            len(current_field_names),
            desired_nwp_model_name,
            k_start,
            k_end - 1
        ))

        desired_model_data_matrix = data_matrix_10km[..., 0, k_start:k_end]
    elif nwp_downsampling_factors[i] == 8:
        print((
            'Found {0:d} fields from model {1:s}, at indices {2:d}-{3:d} of '
            'data_matrix_20km!'
        ).format(
            len(current_field_names),
            desired_nwp_model_name,
            k_start,
            k_end - 1
        ))

        desired_model_data_matrix = data_matrix_20km[..., 0, k_start:k_end]
    else:
        print((
            'Found {0:d} fields from model {1:s}, at indices {2:d}-{3:d} of '
            'data_matrix_40km!'
        ).format(
            len(current_field_names),
            desired_nwp_model_name,
            k_start,
            k_end - 1
        ))

        desired_model_data_matrix = data_matrix_40km[..., 0, k_start:k_end]

    if targets_only:
        all_field_names = TARGET_FIELD_NAMES

        if nwp_model_utils.WIND_GUST_10METRE_NAME not in current_field_names:
            u = current_field_names.index(nwp_model_utils.U_WIND_10METRE_NAME)
            v = current_field_names.index(nwp_model_utils.V_WIND_10METRE_NAME)
            current_field_names.append(nwp_model_utils.WIND_GUST_10METRE_NAME)

            fake_gust_matrix = numpy.sqrt(
                desired_model_data_matrix[..., u] ** 2 +
                desired_model_data_matrix[..., v] ** 2
            )
            desired_model_data_matrix = numpy.concatenate([
                desired_model_data_matrix,
                numpy.expand_dims(fake_gust_matrix, axis=-1)
            ], axis=-1)
    else:
        all_field_names = nwp_model_utils.ALL_FIELD_NAMES

        for this_field_name in all_field_names:
            if this_field_name in current_field_names:
                continue

            current_field_names.append(this_field_name)

            nan_matrix = numpy.full(
                desired_model_data_matrix[..., 0].shape, numpy.nan
            )
            desired_model_data_matrix = numpy.concatenate([
                desired_model_data_matrix,
                numpy.expand_dims(nan_matrix, axis=-1)
            ], axis=-1)

    sort_indices = numpy.array(
        [current_field_names.index(f) for f in all_field_names],
        dtype=int
    )
    return desired_model_data_matrix[..., sort_indices]


def _create_ensemble_file_1init_1valid(
        init_time_unix_sec, lead_time_hours, targets_only,
        nwp_model_to_dir_name, output_ensemble_dir_name):
    """Creates ensemble file for one init time and one valid time.

    :param init_time_unix_sec: Init time.
    :param lead_time_hours: Lead time.
    :param targets_only: Boolean flag.  If True, will ensemble only the five
        target variables.  If False, will ensemble all variables.
    :param nwp_model_to_dir_name: Dictionary, where each key is the name of an
        NWP model and the corresponding value is the directory path for data
        from said model.  NWP-model names must be accepted by
        `nwp_model_utils.check_model_name`, and within each directory, relevant
        files will be found by `interp_nwp_model_io.find_file`.
    :param output_ensemble_dir_name: See documentation at top of this script.
    """

    if targets_only:
        all_field_names = TARGET_FIELD_NAMES
    else:
        all_field_names = nwp_model_utils.ALL_FIELD_NAMES

    nwp_model_names = list(nwp_model_to_dir_name.keys())
    nwp_model_names.sort()
    nwp_model_to_field_names = dict()

    for this_model_name in nwp_model_names:
        missing_field_names = nwp_model_utils.model_to_maybe_missing_fields(
            this_model_name
        )
        desired_field_names = list(
            set(all_field_names) - set(missing_field_names)
        )
        desired_field_names.sort()

        nwp_model_to_field_names[this_model_name] = desired_field_names

    nwp_downsampling_factors = numpy.array([
        nwp_model_utils.model_to_nbm_downsampling_factor(m)
        for m in nwp_model_names
    ], dtype=int)

    (
        data_matrix_2pt5km,
        data_matrix_10km,
        data_matrix_20km,
        data_matrix_40km,
        found_any_data,
        _
    ) = nwp_input.read_predictors_one_example(
        init_time_unix_sec=init_time_unix_sec,
        nwp_model_names=nwp_model_names,
        nwp_lead_times_hours=numpy.array([lead_time_hours], dtype=int),
        nwp_model_to_field_names=nwp_model_to_field_names,
        nwp_model_to_dir_name=nwp_model_to_dir_name,
        nwp_norm_param_table_xarray=None,
        nwp_resid_norm_param_table_xarray=None,
        use_quantile_norm=False,
        backup_nwp_model_name=None,
        backup_nwp_directory_name=None,
        patch_location_dict=None,
        rigid_flag=True
    )

    if not found_any_data:
        warning_string = (
            'POTENTIAL MAJOR ERROR: Could not find any data for init time '
            '{0:s} and lead time of {1:d} hours.  Tried the following models:'
            '\n{2:s}'
        ).format(
            time_conversion.unix_sec_to_string(init_time_unix_sec, TIME_FORMAT),
            lead_time_hours,
            str(list(nwp_model_to_dir_name.keys()))
        )

        warnings.warn(warning_string)
        return

    num_models = len(nwp_model_names)
    num_rows = data_matrix_2pt5km.shape[0]
    num_columns = data_matrix_2pt5km.shape[1]
    num_fields = len(all_field_names)

    these_dims = (num_rows, num_columns, num_fields, num_models)
    ensemble_data_matrix_2pt5km = numpy.full(these_dims, numpy.nan)

    for i in range(num_models):
        current_data_matrix = _extract_1model_from_matrices(
            nwp_model_to_field_names=nwp_model_to_field_names,
            desired_nwp_model_name=nwp_model_names[i],
            targets_only=targets_only,
            data_matrix_2pt5km=data_matrix_2pt5km,
            data_matrix_10km=data_matrix_10km,
            data_matrix_20km=data_matrix_20km,
            data_matrix_40km=data_matrix_40km
        )

        if nwp_downsampling_factors[i] == 1:
            ensemble_data_matrix_2pt5km[..., i] = current_data_matrix + 0.
            continue

        source_latitude_matrix_deg_n, source_longitude_matrix_deg_e = (
            nbm_utils.read_coords()
        )

        dsf = nwp_downsampling_factors[i]
        source_latitude_matrix_deg_n = (
            source_latitude_matrix_deg_n[::dsf, ::dsf][:-1, :-1]
        )
        source_longitude_matrix_deg_e = (
            source_longitude_matrix_deg_e[::dsf, ::dsf][:-1, :-1]
        )

        num_source_rows = source_latitude_matrix_deg_n.shape[0]
        num_source_columns = source_latitude_matrix_deg_n.shape[1]

        coord_dict = {
            nwp_model_utils.ROW_DIM: numpy.linspace(
                0, num_source_rows - 1, num=num_source_rows, dtype=int
            ),
            nwp_model_utils.COLUMN_DIM: numpy.linspace(
                0, num_source_columns - 1, num=num_source_columns, dtype=int
            ),
            nwp_model_utils.FORECAST_HOUR_DIM: numpy.array(
                [lead_time_hours], dtype=int
            ),
            nwp_model_utils.FIELD_DIM: all_field_names
        }

        these_dims_2d = (nwp_model_utils.ROW_DIM, nwp_model_utils.COLUMN_DIM)
        these_dims_4d = (
            nwp_model_utils.FORECAST_HOUR_DIM, nwp_model_utils.ROW_DIM,
            nwp_model_utils.COLUMN_DIM, nwp_model_utils.FIELD_DIM
        )

        main_data_dict = {
            nwp_model_utils.LATITUDE_KEY: (
                these_dims_2d, source_latitude_matrix_deg_n
            ),
            nwp_model_utils.LONGITUDE_KEY: (
                these_dims_2d, source_longitude_matrix_deg_e
            ),
            nwp_model_utils.DATA_KEY: (
                these_dims_4d, numpy.expand_dims(current_data_matrix, axis=0)
            )
        }

        nwp_forecast_table_xarray = xarray.Dataset(
            data_vars=main_data_dict, coords=coord_dict
        )

        nwp_forecast_table_xarray = nwp_model_utils.interp_data_to_nbm_grid(
            nwp_forecast_table_xarray=nwp_forecast_table_xarray,
            model_name=nwp_model_names[i],
            use_nearest_neigh=False,
            interp_to_full_resolution=True,
            proj_object=nbm_utils.NBM_PROJECTION_OBJECT
        )
        ensemble_data_matrix_2pt5km[..., i] = (
            nwp_forecast_table_xarray[nwp_model_utils.DATA_KEY].values[0, ...]
        )

    if not targets_only:
        ensemble_data_matrix_2pt5km = numpy.nanmean(
            ensemble_data_matrix_2pt5km, axis=-1, keepdims=True
        )

    num_rows = ensemble_data_matrix_2pt5km.shape[0]
    num_columns = ensemble_data_matrix_2pt5km.shape[1]

    coord_dict = {
        nwp_model_utils.ROW_DIM: numpy.linspace(
            0, num_rows - 1, num=num_rows, dtype=int
        ),
        nwp_model_utils.COLUMN_DIM: numpy.linspace(
            0, num_columns - 1, num=num_columns, dtype=int
        ),
        nwp_model_utils.FORECAST_HOUR_DIM: numpy.array(
            [lead_time_hours], dtype=int
        ),
        nwp_model_utils.FIELD_DIM: all_field_names,
        interp_nwp_model_io.ENSEMBLE_MEMBER_DIM: numpy.linspace(
            0, num_models - 1, num=num_models, dtype=int
        )
    }

    these_dims_2d = (nwp_model_utils.ROW_DIM, nwp_model_utils.COLUMN_DIM)
    these_dims_5d = (
        nwp_model_utils.FORECAST_HOUR_DIM, nwp_model_utils.ROW_DIM,
        nwp_model_utils.COLUMN_DIM, nwp_model_utils.FIELD_DIM,
        interp_nwp_model_io.ENSEMBLE_MEMBER_DIM
    )

    nbm_latitude_matrix_deg_n, nbm_longitude_matrix_deg_e = (
        nbm_utils.read_coords()
    )

    main_data_dict = {
        nwp_model_utils.LATITUDE_KEY: (
            these_dims_2d, nbm_latitude_matrix_deg_n
        ),
        nwp_model_utils.LONGITUDE_KEY: (
            these_dims_2d, nbm_longitude_matrix_deg_e
        ),
        nwp_model_utils.DATA_KEY: (
            these_dims_5d,
            numpy.expand_dims(ensemble_data_matrix_2pt5km, axis=0)
        )
    }

    ensemble_forecast_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=coord_dict
    )

    output_file_name = interp_nwp_model_io.find_file(
        directory_name=output_ensemble_dir_name,
        init_time_unix_sec=init_time_unix_sec,
        forecast_hour=lead_time_hours,
        model_name=nwp_model_utils.ENSEMBLE_MODEL_NAME,
        raise_error_if_missing=False
    )

    print('Writing ensemble forecast to: "{0:s}"...'.format(output_file_name))
    interp_nwp_model_io.write_file(
        nwp_forecast_table_xarray=ensemble_forecast_table_xarray,
        netcdf_file_name=output_file_name
    )


def _run(input_nwp_dir_names, nwp_model_names, lead_times_hours, targets_only,
         first_init_time_string, last_init_time_string,
         output_ensemble_dir_name):
    """Creates NWP-ensemble files.

    This is effectively the main method.

    :param input_nwp_dir_names: See documentation at top of this script.
    :param nwp_model_names: Same.
    :param lead_times_hours: Same.
    :param targets_only: Same.
    :param first_init_time_string: Same.
    :param last_init_time_string: Same.
    :param output_ensemble_dir_name: Same.
    """

    # Check input args.
    error_checking.assert_is_greater_numpy_array(lead_times_hours, 0)
    error_checking.assert_equals(
        len(input_nwp_dir_names),
        len(nwp_model_names)
    )

    for n in nwp_model_names:
        nwp_model_utils.check_model_name(n)

    first_init_time_unix_sec = time_conversion.string_to_unix_sec(
        first_init_time_string, TIME_FORMAT
    )
    last_init_time_unix_sec = time_conversion.string_to_unix_sec(
        last_init_time_string, TIME_FORMAT
    )
    init_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_init_time_unix_sec,
        end_time_unix_sec=last_init_time_unix_sec,
        time_interval_sec=6 * HOURS_TO_SECONDS,
        include_endpoint=True
    )

    init_times_unix_sec = misc_utils.remove_unused_days(init_times_unix_sec)

    # Do actual stuff.
    nwp_model_to_dir_name = dict(zip(nwp_model_names, input_nwp_dir_names))

    for this_init_time_unix_sec in init_times_unix_sec:
        for this_lead_time_hours in lead_times_hours:
            _create_ensemble_file_1init_1valid(
                init_time_unix_sec=this_init_time_unix_sec,
                lead_time_hours=this_lead_time_hours,
                targets_only=targets_only,
                nwp_model_to_dir_name=nwp_model_to_dir_name,
                output_ensemble_dir_name=output_ensemble_dir_name
            )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_nwp_dir_names=getattr(INPUT_ARG_OBJECT, INPUT_DIRS_ARG_NAME),
        nwp_model_names=getattr(INPUT_ARG_OBJECT, NWP_MODELS_ARG_NAME),
        lead_times_hours=numpy.array(
            getattr(INPUT_ARG_OBJECT, LEAD_TIMES_ARG_NAME), dtype=int
        ),
        targets_only=bool(getattr(INPUT_ARG_OBJECT, TARGETS_ONLY_ARG_NAME)),
        first_init_time_string=getattr(
            INPUT_ARG_OBJECT, FIRST_INIT_TIME_ARG_NAME
        ),
        last_init_time_string=getattr(
            INPUT_ARG_OBJECT, LAST_INIT_TIME_ARG_NAME
        ),
        output_ensemble_dir_name=getattr(
            INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME
        )
    )
