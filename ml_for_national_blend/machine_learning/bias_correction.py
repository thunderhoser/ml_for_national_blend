"""Training and inference code for bias-correction models.

This module includes both isotonic regression, for bias-correcting the ensemble
mean, and uncertainty calibration, for bias-correcting the ensemble spread.
"""

import time
from multiprocessing import Pool
import dill
import numpy
import xarray
from sklearn.isotonic import IsotonicRegression
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml_for_national_blend.io import prediction_io
from ml_for_national_blend.utils import bias_clustering
from ml_for_national_blend.utils import urma_utils

TOLERANCE = 1e-6
MASK_PIXEL_IF_WEIGHT_BELOW = 0.01
NUM_SLICES_FOR_MULTIPROCESSING = 24
MAX_STDEV_INFLATION_FACTOR = 1000.

EVALUATION_WEIGHT_KEY = 'evaluation_weight'

MODEL_KEY = 'model_object'
CLUSTER_TO_MODEL_KEY = 'cluster_id_to_model_object'
CLUSTER_IDS_KEY = 'cluster_id_matrix'
FIELD_NAMES_KEY = 'field_names'
DO_UNCERTAINTY_CALIB_KEY = 'do_uncertainty_calibration'
DO_IR_BEFORE_UC_KEY = 'do_iso_reg_before_uncertainty_calib'

ALL_KEYS = [
    MODEL_KEY, CLUSTER_TO_MODEL_KEY, CLUSTER_IDS_KEY, FIELD_NAMES_KEY,
    DO_UNCERTAINTY_CALIB_KEY, DO_IR_BEFORE_UC_KEY
]


def __get_slices_for_multiprocessing(cluster_ids):
    """Returns slices for multiprocessing.

    Each "slice" consists of several clusters.

    K = number of slices

    :param cluster_ids: 1-D numpy array of cluster IDs (positive integers).
    :return: slice_to_cluster_ids: Dictionary, where each key is a slice index
        (non-negative integer) and the corresponding value is a list of cluster
        IDs.
    """

    shuffled_cluster_ids = cluster_ids + 0
    numpy.random.shuffle(shuffled_cluster_ids)
    num_clusters = len(shuffled_cluster_ids)

    slice_indices_normalized = numpy.linspace(
        0, 1, num=NUM_SLICES_FOR_MULTIPROCESSING + 1, dtype=float
    )

    start_indices = numpy.round(
        num_clusters * slice_indices_normalized[:-1]
    ).astype(int)

    end_indices = numpy.round(
        num_clusters * slice_indices_normalized[1:]
    ).astype(int)

    slice_to_cluster_ids = dict()

    for i in range(len(start_indices)):
        slice_to_cluster_ids[i] = (
            shuffled_cluster_ids[start_indices[i]:end_indices[i]]
        )

    return slice_to_cluster_ids


def _subset_predictions_to_cluster(
        prediction_tables_xarray, cluster_table_xarray, desired_cluster_id):
    """Subsets predictions by location.

    :param prediction_tables_xarray: See input documentation for
        `train_model_suite`.
    :param cluster_table_xarray: Same.
    :param desired_cluster_id: Will subset predictions to cluster k, where
        k = `desired_cluster_id`.
    :return: new_prediction_tables_xarray: Same as input, except with a smaller
        grid and different evaluation weights.
    """

    # This method automatically handles NaN predictions, because only grid
    # points without NaN predictions (i.e., where NaN is impossible) are
    # assigned to a real cluster (with index > 0).
    assert desired_cluster_id > 0

    cluster_id_matrix = (
        cluster_table_xarray[bias_clustering.CLUSTER_ID_KEY].values[..., 0]
    )
    good_rows, good_columns = numpy.where(
        cluster_id_matrix == desired_cluster_id
    )
    good_rows = numpy.unique(good_rows)
    good_columns = numpy.unique(good_columns)

    first_ptx = prediction_tables_xarray[0]
    eval_weight_submatrix = (
        first_ptx[EVALUATION_WEIGHT_KEY].values[good_rows, :][:, good_columns]
    )
    cluster_id_submatrix = cluster_id_matrix[good_rows, :][:, good_columns]
    eval_weight_submatrix[cluster_id_submatrix != desired_cluster_id] = 0.

    num_tables = len(prediction_tables_xarray)
    new_prediction_tables_xarray = [xarray.Dataset()] * num_tables

    for k in range(num_tables):
        new_prediction_tables_xarray[k] = prediction_tables_xarray[k].isel(
            {prediction_io.ROW_DIM: good_rows}
        )
        new_prediction_tables_xarray[k] = new_prediction_tables_xarray[k].isel(
            {prediction_io.COLUMN_DIM: good_columns}
        )
        new_prediction_tables_xarray[k] = new_prediction_tables_xarray[k].assign({
            EVALUATION_WEIGHT_KEY: (
                new_prediction_tables_xarray[k][EVALUATION_WEIGHT_KEY].dims,
                eval_weight_submatrix
            )
        })

    return new_prediction_tables_xarray


def _train_one_model(prediction_tables_xarray):
    """Trains one model.

    One model corresponds to either [1] one field or [2] one field at one grid
    point.

    :param prediction_tables_xarray: Same as input to `train_model_suite`,
        except containing only the relevant data (i.e., the relevant fields and
        grid points).
    :return: model_object: Trained instance of
        `sklearn.isotonic.IsotonicRegression`.
    """

    eval_weight_matrix = (
        prediction_tables_xarray[0][EVALUATION_WEIGHT_KEY].values
    )
    good_spatial_inds = numpy.where(
        eval_weight_matrix >= MASK_PIXEL_IF_WEIGHT_BELOW
    )
    if len(good_spatial_inds[0]) == 0:
        print('Num training pixels/samples = 0/0')
        return None

    predicted_values = numpy.concatenate([
        ptx[prediction_io.PREDICTION_KEY].values[..., 0, 0][good_spatial_inds]
        for ptx in prediction_tables_xarray
    ])
    target_values = numpy.concatenate([
        ptx[prediction_io.TARGET_KEY].values[..., 0][good_spatial_inds]
        for ptx in prediction_tables_xarray
    ])
    eval_weights = numpy.concatenate([
        ptx[EVALUATION_WEIGHT_KEY].values[good_spatial_inds]
        for ptx in prediction_tables_xarray
    ])

    real_mask = numpy.invert(numpy.logical_or(
        numpy.isnan(predicted_values),
        numpy.isnan(target_values)
    ))
    predicted_values = predicted_values[real_mask]
    target_values = target_values[real_mask]
    eval_weights = eval_weights[real_mask]

    percentile_levels = numpy.linspace(0, 100, num=11, dtype=float)
    print((
        'Num training pixels/samples = {0:d}/{1:d}; '
        'percentiles {2:s} of sample weights = {3:s}'
    ).format(
        len(good_spatial_inds[0]),
        len(predicted_values),
        str(percentile_levels),
        str(numpy.percentile(eval_weights, percentile_levels))
    ))

    # TODO(thunderhoser): Deal with physical constraints here.
    model_object = IsotonicRegression(
        increasing=True, out_of_bounds='clip'
    )
    model_object.fit(
        X=predicted_values, y=target_values, sample_weight=eval_weights
    )
    return model_object


def _train_one_model_per_cluster(prediction_tables_xarray, cluster_table_xarray,
                                 train_for_cluster_ids):
    """Trains one model per cluster, using multiprocessing.

    :param prediction_tables_xarray: See documentation for `train_model_suite`.
    :param cluster_table_xarray: Same.
    :param train_for_cluster_ids: 1-D numpy array of cluster IDs for which to
        train a model.
    :return: cluster_id_to_model_object: See output doc for `train_model_suite`.
        This will be a subset of that final dictionary.
    """

    cluster_id_to_model_object = dict()
    num_clusters = len(train_for_cluster_ids)

    for k in range(num_clusters):
        print((
            'Training bias-correction model for {0:d}th of {1:d} clusters...'
        ).format(
            k + 1, num_clusters
        ))

        these_prediction_tables_xarray = _subset_predictions_to_cluster(
            prediction_tables_xarray=prediction_tables_xarray,
            cluster_table_xarray=cluster_table_xarray,
            desired_cluster_id=train_for_cluster_ids[k]
        )

        cluster_id_to_model_object[train_for_cluster_ids[k]] = (
            _train_one_model(these_prediction_tables_xarray)
        )

    return cluster_id_to_model_object


def train_model_suite(
        prediction_tables_xarray, target_field_name, do_uncertainty_calibration,
        do_iso_reg_before_uncertainty_calib=None,
        do_multiprocessing=True,
        cluster_table_xarray=None):
    """Trains one model suite.

    The model suite will contain either [1] one model per target field or
    [2] one model per target field per cluster.

    M = number of rows in spatial grid
    N = number of columns in spatial grid

    :param prediction_tables_xarray: 1-D list of xarray tables in format
        returned by `prediction_io.read_file`.
    :param target_field_name: Will train models only for this target field
        (name must be accepted by `urma_utils.check_field_name`).
    :param do_uncertainty_calibration: Boolean flag.  If True, this method will
        [1] assume that every "prediction" in `prediction_tables_xarray` is the
        ensemble variance; [2] assume that every "target" in
        `prediction_tables_xarray` is the squared error of the ensemble mean;
        [3] train IR models to adjust the ensemble variance, i.e., to do
        uncertainty calibration.  If False, this method will do standard
        isotonic regression, correcting only the ensemble mean.
    :param do_iso_reg_before_uncertainty_calib:
        [used only if `do_uncertainty_calibration == True`]
        Boolean flag, indicating whether isotonic regression has been done
        before uncertainty calibration.
    :param do_multiprocessing: [used only if `cluster_table_xarray is not None`]
        Boolean flag.  If True, will do multi-threaded processing to make this
        go faster.
    :param cluster_table_xarray: xarray table in format returned by
        `bias_clustering.read_file`.  If you want to train one model for the
        whole domain -- instead of one model per spatial cluster -- just make
        this None.

    :return: model_dict: Dictionary with the following keys.
    model_dict["model_object"]: Trained bias-correction model (instance of
        `sklearn.isotonic.IsotonicRegression`).  If one model per cluster, this
        is None.
    model_dict["cluster_id_to_model_object"]: Dictionary, where each key is a
        cluster ID (positive integer) and the corresponding value is a trained
        bias-correction model (instance of
        `sklearn.isotonic.IsotonicRegression`).  If *not* one model per cluster,
        this is None.
    model_dict["cluster_id_matrix"]: M-by-N numpy array of cluster IDs.  If
        *not* one model per cluster, this is None.
    model_dict["field_names"]: length-1 list with names of target fields.
    model_dict["do_uncertainty_calibration"]: Same as input arg.
    model_dict["do_iso_reg_before_uncertainty_calib"]: Same as input arg.
    """

    # Check input args.
    urma_utils.check_field_name(target_field_name)
    error_checking.assert_is_boolean(do_uncertainty_calibration)
    if cluster_table_xarray is None:
        do_multiprocessing = False
    if not do_uncertainty_calibration:
        do_iso_reg_before_uncertainty_calib = None

    error_checking.assert_is_boolean(do_multiprocessing)

    # Subset all data to the one field.
    first_ptx = prediction_tables_xarray[0]
    num_grid_rows = len(first_ptx.coords[prediction_io.ROW_DIM].values)
    num_grid_columns = len(first_ptx.coords[prediction_io.COLUMN_DIM].values)
    eval_weight_matrix = numpy.full((num_grid_rows, num_grid_columns), 1.)

    num_tables = len(prediction_tables_xarray)

    for i in range(num_tables):
        field_index = numpy.where(
            prediction_tables_xarray[i][prediction_io.FIELD_NAME_KEY].values
            == target_field_name
        )[0][0]

        prediction_tables_xarray[i] = prediction_tables_xarray[i].isel({
            prediction_io.FIELD_DIM: numpy.array([field_index], dtype=int)
        })

        prediction_tables_xarray[i] = prediction_tables_xarray[i].assign({
            EVALUATION_WEIGHT_KEY: (
                (prediction_io.ROW_DIM, prediction_io.COLUMN_DIM),
                eval_weight_matrix
            )
        })

    if cluster_table_xarray is not None:
        field_index = numpy.where(
            cluster_table_xarray[bias_clustering.FIELD_NAME_KEY].values
            == target_field_name
        )[0][0]

        cluster_table_xarray = cluster_table_xarray.isel({
            bias_clustering.FIELD_DIM: numpy.array([field_index], dtype=int)
        })

    # Do actual stuff.
    if cluster_table_xarray is None:
        cluster_id_matrix = None
        unique_cluster_ids = numpy.array([-1], dtype=int)
    else:
        cluster_id_matrix = (
            cluster_table_xarray[bias_clustering.CLUSTER_ID_KEY].values
        )
        unique_cluster_ids = numpy.unique(
            cluster_table_xarray[bias_clustering.CLUSTER_ID_KEY].values
        )
        unique_cluster_ids = unique_cluster_ids[unique_cluster_ids > 0]

    model_object = None
    cluster_id_to_model_object = dict()
    for this_id in unique_cluster_ids:
        cluster_id_to_model_object[this_id] = None

    num_clusters = len(unique_cluster_ids)

    if do_multiprocessing:
        slice_to_cluster_ids = __get_slices_for_multiprocessing(
            cluster_ids=unique_cluster_ids
        )

        argument_list = []
        for this_slice in slice_to_cluster_ids:
            argument_list.append((
                prediction_tables_xarray,
                cluster_table_xarray,
                slice_to_cluster_ids[this_slice]
            ))

        with Pool() as pool_object:
            subdicts = pool_object.starmap(
                _train_one_model_per_cluster, argument_list
            )

            for k in range(len(subdicts)):
                cluster_id_to_model_object.update(subdicts[k])

        for this_cluster_id in cluster_id_to_model_object:
            assert cluster_id_to_model_object[this_cluster_id] is not None

        return {
            MODEL_KEY: model_object,
            CLUSTER_TO_MODEL_KEY: cluster_id_to_model_object,
            CLUSTER_IDS_KEY: cluster_id_matrix,
            FIELD_NAMES_KEY: [target_field_name],
            DO_UNCERTAINTY_CALIB_KEY: do_uncertainty_calibration,
            DO_IR_BEFORE_UC_KEY: do_iso_reg_before_uncertainty_calib
        }

    for k in range(num_clusters):
        if cluster_table_xarray is None:
            print('Training bias-correction model...')
        else:
            print((
                'Training bias-correction model for {0:d}th of {1:d} '
                'clusters...'
            ).format(
                k + 1, num_clusters
            ))

        if cluster_table_xarray is None:
            these_prediction_tables_xarray = prediction_tables_xarray
            model_object = _train_one_model(these_prediction_tables_xarray)
        else:
            these_prediction_tables_xarray = _subset_predictions_to_cluster(
                prediction_tables_xarray=prediction_tables_xarray,
                cluster_table_xarray=cluster_table_xarray,
                desired_cluster_id=unique_cluster_ids[k]
            )

            cluster_id_to_model_object[unique_cluster_ids[k]] = (
                _train_one_model(these_prediction_tables_xarray)
            )

    if cluster_table_xarray is None:
        cluster_id_to_model_object = None

    return {
        MODEL_KEY: model_object,
        CLUSTER_TO_MODEL_KEY: cluster_id_to_model_object,
        CLUSTER_IDS_KEY: cluster_id_matrix,
        FIELD_NAMES_KEY: [target_field_name],
        DO_UNCERTAINTY_CALIB_KEY: do_uncertainty_calibration,
        DO_IR_BEFORE_UC_KEY: do_iso_reg_before_uncertainty_calib
    }


def apply_model_suite(prediction_table_xarray, model_dict_by_field, verbose):
    """Applies model suite to new data in inference mode.

    F = number of target fields

    :param prediction_table_xarray: xarray table in format returned by
        `prediction_io.read_file`.
    :param model_dict_by_field: length-F list of dictionaries, each in format
        created by `train_model_suite`.
    :param verbose: Boolean flag.
    :return: prediction_table_xarray: Same as input but with new predictions.
    """

    exec_start_time_unix_sec = time.time()
    error_checking.assert_is_boolean(verbose)

    num_fields = len(model_dict_by_field)
    one_model_per_cluster = None
    field_names = [None] * num_fields
    do_uncertainty_calibration = None
    do_ir_before_uc = None

    for f in range(num_fields):
        # model_object = model_dict[MODEL_KEY]
        # cluster_id_to_model_object = model_dict[CLUSTER_TO_MODEL_KEY]
        # cluster_id_matrix = model_dict[CLUSTER_IDS_KEY]

        these_field_names = model_dict_by_field[f][FIELD_NAMES_KEY]
        assert len(these_field_names) == 1
        field_names[f] = these_field_names[0]

        if f == 0:
            do_uncertainty_calibration = (
                model_dict_by_field[f][DO_UNCERTAINTY_CALIB_KEY]
            )
            do_ir_before_uc = model_dict_by_field[f][DO_IR_BEFORE_UC_KEY]
            one_model_per_cluster = model_dict_by_field[f][MODEL_KEY] is None

        assert (
            do_uncertainty_calibration ==
            model_dict_by_field[f][DO_UNCERTAINTY_CALIB_KEY]
        )
        assert do_ir_before_uc == model_dict_by_field[f][DO_IR_BEFORE_UC_KEY]
        assert (
            one_model_per_cluster ==
            model_dict_by_field[f][MODEL_KEY] is None
        )

    ptx = prediction_table_xarray
    assert (
        set(field_names) ==
        set(ptx[prediction_io.FIELD_NAME_KEY].values.tolist())
    )

    sort_indices = numpy.array([
        numpy.where(ptx[prediction_io.FIELD_NAME_KEY].values == f)[0][0]
        for f in field_names
    ], dtype=int)

    prediction_table_xarray = prediction_table_xarray.isel({
        prediction_io.FIELD_DIM: sort_indices
    })
    ptx = prediction_table_xarray

    prediction_matrix = ptx[prediction_io.PREDICTION_KEY].values
    mean_prediction_matrix = numpy.mean(prediction_matrix, axis=-1)

    if do_uncertainty_calibration:
        ensemble_size = prediction_matrix.shape[-1]
        assert ensemble_size > 1

        prediction_stdev_matrix = numpy.std(
            prediction_matrix, axis=-1, ddof=1
        )
    else:
        mean_prediction_matrix = numpy.array([], dtype=float)
        prediction_stdev_matrix = numpy.array([], dtype=float)

    for f in range(num_fields):
        model_object = model_dict_by_field[f][MODEL_KEY]
        cluster_id_to_model_object = (
            model_dict_by_field[f][CLUSTER_TO_MODEL_KEY]
        )
        cluster_id_matrix = model_dict_by_field[f][CLUSTER_IDS_KEY]

        if one_model_per_cluster:
            unique_cluster_ids = numpy.array(
                list(cluster_id_to_model_object.keys()),
                dtype=int
            )
        else:
            unique_cluster_ids = numpy.array([-1], dtype=int)

        num_clusters = len(unique_cluster_ids)

        for k in range(num_clusters):
            if verbose:
                print((
                    'Applying bias-correction model for '
                    '{0:d}th of {1:d} fields,'
                    '{2:d}th of {3:d} clusters...'
                ).format(
                    f + 1, num_fields,
                    k + 1, num_clusters
                ))

            if do_uncertainty_calibration:
                if one_model_per_cluster:
                    this_model_object = cluster_id_to_model_object[
                        unique_cluster_ids[k]
                    ]
                    this_cluster_mask = (
                        cluster_id_matrix[..., f] == unique_cluster_ids[k]
                    )

                    orig_stdev_vector = (
                        prediction_stdev_matrix[..., f][this_cluster_mask]
                    )
                    real_mask = numpy.invert(numpy.isnan(orig_stdev_vector))

                    new_stdev_vector = orig_stdev_vector + 0.
                    new_stdev_vector[real_mask] = numpy.sqrt(
                        this_model_object.predict(
                            orig_stdev_vector[real_mask] ** 2
                        )
                    )

                    stdev_inflation_vector = (
                        new_stdev_vector / orig_stdev_vector
                    )
                    stdev_inflation_vector[
                        numpy.isnan(stdev_inflation_vector)
                    ] = 1.
                    stdev_inflation_vector = numpy.minimum(
                        stdev_inflation_vector, MAX_STDEV_INFLATION_FACTOR
                    )

                    stdev_inflation_matrix = numpy.expand_dims(
                        stdev_inflation_vector, axis=-1
                    )
                    this_mean_pred_matrix = numpy.expand_dims(
                        mean_prediction_matrix[..., f][this_cluster_mask],
                        axis=-1
                    )
                    prediction_matrix[this_cluster_mask, f, :] = (
                        this_mean_pred_matrix +
                        stdev_inflation_matrix * (
                            prediction_matrix[this_cluster_mask, f, :] -
                            this_mean_pred_matrix
                        )
                    )
                else:
                    orig_stdev_matrix = prediction_stdev_matrix[..., f]
                    these_dims = orig_stdev_matrix.shape
                    orig_stdev_vector = numpy.ravel(orig_stdev_matrix)
                    real_mask = numpy.invert(numpy.isnan(orig_stdev_vector))

                    new_stdev_vector = orig_stdev_vector + 0.
                    new_stdev_vector[real_mask] = numpy.sqrt(
                        model_object.predict(orig_stdev_vector[real_mask] ** 2)
                    )
                    new_stdev_matrix = numpy.reshape(
                        new_stdev_vector, these_dims
                    )

                    stdev_inflation_matrix = (
                        new_stdev_matrix / orig_stdev_matrix
                    )
                    stdev_inflation_matrix[
                        numpy.isnan(stdev_inflation_matrix)
                    ] = 1.
                    stdev_inflation_matrix = numpy.minimum(
                        stdev_inflation_matrix, MAX_STDEV_INFLATION_FACTOR
                    )

                    stdev_inflation_matrix = numpy.expand_dims(
                        stdev_inflation_matrix, axis=-1
                    )
                    this_mean_pred_matrix = numpy.expand_dims(
                        mean_prediction_matrix[..., f], axis=-1
                    )
                    prediction_matrix[..., f, :] = (
                        this_mean_pred_matrix +
                        stdev_inflation_matrix *
                        (prediction_matrix[..., f, :] - this_mean_pred_matrix)
                    )

                continue

            if one_model_per_cluster:
                this_model_object = cluster_id_to_model_object[
                    unique_cluster_ids[k]
                ]
                this_cluster_mask = (
                    cluster_id_matrix[..., f] == unique_cluster_ids[k]
                )

                orig_mean_vector = (
                    mean_prediction_matrix[..., f][this_cluster_mask]
                )
                real_mask = numpy.invert(numpy.isnan(orig_mean_vector))

                new_mean_vector = orig_mean_vector + 0.
                new_mean_vector[real_mask] = this_model_object.predict(
                    orig_mean_vector[real_mask]
                )

                mean_diff_vector = new_mean_vector - orig_mean_vector
                mean_diff_vector[numpy.isnan(mean_diff_vector)] = 0.

                mean_diff_matrix = numpy.expand_dims(mean_diff_vector, axis=-1)
                prediction_matrix[this_cluster_mask, f, :] = (
                    prediction_matrix[this_cluster_mask, f, :] +
                    mean_diff_matrix
                )
            else:
                orig_mean_matrix = mean_prediction_matrix[..., f]
                these_dims = orig_mean_matrix.shape
                orig_mean_vector = numpy.ravel(orig_mean_matrix)
                real_mask = numpy.invert(numpy.isnan(orig_mean_vector))

                new_mean_vector = orig_mean_vector + 0.
                new_mean_vector[real_mask] = model_object.predict(
                    orig_mean_vector[real_mask]
                )
                new_mean_matrix = numpy.reshape(new_mean_vector, these_dims)

                mean_diff_matrix = new_mean_matrix - orig_mean_matrix
                mean_diff_matrix[numpy.isnan(mean_diff_matrix)] = 0.
                mean_diff_matrix = numpy.expand_dims(
                    mean_diff_matrix, axis=-1
                )
                prediction_matrix[..., f, :] = (
                    prediction_matrix[..., f, :] + mean_diff_matrix
                )

    if (
            urma_utils.DEWPOINT_2METRE_NAME in field_names and
            urma_utils.TEMPERATURE_2METRE_NAME in field_names
    ):
        dewp_idx = field_names.index(urma_utils.DEWPOINT_2METRE_NAME)
        temp_idx = field_names.index(urma_utils.TEMPERATURE_2METRE_NAME)
        prediction_matrix[..., dewp_idx, :] = numpy.minimum(
            prediction_matrix[..., dewp_idx, :],
            prediction_matrix[..., temp_idx, :]
        )

    if (
            urma_utils.U_WIND_10METRE_NAME in field_names and
            urma_utils.V_WIND_10METRE_NAME in field_names and
            urma_utils.WIND_GUST_10METRE_NAME in field_names
    ):
        u_idx = field_names.index(urma_utils.U_WIND_10METRE_NAME)
        v_idx = field_names.index(urma_utils.V_WIND_10METRE_NAME)
        gust_idx = field_names.index(urma_utils.WIND_GUST_10METRE_NAME)

        sustained_speed_matrix = numpy.sqrt(
            prediction_matrix[..., u_idx, :] ** 2 +
            prediction_matrix[..., v_idx, :] ** 2
        )
        prediction_matrix[..., gust_idx, :] = numpy.maximum(
            prediction_matrix[..., gust_idx, :],
            sustained_speed_matrix
        )

    ptx = ptx.assign({
        prediction_io.PREDICTION_KEY: (
            ptx[prediction_io.PREDICTION_KEY].dims,
            prediction_matrix
        )
    })

    print('Applying bias-correction model took {0:.4f} seconds.'.format(
        time.time() - exec_start_time_unix_sec
    ))
    return ptx


def write_file(dill_file_name, model_dict):
    """Writes suite of bias-correction models to Dill file.

    :param dill_file_name: Path to output file.
    :param model_dict: Dictionary in format created by `train_model_suite`.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=dill_file_name)

    dill_file_handle = open(dill_file_name, 'wb')
    dill.dump(model_dict, dill_file_handle)
    dill_file_handle.close()


def read_file(dill_file_name):
    """Reads suite of bias-correction models from Dill file.

    :param dill_file_name: Path to input file.
    :return: model_dict: Dictionary in format created by `train_model_suite`.
    """

    error_checking.assert_file_exists(dill_file_name)

    dill_file_handle = open(dill_file_name, 'rb')
    model_dict = dill.load(dill_file_handle)
    dill_file_handle.close()

    if DO_UNCERTAINTY_CALIB_KEY not in model_dict:
        model_dict[DO_UNCERTAINTY_CALIB_KEY] = False
    if DO_IR_BEFORE_UC_KEY not in model_dict:
        model_dict[DO_IR_BEFORE_UC_KEY] = None

    missing_keys = list(set(ALL_KEYS) - set(model_dict.keys()))
    if len(missing_keys) == 0:
        return model_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), dill_file_name)

    raise ValueError(error_string)
