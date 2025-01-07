"""'Tests' data-generator by plotting output."""

import copy
import json
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from ml_for_national_blend.outside_code import time_conversion
from ml_for_national_blend.outside_code import gg_plotting_utils
from ml_for_national_blend.outside_code import file_system_utils
from ml_for_national_blend.outside_code import error_checking
from ml_for_national_blend.io import border_io
from ml_for_national_blend.utils import nwp_model_utils
from ml_for_national_blend.utils import nbm_constant_utils
from ml_for_national_blend.machine_learning import neural_net
from ml_for_national_blend.machine_learning import nwp_input
from ml_for_national_blend.plotting import plotting_utils
from ml_for_national_blend.scripts import test_generator_args

TOLERANCE = 1e-6
TIME_FORMAT = '%Y-%m-%d-%H'

SEQUENTIAL_COLOUR_MAP_OBJECT = pyplot.get_cmap('viridis')
DIVERGING_COLOUR_MAP_OBJECT = pyplot.get_cmap('seismic')

DEFAULT_FONT_SIZE = 30
pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER = test_generator_args.add_input_args(
    parser_object=INPUT_ARG_PARSER
)


def _process_nwp_directories(nwp_directory_names, nwp_model_names):
    """Processes NWP directories for either training or validation data.

    :param nwp_directory_names: See documentation for input arg
        "nwp_dir_names_for_training" to this script.
    :param nwp_model_names: See documentation for input arg to this script.
    :return: nwp_model_to_dir_name: Dictionary, where each key is the name of an
        NWP model and the corresponding value is the input directory.
    """

    # assert len(nwp_model_names) == len(nwp_directory_names)
    nwp_directory_names = nwp_directory_names[:len(nwp_model_names)]

    if len(nwp_directory_names) == 1:
        found_any_model_name_in_dir_name = any([
            m in nwp_directory_names[0]
            for m in nwp_model_utils.ALL_MODEL_NAMES_WITH_ENSEMBLE
        ])
        infer_directories = (
            len(nwp_model_names) > 1 or
            (len(nwp_model_names) == 1 and not found_any_model_name_in_dir_name)
        )
    else:
        infer_directories = False

    if infer_directories:
        top_directory_name = copy.deepcopy(nwp_directory_names[0])
        nwp_directory_names = [
            '{0:s}/{1:s}/processed/interp_to_nbm_grid'.format(
                top_directory_name, m
            ) for m in nwp_model_names
        ]

    return dict(zip(nwp_model_names, nwp_directory_names))


def _run(output_dir_name, nwp_lead_times_hours,
         nwp_model_names, nwp_model_to_field_names,
         nwp_normalization_file_name, nwp_use_quantile_norm,
         backup_nwp_model_name, backup_nwp_dir_name,
         target_lead_time_hours, target_field_names, target_lag_times_hours,
         target_normalization_file_name, targets_use_quantile_norm,
         recent_bias_init_time_lags_hours, recent_bias_lead_times_hours,
         nbm_constant_field_names, nbm_constant_file_name,
         num_examples_per_batch, sentinel_value,
         patch_size_2pt5km_pixels, patch_buffer_size_2pt5km_pixels,
         use_fast_patch_generator, patch_overlap_size_2pt5km_pixels,
         require_all_predictors,
         predict_dewpoint_depression, predict_gust_excess,
         do_residual_prediction, resid_baseline_model_name,
         resid_baseline_lead_time_hours, resid_baseline_model_dir_name,
         first_init_time_strings_for_training,
         last_init_time_strings_for_training,
         nwp_dir_names_for_training, target_dir_name_for_training):
    """'Tests' data-generator by plotting output.

    This is effectively the main method.

    :param output_dir_name: See documentation at top of this script.
    :param nwp_lead_times_hours: Same.
    :param nwp_model_names: Same.
    :param nwp_model_to_field_names: Same.
    :param nwp_normalization_file_name: Same.
    :param nwp_use_quantile_norm: Same.
    :param backup_nwp_model_name: Same.
    :param backup_nwp_dir_name: Same.
    :param target_lead_time_hours: Same.
    :param target_field_names: Same.
    :param target_lag_times_hours: Same.
    :param target_normalization_file_name: Same.
    :param targets_use_quantile_norm: Same.
    :param recent_bias_init_time_lags_hours: Same.
    :param recent_bias_lead_times_hours: Same.
    :param nbm_constant_field_names: Same.
    :param nbm_constant_file_name: Same.
    :param num_examples_per_batch: Same.
    :param sentinel_value: Same.
    :param patch_size_2pt5km_pixels: Same.
    :param patch_buffer_size_2pt5km_pixels: Same.
    :param use_fast_patch_generator: Same.
    :param patch_overlap_size_2pt5km_pixels: Same.
    :param require_all_predictors: Same.
    :param predict_dewpoint_depression: Same.
    :param predict_gust_excess: Same.
    :param do_residual_prediction: Same.
    :param resid_baseline_model_name: Same.
    :param resid_baseline_lead_time_hours: Same.
    :param resid_baseline_model_dir_name: Same.
    :param first_init_time_strings_for_training: Same.
    :param last_init_time_strings_for_training: Same.
    :param nwp_dir_names_for_training: Same.
    :param target_dir_name_for_training: Same.
    """

    # TODO(thunderhoser): Make sure to use unnormalized NBM.
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    if nwp_normalization_file_name == '':
        nwp_normalization_file_name = None
    if target_normalization_file_name == '':
        target_normalization_file_name = None
    if resid_baseline_model_name == '':
        resid_baseline_model_name = None
    if resid_baseline_model_dir_name == '':
        resid_baseline_model_dir_name = None
    if resid_baseline_lead_time_hours <= 0:
        resid_baseline_lead_time_hours = None
    if not use_fast_patch_generator:
        patch_overlap_size_2pt5km_pixels = None
    if patch_size_2pt5km_pixels < 0:
        patch_size_2pt5km_pixels = None
    if len(target_lag_times_hours) == 1 and target_lag_times_hours[0] < 0:
        target_lag_times_hours = None

    error_checking.assert_is_string(nbm_constant_file_name)
    error_checking.assert_is_list(nbm_constant_field_names)
    assert nbm_constant_utils.LATITUDE_NAME in nbm_constant_field_names
    assert nbm_constant_utils.LONGITUDE_NAME in nbm_constant_field_names

    if (
            len(recent_bias_init_time_lags_hours) == 1 and
            recent_bias_init_time_lags_hours[0] < 0
    ):
        recent_bias_init_time_lags_hours = None

    if (
            len(recent_bias_lead_times_hours) == 1 and
            recent_bias_lead_times_hours[0] < 0
    ):
        recent_bias_lead_times_hours = None

    nwp_model_to_training_dir_name = _process_nwp_directories(
        nwp_directory_names=nwp_dir_names_for_training,
        nwp_model_names=nwp_model_names
    )

    first_init_times_for_training_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, TIME_FORMAT)
        for t in first_init_time_strings_for_training
    ], dtype=int)
    last_init_times_for_training_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, TIME_FORMAT)
        for t in last_init_time_strings_for_training
    ], dtype=int)

    training_option_dict = {
        neural_net.FIRST_INIT_TIMES_KEY: first_init_times_for_training_unix_sec,
        neural_net.LAST_INIT_TIMES_KEY: last_init_times_for_training_unix_sec,
        neural_net.NWP_LEAD_TIMES_KEY: nwp_lead_times_hours,
        neural_net.NWP_MODEL_TO_DIR_KEY: nwp_model_to_training_dir_name,
        neural_net.NWP_MODEL_TO_FIELDS_KEY: nwp_model_to_field_names,
        neural_net.NWP_NORM_FILE_KEY: nwp_normalization_file_name,
        neural_net.NWP_USE_QUANTILE_NORM_KEY: nwp_use_quantile_norm,
        neural_net.BACKUP_NWP_MODEL_KEY: backup_nwp_model_name,
        neural_net.BACKUP_NWP_DIR_KEY: backup_nwp_dir_name,
        neural_net.TARGET_LEAD_TIME_KEY: target_lead_time_hours,
        neural_net.TARGET_FIELDS_KEY: target_field_names,
        neural_net.TARGET_LAG_TIMES_KEY: target_lag_times_hours,
        neural_net.TARGET_DIR_KEY: target_dir_name_for_training,
        neural_net.TARGET_NORM_FILE_KEY: target_normalization_file_name,
        neural_net.TARGETS_USE_QUANTILE_NORM_KEY: targets_use_quantile_norm,
        neural_net.RECENT_BIAS_LAG_TIMES_KEY: recent_bias_init_time_lags_hours,
        neural_net.RECENT_BIAS_LEAD_TIMES_KEY: recent_bias_lead_times_hours,
        neural_net.NBM_CONSTANT_FIELDS_KEY: nbm_constant_field_names,
        neural_net.NBM_CONSTANT_FILE_KEY: nbm_constant_file_name,
        neural_net.COMPARE_TO_BASELINE_IN_LOSS_KEY: True,
        neural_net.BATCH_SIZE_KEY: num_examples_per_batch,
        neural_net.SENTINEL_VALUE_KEY: sentinel_value,
        neural_net.PREDICT_DEWPOINT_DEPRESSION_KEY: predict_dewpoint_depression,
        neural_net.PREDICT_GUST_EXCESS_KEY: predict_gust_excess,
        neural_net.DO_RESIDUAL_PREDICTION_KEY: do_residual_prediction,
        neural_net.RESID_BASELINE_MODEL_KEY: resid_baseline_model_name,
        neural_net.RESID_BASELINE_LEAD_TIME_KEY: resid_baseline_lead_time_hours,
        neural_net.RESID_BASELINE_MODEL_DIR_KEY: resid_baseline_model_dir_name,
        neural_net.PATCH_SIZE_KEY: patch_size_2pt5km_pixels,
        neural_net.PATCH_BUFFER_SIZE_KEY: patch_buffer_size_2pt5km_pixels,
        neural_net.REQUIRE_ALL_PREDICTORS_KEY: require_all_predictors,
        neural_net.NWP_RESID_NORM_FILE_KEY: None,
        neural_net.TARGET_RESID_NORM_FILE_KEY: None
    }

    use_recent_biases = not (
        recent_bias_init_time_lags_hours is None
        or recent_bias_lead_times_hours is None
    )

    if use_recent_biases:
        nwp_model_to_target_names = nwp_input.nwp_models_to_target_fields(
            nwp_model_names=nwp_model_names,
            target_field_names=target_field_names
        )
    else:
        nwp_model_to_target_names = dict()

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    if patch_overlap_size_2pt5km_pixels is None:
        training_generator = neural_net.data_generator(
            option_dict=training_option_dict,
            return_predictors_as_dict=True
        )
    else:
        training_generator = neural_net.data_generator_fast_patches(
            option_dict=training_option_dict,
            patch_overlap_size_2pt5km_pixels=patch_overlap_size_2pt5km_pixels,
            return_predictors_as_dict=True
        )

    for _ in range(100):
        predictor_matrix_dict, target_matrix = next(training_generator)
    target_matrix = target_matrix[0, ...]

    nwp_model_names = list(nwp_model_to_training_dir_name.keys())
    nwp_model_names.sort()

    predictor_matrix_2pt5km = predictor_matrix_dict['2pt5km_inputs'][0, ...]
    predictor_matrix_2pt5km[predictor_matrix_2pt5km < -9000] = numpy.nan

    nbm_constant_matrix = predictor_matrix_dict['const_inputs'][0, ...]
    nbm_constant_matrix[nbm_constant_matrix < -9000] = numpy.nan
    lat_index = nbm_constant_field_names.index(nbm_constant_utils.LATITUDE_NAME)
    lng_index = nbm_constant_field_names.index(
        nbm_constant_utils.LONGITUDE_NAME
    )

    latitude_matrix_deg_n = nbm_constant_matrix[..., lat_index]
    longitude_matrix_deg_e = nbm_constant_matrix[..., lng_index]

    field_names_2pt5km = []
    nwp_model_names_2pt5km = []
    for this_model_name in nwp_model_names:
        if nwp_model_utils.model_to_nbm_downsampling_factor(this_model_name) != 1:
            continue

        field_names_2pt5km += nwp_model_to_field_names[this_model_name]
        nwp_model_names_2pt5km += [this_model_name] * len(nwp_model_to_field_names[this_model_name])

    for i in range(len(nwp_lead_times_hours)):
        for j in range(len(field_names_2pt5km)):
            this_data_matrix = predictor_matrix_2pt5km[..., i, j]
            figure_object, axes_object = pyplot.subplots(
                1, 1, figsize=(15, 15)
            )

            if '_wind' in field_names_2pt5km[j]:
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
                longitude_matrix_deg_e, latitude_matrix_deg_n,
                data_matrix_to_plot,
                cmap=colour_map_object, norm=colour_norm_object,
                edgecolors='None', zorder=-1e11
            )

            gg_plotting_utils.plot_colour_bar(
                axes_object_or_matrix=axes_object,
                data_matrix=this_data_matrix,
                colour_map_object=colour_map_object,
                colour_norm_object=colour_norm_object,
                orientation_string='vertical',
                extend_min=True, extend_max=True
            )

            plotting_utils.plot_borders(
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                axes_object=axes_object,
                line_colour=numpy.full(3, 0.)
            )
            plotting_utils.plot_grid_lines(
                plot_latitudes_deg_n=numpy.ravel(latitude_matrix_deg_n),
                plot_longitudes_deg_e=numpy.ravel(longitude_matrix_deg_e),
                axes_object=axes_object,
                meridian_spacing_deg=2.,
                parallel_spacing_deg=1.
            )

            axes_object.set_xlim(
                numpy.min(longitude_matrix_deg_e),
                numpy.max(longitude_matrix_deg_e)
            )
            axes_object.set_ylim(
                numpy.min(latitude_matrix_deg_n),
                numpy.max(latitude_matrix_deg_n)
            )

            axes_object.set_title('{0:s} {1:s} at {2:d}-hour lead time'.format(
                nwp_model_names_2pt5km[j],
                field_names_2pt5km[j],
                nwp_lead_times_hours[i]
            ), fontsize=15)

            output_file_name = '{0:s}/{1:s}_{2:03d}hours_{3:s}.jpg'.format(
                output_dir_name,
                field_names_2pt5km[j],
                nwp_lead_times_hours[i],
                nwp_model_names_2pt5km[j].replace('_', '-')
            )

            print('Saving figure to: "{0:s}"...'.format(output_file_name))
            figure_object.savefig(
                output_file_name, dpi=300, pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)

    for j in range(len(nbm_constant_field_names)):
        this_data_matrix = nbm_constant_matrix[..., j]
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
            longitude_matrix_deg_e, latitude_matrix_deg_n,
            data_matrix_to_plot,
            cmap=colour_map_object, norm=colour_norm_object,
            edgecolors='None', zorder=-1e11
        )

        gg_plotting_utils.plot_colour_bar(
            axes_object_or_matrix=axes_object,
            data_matrix=this_data_matrix,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            orientation_string='vertical',
            extend_min=True, extend_max=True
        )

        plotting_utils.plot_borders(
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            axes_object=axes_object,
            line_colour=numpy.full(3, 0.)
        )
        plotting_utils.plot_grid_lines(
            plot_latitudes_deg_n=numpy.ravel(latitude_matrix_deg_n),
            plot_longitudes_deg_e=numpy.ravel(longitude_matrix_deg_e),
            axes_object=axes_object,
            meridian_spacing_deg=2.,
            parallel_spacing_deg=1.
        )

        axes_object.set_xlim(
            numpy.min(longitude_matrix_deg_e),
            numpy.max(longitude_matrix_deg_e)
        )
        axes_object.set_ylim(
            numpy.min(latitude_matrix_deg_n),
            numpy.max(latitude_matrix_deg_n)
        )
        axes_object.set_title('NBM {0:s}'.format(nbm_constant_field_names[j]), fontsize=15)

        output_file_name = '{0:s}/{1:s}_nbm.jpg'.format(
            output_dir_name, nbm_constant_field_names[j]
        )

        print('Saving figure to: "{0:s}"...'.format(output_file_name))
        figure_object.savefig(
            output_file_name, dpi=300, pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

    if target_lag_times_hours is not None:
        lagged_target_predictor_matrix = (
            predictor_matrix_dict['lagtgt_inputs'][0, ...]
        )

        for i in range(len(target_lag_times_hours)):
            for j in range(len(target_field_names)):
                this_data_matrix = lagged_target_predictor_matrix[..., i, j]
                figure_object, axes_object = pyplot.subplots(
                    1, 1, figsize=(15, 15)
                )

                if '_wind' in target_field_names[j]:
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
                    longitude_matrix_deg_e, latitude_matrix_deg_n,
                    data_matrix_to_plot,
                    cmap=colour_map_object, norm=colour_norm_object,
                    edgecolors='None', zorder=-1e11
                )

                gg_plotting_utils.plot_colour_bar(
                    axes_object_or_matrix=axes_object,
                    data_matrix=this_data_matrix,
                    colour_map_object=colour_map_object,
                    colour_norm_object=colour_norm_object,
                    orientation_string='vertical',
                    extend_min=True, extend_max=True
                )

                plotting_utils.plot_borders(
                    border_latitudes_deg_n=border_latitudes_deg_n,
                    border_longitudes_deg_e=border_longitudes_deg_e,
                    axes_object=axes_object,
                    line_colour=numpy.full(3, 0.)
                )
                plotting_utils.plot_grid_lines(
                    plot_latitudes_deg_n=numpy.ravel(latitude_matrix_deg_n),
                    plot_longitudes_deg_e=numpy.ravel(longitude_matrix_deg_e),
                    axes_object=axes_object,
                    meridian_spacing_deg=2.,
                    parallel_spacing_deg=1.
                )

                axes_object.set_xlim(
                    numpy.min(longitude_matrix_deg_e),
                    numpy.max(longitude_matrix_deg_e)
                )
                axes_object.set_ylim(
                    numpy.min(latitude_matrix_deg_n),
                    numpy.max(latitude_matrix_deg_n)
                )

                axes_object.set_title('URMA {0:s} at {1:d}-hour lag time'.format(
                    target_field_names[j],
                    target_lag_times_hours[i]
                ), fontsize=15)

                output_file_name = '{0:s}/{1:s}_{2:03d}hours_urma.jpg'.format(
                    output_dir_name,
                    target_field_names[j],
                    target_lag_times_hours[i]
                )

                print('Saving figure to: "{0:s}"...'.format(output_file_name))
                figure_object.savefig(
                    output_file_name, dpi=300, pad_inches=0, bbox_inches='tight'
                )
                pyplot.close(figure_object)

    field_names_2pt5km = []
    nwp_model_names_2pt5km = []
    for this_model_name in nwp_model_names:
        if nwp_model_utils.model_to_nbm_downsampling_factor(this_model_name) != 1:
            continue
        if not use_recent_biases:
            continue

        field_names_2pt5km += nwp_model_to_target_names[this_model_name]
        nwp_model_names_2pt5km += [this_model_name] * len(nwp_model_to_target_names[this_model_name])

    if len(field_names_2pt5km) > 0:
        recent_bias_matrix_2pt5km = predictor_matrix_dict['2pt5km_rctbias'][0, ...]

        for i in range(len(recent_bias_init_time_lags_hours)):
            for j in range(len(field_names_2pt5km)):
                this_data_matrix = recent_bias_matrix_2pt5km[..., i, j]
                figure_object, axes_object = pyplot.subplots(
                    1, 1, figsize=(15, 15)
                )

                colour_map_object = DIVERGING_COLOUR_MAP_OBJECT
                max_colour_value = numpy.nanpercentile(
                    numpy.absolute(this_data_matrix), 99.9
                )
                if numpy.isnan(max_colour_value):
                    max_colour_value = TOLERANCE

                max_colour_value = max([max_colour_value, TOLERANCE])
                min_colour_value = -1 * max_colour_value

                colour_norm_object = pyplot.Normalize(
                    vmin=min_colour_value, vmax=max_colour_value
                )
                data_matrix_to_plot = this_data_matrix + 0.
                data_matrix_to_plot = numpy.ma.masked_where(
                    numpy.isnan(data_matrix_to_plot), data_matrix_to_plot
                )

                axes_object.pcolor(
                    longitude_matrix_deg_e, latitude_matrix_deg_n,
                    data_matrix_to_plot,
                    cmap=colour_map_object, norm=colour_norm_object,
                    edgecolors='None', zorder=-1e11
                )

                gg_plotting_utils.plot_colour_bar(
                    axes_object_or_matrix=axes_object,
                    data_matrix=this_data_matrix,
                    colour_map_object=colour_map_object,
                    colour_norm_object=colour_norm_object,
                    orientation_string='vertical',
                    extend_min=True, extend_max=True
                )

                plotting_utils.plot_borders(
                    border_latitudes_deg_n=border_latitudes_deg_n,
                    border_longitudes_deg_e=border_longitudes_deg_e,
                    axes_object=axes_object,
                    line_colour=numpy.full(3, 0.)
                )
                plotting_utils.plot_grid_lines(
                    plot_latitudes_deg_n=numpy.ravel(latitude_matrix_deg_n),
                    plot_longitudes_deg_e=numpy.ravel(longitude_matrix_deg_e),
                    axes_object=axes_object,
                    meridian_spacing_deg=2.,
                    parallel_spacing_deg=1.
                )

                axes_object.set_xlim(
                    numpy.min(longitude_matrix_deg_e),
                    numpy.max(longitude_matrix_deg_e)
                )
                axes_object.set_ylim(
                    numpy.min(latitude_matrix_deg_n),
                    numpy.max(latitude_matrix_deg_n)
                )

                axes_object.set_title((
                    'Bias in {0:s} {1:s} at {2:d}-hour lag time '
                    'and {3:d}-hour lead time'
                ).format(
                    nwp_model_names_2pt5km[j],
                    field_names_2pt5km[j],
                    recent_bias_init_time_lags_hours[i],
                    recent_bias_lead_times_hours[i]
                ), fontsize=15)

                output_file_name = (
                    '{0:s}/{1:s}_bias_lag={2:03d}hours_lead={3:03d}hours_'
                    '{4:s}.jpg'
                ).format(
                    output_dir_name,
                    field_names_2pt5km[j],
                    recent_bias_init_time_lags_hours[i],
                    recent_bias_lead_times_hours[i],
                    nwp_model_names_2pt5km[j].replace('_', '-')
                )

                print('Saving figure to: "{0:s}"...'.format(output_file_name))
                figure_object.savefig(
                    output_file_name, dpi=300, pad_inches=0, bbox_inches='tight'
                )
                pyplot.close(figure_object)

    field_names_10km = []
    nwp_model_names_10km = []
    for this_model_name in nwp_model_names:
        if nwp_model_utils.model_to_nbm_downsampling_factor(
                this_model_name) != 4:
            continue

        field_names_10km += nwp_model_to_field_names[this_model_name]
        nwp_model_names_10km += [this_model_name] * len(nwp_model_to_field_names[this_model_name])

    if len(field_names_10km) > 0:
        predictor_matrix_10km = predictor_matrix_dict['10km_inputs'][0, ...]
        this_latitude_matrix_deg_n = latitude_matrix_deg_n[::4, ::4]
        this_longitude_matrix_deg_e = longitude_matrix_deg_e[::4, ::4]

        for i in range(len(nwp_lead_times_hours)):
            for j in range(len(field_names_10km)):
                this_data_matrix = predictor_matrix_10km[..., i, j]
                figure_object, axes_object = pyplot.subplots(
                    1, 1, figsize=(15, 15)
                )

                if '_wind' in field_names_10km[j]:
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
                    this_longitude_matrix_deg_e, this_latitude_matrix_deg_n,
                    data_matrix_to_plot,
                    cmap=colour_map_object, norm=colour_norm_object,
                    edgecolors='None', zorder=-1e11
                )

                gg_plotting_utils.plot_colour_bar(
                    axes_object_or_matrix=axes_object,
                    data_matrix=this_data_matrix,
                    colour_map_object=colour_map_object,
                    colour_norm_object=colour_norm_object,
                    orientation_string='vertical',
                    extend_min=True, extend_max=True
                )

                plotting_utils.plot_borders(
                    border_latitudes_deg_n=border_latitudes_deg_n,
                    border_longitudes_deg_e=border_longitudes_deg_e,
                    axes_object=axes_object,
                    line_colour=numpy.full(3, 0.)
                )
                plotting_utils.plot_grid_lines(
                    plot_latitudes_deg_n=numpy.ravel(latitude_matrix_deg_n),
                    plot_longitudes_deg_e=numpy.ravel(longitude_matrix_deg_e),
                    axes_object=axes_object,
                    meridian_spacing_deg=2.,
                    parallel_spacing_deg=1.
                )

                axes_object.set_xlim(
                    numpy.min(this_longitude_matrix_deg_e),
                    numpy.max(this_longitude_matrix_deg_e)
                )
                axes_object.set_ylim(
                    numpy.min(this_latitude_matrix_deg_n),
                    numpy.max(this_latitude_matrix_deg_n)
                )

                axes_object.set_title('{0:s} {1:s} at {2:d}-hour lead time'.format(
                    nwp_model_names_10km[j],
                    field_names_10km[j],
                    nwp_lead_times_hours[i]
                ), fontsize=15)

                output_file_name = '{0:s}/{1:s}_{2:03d}hours_{3:s}.jpg'.format(
                    output_dir_name,
                    field_names_10km[j],
                    nwp_lead_times_hours[i],
                    nwp_model_names_10km[j].replace('_', '-')
                )

                print('Saving figure to: "{0:s}"...'.format(output_file_name))
                figure_object.savefig(
                    output_file_name, dpi=300, pad_inches=0, bbox_inches='tight'
                )
                pyplot.close(figure_object)

    field_names_10km = []
    nwp_model_names_10km = []
    for this_model_name in nwp_model_names:
        if nwp_model_utils.model_to_nbm_downsampling_factor(this_model_name) != 4:
            continue
        if not use_recent_biases:
            continue

        field_names_10km += nwp_model_to_target_names[this_model_name]
        nwp_model_names_10km += [this_model_name] * len(nwp_model_to_target_names[this_model_name])

    if len(field_names_10km) > 0:
        recent_bias_matrix_10km = predictor_matrix_dict['10km_rctbias'][0, ...]
        this_latitude_matrix_deg_n = latitude_matrix_deg_n[::4, ::4]
        this_longitude_matrix_deg_e = longitude_matrix_deg_e[::4, ::4]

        for i in range(len(recent_bias_init_time_lags_hours)):
            for j in range(len(field_names_10km)):
                this_data_matrix = recent_bias_matrix_10km[..., i, j]
                figure_object, axes_object = pyplot.subplots(
                    1, 1, figsize=(15, 15)
                )

                colour_map_object = DIVERGING_COLOUR_MAP_OBJECT
                max_colour_value = numpy.nanpercentile(
                    numpy.absolute(this_data_matrix), 99.9
                )
                if numpy.isnan(max_colour_value):
                    max_colour_value = TOLERANCE

                max_colour_value = max([max_colour_value, TOLERANCE])
                min_colour_value = -1 * max_colour_value

                colour_norm_object = pyplot.Normalize(
                    vmin=min_colour_value, vmax=max_colour_value
                )
                data_matrix_to_plot = this_data_matrix + 0.
                data_matrix_to_plot = numpy.ma.masked_where(
                    numpy.isnan(data_matrix_to_plot), data_matrix_to_plot
                )

                axes_object.pcolor(
                    this_longitude_matrix_deg_e, this_latitude_matrix_deg_n,
                    data_matrix_to_plot,
                    cmap=colour_map_object, norm=colour_norm_object,
                    edgecolors='None', zorder=-1e11
                )

                gg_plotting_utils.plot_colour_bar(
                    axes_object_or_matrix=axes_object,
                    data_matrix=this_data_matrix,
                    colour_map_object=colour_map_object,
                    colour_norm_object=colour_norm_object,
                    orientation_string='vertical',
                    extend_min=True, extend_max=True
                )

                plotting_utils.plot_borders(
                    border_latitudes_deg_n=border_latitudes_deg_n,
                    border_longitudes_deg_e=border_longitudes_deg_e,
                    axes_object=axes_object,
                    line_colour=numpy.full(3, 0.)
                )
                plotting_utils.plot_grid_lines(
                    plot_latitudes_deg_n=numpy.ravel(latitude_matrix_deg_n),
                    plot_longitudes_deg_e=numpy.ravel(longitude_matrix_deg_e),
                    axes_object=axes_object,
                    meridian_spacing_deg=2.,
                    parallel_spacing_deg=1.
                )

                axes_object.set_xlim(
                    numpy.min(this_longitude_matrix_deg_e),
                    numpy.max(this_longitude_matrix_deg_e)
                )
                axes_object.set_ylim(
                    numpy.min(this_latitude_matrix_deg_n),
                    numpy.max(this_latitude_matrix_deg_n)
                )

                axes_object.set_title((
                    'Bias in {0:s} {1:s} at {2:d}-hour lag time '
                    'and {3:d}-hour lead time'
                ).format(
                    nwp_model_names_10km[j],
                    field_names_10km[j],
                    recent_bias_init_time_lags_hours[i],
                    recent_bias_lead_times_hours[i]
                ), fontsize=15)

                output_file_name = (
                    '{0:s}/{1:s}_bias_lag={2:03d}hours_lead={3:03d}hours_'
                    '{4:s}.jpg'
                ).format(
                    output_dir_name,
                    field_names_10km[j],
                    recent_bias_init_time_lags_hours[i],
                    recent_bias_lead_times_hours[i],
                    nwp_model_names_10km[j].replace('_', '-')
                )

                print('Saving figure to: "{0:s}"...'.format(output_file_name))
                figure_object.savefig(
                    output_file_name, dpi=300, pad_inches=0, bbox_inches='tight'
                )
                pyplot.close(figure_object)

    field_names_20km = []
    nwp_model_names_20km = []
    for this_model_name in nwp_model_names:
        if nwp_model_utils.model_to_nbm_downsampling_factor(this_model_name) != 8:
            continue

        field_names_20km += nwp_model_to_field_names[this_model_name]
        nwp_model_names_20km += [this_model_name] * len(nwp_model_to_field_names[this_model_name])

    if len(field_names_20km) > 0:
        predictor_matrix_20km = predictor_matrix_dict['20km_inputs'][0, ...]
        this_latitude_matrix_deg_n = latitude_matrix_deg_n[::8, ::8]
        this_longitude_matrix_deg_e = longitude_matrix_deg_e[::8, ::8]

        for i in range(len(nwp_lead_times_hours)):
            for j in range(len(field_names_20km)):
                this_data_matrix = predictor_matrix_20km[..., i, j]
                figure_object, axes_object = pyplot.subplots(
                    1, 1, figsize=(15, 15)
                )

                if '_wind' in field_names_20km[j]:
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
                    this_longitude_matrix_deg_e, this_latitude_matrix_deg_n,
                    data_matrix_to_plot,
                    cmap=colour_map_object, norm=colour_norm_object,
                    edgecolors='None', zorder=-1e11
                )

                gg_plotting_utils.plot_colour_bar(
                    axes_object_or_matrix=axes_object,
                    data_matrix=this_data_matrix,
                    colour_map_object=colour_map_object,
                    colour_norm_object=colour_norm_object,
                    orientation_string='vertical',
                    extend_min=True, extend_max=True
                )

                plotting_utils.plot_borders(
                    border_latitudes_deg_n=border_latitudes_deg_n,
                    border_longitudes_deg_e=border_longitudes_deg_e,
                    axes_object=axes_object,
                    line_colour=numpy.full(3, 0.)
                )
                plotting_utils.plot_grid_lines(
                    plot_latitudes_deg_n=numpy.ravel(latitude_matrix_deg_n),
                    plot_longitudes_deg_e=numpy.ravel(longitude_matrix_deg_e),
                    axes_object=axes_object,
                    meridian_spacing_deg=2.,
                    parallel_spacing_deg=1.
                )

                axes_object.set_xlim(
                    numpy.min(this_longitude_matrix_deg_e),
                    numpy.max(this_longitude_matrix_deg_e)
                )
                axes_object.set_ylim(
                    numpy.min(this_latitude_matrix_deg_n),
                    numpy.max(this_latitude_matrix_deg_n)
                )

                axes_object.set_title('{0:s} {1:s} at {2:d}-hour lead time'.format(
                    nwp_model_names_20km[j],
                    field_names_20km[j],
                    nwp_lead_times_hours[i]
                ), fontsize=15)

                output_file_name = '{0:s}/{1:s}_{2:03d}hours_{3:s}.jpg'.format(
                    output_dir_name,
                    field_names_20km[j],
                    nwp_lead_times_hours[i],
                    nwp_model_names_20km[j].replace('_', '-')
                )

                print('Saving figure to: "{0:s}"...'.format(output_file_name))
                figure_object.savefig(
                    output_file_name, dpi=300, pad_inches=0, bbox_inches='tight'
                )
                pyplot.close(figure_object)

    field_names_20km = []
    nwp_model_names_20km = []
    for this_model_name in nwp_model_names:
        if nwp_model_utils.model_to_nbm_downsampling_factor(this_model_name) != 8:
            continue
        if not use_recent_biases:
            continue

        field_names_20km += nwp_model_to_target_names[this_model_name]
        nwp_model_names_20km += [this_model_name] * len(nwp_model_to_target_names[this_model_name])

    if len(field_names_20km) > 0:
        recent_bias_matrix_20km = predictor_matrix_dict['20km_rctbias'][0, ...]
        this_latitude_matrix_deg_n = latitude_matrix_deg_n[::8, ::8]
        this_longitude_matrix_deg_e = longitude_matrix_deg_e[::8, ::8]

        for i in range(len(recent_bias_init_time_lags_hours)):
            for j in range(len(field_names_20km)):
                this_data_matrix = recent_bias_matrix_20km[..., i, j]
                figure_object, axes_object = pyplot.subplots(
                    1, 1, figsize=(15, 15)
                )

                colour_map_object = DIVERGING_COLOUR_MAP_OBJECT
                max_colour_value = numpy.nanpercentile(
                    numpy.absolute(this_data_matrix), 99.9
                )
                if numpy.isnan(max_colour_value):
                    max_colour_value = TOLERANCE

                max_colour_value = max([max_colour_value, TOLERANCE])
                min_colour_value = -1 * max_colour_value

                colour_norm_object = pyplot.Normalize(
                    vmin=min_colour_value, vmax=max_colour_value
                )
                data_matrix_to_plot = this_data_matrix + 0.
                data_matrix_to_plot = numpy.ma.masked_where(
                    numpy.isnan(data_matrix_to_plot), data_matrix_to_plot
                )

                axes_object.pcolor(
                    this_longitude_matrix_deg_e, this_latitude_matrix_deg_n,
                    data_matrix_to_plot,
                    cmap=colour_map_object, norm=colour_norm_object,
                    edgecolors='None', zorder=-1e11
                )

                gg_plotting_utils.plot_colour_bar(
                    axes_object_or_matrix=axes_object,
                    data_matrix=this_data_matrix,
                    colour_map_object=colour_map_object,
                    colour_norm_object=colour_norm_object,
                    orientation_string='vertical',
                    extend_min=True, extend_max=True
                )

                plotting_utils.plot_borders(
                    border_latitudes_deg_n=border_latitudes_deg_n,
                    border_longitudes_deg_e=border_longitudes_deg_e,
                    axes_object=axes_object,
                    line_colour=numpy.full(3, 0.)
                )
                plotting_utils.plot_grid_lines(
                    plot_latitudes_deg_n=numpy.ravel(latitude_matrix_deg_n),
                    plot_longitudes_deg_e=numpy.ravel(longitude_matrix_deg_e),
                    axes_object=axes_object,
                    meridian_spacing_deg=2.,
                    parallel_spacing_deg=1.
                )

                axes_object.set_xlim(
                    numpy.min(this_longitude_matrix_deg_e),
                    numpy.max(this_longitude_matrix_deg_e)
                )
                axes_object.set_ylim(
                    numpy.min(this_latitude_matrix_deg_n),
                    numpy.max(this_latitude_matrix_deg_n)
                )

                axes_object.set_title((
                    'Bias in {0:s} {1:s} at {2:d}-hour lag time '
                    'and {3:d}-hour lead time'
                ).format(
                    nwp_model_names_20km[j],
                    field_names_20km[j],
                    recent_bias_init_time_lags_hours[i],
                    recent_bias_lead_times_hours[i]
                ), fontsize=15)

                output_file_name = (
                    '{0:s}/{1:s}_bias_lag={2:03d}hours_lead={3:03d}hours_'
                    '{4:s}.jpg'
                ).format(
                    output_dir_name,
                    field_names_20km[j],
                    recent_bias_init_time_lags_hours[i],
                    recent_bias_lead_times_hours[i],
                    nwp_model_names_20km[j].replace('_', '-')
                )

                print('Saving figure to: "{0:s}"...'.format(output_file_name))
                figure_object.savefig(
                    output_file_name, dpi=300, pad_inches=0, bbox_inches='tight'
                )
                pyplot.close(figure_object)

    field_names_40km = []
    nwp_model_names_40km = []
    for this_model_name in nwp_model_names:
        if nwp_model_utils.model_to_nbm_downsampling_factor(this_model_name) != 16:
            continue

        field_names_40km += nwp_model_to_field_names[this_model_name]
        nwp_model_names_40km += [this_model_name] * len(nwp_model_to_field_names[this_model_name])

    if len(field_names_40km) > 0:
        predictor_matrix_40km = predictor_matrix_dict['40km_inputs'][0, ...]
        this_latitude_matrix_deg_n = latitude_matrix_deg_n[::16, ::16]
        this_longitude_matrix_deg_e = longitude_matrix_deg_e[::16, ::16]

        for i in range(len(nwp_lead_times_hours)):
            for j in range(len(field_names_40km)):
                this_data_matrix = predictor_matrix_40km[..., i, j]
                figure_object, axes_object = pyplot.subplots(
                    1, 1, figsize=(15, 15)
                )

                if '_wind' in field_names_40km[j]:
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
                    this_longitude_matrix_deg_e, this_latitude_matrix_deg_n,
                    data_matrix_to_plot,
                    cmap=colour_map_object, norm=colour_norm_object,
                    edgecolors='None', zorder=-1e11
                )

                gg_plotting_utils.plot_colour_bar(
                    axes_object_or_matrix=axes_object,
                    data_matrix=this_data_matrix,
                    colour_map_object=colour_map_object,
                    colour_norm_object=colour_norm_object,
                    orientation_string='vertical',
                    extend_min=True, extend_max=True
                )

                plotting_utils.plot_borders(
                    border_latitudes_deg_n=border_latitudes_deg_n,
                    border_longitudes_deg_e=border_longitudes_deg_e,
                    axes_object=axes_object,
                    line_colour=numpy.full(3, 0.)
                )
                plotting_utils.plot_grid_lines(
                    plot_latitudes_deg_n=numpy.ravel(latitude_matrix_deg_n),
                    plot_longitudes_deg_e=numpy.ravel(longitude_matrix_deg_e),
                    axes_object=axes_object,
                    meridian_spacing_deg=2.,
                    parallel_spacing_deg=1.
                )

                axes_object.set_xlim(
                    numpy.min(this_longitude_matrix_deg_e),
                    numpy.max(this_longitude_matrix_deg_e)
                )
                axes_object.set_ylim(
                    numpy.min(this_latitude_matrix_deg_n),
                    numpy.max(this_latitude_matrix_deg_n)
                )

                axes_object.set_title('{0:s} {1:s} at {2:d}-hour lead time'.format(
                    nwp_model_names_40km[j],
                    field_names_40km[j],
                    nwp_lead_times_hours[i]
                ), fontsize=15)

                output_file_name = '{0:s}/{1:s}_{2:03d}hours_{3:s}.jpg'.format(
                    output_dir_name,
                    field_names_40km[j],
                    nwp_lead_times_hours[i],
                    nwp_model_names_40km[j].replace('_', '-')
                )

                print('Saving figure to: "{0:s}"...'.format(output_file_name))
                figure_object.savefig(
                    output_file_name, dpi=300, pad_inches=0, bbox_inches='tight'
                )
                pyplot.close(figure_object)

    field_names_40km = []
    nwp_model_names_40km = []
    for this_model_name in nwp_model_names:
        if nwp_model_utils.model_to_nbm_downsampling_factor(this_model_name) != 16:
            continue
        if not use_recent_biases:
            continue

        field_names_40km += nwp_model_to_target_names[this_model_name]
        nwp_model_names_40km += [this_model_name] * len(nwp_model_to_target_names[this_model_name])

    if len(field_names_40km) > 0:
        recent_bias_matrix_40km = predictor_matrix_dict['40km_rctbias'][0, ...]
        this_latitude_matrix_deg_n = latitude_matrix_deg_n[::16, ::16]
        this_longitude_matrix_deg_e = longitude_matrix_deg_e[::16, ::16]

        for i in range(len(recent_bias_init_time_lags_hours)):
            for j in range(len(field_names_40km)):
                this_data_matrix = recent_bias_matrix_40km[..., i, j]
                figure_object, axes_object = pyplot.subplots(
                    1, 1, figsize=(15, 15)
                )

                colour_map_object = DIVERGING_COLOUR_MAP_OBJECT
                max_colour_value = numpy.nanpercentile(
                    numpy.absolute(this_data_matrix), 99.9
                )
                if numpy.isnan(max_colour_value):
                    max_colour_value = TOLERANCE

                max_colour_value = max([max_colour_value, TOLERANCE])
                min_colour_value = -1 * max_colour_value

                colour_norm_object = pyplot.Normalize(
                    vmin=min_colour_value, vmax=max_colour_value
                )
                data_matrix_to_plot = this_data_matrix + 0.
                data_matrix_to_plot = numpy.ma.masked_where(
                    numpy.isnan(data_matrix_to_plot), data_matrix_to_plot
                )

                axes_object.pcolor(
                    this_longitude_matrix_deg_e, this_latitude_matrix_deg_n,
                    data_matrix_to_plot,
                    cmap=colour_map_object, norm=colour_norm_object,
                    edgecolors='None', zorder=-1e11
                )

                gg_plotting_utils.plot_colour_bar(
                    axes_object_or_matrix=axes_object,
                    data_matrix=this_data_matrix,
                    colour_map_object=colour_map_object,
                    colour_norm_object=colour_norm_object,
                    orientation_string='vertical',
                    extend_min=True, extend_max=True
                )

                plotting_utils.plot_borders(
                    border_latitudes_deg_n=border_latitudes_deg_n,
                    border_longitudes_deg_e=border_longitudes_deg_e,
                    axes_object=axes_object,
                    line_colour=numpy.full(3, 0.)
                )
                plotting_utils.plot_grid_lines(
                    plot_latitudes_deg_n=numpy.ravel(latitude_matrix_deg_n),
                    plot_longitudes_deg_e=numpy.ravel(longitude_matrix_deg_e),
                    axes_object=axes_object,
                    meridian_spacing_deg=2.,
                    parallel_spacing_deg=1.
                )

                axes_object.set_xlim(
                    numpy.min(this_longitude_matrix_deg_e),
                    numpy.max(this_longitude_matrix_deg_e)
                )
                axes_object.set_ylim(
                    numpy.min(this_latitude_matrix_deg_n),
                    numpy.max(this_latitude_matrix_deg_n)
                )

                axes_object.set_title((
                    'Bias in {0:s} {1:s} at {2:d}-hour lag time '
                    'and {3:d}-hour lead time'
                ).format(
                    nwp_model_names_40km[j],
                    field_names_40km[j],
                    recent_bias_init_time_lags_hours[i],
                    recent_bias_lead_times_hours[i]
                ), fontsize=15)

                output_file_name = (
                    '{0:s}/{1:s}_bias_lag={2:03d}hours_lead={3:03d}hours_'
                    '{4:s}.jpg'
                ).format(
                    output_dir_name,
                    field_names_40km[j],
                    recent_bias_init_time_lags_hours[i],
                    recent_bias_lead_times_hours[i],
                    nwp_model_names_40km[j].replace('_', '-')
                )

                print('Saving figure to: "{0:s}"...'.format(output_file_name))
                figure_object.savefig(
                    output_file_name, dpi=300, pad_inches=0, bbox_inches='tight'
                )
                pyplot.close(figure_object)

    if do_residual_prediction:
        predictor_matrix_resid_baseline = (
            predictor_matrix_dict['resid_baseline_inputs'][0, ...]
        )

        for j in range(len(target_field_names)):
            this_data_matrix = predictor_matrix_resid_baseline[..., j]
            figure_object, axes_object = pyplot.subplots(
                1, 1, figsize=(15, 15)
            )

            if '_wind' in target_field_names[j]:
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
                longitude_matrix_deg_e, latitude_matrix_deg_n,
                data_matrix_to_plot,
                cmap=colour_map_object, norm=colour_norm_object,
                edgecolors='None', zorder=-1e11
            )

            gg_plotting_utils.plot_colour_bar(
                axes_object_or_matrix=axes_object,
                data_matrix=this_data_matrix,
                colour_map_object=colour_map_object,
                colour_norm_object=colour_norm_object,
                orientation_string='vertical',
                extend_min=True, extend_max=True
            )

            plotting_utils.plot_borders(
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                axes_object=axes_object,
                line_colour=numpy.full(3, 0.)
            )
            plotting_utils.plot_grid_lines(
                plot_latitudes_deg_n=numpy.ravel(latitude_matrix_deg_n),
                plot_longitudes_deg_e=numpy.ravel(longitude_matrix_deg_e),
                axes_object=axes_object,
                meridian_spacing_deg=2.,
                parallel_spacing_deg=1.
            )

            axes_object.set_xlim(
                numpy.min(longitude_matrix_deg_e),
                numpy.max(longitude_matrix_deg_e)
            )
            axes_object.set_ylim(
                numpy.min(latitude_matrix_deg_n),
                numpy.max(latitude_matrix_deg_n)
            )
            axes_object.set_title('Residual-baseline {0:s}'.format(
                target_field_names[j]
            ), fontsize=15)

            output_file_name = '{0:s}/{1:s}_residual.jpg'.format(
                output_dir_name, target_field_names[j]
            )

            print('Saving figure to: "{0:s}"...'.format(output_file_name))
            figure_object.savefig(
                output_file_name, dpi=300, pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)

    for j in range(len(target_field_names)):
        this_data_matrix = target_matrix[..., j]
        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(15, 15)
        )

        if '_wind' in target_field_names[j]:
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
            longitude_matrix_deg_e, latitude_matrix_deg_n,
            data_matrix_to_plot,
            cmap=colour_map_object, norm=colour_norm_object,
            edgecolors='None', zorder=-1e11
        )

        gg_plotting_utils.plot_colour_bar(
            axes_object_or_matrix=axes_object,
            data_matrix=this_data_matrix,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            orientation_string='vertical',
            extend_min=True, extend_max=True
        )

        plotting_utils.plot_borders(
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            axes_object=axes_object,
            line_colour=numpy.full(3, 0.)
        )
        plotting_utils.plot_grid_lines(
            plot_latitudes_deg_n=numpy.ravel(latitude_matrix_deg_n),
            plot_longitudes_deg_e=numpy.ravel(longitude_matrix_deg_e),
            axes_object=axes_object,
            meridian_spacing_deg=2.,
            parallel_spacing_deg=1.
        )

        axes_object.set_xlim(
            numpy.min(longitude_matrix_deg_e),
            numpy.max(longitude_matrix_deg_e)
        )
        axes_object.set_ylim(
            numpy.min(latitude_matrix_deg_n),
            numpy.max(latitude_matrix_deg_n)
        )
        axes_object.set_title('Target {0:s}'.format(target_field_names[j]), fontsize=15)

        output_file_name = '{0:s}/{1:s}_target.jpg'.format(
            output_dir_name, target_field_names[j]
        )

        print('Saving figure to: "{0:s}"...'.format(output_file_name))
        figure_object.savefig(
            output_file_name, dpi=300, pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

    for j in range(len(target_field_names)):
        this_data_matrix = target_matrix[..., j + len(target_field_names)]
        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(15, 15)
        )

        if '_wind' in target_field_names[j]:
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
            longitude_matrix_deg_e, latitude_matrix_deg_n,
            data_matrix_to_plot,
            cmap=colour_map_object, norm=colour_norm_object,
            edgecolors='None', zorder=-1e11
        )

        gg_plotting_utils.plot_colour_bar(
            axes_object_or_matrix=axes_object,
            data_matrix=this_data_matrix,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            orientation_string='vertical',
            extend_min=True, extend_max=True
        )

        plotting_utils.plot_borders(
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            axes_object=axes_object,
            line_colour=numpy.full(3, 0.)
        )
        plotting_utils.plot_grid_lines(
            plot_latitudes_deg_n=numpy.ravel(latitude_matrix_deg_n),
            plot_longitudes_deg_e=numpy.ravel(longitude_matrix_deg_e),
            axes_object=axes_object,
            meridian_spacing_deg=2.,
            parallel_spacing_deg=1.
        )

        axes_object.set_xlim(
            numpy.min(longitude_matrix_deg_e),
            numpy.max(longitude_matrix_deg_e)
        )
        axes_object.set_ylim(
            numpy.min(latitude_matrix_deg_n),
            numpy.max(latitude_matrix_deg_n)
        )
        axes_object.set_title('Baseline-comparison {0:s}'.format(target_field_names[j]), fontsize=15)

        output_file_name = '{0:s}/{1:s}_comparison.jpg'.format(
            output_dir_name, target_field_names[j]
        )

        print('Saving figure to: "{0:s}"...'.format(output_file_name))
        figure_object.savefig(
            output_file_name, dpi=300, pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

    this_data_matrix = target_matrix[..., -1]
    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(15, 15)
    )

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
        longitude_matrix_deg_e, latitude_matrix_deg_n,
        data_matrix_to_plot,
        cmap=colour_map_object, norm=colour_norm_object,
        edgecolors='None', zorder=-1e11
    )

    gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object,
        data_matrix=this_data_matrix,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical',
        extend_min=True, extend_max=True
    )

    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object,
        line_colour=numpy.full(3, 0.)
    )
    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=numpy.ravel(latitude_matrix_deg_n),
        plot_longitudes_deg_e=numpy.ravel(longitude_matrix_deg_e),
        axes_object=axes_object,
        meridian_spacing_deg=2.,
        parallel_spacing_deg=1.
    )

    axes_object.set_xlim(
        numpy.min(longitude_matrix_deg_e),
        numpy.max(longitude_matrix_deg_e)
    )
    axes_object.set_ylim(
        numpy.min(latitude_matrix_deg_n),
        numpy.max(latitude_matrix_deg_n)
    )
    axes_object.set_title('Evaluation weights', fontsize=15)

    output_file_name = '{0:s}/evaluation_weights.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=300, pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        output_dir_name=getattr(
            INPUT_ARG_OBJECT, test_generator_args.OUTPUT_DIR_ARG_NAME
        ),
        nwp_lead_times_hours=numpy.array(
            getattr(INPUT_ARG_OBJECT, test_generator_args.NWP_LEAD_TIMES_ARG_NAME),
            dtype=int
        ),
        nwp_model_names=getattr(
            INPUT_ARG_OBJECT, test_generator_args.NWP_MODELS_ARG_NAME
        ),
        nwp_model_to_field_names=json.loads(getattr(
            INPUT_ARG_OBJECT, test_generator_args.NWP_MODEL_TO_FIELDS_ARG_NAME
        )),
        nwp_normalization_file_name=getattr(
            INPUT_ARG_OBJECT, test_generator_args.NWP_NORMALIZATION_FILE_ARG_NAME
        ),
        nwp_use_quantile_norm=bool(getattr(
            INPUT_ARG_OBJECT, test_generator_args.NWP_USE_QUANTILE_NORM_ARG_NAME
        )),
        backup_nwp_model_name=getattr(
            INPUT_ARG_OBJECT, test_generator_args.BACKUP_NWP_MODEL_ARG_NAME
        ),
        backup_nwp_dir_name=getattr(
            INPUT_ARG_OBJECT, test_generator_args.BACKUP_NWP_DIR_ARG_NAME
        ),
        target_lead_time_hours=getattr(
            INPUT_ARG_OBJECT, test_generator_args.TARGET_LEAD_TIME_ARG_NAME
        ),
        target_field_names=getattr(
            INPUT_ARG_OBJECT, test_generator_args.TARGET_FIELDS_ARG_NAME
        ),
        target_lag_times_hours=numpy.array(
            getattr(INPUT_ARG_OBJECT, test_generator_args.TARGET_LAG_TIMES_ARG_NAME),
            dtype=int
        ),
        target_normalization_file_name=getattr(
            INPUT_ARG_OBJECT, test_generator_args.TARGET_NORMALIZATION_FILE_ARG_NAME
        ),
        targets_use_quantile_norm=bool(getattr(
            INPUT_ARG_OBJECT, test_generator_args.TARGETS_USE_QUANTILE_NORM_ARG_NAME
        )),
        recent_bias_init_time_lags_hours=numpy.array(
            getattr(
                INPUT_ARG_OBJECT,
                test_generator_args.RECENT_BIAS_LAG_TIMES_ARG_NAME
            ),
            dtype=int
        ),
        recent_bias_lead_times_hours=numpy.array(
            getattr(
                INPUT_ARG_OBJECT,
                test_generator_args.RECENT_BIAS_LEAD_TIMES_ARG_NAME
            ),
            dtype=int
        ),
        nbm_constant_field_names=getattr(
            INPUT_ARG_OBJECT, test_generator_args.NBM_CONSTANT_FIELDS_ARG_NAME
        ),
        nbm_constant_file_name=getattr(
            INPUT_ARG_OBJECT, test_generator_args.NBM_CONSTANT_FILE_ARG_NAME
        ),
        num_examples_per_batch=getattr(
            INPUT_ARG_OBJECT, test_generator_args.BATCH_SIZE_ARG_NAME
        ),
        sentinel_value=getattr(
            INPUT_ARG_OBJECT, test_generator_args.SENTINEL_VALUE_ARG_NAME
        ),
        patch_size_2pt5km_pixels=getattr(
            INPUT_ARG_OBJECT, test_generator_args.PATCH_SIZE_ARG_NAME
        ),
        patch_buffer_size_2pt5km_pixels=getattr(
            INPUT_ARG_OBJECT, test_generator_args.PATCH_BUFFER_SIZE_ARG_NAME
        ),
        use_fast_patch_generator=bool(getattr(
            INPUT_ARG_OBJECT, test_generator_args.USE_FAST_PATCH_GENERATOR_ARG_NAME
        )),
        patch_overlap_size_2pt5km_pixels=getattr(
            INPUT_ARG_OBJECT, test_generator_args.PATCH_OVERLAP_SIZE_ARG_NAME
        ),
        require_all_predictors=bool(getattr(
            INPUT_ARG_OBJECT, test_generator_args.REQUIRE_ALL_PREDICTORS_ARG_NAME
        )),
        predict_dewpoint_depression=bool(getattr(
            INPUT_ARG_OBJECT, test_generator_args.PREDICT_DEWPOINT_DEPRESSION_ARG_NAME
        )),
        predict_gust_excess=bool(getattr(
            INPUT_ARG_OBJECT, test_generator_args.PREDICT_GUST_EXCESS_ARG_NAME
        )),
        do_residual_prediction=bool(getattr(
            INPUT_ARG_OBJECT, test_generator_args.DO_RESIDUAL_PREDICTION_ARG_NAME
        )),
        resid_baseline_model_name=getattr(
            INPUT_ARG_OBJECT, test_generator_args.RESID_BASELINE_MODEL_ARG_NAME
        ),
        resid_baseline_lead_time_hours=getattr(
            INPUT_ARG_OBJECT, test_generator_args.RESID_BASELINE_LEAD_TIME_ARG_NAME
        ),
        resid_baseline_model_dir_name=getattr(
            INPUT_ARG_OBJECT, test_generator_args.RESID_BASELINE_MODEL_DIR_ARG_NAME
        ),
        first_init_time_strings_for_training=getattr(
            INPUT_ARG_OBJECT, test_generator_args.FIRST_TRAINING_TIMES_ARG_NAME
        ),
        last_init_time_strings_for_training=getattr(
            INPUT_ARG_OBJECT, test_generator_args.LAST_TRAINING_TIMES_ARG_NAME
        ),
        nwp_dir_names_for_training=getattr(
            INPUT_ARG_OBJECT, test_generator_args.TRAINING_NWP_DIRS_ARG_NAME
        ),
        target_dir_name_for_training=getattr(
            INPUT_ARG_OBJECT, test_generator_args.TRAINING_TARGET_DIR_ARG_NAME
        )
    )
