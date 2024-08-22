"""Analyzes bias-clustering experiment.

USE ONCE AND DESTROY.
"""

import os
import sys
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import matplotlib.colors

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import imagemagick_utils
import border_io
import bias_clustering
import plotting_utils
import target_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

BIAS_DISCRETIZATION_INTERVAL_INTERVALS_AXIS1 = numpy.array(
    [0.025, 0.050, 0.075, 0.100]
)
MIN_CLUSTER_SIZES_PX_AXIS2 = numpy.array([10, 25, 50, 100, 250], dtype=int)
BUFFER_DISTANCES_PX_AXIS3 = numpy.array([0, 2, 4, 6], dtype=int)
DO_BACKWARDS_FLAGS_AXIS4 = numpy.array([0, 1], dtype=int)

HISTOGRAM_BIN_CENTERS = numpy.concatenate([
    numpy.linspace(1, 20, num=20, dtype=float),
    numpy.linspace(25, 100, num=16, dtype=float),
    numpy.linspace(110, 250, num=15, dtype=float)
])

HISTOGRAM_FACE_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
HISTOGRAM_FACE_COLOUR = matplotlib.colors.to_rgba(HISTOGRAM_FACE_COLOUR, 0.5)
HISTOGRAM_EDGE_COLOUR = numpy.full(3, 0.)
HISTOGRAM_EDGE_WIDTH = 1.5

PANEL_SIZE_PX = int(2.5e6)
CONCAT_FIGURE_SIZE_PX = int(1e7)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

DEFAULT_FONT_SIZE = 30
pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

EXPERIMENT_DIR_ARG_NAME = 'experiment_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

EXPERIMENT_DIR_HELP_STRING = (
    'Path to top-level directory with experiment results for every '
    'hyperparameter set.'
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Analysis figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXPERIMENT_DIR_ARG_NAME, type=str, required=True,
    help=EXPERIMENT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _make_plots_one_hyperparam_set(cluster_file_name, title_string,
                                   histogram_file_name, map_file_name):
    """Makes histogram and spatial map for one hyperparameter set.

    :param cluster_file_name: Path to input file (will be read by
        `bias_clustering.read_file`).
    :param title_string: Figure title.
    :param histogram_file_name: Image file with histogram will be saved to this
        path.
    :param map_file_name: Image file with spatial map will be saved to this
        path.
    """

    print('Reading data from file: "{0:s}"...'.format(cluster_file_name))
    cluster_table_xarray = bias_clustering.read_file(cluster_file_name)
    ctx = cluster_table_xarray

    assert len(ctx.coords[bias_clustering.FIELD_DIM].values) == 1

    _, pixel_counts = numpy.unique(
        ctx[bias_clustering.CLUSTER_ID_KEY].values[..., 0], return_counts=True
    )

    histogram_bin_edges = (
        (HISTOGRAM_BIN_CENTERS[:-1] + HISTOGRAM_BIN_CENTERS[1:]) / 2
    )
    histogram_bin_edges = numpy.concatenate([
        numpy.array([0.]),
        histogram_bin_edges,
        numpy.array([numpy.inf])
    ])
    num_bins = len(HISTOGRAM_BIN_CENTERS)

    bin_indices = numpy.digitize(
        x=pixel_counts, bins=histogram_bin_edges, right=False
    ) - 1
    assert numpy.all(bin_indices >= 0)
    assert numpy.all(bin_indices < num_bins)

    bin_counts = numpy.array([
        numpy.sum(pixel_counts[bin_indices == k])
        for k in numpy.linspace(0, num_bins - 1, num=num_bins, dtype=int)
    ], dtype=int)

    bin_frequencies = bin_counts.astype(float) / numpy.sum(bin_counts)

    k = numpy.argmin(numpy.absolute(HISTOGRAM_BIN_CENTERS - 10))
    fraction_of_px_in_small_cluster = numpy.sum(bin_frequencies[:(k + 1)])
    print((
        'Fraction of pixels in cluster with 10 pixels or less = {0:.4f}'
    ).format(
        fraction_of_px_in_small_cluster
    ))

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    x_tick_values = numpy.linspace(
        0, num_bins - 1, num=num_bins, dtype=float
    ) + 0.5

    axes_object.bar(
        x=x_tick_values, height=bin_frequencies, width=1.,
        color=HISTOGRAM_FACE_COLOUR,
        edgecolor=HISTOGRAM_EDGE_COLOUR,
        linewidth=HISTOGRAM_EDGE_WIDTH
    )

    x_tick_labels = ['{0:.0f}'.format(c) for c in HISTOGRAM_BIN_CENTERS]
    axes_object.set_xticks(x_tick_values)
    axes_object.set_xticklabels(x_tick_labels, fontsize=20, rotation=90)
    # axes_object.set_xlim([0, num_bins])

    axes_object.set_xlabel('Pixels in cluster')
    axes_object.set_ylabel('Fraction of pixels')
    axes_object.set_title(title_string)

    print('Saving figure to: "{0:s}"...'.format(histogram_file_name))
    figure_object.savefig(
        histogram_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    imagemagick_utils.resize_image(
        input_file_name=histogram_file_name,
        output_file_name=histogram_file_name,
        output_size_pixels=PANEL_SIZE_PX
    )

    cluster_id_matrix = ctx[bias_clustering.CLUSTER_ID_KEY].values[..., 0]
    latitude_matrix_deg_n = ctx[bias_clustering.LATITUDE_KEY].values
    longitude_matrix_deg_e = ctx[bias_clustering.LONGITUDE_KEY].values

    unique_cluster_ids = numpy.unique(cluster_id_matrix)
    random_colours = numpy.random.rand(len(unique_cluster_ids), 3)
    colour_map_object = matplotlib.colors.ListedColormap(random_colours)
    colour_norm_object = matplotlib.colors.Normalize(
        vmin=numpy.min(cluster_id_matrix),
        vmax=numpy.max(cluster_id_matrix)
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    target_plotting.plot_field(
        data_matrix=cluster_id_matrix,
        latitude_matrix_deg_n=latitude_matrix_deg_n,
        longitude_matrix_deg_e=longitude_matrix_deg_e,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        axes_object=axes_object,
        plot_colour_bar=False
    )

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

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
        meridian_spacing_deg=20.,
        parallel_spacing_deg=10.
    )

    axes_object.set_xlim(
        numpy.min(longitude_matrix_deg_e),
        numpy.max(longitude_matrix_deg_e)
    )
    axes_object.set_ylim(
        numpy.min(latitude_matrix_deg_n),
        numpy.max(latitude_matrix_deg_n)
    )
    axes_object.set_title(title_string)

    print('Saving figure to: "{0:s}"...'.format(map_file_name))
    figure_object.savefig(
        map_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    imagemagick_utils.resize_image(
        input_file_name=map_file_name,
        output_file_name=map_file_name,
        output_size_pixels=PANEL_SIZE_PX
    )


def _run(experiment_dir_name, output_dir_name):
    """Analyzes bias-clustering experiment.

    This is effectively the main method.

    :param experiment_dir_name: See documentation at top of this script.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    axis1_length = len(BIAS_DISCRETIZATION_INTERVAL_INTERVALS_AXIS1)
    axis2_length = len(MIN_CLUSTER_SIZES_PX_AXIS2)
    axis3_length = len(BUFFER_DISTANCES_PX_AXIS3)
    axis4_length = len(DO_BACKWARDS_FLAGS_AXIS4)

    these_dim = (axis1_length, axis2_length, axis3_length, axis4_length)
    histogram_file_name_matrix = numpy.full(these_dim, '', dtype=object)
    map_file_name_matrix = numpy.full(these_dim, '', dtype=object)

    for k in range(axis3_length):
        for m in range(axis4_length):
            for i in range(axis1_length):
                for j in range(axis2_length):
                    description_string = (
                        'bias-discretization-interval-interval={0:.3f}_'
                        'min-cluster-size-px={1:03d}_buffer-distance-px={2:d}_'
                        'do-backwards-clustering={3:d}'
                    ).format(
                        BIAS_DISCRETIZATION_INTERVAL_INTERVALS_AXIS1[i],
                        MIN_CLUSTER_SIZES_PX_AXIS2[j],
                        BUFFER_DISTANCES_PX_AXIS3[k],
                        DO_BACKWARDS_FLAGS_AXIS4[m]
                    )

                    this_cluster_file_name = (
                        '{0:s}/{1:s}/bias_clustering.nc'
                    ).format(
                        experiment_dir_name,
                        description_string
                    )

                    if not os.path.isfile(this_cluster_file_name):
                        continue

                    histogram_file_name_matrix[i, j, k, m] = (
                        '{0:s}/histogram_{1:s}.jpg'
                    ).format(
                        output_dir_name,
                        description_string
                    )

                    map_file_name_matrix[i, j, k, m] = (
                        '{0:s}/map_{1:s}.jpg'
                    ).format(
                        output_dir_name,
                        description_string
                    )

                    this_title_string = (
                        r'Discretization $\Delta\Delta$ = ' +
                        '{0:.3f}'.format(BIAS_DISCRETIZATION_INTERVAL_INTERVALS_AXIS1[i]) +
                        r'; $K_{min}$ = ' +
                        '{0:d}'.format(MIN_CLUSTER_SIZES_PX_AXIS2[j]) +
                        '; buffer = {0:d} px; {1:s}'.format(
                            BUFFER_DISTANCES_PX_AXIS3[k],
                            'BACKWARDS' if DO_BACKWARDS_FLAGS_AXIS4[m] == 1
                            else 'FORWARD'
                        )
                    )

                    _make_plots_one_hyperparam_set(
                        cluster_file_name=this_cluster_file_name,
                        title_string=this_title_string,
                        histogram_file_name=
                        histogram_file_name_matrix[i, j, k, m],
                        map_file_name=map_file_name_matrix[i, j, k, m]
                    )

            concat_figure_file_name = (
                '{0:s}/histograms_buffer-distance-px={1:d}_'
                'do-backwards-clustering={2:d}.jpg'
            ).format(
                output_dir_name,
                BUFFER_DISTANCES_PX_AXIS3[k],
                DO_BACKWARDS_FLAGS_AXIS4[m]
            )

            panel_file_names = numpy.ravel(
                histogram_file_name_matrix[..., k, m]
            ).tolist()

            panel_file_names = [f for f in panel_file_names if f != '']

            print('Concatenating panels to: "{0:s}"...'.format(
                concat_figure_file_name
            ))
            imagemagick_utils.concatenate_images(
                input_file_names=panel_file_names,
                output_file_name=concat_figure_file_name,
                num_panel_rows=axis2_length,
                num_panel_columns=axis1_length
            )

            concat_figure_file_name = (
                '{0:s}/spatial_maps_buffer-distance-px={1:d}_'
                'do-backwards-clustering={2:d}.jpg'
            ).format(
                output_dir_name,
                BUFFER_DISTANCES_PX_AXIS3[k],
                DO_BACKWARDS_FLAGS_AXIS4[m]
            )

            panel_file_names = numpy.ravel(
                map_file_name_matrix[..., k, m]
            ).tolist()

            panel_file_names = [f for f in panel_file_names if f != '']

            print('Concatenating panels to: "{0:s}"...'.format(
                concat_figure_file_name
            ))
            imagemagick_utils.concatenate_images(
                input_file_names=panel_file_names,
                output_file_name=concat_figure_file_name,
                num_panel_rows=axis2_length,
                num_panel_columns=axis1_length
            )

    print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        experiment_dir_name=getattr(INPUT_ARG_OBJECT, EXPERIMENT_DIR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
