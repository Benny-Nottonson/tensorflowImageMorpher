"""Assorted utility functions for the project."""
from subprocess import call, DEVNULL
import tensorflow as tf


def combine_videos(filenames, output_filename, fps, loop=False):
    """
    The combine_videos function takes a list of filenames, an output filename, and a framerate.
    It then combines the videos into one video with the given framerate.
    If loop is set to True, it will create a gif instead of a mp4.

    :param filenames: Specify the filenames of the videos to be combined
    :param output_filename: Name the output file
    :param fps: Set the frames per second of the output video
    :param loop: Determine whether the video should loop or not
    :return: A video
    """
    input_files = " ".join(f"-i {filename}" for filename in filenames)
    filter_complex = "".join(f"[{i}:v] " for i in range(len(filenames)))
    filter_complex += f"concat=n={len(filenames)}:v=1 [v]"
    if loop:
        command = f"ffmpeg {input_files} -filter_complex \"{filter_complex}\" -map [v] " \
                  f"-r {fps} -loop 0 {output_filename + '.gif'}"
    else:
        command = f"ffmpeg {input_files} -filter_complex \"{filter_complex}\" -map [v] " \
                  f"{output_filename + '.mp4'}"
    call(command, shell=True, stdout=DEVNULL, stderr=DEVNULL)


def _get_dim(x_dim, idx):
    """
    The _get_dim function is used to get the dimension of a tensor.

    :param x_dim: Define the input tensor
    :param idx: Specify the index of the dimension to be returned
    :return: The size of a dimension in the input tensor
    """
    if x_dim.shape.ndims is None:
        return tf.shape(x_dim)[idx]
    return x_dim.shape[idx] or tf.shape(x_dim)[idx]


@tf.function
def dense_image_warp(image, flow, name=None):
    """
    The dense_image_warp function warps an image from one coordinate frame to another.

    :param image: Warp the image
    :param flow: Warp the image
    :param name: Give a name to the operation
    :return: The image warped by the flow
    """
    with tf.name_scope(name or "dense_image_warp"):
        batch_size, height, width, channels = image.shape
        grid_x, grid_y = tf.meshgrid(tf.range(width), tf.range(height))
        stacked_grid = tf.cast(tf.stack([grid_y, grid_x], axis=2), flow.dtype)
        batched_grid = tf.expand_dims(stacked_grid, axis=0)
        query_points_on_grid = batched_grid - flow
        query_points_flattened = tf.reshape(query_points_on_grid, [batch_size, -1, 2])
        interpolated = interpolate_bilinear(image, query_points_flattened)
        interpolated = tf.reshape(interpolated, [batch_size, height, width, channels])
        return interpolated


def interpolate_bilinear(grid, query_points):
    """
    The interpolate_bilinear function takes a grid of values and interpolates
    new values from it.  The grid is defined by the shape of the first two
    dimensions, and the third dimension defines channels at each point in that
    grid.

    :param grid: Define the grid of values to be interpolated
    :param query_points: Specify the points in the grid that we want to sample
    :return: The interpolated value of the query points
    """
    grid_shape = tf.shape(grid)
    query_shape = tf.shape(query_points)

    batch_size, height, width, channels = grid_shape[0], grid_shape[1], grid_shape[2], grid_shape[3]
    num_queries = query_shape[1]

    alphas, floors, ceilings = [], [], []
    index_order = [0, 1]
    unstacked_query_points = tf.unstack(query_points, axis=2, num=2)

    for dim in index_order:
        queries = unstacked_query_points[dim]

        size_in_indexing_dimension = grid_shape[dim + 1]
        max_floor = tf.cast(size_in_indexing_dimension - 2, queries.dtype)
        floor = tf.clip_by_value(tf.floor(queries), 0.0, max_floor)
        int_floor = tf.cast(floor, tf.int32)
        floors.append(int_floor)
        ceil = int_floor + 1
        ceilings.append(ceil)

        alpha = tf.clip_by_value(queries - floor, 0.0, 1.0)
        alpha = tf.expand_dims(alpha, axis=2)
        alphas.append(alpha)

    flattened_grid = tf.reshape(grid, [batch_size * height * width, channels])
    batch_offsets = tf.reshape(tf.range(batch_size) * height * width, [batch_size, 1])

    def gather(y_coords, x_coords):
        """
        The gather function takes a tensor of linear indices and returns the
        corresponding values from the flattened grid. The shape of the output is
        [batch_size, num_queries, channels].

        :param y_coords: Specify the y coordinates of the query points
        :param x_coords: Specify the x-coordinates of the query points
        :return: The values of the pixels at the given coordinates
        """
        linear_coordinates = batch_offsets + y_coords * width + x_coords
        return tf.reshape(tf.gather(flattened_grid, linear_coordinates),
                          [batch_size, num_queries, channels]
                          )

    top_left = gather(floors[0], floors[1])
    top_right = gather(floors[0], ceilings[1])
    bottom_left = gather(ceilings[0], floors[1])
    bottom_right = gather(ceilings[0], ceilings[1])

    interp_top = alphas[1] * (top_right - top_left) + top_left
    interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
    interp = alphas[0] * (interp_bottom - interp_top) + interp_top

    return interp
