import tensorflow as tf
import numpy as np


# Index Proposal Layer
def ip_layer(scores, window_size, kernels):

    exponential_value = np.e

    shape_scores = tf.shape(scores)

    weights = tf.nn.max_pool(tf.stop_gradient(scores), [1, window_size, window_size, 1], strides=[1, window_size, window_size, 1], padding='VALID')

    max_pool_unpool = tf.nn.conv2d_transpose(weights, kernels['upsample_filter_np_'+str(window_size)],
                                              output_shape=[shape_scores[0], shape_scores[1], shape_scores[2], 1],
                                              strides=[1, window_size, window_size, 1])

    exp_map_1 = tf.add(tf.pow(exponential_value, tf.div(scores, max_pool_unpool+1e-6)), -1*(1.-1e-6))

    sum_exp_map_1 = tf.nn.conv2d(exp_map_1, kernels['ones_kernel_'+str(window_size)], [1, window_size, window_size, 1], padding='VALID')

    indexes_map = tf.nn.conv2d(exp_map_1, kernels['indexes_kernel_' + str(window_size)], [1, window_size, window_size, 1], padding='VALID')

    indexes_map = tf.divide(indexes_map, tf.add(sum_exp_map_1, 1e-6))

    max_weights = tf.reduce_max(weights, axis=[1, 2, 3], keepdims=True)

    norm_weights = tf.divide(weights, max_weights + 1e-6)

    return indexes_map, [weights, norm_weights]


def ip_softscores(scores, window_size, kernels):

    exponential_value = np.e

    shape_scores = tf.shape(scores)

    weights = tf.nn.max_pool(scores, [1, window_size, window_size, 1], strides=[1, window_size, window_size, 1], padding='VALID')

    max_pool_unpool = tf.nn.conv2d_transpose(weights, kernels['upsample_filter_np_'+str(window_size)],
                                              output_shape=[shape_scores[0], shape_scores[1], shape_scores[2], 1],
                                              strides=[1, window_size, window_size, 1])

    exp_map_1 = tf.add(tf.pow(exponential_value, tf.div(scores, tf.add(max_pool_unpool, 1e-6))), -1*(1. - 1e-6))

    sum_exp_map_1 = tf.nn.conv2d(exp_map_1, kernels['ones_kernel_'+str(window_size)], [1, window_size, window_size, 1], padding='VALID')

    sum_scores_map_1 = tf.nn.conv2d(exp_map_1*scores, kernels['ones_kernel_'+str(window_size)], [1, window_size, window_size, 1], padding='VALID')

    soft_scores = tf.divide(sum_scores_map_1, tf.add(sum_exp_map_1, 1e-6))

    return soft_scores


def unpool(pool, ind, ksize=[1, 2, 2, 1], scope='unpool'):

    with tf.variable_scope(scope):
        input_shape = tf.shape(pool)
        output_shape = [input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3]]

        flat_input_size = tf.reduce_prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype),
                                 shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b1 = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b1, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
        ret = tf.reshape(ret, output_shape)

        set_input_shape = pool.get_shape()
        set_output_shape = [set_input_shape[0], set_input_shape[1] * ksize[1], set_input_shape[2] * ksize[2],
                            set_input_shape[3]]
        ret.set_shape(set_output_shape)
        return ret


def grid_indexes_nms_conv(scores, kernels, window_size):

    weights, indexes = tf.nn.max_pool_with_argmax(scores, ksize=[1, window_size, window_size, 1],
                                               strides=[1, window_size, window_size, 1], padding='VALID')

    weights_norm = tf.divide(weights, tf.add(weights, np.finfo(float).eps))

    score_map = unpool(weights_norm, indexes, ksize=[1, window_size, window_size, 1], scope='unpool')

    indexes_label = tf.nn.conv2d(score_map, kernels['indexes_kernel_'+str(window_size)], [1, window_size, window_size, 1], padding='VALID')

    ind_rand = tf.cast(tf.random_uniform(tf.shape(indexes_label), minval=0, maxval=window_size, dtype=tf.int32), tf.float32)
    indexes_label = tf.where(tf.equal(indexes_label, tf.zeros_like(indexes_label)), ind_rand, indexes_label)

    return indexes_label, weights, score_map


def loss_ln_indexes_norm(src_indexes, label_indexes, weights_indexes, window_size, n=2):

    norm_sq = tf.reduce_sum(((src_indexes-label_indexes)/window_size)**n, axis=-1, keepdims=True)
    weigthed_norm_sq = 1000*(tf.multiply(weights_indexes, norm_sq))
    loss = tf.reduce_mean(weigthed_norm_sq)

    return loss


def msip_loss_function(src_im, src_score_maps, dst_score_maps, window_size, kernels, h_src_2_dst, h_dst_2_src,
                       coordinate_weighting, patch_size, mask_borders):
    tf.set_random_seed(12345)
    np.random.seed(12345)

    src_maps = tf.nn.relu(src_score_maps['output'])
    dst_maps = tf.nn.relu(dst_score_maps['output'])

    # Check if patch size is divisible by the window size
    if patch_size % window_size > 0:
        batch_shape = tf.shape(src_maps)
        new_size = patch_size - (patch_size % window_size)
        src_maps = tf.slice(src_maps, [0, 0, 0, 0], [batch_shape[0], new_size, new_size, batch_shape[3]])
        dst_maps = tf.slice(dst_maps, [0, 0, 0, 0], [batch_shape[0], new_size, new_size, batch_shape[3]])
        mask_borders = tf.slice(mask_borders, [0, 0, 0, 0], [batch_shape[0], new_size, new_size, batch_shape[3]])

    # Tensorflow inverts homography
    src_maps_warped = tf.contrib.image.transform(src_maps * mask_borders, h_dst_2_src, interpolation='BILINEAR')
    src_im_warped = tf.contrib.image.transform(src_im, h_dst_2_src, interpolation='BILINEAR')
    dst_maps_warped = tf.contrib.image.transform(dst_maps * mask_borders, h_src_2_dst, interpolation='BILINEAR')
    visible_src_mask = tf.contrib.image.transform(mask_borders, h_src_2_dst, interpolation='BILINEAR')
    visible_dst_mask = tf.contrib.image.transform(mask_borders, h_dst_2_src, interpolation='BILINEAR')

    # Remove borders and stop gradients to only backpropagate on the unwarped maps
    src_maps_warped = tf.stop_gradient(src_maps_warped)
    dst_maps_warped = tf.stop_gradient(dst_maps_warped)
    visible_src_mask = visible_src_mask * mask_borders
    visible_dst_mask = visible_dst_mask * mask_borders

    src_maps *= visible_src_mask
    dst_maps *= visible_dst_mask
    src_maps_warped *= visible_dst_mask
    dst_maps_warped *= visible_src_mask

    # Compute visible coordinates to discard uncommon regions
    _, weights_visible_src, map_nms = grid_indexes_nms_conv(visible_src_mask, kernels, window_size)
    _, weights_visible_dst, _ = grid_indexes_nms_conv(visible_dst_mask, kernels, window_size)

    # Extract NMS coordinates from warped maps
    src_indexes_nms_warped, weights_src_warped, _ = grid_indexes_nms_conv(src_maps_warped, kernels, window_size)
    dst_indexes_nms_warped, weights_dst_warped, _ = grid_indexes_nms_conv(dst_maps_warped, kernels, window_size)

    # Use IP Layer to extract soft coordinates
    src_indexes, _ = ip_layer(src_maps, window_size, kernels)
    dst_indexes, _ = ip_layer(dst_maps, window_size, kernels)

    # Compute soft weights
    weights_src = tf.stop_gradient(ip_softscores(src_maps, window_size, kernels))
    weights_dst = tf.stop_gradient(ip_softscores(dst_maps, window_size, kernels))

    if coordinate_weighting:
        shape = tf.shape(weights_src)

        weights_src = tf.layers.flatten(weights_src)
        weights_dst = tf.layers.flatten(weights_dst)

        weights_src = tf.nn.softmax(weights_src)
        weights_dst = tf.nn.softmax(weights_dst)

        weights_src = 100 * weights_visible_src * tf.reshape(weights_src, shape)
        weights_dst = 100 * weights_visible_dst * tf.reshape(weights_dst, shape)
    else:
        weights_src = weights_visible_src
        weights_dst = weights_visible_dst

    loss_src = loss_ln_indexes_norm(src_indexes, dst_indexes_nms_warped, weights_src, window_size, n=2)
    loss_dst = loss_ln_indexes_norm(dst_indexes, src_indexes_nms_warped, weights_dst, window_size, n=2)

    loss_indexes = (loss_src + loss_dst) / 2.

    loss_elements = {}
    loss_elements['src_im'] = src_im
    loss_elements['src_im_warped'] = src_im_warped
    loss_elements['map_nms'] = map_nms
    loss_elements['src_maps'] = src_maps
    loss_elements['dst_maps'] = dst_maps
    loss_elements['src_maps_warped'] = src_maps_warped
    loss_elements['dst_maps_warped'] = dst_maps_warped
    loss_elements['weights_src'] = weights_src
    loss_elements['weights_src_warped'] = weights_src_warped
    loss_elements['weights_visible_src'] = weights_visible_src
    loss_elements['weights_dst'] = weights_dst
    loss_elements['weights_visible_dst'] = weights_visible_dst
    loss_elements['weights_dst_warped'] = weights_dst_warped
    loss_elements['src_indexes'] = src_indexes
    loss_elements['dst_indexes'] = dst_indexes
    loss_elements['dst_indexes_nms_warped'] = dst_indexes_nms_warped

    return loss_indexes, loss_elements
