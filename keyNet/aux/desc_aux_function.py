import tensorflow as tf

def _meshgrid(height, width):
    with tf.name_scope('meshgrid'):
        # This should be equivalent to:
        #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
        #                         np.linspace(-1, 1, height))
        #  ones = np.ones(np.prod(x_t.shape))
        #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                        tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
        y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                        tf.ones(shape=tf.stack([1, width])))

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        ones = tf.ones_like(x_t_flat)
        grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
        return grid


def transformer_crop(images, out_size, batch_inds, kpts_xy, kpts_scale=None, kpts_ori=None, thetas=None,
                     name='SpatialTransformCropper'):
    # images : [B,H,W,C]
    # out_size : (out_width, out_height)
    # batch_inds : [B*K,] tf.int32 [0,B)
    # kpts_xy : [B*K,2] tf.float32 or whatever
    # kpts_scale : [B*K,] tf.float32
    # kpts_ori : [B*K,2] tf.float32 (cos,sin)
    if isinstance(out_size, int):
        out_width = out_height = out_size
    else:
        out_width, out_height = out_size
    hoW = out_width // 2
    hoH = out_height // 2

    with tf.name_scope(name):

        num_batch = tf.shape(images)[0]
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]
        C = tf.shape(images)[3]
        num_kp = tf.shape(kpts_xy)[0]  # B*K
        zero = tf.zeros([], dtype=tf.int32)
        max_y = tf.cast(tf.shape(images)[1] - 1, tf.int32)
        max_x = tf.cast(tf.shape(images)[2] - 1, tf.int32)

        grid = _meshgrid(out_height, out_width)  # normalized -1~1
        grid = tf.expand_dims(grid, 0)
        grid = tf.reshape(grid, [-1])
        grid = tf.tile(grid, tf.stack([num_kp]))
        grid = tf.reshape(grid, tf.stack([num_kp, 3, -1]))

        # create 6D affine from scale and orientation
        # [s, 0, 0]   [cos, -sin, 0]
        # [0, s, 0] * [sin,  cos, 0]
        # [0, 0, 1]   [0,    0,   1]

        if thetas is None:
            thetas = tf.eye(2, 3, dtype=tf.float32)
            thetas = tf.tile(thetas[None], [num_kp, 1, 1])
            if kpts_scale is not None:
                thetas = thetas * kpts_scale[:, None, None]
            ones = tf.tile(tf.constant([[[0, 0, 1]]], tf.float32), [num_kp, 1, 1])
            thetas = tf.concat([thetas, ones], axis=1)  # [num_kp, 3,3]

            if kpts_ori is not None:
                cos = tf.slice(kpts_ori, [0, 0], [-1, 1])  # [num_kp, 1]
                sin = tf.slice(kpts_ori, [0, 1], [-1, 1])
                zeros = tf.zeros_like(cos)
                ones = tf.ones_like(cos)
                R = tf.concat([cos, -sin, zeros, sin, cos, zeros, zeros, zeros, ones], axis=-1)
                R = tf.reshape(R, [-1, 3, 3])
                thetas = tf.matmul(thetas, R)
        # Apply transformation to regular grid
        T_g = tf.matmul(thetas, grid)  # [num_kp,3,3] * [num_kp,3,H*W]
        x = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])  # [num_kp,1,H*W]
        y = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])

        # unnormalization [-1,1] --> [-out_size/2,out_size/2]
        x = x * out_width / 2.0
        y = y * out_height / 2.0

        if kpts_xy.dtype != tf.float32:
            kpts_xy = tf.cast(kpts_xy, tf.float32)

        kp_x_ofst = tf.expand_dims(tf.slice(kpts_xy, [0, 0], [-1, 1]), axis=1)  # [B*K,1,1]
        kp_y_ofst = tf.expand_dims(tf.slice(kpts_xy, [0, 1], [-1, 1]), axis=1)  # [B*K,1,1]

        # centerize on keypoints
        x = x + kp_x_ofst
        y = y + kp_y_ofst
        x = tf.reshape(x, [-1])  # num_kp*out_height*out_width
        y = tf.reshape(y, [-1])

        # interpolation
        x0 = tf.cast(tf.floor(x), tf.int32)
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), tf.int32)
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        dim2 = width
        dim1 = width * height
        base = tf.tile(batch_inds[:, None], [1, out_height * out_width])  # [B*K,out_height*out_width]
        base = tf.reshape(base, [-1]) * dim1
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        im_flat = tf.reshape(images, tf.stack([-1, C]))  # [B*height*width,C]
        im_flat = tf.cast(im_flat, tf.float32)

        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)

        x0_f = tf.cast(x0, tf.float32)
        x1_f = tf.cast(x1, tf.float32)
        y0_f = tf.cast(y0, tf.float32)
        y1_f = tf.cast(y1, tf.float32)

        wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
        wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
        wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
        wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)

        output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
        output = tf.reshape(output, tf.stack([num_kp, out_height, out_width, C]))
        output.set_shape([batch_inds.shape[0], out_height, out_width, images.shape[-1]])
        return output


def build_patch_extraction(kpts, batch_inds, images, kpts_scale, name='PatchExtract', patch_size=32):

    with tf.name_scope(name):
        patches = transformer_crop(images, patch_size, batch_inds, kpts, kpts_scale=kpts_scale)

        return patches
