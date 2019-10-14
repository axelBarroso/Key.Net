import os
import cv2
import numpy as np
import tensorflow as tf
import keyNet.datasets.dataset_utils as tools
from tqdm import tqdm

class tf_dataset(object):

    def __init__(self, dataset_root, tfrecord_root, size_patches, batch_size, max_angle, max_scaling, max_shearing, random_seed, is_debugging=False):

        self.size_patches = size_patches
        self.batch_size = batch_size
        self.dataset_root = dataset_root
        self.num_examples = 0
        self.num_val_examples = 0
        self.max_angle = max_angle
        self.max_scaling = max_scaling
        self.max_shearing = max_shearing
        self.is_debugging = is_debugging
        tf.set_random_seed(random_seed)
        np.random.seed(random_seed)

        self.tfrecord_root = tfrecord_root
        if is_debugging:
            self.tfrecord_path = self.tfrecord_root + '/train_dataset_debug.tfrecord'
            self.tfrecord_val_path = self.tfrecord_root + '/val_dataset_debug.tfrecord'
        else:
            self.tfrecord_path = self.tfrecord_root + '/train_dataset.tfrecord'
            self.tfrecord_val_path = self.tfrecord_root + '/val_dataset.tfrecord'

        tfrecord_exists = os.path.isfile(self.tfrecord_path)

        if not tfrecord_exists:

            self.data_path = self._find_data_path(self.dataset_root)

            self.images_info = self._load_data_names(self.data_path)
            self._create_tfrecords(False)
            self._create_tfrecords(True)

        self._compute_num_examples()

        self.feature_description = {
        'im_src_patch': tf.FixedLenFeature([], tf.string),
        'im_dst_patch': tf.FixedLenFeature([], tf.string),
        'homography_src_2_dst': tf.FixedLenFeature([], tf.string),
        'homography_dst_2_src': tf.FixedLenFeature([], tf.string),
        }

    def get_num_patches(self, is_val=False):
        if is_val:
            return self.num_val_examples
        else:
            return self.num_examples

    def create_dataset_object(self, is_val=False):

        self.is_val = is_val

        if self.is_val:
            dataset = tf.data.TFRecordDataset([self.tfrecord_val_path])
            batch_size = 1
        else:
            dataset = tf.data.TFRecordDataset([self.tfrecord_path])
            batch_size = self.batch_size

        dataset = dataset.map(self._prepare_data)
        dataset = dataset.shuffle(buffer_size=500)
        dataset = dataset.batch(batch_size)

        return dataset.repeat()

    def _compute_num_examples(self):

        self.num_examples = 0
        for _ in tf.python_io.tf_record_iterator(self.tfrecord_path):
            self.num_examples += 1

        self.num_val_examples = 0
        for _ in tf.python_io.tf_record_iterator(self.tfrecord_val_path):
            self.num_val_examples += 1

    def _parse_function(self, sample_pair):

        return tf.parse_single_example(sample_pair, self.feature_description)

    def _prepare_data(self, sample_pair):

        if self.is_val:
            patch_size = 2 * self.size_patches
        else:
            patch_size = self.size_patches

        features = tf.parse_single_example(sample_pair, self.feature_description)

        im_src_patch = tf.decode_raw(features['im_src_patch'], tf.float64)
        im_src_patch = tf.reshape(im_src_patch, [patch_size, patch_size, 1])

        im_dst_patch = tf.decode_raw(features['im_dst_patch'], tf.float64)
        im_dst_patch = tf.reshape(im_dst_patch, [patch_size, patch_size, 1])

        homography_src_2_dst = tf.decode_raw(features['homography_src_2_dst'], tf.float32)
        homography_src_2_dst = tf.reshape(homography_src_2_dst, [8])

        homography_dst_2_src = tf.decode_raw(features['homography_dst_2_src'], tf.float32)
        homography_dst_2_src = tf.reshape(homography_dst_2_src, [8])

        return im_src_patch, im_dst_patch, homography_src_2_dst, homography_dst_2_src

    def _find_data_path(self, data_path):
        assert os.path.isdir(data_path), \
            "Invalid directory: {}".format(data_path)
        return data_path

    def _load_data_names(self, data_path):
        count = 0
        images_info = []

        for r, d, f in os.walk(data_path):
            for file_name in f:
                if file_name.endswith(".JPEG") or file_name.endswith(".jpg") or file_name.endswith(".png"):
                    images_info.append(os.path.join(data_path, r, file_name))
                    count += 1

        src_idx = np.random.permutation(len(np.asarray(images_info)))
        images_info = np.asarray(images_info)[src_idx]
        return images_info

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _create_tfrecords(self, is_val):

        self._create_pair_images(is_val)

    def _create_pair_images(self, is_val):

        # More stable repeatability when using bigger size patches
        if is_val:
            size_patches = 2 * self.size_patches
            self.counter += 1
        else:
            size_patches = self.size_patches
            self.counter = 0

        counter_patches = 0

        print('Writing TFrecords . . .')

        if is_val:
            writer = tf.python_io.TFRecordWriter(self.tfrecord_val_path)
        else:
            writer = tf.python_io.TFRecordWriter(self.tfrecord_path)

        for path_image_idx in tqdm(range(len(self.images_info))):

            name_image_path = self.images_info[(self.counter+path_image_idx)%len(self.images_info)]

            correct_patch = False
            counter = -1
            while counter < 10:

                counter += 1
                incorrect_h = True

                while incorrect_h:

                    scr_c = tools.read_color_image(name_image_path)
                    source_shape = scr_c.shape
                    h = tools.generate_composed_homography(self.max_angle, self.max_scaling, self.max_shearing)

                    inv_h = np.linalg.inv(h)
                    inv_h = inv_h / inv_h[2, 2]

                    scr = tools.to_black_and_white(scr_c)
                    dst = tools.color_distorsion(scr_c)
                    dst = tools.apply_h_2_source_image(dst, inv_h)

                    if dst.max() > 0.0:
                        incorrect_h = False

                scr_sobelx = cv2.Sobel(scr, cv2.CV_64F, 1, 0, ksize=3)
                scr_sobelx = abs(scr_sobelx.reshape((scr.shape[0], scr.shape[1], 1)))
                scr_sobelx = scr_sobelx.astype(float) / scr_sobelx.max()
                dst_sobelx = cv2.Sobel(dst, cv2.CV_64F, 1, 0, ksize=3)
                dst_sobelx = abs(dst_sobelx.reshape((dst.shape[0], dst.shape[1], 1)))
                dst_sobelx = dst_sobelx.astype(float) / dst_sobelx.max()

                scr = scr.astype(float) / scr.max()
                dst = dst.astype(float) / dst.max()

                if size_patches/2 >= scr.shape[0]-size_patches/2 or size_patches/2 >= scr.shape[1]-size_patches/2:
                    break

                window_point = [scr.shape[0]/2, scr.shape[1]/2]

                # Define points
                point_src = [window_point[0], window_point[1], 1.0]

                im_src_patch = scr[int(point_src[0] - size_patches / 2): int(point_src[0] + size_patches / 2),
                               int(point_src[1] - size_patches / 2): int(point_src[1] + size_patches / 2)]

                point_dst = inv_h.dot([point_src[1], point_src[0], 1.0])
                point_dst = [point_dst[1] / point_dst[2], point_dst[0] / point_dst[2]]

                if (point_dst[0] - size_patches / 2) < 0 or (point_dst[1] - size_patches / 2) < 0:
                    continue
                if (point_dst[0] + size_patches / 2) > source_shape[0] or (point_dst[1] + size_patches / 2) > \
                        source_shape[1]:
                    continue

                h_src_translation = np.asanyarray([[1., 0., -(int(point_src[1]) - size_patches / 2)],
                                                   [0., 1., -(int(point_src[0]) - size_patches / 2)], [0., 0., 1.]])
                h_dst_translation = np.asanyarray(
                    [[1., 0., int(point_dst[1] - size_patches / 2)], [0., 1., int(point_dst[0] - size_patches / 2)],
                     [0., 0., 1.]])

                im_dst_patch = dst[int(point_dst[0] - size_patches / 2): int(point_dst[0] + size_patches / 2),
                               int(point_dst[1] - size_patches / 2): int(point_dst[1] + size_patches / 2)]
                label_dst_patch = dst_sobelx[
                                  int(point_dst[0] - size_patches / 2): int(point_dst[0] + size_patches / 2),
                                  int(point_dst[1] - size_patches / 2): int(point_dst[1] + size_patches / 2)]
                label_scr_patch = scr_sobelx[
                                  int(point_src[0] - size_patches / 2): int(point_src[0] + size_patches / 2),
                                  int(point_src[1] - size_patches / 2): int(point_src[1] + size_patches / 2)]

                if im_src_patch.shape[0] != size_patches or im_src_patch.shape[1] != size_patches:
                    continue
                if label_dst_patch.max() < 0.25:
                    continue
                if label_scr_patch.max() < 0.25:
                    continue

                correct_patch = True
                break

            if correct_patch:
                im_src_patch = im_src_patch.reshape((1, im_src_patch.shape[0], im_src_patch.shape[1], 1))
                im_dst_patch = im_dst_patch.reshape((1, im_dst_patch.shape[0], im_dst_patch.shape[1], 1))

                homography = np.dot(h_src_translation, np.dot(h, h_dst_translation))

                homography_dst_2_src = homography.astype('float32')
                homography_dst_2_src = homography_dst_2_src.flatten()
                homography_dst_2_src = homography_dst_2_src / homography_dst_2_src[8]
                homography_dst_2_src = homography_dst_2_src[:8]

                homography_src_2_dst = np.linalg.inv(homography)
                homography_src_2_dst = homography_src_2_dst.astype('float32')
                homography_src_2_dst = homography_src_2_dst.flatten()
                homography_src_2_dst = homography_src_2_dst / homography_src_2_dst[8]
                homography_src_2_dst = homography_src_2_dst[:8]

                homography_src_2_dst = homography_src_2_dst.reshape((1, homography_src_2_dst.shape[0]))
                homography_dst_2_src = homography_dst_2_src.reshape((1, homography_dst_2_src.shape[0]))

                sample = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'im_src_patch': self._bytes_feature(im_src_patch.tostring()),
                            'im_dst_patch': self._bytes_feature(im_dst_patch.tostring()),
                            'homography_src_2_dst': self._bytes_feature(homography_src_2_dst.tostring()),
                            'homography_dst_2_src': self._bytes_feature(homography_dst_2_src.tostring())
                        }))
                writer.write(sample.SerializeToString())

                counter_patches += 1

            if is_val and counter_patches > 1500:
                break
            elif counter_patches > 4000:
                break
            if is_val and self.is_debugging and counter_patches > 100:
                break
            elif not is_val and self.is_debugging and counter_patches > 400:
                break

        writer.close()
        self.counter = counter_patches
