import os
import json
import numpy as np
from skimage import io


class HSequences_dataset(object):

    def __init__(self, dataset_path, split, split_path):

        self.dataset_path = dataset_path
        self.split = split

        self.splits = json.load(open(split_path))
        self.sequences = self.splits[self.split]['test']

        self.count = 0

    def read_image(self, path):
        im = io.imread(path, as_gray=True)
        return im.reshape(im.shape[0], im.shape[1], 1)

    def read_homography(self, h_name):

        h = np.zeros((3, 3))
        h_file = open(h_name, 'r')

        # Prepare Homography
        for j in range(3):
            line = h_file.readline()
            line = str.split(line);
            for i in range(3):
                h[j, i] = float(line[i])

        inv_h = np.linalg.inv(h)
        inv_h = inv_h / inv_h[2, 2]

        return h, inv_h

    def get_sequence(self, folder_id):

        images_dst = []
        h_src_2_dst = []
        h_dst_2_src = []

        sequence_path = os.path.join(self.dataset_path, self.sequences[folder_id])
        name_image_src_path = sequence_path + '/1.ppm'

        im_src = self.read_image(name_image_src_path)
        im_src = im_src.astype(float) / im_src.max()

        for i in range(5):

            name_image_dst_path = sequence_path + '/' + str(i+2) + '.ppm'

            dst = self.read_image(name_image_dst_path)
            dst = dst.astype(float) / dst.max()

            images_dst.append(dst)

            homography_path = sequence_path + '/H_1_'+str(i+2)
            src_2_dst, dst_2_src = self.read_homography(homography_path)
            h_src_2_dst.append(src_2_dst)
            h_dst_2_src.append(dst_2_src)

        images_dst = np.asarray(images_dst)
        h_src_2_dst = np.asarray(h_src_2_dst)
        h_dst_2_src = np.asarray(h_dst_2_src)

        return {'im_src': im_src, 'images_dst': images_dst, 'h_src_2_dst': h_src_2_dst, 'h_dst_2_src': h_dst_2_src,
                'sequence_name': self.sequences[folder_id]}

    def extract_hsequences(self):

        for idx_sequence in range(len(self.sequences)):

            yield self.get_sequence(idx_sequence)
