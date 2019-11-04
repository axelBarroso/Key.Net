from os import path, mkdir
import numpy as np
import cv2

def convert_opencv_matches_to_numpy(matches):
    """Returns a np.ndarray array with points indices correspondences
       with the shape of Nx2 which each N feature is a vector containing
       the keypoints id [id_ref, id_dst].
    """
    assert isinstance(matches, list), type(matches)
    correspondences = []
    for match in matches:
        assert isinstance(match, cv2.DMatch), type(match)
        correspondences.append([match.queryIdx, match.trainIdx])
    return np.asarray(correspondences)


def create_results():
   return {
       'num_features': [],
       'rep_single_scale': [],
       'rep_multi_scale': [],
       'num_points_single_scale': [],
       'num_points_multi_scale': [],
       'error_overlap_single_scale': [],
       'error_overlap_multi_scale': [],
       'mma': [],
       'mma_corr': [],
       'num_matches': [],
       'num_mutual_corresp': [],
       'avg_mma': [],
       'num_matches': [],
    }


def create_overlapping_results(detector_name, overlap):

    results = create_results()
    results['detector'] = detector_name
    results['overlap'] = overlap
    return results


def check_directory(dir):
    if not path.isdir(dir):
        mkdir(dir)


def convert_openCV_to_np(pts, dsc, order_coord):
    for idx, kp in enumerate(pts):
        if order_coord == 'xysr':
            kp_np = np.asarray([kp.pt[0], kp.pt[1], kp.size, kp.response, kp.angle])
        else:
            kp_np = np.asarray([kp.pt[1], kp.pt[0], kp.size, kp.response, kp.angle])

        dsc_np = np.asarray(dsc[idx], np.uint8).flatten()

        if idx == 0:
            kps_np = kp_np
            dscs_np = dsc_np
        else:
            kps_np = np.vstack([kps_np, kp_np])
            dscs_np = np.vstack([dscs_np, dsc_np])

    return kps_np, dscs_np