import os
import argparse
import numpy as np
import pickle
from tqdm import tqdm
import HSequences_bench.tools.aux_tools as aux
import HSequences_bench.tools.geometry_tools as geo_tools
import HSequences_bench.tools.repeatability_tools as rep_tools
import HSequences_bench.tools.matching_tools as match_tools
from HSequences_bench.tools.HSequences_reader import HSequences_dataset
from HSequences_bench.tools.opencv_matcher import OpencvBruteForceMatcher


def hsequences_metrics():

    parser = argparse.ArgumentParser(description='HSequences Compute Repeatability')

    parser.add_argument('--data-dir', type=str, default='hpatches-sequences-release/',
                        help='The root path to HSequences dataset.')

    parser.add_argument('--results-bench-dir', type=str, default='HSequences_bench/results/',
                        help='The output path to save the results.')

    parser.add_argument('--detector-name', type=str, default='KeyNet_default',
                        help='The name of the detector to compute metrics.')

    parser.add_argument('--results-dir', type=str, default='extracted_features/',
                        help='The path to the extracted points.')

    parser.add_argument('--split', type=str, default='view',
                        help='The name of the HPatches (HSequences) split. Use full, debug_view, debug_illum, view or illum.')

    parser.add_argument('--split-path', type=str, default='HSequences_bench/splits.json',
                        help='The path to the split json file.')

    parser.add_argument('--top-k-points', type=int, default=1000,
                        help='The number of top points to use for evaluation. Set to None to use all points')

    parser.add_argument('--overlap', type=float, default=0.6,
                        help='The overlap threshold for a correspondence to be considered correct.')

    parser.add_argument('--pixel-threshold', type=int, default=5,
                        help='The distance of pixels for a matching correspondence to be considered correct.')

    parser.add_argument('--dst-to-src-evaluation', type=bool, default=True,
                        help='Order to apply homography to points. Use True for dst to src, False otherwise.')

    parser.add_argument('--order-coord', type=str, default='xysr',
                        help='The coordinate order that follows the extracted points. Use either xysr or yxsr.')

    args = parser.parse_args()

    print(args.detector_name + ': ' + args.split)

    aux.check_directory(args.results_bench_dir)

    # create the dataloader
    data_loader = HSequences_dataset(args.data_dir, args.split, args.split_path)
    results = aux.create_overlapping_results(args.detector_name, args.overlap)

    # matching method
    matcher = OpencvBruteForceMatcher('l2')

    count_seq = 0

    # load data and compute the keypoints
    for sample_id, sample_data in enumerate(data_loader.extract_hsequences()):

        sequence = sample_data['sequence_name']

        count_seq += 1
        image_src = sample_data['im_src']
        images_dst = sample_data['images_dst']
        h_src_2_dst = sample_data['h_src_2_dst']
        h_dst_2_src = sample_data['h_dst_2_src']

        print('\nComputing ' + sequence + ' sequence {0} / {1} \n'.format(count_seq, len(data_loader.sequences)))

        for idx_im in tqdm(range(len(images_dst))):

            # create the mask to filter out the points outside of the common areas
            mask_src, mask_dst = geo_tools.create_common_region_masks(h_dst_2_src[idx_im], image_src.shape, images_dst[idx_im].shape)

            # compute the files paths
            src_pts_filename = os.path.join(args.results_dir, args.detector_name,
                            'hpatches-sequences-release', '{}/1.ppm.kpt.npy'.format(sample_data['sequence_name']))
            src_dsc_filename = os.path.join(args.results_dir, args.detector_name,
                            'hpatches-sequences-release', '{}/1.ppm.dsc.npy'.format(sample_data['sequence_name']))
            dst_pts_filename = os.path.join(args.results_dir, args.detector_name,
                            'hpatches-sequences-release', '{}/{}.ppm.kpt.npy'.format(sample_data['sequence_name'], idx_im+2))
            dst_dsc_filename = os.path.join(args.results_dir, args.detector_name,
                            'hpatches-sequences-release', '{}/{}.ppm.dsc.npy'.format(sample_data['sequence_name'], idx_im+2))

            if not os.path.isfile(src_pts_filename):
                print("Could not find the file: " + src_pts_filename)
                return False

            if not os.path.isfile(src_dsc_filename):
                print("Could not find the file: " + src_dsc_filename)
                return False

            if not os.path.isfile(dst_pts_filename):
                print("Could not find the file: " + dst_pts_filename)
                return False

            if not os.path.isfile(dst_dsc_filename):
                print("Could not find the file: " + dst_dsc_filename)
                return False

            # load the points
            src_pts = np.load(src_pts_filename)
            src_dsc = np.load(src_dsc_filename)

            dst_pts = np.load(dst_pts_filename)
            dst_dsc = np.load(dst_dsc_filename)

            if args.order_coord == 'xysr':
                src_pts = np.asarray(list(map(lambda x: [x[1], x[0], x[2], x[3]], src_pts)))
                dst_pts = np.asarray(list(map(lambda x: [x[1], x[0], x[2], x[3]], dst_pts)))

            # Check Common Points
            src_idx = rep_tools.check_common_points(src_pts, mask_src)
            src_pts = src_pts[src_idx]
            src_dsc = src_dsc[src_idx]

            dst_idx = rep_tools.check_common_points(dst_pts, mask_dst)
            dst_pts = dst_pts[dst_idx]
            dst_dsc = dst_dsc[dst_idx]

            # Select top K points
            if args.top_k_points:
                src_idx = rep_tools.select_top_k(src_pts, args.top_k_points)
                src_pts = src_pts[src_idx]
                src_dsc = src_dsc[src_idx]

                dst_idx = rep_tools.select_top_k(dst_pts, args.top_k_points)
                dst_pts = dst_pts[dst_idx]
                dst_dsc = dst_dsc[dst_idx]

            src_pts = np.asarray(list(map(lambda x: [x[1], x[0], x[2], x[3]], src_pts)))
            dst_pts = np.asarray(list(map(lambda x: [x[1], x[0], x[2], x[3]], dst_pts)))

            src_to_dst_pts = geo_tools.apply_homography_to_points(
                src_pts, h_src_2_dst[idx_im])

            dst_to_src_pts = geo_tools.apply_homography_to_points(
                dst_pts, h_dst_2_src[idx_im])

            if args.dst_to_src_evaluation:
                points_src = src_pts
                points_dst = dst_to_src_pts
            else:
                points_src = src_to_dst_pts
                points_dst = dst_pts

            # compute repeatability
            repeatability_results = rep_tools.compute_repeatability(points_src, points_dst, overlap_err=1-args.overlap,
                                                                    dist_match_thresh=args.pixel_threshold)

            # match descriptors
            matches = matcher.match(src_dsc, dst_dsc)
            matches_np = aux.convert_opencv_matches_to_numpy(matches)

            matches_inv = matcher.match(dst_dsc, src_dsc)
            matches_inv_np = aux.convert_opencv_matches_to_numpy(matches_inv)

            mask = matches_np[:, 0] == matches_inv_np[matches_np[:, 1], 1]
            matches_np = matches_np[mask]

            match_score, match_score_corr, num_matches = {}, {}, {}

            # compute matching based on pixel distance
            for th_i in range(1, 11):
                match_score_i, match_score_corr_i, num_matches_i = match_tools.compute_matching_based_distance(points_src, points_dst, matches_np,
                                                                       repeatability_results['total_num_points'],
                                                                       pixel_threshold=th_i,
                                                                       possible_matches=repeatability_results['possible_matches'])
                match_score[str(th_i)] = match_score_i
                match_score_corr[str(th_i)] = match_score_corr_i
                num_matches[str(th_i)] = num_matches_i

            mma = np.mean([match_score[str(idx)] for idx in match_score])

            results['rep_single_scale'].append(
                repeatability_results['rep_single_scale'])
            results['rep_multi_scale'].append(
                repeatability_results['rep_multi_scale'])
            results['num_points_single_scale'].append(
                repeatability_results['num_points_single_scale'])
            results['num_points_multi_scale'].append(
                repeatability_results['num_points_multi_scale'])
            results['error_overlap_single_scale'].append(
                repeatability_results['error_overlap_single_scale'])
            results['error_overlap_multi_scale'].append(
                repeatability_results['error_overlap_multi_scale'])

            results['mma'].append(match_score[str(args.pixel_threshold)])
            results['mma_corr'].append(match_score_corr[str(args.pixel_threshold)])
            results['num_matches'].append(num_matches[str(args.pixel_threshold)])
            results['num_mutual_corresp'].append(len(matches_np))
            results['avg_mma'].append(mma)
            results['num_features'].append(repeatability_results['total_num_points'])

    # average the results
    rep_single = np.array(results['rep_single_scale']).mean()
    rep_multi = np.array(results['rep_multi_scale']).mean()
    error_overlap_s = np.array(results['error_overlap_single_scale']).mean()
    error_overlap_m = np.array(results['error_overlap_multi_scale']).mean()
    mma = np.array(results['mma']).mean()
    mma_corr = np.array(results['mma_corr']).mean()
    num_matches = np.array(results['num_matches']).mean()
    num_mutual_corresp = np.array(results['num_mutual_corresp']).mean()
    avg_mma = np.array(results['avg_mma']).mean()
    num_features = np.array(results['num_features']).mean()

    # Matching Score: Matching Score taking into account all features that have been
    # detected in any of the two images.
    # Matching Score (possible matches): Matching Score only taking into account those features that have been
    # detected in both images.
    # MMA Score is computed based on the Matching Score (all detected features)

    print('\n## Overlap @{0}:\n \
           #### Rep. Multi: {1:.4f}\n \
           #### Rep. Single: {2:.4f}\n \
           #### Overlap Multi: {3:.4f}\n \
           #### Overlap Single: {4:.4f}\n \
           #### MMA: {5:.4f}\n \
           #### MMA (possible matches): {6:.4f}\n \
           #### Num matches: {7:.4f}\n \
           #### Num Mutual Correspondences: {8:.4f}\n \
           #### Avg. over Threshold MMA: {9:.4f}\n \
           #### Num Feats: {10:.4f}'.format(
        args.overlap, rep_multi, rep_single, error_overlap_s, error_overlap_m, mma,
        mma_corr, num_matches, num_mutual_corresp, avg_mma, num_features))

    # Store data (serialize)
    output_file_path = os.path.join(args.results_bench_dir, '{0}_{1}.pickle'
        .format(args.detector_name, args.split))
    with open(output_file_path, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':

    hsequences_metrics()