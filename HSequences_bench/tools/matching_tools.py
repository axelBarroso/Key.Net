import numpy as np


def create_precision_recall_results():
    return {
        'recall': 0.0,
        'precision': 0.0,
        'correct_matches': [],
        'false_matches': [],
    }


# retrieve the true correspondences
def compute_matching_based_distance(points_src, points_dst, matches, num_points, pixel_threshold, possible_matches):
    dist = np.sqrt((points_src[matches[:, 0], 0] - points_dst[matches[:, 1], 0]) ** 2 + (
            points_src[matches[:, 0], 1] - points_dst[matches[:, 1], 1]) ** 2)
    matches_dist = np.sum(np.where(dist < pixel_threshold, 1.0, 0.0))
    match_score = matches_dist / num_points
    match_score_corr = matches_dist / (possible_matches+1e-6)
    return match_score, match_score_corr, matches_dist


def compute_precision_recall(matches, true_matches, num_points, eps=1e-6):
    results = create_precision_recall_results()
    if len(true_matches) == 0:
        return results

    num_correct_matches, num_false_matches = 0.0, 0.0
    for match in matches:
        found_match = np.where(match[0] == true_matches[:, 0]) == \
                      np.where(match[1] == true_matches[:, 1])
        if found_match:
            num_correct_matches += 1
            results['correct_matches'].append(match)
        else:
            num_false_matches += 1
            results['false_matches'].append(match)
    # stack matches
    results['correct_matches'] = np.array(results['correct_matches'])
    results['false_matches'] = np.array(results['false_matches'])
    # compute the actual statistics
    num_correspondences = true_matches.shape[0] + eps
    sum_matches = num_correct_matches + num_false_matches + eps
    # return a dictionary with all the results
    results['recall'] = num_correct_matches / num_correspondences
    results['recall_total'] = num_correct_matches / num_points
    results['precision'] = 1. - num_false_matches / sum_matches
    return results

# find matches
def find_matches(dsc_src, dsc_dst):

    dsc_src = np.reshape(dsc_src, (dsc_src.shape[0], 1, 128))
    dsc_dst = np.reshape(dsc_dst, (1, dsc_dst.shape[0], 128))
    dsc_src = np.repeat(dsc_src, dsc_dst.shape[1], axis=1)
    dsc_dst = np.repeat(dsc_dst, dsc_src.shape[0], axis=0)

    l2_matrix = np.sum((dsc_src - dsc_dst)**2, axis=-1)
    matches = l2_matrix.argmin(axis=1)
    return [np.arange(len(dsc_src)), matches]