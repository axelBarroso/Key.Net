import os, argparse, math, cv2, sys, time
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from keyNet.model.keynet_architecture import keynet
from keyNet.loss.score_loss_function import msip_loss_function
import keyNet.aux.tools as aux
import HSequences_bench.tools.geometry_tools as geo_tools
import HSequences_bench.tools.repeatability_tools as rep_tools
from keyNet.datasets.tf_dataset import tf_dataset as tf_dataset
from contextlib import contextmanager

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def save_log(str, file):
    print(str)
    file.write(str+'\n')
    file.flush()

try:
    # python 3.4+ should use builtin unittest.mock not mock package
    from unittest.mock import patch
except ImportError:
    from mock import patch

def train_keynet_architecture():

    parser = argparse.ArgumentParser(description='Train Key.Net Architecture')

    parser.add_argument('--data-dir', type=str, default='path-to-ImageNet',
                        help='The root path to the data from which the synthetic dataset will be created.')

    parser.add_argument('--tfrecord-dir', type=str, default='keyNet/tfrecords/',
                        help='The path to save the generated tfrecords.')

    parser.add_argument('--weights-dir', type=str, default='keyNet/weights',
                        help='The path to save the Key.Net weights.')

    parser.add_argument('--write-summary', type=bool, default=False,
                        help='Set to True if you desire to save the summary of the training.')

    parser.add_argument('--network-version', type=str, default='KeyNet_default',
                        help='The Key.Net network version name')

    parser.add_argument('--num-epochs', type=int, default=25,
                        help='Number of epochs for training.')

    parser.add_argument('--epochs-val', type=int, default=3,
                        help='Set the number of training epochs between repeteability checks on the validation set.')

    parser.add_argument('--batch-size', type=int, default=32,
                        help='The batch size for training.')

    parser.add_argument('--init-initial-learning-rate', type=float, default=1e-3,
                        help='The init initial learning rate value.')

    parser.add_argument('--weights-decay', type=float, default=1e-5,
                        help='The weight decay value.')

    parser.add_argument('--num-epochs-before-decay', type=int, default=10,
                        help='The number of epochs before decay.')

    parser.add_argument('--learning-rate-decay-factor', type=float, default=0.7,
                        help='The learning rate decay factor.')

    parser.add_argument('--random-seed', type=int, default=12345,
                        help='The random seed value for TensorFlow and Numpy.')

    parser.add_argument('--resume-training', type=bool, default=False,
                        help='Set True if resume training is desired.')

    parser.add_argument('--num-filters', type=int, default=8,
                        help='The number of filters in each learnable block.')

    parser.add_argument('--num-learnable-blocks', type=int, default=3,
                        help='The number of learnable blocks after handcrafted block.')

    parser.add_argument('--num-levels-within-net', type=int, default=3,
                        help='The number of pyramid levels inside the architecture.')

    parser.add_argument('--factor-scaling-pyramid', type=float, default=1.2,
                        help='The scale factor between the multi-scale pyramid levels in the architecture.')

    parser.add_argument('--conv-kernel-size', type=int, default=5,
                        help='The size of the convolutional filters in each of the learnable blocks.')

    parser.add_argument('--nms-size', type=int, default=15,
                        help='The NMS size for computing the validation repeatability.')

    parser.add_argument('--border-size', type=int, default=15,
                        help='The number of pixels to remove from the borders to compute the repeatability.')

    parser.add_argument('--max-angle', type=int, default=45,
                        help='The max angle value for generating a synthetic view to train Key.Net.')

    parser.add_argument('--max-scale', type=int, default=2.0,
                        help='The max scale value for generating a synthetic view to train Key.Net.')

    parser.add_argument('--max-shearing', type=int, default=0.8,
                        help='The max shearing value for generating a synthetic view to train Key.Net.')

    parser.add_argument('--patch-size', type=int, default=192,
                        help='The patch size of the generated dataset.')

    parser.add_argument('--weight-coordinates', type=bool, default=True,
                        help='Weighting coordinates by their scores.')

    parser.add_argument('--is-debugging', type=bool, default=False,
                        help='Set variable to True if you desire to train network on a smaller dataset.')

    parser.add_argument('--gpu-memory-fraction', type=float, default=0.9,
                        help='The fraction of GPU used by the script.')

    parser.add_argument('--gpu-visible-devices', type=str, default="0",
                        help='Set CUDA_VISIBLE_DEVICES variable.')

    args = parser.parse_args()

    aux.check_directory('logs')
    log_file = open('logs/'+args.network_version + ".txt", "w+")

    # Set CUDA GPU environment
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_visible_devices

    version_network_name = args.network_version

    # Check directories
    aux.check_directory('keyNet/data')
    aux.check_directory(args.weights_dir)
    aux.check_directory(args.weights_dir + '/' + version_network_name)
    aux.check_directory(args.weights_dir + '/' + version_network_name + '_best')
    aux.check_directory(args.tfrecord_dir)
    aux.check_tensorboard_directory(version_network_name)

    # Set random seeds
    tf.set_random_seed(args.random_seed)
    np.random.seed(args.random_seed)

    print('Start training Key.Net Architecture: ' + version_network_name)

    def check_val_rep(num_points=25):
        total_rep_avg = []
        num_examples = dataset_class.get_num_patches(True)
        fetches = [src_score_maps_activation, dst_score_maps_activation]

        for _ in tqdm(range(num_examples)):
            images_batch, images_dst_batch, h_src_2_dst_batch, h_dst_2_src_batch = sess.run(next_val_batch)

            feed_dict = {
                input_network_src: images_batch,
                input_network_dst: images_dst_batch,
                h_src_2_dst: h_src_2_dst_batch,
                h_dst_2_src: h_dst_2_src_batch,
                phase_train: False,
                dimension_image: np.array(
                    [images_batch.shape[0], images_batch.shape[1], images_batch.shape[2]], dtype=np.int32),
                dimension_image_dst: np.array(
                    [images_dst_batch.shape[0], images_dst_batch.shape[1], images_dst_batch.shape[2]], dtype=np.int32),
            }

            src_scores, dst_scores = sess.run(fetches, feed_dict=feed_dict)

            # Apply NMS
            src_scores = rep_tools.apply_nms(src_scores[0, :, :, 0], args.nms_size)
            dst_scores = rep_tools.apply_nms(dst_scores[0, :, :, 0], args.nms_size)

            hom = geo_tools.prepare_homography(h_dst_2_src_batch[0])
            mask_src, mask_dst = geo_tools.create_common_region_masks(hom, images_batch[0].shape, images_dst_batch[0].shape)

            src_scores = np.multiply(src_scores, mask_src)
            dst_scores = np.multiply(dst_scores, mask_dst)

            src_pts = geo_tools.get_point_coordinates(src_scores, num_points=num_points, order_coord='xysr')
            dst_pts = geo_tools.get_point_coordinates(dst_scores, num_points=num_points, order_coord='xysr')

            dst_to_src_pts = geo_tools.apply_homography_to_points(dst_pts, hom)

            repeatability_results = rep_tools.compute_repeatability(src_pts, dst_to_src_pts)

            total_rep_avg.append(repeatability_results['rep_single_scale'])
        return np.asarray(total_rep_avg).mean()

    def train_epoch():

        total_loss_avg = []
        num_examples = dataset_class.get_num_patches()

        for step in tqdm(range(int(math.ceil(num_examples / args.batch_size))+1)):

            images_batch, images_dst_batch, h_src_2_dst_batch, h_dst_2_src_batch = sess.run(next_batch)

            feed_dict = {
                input_network_src: images_batch,
                input_network_dst: images_dst_batch,
                input_border_mask: aux.remove_borders(np.ones_like(images_batch), 16),
                h_src_2_dst: h_src_2_dst_batch,
                h_dst_2_src: h_dst_2_src_batch,
                phase_train: True,
                dimension_image: np.array([images_batch.shape[0], images_batch.shape[1], images_batch.shape[2]], dtype=np.int32),
                dimension_image_dst: np.array([images_dst_batch.shape[0], images_dst_batch.shape[1], images_dst_batch.shape[2]],dtype=np.int32),
                }

            fetches = [train_op, loss_net, global_step, merged_summary]
            _, loss, global_step_count, summary = sess.run(fetches, feed_dict=feed_dict)

            if args.write_summary:
                train_writer.add_summary(summary, global_step_count)

            total_loss_avg.append(loss)

            if step % 50 == 0:

                feed_dict = {
                    input_network_src: np.reshape(images_batch[0, :, :, :], (1, images_batch.shape[1], images_batch.shape[2], images_batch.shape[3])),
                    input_network_dst: np.reshape(images_dst_batch[0, :, :, :], (1, images_dst_batch.shape[1], images_dst_batch.shape[2], images_dst_batch.shape[3])),
                    phase_train: False,
                    dimension_image: np.array([1, images_batch.shape[1], images_batch.shape[2]],dtype=np.int32),
                    dimension_image_dst: np.array([1, images_dst_batch.shape[1], images_dst_batch.shape[2]],dtype=np.int32),
                }

                fetches = [src_score_maps_activation, dst_score_maps_activation]
                deep_src, deep_dst = sess.run(fetches, feed_dict=feed_dict)

                deep_src = aux.remove_borders(deep_src, 16)
                deep_dst = aux.remove_borders(deep_dst, 16)

                cv2.imwrite('keyNet/data/image_dst_' + version_network_name + '.png', 255 * images_dst_batch[0,:,:,0])
                cv2.imwrite('keyNet/data/KeyNet_dst_' + version_network_name + '.png', 255 * deep_dst[0,:,:, 0] / deep_dst[0,:,:,0].max())
                cv2.imwrite('keyNet/data/image_src_' + version_network_name + '.png', 255 * images_batch[0,:,:,0])
                cv2.imwrite('keyNet/data/KeyNet_src_' + version_network_name + '.png', 255 * deep_src[0,:,:, 0] / deep_src[0,:,:, 0].max())

        return np.asarray(total_loss_avg).mean()

    with tf.Graph().as_default():

        with tf.name_scope('inputs'):

            # Define the input tensor shape
            tensor_input_shape = (None, None, None, 1)
            tensor_homography_shape = (None, 8)

            # Define Placeholders
            input_network_src = tf.placeholder(dtype=tf.float32, shape=tensor_input_shape, name='input_network_src')
            input_network_dst = tf.placeholder(dtype=tf.float32, shape=tensor_input_shape, name='input_network_dst')
            input_border_mask = tf.placeholder(dtype=tf.float32, shape=tensor_input_shape, name='input_border_mask')
            h_src_2_dst = tf.placeholder(dtype=tf.float32, shape=tensor_homography_shape, name='H_scr_2_dst')
            h_dst_2_src = tf.placeholder(dtype=tf.float32, shape=tensor_homography_shape, name='H_dst_2_src')
            dimension_image = tf.placeholder(dtype=tf.int32, shape=(3,), name='dimension_image')
            dimension_image_dst = tf.placeholder(dtype=tf.int32, shape=(3,), name='dimension_image_dst')
            phase_train = tf.placeholder(tf.bool, name='phase_train')

        with tf.name_scope('model_deep_detector'):

            MSIP_sizes = [8, 16, 24, 32, 40]
            MSIP_factor_loss = [256.0, 64.0, 16.0, 4.0, 1.0]

            deep_architecture = keynet(args, MSIP_sizes)

            src_score_maps = deep_architecture.model(input_network_src, phase_train, dimension_image, reuse=False)
            dst_score_maps = deep_architecture.model(input_network_dst, phase_train, dimension_image_dst, reuse=True)

            kernels = deep_architecture.get_kernels()

        # Create Dataset
        dataset_class = tf_dataset(args.data_dir, args.tfrecord_dir, args.patch_size, args.batch_size,
                                   args.max_angle, args.max_scale, args.max_shearing, args.random_seed, args.is_debugging)
        train_dataset = dataset_class.create_dataset_object()
        dataset_it = train_dataset.make_one_shot_iterator()
        next_batch = dataset_it.get_next()
        val_dataset = dataset_class.create_dataset_object(is_val=True)
        dataset_val_it = val_dataset.make_one_shot_iterator()
        next_val_batch = dataset_val_it.get_next()

        # Learning Settings
        num_batches_per_epoch = dataset_class.get_num_patches() / args.batch_size
        num_steps_per_epoch = num_batches_per_epoch  # Because one step is one batch processed
        decay_steps = int(args.num_epochs_before_decay * num_steps_per_epoch)

        global_step = tf.train.get_or_create_global_step()

        lr = tf.train.exponential_decay(
            learning_rate   = args.init_initial_learning_rate,
            global_step     = global_step,
            decay_steps     = decay_steps,
            decay_rate      = args.learning_rate_decay_factor,
            staircase       = True)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        src_score_maps_activation = src_score_maps['output']
        dst_score_maps_activation = dst_score_maps['output']

        # Loss Function
        MSIP_elements = {}
        loss_net = 0.0

        for MSIP_idx in range(len(MSIP_sizes)):
            MSIP_loss, loss_elements = msip_loss_function(input_network_src, src_score_maps, dst_score_maps,
                                                                      MSIP_sizes[MSIP_idx], kernels, h_src_2_dst, h_dst_2_src,
                                                                      args.weight_coordinates, args.patch_size, input_border_mask)
            MSIP_level_name = "MSIP_ws_{}".format(MSIP_sizes[MSIP_idx])
            MSIP_elements[MSIP_level_name] = loss_elements
            tf.summary.scalar(MSIP_level_name, MSIP_loss)
            tf.losses.add_loss(MSIP_factor_loss[MSIP_idx] * MSIP_loss)
            loss_net += MSIP_factor_loss[MSIP_idx] * MSIP_loss

        total_loss = tf.losses.get_total_loss(add_regularization_losses=False)
        train_op = tf.contrib.training.create_train_op(total_loss, optimizer)
        merged_summary = tf.summary.merge_all()

        # Restore Variables
        if args.resume_training:
            checkpoint_file_path = os.path.join(args.weights_dir, version_network_name)
            variables_to_restore = tf.contrib.framework.get_variables_to_restore()
            if os.listdir(checkpoint_file_path):
                init_assign_op, init_feed_dict = tf.contrib.framework.assign_from_checkpoint(
                    tf.train.latest_checkpoint(checkpoint_file_path), variables_to_restore)

        # GPU Usage
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory_fraction

        with tf.Session(config=config) as sess:

            count = 0
            max_counts = 3

            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver()
            saver_best = tf.train.Saver()

            if args.write_summary:
                train_writer = tf.summary.FileWriter('keyNet/logs_network/' + version_network_name + '/train ', sess.graph)

            if args.resume_training and os.listdir(checkpoint_file_path):
                sess.run(init_assign_op, init_feed_dict)
                keynet_rep_best = check_val_rep()
            else:
                keynet_rep_best = 0.0

            print('Start training . . .')

            for epoch in range(0, args.num_epochs):

                start_time = time.time()
                loss = train_epoch()

                aux.check_directory(args.weights_dir + '/' + version_network_name + '/')
                saver.save(sess, args.weights_dir + '/' + version_network_name + '/model-', global_step)

                if epoch % args.epochs_val == 0:
                    with suppress_stdout():
                        keynet_rep_val = check_val_rep()
                    save_log('\nRepeatability Validation: {:.3f}.'.format(keynet_rep_val), log_file)
                else:
                    keynet_rep_val = 0

                # Control the early stopping
                if epoch == 0:
                    loss_best = loss
                else:
                    if keynet_rep_best < keynet_rep_val:
                        keynet_rep_best = keynet_rep_val
                        saver_best.save(sess, args.weights_dir + '/' + version_network_name + '_best' + '/model-', global_step)
                        count = 0
                    elif keynet_rep_val > 0:
                        if loss_best > loss:
                            loss_best = loss
                        else:
                            count += 1

                time_elapsed = time.time() - start_time

                save_log('\nEpoch ' + str(epoch) + '. Loss: ' + str(loss) + '. Time per epoch: ' + str(time_elapsed), log_file)
                if keynet_rep_val > 0:
                    print('Repeatability Val: {:.3f}\n'.format(keynet_rep_val))
                else:
                    print('')

                if count > max_counts:
                    break

            save_log('\nRepeatability Val: {:.3f}. Best iteration'.format(keynet_rep_best), log_file)
            log_file.close()
            print('End training')


if __name__ == '__main__':

    train_keynet_architecture()
