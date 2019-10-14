import os


def remove_borders(images, borders=3):

    shape = images.shape

    if len(shape) == 4:
        for batch_id in range(shape[0]):
            images[batch_id, 0:borders, :, :] = 0
            images[batch_id, :, 0:borders, :] = 0
            images[batch_id, shape[1] - borders:shape[1], :, :] = 0
            images[batch_id, :, shape[2] - borders:shape[2], :] = 0
    elif len(shape) == 3:
        images[0:borders, :, :] = 0
        images[:, 0:borders, :] = 0
        images[shape[1] - borders:shape[1], :, :] = 0
        images[:, shape[2] - borders:shape[2], :] = 0
    else:
        images[0:borders, :] = 0
        images[:, 0:borders] = 0
        images[shape[0] - borders:shape[0], :] = 0
        images[:, shape[1] - borders:shape[1]] = 0

    return images


def check_directory(file_path):
    if not os.path.exists(file_path):
        os.mkdir(file_path)


def check_tensorboard_directory(version_network_name):
    check_directory('keyNet/logs_network')
    check_directory('keyNet/logs_network/' + version_network_name)

