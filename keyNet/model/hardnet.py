import tensorflow as tf
import pickle

class HardNet(object):
    def __init__(self, checkpoint):

        file = open(checkpoint, 'rb')
        pickle_dataset = pickle.load(file)
        file.close()

        self.create_weights(pickle_dataset['weights'])

    def create_weights(self, weights):
        self.weights = {}

        for name, value in weights.items():
            self.weights[name] = value.T

    def features(self, input):

        features = tf.nn.conv2d(input, self.weights['features.0.weight'], strides=[1,1,1,1], padding='SAME')
        features = tf.nn.batch_normalization(features, self.weights['features.1.running_mean'], self.weights['features.1.running_var'], None, None, 1e-6)
        features = tf.nn.relu(features) # 2

        features = tf.nn.conv2d(features, self.weights['features.3.weight'], strides=[1,1,1,1], padding='SAME')
        features = tf.nn.batch_normalization(features, self.weights['features.4.running_mean'], self.weights['features.4.running_var'], None, None, 1e-6)
        features = tf.nn.relu(features) # 5

        features = tf.nn.conv2d(features, self.weights['features.6.weight'], strides=[1,2,2,1], padding='SAME')
        features = tf.nn.batch_normalization(features, self.weights['features.7.running_mean'], self.weights['features.7.running_var'], None, None, 1e-6)
        features = tf.nn.relu(features) # 8

        features = tf.nn.conv2d(features, self.weights['features.9.weight'], strides=[1,1,1,1], padding='SAME')
        features = tf.nn.batch_normalization(features, self.weights['features.10.running_mean'], self.weights['features.10.running_var'], None, None, 1e-6)
        features = tf.nn.relu(features) # 11

        features = tf.nn.conv2d(features, self.weights['features.12.weight'], strides=[1,2,2,1], padding='SAME')
        features = tf.nn.batch_normalization(features, self.weights['features.13.running_mean'], self.weights['features.13.running_var'], None, None, 1e-6)
        features = tf.nn.relu(features) # 14

        features = tf.nn.conv2d(features, self.weights['features.15.weight'], strides=[1,1,1,1], padding='SAME')
        features = tf.nn.batch_normalization(features, self.weights['features.16.running_mean'], self.weights['features.16.running_var'], None, None, 1e-6)
        features = tf.nn.relu(features) # 17

        features = tf.nn.conv2d(features, self.weights['features.19.weight'], strides=[1,1,1,1], padding='VALID')
        features = tf.nn.batch_normalization(features, self.weights['features.20.running_mean'], self.weights['features.20.running_var'], None, None, 1e-6)

        return features

    def input_norm(self, x, eps=1e-6):                      
        x_flatten = tf.layers.flatten(x)
        x_mu, x_std = tf.nn.moments(x_flatten, axes=[1])
        # Add extra dimension
        x_mu = tf.expand_dims(x_mu, axis=1)
        x_std = tf.expand_dims(x_std, axis=1)
        x_norm = (x_flatten - x_mu) / (x_std + eps)         
        return tf.reshape(x_norm, shape=(tf.shape(x)[0], x.shape[1], x.shape[2], x.shape[3]))

    def model(self, input):

        input_norm = self.input_norm(input)

        x_features = self.features(input_norm)
        x = tf.reshape(x_features, shape=(tf.shape(x_features)[0], 128))
        x_norm = tf.nn.l2_normalize(x, axis=1)
        return x_norm