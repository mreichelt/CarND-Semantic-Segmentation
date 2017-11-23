import os.path
import tensorflow as tf
import helper
import warnings
import time
from distutils.version import LooseVersion
import project_tests as tests

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    graph = tf.get_default_graph()
    image = graph.get_tensor_by_name('image_input:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    layer3_out = graph.get_tensor_by_name('layer3_out:0')
    layer4_out = graph.get_tensor_by_name('layer4_out:0')
    layer7_out = graph.get_tensor_by_name('layer7_out:0')

    return image, keep_prob, layer3_out, layer4_out, layer7_out


tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes, print_shape=False):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :param print_shape: Whether to print some shapes during training (for debugging).
    :return: The Tensor for the last layer of output
    """

    # define some helper functions here so the code is more readable
    def conv1x1(input):
        return tf.layers.conv2d(input, num_classes, 1, padding='same',
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    def upsample(input, factor):
        return tf.layers.conv2d_transpose(input, num_classes, factor * 2, factor, padding='same',
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # create 1x1 convolutions of desired VGG layers for later use
    layer7_1x1 = conv1x1(vgg_layer7_out)
    layer4_1x1 = conv1x1(vgg_layer4_out)
    layer3_1x1 = conv1x1(vgg_layer3_out)

    # Decoder layer 7
    x = layer7_1x1
    if print_shape:
        x = tf.Print(x, [tf.shape(vgg_layer7_out), tf.shape(vgg_layer4_out), tf.shape(vgg_layer3_out)],
                     message="VGG layers 7, 4, 3: ", summarize=10)
        x = tf.Print(x, [tf.shape(layer7_1x1), tf.shape(layer4_1x1), tf.shape(layer3_1x1)],
                     message="1x1 of VGG layers 7, 4, 3: ", summarize=10)

    # Decoder layer 4 with skip connection
    x = tf.add(
        upsample(x, factor=2),
        layer4_1x1)

    # Decoder layer 3 with skip connection
    x = tf.add(
        upsample(x, factor=2),
        layer3_1x1)

    # Final upsampling of decoder
    x = upsample(x, factor=8)

    if print_shape:
        x = tf.Print(x, [tf.shape(x)], message="Final output: ", summarize=10)

    return x


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss


tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        print('Training epoch {}'.format(epoch))
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image,
                                          correct_label: label,
                                          keep_prob: 0.5,
                                          learning_rate: 0.001})
            print('Loss: {}'.format(loss))
    pass


tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    epochs = 42
    batch_size = 4
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        image, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        nn_last_layer = layers(layer3, layer4, layer7, num_classes)

        # use names here so we can use feed_dict later
        correct_label = tf.placeholder(tf.int8, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image, correct_label,
                 keep_prob, learning_rate)

        # Save inference data using helper.save_inference_samples
        timestamp = str(time.time())
        save_path = tf.train.Saver().save(sess, timestamp + ".ckpt")
        print("Model saved in file: %s" % save_path)
        helper.save_inference_samples(runs_dir, timestamp, data_dir, sess, image_shape, logits, keep_prob, image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
