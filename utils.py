import tensorflow as tf


# Ref - https://github.com/xingjunm/lid_adversarial_subspace_detection
# Ref - "Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality". ICLR 2018
def get_lid_score(data, k=21):
    # pairwise distance
    r = tf.reduce_sum(data ** 2, axis=1)
    r = tf.reshape(r, [-1, 1])
    distance = tf.sqrt(r - 2 * tf.matmul(data, tf.transpose(data)) + tf.transpose(r) + 1e-9)

    # find the k nearest neighbor
    temp, _ = tf.nn.top_k(-distance, k=k, sorted=True)
    top_k_distance = -temp[:, 1:]

    m = tf.transpose(tf.multiply(tf.transpose(top_k_distance), 1.0 / top_k_distance[:, -1]))
    lids = (1 - k) / tf.reduce_sum(tf.math.log(m + 1e-9), axis=1)
    return tf.reduce_mean(lids)


def euclidean_distance(x, y):
    return tf.broadcast_to(tf.reshape(tf.reduce_sum(x ** 2, axis=-1), (1, x.shape[0])), (y.shape[0], x.shape[0])) + \
           tf.broadcast_to(tf.reshape(tf.reduce_sum(y ** 2, axis=-1), (y.shape[0], 1)), (y.shape[0], x.shape[0])) - \
           2 * tf.matmul(y, tf.transpose(x))


def accuracy(y_true, y_pred):
    same = tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1))
    return tf.reduce_mean(tf.cast(same, tf.float32))


def generate_exp_string(profile: dict, data_type=True):
    exp_string = ''
    for x in list(profile):
        value = profile[x].__name__ if hasattr(profile[x], '__name__') else str(profile[x])
        if data_type:
            exp_string += '{}:{}:{}-'.format(x, type(profile[x]).__name__, value)
        else:
            exp_string += '{}:{}-'.format(x, value)
    return exp_string[:-1]