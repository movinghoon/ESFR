import tensorflow as tf
from utils import accuracy, euclidean_distance


# Nearest Centroid
class NN(object):
    def __init__(self, centering=False, l2_normalize=False):
        self.centering = centering
        self.l2_normalize = l2_normalize

    @tf.function
    def get_logits_and_feats(self, feats, lbs):
        # Pre-processing
        feat_dim = feats.shape[-1]
        feats = tf.reshape(feats, (-1, feat_dim))
        if self.centering:
            feats = feats - tf.reduce_mean(feats, axis=0)

        if self.l2_normalize:
            feats = tf.math.l2_normalize(feats, axis=-1)

        feats = tf.reshape(feats, (5, -1, feat_dim))

        # CM
        supp_feats = feats[:, :-15]
        cm = tf.reduce_mean(supp_feats, axis=1)

        # Qry logit
        qry_feats = tf.reshape(feats[:, -15:], (-1, feat_dim))
        qry_logit = -1 * euclidean_distance(cm, qry_feats)

        # Supp logit
        supp_feats = tf.reshape(supp_feats, (-1, feat_dim))
        supp_logit = -1 * euclidean_distance(cm, supp_feats)
        return qry_logit, supp_logit, qry_feats, supp_feats

    def __call__(self, feats, lbs):
        qry_logit, _, _, _ = self.get_logits_and_feats(feats, lbs)
        qry_lbs = tf.reshape(tf.squeeze(lbs)[:, -15:], (-1, 5))
        return accuracy(qry_lbs, qry_logit)


# Nearest Centroid with Cosine-distance
class CSPN(NN):
    def __init__(self, centering=False, l2_normalize=False):
        super(CSPN, self).__init__(centering, l2_normalize)

    @tf.function
    def get_logits_and_feats(self, feats, lbs):
        # Pre-processing
        feat_dim = feats.shape[-1]
        feats = tf.reshape(feats, (-1, feat_dim))
        if self.centering:
            feats = feats - tf.reduce_mean(feats, axis=0)

        if self.l2_normalize:
            feats = tf.math.l2_normalize(feats, axis=-1)

        feats = tf.reshape(feats, (5, -1, feat_dim))

        # CM
        supp_feats = feats[:, :-15]
        cm = tf.reduce_mean(supp_feats, axis=1)

        # Cosine
        cm = tf.math.l2_normalize(cm, axis=-1)
        qry_feats = tf.reshape(feats[:, -15:], (-1, feat_dim))
        qry_feats = tf.math.l2_normalize(qry_feats, axis=-1)
        supp_feats = tf.math.l2_normalize(supp_feats, axis=-1)

        # Qry logit
        qry_logit = -1 * euclidean_distance(cm, qry_feats)

        # Supp logit
        supp_feats = tf.reshape(supp_feats, (-1, feat_dim))
        supp_logit = -1 * euclidean_distance(cm, supp_feats)
        return qry_logit, supp_logit, qry_feats, supp_feats


# Prototype Rectification Wrapper
# Also includes shiting-term option
class PRWrapper(object):
    def __init__(self, algorithm, alpha=0.2, supp_logit_by_label=True, shift=True):
        self.algorithm = algorithm
        self.alpha = alpha
        self.supp_logit_by_label = supp_logit_by_label
        self.shift = shift

    def rectified_prototype(self, supp_feats, supp_logits, qry_feats, qry_logits):
        feats = tf.concat([supp_feats, qry_feats], axis=0)

        # Pseudo Label
        pseudo_supp_lbs = supp_logits if self.supp_logit_by_label else tf.nn.softmax(supp_logits / self.alpha, axis=-1)
        pseudo_qry_lbs = tf.nn.softmax(qry_logits / self.alpha, axis=-1)
        pseudo_lbs = tf.concat([pseudo_supp_lbs, pseudo_qry_lbs], axis=0)
        predict = tf.argmax(pseudo_lbs, axis=-1)

        # rectified prototype
        mask = tf.one_hot(predict, depth=5)
        return tf.matmul(tf.transpose(mask * pseudo_lbs), feats) / tf.reshape(tf.reduce_sum(mask * pseudo_lbs, axis=0),
                                                                              (5, 1))

    def rectified_inference(self, supp_feats, supp_logits, qry_feats, qry_logits, lbs):
        # Reshape
        feat_dim = supp_feats.shape[-1]
        supp_feats = tf.reshape(supp_feats, (-1, feat_dim))
        qry_feats = tf.reshape(qry_feats, (-1, feat_dim))
        supp_logits = tf.reshape(supp_logits, (-1, 5))
        qry_logits = tf.reshape(qry_logits, (-1, 5))

        if self.shift:
            qry_feats = qry_feats - tf.reduce_mean(qry_feats, axis=0) + tf.reduce_mean(supp_feats, axis=0)  # shift

        # Labels
        qry_lbs = tf.reshape(tf.squeeze(lbs)[:, -15:], (-1, 5))
        supp_lbs = tf.reshape(tf.squeeze(lbs)[:, :-15], (-1, 5))
        if self.supp_logit_by_label:
            supp_logits = supp_lbs

        # Classify
        cm = self.rectified_prototype(supp_feats,
                                      supp_logits,
                                      qry_feats,
                                      qry_logits)
        qry_logit = -1 * euclidean_distance(cm, qry_feats)
        return accuracy(qry_lbs, qry_logit)

    def __call__(self, feats, lbs):
        qry_logit, supp_logit, qry_feats, supp_feats = self.algorithm.get_logits_and_feats(feats, lbs)
        return self.rectified_inference(supp_feats, supp_logit, qry_feats, qry_logit, lbs)


# BDCSPN algorithm
class BDCSPN(PRWrapper):
    def __init__(self, centering=False, l2_normalize=False, alpha=0.2, supp_logit_by_label=True, shift=False):
        algorithm = CSPN(centering=centering, l2_normalize=l2_normalize)
        super(BDCSPN, self).__init__(algorithm=algorithm, alpha=alpha, supp_logit_by_label=supp_logit_by_label,
                                     shift=shift)
