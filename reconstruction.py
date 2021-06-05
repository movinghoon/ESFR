import tensorflow as tf
from tensorflow.keras.layers import Dense
from utils import get_lid_score


class ReconstructionModule(tf.keras.Model):
    def __init__(self, input_dim, hidden=(640, 640, 640, 640)):
        super(ReconstructionModule, self).__init__()
        self.feat_dim = list(hidden)[-1]
        self.layer_dict = dict()

        # Layers
        self.num_dense = len(hidden)
        for i in range(self.num_dense):
            self.layer_dict['dense_{}'.format(i)] = Dense(list(hidden)[i], activation=tf.nn.relu)

        # Build
        self.build((None, input_dim))

    def call(self, inputs, training=True, mask=None):
        x = inputs
        for i in range(self.num_dense):
            x = self.layer_dict['dense_{}'.format(i)](x)
        return x

    @tf.function
    def get_lid(self, inputs, k):
        x = inputs
        for i in range(self.num_dense - 2):     # LID from hidden representations of second-to-last layer
            x = self.layer_dict['dense_{}'.format(i)](x)
        return get_lid_score(x, k)

    @tf.function
    def get_feats(self, inputs):
        return self.call(inputs, training=False, mask=None)

    def reset(self):
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.kernel.assign(layer.kernel_initializer(layer.kernel.shape))
                if layer.use_bias:
                    layer.bias.assign(tf.zeros_like(layer.bias))


class ESFR:
    def __init__(self,
                 net,
                 drop_rate=0.5,
                 num_ensemble=5,
                 centering=True,
                 lam=1.,
                 weight_decay=0.,
                 k=21,
                 period=2,
                 max_updates=100,
                 way=5):
        self.net = net

        # Params I
        self.way = way
        self.lam = lam
        self.weight_decay = weight_decay
        self.centering = centering
        if isinstance(drop_rate, float):
            drop_rate = [drop_rate] * num_ensemble
        assert isinstance(drop_rate, list)
        if len(drop_rate) < num_ensemble:
            drop_rate += [drop_rate[-1]] * (num_ensemble - len(drop_rate))
        self.drop_rate = [float(x) for x in drop_rate]

        # Params II
        self.num_ensemble = num_ensemble
        self.k = k
        self.period = period
        self.max_updates = max_updates

        # For support loss
        self.weight = tf.Variable(tf.zeros(shape=(self.net.feat_dim, self.way)), trainable=True)
        self.bias = tf.Variable(tf.zeros(shape=(1, self.way)), trainable=True)

    def _reset(self):
        self.net.reset()
        if self.lam > 0.:
            self.weight.assign(tf.zeros(shape=(self.net.feat_dim, self.way)))
            self.bias.assign(tf.zeros(shape=(1, self.way)))

    def get_trainable_variables(self):
        trainable_variables = self.net.trainable_variables
        return trainable_variables + [self.weight, self.bias] if self.lam > 0. else trainable_variables

    def _support_loss(self, feats, supp_lbs):
        feat_dim = feats.shape[-1]
        reconstructed_feats = self.net(feats)

        # Pre-process
        reconstructed_feats = tf.math.l2_normalize(reconstructed_feats, axis=-1)
        supp_feats = tf.reshape(tf.reshape(reconstructed_feats, (self.way, -1, feat_dim))[:, :-15], (-1, feat_dim))

        # Inference
        supp_logit = tf.matmul(supp_feats, self.weight) + self.bias
        supp_lbs = tf.reshape(supp_lbs, (-1, self.way))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=supp_lbs, logits=supp_logit))
        return loss

    def _reconstruction_loss(self, feats, drop_rate):
        # Noisy feat
        noisy_feats = tf.nn.dropout(feats, rate=drop_rate)

        # Feats
        reconstructed_feats = self.net(noisy_feats, training=True)
        if self.centering:
            reconstructed_feats = reconstructed_feats - tf.reduce_mean(reconstructed_feats, axis=0, keepdims=True)
        feats = tf.math.l2_normalize(feats, axis=-1)
        reconstructed_feats = tf.math.l2_normalize(reconstructed_feats, axis=-1)
        loss = -1 * tf.reduce_mean(tf.reduce_sum(feats * reconstructed_feats, axis=-1))
        return loss

    def get_loss(self, feats, supp_lbs, drop_rate):
        feat_dim = feats.shape[-1]
        feats = tf.reshape(feats, (-1, feat_dim))

        # Loss
        loss = self._reconstruction_loss(feats, drop_rate=drop_rate)
        if self.lam > 0.:
            loss += self.lam * self._support_loss(feats, supp_lbs)
        return loss

    @tf.function
    def get_grad(self, feats, supp_lbs, drop_rate):
        trainable_variables = self.get_trainable_variables()
        with tf.GradientTape() as tape:
            tape.watch(trainable_variables)
            loss = self.get_loss(feats, supp_lbs, drop_rate)
            if self.weight_decay > 0.:
                for x in self.net.trainable_variables:
                    loss += self.weight_decay * tf.nn.l2_loss(x)
        grad = tape.gradient(loss, trainable_variables)
        return grad

    def _get_feats(self, feats, supp_lbs, drop_rate):
        # Reset
        self._reset()
        optimizer = tf.optimizers.Adam(learning_rate=1e-3)

        # Init
        new_feats = self.net.get_feats(feats)
        prev_lid = self.net.get_lid(feats, k=self.k)
        for i in range(self.max_updates):
            grad = self.get_grad(feats, supp_lbs, drop_rate=drop_rate)
            optimizer.apply_gradients(zip(grad, self.get_trainable_variables()))

            if (i + 1) % self.period == 0:
                lid = self.net.get_lid(feats, k=self.k)
                new_feats = self.net.get_feats(feats)
                if lid > prev_lid:
                    break
                prev_lid = lid
        return new_feats

    def get_feats(self, feats, supp_lbs, centering=False):
        # Reshape
        feats = tf.reshape(feats, (-1, feats.shape[-1]))

        # Ensemble
        temp = []
        for i in range(self.num_ensemble):
            new_feats = self._get_feats(feats, supp_lbs, drop_rate=self.drop_rate[i])
            if centering:
                new_feats = new_feats - tf.reduce_mean(new_feats, axis=0)
            new_feats = tf.math.l2_normalize(new_feats, axis=-1)
            temp.append(tf.expand_dims(new_feats, axis=0))
        return tf.reduce_mean(temp, axis=0)