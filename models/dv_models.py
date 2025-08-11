import numpy as np
import tensorflow as tf
from tensorflow import keras
from models.layers import ContrastiveGaussianNoiseLayer, ContrastiveDiscreteUniformNoiseLayer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Layer, Dense, Concatenate, Lambda, LSTM, LSTMCell, RNN, Dropout, \
    LayerNormalization, Activation
from tensorflow.python.keras import initializers
import wandb
from time import time

tf.keras.backend.set_floatx('float64')


class MINE(Model):
    """
    """

    def __init__(self, channel,
                 hidden_size=500,
                 num_layers=2,
                 embedding_size=50,
                 contrastive_duplicates=5):
        super(MINE, self).__init__()
        self.channel = channel
        self.contrastive_duplicates = contrastive_duplicates
        self.dense_layers = [Dense(hidden_size, activation='elu') for k in range(num_layers)]
        self.dense_layers += [Dense(embedding_size, activation='elu')]
        self.last_layer = Dense(2, activation=None)

    def get_config(self):
        pass

    @tf.function
    def llr(self, y):
        e = self.__call__(y)
        t = self.last_layer(e)

        t0 = tf.slice(t, [0, 0, 0], [tf.shape(t)[0], tf.shape(t)[1], 1])
        t1 = tf.slice(t, [0, 0, 1], [tf.shape(t)[0], tf.shape(t)[1], 1])
        return t0 - t1

    @tf.function
    def _forward(self, y, training=None, **kwargs):
        e = y
        for layer in self.dense_layers:
            e = layer(e, training=training)
        return e

    @tf.function
    def __call__(self, inputs, training=None, **kwargs):
        e = self._forward(inputs, training=training, **kwargs)
        return e

    @tf.function
    def channel_stats(self, inputs, training=None, **kwargs):
        e = self._forward(inputs, training=training, **kwargs)
        return e

    @tf.function
    def train_step(self, data):
        x, y = data
        y_tilde = tf.concat([tf.random.shuffle(y) for _ in range(self.contrastive_duplicates)], axis=0)
        X = tf.concat([1 - x, x], axis=-1)
        with tf.GradientTape() as tape:
            e = self.__call__(y)
            t = self.last_layer(e)
            t = tf.reduce_sum(X * t, axis=-1, keepdims=True)

            e_ = self.__call__(y_tilde)
            t_ = self.last_layer(e_)
            t_ = tf.reduce_sum(tf.tile(X, [self.contrastive_duplicates, 1]) * t_, axis=-1, keepdims=True)

            loss = self.compiled_loss(t, t_)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(t, t_)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def train(self, num_iters, batch, log_freq=100, wandb=None):
        mi = 0.0
        for i in range(num_iters):
            data = self.channel.gen_inputs_and_outputs(batch, )
            dic = self.train_step(data)
            mi = dic["mi"].numpy()
            if wandb is not None:
                wandb.log({"iter_mi": iter, "mi": mi})
            if i % log_freq == 0:
                print(f'iter: {i: 4d} MI: {mi: 5.4f}')
                self.compiled_metrics.reset_state()

        print(f'channel: {self.channel.name}, est. rate: {mi: 5.4f}')
        return mi


class RelativeEntropyModel(Model):
    """
    """

    def __init__(self, channel, num_layers_dine=1, lstm_hidden_size=100, fc_hidden_size=(100,),
                 max_norm=None, contrastive_duplicates=5, clip_grad_norm=10.0,
                 activation="tanh", contrastive_noise_dist='uniform',
                 noise_params=None, batch_size_eval=None,  input_shape=(1, 1, 1), *args, **kwargs):
        super(RelativeEntropyModel, self).__init__()

        self.channel = channel
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else input_shape[0]
        self.contrastive_duplicates = contrastive_duplicates
        self.clip_grad_norm = clip_grad_norm

        if contrastive_noise_dist == "uniform_discrete":
            self.contrastive_noise_layer = ContrastiveDiscreteUniformNoiseLayer(minval=noise_params["minval"],
                                                                                maxval=noise_params["maxval"],
                                                                                name="contrastive_uniform_noise")
        elif contrastive_noise_dist == "normal":
            self.contrastive_noise_layer = ContrastiveGaussianNoiseLayer(mean=noise_params["mean"],
                                                                         std=noise_params["std"],
                                                                         name="contrastive_gaussian_noise")
        else:
            raise ValueError("invalid contrastive noise distribution was selected")

        self.input_shape_ = input_shape
        self.input_shape_lstm = input_shape
        lstm_layers = list()
        prev_layer_dim = input_shape[-1]
        for k in range(num_layers_dine):
            cell = LSTMCell(lstm_hidden_size)
            lstm = RNN(cell, return_sequences=True, stateful=True)
            layer_input_shape = tuple([input_shape[0], input_shape[1], prev_layer_dim])
            lstm.build(layer_input_shape)
            lstm_layers.append(lstm)
            prev_layer_dim = lstm_hidden_size

        self.lstm = Sequential(lstm_layers)
        self.lstm.build(self.input_shape_)

        self.input_shape_fc = list(np.insert(self.input_shape_, 2, 1))
        self.input_shape_fc[2] = None
        self.input_shape_fc[3] = input_shape[-1] + self.lstm.output_shape[-1]

        last_layer_kwargs = dict()
        if max_norm is not None:
            last_layer_kwargs.update({"kernel_constraint": keras.constraints.MaxNorm(max_norm)})
        self.fc = Sequential([Dense(fc_hidden_size, activation=activation, kernel_initializer='he_uniform')
                              for _ in range(num_layers_dine)])
        self.last_layer = Dense(1, **last_layer_kwargs)

    def get_config(self):
        pass

    @tf.function
    def forward(self, y, training=None, n_aux=None):
        n_aux = self.contrastive_duplicates if n_aux is None else n_aux
        noise = self.contrastive_noise_layer(y, n_aux)

        last_state = tf.identity(self.lstm.layers[-1].states[0])
        o1 = self.lstm(y, training=training)
        o1_ = tf.slice(o1, [0, 0, 0], [tf.shape(o1)[0], tf.shape(o1)[1] - 1, tf.shape(o1)[2]])
        s = tf.concat([tf.expand_dims(last_state, 1), o1_], axis=1)

        true_data = tf.concat([tf.expand_dims(s, 2), tf.expand_dims(y, 2)], axis=-1)
        noise_list = tf.split(noise, num_or_size_splits=n_aux, axis=2)
        fake_data = [tf.concat([tf.expand_dims(s, 2), n], axis=-1) for n in noise_list]
        fake_data = tf.concat(fake_data, axis=2)
        data = tf.concat([true_data, fake_data], axis=2)
        o2 = self.fc(data, training=training)
        t, t_ = tf.split(o2, num_or_size_splits=[1, n_aux], axis=2)

        return t, t_

    @tf.function
    def __call__(self, inputs, training=None, **kwargs):
        y = inputs
        o, o_ = self.forward(y, training=training)
        t, t_ = self.last_layer(o), self.last_layer(o_)
        t = tf.squeeze(t, -1)
        t_ = tf.squeeze(t_, -1)
        return t, t_

    @tf.function
    def channel_stats(self, inputs, training=None, **kwargs):
        y = inputs
        o, _ = self.forward(y, training=training)
        return tf.squeeze(o, axis=2)

    def reset_states(self):
        self.lstm.reset_states()

    @tf.function
    def train_step(self, data):
        y = data
        with tf.GradientTape() as tape:
            t, t_ = self(y, training=True)
            loss = self.compiled_loss(t, t_)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        gradients, grad_norm = tf.clip_by_global_norm(gradients, self.clip_grad_norm)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(t, t_)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def train(self, num_iters, batch, log_freq=100, wandb=None, decay=0.99):
        bptt = self.input_shape_[1]
        mi = 0.0
        self.reset_states()
        for i in range(num_iters):
            x, y = self.channel.gen_inputs_and_outputs(batch, bptt=bptt)
            dic = self.train_step(y)
            mi = dic["mi"].numpy()
            if wandb is not None:
                wandb.log({"iter_mi": iter, "mi": mi})
            if i % log_freq == 0:
                print(f'iter: {i: 4d} MI: {mi: 5.4f}')
                self.compiled_metrics.reset_state()

        print(f'channel: {self.channel.name}, est. rate: {mi: 5.4f}')


class CausallyConditionedRelativeEntropyModel(Model):
    """
    """

    def __init__(self, channel, num_layers_dine=1, lstm_hidden_size=100, fc_hidden_size=(100,),
                 max_norm=None, contrastive_duplicates=5, clip_grad_norm=10.0,
                 activation="tanh", contrastive_noise_dist='uniform', embedding_size=50,
                 noise_params=None, batch_size_eval=None,  input_shape=(1, 1, 1), *args, **kwargs):
        super(CausallyConditionedRelativeEntropyModel, self).__init__()

        self.channel = channel
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else input_shape[0]
        self.contrastive_duplicates = contrastive_duplicates
        self.clip_grad_norm = clip_grad_norm

        if contrastive_noise_dist == "uniform_discrete":
            self.contrastive_noise_layer = ContrastiveDiscreteUniformNoiseLayer(minval=noise_params["minval"],
                                                                                maxval=noise_params["maxval"],
                                                                                name="contrastive_uniform_noise")
        elif contrastive_noise_dist == "normal":
            self.contrastive_noise_layer = ContrastiveGaussianNoiseLayer(mean=noise_params["mean"],
                                                                         std=noise_params["std"],
                                                                         name="contrastive_gaussian_noise")
        else:
            raise ValueError("invalid contrastive noise distribution was selected")

        self.prev_e = self.add_weight("prev_e", shape=(input_shape[0], 1, embedding_size), dtype=tf.float64, trainable=False,
                                      initializer=initializers.constant(0.0, dtype=tf.float64))
        self.input_shape_ = input_shape
        self.input_shape_lstm = (input_shape[0], input_shape[1], input_shape[2] + embedding_size)
        lstm_layers = list()
        prev_layer_dim = self.input_shape_lstm[-1]
        for k in range(num_layers_dine):
            cell = LSTMCell(lstm_hidden_size)
            lstm = RNN(cell, return_sequences=True, stateful=True)
            layer_input_shape = tuple([input_shape[0], input_shape[1], prev_layer_dim])
            lstm.build(layer_input_shape)
            lstm_layers.append(lstm)
            prev_layer_dim = lstm_hidden_size

        self.lstm = Sequential(lstm_layers)
        self.lstm.build(self.input_shape_lstm)

        self.input_shape_fc = list(np.insert(self.input_shape_, 2, 1))
        self.input_shape_fc[2] = None
        self.input_shape_fc[3] = input_shape[-1] + self.lstm.output_shape[-1]

        last_layer_kwargs = dict()
        if max_norm is not None:
            last_layer_kwargs.update({"kernel_constraint": keras.constraints.MaxNorm(max_norm)})
        self.fc = Sequential([Dense(fc_hidden_size, activation=activation, kernel_initializer='he_uniform')
                              for _ in range(num_layers_dine)])
        self.last_layer = Dense(1, **last_layer_kwargs)
        cell = LSTMCell(embedding_size)
        lstm = RNN(cell, return_sequences=True, stateful=True)
        layer_input_shape = tuple([input_shape[0], None, input_shape[-1]])
        lstm.build(layer_input_shape)
        # self.f_enc = Sequential([lstm])
        self.f_enc = Sequential([Dense(embedding_size, activation=None, kernel_initializer='he_uniform')])
        # self.f_enc = Sequential([Dense(embedding_size, activation=None, kernel_initializer='he_uniform')])

    def get_config(self):
        pass

    @tf.function
    def forward(self, x, y, training=None, n_aux=None):
        n_aux = self.contrastive_duplicates if n_aux is None else n_aux
        noise = self.contrastive_noise_layer(y, n_aux)
        e = self.f_enc(y)
        # if training:
        #     e += tf.random.normal(tf.shape(e), stddev=0.05, dtype=tf.float64)
        noise_list = tf.split(noise, num_or_size_splits=n_aux, axis=2)
        e_noise = tf.stack([self.f_enc(tf.squeeze(n, 2)) for n in noise_list], axis=2)
        e_prev = tf.identity(self.prev_e)
        last_e = tf.slice(e, [0, tf.shape(e)[1] - 1, 0], [tf.shape(e)[0], 1, tf.shape(e)[2]])
        self.prev_e.assign(last_e)
        e_ = tf.slice(e, [0, 0, 0], [tf.shape(e)[0], tf.shape(e)[1] - 1, tf.shape(e)[2]])
        e_state = tf.concat([e_prev, e_], axis=1)
        xy_state = tf.concat([x, e_state], axis=-1)
        s = self.lstm(xy_state, training=training)

        true_data = tf.concat([tf.expand_dims(s, 2), tf.expand_dims(e, 2)], axis=-1)
        noise_list = tf.split(e_noise, num_or_size_splits=n_aux, axis=2)
        fake_data = [tf.concat([tf.expand_dims(s, 2), n], axis=-1) for n in noise_list]
        fake_data = tf.concat(fake_data, axis=2)
        data = tf.concat([true_data, fake_data], axis=2)
        o2 = self.fc(data, training=training)
        t, t_ = tf.split(o2, num_or_size_splits=[1, n_aux], axis=2)

        return t, t_

    @tf.function
    def __call__(self, inputs, training=None, **kwargs):
        x, y = inputs
        o, o_ = self.forward(x, y, training=training)
        t, t_ = self.last_layer(o), self.last_layer(o_)
        t = tf.squeeze(t, -1)
        t_ = tf.squeeze(t_, -1)
        return t, t_

    @tf.function
    def channel_stats(self, y, training=None, **kwargs):
        # x, y = inputs
        o = self.f_enc(y, training=training)
        return o
        # return tf.squeeze(o, axis=2)

    def reset_states(self):
        self.lstm.reset_states()

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            t, t_ = self(data, training=True)
            loss = self.compiled_loss(t, t_)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        gradients, grad_norm = tf.clip_by_global_norm(gradients, self.clip_grad_norm)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(t, t_)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def train(self, num_iters, batch, decay=1.0, log_freq=100):
        bptt = 10 # self.input_shape_[1]
        mi = 0.0
        self.reset_states()
        for i in range(num_iters):
            x, y = self.channel.gen_inputs_and_outputs(batch, bptt=bptt)
            dic = self.train_step((x, y))
            self.optimizer.learning_rate = self.optimizer.learning_rate * decay

            mi = dic["mi"].numpy()
            wandb.log({"iter_mi": i, "D_xy": mi})
            if i % log_freq == 0:
                print(f'iter: {i: 4d} MI: {mi: 5.4f} Lr:{self.optimizer.learning_rate.numpy(): 5.4e}')
                self.compiled_metrics.reset_state()
            # if i % (num_iters//3) ==0:

        print(f'channel: {self.channel.name}, est. rate: {mi: 5.4f}')


class DINE(Model):
    """
    """

    def __init__(self, data_gen, num_layers_dine=1, lstm_hidden_size=100, fc_hidden_size=(100,),
                 max_norm=None, contrastive_duplicates=5, clip_grad_norm=10.0,
                 activation="tanh", contrastive_noise_dist='uniform', embedding_size=50,
                 noise_params=None, batch_size_eval=None,  input_shape=(1, 1, 1), *args, **kwargs):
        super(DINE, self).__init__()

        self.data_gen = data_gen
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else input_shape[0]
        self.contrastive_duplicates = contrastive_duplicates
        self.clip_grad_norm = clip_grad_norm

        if contrastive_noise_dist == "uniform_discrete":
            self.contrastive_noise_layer = ContrastiveDiscreteUniformNoiseLayer(minval=noise_params["minval"],
                                                                                maxval=noise_params["maxval"],
                                                                                name="contrastive_uniform_noise")
        elif contrastive_noise_dist == "normal":
            self.contrastive_noise_layer = ContrastiveGaussianNoiseLayer(mean=noise_params["mean"],
                                                                         std=noise_params["std"],
                                                                         name="contrastive_gaussian_noise")
        else:
            raise ValueError("invalid contrastive noise distribution was selected")

        self.input_shape_ = input_shape
        self.input_shape_lstm_y = (input_shape[0], input_shape[1], input_shape[2])
        lstm_layers = list()
        prev_layer_dim = self.input_shape_lstm_y[-1]
        for k in range(num_layers_dine):
            hidden_size = embedding_size if k == num_layers_dine-1 else lstm_hidden_size
            cell = LSTMCell(hidden_size)
            lstm = RNN(cell, return_sequences=True, stateful=True)
            layer_input_shape = tuple([input_shape[0], input_shape[1], prev_layer_dim])
            lstm.build(layer_input_shape)
            lstm_layers.append(lstm)
            prev_layer_dim = hidden_size

        self.lstm_y = Sequential(lstm_layers, name="lstm_y")
        self.lstm_y.build(self.input_shape_lstm_y)

        self.input_shape_lstm_xy = (input_shape[0], input_shape[1], input_shape[2] + embedding_size)
        lstm_layers = list()
        prev_layer_dim = self.input_shape_lstm_xy[-1]
        for k in range(num_layers_dine):
            hidden_size = embedding_size if k == num_layers_dine-1 else lstm_hidden_size
            cell = LSTMCell(hidden_size)
            lstm = RNN(cell, return_sequences=True, stateful=True)
            layer_input_shape = tuple([input_shape[0], input_shape[1], prev_layer_dim])
            lstm.build(layer_input_shape)
            lstm_layers.append(lstm)
            prev_layer_dim = hidden_size

        self.lstm_xy = Sequential(lstm_layers, name="lstm_xy")
        self.lstm_xy.build(self.input_shape_lstm_xy)

        self.input_shape_fc = list(np.insert(self.input_shape_, 2, 1))
        self.input_shape_fc[2] = None
        self.input_shape_fc[3] = input_shape[-1] + self.lstm_y.output_shape[-1]

        last_layer_kwargs = dict()
        if max_norm is not None:
            last_layer_kwargs.update({"kernel_constraint": keras.constraints.MaxNorm(max_norm)})
        self.fc = Sequential([Dense(fc_hidden_size,
                                    activation=activation,
                                    kernel_initializer='he_uniform')] +
                             [Dense(embedding_size,
                                    activation=activation,
                                    kernel_initializer='he_uniform')]
                             )
        self.last_layer = Dense(1, **last_layer_kwargs)

    def get_config(self):
        pass

    @tf.function
    def forward(self, x, y, training=None, n_aux=None):
        n_aux = self.contrastive_duplicates if n_aux is None else n_aux
        noise = self.contrastive_noise_layer(y, n_aux)

        last_state = tf.identity(self.lstm_y.layers[-1].states[0])
        o1 = self.lstm_y(y, training=training)
        o1_ = tf.slice(o1, [0, 0, 0], [tf.shape(o1)[0], tf.shape(o1)[1] - 1, tf.shape(o1)[2]])
        s = tf.concat([tf.expand_dims(last_state, 1), o1_], axis=1)

        true_data = tf.concat([tf.expand_dims(s, 2), tf.expand_dims(y, 2)], axis=-1)
        noise_list = tf.split(noise, num_or_size_splits=n_aux, axis=2)
        fake_data = [tf.concat([tf.expand_dims(s, 2), n], axis=-1) for n in noise_list]
        fake_data = tf.concat(fake_data, axis=2)
        data = tf.concat([true_data, fake_data], axis=2)
        o2 = self.fc(data, training=training)
        ey, ey_ = tf.split(o2, num_or_size_splits=[1, n_aux], axis=2)

        xy_state = tf.concat([x, tf.squeeze(ey, axis=2)], axis=-1)
        s = self.lstm_xy(xy_state, training=training)

        true_data = tf.concat([tf.expand_dims(s, 2), tf.expand_dims(y, 2)], axis=-1)
        fake_data = [tf.concat([tf.expand_dims(s, 2), n], axis=-1) for n in noise_list]
        fake_data = tf.concat(fake_data, axis=2)
        data = tf.concat([true_data, fake_data], axis=2)
        o2 = self.fc(data, training=training)
        exy, exy_ = tf.split(o2, num_or_size_splits=[1, n_aux], axis=2)

        return ey, ey_, exy, exy_

    @tf.function
    def __call__(self, inputs, training=None, **kwargs):
        x, y = inputs
        ey, ey_, exy, exy_ = self.forward(x, y, training=training)
        ty, ty_, txy, txy_ = self.last_layer(ey), self.last_layer(ey_), self.last_layer(exy), self.last_layer(exy_)
        ty = tf.squeeze(ty, -1)
        ty_ = tf.squeeze(ty_, -1)
        txy = tf.squeeze(txy, -1)
        txy_ = tf.squeeze(txy_, -1)
        return ty, ty_, txy, txy_

    @tf.function
    def channel_stats(self, y, training=None, **kwargs):
        last_state = tf.identity(self.lstm_y.layers[-1].states[0])
        o1 = self.lstm_y(y, training=training)
        o1_ = tf.slice(o1, [0, 0, 0], [tf.shape(o1)[0], tf.shape(o1)[1] - 1, tf.shape(o1)[2]])
        s = tf.concat([tf.expand_dims(last_state, 1), o1_], axis=1)

        true_data = tf.concat([tf.expand_dims(s, 2), tf.expand_dims(y, 2)], axis=-1)
        ey = self.fc(true_data, training=training)
        return tf.squeeze(ey, axis=2)
        # return tf.squeeze(o, axis=2)

    def reset_states(self):
        self.lstm_y.reset_states()
        self.lstm_xy.reset_states()

    @tf.function
    def train_step(self, data):
        with tf.GradientTape(persistent=True) as tape:
            ty, ty_, txy, txy_ = self(data, training=True)
            loss_xy = self.compiled_loss(txy, txy_)
            loss_y = self.compiled_loss(ty, ty_)
            loss = loss_xy + loss_y

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        gradients, grad_norm = tf.clip_by_global_norm(gradients, self.clip_grad_norm)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(tf.concat([txy, ty], axis=-1), tf.concat([txy_, ty_], axis=-1))
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def train(self, num_iters, batch, bptt=10,  decay=1.0, log_freq=100, verbose=True):
        mi = 0.0

        def permute(xx, yy):
            return tf.transpose(xx, [1, 0, 2]), tf.transpose(yy, [1, 0, 2])

        ds = tf.data.Dataset.from_generator(lambda: self.data_gen,
                                            output_types=(tf.float64, tf.float64),
                                            output_shapes=(tf.TensorShape([None, 1]),
                                                           tf.TensorShape([None, 1]))
                                            ).repeat().batch(bptt, drop_remainder=True).map(
            permute)

        self.reset_states()
        t = time()
        for i, (x, y) in enumerate(ds.take(num_iters)):
            # x, y = self.channel.gen_inputs_and_outputs(batch, bptt=bptt)
            dic = self.train_step((x, y))
            self.optimizer.learning_rate = self.optimizer.learning_rate * decay

            if i % log_freq == 0:
                mi = dic["di"].numpy()
                wandb.log({"iter_di": i, "di": mi})
                if verbose:
                    print(f'iter: {i: 4d} DI: {mi: 5.4f} Lr:{self.optimizer.learning_rate.numpy(): 5.4e} ' 
                          f'Elapsed time: {time() - t: 5.3f}')
                t = time()
                self.compiled_metrics.reset_state()
        if verbose:
            print(f'channel: {self.data_gen.channel.name}, est. rate: {mi: 5.4f}')


