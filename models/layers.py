
import tensorflow as tf
keras = tf.keras
# from tensorflow.python import keras
import numpy as np
import settings as st

_INITIALIZER = 'glorot_normal'
# _INITIALIZER = 'glorot_uniform'
# _INITIALIZER = keras.initializers.TruncatedNormal(mean=0., stddev=0.001)

class pixel_bias(keras.layers.Layer):
    def __init__(self, trainable=True, **kwargs):
        super(pixel_bias, self).__init__(trainable=trainable, **kwargs)
        self.init_value = np.load(st.preprocessed_data_path+"%s/marg.npy"%(st.dataset_dict[st.dataset]))

    def build(self, inshape):
        self.b = self.add_weight(shape=self.init_value.shape,
                                    initializer=keras.initializers.constant(self.init_value),
                                    trainable=self.trainable,
                                 name="bias")
        super(pixel_bias, self).build(inshape)
    def call(self, inputs):
        return inputs + self.b

    def get_config(self):
        config = super(pixel_bias, self).get_config()
        return config

def reparameterize_func(inputs):
    mu = keras.layers.Reshape((1, 1, st.z_dim))(inputs[...,:st.z_dim])
    sig = keras.activations.exponential(inputs[...,st.z_dim:])
    sig = keras.layers.Reshape((1, 1, st.z_dim))(sig)
    esp = tf.random.normal(shape=tf.shape(sig), mean=0., stddev=0.1)
    return mu, sig, keras.layers.Reshape((1, 1, st.z_dim))(sig*esp+mu)

class reparameterize(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(reparameterize, self).__init__(**kwargs)
    def call(self, inputs):
        mu = inputs[...,:st.z_dim]
        sig = keras.activations.exponential(inputs[...,st.z_dim:])
        esp = tf.random.normal(shape=tf.shape(sig), mean=0., stddev=0.1)
        return sig*esp+mu

class shuffle_batch(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(shuffle_batch, self).__init__(**kwargs)

    def call(self, inputs):
        # rand_idx = tf.random.shuffle(tf.range(tf.shape(inputs)[0]))[:, None]
        return tf.gather_nd(inputs, tf.random.shuffle(tf.range(tf.shape(inputs)[0]))[:, None])
    def get_config(self):
        config = super(shuffle_batch, self).get_config()
        return config

class Layers:
    def _imperative(self, out, i):
        if i is not None:
            out = out(i)
        return out

    def concat(self, a, b, axis=-1):
        return keras.backend.concatenate([a, b], axis=axis)

    def reshape_choose_one(self, a, shape):
        a = keras.layers.Reshape((shape[1], shape[2], a.shape[-1]))(a)
        return keras.layers.Reshape((1, 1, a.shape[-1]))(a[:, 0, 0])

    def z_diff(self, a, b, shape, mode=0):
        a = tf.reshape(a, (-1, shape[1],shape[2], a.shape[-1]))
        b = keras.layers.Reshape((1, 1, b.shape[-1]))(b)

        if mode==0:
            tile_b = keras.backend.tile(b, [1, shape[1], shape[2], 1])
            out = self.concat(a, tile_b)
        elif mode==1:
            out = keras.backend.abs(a-b)
        elif mode==2:
            out = a+b
        return out

    def batch_norm(self, i=None, trainable=True, name=None, momentum=0.9, epsilon=1e-5):
        out = keras.layers.BatchNormalization(trainable=trainable, name=name, momentum=momentum, epsilon=epsilon)
        return self._imperative(out, i)

    def dropout(self, r, i=None, trainable=True, name=None):
        out = keras.layers.Dropout(rate=r, trainable=trainable, name=name)
        return self._imperative(out, i)

    def conv2d(self, f, k, s=1, p = "valid", act= None, bias=True, i=None, trainable=True, name=None):
        out = keras.layers.Conv2D(filters=f, kernel_size=k, strides=s, padding=p, activation=act, use_bias=bias,
                                  trainable=trainable, name=name,
                                  kernel_initializer=_INITIALIZER, bias_initializer='zeros',
                                  kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                  kernel_constraint=None, bias_constraint=None)
        return self._imperative(out, i)

    def deconv2d(self, f, k, s, p="valid", act= None, bias=True, bias_init="zeros", i=None, name=None, trainable=True):
        out = keras.layers.Conv2DTranspose(filters=f, kernel_size=k, strides=s, padding=p, output_padding=None, activation=act,
                                           trainable = trainable, name = name,use_bias=bias,
                                           data_format=None, dilation_rate=(1, 1),
                                           kernel_initializer=_INITIALIZER, bias_initializer=bias_init,
                                           kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                           kernel_constraint=None, bias_constraint=None)
        return self._imperative(out, i)

    def conv1d(self, f, k, s=1, p = "valid", act= None, bias=True, i=None, trainable=True, name=None):
        out = keras.layers.Conv1D(filters=f, kernel_size=k, strides=s, padding=p, activation=act, use_bias=bias,
                                  trainable=trainable, name=name,
                                  kernel_initializer=_INITIALIZER, bias_initializer='zeros',
                                  kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                  kernel_constraint=None, bias_constraint=None)
        return self._imperative(out, i)

    def dense(self, f, act= None, bias=True, i=None, trainable=True, name=None):
        out = keras.layers.Dense(units=f, activation=act, use_bias=bias,
                                 kernel_initializer=_INITIALIZER, bias_initializer='zeros', kernel_regularizer=None,
                                 trainable=trainable, name=name,
                                 bias_regularizer=None, activity_regularizer=None,
                                 kernel_constraint=None, bias_constraint=None)
        return self._imperative(out, i)

    def dense_layer(self, f, do = 0., bn=True, act = keras.layers.LeakyReLU(st.lrelu_alpha), bias = True, trainable=True, name=None):
        if name is not None: name="dense_%s"%name
        model = keras.Sequential(name=name)
        if do and st.do:
            model.add(self.dropout(r=do, trainable=trainable))
        model.add(self.dense(f=f, bias=bias, trainable=trainable))
        if bn and st.bn:
            model.add(self.batch_norm(trainable=trainable))
        if act is not None:
            model.add(act)
        return model

    def conv1d_layer(self, f, k, s=1, p = "valid", do = 0., bn=True, act = keras.layers.LeakyReLU(st.lrelu_alpha), bias = True, trainable=True, name=None):
        if name is not None: name="conv1d_%s"%name
        model = keras.Sequential(name=name)
        if do and st.do:
            model.add(self.dropout(r=do, trainable=trainable, name=name))
        model.add(self.conv1d(f=f, k=k, s=s, p = p, bias=bias, trainable=trainable, name=name))
        if bn and st.bn:
            model.add(self.batch_norm(trainable=trainable, name=name))
        if act is not None:
            model.add(act)
        return model

    def conv2d_layer(self, f, k, s=1, p = "valid", do = 0., bn=True, act = keras.layers.LeakyReLU(st.lrelu_alpha), bias = True, trainable=True, name=None):
        if name is not None: name="conv2d_%s"%name
        model = keras.Sequential(name=name)
        if do and st.do:
            model.add(self.dropout(r=do, trainable=trainable, name=name))
        model.add(self.conv2d(f=f, k=k, s=s, p = p, bias=bias, trainable=trainable, name=name))
        if bn and st.bn:
            model.add(self.batch_norm(trainable=trainable, name=name))
        if act is not None:
            model.add(act)
        return model

    def deconv2d_layer(self, f, k, s=1, p = "valid", do = 0., bn=True, act = keras.layers.LeakyReLU(st.lrelu_alpha), bias = True, bias_init= "zeros", trainable=True, name=None):
        if name is not None: name="deconv2d_%s"%name
        model = keras.Sequential(name=name)
        if do and st.do:
            model.add(self.dropout(r=do, trainable=trainable, name=name))
        model.add(self.deconv2d(f=f, k=k, s=s, p=p, bias=bias, bias_init=bias_init, name=name, trainable=trainable))
        if bn and st.bn:
            model.add(self.batch_norm(trainable=trainable, name=name))
        if act is not None:
            model.add(act)
        return model
