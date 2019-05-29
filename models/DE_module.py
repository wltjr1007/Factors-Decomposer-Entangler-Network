

import settings as st
import tensorflow as tf
keras = tf.keras
import models.AE_module
import models.layers


class DE_omniglot(models.AE_module.AE_omniglot):
    def __init__(self, trainable):
        super(DE_omniglot, self).__init__(trainable=False)
        self.trainable = trainable
        self.dim = 128

    def _disentangler(self, name):
        m = keras.Sequential(name=name)
        m.add(self.dense_layer(f=self.dim, do = 0.2, bn=True, bias = False, trainable=self.trainable, name="1"))
        m.add(self.dense_layer(f=self.dim, do = 0.2, bn=False, bias = False, trainable=self.trainable, name="2"))
        m.add(self.dense_layer(f=self.dim, do = 0.2, bn=False, bias = False, trainable=self.trainable, name="3"))
        m.add(self.dense_layer(f=st.z_dim*2, do = 0.2, bn=False, bias = False, trainable=self.trainable, name="4", act=None))
        return m
    def _disentangler_id(self, name):
        m = keras.Sequential(name=name)
        m.add(self.dense_layer(f=self.dim, do = 0.2, bn=True, bias = False, trainable=self.trainable, name="1"))
        m.add(self.dense_layer(f=self.dim, do = 0.2, bn=False, bias = False, trainable=self.trainable, name="2"))
        m.add(self.dense_layer(f=self.dim, do = 0.2, bn=False, bias = False, trainable=self.trainable, name="3"))
        m.add(self.dense_layer(f=st.z_dim, do = 0.2, bn=False, bias = False, trainable=self.trainable, name="4", act=None))
        return m
    def _disentangler_sty(self, name):
        m = keras.Sequential(name=name)
        m.add(self.dense_layer(f=self.dim, do = 0.2, bn=True, bias = False, trainable=self.trainable, name="1"))
        m.add(self.dense_layer(f=self.dim, do = 0.2, bn=False, bias = False, trainable=self.trainable, name="2"))
        m.add(self.dense_layer(f=self.dim, do = 0.2, bn=False, bias = False, trainable=self.trainable, name="3"))
        m.add(self.dense_layer(f=st.z_dim, do = 0.2, bn=False, bias = False, trainable=self.trainable, name="4", act=None))
        return m

    def _entangler(self, name):
        m = keras.Sequential(name=name)
        m.add(self.dense_layer(f=self.dim, do = 0.2, bn=True, bias = False, trainable=self.trainable, name="1"))
        m.add(self.dense_layer(f=self.dim, do = 0.2, bn=False, bias = False, trainable=self.trainable, name="2"))
        m.add(self.dense_layer(f=self.dim, do = 0.2, bn=False, bias = False, trainable=self.trainable, name="3"))
        m.add(self.dense_layer(f=st.z_dim*2, do = 0.2, bn=False, bias = False, trainable=self.trainable, name="4", act=None))
        # m = models.layers.reparameterize_func()
        # m.add(models.layers.reparameterize())
        m.add(keras.layers.Reshape((1, 1, st.z_dim*2)))
        return m
    def _entangler_id(self, name):
        m = keras.Sequential(name=name)
        m.add(self.dense_layer(f=self.dim, do = 0.2, bn=True, bias = False, trainable=self.trainable, name="1"))
        m.add(self.dense_layer(f=self.dim, do = 0.2, bn=False, bias = False, trainable=self.trainable, name="2"))
        m.add(self.dense_layer(f=self.dim, do = 0.2, bn=False, bias = False, trainable=self.trainable, name="3"))
        m.add(self.dense_layer(f=st.z_dim, do = 0.2, bn=False, bias = False, trainable=self.trainable, name="4", act=None))
        return m
    def _entangler_sty(self, name):
        m = keras.Sequential(name=name)
        m.add(self.dense_layer(f=self.dim, do = 0.2, bn=True, bias = False, trainable=self.trainable, name="1"))
        m.add(self.dense_layer(f=self.dim, do = 0.2, bn=False, bias = False, trainable=self.trainable, name="2"))
        m.add(self.dense_layer(f=self.dim, do = 0.2, bn=False, bias = False, trainable=self.trainable, name="3"))
        m.add(self.dense_layer(f=st.z_dim, do = 0.2, bn=False, bias = False, trainable=self.trainable, name="4", act=None))
        return m

    def _siamese(self, name):
        m = keras.Sequential(name=name)
        m.add(self.dense_layer(f=st.z_dim, do = 0.5, bn=True, bias = False, trainable=self.trainable, name="1"))
        m.add(self.dense_layer(f=st.z_dim//2, do = 0.5, bn=False, bias = False, trainable=self.trainable, name="2"))
        m.add(self.dense_layer(f=st.z_dim//3, do = 0.5, bn=False, bias = False, trainable=self.trainable, name="3"))
        m.add(self.dense_layer(f=st.z_dim//4, do = 0.5, bn=False, bias = False, trainable=self.trainable, name="4"))
        m.add(self.dense_layer(f=1, do = 0.5, bn=False, bias = False, trainable=self.trainable, name="5", act=None))
        return m
    def _prototypical(self, name):
        m = keras.Sequential(name=name)
        m.add(self.dense_layer(f=st.z_dim, do = 0.5, bn=True, bias = False, trainable=self.trainable, name="1"))
        m.add(self.dense_layer(f=st.z_dim, do = 0.5, bn=False, bias = False, trainable=self.trainable, name="2"))
        m.add(self.dense_layer(f=st.z_dim//2, do = 0.5, bn=False, bias = False, trainable=self.trainable, name="3"))
        m.add(self.dense_layer(f=st.z_dim//2, do = 0.5, bn=False, bias = False, trainable=self.trainable, name="4", act=None))
        return m

    def _statistician_z(self, name):
        m = keras.Sequential(name=name)
        m.add(self.dense_layer(f=st.z_dim, do = 0.5, bn=True, bias = False, trainable=self.trainable, name="1"))
        m.add(self.dense_layer(f=st.z_dim, do = 0.5, bn=False, bias = False, trainable=self.trainable, name="2", act=None))
        return m
    def _statistician_all(self, name):
        m = keras.Sequential(name=name)
        m.add(self.dense_layer(f=st.z_dim*2, do = 0.5, bn=True, bias = False, trainable=self.trainable, name="1"))
        m.add(self.dense_layer(f=st.z_dim, do = 0.5, bn=False, bias = False, trainable=self.trainable, name="2"))
        m.add(self.dense_layer(f=st.z_dim//2, do = 0.5, bn=False, bias = False, trainable=self.trainable, name="3"))
        m.add(self.dense_layer(f=1, do = 0.5, bn=False, bias = False, trainable=self.trainable, name="4", act=None))
        return m

class DE_msceleb(models.AE_module.AE_msceleb):
    def __init__(self, trainable):
        super(DE_msceleb, self).__init__(trainable=False)
        self.trainable = trainable

    def _entangler(self, name):
        a=0
    def _disentangler(self, name):
        a=0
    def _siamese(self, name):
        a=0
    def _prototypical(self, name):
        a=0

    def _statistician_x(self, name):
        a=0
    def _statistician_z(self, name):
        a=0
    def _statistician_all(self, name):
        a=0

class DE_imagenet(models.AE_module.AE_imagenet):
    def __init__(self, trainable):
        super(DE_imagenet, self).__init__(trainable=False)
        self.trainable = trainable

    def _entangler(self, name):
        a=0
    def _disentangler(self, name):
        a=0
    def _siamese(self, name):
        a=0
    def _prototypical(self, name):
        a=0

    def _statistician_x(self, name):
        a=0
    def _statistician_z(self, name):
        a=0
    def _statistician_all(self, name):
        a=0


# if st.dataset==0: de_network=DE_omniglot
# elif st.dataset==1: de_network=DE_msceleb
# elif st.dataset==2: de_network=DE_imagenet
# else: de_network=DE_imagenet
de_network = DE_omniglot
