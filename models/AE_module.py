
import models.layers
import settings as st
import tensorflow as tf
keras = tf.keras


class AE_omniglot(models.layers.Layers):
    def __init__(self, trainable):
        super(AE_omniglot, self).__init__()
        self.trainable = trainable

    def _encoder(self, name):
        m = keras.Sequential(name=name)
        m.add(self.conv2d_layer(f=32, k=5, s=1, do = 0., bn=True, bias = False, trainable=self.trainable, name="1"))
        m.add(self.conv2d_layer(f=64, k=4, s=2, do = 0., bn=True, bias = False, trainable=self.trainable, name="2"))
        m.add(self.conv2d_layer(f=64, k=4, s=1, do = 0., bn=True, bias = False, trainable=self.trainable, name="3"))
        m.add(self.conv2d_layer(f=128, k=4, s=2, do = 0., bn=True, bias = False, trainable=self.trainable, name="4"))
        m.add(self.conv2d_layer(f=128, k=4, s=1, do = 0., bn=True, bias = False, trainable=self.trainable, name="5"))
        m.add(self.conv2d_layer(f=256, k=1, s=1, do = 0., bn=True, bias = False, trainable=self.trainable, name="6"))
        m.add(self.conv2d_layer(f=st.z_dim*2, k=1, s=1, do = 0., bn=False, bias = True, trainable=self.trainable, act=None, name="7"))
        # m.add(models.layers.reparameterize())
        # m.add(self.conv2d_layer(f=st.z_dim, k=1, s=1, do = 0., bn=False, bias = True, trainable=self.trainable, act=None, name="7"))
        # m.add(keras.layers.Activation("tanh"))
        return m

    def _decoder(self, name):
        m = keras.Sequential(name=name)
        m.add(self.conv2d_layer(f=st.z_dim*2, k=1, s=1, do = 0., bn=True, bias = False, trainable=self.trainable, name="0"))
        m.add(self.deconv2d_layer(f=256, k=4, s=1, bn=True, bias = False, trainable=self.trainable, name="1"))
        m.add(self.deconv2d_layer(f=128, k=4, s=2, bn=True, bias = False, trainable=self.trainable, name="2"))
        m.add(self.deconv2d_layer(f=64, k=4, s=1, bn=True, bias = False, trainable=self.trainable, name="3"))
        m.add(self.deconv2d_layer(f=32, k=4, s=2, bn=True, bias = False, trainable=self.trainable, name="4"))
        m.add(self.deconv2d_layer(f=32, k=5, s=1, bn=True, bias = False, trainable=self.trainable, name="5"))
        m.add(self.conv2d_layer(f=32, k=1, s=1, bn=True, bias = False, trainable=self.trainable, name="6"))
        m.add(self.conv2d_layer(f=st.C, k=1, s=1, bn=False, bias = False, trainable=self.trainable, name="7", act=None))
        m.add(models.layers.pixel_bias(trainable=self.trainable))
        m.add(keras.layers.Activation("sigmoid"))
        return m

    def _x_disc(self, name):
        m = keras.Sequential(name=name)
        m.add(self.conv2d_layer(f=32, k=5, s=1, do = 0.2, bn=False, bias = True, trainable=self.trainable, name="1"))
        m.add(self.conv2d_layer(f=64, k=4, s=2, do = 0.2, bn=True, bias = False, trainable=self.trainable, name="2"))
        m.add(self.conv2d_layer(f=128, k=4, s=1, do = 0.2, bn=True, bias = False, trainable=self.trainable, name="3"))
        m.add(self.conv2d_layer(f=256, k=4, s=2, do = 0.2, bn=True, bias = False, trainable=self.trainable, name="4"))
        m.add(self.conv2d_layer(f=512, k=4, s=1, do = 0.2, bn=True, bias = False, trainable=self.trainable, name="5"))
        return m

    def _z_disc(self, name):
        m = keras.Sequential(name=name)
        m.add(self.conv2d_layer(f=512, k=1, s=1, do = 0.2, bn=False, bias = True, trainable=self.trainable, name="1"))
        m.add(self.conv2d_layer(f=512, k=1, s=1, do = 0.2, bn=False, bias = True, trainable=self.trainable, name="2"))
        return m

    def _discriminator(self, name):
        m = keras.Sequential(name=name)
        m.add(self.conv2d_layer(f=1024, k=1, s=1, do = 0.2, bn=False, bias = True, trainable=self.trainable, name="1"))
        m.add(self.conv2d_layer(f=1024, k=1, s=1, do = 0.2, bn=False, bias = True, trainable=self.trainable, name="2"))
        m.add(self.conv2d_layer(f=1, k=1, s=1, do = 0.2, bn=False, bias = True, trainable=self.trainable, name="3", act=None))
        # m.add(keras.layers.Activation("sigmoid"))
        return m

class AE_msceleb(models.layers.Layers):
    def __init__(self, trainable):
        super(AE_msceleb, self).__init__()
        self.trainable = trainable

    def _encoder(self, name):
        m = keras.Sequential(name=name)
        m.add(self.conv2d_layer(f=64, k=2, s=1, do = 0., bn=True, bias = False, trainable=self.trainable, name="1"))
        m.add(self.conv2d_layer(f=128, k=7, s=2, do = 0., bn=True, bias = False, trainable=self.trainable, name="2"))
        m.add(self.conv2d_layer(f=256, k=5, s=2, do = 0., bn=True, bias = False, trainable=self.trainable, name="3"))
        m.add(self.conv2d_layer(f=256, k=7, s=2, do = 0., bn=True, bias = False, trainable=self.trainable, name="4"))
        m.add(self.conv2d_layer(f=512, k=4, s=1, do = 0., bn=True, bias = False, trainable=self.trainable, name="5"))
        # m.add(self.conv2d_layer(f=st.z_dim, k=1, s=1, do = 0., bn=False, bias = True, trainable=self.trainable, act=None, name="6"))
        # m.add(keras.layers.Activation("tanh"))
        m.add(self.conv2d_layer(f=st.z_dim*2, k=1, s=1, do = 0., bn=False, bias = True, trainable=self.trainable, act=None, name="6"))
        # m.add(models.layers.reparameterize())
        return m

    def _decoder(self, name):
        m = keras.Sequential(name=name)
        m.add(self.conv2d_layer(f=st.z_dim, k=1, s=1, do = 0., bn=True, bias = False, trainable=self.trainable, name="0"))
        m.add(self.deconv2d_layer(f=512, k=4, s=1, bn=True, bias = False, trainable=self.trainable, name="1"))
        m.add(self.deconv2d_layer(f=256, k=7, s=2, bn=True, bias = False, trainable=self.trainable, name="2"))
        m.add(self.deconv2d_layer(f=256, k=5, s=2, bn=True, bias = False, trainable=self.trainable, name="3"))
        m.add(self.deconv2d_layer(f=128, k=7, s=2, bn=True, bias = False, trainable=self.trainable, name="4"))
        m.add(self.deconv2d_layer(f=64, k=2, s=1, bn=True, bias = False, trainable=self.trainable, name="5"))
        m.add(self.conv2d_layer(f=st.C, k=1, s=1, bn=False, bias = False, trainable=self.trainable, name="6", act=None))
        m.add(models.layers.pixel_bias(trainable=self.trainable))
        m.add(keras.layers.Activation("sigmoid"))
        return m

    def _x_disc(self, name):
        m = keras.Sequential(name=name)
        m.add(self.conv2d_layer(f=64, k=2, s=1, do = 0., bn=True, bias = False, trainable=self.trainable, name="1"))
        m.add(self.conv2d_layer(f=128, k=7, s=2, do = 0., bn=True, bias = False, trainable=self.trainable, name="2"))
        m.add(self.conv2d_layer(f=256, k=5, s=2, do = 0., bn=True, bias = False, trainable=self.trainable, name="3"))
        m.add(self.conv2d_layer(f=256, k=7, s=2, do = 0., bn=True, bias = False, trainable=self.trainable, name="4"))
        m.add(self.conv2d_layer(f=512, k=4, s=1, do = 0., bn=True, bias = False, trainable=self.trainable, name="5"))
        return m

    def _z_disc(self, name):
        m = keras.Sequential(name=name)
        m.add(self.conv2d_layer(f=1024, k=1, s=1, do = 0.2, bn=False, bias = True, trainable=self.trainable, name="1"))
        m.add(self.conv2d_layer(f=1024, k=1, s=1, do = 0.2, bn=False, bias = True, trainable=self.trainable, name="2"))
        return m

    def _discriminator(self, name):
        m = keras.Sequential(name=name)
        m.add(self.conv2d_layer(f=2048, k=1, s=1, do = 0.2, bn=False, bias = True, trainable=self.trainable, name="1"))
        m.add(self.conv2d_layer(f=2048, k=1, s=1, do = 0.2, bn=False, bias = True, trainable=self.trainable, name="2"))
        m.add(self.conv2d_layer(f=1, k=1, s=1, do = 0.2, bn=False, bias = True, trainable=self.trainable, name="3", act=None))
        # m.add(keras.layers.Activation("sigmoid"))
        return m

class AE_imagenet(models.layers.Layers):
    def __init__(self, trainable):
        super(AE_imagenet, self).__init__()
        self.trainable = trainable

    def _encoder(self, name):
        m = keras.Sequential(name=name)
        m.add(self.conv2d_layer(f=64, k=4, s=2, do = 0., bn=True, bias = False, trainable=self.trainable, name="1"))
        m.add(self.conv2d_layer(f=64, k=4, s=1, do = 0., bn=True, bias = False, trainable=self.trainable, name="2"))
        m.add(self.conv2d_layer(f=128, k=4, s=2, do = 0., bn=True, bias = False, trainable=self.trainable, name="3"))
        m.add(self.conv2d_layer(f=128, k=4, s=1, do = 0., bn=True, bias = False, trainable=self.trainable, name="4"))
        m.add(self.conv2d_layer(f=256, k=4, s=2, do = 0., bn=True, bias = False, trainable=self.trainable, name="5"))
        m.add(self.conv2d_layer(f=256, k=4, s=1, do = 0., bn=True, bias = False, trainable=self.trainable, name="6"))
        m.add(self.conv2d_layer(f=2048, k=1, s=1, do = 0., bn=True, bias = False, trainable=self.trainable, name="7"))
        m.add(self.conv2d_layer(f=2048, k=1, s=1, do = 0., bn=True, bias = False, trainable=self.trainable, name="8"))
        # m.add(self.conv2d_layer(f=st.z_dim, k=1, s=1, do = 0., bn=False, bias = True, trainable=self.trainable, act=None, name="9"))
        # m.add(keras.layers.Activation("tanh"))
        m.add(self.conv2d_layer(f=st.z_dim*2, k=1, s=1, do = 0., bn=False, bias = True, trainable=self.trainable, act=None, name="9"))
        m.add(models.layers.reparameterize_func())
        return m

    def _decoder(self, name):
        m = keras.Sequential(name=name)
        m.add(self.conv2d_layer(f=st.z_dim*2, k=1, s=1, bn=True, bias = False, trainable=self.trainable, name="1"))
        m.add(self.conv2d_layer(f=256, k=1, s=1, bn=True, bias = False, trainable=self.trainable, name="2"))
        m.add(self.deconv2d_layer(f=256, k=4, s=1, bn=True, bias = False, trainable=self.trainable, name="3"))
        m.add(self.deconv2d_layer(f=128, k=4, s=2, bn=True, bias = False, trainable=self.trainable, name="4"))
        m.add(self.deconv2d_layer(f=128, k=4, s=1, bn=True, bias = False, trainable=self.trainable, name="5"))
        m.add(self.deconv2d_layer(f=64, k=4, s=2, bn=True, bias = False, trainable=self.trainable, name="6"))
        m.add(self.deconv2d_layer(f=64, k=4, s=1, bn=True, bias = False, trainable=self.trainable, name="7"))
        m.add(self.deconv2d_layer(f=64, k=4, s=2, bn=True, bias = False, trainable=self.trainable, name="8"))
        m.add(self.conv2d_layer(f=st.C, k=1, s=1, bn=False, bias = False, trainable=self.trainable, name="9", act=None))
        m.add(models.layers.pixel_bias(trainable=self.trainable))
        m.add(keras.layers.Activation("sigmoid"))
        return m

    def _x_disc(self, name):
        m = keras.Sequential(name=name)
        m.add(self.conv2d_layer(f=64, k=4, s=2, do = 0.2, bn=False, bias = True, trainable=self.trainable, name="1"))
        m.add(self.conv2d_layer(f=64, k=4, s=1, do = 0.2, bn=True, bias = False, trainable=self.trainable, name="2"))
        m.add(self.conv2d_layer(f=128, k=4, s=2, do = 0.2, bn=True, bias = False, trainable=self.trainable, name="3"))
        m.add(self.conv2d_layer(f=128, k=4, s=1, do = 0.2, bn=True, bias = False, trainable=self.trainable, name="4"))
        m.add(self.conv2d_layer(f=256, k=4, s=2, do = 0.2, bn=True, bias = False, trainable=self.trainable, name="5"))
        m.add(self.conv2d_layer(f=256, k=4, s=1, do = 0.2, bn=True, bias = False, trainable=self.trainable, name="6"))
        return m

    def _z_disc(self, name):
        m = keras.Sequential(name=name)
        m.add(self.conv2d_layer(f=2048, k=1, s=1, do = 0.2, bn=False, bias = True, trainable=self.trainable, name="1"))
        m.add(self.conv2d_layer(f=2048, k=1, s=1, do = 0.2, bn=False, bias = True, trainable=self.trainable, name="2"))
        return m

    def _discriminator(self, name):
        m = keras.Sequential(name=name)
        m.add(self.conv2d_layer(f=4096, k=1, s=1, do = 0.2, bn=False, bias = True, trainable=self.trainable, name="1"))
        m.add(self.conv2d_layer(f=4096, k=1, s=1, do = 0.2, bn=False, bias = True, trainable=self.trainable, name="2"))
        m.add(self.conv2d_layer(f=1, k=1, s=1, do = 0.2, bn=False, bias = True, trainable=self.trainable, name="3"))
        # m.add(keras.layers.Activation("sigmoid"))
        return m

if st.dataset==0: ae_network=AE_omniglot
elif st.dataset==1: ae_network=AE_msceleb
elif st.dataset==2: ae_network=AE_msceleb
else: ae_network=AE_msceleb