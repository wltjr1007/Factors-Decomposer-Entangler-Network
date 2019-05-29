

import settings as st
import tensorflow as tf
keras = tf.keras
import models.layers

from models.AE_module import ae_network
import numpy as np
import os


class BEGAN(ae_network):
    def __init__(self, trainable):
        super(BEGAN, self).__init__(trainable=trainable)

        self.trainable = trainable

        self.input_x = keras.Input(name="x", shape=(st.W, st.H, st.C), batch_size=None)
        self.input_y = keras.Input(name="y", shape=(1, 1, st.z_dim), batch_size=None)

        self.model_name = st.mode_ae_dict[st.ae_mode]

    def build(self):
        self.gen_decoder = self._decoder(name=self.model_name+"/gen/dec")
        self.disc_encoder = self._encoder(name=self.model_name+"/disc/enc")
        self.disc_decoder = self._decoder(name=self.model_name+"/disc/dec")

        real_z_gen_all = self.disc_encoder(self.input_x)
        real_z_gen_mu, real_z_gen_sig, real_z_gen  = models.layers.reparameterize_func(real_z_gen_all)
        real_x_gen = self.disc_decoder(real_z_gen)

        fake_x_gen = self.gen_decoder(self.input_y)
        fake_z_gen_all = self.disc_encoder(fake_x_gen)
        fake_z_gen_mu, fake_z_gen_sig, fake_z_gen  = models.layers.reparameterize_func(fake_z_gen_all)
        fake_x_gen_gen = self.disc_decoder(fake_z_gen)
        # real_z_gen_all = self.disc_encoder(self.input_x)
        # real_z_gen_mu, real_z_gen_sig, real_z_gen  = models.layers.reparameterize_func(real_z_gen_all)
        # real_x_gen = self.gen_decoder(real_z_gen)
        #
        # fake_x_gen = self.gen_decoder(self.input_y)
        # fake_z_gen_all = self.disc_encoder(fake_x_gen)
        # fake_z_gen_mu, fake_z_gen_sig, fake_z_gen  = models.layers.reparameterize_func(fake_z_gen_all)
        # fake_x_gen_gen = self.gen_decoder(fake_z_gen)

        self.kt = tf.Variable(initial_value=0., trainable=False, name="kt", dtype=tf.float32)

        self.ae_model = keras.Model({"x": self.input_x, "y": self.input_y},
                                 {"real_x_gen": real_x_gen, "fake_x_gen": fake_x_gen, "fake_x_gen_gen": fake_x_gen_gen,
                                  "real_z_gen": real_z_gen, "fake_z_gen": fake_z_gen})


    def train_one_batch(self, x, y, step, epoch):
        with tf.GradientTape() as g_grad, tf.GradientTape() as d_grad:
            outputs = self.ae_model({"x": x, "y": y})
            d_real = keras.backend.mean(keras.backend.abs(x-outputs["real_x_gen"]))
            d_fake = keras.backend.mean(keras.backend.abs(outputs["fake_x_gen"]-outputs["fake_x_gen_gen"]))

            self.g_loss = d_fake
            self.d_loss = d_real-self.kt*d_fake

        losses = {"g_loss": self.g_loss.numpy(), "d_loss": self.d_loss.numpy()}

        g_grads = g_grad.gradient(self.g_loss, self.gen_var)
        self.optim.apply_gradients(zip(g_grads, self.gen_var))

        # if step%3==0:
        d_grads = d_grad.gradient(self.d_loss, self.disc_var)
        self.optim.apply_gradients(zip(d_grads, self.disc_var))

        self.kt = keras.backend.clip(self.kt + st.lambda_k * (st.gamma_k * d_real - d_fake), 0, 1)


        return losses, outputs


    def save_architecture(self, summ_path):
        def save_config(model, n):
            with open(summ_path+"ae_"+n+".json", "w") as f:
                f.write(model.to_json())

        save_config(self.gen_decoder, "gen_decoder")
        save_config(self.disc_encoder, "encoder")
        save_config(self.disc_decoder, "decoder")

    def save_all(self, pn, fn):
        if not os.path.exists(pn):
            os.makedirs(pn)
        name = pn+fn

        self.gen_decoder.save_weights(name%"gen_decoder")
        self.disc_encoder.save_weights(name%"encoder")
        self.disc_decoder.save_weights(name%"decoder")


    def logger(self, x, y, step, img=True):
        outputs = self.ae_model({"x": x, "y": y}, training=False)
        if img:
            summ_img = tf.concat((x, outputs["real_x_gen"], outputs["fake_x_gen"]), axis=-2)
            summ_img = tf.clip_by_value(summ_img, 0, 1.)
            self.summ_img = tf.reshape(summ_img, [1, -1, summ_img.shape[-2], summ_img.shape[-1]])
            tf.summary.image(name="summ_img", data=self.summ_img, step=step, max_outputs=1)
            tf.summary.histogram(name="real_z_gen", data=outputs["real_z_gen"], step=step)
            tf.summary.flush()

        d_real = keras.backend.mean(keras.backend.abs(x-outputs["real_x_gen"]))
        d_fake = keras.backend.mean(keras.backend.abs(outputs["fake_x_gen"]-outputs["fake_x_gen_gen"]))

        g_loss = d_fake
        d_loss = d_real-self.kt*d_fake

        tf.summary.scalar("real_d", d_real, step=step)
        tf.summary.scalar("fake_d", d_fake, step=step)
        tf.summary.scalar("kt", self.kt, step=step)

        tf.summary.scalar('d_loss', d_loss, step=step)
        tf.summary.scalar('g_loss', g_loss, step=step)

class ALI(ae_network):
    def __init__(self, trainable):
        super(ALI, self).__init__(trainable=trainable)

        self.trainable = trainable

        self.input_x = keras.Input(name="x", shape=(st.W, st.H, st.C), batch_size=None)
        self.input_y = keras.Input(name="y", shape=(1, 1, st.z_dim), batch_size=None)
        self.model_name = st.mode_ae_dict[st.ae_mode]


    def build(self):
        self.encoder = self._encoder(name=self.model_name+"/gen/enc")
        self.decoder = self._decoder(name=self.model_name+"/gen/dec")
        self.x_disc = self._x_disc(name=self.model_name+"/disc/x")
        self.z_disc = self._z_disc(name=self.model_name+"/disc/z")
        self.all_disc = self._discriminator(name=self.model_name+"/disc/all")


        real_z_gen_mu_sig = self.encoder(self.input_x)
        real_z_gen_mu, real_z_gen_sig, real_z_gen  = models.layers.reparameterize_func(real_z_gen_mu_sig)
        real_x_gen = self.decoder(real_z_gen)

        real_x_disc = self.x_disc(self.input_x)
        real_z_disc = self.z_disc(real_z_gen)
        real_disc = self.all_disc(keras.layers.concatenate([real_x_disc, real_z_disc]))

        fake_x_gen = self.decoder(self.input_y)
        fake_z_gen_mu_sig = self.encoder(fake_x_gen)
        fake_z_gen_mu, fake_z_gen_sig, fake_z_gen  = models.layers.reparameterize_func(fake_z_gen_mu_sig)

        fake_x_disc = self.x_disc(fake_x_gen)
        fake_z_disc = self.z_disc(self.input_y)
        fake_disc = self.all_disc(keras.layers.concatenate([fake_x_disc, fake_z_disc]))

        model_output = {"real_d": real_disc, "fake_d": fake_disc,
                        "real_x_gen": real_x_gen, "fake_x_gen": fake_x_gen,
                        "real_z_gen": real_z_gen, "real_z_gen_mu":real_z_gen_mu, "real_z_gen_sig":real_z_gen_sig,
                        "fake_z_gen": fake_z_gen, "fake_z_gen_mu":fake_z_gen_mu, "fake_z_gen_sig":fake_z_gen_sig}


        if st.ae_mode==2:
            self.x_T = self._x_disc(name=self.model_name+"/mine/x")
            self.z_T = self._z_disc(name=self.model_name+"/mine/z")
            self.all_T = self._discriminator(name=self.model_name+"/mine/all")
            shuffler = models.layers.shuffle_batch()

            fake_T_x = self.x_T(fake_x_gen)
            fake_T_z = self.z_T(self.input_y)
            d_pos = self.all_T(keras.layers.concatenate([fake_T_x, fake_T_z]))

            fake_T_x_shuf=shuffler(fake_T_x)
            d_neg = self.all_T(keras.layers.concatenate([fake_T_x_shuf, fake_T_z]))

            model_output.update({"d_pos": d_pos, "d_neg": d_neg})


        self.ae_model = keras.Model({"x": self.input_x, "y": self.input_y}, model_output)

    def save_architecture(self, summ_path):
        def save_config(model, n):
            with open(summ_path+"ae_"+n+".json", "w") as f:
                f.write(model.to_json())

        save_config(self.encoder, "encoder")
        save_config(self.decoder, "decoder")
        save_config(self.x_disc, "x_disc")
        save_config(self.z_disc, "z_disc")
        save_config(self.all_disc, "all_disc")
        if st.ae_mode==2:
            save_config(self.x_T, "x_T")
            save_config(self.z_T, "z_T")
            save_config(self.all_T, "all_T")

    def save_all(self, pn, fn):
        if not os.path.exists(pn):
            os.makedirs(pn)
        name = pn+fn

        self.encoder.save_weights(name%"encoder")
        self.decoder.save_weights(name%"decoder")
        self.x_disc.save_weights(name%"x_disc")
        self.z_disc.save_weights(name%"z_disc")
        self.all_disc.save_weights(name%"all_disc")
        if st.ae_mode==2:
            self.x_T.save_weights(name%"x_T")
            self.z_T.save_weights(name%"z_T")
            self.all_T.save_weights(name%"all_T")


    def train_one_batch(self, x, y, step, epoch):
        with tf.GradientTape() as g_grad, tf.GradientTape() as d_grad, tf.GradientTape() as m_grad:
            outputs = self.ae_model({"x": x, "y": y}, training=True)
            real_d = outputs["real_d"]
            fake_d = outputs["fake_d"]
            real_z_gen_mu, real_z_gen_sig = outputs["real_z_gen_mu"], outputs["real_z_gen_mu"]
            fake_z_gen_mu, fake_z_gen_sig = outputs["fake_z_gen_mu"], outputs["fake_z_gen_sig"]
            # self.d_loss = -(tf.math.log(keras.backend.mean(real_d)+1e-8)+tf.math.log(keras.backend.mean(1.-fake_d)+1e-8))
            # self.g_loss = -(tf.math.log(keras.backend.mean(1-real_d)+1e-8)+tf.math.log(keras.backend.mean(fake_d)+1e-8))
            # self.d_loss = keras.losses.MeanSquaredError()(0, fake_d)
            # self.d_loss += keras.losses.MeanSquaredError()(1, real_d)
            # self.g_loss = keras.losses.MeanSquaredError()(1, fake_d)
            # self.g_loss += keras.losses.MeanSquaredError()(0, real_d)
            # self.d_loss = -(keras.backend.mean(tf.math.log(real_d+0.000001))+keras.backend.mean(tf.math.log(1.-fake_d+0.000001)))
            # self.g_loss = -(keras.backend.mean(tf.math.log(1.-real_d+0.000001))+keras.backend.mean(tf.math.log(fake_d+0.000001)))
            self.g_loss = keras.backend.mean(keras.activations.softplus(real_d)+keras.activations.softplus(-fake_d))
            self.d_loss = keras.backend.mean(keras.activations.softplus(-real_d)+keras.activations.softplus(fake_d))

            self.kl_loss = keras.backend.mean(0.5 * tf.reduce_sum(tf.square(real_z_gen_mu) + tf.square(real_z_gen_sig) - keras.backend.log(tf.square(real_z_gen_sig)) - 1,1))
            self.kl_loss += keras.backend.mean(0.5 * tf.reduce_sum(tf.square(fake_z_gen_mu) + tf.square(fake_z_gen_sig) - keras.backend.log(tf.square(fake_z_gen_sig)) - 1,1))

            self.recon_loss = keras.backend.mean(keras.losses.MeanSquaredError()(x, outputs["real_x_gen"]))
            # self.recon_loss += keras.backend.mean(keras.losses.MeanAbsoluteError()(y, outputs["fake_z_gen"]))

            self.g_loss+=self.recon_loss
            # self.g_loss+=self.kl_loss

            if st.ae_mode ==2:
                outputs = self.ae_model({"x": x, "y": y})
                d_pos, d_neg = outputs["d_pos"], outputs["d_neg"]

                self.g_mine = -(tf.reduce_mean(d_pos) - tf.math.log(tf.reduce_mean(tf.math.exp(d_neg))+0.000001))


        g_grads = g_grad.gradient(self.g_loss, self.gen_var)
        d_grads = d_grad.gradient(self.d_loss, self.disc_var)

        losses = {"g_loss": self.g_loss.numpy(), "d_loss": self.d_loss.numpy(), "recon": self.recon_loss.numpy(), "kl": self.kl_loss.numpy()}

        g_grad_var = zip(g_grads, self.gen_var)
        d_grad_var = zip(d_grads, self.disc_var)

        if st.ae_mode == 2:
            g_name_grad = {}
            g_name_vars = {}
            for g, v in zip(g_grads, self.gen_var):
                if g is None:
                    continue
                g_name_grad[v.name] = g
                g_name_vars[v.name] = v
            self.gen_var += self.mine_var
            m_grads = m_grad.gradient(self.g_mine, self.gen_var)

            for g, v in zip(m_grads, self.gen_var):
                if g is None:
                    continue
                if v.name in g_name_grad:
                    temp_grad = tf.stack((g, g_name_grad[v.name]), axis=0)
                    temp_grad = tf.reshape(temp_grad, (2, -1))
                    norm_grad = tf.norm(temp_grad, axis=-1)
                    clip_norm = tf.reduce_min(norm_grad)
                    temp_grad = tf.clip_by_norm(g, clip_norm=clip_norm)

                    g_name_grad[v.name] += temp_grad
                else:
                    g_name_grad[v.name] = g
                    g_name_vars[v.name] = v

            losses.update({"mine": self.g_mine.numpy()})

            g_grad_var = [(g_name_grad[name], g_name_vars[name]) for name in g_name_vars.keys()]

        self.optim.apply_gradients(g_grad_var)
        if step%10==0 or st.dataset==0:
            self.optim.apply_gradients(d_grad_var)

        return losses, outputs


    def logger(self, x, y, step, img=True):
        outputs = self.ae_model({"x": x, "y": y}, training=False)
        if img:
            summ_img = tf.concat((x, outputs["real_x_gen"], outputs["fake_x_gen"]), axis=-2)
            summ_img = tf.clip_by_value(summ_img, 0, 1.)
            self.summ_img = tf.reshape(summ_img, [1, -1, summ_img.shape[-2], summ_img.shape[-1]])
            tf.summary.image(name="summ_img", data=self.summ_img, step=step, max_outputs=1)
            tf.summary.histogram(name="real_z_gen", data=outputs["real_z_gen"], step=step)
            tf.summary.flush()

        tf.summary.scalar("real_d", keras.backend.mean(outputs["real_d"]), step=step)
        tf.summary.scalar("fake_d", keras.backend.mean(outputs["fake_d"]), step=step)

        tf.summary.scalar('d_loss', self.d_loss, step=step)
        tf.summary.scalar('g_loss', self.g_loss, step=step)
        tf.summary.scalar('recon_loss', self.recon_loss, step=step)
        if st.ae_mode==2:
            tf.summary.scalar("mine", self.g_mine, step=step)

if st.ae_mode==0: ae_model = BEGAN
else: ae_model= ALI