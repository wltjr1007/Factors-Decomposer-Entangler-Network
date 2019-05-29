
import settings as st
import tensorflow as tf
import numpy as np
keras = tf.keras

from glob import glob
import os

from models.DE_module import de_network
import models.layers

class DE_model(de_network):
    def __init__(self, trainable):
        super(DE_model, self).__init__(trainable=trainable)

        self.trainable = trainable

        self.input_x_s = keras.Input(name="x_support", shape=(None, None, st.W, st.H, st.C), batch_size=None)
        self.input_x_q = keras.Input(name="x_query", shape=(st.W, st.H, st.C), batch_size=None)

        self.ae_model_name = st.mode_ae_dict[st.ae_mode]
        if st.ae_mode==0: self.ae_model_name+="/disc/"
        else: self.ae_model_name+="/gen/"

        self.cls_name = st.mode_de_dict[st.de_mode]

    def build(self):
        ae_model_path = st.result_path+"%d00/%s/best_model/"%(st.ae_mode, st.dataset_dict[st.dataset])
        print(ae_model_path)

        with open(ae_model_path+"ae_encoder.json", "r") as f:
            try:
                self.encoder = keras.models.model_from_json(f.read())
            except:
                self.encoder = keras.models.model_from_json(f.read(), custom_objects={"reparameterize": models.layers.reparameterize})
        with open(ae_model_path+"ae_decoder.json", "r") as f:
            self.decoder = keras.models.model_from_json(f.read(),custom_objects={"pixel_bias":models.layers.pixel_bias})

        self.disen_all = self._disentangler(name="de/disen/all")
        self.disen_id = self._disentangler_id(name="de/disen/id")
        self.disen_sty = self._disentangler_sty(name="de/disen/sty")
        self.entan_all = self._entangler(name="de/entan/all")
        self.entan_id = self._entangler_id(name="de/entan/id")
        self.entan_sty = self._entangler_sty(name="de/entan/sty")
        self.mine_id = self._statistician_z(name="de/mine/id")
        self.mine_sty = self._statistician_z(name="de/mine/sty")
        self.mine_all = self._statistician_all(name="de/mine/all")

        if st.de_mode==0: classifier = self._siamese(name="de/"+self.cls_name)
        else: classifier = self._prototypical(name="de/"+self.cls_name)

        ### Support ###
        sup_shape = tf.shape(self.input_x_s)
        x_sup_flat = tf.reshape(self.input_x_s, (-1, st.W, st.H, st.C), name="sgfdiobjde")
        z_sup_all = self.encoder(x_sup_flat)
        z_sup_mu, z_sup_sig, z_sup  = models.layers.reparameterize_func(z_sup_all)

        z_sup_dis = self.disen_all(z_sup)
        z_sup_dis_id, z_sup_dis_sty = tf.split(z_sup_dis, 2, axis=-1)
        z_sup_dis_id, z_sup_dis_sty = self.disen_id(z_sup_dis_id), self.disen_sty(z_sup_dis_sty)

        z_sup_ent_id, z_sup_ent_sty = self.entan_id(z_sup_dis_id), self.entan_sty(z_sup_dis_sty)
        z_sup_ent_all = self.entan_all(keras.layers.concatenate([z_sup_ent_id, z_sup_ent_sty], axis=-1))
        z_sup_ent_mu, z_sup_ent_sig, z_sup_ent = models.layers.reparameterize_func(z_sup_ent_all)

        ### Query ###
        z_qry_all = self.encoder(self.input_x_q)
        z_qry_mu, z_qry_sig, z_qry  = models.layers.reparameterize_func(z_qry_all)

        z_qry_dis = self.disen_all(z_qry)
        z_qry_dis_id, z_qry_dis_sty = tf.split(z_qry_dis, 2, axis=-1)
        z_qry_dis_id, z_qry_dis_sty = self.disen_id(z_qry_dis_id), self.disen_sty(z_qry_dis_sty)

        z_qry_ent_id, z_qry_ent_sty = self.entan_id(z_qry_dis_id), self.entan_sty(z_qry_dis_sty)
        z_qry_ent_all = self.entan_all(keras.layers.concatenate([z_qry_ent_id, z_qry_ent_sty], axis=-1))
        z_qry_ent_mu, z_qry_ent_sig, z_qry_ent = models.layers.reparameterize_func(z_qry_ent_all)


        ### classifier ###
        logit = classifier(self.z_diff(z_sup_dis_id, z_qry_dis_id, shape=sup_shape, mode=0))
        logit = keras.backend.mean(logit, axis=(-1, -2))
        logit = keras.activations.softmax(logit)

        ### mine ###
        z_all_dis_id = keras.layers.concatenate([z_sup_dis_id, z_qry_dis_id], axis=0)
        z_all_dis_sty = keras.layers.concatenate([z_sup_dis_sty, z_qry_dis_sty], axis=0)

        d_id, d_sty = self.mine_id(z_all_dis_id), self.mine_sty(z_all_dis_sty)
        d_pos = self.mine_all(keras.layers.concatenate([d_id, d_sty], axis=-1))

        d_sty_shuf = models.layers.shuffle_batch()(d_sty)
        d_neg = self.mine_all(keras.layers.concatenate([d_id, d_sty_shuf], axis=-1))

        model_output = {"logit": logit, "d_pos": d_pos, "d_neg": d_neg,
                        "z_sup": z_sup, "z_sup_ent": z_sup_ent,
                        "z_qry": z_qry, "z_qry_ent": z_qry_ent}
        self.de_model = keras.Model({"x_sup": self.input_x_s, "x_qry": self.input_x_q}, model_output)

        ## interpolate
        x_sup_one = self.decoder(tf.reshape(z_sup[0], (1, 1, 1, st.z_dim)))
        x_qry_one = self.decoder(tf.reshape(z_qry[0], (1, 1, 1, st.z_dim)))

        x_inter = self._interpolate(x_id=z_qry_dis_id, x_sty=z_qry_dis_sty, y_id=z_sup_dis_id, y_sty=z_sup_dis_sty,
                                    x_s_ali=x_sup_one, x_q_ali=x_qry_one)

        vis_output = {"logit": logit, "d_pos": d_pos, "d_neg": d_neg,
                      "z_sup": z_sup, "z_sup_ent": z_sup_ent,
                      "z_qry": z_qry, "z_qry_ent": z_qry_ent,
                      "x_inter": x_inter}
        self.de_vis_model = keras.Model({"x_sup": self.input_x_s, "x_qry": self.input_x_q}, vis_output)

        ae_dec_path = sorted(glob(ae_model_path+"ae_decoder*.h5"), key=os.path.basename)[-1]
        ae_enc_path = sorted(glob(ae_model_path+"ae_encoder*.h5"), key=os.path.basename)[-1]

        self.decoder.load_weights(ae_dec_path, by_name=True)
        self.encoder.load_weights(ae_enc_path, by_name=True)

    def save_architecture(self, summ_path):
        def save_config(model, n):
            with open(summ_path+"de_"+n+".json", "w") as f:
                f.write(model.to_json())

        save_config(self.disen_all, "disen_all")
        save_config(self.disen_id, "disen_id")
        save_config(self.disen_sty, "disen_sty")

        save_config(self.entan_all, "entan_all")
        save_config(self.entan_id, "entan_id")
        save_config(self.entan_sty, "entan_sty")

        save_config(self.mine_id, "mine_id")
        save_config(self.mine_sty, "mine_sty")
        save_config(self.mine_all, "mine_all")

    def save_all(self, pn, fn):
        if not os.path.exists(pn):
            os.makedirs(pn)
        name = pn+fn

        print(name)

        self.disen_all.save_weights(name%"disen_all")
        self.disen_id.save_weights(name%"disen_id")
        self.disen_sty.save_weights(name%"disen_sty")

        self.entan_all.save_weights(name%"entan_all")
        self.entan_id.save_weights(name%"entan_id")
        self.entan_sty.save_weights(name%"entan_sty")

        self.mine_id.save_weights(name%"mine_id")
        self.mine_sty.save_weights(name%"mine_sty")
        self.mine_all.save_weights(name%"mine_all")

    # @tf.function
    def _interpolate(self, x_id, x_sty, y_id, y_sty, x_s_ali, x_q_ali, inter_cnt=3):
        all_id = []
        all_sty = []

        all_img = []

        x_id = keras.layers.Reshape((1, 1, st.z_dim))(x_id[0][None])
        x_sty = keras.layers.Reshape((1, 1, st.z_dim))(x_sty[0][None])
        y_id = keras.layers.Reshape((1, 1, st.z_dim))(y_id[0][None])
        y_sty = keras.layers.Reshape((1, 1, st.z_dim))(y_sty[0][None])

        for i in np.linspace(0., 1., inter_cnt):
            all_id += [self.entan_id((1. - i) * x_id + i * y_id)]
        for j in np.linspace(0., 1., inter_cnt):
            all_sty += [self.entan_sty((1. - j) * x_sty + j * y_sty)]
        for i in range(inter_cnt):
            temp_img = []
            for j in range(inter_cnt):
                te = self.entan_all(keras.layers.concatenate([all_id[i], all_sty[j]], axis=-1))
                _, _, te = models.layers.reparameterize_func(te)

                img = self.decoder(te)
                temp_img += [img]
            all_img += [temp_img]

        all_img = tf.convert_to_tensor(all_img)
        all_img = tf.reshape(all_img, (inter_cnt, inter_cnt*st.W, st.H, st.C), name="soweirgj")
        all_img = tf.transpose(all_img, (1, 0, 2, 3))
        all_img = tf.reshape(all_img, (1, inter_cnt*st.W, inter_cnt*st.H, st.C), name="saoeirgjrtyh")

        real_x_s = self.input_x_s[0,0,0][None]
        real_x_q = self.input_x_q[0][None]

        real_x_q = tf.pad(real_x_q, ((0,0), (0, (inter_cnt-1)*st.W), (0,0), (0,0)), constant_values=1.)
        x_q_ali = tf.pad(x_q_ali, ((0,0), (0, (inter_cnt-1)*st.W), (0,0), (0,0)), constant_values=1.)

        real_x_s = tf.pad(real_x_s, ((0,0), ((inter_cnt-1)*st.W,0), (0,0), (0,0)), constant_values=1.)
        x_s_ali = tf.pad(x_s_ali, ((0,0), ((inter_cnt-1)*st.W,0), (0,0), (0,0)), constant_values=1.)

        all_img = tf.concat((real_x_q, x_q_ali, all_img, x_s_ali, real_x_s), axis=2)

        # a1 = np.pad(a1, ((0,0), (0,2*64), (0,0), (0,0)), mode="constant", constant_values=0)
        # asdf = np.concatenate((a1, asdf), axis=2)
        # a2 = np.pad(a2, ((0,0), (2*64, 0), (0,0), (0,0)), mode="constant", constant_values=0)
        # asdf = np.concatenate((asdf, a2), axis=2)[0]
        return all_img


    def train_one_batch(self, x_sup, x_qry, y_lbl, step):
        with tf.GradientTape() as recon_grad, tf.GradientTape() as mine_grad, tf.GradientTape() as cls_grad:
            outputs = self.de_model({"x_sup": x_sup, "x_qry": x_qry})

            recon_loss = keras.losses.MeanSquaredError()(outputs["z_sup"], outputs["z_sup_ent"])
            recon_loss += keras.losses.MeanSquaredError()(outputs["z_qry"], outputs["z_qry_ent"])
            cls_loss = keras.losses.CategoricalCrossentropy()(y_lbl, outputs["logit"])
            mine_loss = -(keras.backend.mean(outputs["d_pos"]) - keras.backend.log(
                keras.backend.mean(keras.backend.exp(outputs["d_neg"])) + 0.000001))

        cls_acc = keras.backend.mean(keras.metrics.categorical_accuracy(y_lbl,outputs["logit"]))


        losses = {"recon": recon_loss.numpy(), "cls": cls_loss.numpy(), "mine": mine_loss.numpy(), "acc":cls_acc.numpy()}

        recon_grads = recon_grad.gradient(recon_loss, self.de_all)
        cls_grads = cls_grad.gradient(cls_loss, self.de_all)
        mine_grads = mine_grad.gradient(mine_loss, self.de_all)

        name_grad = {}
        name_var = {}
        for g, v in zip(recon_grads, self.de_all):
            if g is not None:
                if v.name in name_grad:
                    name_grad[v.name] += g
                else:
                    name_grad[v.name] = g
                    name_var[v.name] = v
        for g, v in zip(cls_grads, self.de_all):
            if g is not None:
                if v.name in name_grad:
                    name_grad[v.name] += g
                else:
                    name_grad[v.name] = g
                    name_var[v.name] = v

        for g, v in zip(mine_grads, self.de_all):
            temp_grad = g
            if g is not None:
                if v.name in name_grad:
                    temp_grad = tf.stack((g, name_grad[v.name]), axis=0)
                    norm_grad = tf.norm(temp_grad, axis=(-1, -2))
                    clip_norm = tf.reduce_min(norm_grad)
                    temp_grad = tf.clip_by_norm(g, clip_norm=clip_norm)
                if "mine" not in v.name:
                    temp_grad = tf.negative(temp_grad)
                if v.name in name_grad:
                    name_grad[v.name] += temp_grad
                else:
                    name_grad[v.name] = temp_grad
                    name_var[v.name] = v

        grad_vars = [(name_grad[name], name_var[name]) for name in name_grad.keys()]
        self.all_optim = self.optim.apply_gradients(grad_vars)

        return losses, outputs

    def logger(self, x_sup, x_qry, y_lbl, step):
        outputs = self.de_vis_model({"x_sup": x_sup, "x_qry": x_qry}, training=False)

        recon_loss = keras.losses.MeanSquaredError()(outputs["z_sup"], outputs["z_sup_ent"])
        recon_loss += keras.losses.MeanSquaredError()(outputs["z_qry"], outputs["z_qry_ent"])
        cls_loss = keras.losses.categorical_crossentropy(y_lbl, outputs["logit"])
        cls_loss = keras.backend.mean(cls_loss)
        mine_loss = -(tf.reduce_mean(outputs["d_pos"]) - tf.math.log(tf.reduce_mean(tf.math.exp(outputs["d_neg"])) + 0.000001))

        cls_acc = keras.backend.mean(keras.metrics.categorical_accuracy(y_lbl,outputs["logit"]))

        summ_img = outputs["x_inter"]

        tf.summary.scalar("recon", recon_loss, step=step)
        tf.summary.scalar("cls", cls_loss, step=step)
        tf.summary.scalar("mine", mine_loss, step=step)
        tf.summary.scalar("cls_acc", cls_acc, step=step)
        tf.summary.image(name="summ_img", data=summ_img, step=step, max_outputs=1)




de_model = DE_model
