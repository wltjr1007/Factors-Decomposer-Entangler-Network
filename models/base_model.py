
import settings as st
import tensorflow as tf
import numpy as np
keras = tf.keras
from time import time

from models.AE_model import ae_model
from models.DE_model import de_model


class AE(ae_model):
    def __init__(self, trainable):
        super(AE, self).__init__(trainable)

    def _optimizer(self):
        self.gen_var = []
        self.disc_var = []
        self.mine_var = []
        for var in self.ae_model.trainable_variables:
            if "gen" in var.name:
                self.gen_var += [var]
            if "disc" in var.name:
                self.disc_var += [var]
            if "mine" in var.name:
                self.mine_var += [var]
        for var in self.ae_model.variables:
            if "mean" in var.name:
                self.bn_mean = var
                break

        self.optim = keras.optimizers.Adam(learning_rate=st.lr, beta_1=st.beta1, beta_2=st.beta2)

    def _train(self, summ_path):
        global_step = 0
        global_time = time()
        local_time = time()
        for cur_epoch in range(st.epoch):
            rand_idx = np.random.permutation(len(self.trn_dat))
            for step in range(0, len(self.trn_dat)-1, st.batch_size):
                cur_dat = self.trn_dat[rand_idx[step:step + st.batch_size]]
                rand_y = np.random.normal(loc=0., scale=1., size=(len(cur_dat), 1, 1, st.z_dim)).astype(np.float32)
                # rand_y = np.random.uniform(low=-1., high=1., size=(len(cur_dat), 1, 1, st.z_dim)).astype(np.float32)
                losses, outputs = self.train_one_batch(x=cur_dat, y=rand_y, step=global_step, epoch=cur_epoch)

                if time()-local_time >15 or global_step==0:
                    self.logger(x=cur_dat[:st.vis_cnt], y=rand_y[:st.vis_cnt], step=global_step)
                    print("Epoch %d/%d(%d), Step %d/%d(%d)"%(cur_epoch, st.epoch, time()-global_time,
                                                             step, len(self.trn_dat), time()-local_time),
                          losses)

                    local_time=time()
                global_step+=1
            # self.ae_model.save_weights(summ_path+"ae_weights_%d_%d_%d.h5"%(st.mode, st.dataset, cur_epoch))
            self.save_all(summ_path+"weights/%d/"%cur_epoch,"ae_%s_"+"%d_%d_%d.h5"%(st.mode, st.dataset, cur_epoch))
            print("\n\033[1;31m", end="")
            print("Model saved in", summ_path)
            print("\033[0;m", end="")

class DE(de_model):
    def __init__(self, trainable):
        super(DE, self).__init__(trainable)

    def _optimizer(self):
        self.ae_enc = []
        self.ae_dec = []
        self.de_disen = []
        self.de_entan = []
        self.de_mine = []
        self.de_cls = []
        self.de_all = []

        for var in self.de_model.trainable_variables:
            if "enc" in var.name:
                self.ae_enc += [var]
            elif "dec" in var.name:
                self.ae_dec += [var]
            elif "disen" in var.name:
                self.de_disen += [var]
            elif "entan" in var.name:
                self.de_entan += [var]
            elif self.cls_name in var.name:
                self.de_cls += [var]
            elif "de/mine" in var.name:
                self.de_mine += [var]
            if "de" in var.name:
                self.de_all += [var]

        self.optim = keras.optimizers.Adam(learning_rate=st.lr, beta_1=st.beta1, beta_2=st.beta2)

    def _train(self, summ_path):
        print(summ_path)
        global_step = 0
        global_time = time()
        local_time = time()
        for cur_epoch in range(st.epoch):
            print("start_training!!!", 1)
            trn_idx, tst_idx, y_lbl_all = self.extract_idx(c_way=st.train_c_way, k_shot=st.train_k_shot)
            rand_idx = np.random.permutation(len(trn_idx))
            print("start_training!!!", 2)
            for step in range(0, st.batch_cnt-1, st.batch_size):
                cur_idx = rand_idx[step:step+st.batch_size]
                x_sup = self.trn_dat[trn_idx[cur_idx]]
                x_qry = self.trn_dat[tst_idx[cur_idx]]
                y_lbl = y_lbl_all[cur_idx]

                losses, outputs = self.train_one_batch(x_sup=x_sup, x_qry=x_qry, y_lbl=y_lbl, step=global_step)

                if time()-local_time >15 or global_step==0:
                    val_idx = np.random.permutation(st.tst_cnt)[:st.val_cnt]
                    self.logger(x_sup=self.one_dat[val_idx], x_qry=self.tst_dat[val_idx], y_lbl=self.tst_lbl[val_idx], step=global_step)
                    print("Epoch %d/%d(%d), Step %d/%d(%d)"%(cur_epoch, st.epoch, time()-global_time,
                                                             step, len(self.trn_dat), time()-local_time),
                          losses)


                    local_time=time()
                global_step+=1
            self.save_all(summ_path+"weights/%d/"%cur_epoch,"de_%s_"+"%d_%d_%d.h5"%(st.mode, st.dataset, cur_epoch))
            print("\n\033[1;31m", end="")
            print("Model saved in", summ_path)
            print("\033[0;m", end="")
