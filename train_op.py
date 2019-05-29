

import settings as st
from datetime import datetime

import tensorflow as tf
from zipfile import ZipFile
from glob import glob

from models import base_model
import os

if st.part_mode==0: cur_model = base_model.AE
else: cur_model = base_model.DE

import numpy as np

class Trainer(cur_model):
    def __init__(self, trn_dat, trn_lbl, one_dat, tst_dat, tst_lbl, summ_path):
        super(Trainer, self).__init__(trainable=True)


        self.trn_dat = trn_dat
        self.one_dat = one_dat
        self.tst_dat = tst_dat
        self.trn_lbl = trn_lbl
        self.tst_lbl = tst_lbl

        self.summ_path = summ_path

        self.build()
        self.save_architecture(self.summ_path)
        self._optimizer()



    def train(self):
        self.train_summary_writer = tf.summary.create_file_writer(self.summ_path+"train")

        with self.train_summary_writer.as_default():
            self._train(self.summ_path)
        self.train_summary_writer.close()

    def extract_idx(self, c_way=st.c_way, k_shot=st.k_shot):
        trn_unique_lbl = np.unique(self.trn_lbl)

        if not os.path.exists(st.preprocessed_data_path + "%s/trn_mask_cnt.npy" % (st.dataset_dict[st.dataset])) or True:
            trn_mask = np.array([self.trn_lbl == i for i in range(max(trn_unique_lbl)+1)], dtype=np.bool)

            new_trn_mask = []
            trn_mask_cnt = []
            for m in trn_mask:
                temp_mask = np.argwhere(m).squeeze()
                trn_mask_cnt += [len(temp_mask)]
                new_trn_mask += [temp_mask]
            new_trn_mask = np.array(new_trn_mask)
            trn_mask_cnt = np.array(trn_mask_cnt)

            np.save(st.preprocessed_data_path + "%s/new_trn_mask.npy" % (st.dataset_dict[st.dataset]), new_trn_mask)
            np.save(st.preprocessed_data_path + "%s/trn_mask_cnt.npy" % (st.dataset_dict[st.dataset]), trn_mask_cnt)


        else:
            new_trn_mask = np.load(st.preprocessed_data_path + "%s/new_trn_mask.npy" % (st.dataset_dict[st.dataset]))
            trn_mask_cnt = np.load(st.preprocessed_data_path + "%s/trn_mask_cnt.npy" % (st.dataset_dict[st.dataset]), mmap_mode="r")

        trn_mask_cnt = np.array([np.arange(c) for c in trn_mask_cnt])

        y_lbl = np.zeros(shape=(st.batch_cnt, c_way), dtype=np.bool)
        way_lbl = np.zeros(shape=(st.batch_cnt, c_way), dtype=np.int32)
        tst_way_lbl = np.zeros(shape=st.batch_cnt, dtype=np.int32)

        trn_idx = np.zeros(shape=(st.batch_cnt, c_way, k_shot), dtype=np.uint32)
        tst_idx = np.zeros(shape=st.batch_cnt, dtype=np.uint32)


        for batch_cnt in range(st.batch_cnt):
            way_lbl[batch_cnt] = np.random.choice(trn_unique_lbl, size=c_way, replace=False)
            cur_tst_idx = np.random.permutation(c_way)[0]
            tst_way_lbl[batch_cnt] = way_lbl[batch_cnt, cur_tst_idx]
            y_lbl[batch_cnt, cur_tst_idx] = True


            for w_cnt, (m, c, w) in enumerate(zip(new_trn_mask[way_lbl[batch_cnt]], trn_mask_cnt[way_lbl[batch_cnt]], way_lbl[batch_cnt])):
                rand_cnt = k_shot
                if cur_tst_idx == w_cnt: rand_cnt +=1

                rand_idx = m[np.random.choice(c, k_shot + 1, replace=False)]

                trn_idx[batch_cnt, w_cnt] = rand_idx[:k_shot]

                if cur_tst_idx==w_cnt: tst_idx[batch_cnt] = rand_idx[-1]
        return trn_idx, tst_idx, y_lbl.astype(np.float32)