from tensorflow.python.data.experimental.ops import unique

import settings as st
import tarfile
import zipfile
import os
from glob import glob
import numpy as np
import scipy.io
import scipy.misc
from PIL import Image, ImageOps

def process_omniglot():
    raw_path = st.raw_data_path+"omniglot/"
    #Extract if not already extracted
    if not os.path.exists(raw_path+"images_evaluation"):
        with zipfile.ZipFile(raw_path+"images_background.zip", 'r') as zip_ref:
            zip_ref.extractall(raw_path)
        with zipfile.ZipFile(raw_path+"images_evaluation.zip", "r") as zip_ref:
            zip_ref.extractall(raw_path)


    with open(raw_path+"trainval.txt", "r") as f:
        trn_path_fn = np.unique([fn.split("/rot")[0] for fn in f.readlines()])
    with open(raw_path+"test.txt", "r") as f:
        tst_path_fn = np.unique([fn.split("/rot")[0] for fn in f.readlines()])

    all_path = glob(raw_path+"**/*.png", recursive=True)
    all_path = sorted(all_path, key=os.path.basename)

    trn_dat = np.zeros(shape=(len(trn_path_fn)*20, st.W, st.H), dtype=np.uint8)
    trn_lbl = np.zeros(shape=(len(trn_path_fn)*20), dtype=np.int16)

    tst_dat = np.zeros(shape=(len(tst_path_fn)*20, st.W, st.H), dtype=np.uint8)
    tst_lbl = np.zeros(shape=(len(tst_path_fn)*20), dtype=np.int16)

    trn_cnt = 0
    tst_cnt = 0
    for cnt, fn in enumerate(all_path):
        temp_fn = fn.split("/")
        temp_fn = "%s/%s"%(temp_fn[-3], temp_fn[-2])
        if temp_fn in trn_path_fn:
            trn_dat[trn_cnt] = Image.open(fn).convert("RGBA").resize((st.W, st.H), resample=Image.LANCZOS).convert("L")
            trn_lbl[trn_cnt] = fn.split("/")[-1].split("_")[0]
            trn_cnt += 1
        elif temp_fn in tst_path_fn:
            tst_dat[tst_cnt] = Image.open(fn).convert("RGBA").resize((st.W, st.H), resample=Image.LANCZOS).convert("L")
            tst_lbl[tst_cnt] = fn.split("/")[-1].split("_")[0]
            tst_cnt +=1

    trn_lbl -= 1
    tst_lbl -= 1

    augment_trn_dat = np.zeros(shape=(len(trn_dat), 4, st.W, st.H), dtype=trn_dat.dtype)
    augment_trn_lbl = np.zeros(shape=(len(trn_dat), 4), dtype=trn_lbl.dtype)
    for cnt, (dat, lbl) in enumerate(zip(trn_dat, trn_lbl)):
        for k in range(4):
            augment_trn_dat[cnt, k] = np.rot90(dat, k=k)
            augment_trn_lbl[cnt, k] = lbl*4+k
    augment_trn_dat = augment_trn_dat.reshape((-1, st.W, st.H, st.C))
    augment_trn_lbl = augment_trn_lbl.reshape(-1)

    augment_tst_dat = np.zeros(shape=(len(tst_dat), 4, st.W, st.H), dtype=tst_dat.dtype)
    augment_tst_lbl = np.zeros(shape=(len(tst_dat), 4), dtype=tst_lbl.dtype)
    for cnt, (dat, lbl) in enumerate(zip(tst_dat, tst_lbl)):
        for k in range(4):
            augment_tst_dat[cnt, k] = np.rot90(dat, k=k)
            augment_tst_lbl[cnt, k] = lbl*4+k
    augment_tst_dat = augment_tst_dat.reshape((-1, st.W, st.H, st.C))
    augment_tst_lbl = augment_tst_lbl.reshape(-1)

    if not os.path.exists(st.preprocessed_data_path+"omniglot/"):
        os.makedirs(st.preprocessed_data_path+"omniglot")

    np.save(st.preprocessed_data_path+"omniglot/trn_dat.npy", trn_dat.astype(np.float32)/255.)
    np.save(st.preprocessed_data_path+"omniglot/trn_lbl.npy", trn_lbl)
    np.save(st.preprocessed_data_path+"omniglot/tst_dat.npy", tst_dat.astype(np.float32)/255.)
    np.save(st.preprocessed_data_path+"omniglot/tst_lbl.npy", tst_lbl)

    np.save(st.preprocessed_data_path+"omniglot/augment_trn_dat.npy", augment_trn_dat.astype(np.float32)/255.)
    np.save(st.preprocessed_data_path+"omniglot/augment_trn_lbl.npy", augment_trn_lbl)
    np.save(st.preprocessed_data_path+"omniglot/augment_tst_dat.npy", augment_tst_dat.astype(np.float32)/255.)
    np.save(st.preprocessed_data_path+"omniglot/augment_tst_lbl.npy", augment_tst_lbl)

def process_imagenet():
    raw_path = st.raw_data_path+"imagenet/"
    if not os.path.exists(raw_path+"images"):
        with zipfile.ZipFile(raw_path+"images.zip", 'r') as zip_ref:
            zip_ref.extractall(raw_path)

    with open(raw_path + "train.csv", "r") as f:
        f.readline()
        trn_csv = f.readlines()
    with open(raw_path + "val.csv", "r") as f:
        f.readline()
        val_csv = f.readlines()
    with open(raw_path + "test.csv", "r") as f:
        f.readline()
        tst_csv = f.readlines()


    trn_dat = np.zeros(shape=(len(trn_csv), st.W, st.H, st.C), dtype=np.uint8)
    trn_lbl = np.zeros(shape=len(trn_csv), dtype=np.uint32)
    val_dat = np.zeros(shape=(len(val_csv), st.W, st.H, st.C), dtype=np.uint8)
    val_lbl = np.zeros(shape=len(val_csv), dtype=np.uint32)
    tst_dat = np.zeros(shape=(len(tst_csv), st.W, st.H, st.C), dtype=np.uint8)
    tst_lbl = np.zeros(shape=len(tst_csv), dtype=np.uint32)

    lbl_dict = {}
    lbl_cnt = 0
    for cnt, line in enumerate(trn_csv):
        fn, lbl_str = line.split(",")

        if lbl_str not in lbl_dict:
            lbl_dict[lbl_str] = lbl_cnt
            lbl_cnt += 1

        trn_lbl[cnt] = lbl_dict[lbl_str]
        trn_dat[cnt] = ImageOps.fit(Image.open(raw_path + "images/%s" % fn), (st.W, st.H), method=Image.LANCZOS)

    for cnt, line in enumerate(val_csv):
        fn, lbl_str = line.split(",")

        if lbl_str not in lbl_dict:
            lbl_dict[lbl_str] = lbl_cnt
            lbl_cnt += 1

        val_lbl[cnt] = lbl_dict[lbl_str]
        val_dat[cnt] = ImageOps.fit(Image.open(raw_path + "images/%s" % fn), (st.W, st.H), method=Image.LANCZOS)

    for cnt, line in enumerate(tst_csv):
        fn, lbl_str = line.split(",")

        if lbl_str not in lbl_dict:
            lbl_dict[lbl_str] = lbl_cnt
            lbl_cnt += 1

        tst_lbl[cnt] = lbl_dict[lbl_str]
        tst_dat[cnt] = ImageOps.fit(Image.open(raw_path + "images/%s" % fn), (st.W, st.H), method=Image.LANCZOS)



    if not os.path.exists(st.preprocessed_data_path+"imagenet/"):
        os.makedirs(st.preprocessed_data_path+"imagenet")

    np.save(st.preprocessed_data_path+"imagenet/trn_dat.npy", trn_dat.astype(np.float32)/255.)
    np.save(st.preprocessed_data_path+"imagenet/trn_lbl.npy", trn_lbl)
    np.save(st.preprocessed_data_path+"imagenet/tst_dat.npy", tst_dat.astype(np.float32)/255.)
    np.save(st.preprocessed_data_path+"imagenet/tst_lbl.npy", tst_lbl)
    np.save(st.preprocessed_data_path+"imagenet/val_dat.npy", val_dat.astype(np.float32)/255.)
    np.save(st.preprocessed_data_path+"imagenet/val_lbl.npy", val_lbl)
    np.save(st.preprocessed_data_path+"imagenet/lbl_dict.npy", lbl_dict)

def process_msceleb():
    import base64
    from io import BytesIO

    trn_dat = np.zeros(shape=(1155175, st.W, st.H, st.C), dtype=np.uint8)
    trn_lbl = np.zeros(shape=len(trn_dat), dtype=np.uint16)
    tst_dat = np.zeros(shape=(25000, st.W, st.H, st.C), dtype=np.uint8)
    tst_lbl = np.zeros(shape=len(tst_dat), dtype=np.uint16)
    val_dat = np.zeros(shape=(1000, st.W, st.H, st.C), dtype=np.uint8)
    val_lbl = np.zeros(shape=len(val_dat), dtype=np.uint16)

    lbl_dict = {}
    lbl_cnt = 0

    raw_path = st.raw_data_path +"msceleb/"

    with open(raw_path + "TrainData_Base.tsv", "r") as f:
        for cnt, line in enumerate(f):
            words = line.split("\t")
            if words[2] not in lbl_dict:
                lbl_dict[words[2]] = lbl_cnt
                lbl_cnt += 1

            trn_lbl[cnt] = lbl_dict[words[2]]
            trn_dat[cnt] = ImageOps.fit(Image.open(BytesIO(base64.b64decode(words[1]))), (64, 64), method=Image.LANCZOS)

    with open(raw_path + "TrainData_lowshot.tsv", "r") as f:
        for cnt, line in enumerate(f):
            words = line.split("\t")
            if words[2] not in lbl_dict:
                lbl_dict[words[2]] = lbl_cnt
                lbl_cnt += 1

            val_lbl[cnt] = lbl_dict[words[2]]
            val_dat[cnt] = ImageOps.fit(Image.open(BytesIO(base64.b64decode(words[1]))), (64, 64),
                                        method=Image.ANTIALIAS)

    with open(raw_path + "DevelopmentSet.tsv", "r") as f:
        for cnt, line in enumerate(f):
            words = line.split("\t")

            tst_lbl[cnt] = lbl_dict[words[3]]
            tst_dat[cnt] = ImageOps.fit(Image.open(BytesIO(base64.b64decode(words[2]))), (64, 64),
                                        method=Image.ANTIALIAS)




    if not os.path.exists(st.preprocessed_data_path+"msceleb"):
        os.makedirs(st.preprocessed_data_path+"msceleb")

    np.save(st.preprocessed_data_path + "msceleb/trn_dat.npy", trn_dat.astype(np.float32)/255.)
    np.save(st.preprocessed_data_path + "msceleb/trn_lbl.npy", trn_lbl)
    np.save(st.preprocessed_data_path + "msceleb/val_dat.npy", val_dat.astype(np.float32)/255.)
    np.save(st.preprocessed_data_path + "msceleb/val_lbl.npy", val_lbl)
    np.save(st.preprocessed_data_path + "msceleb/tst_dat.npy", tst_dat.astype(np.float32)/255.)
    np.save(st.preprocessed_data_path + "msceleb/tst_lbl.npy", tst_lbl)
    np.save(st.preprocessed_data_path + "msceleb/lbl_dict.npy", lbl_dict)

def process_oxford():
    raw_path = st.raw_data_path+"oxford/"
    if not os.path.exists(raw_path+"jpg"):
        with tarfile.open(raw_path+"102flowers.tgz", "r:gz") as f:
            f.extractall(raw_path)

    all_lbl = scipy.io.loadmat(raw_path+"imagelabels.mat")["labels"].squeeze()
    all_dat_64 = np.zeros(shape=(len(all_lbl), 64, 64, st.C), dtype=np.uint8)
    all_dat_128 = np.zeros(shape=(len(all_lbl), 128, 128, st.C), dtype=np.uint8)
    all_dat_256 = np.zeros(shape=(len(all_lbl), 256, 256, st.C), dtype=np.uint8)
    all_dat_500 = np.zeros(shape=(len(all_lbl), 500, 500, st.C), dtype=np.uint8)

    for fn in sorted(glob(raw_path+"jpg/*.jpg"), key=os.path.basename):
        cnt = int(fn.split("_")[-1].split(".jpg")[0])-1
        cur_dat = Image.open(fn)
        all_dat_64[cnt] = ImageOps.fit(cur_dat, (64, 64), method=Image.LANCZOS)
        all_dat_128[cnt] = ImageOps.fit(cur_dat, (128, 128), method=Image.LANCZOS)
        all_dat_256[cnt] = ImageOps.fit(cur_dat, (256, 256), method=Image.LANCZOS)
        all_dat_500[cnt] = ImageOps.fit(cur_dat, (500, 500), method=Image.LANCZOS)
        if cnt%500==0:
            print(cnt, fn)


    if not os.path.exists(st.preprocessed_data_path+"oxford"):
        os.makedirs(st.preprocessed_data_path+"oxford")

    np.save(st.preprocessed_data_path + "oxford/all_dat_64.npy", all_dat_64.astype(np.float32)/255.)
    np.save(st.preprocessed_data_path + "oxford/all_dat_128.npy", all_dat_128.astype(np.float32)/255.)
    np.save(st.preprocessed_data_path + "oxford/all_dat_256.npy", all_dat_256.astype(np.float32)/255.)
    np.save(st.preprocessed_data_path + "oxford/all_dat_500.npy", all_dat_500.astype(np.float32)/255.)
    np.save(st.preprocessed_data_path + "oxford/all_lbl.npy", all_lbl-1)



def load_omniglot():
    if not os.path.exists(st.preprocessed_data_path+"omniglot/augment_tst_lbl.npy"):
        process_omniglot()

    rsync_data()

    trn_dat = np.load(st.temp_data_path+"omniglot/augment_trn_dat.npy", mmap_mode="r")
    trn_lbl = np.load(st.temp_data_path+"omniglot/augment_trn_lbl.npy", mmap_mode="r")

    tst_dat = np.load(st.temp_data_path+"omniglot/augment_tst_dat.npy", mmap_mode="r")
    tst_lbl = np.load(st.temp_data_path+"omniglot/augment_tst_lbl.npy", mmap_mode="r")

    return trn_dat, trn_lbl, tst_dat, tst_lbl

def load_imagenet():
    if not os.path.exists(st.preprocessed_data_path+"imagenet/lbl_dict.npy"):
        process_imagenet()

    rsync_data()

    trn_dat = np.load(st.temp_data_path+"imagenet/trn_dat.npy", mmap_mode="r")
    trn_lbl = np.load(st.temp_data_path+"imagenet/trn_lbl.npy", mmap_mode="r")

    _val_dat = np.load(st.temp_data_path+"imagenet/val_dat.npy", mmap_mode="r")
    _val_lbl = np.load(st.temp_data_path+"imagenet/val_lbl.npy", mmap_mode="r")

    tst_dat = np.load(st.temp_data_path+"imagenet/tst_dat.npy", mmap_mode="r")
    tst_lbl = np.load(st.temp_data_path+"imagenet/tst_lbl.npy", mmap_mode="r")

    trn_dat = np.concatenate((trn_dat, _val_dat), axis=0)
    trn_lbl = np.concatenate((trn_lbl, _val_lbl), axis=0)

    return trn_dat, trn_lbl, tst_dat, tst_lbl

def load_msceleb():
    if not os.path.exists(st.preprocessed_data_path+"msceleb/lbl_dict.npy"):
        process_msceleb()

    rsync_data()

    trn_dat = np.load(st.temp_data_path+"msceleb/trn_dat.npy", mmap_mode="r")
    trn_lbl = np.load(st.temp_data_path+"msceleb/trn_lbl.npy", mmap_mode="r")

    tst_dat = np.load(st.temp_data_path+"msceleb/tst_dat.npy", mmap_mode="r")
    tst_lbl = np.load(st.temp_data_path+"msceleb/tst_lbl.npy", mmap_mode="r")
    tst_dat = tst_dat[tst_lbl>19999]
    tst_lbl = tst_lbl[tst_lbl>19999]

    val_dat = np.load(st.temp_data_path+"msceleb/val_dat.npy", mmap_mode="r")
    val_lbl = np.load(st.temp_data_path+"msceleb/val_lbl.npy", mmap_mode="r")

    tst_dat = np.concatenate((tst_dat, val_dat), axis=0)
    tst_lbl = np.concatenate((tst_lbl, val_lbl), axis=0)

    return trn_dat, trn_lbl, tst_dat, tst_lbl

def load_oxford():
    if not os.path.exists(st.preprocessed_data_path+"oxford/all_lbl.npy"):
        process_oxford()

    rsync_data()

    all_dat = np.load(st.temp_data_path+"oxford/all_dat_%d.npy"%st.H, mmap_mode="r")
    all_lbl = np.load(st.temp_data_path+"oxford/all_lbl.npy", mmap_mode="r")

    with open(st.raw_data_path+"oxford/temp/trainvalids.txt", "r") as f:
        trn_lbl_idx = np.array(f.readlines(), dtype=np.uint8)-1
    tst_lbl_idx = np.array([lbl for lbl in np.unique(all_lbl) if lbl not in trn_lbl_idx])

    lbl_mask = np.array([all_lbl==lbl for lbl in np.unique(all_lbl)])

    trn_idx = np.any(lbl_mask[trn_lbl_idx], axis=0)
    tst_idx = np.any(lbl_mask[tst_lbl_idx], axis=0)

    trn_dat = all_dat[trn_idx]
    trn_lbl = all_lbl[trn_idx]
    tst_dat = all_dat[tst_idx]
    tst_lbl = all_lbl[tst_idx]

    return trn_dat, trn_lbl, tst_dat, tst_lbl


def split_way_shot(tst_dat, tst_lbl, way_cnt=-1, shot=-1, seed=-1, saved=True):
    if saved and os.path.exists(st.temp_data_path+"%s/new_tst_lbl_%d_%d.npy"%(st.dataset_dict[st.dataset], way_cnt, shot)):
        new_one_dat = np.load(st.temp_data_path+"%s/new_one_dat_%d_%d.npy"%(st.dataset_dict[st.dataset], way_cnt, shot), mmap_mode="r")
        new_tst_dat = np.load(st.temp_data_path+"%s/new_tst_dat_%d_%d.npy"%(st.dataset_dict[st.dataset], way_cnt, shot), mmap_mode="r")
        new_tst_lbl = np.load(st.temp_data_path+"%s/new_tst_lbl_%d_%d.npy"%(st.dataset_dict[st.dataset], way_cnt, shot), mmap_mode="r")
        return new_one_dat, new_tst_dat, new_tst_lbl

    if way_cnt==-1: way_cnt=st.c_way
    if shot==-1: shot=st.k_shot
    if seed==-1: seed=st.seed

    '''
    Omni 1692 classes 33840 samples (20 per class)
    MSCeleb 1000 clsses 6000 samples (6 per class)
    Imagenet 20 classes 12000 samples (600 per class)
    Oxford 20 classes 1155 samples (40 60 40 56 65 45 40 85 46 45 87 87 49 48 49 41 85 82 49 56)
    '''
    np.random.seed(seed)

    new_tst_dat = np.zeros(shape=(st.tst_cnt, st.H, st.W, st.C), dtype=tst_dat.dtype)
    new_tst_lbl = np.zeros(shape=(st.tst_cnt), dtype=np.uint16)

    unique_lbl = np.unique(tst_lbl)
    lbl_mask = np.array([tst_lbl==lbl for lbl in range(max(unique_lbl)+1)])

    query_lbl = np.random.choice(unique_lbl, size=st.tst_cnt, replace=len(unique_lbl)<st.tst_cnt)
    rest_lbl = (unique_lbl[..., None]!= query_lbl[None]).T
    dat_idx = np.zeros(shape=(st.tst_cnt, way_cnt, shot), dtype=np.int32)

    for cnt, (qlbl, rlbl) in enumerate(zip(query_lbl, rest_lbl)):
        slbl = unique_lbl[np.random.choice(np.argwhere(rlbl).squeeze(), way_cnt-1, replace=False)]
        slbl = np.random.permutation(np.concatenate((slbl, [qlbl]), axis=0))

        new_tst_lbl[cnt] = np.argwhere(slbl==qlbl)

        temp_idx = np.array([np.random.choice(np.argwhere(idx).squeeze(), shot+1, replace=False) for idx in lbl_mask[slbl]])
        new_tst_dat[cnt] = tst_dat[temp_idx[new_tst_lbl[cnt], -1]]
        dat_idx[cnt] = temp_idx[:, :-1]

    new_one_dat= tst_dat[dat_idx]

    np.save(st.temp_data_path+"%s/new_one_dat_%d_%d.npy"%(st.dataset_dict[st.dataset], way_cnt, shot), new_one_dat)
    np.save(st.temp_data_path+"%s/new_tst_dat_%d_%d.npy"%(st.dataset_dict[st.dataset], way_cnt, shot), new_tst_dat)
    np.save(st.temp_data_path+"%s/new_tst_lbl_%d_%d.npy"%(st.dataset_dict[st.dataset], way_cnt, shot), new_tst_lbl)

    return new_one_dat, new_tst_dat, new_tst_lbl

def rsync_data():
    import subprocess

    if not os.path.exists(st.temp_data_path+st.dataset_dict[st.dataset]):
        os.makedirs(st.temp_data_path+st.dataset_dict[st.dataset])

    subprocess.call(["rsync", "-a", st.preprocessed_data_path+st.dataset_dict[st.dataset]+"/",
                     st.temp_data_path+st.dataset_dict[st.dataset]])

def load_marg(dat):
    if not os.path.exists(st.preprocessed_data_path+"%s/marg.npy"%(st.dataset_dict[st.dataset])):
        eps = 1e-7
        marg = np.clip(dat.mean(axis=0), eps, 1. - eps)
        marg = np.log(marg / (1. - marg))
        np.save(st.preprocessed_data_path+"%s/marg.npy"%(st.dataset_dict[st.dataset]), marg)

def load_data():
    if st.dataset == 0: loader = load_omniglot
    elif st.dataset == 1: loader = load_msceleb
    elif st.dataset== 2: loader = load_imagenet
    elif st.dataset == 3 : loader = load_oxford


    trn_dat, trn_lbl, tst_dat, tst_lbl = loader()

    load_marg(trn_dat)

    one_dat = None

    if st.part_mode==1:
        one_dat, tst_dat, tst_lbl = split_way_shot(tst_dat=tst_dat, tst_lbl=tst_lbl, saved=True)
        tst_lbl = np.eye(st.c_way)[tst_lbl]

    return trn_dat, trn_lbl, one_dat, tst_dat, tst_lbl



