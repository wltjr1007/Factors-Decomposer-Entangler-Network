
import argparse
import GPUtil
import os



parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=int, default=0)

parser.set_defaults(train=True)
parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--test', dest='train', action='store_false')

parser.set_defaults(deploy=True)
parser.add_argument('--develop', dest='deploy', action='store_false')

parser.add_argument('--mode', type=int, default=0)

parser.add_argument('--way', type=int, default=20)
parser.add_argument('--shot', type=int, default=1)

parser.add_argument('--GPU', type=int, default=-1)

parser.add_argument('--seed', type=int, default=0)

ARGS = parser.parse_args()

dataset_dict = {0: "omniglot", 1: "msceleb", 2: "imagenet", 3 : "oxford"}

mode_ae_dict = {0: "began", 1: "ali", 2:"ali_mine"}
mode_de_dict = {0: "siamese", 1: "proto"}
mode_part_dict = {0: "ae", 1: "style"}


dataset = ARGS.dataset
is_train= ARGS.train
mode = ARGS.mode
GPU = ARGS.GPU
c_way = ARGS.way
k_shot = ARGS.shot
seed = ARGS.seed
deploy = ARGS.deploy

ae_mode = mode//100
de_mode = (mode%100)//10
part_mode = mode%10




root_path = "/home/jsyoon/project/"
project_path = root_path+"FDAE/"
raw_data_path = root_path+"data/raw/"
preprocessed_data_path = root_path+"data/preprocessed/"
temp_data_path = "/home/jsyoon/temp/preprocessed/"
result_path = root_path+"results/FDAE/"

vis_cnt = 12
train_c_way = c_way
train_k_shot = k_shot

# ##########
# dataset=3
# mode=000
# #########

if GPU == -1:
    devices = "%d" % GPUtil.getFirstAvailable(order="memory")[0]
else:
    devices = "%d" % GPU
os.environ["CUDA_VISIBLE_DEVICES"] = devices
try:
    import tensorflow as tf
    tf.config.gpu.set_per_process_memory_growth(True)
except:
    pass


print("Mode %d, Dataset %d, GPU %d"%(mode, dataset, GPU), "//////////////////////////////////////")


bn = True
do = True
batch_cnt = 1000
tst_cnt = 1000
val_cnt = 16
batch_size=32

lambda_k = 0.001
gamma_k = 0.5

epoch =10000

lr = 0.0001
beta1 = 0.5
beta2 = 0.999

if dataset==0:
    W=H=32
    C=1
    z_dim = 128
    batch_size=100

    lrelu_alpha = 0.001

elif dataset ==1:
    W=H=64
    C=3
    z_dim = 64
    batch_size=100
    lrelu_alpha = 0.02

elif dataset ==2:
    W=H=64
    C=3
    z_dim = 128
    batch_size=100
    lrelu_alpha = 0.02

elif dataset==3:
    W=H=64
    C=3
    z_dim = 128
    batch_size=100
    lrelu_alpha = 0.01

if part_mode==1:
    batch_size = 16