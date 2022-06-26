from easydict import EasyDict as edict

config = edict()
config.dataset = "emoreIresNetTunning" # training dataset
config.embedding_size = 512 # embedding size of model
config.momentum = 0.9
config.weight_decay =5e-4
config.batch_size = 128
# batch size per GPU
config.lr = 0.0001
config.output = "output_quant_syn_8_8" # train model output folder
config.goutput="output"
config.output32="/home/fboutros/ElasticFace/output/r100ElasticFaceArc"
config.global_step=181952 # step to resume
config.s=64.0
config.m=0.5
config.std=0.05
config.wq=8
config.aq=8


config.loss="ArcFace"  #  Option : ElasticArcFace, ArcFace, ElasticCosFace, CosFace, MLLoss

# type of network to train [iresnet100 | iresnet50]
config.network = "iresnet100"
config.SE=False # SEModule


if config.dataset == "emoreIresNetTunning":
    config.rec = "/data/fboutros/faces_emore"
    config.num_classes = 85742
    config.num_image = 8574200
    config.num_epoch =  10
    config.warmup_epoch = -1
    config.val_targets =["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
    config.eval_step=1000
    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
            [m for m in [3, 5,7] if m - 1 <= epoch])  # [m for m in [8, 14,20,25] if m - 1 <= epoch])
    config.lr_func = lr_step_func

if config.dataset == "emoreIresNet":
    config.rec = "/data/psiebke/faces_emore"
    config.num_classes = 85742
    config.num_image = 5822653
    config.num_epoch =  400
    config.warmup_epoch = -1
    config.val_targets =  []
    config.eval_step=500
    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
            [m for m in [100, 200,300] if m - 1 <= epoch])  # [m for m in [8, 14,20,25] if m - 1 <= epoch])
    config.lr_func = lr_step_func

elif config.dataset == "webface":
    config.rec = "/home/psiebke/faces_webface_112x112" #"/home/psiebke/faces_webface_112x112"
    config.num_classes = 10572
    config.num_image = 501195
    config.num_epoch = 34   #  [22, 30, 35]
    config.warmup_epoch = -1
    config.val_targets = ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
    config.eval_step= 958 #33350
    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
            [m for m in [20, 28, 32] if m - 1 <= epoch])
    config.lr_func = lr_step_func
