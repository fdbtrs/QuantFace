from easydict import EasyDict as edict

config = edict()
config.dataset = "emoreIresNetTunningSyntheticFP32" # training dataset
config.embedding_size = 512 # embedding size of model
config.momentum = 0.9
config.weight_decay =5e-4
config.batch_size = 128
# batch size per GPU
config.lr = 0.1
config.output = "output/output_r50_FP32_Synthetic" # train model output folder
config.goutput="output"
config.output32=  "/r50_fp32" #"/r50_fp32" |   /r100_fp32"
config.global_step=  181952#181952 # step to resume
config.s=64.0
config.m=0.5
config.std=0.05
config.wq=6
config.aq=6


config.loss="ArcFace"  #  Option : ElasticArcFace, ArcFace, ElasticCosFace, CosFace, MLLoss

# type of network to train [iresnet100 | iresnet50| iresnet18 | mobilefacenet]
config.network = "iresnet50"
config.SE=False # SEModule
config.loss ="ArcFace"
if (config.network == "mobilefacenet"):
    config.embedding_size = 128

if config.dataset == "emoreIresNetTunningSynthetic":
    config.rec = "./data/synthetic/training"
    config.num_classes = 85742
    config.num_image = 528227
    config.num_epoch =  11
    config.warmup_epoch = -1
    config.val_targets =["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
    config.eval_step= 5686
    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
            [m for m in [11] if m - 1 <= epoch])  # [m for m in [8, 14,20,25] if m - 1 <= epoch])
    config.lr_func = lr_step_func

