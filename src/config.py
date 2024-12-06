import torch
import os

class TrainGlobalConfig:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    data_path = "/mnt/d/study/dataset/daphnet/dataset"
    weight = "./weight"
    os.makedirs(weight, exist_ok=True)
    pertrain_weight = "./weight/weight_pretrain.pth"
    train_weight = "./weight/weight_train.pth"

    i_mix = True
    window_size = 50
    num_workers = 4
    batch_size = 64
    n_epochs = 100
    lr = 0.001
    verbose = True
    verbose_step = 1


    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau 
    scheduler_params = dict(
        mode="min", factor=0.7, patience=1, min_lr=0.000001
    ) 
