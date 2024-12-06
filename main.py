import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.config import TrainGlobalConfig
from unet import *
from src.Read_data import *
from src.FoG_MoCo import MoCo
from mamba_ssm.models.config_mamba import *
from inference import *
from train import *
from pre_train import *
from src.FoG_Mix import *

config = TrainGlobalConfig()
mconfig = MambaConfig()
Pre_train, Train, Inference = False, False, False

if Pre_train:
    total_set,train_set, valid_set, _ = read_data(config.data_path, stride = 50)
    train_set = Mixup(train_set)
    train_data = Daphnetwindow(train_set)
    valid_data = Daphnetwindow(valid_set)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True , drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=config.batch_size, shuffle=False, drop_last=True)  
    model = unet_18(9, 2, 50, mconfig=mconfig).to(config.device) 
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    Pretrain(model, train_loader, valid_loader, optimizer, config, num_epochs=2)

else:
    total_set,train_set, valid_set, _ = read_data(config.data_path,stride = 50, mode= "label")
    total_data = Daphnetwindow(total_set)
    total_loader = DataLoader(total_data, batch_size=config.batch_size, shuffle=False, drop_last=True)  
    model_pretrain = unet_18(9, 2, 50, mconfig=mconfig).to(config.device)
    model_train = unet_18(9, 3, 50, mconfig=mconfig).to(config.device)
    weight_pretrain = torch.load(config.pertrain_weight)
    model_pretrain.load_state_dict(weight_pretrain)
    index_list_total = find_Pre_FoG_labels(total_loader, model_pretrain, config.device, tn=0.8, tf=0.7)
    train_set_new,valid_set_new,test_set_new = modify_labels(total_set,index_list_total)
    model_test_state_dict = model_pretrain.state_dict()
    model_state_dict = model_train.state_dict()

    train_data_new = Daphnetwindow(train_set_new, mode="train")
    valid_data_new = Daphnetwindow(valid_set_new, mode="train")
    test_data_new = Daphnetwindow(test_set_new, mode="test")
    train_loader_new = DataLoader(train_data_new, batch_size=config.batch_size, shuffle=True ,drop_last=True)
    valid_loader_new = DataLoader(valid_data_new, batch_size=config.batch_size ,shuffle=False,drop_last=True)
    test_loader_new = DataLoader(test_data_new, batch_size=config.batch_size ,shuffle=False,drop_last=True)
    unique_labels = set()

    for _, labels,_ in train_loader_new:  # train_loader 是 DataLoader
        unique_labels.update(labels.numpy().flatten())  # 更新标签集合

    print("Unique labels in DataLoader:", unique_labels)

    if Train:
        for name, param in model_test_state_dict.items():
            if name in model_state_dict and model_state_dict[name].shape == param.shape:
                model_state_dict[name].copy_(param)
                model_train.load_state_dict(model_state_dict)
        optimizer = optim.Adam(model_train.parameters(), lr=config.lr)
        moco = MoCo(encoder=model_train.encoder, dataset = train_data_new, dim=1024, K=666, m=0.99, threshold=0.7, num_pos_samples=32).to(config.device)
        train(model_train, moco, train_loader_new, valid_loader_new, optimizer, config, num_epochs=2)
        
    elif Inference:
        weight_train = torch.load(config.train_weight)  
        model_train.load_state_dict(weight_train) 
        inference(model_train,config,test_loader_new)

    
    





