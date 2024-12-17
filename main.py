import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.config import TrainGlobalConfig
from unet import unet_18
from src.Read_data import read_data, Daphnetwindow
from src.FoG_MoCo import MoCo
from mamba_ssm.models.config_mamba import MambaConfig
from inference import inference
from train import *
from pre_train import Pretrain
from src.FoG_Mix import Mixup
import warnings


def load_data_for_pretrain(config):
    """
    Prepare datasets and dataloaders for pre-training.
    """
    _, train_set, valid_set, _ = read_data(config.data_path, stride=50)
    train_set = Mixup(train_set)
    train_data = Daphnetwindow(train_set, mode="pre")
    valid_data = Daphnetwindow(valid_set, mode="pre")

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=False, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=config.batch_size, shuffle=False, drop_last=True)

    return train_loader, valid_loader


def load_data_for_train(config, model_pretrain):
    """
    Prepare datasets and dataloaders for training with modified labels.
    """
    total_set, _, _, _ = read_data(config.data_path, stride=50, mode="label")
    total_data = Daphnetwindow(total_set, mode="label")
    total_loader = DataLoader(total_data, batch_size=config.batch_size, shuffle=False, drop_last=True)

    index_list_total = find_Pre_FoG_labels(total_loader, model_pretrain, config.device, tn=0.8, tf=0.7)
    train_set_new, valid_set_new, test_set_new = modify_labels(total_set, index_list_total)
    train_data_new = Daphnetwindow(train_set_new)
    valid_data_new = Daphnetwindow(valid_set_new)
    test_data_new = Daphnetwindow(test_set_new)

    train_loader_new = DataLoader(train_data_new, batch_size=config.batch_size, shuffle=False, drop_last=True)
    valid_loader_new = DataLoader(valid_data_new, batch_size=config.batch_size, shuffle=False, drop_last=True)
    test_loader_new = DataLoader(test_data_new, batch_size=config.batch_size, shuffle=False, drop_last=True)

    return train_loader_new, valid_loader_new, test_loader_new


def load_pretrained_weights(model, weight_path):
    """
    Load pre-trained weights into the model.
    """
    weight = torch.load(weight_path)
    model.load_state_dict(weight)
    return model


def main():
    config = TrainGlobalConfig()
    mconfig = MambaConfig()
    task_pretrain = config.task['Pre_train']
    task_train = config.task['Train']
    task_inference = config.task['Inference']
    model_pretrain = unet_18(9, 2, 50, mconfig=mconfig).to(config.device)
    model_train = unet_18(9, 3, 50, mconfig=mconfig).to(config.device)

    if task_pretrain:
        print("Starting pre-training...")
        train_loader, valid_loader = load_data_for_pretrain(config)
        optimizer = optim.Adam(model_pretrain.parameters(), lr=config.lr)
        Pretrain(model_pretrain, train_loader, valid_loader, optimizer, config, num_epochs=config.num_epochs_pretrain)

    elif task_train:
        print("Starting training...")
        model_pretrain = load_pretrained_weights(model_pretrain, config.pretrain_weight)
        train_loader, valid_loader, test_loader = load_data_for_train(config, model_pretrain)

        model_test_state_dict = model_pretrain.state_dict()
        model_state_dict = model_train.state_dict()
        for name, param in model_test_state_dict.items():
            if name in model_state_dict and model_state_dict[name].shape == param.shape:
                model_state_dict[name].copy_(param)
        model_train.load_state_dict(model_state_dict)

        optimizer = optim.Adam(model_train.parameters(), lr=config.lr)
        moco = MoCo(encoder=model_train.encoder, dataset=train_loader.dataset,
                    dim=1024, K=666, m=0.99, threshold=0.7, num_pos_samples=32).to(config.device)
        train(model_train, moco, train_loader, valid_loader, optimizer, config, num_epochs=config.num_epochs)

    elif task_inference:
        print("Starting inference...")
        model_pretrain = load_pretrained_weights(model_pretrain, config.pretrain_weight)
        _, _, test_loader = load_data_for_train(config, model_pretrain)
        train_weight = torch.load(config.train_weight)
        model_train.load_state_dict(train_weight)
        inference(model_train, config, test_loader)

    else:
        warnings.warn("No valid task selected. Please check the configuration.")

if __name__ == "__main__":
    main()
