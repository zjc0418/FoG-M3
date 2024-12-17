import torch
import os

class TrainGlobalConfig:
    """
    Configuration class for training, pre-training, and inference tasks.
    Includes device settings, file paths, training parameters, and scheduler settings.
    """
    # Device setup
    device = (
        "cuda" if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() else 
        "cpu"
    )

    # File paths
    data_path = "/mnt/d/study/dataset/daphnet/dataset"  
    weight_dir = "./weight"  
    os.makedirs(weight_dir, exist_ok=True)
    pretrain_weight = os.path.join(weight_dir, "weight_pretrain.pth")
    train_weight = os.path.join(weight_dir, "weight_train.pth")

    # Task selection
    task = {'Pre_train': False, 'Train': False, 'Inference': True}

    # Data parameters
    window_size = 50  
    batch_size = 16  
    num_workers = 4  
    i_mix = True  

    # Training parameters
    num_epochs_pretrain = 100
    num_epochs = 2  
    lr = 0.001  
    verbose = True  
    verbose_step = 1  

    # Learning rate scheduler configuration
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode="min",        
        factor=0.7,        
        patience=1,        
        min_lr=0.000001    
    )
