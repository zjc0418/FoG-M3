import torch
import torch.nn.functional as F
import tqdm
import numpy as np
from pre_train import Valid
from sklearn.model_selection import train_test_split
import random

def find_Pre_FoG_labels(loader, model, device,tn,tf):
    model.eval()
    index_list = []
    with torch.no_grad():
        for batch_idx, (inputs, labels, valid) in enumerate(tqdm.tqdm(loader, desc="Finding inconsistent labels")):
            inputs, labels, valid = inputs.to(device), labels.to(device), valid.to(device)
            outputs,_ = model(inputs)
            for sample_idx in range(outputs.size(0)):
                output = outputs[sample_idx].reshape(-1, 2)
                valid_seq = valid[sample_idx]
                max_vals, _ = output.max(dim=1, keepdim=True)
                predicted_labels = output - max_vals
                softmax = torch.nn.Softmax(dim=1)
                predicted_labels = softmax(output)
                
                predicted_labels_max = torch.argmax(output, dim=1)
                true_labels = torch.argmax(labels[sample_idx], dim=1)

                for i in range(len(predicted_labels)):
                    normal = (true_labels[i].item()== 0)
                    if valid_seq[i] == 1:
                        if normal:
                            if  predicted_labels[i][0].item() <= tn :
                                index_list.append((batch_idx * loader.batch_size + sample_idx, i)) 
                        else:
                            if  predicted_labels[i][1].item() <= tf:
                                index_list.append((batch_idx * loader.batch_size + sample_idx, i))             
    return index_list

def modify_labels(data, index_list, stride=50, i_test=True):
    for idx in tqdm.tqdm(range(len(data)), desc="Expanding labels to 4 columns"):
        inputs, original_labels = data[idx]
        if original_labels.shape[1] == 3:
            # Add a new column of zeros to expand labels to 4 columns
            expanded_labels = np.hstack((original_labels, np.zeros((original_labels.shape[0], 1))))
            data[idx] = (inputs, expanded_labels)

    for data_idx, row_idx in tqdm.tqdm(index_list, desc="Modifying specified labels"):
        inputs, labels = data[data_idx]
        labels[row_idx] = [0, 0, 0, 1]
        data[data_idx] = (inputs, labels)

    pre_fog = []
    fog = []
    normal = []
    for idx in tqdm.tqdm(range(len(data)), desc="Collecting samples with specific label conditions"):
        if idx == 0 or idx > len(data) - 5:
            continue
        inputs, labels = data[idx]
        col_sum = np.sum(labels, axis=0)

        for i in [1, 2, 3]:
            if col_sum[i] >= stride // 2:
                if i == 3:
                    pre_fog.append((inputs, labels, 2))
                elif i == 2:
                    fog.append((inputs, labels, 1))
                elif i == 1:
                    normal.append((inputs, labels, 0))
                break

    fog.extend(pre_fog)
    sample_size = len(fog) 
    sampled_data = normal[:sample_size]
    fog.extend(sampled_data)

    random.shuffle(fog)
    total_length = 0.8 * len(fog)
    train_count = int(0.8 * total_length)
    validation_count = int(0.1 * total_length)

    train_set = fog[:train_count]
    valid_set = fog[train_count:train_count + validation_count]
    test_set = fog[train_count + validation_count:]

    return train_set, valid_set, test_set


def train(model, moco, train_loader, valid_loader, optimizer, config, num_epochs=50, i_moco=True):
    train_losses = []
    valid_losses = []
    scheduler = config.SchedulerClass(optimizer, **config.scheduler_params)
    for epoch in range(num_epochs):
        model.train() 
        running_loss = 0.0
        train_loader_tqdm = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for step, (inputs, labels, Lab) in enumerate(train_loader_tqdm, 1):
            inputs, labels,Lab = inputs.to(config.device).float(), labels.to(config.device).float(),Lab.to(config.device) 
            moco_loss = 0
            optimizer.zero_grad()  
            predicted, moco_feature = model(inputs)  
            if  i_moco and step < 2 :
                moco_loss = moco(moco_feature,labels)
            labels = torch.argmax(labels.permute(0, 2, 1), dim=1).view(-1,).long()
            predicted = predicted.reshape(-1, 3)
            loss = F.cross_entropy(predicted, labels)
            loss = loss + moco_loss
            loss.backward()  
            optimizer.step() 
            running_loss += loss.item()
            train_loader_tqdm.set_postfix({'Train Loss': loss.item()})
        train_loss = running_loss / len(train_loader)  
        train_losses.append(train_loss)
        val_loss = Valid(model, valid_loader, config, pre_train=False)
        valid_losses.append(val_loss)
        scheduler.step(val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}")
    torch.save(model.state_dict(), config.train_weight)



    

