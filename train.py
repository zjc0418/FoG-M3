import torch
import torch.nn.functional as F
import tqdm
import numpy as np
from pre_train import Valid
from sklearn.model_selection import train_test_split

def find_Pre_FoG_labels(loader, model, device,tn,tf):
    model.eval()
    index_list = []
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm.tqdm(loader, desc="Finding inconsistent labels")):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs,_ = model(inputs)

            for sample_idx in range(outputs.size(0)):
                output = outputs[sample_idx].reshape(-1, 2)

                max_vals, _ = output.max(dim=1, keepdim=True)
                predicted_labels = output - max_vals
                softmax = torch.nn.Softmax(dim=1)
                predicted_labels = softmax(output)
                
                predicted_labels_max = torch.argmax(output, dim=1)
                true_labels = torch.argmax(labels[sample_idx], dim=1)

                for i in range(len(predicted_labels)):
                    normal = (true_labels[i].item()== 0)
                    if normal:
                        if  predicted_labels[i][0].item() <= tn :
                            index_list.append((batch_idx * loader.batch_size + sample_idx, i)) 
                    else:
                        if  predicted_labels[i][1].item() <= tf:
                            index_list.append((batch_idx * loader.batch_size + sample_idx, i))             
    return index_list

def get_valid_indices(original_labels):
    valid_indices = []
    for center_idx in range(1, len(original_labels) - 1):  
        if (
            original_labels[center_idx][1] == 2 and
            original_labels[center_idx - 1][1] == 1 and
            original_labels[center_idx + 1][1] == 2
        ):
            valid_indices.append(center_idx)
    return valid_indices

def is_within_valid_range(valid_indices, target_idx):
    if not valid_indices:
        return False
    
    closest_indices = sorted(valid_indices, key=lambda x: abs(x - target_idx))

    if len(closest_indices) < 2:
        return False

    for idx in closest_indices[:2]:
        if abs(idx - target_idx) <= 576 or abs(idx - target_idx) <= 64:
            return True

    return False

def modify_labels(data, index_list, stride=50):
    for idx in tqdm.tqdm(range(len(data)), desc="Expanding labels to 4 columns"):
        inputs, original_labels = data[idx]
        if original_labels.shape[1] == 3:
            expanded_labels = np.hstack((original_labels, np.zeros((original_labels.shape[0], 1))))
            data[idx] = (inputs, expanded_labels)


    for global_idx, row_idx in tqdm.tqdm(index_list, desc="Modifying specified labels"):
        data_idx = global_idx // data[0][0].shape[0]
        local_sample_idx = global_idx % data[0][0].shape[0]

        if data_idx >= len(data):
            continue

        inputs, labels = data[data_idx]
        original_labels = labels[:, :3]
        col_sum = np.sum(labels, axis=0)

        if col_sum[2] != stride:
            continue

        valid_indices = get_valid_indices(original_labels)

        if is_within_valid_range(valid_indices, row_idx):
            labels[row_idx] = [0, 0, 0, 1]
            data[data_idx] = (inputs, labels)

    processed_data = []
    for inputs, labels in tqdm.tqdm(data, desc="Calculating dominant labels"):
        col_sum = np.sum(labels, axis=0)  
        max_idx = np.argmax(col_sum[1:]) + 1  
        n = max_idx  
        processed_data.append((inputs, labels, n))
    
    train_data, temp_data = train_test_split(processed_data, test_size=0.2, random_state=2024)
    valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=2024)

    return train_data, valid_data, test_data


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
            if  i_moco and step <5 :
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


    

