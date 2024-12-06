import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class MoCo(nn.Module):
    def __init__(self, encoder, dataset, dim, K, m, threshold, num_pos_samples):
        super(MoCo, self).__init__()
        self.encoder_q = encoder 
        self.encoder_k = encoder 
        self.K = K
        self.dim = dim
        self.m = m
        self.threshold = threshold
        self.num_pos_samples = num_pos_samples
        self._init_queue(dataset)
        self.register_buffer("weights", torch.ones(K))
    def _init_queue(self, dataset):
        K_div_3 = self.K // 3
        normal, fog, other = [], [], []
        for feature_i, label_i,_ in dataset:
            label_sum_i = label_i.sum(dim=0) 
            if label_sum_i[0] == 50:
                normal.append((feature_i, label_i))
            elif label_sum_i[1] == 50:
                fog.append((feature_i, label_i))
            else:
                other.append((feature_i, label_i))
        normal_samples = [normal[i] for i in torch.randperm(len(normal))[:K_div_3]]
        fog_samples = [fog[i] for i in torch.randperm(len(fog))[:K_div_3]]
        other_samples = [other[i] for i in torch.randperm(len(other))[:K_div_3]]
        selected_samples = normal_samples + fog_samples + other_samples
        features = []
        labels = []
        self.sample_init = []
        for data, label in selected_samples:
            data = data.to(next(self.encoder_q.parameters()).device)
            self.sample_init.append(data)
            self.encoder_k.eval()
            with torch.no_grad():
                feature = self.encoder_k(data.unsqueeze(0))
            features.append(feature)
            labels.append(label)
        self.queue = torch.cat(features, dim=0)
        self.queue = F.normalize(self.queue, dim=1)
        self.labels = torch.stack(labels)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = self.m * param_k.data + (1. - self.m) * param_q.data
        features = []
        for data in self.sample_init:
            with torch.no_grad():
                self.encoder_k.eval()
                feature = self.encoder_k(data.unsqueeze(0))             
            features.append(feature)
        self.queue = torch.cat(features,dim=0)

    def _update_queue(self, keys, labels):
        ptr = int(self.queue_ptr)
        update_length = keys.size(0) 
        self.queue[ptr: ptr+update_length] = keys.detach().clone()
        self.queue = F.normalize(self.queue, dim=1)
        self.labels[ptr:ptr+update_length] = labels
        ptr = 0
        self.queue_ptr[0] = ptr 

    def contrastive_loss_with_temperature(self, anchor_features, anchor_labels, queue_labels, queue_features, temperature=0.5, threshold=0.5):
        total_pos_loss = 0
        self.drop_pos = 0
        for anchor_feature, anchor_label in zip(anchor_features, anchor_labels):
            anchor_label_float = anchor_label.float().to(anchor_feature.device) 
            positive_samples = []
            # Random start index and step of 10
            index = random.randint(0, queue_labels.size(0) //10)
            for i in range(index, queue_labels.size(0), 10):
                cos_sim = abs(F.cosine_similarity(anchor_label_float, queue_labels[i].float().to(anchor_feature.device), dim=1))
                cos_sim = torch.mean(cos_sim)

                if cos_sim.item() >= threshold:
                    positive_samples.append(queue_features[i])
                if len(positive_samples) == self.num_pos_samples:
                    break
            # Ensure the correct number of positive samples
            if len(positive_samples) < self.num_pos_samples:
                if len(positive_samples) < 10:
                    self.drop_pos += 1
                positive_samples_selected = positive_samples
            else:
                positive_samples_selected = random.sample(positive_samples, self.num_pos_samples)
            # Skip if no positive samples are found
            if len(positive_samples_selected) == 0:
                continue
            positive_samples_selected = torch.stack(positive_samples_selected, dim=0)  
            q = anchor_feature.unsqueeze(0)  
            pos_sim = F.cosine_similarity(q, positive_samples_selected, dim=1) / temperature
            pos_loss = -torch.log(torch.sigmoid(pos_sim)).mean()
            total_pos_loss += pos_loss
        if self.drop_pos >= 5:
            print("Number of anchors with no positive samples in this batch:", self.drop_pos)  
            print(anchor_label)       
        return total_pos_loss / len(anchor_features)
    
    def forward(self, x_q, x_label):
        q = x_q  
        self._momentum_update_key_encoder() 
        k = self.queue  
        labels = self.labels
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        total_loss = self.contrastive_loss_with_temperature(anchor_features=q, anchor_labels=x_label,
                                                        queue_labels=labels, queue_features=k)
        return total_loss
    


