import random
import numpy as np
import tqdm
from torch.distributions.beta import Beta


def _mixup_function(x, y, cigma):
    mix_score = abs(x[1] + y[1] - y[2] - x[2]) 
    if mix_score <= cigma:
        return True
    if mix_score == 100:
        for i in range(3):
            if (x[i] in [0, 50] and x[i] != y[i]) or (y[i] in [0, 50] and y[i] != x[i]):
                return True
    return False

def mix_up(feature_i, feature_j, label_i, label_j, alpha):
    lam = Beta(alpha, alpha).sample().item()
    feature_mixup = feature_i * lam + feature_j * (1 - lam)
    label_mixup = label_i * lam + label_j * (1 - lam)
    return feature_mixup, label_mixup

def Mixup(dataset):
    mix_x, mix_y = [], []
    alpha = 32
    cigma = 30
    a_normal = []
    a_fog = []
    other = []
    for i, (feature_i, label_i,label_sum_i) in tqdm.tqdm(enumerate(dataset), total=len(dataset), desc="Mixup Progress"):
        if label_sum_i[1] == 50 :
            a_normal.append((feature_i, label_i,label_sum_i))
        elif label_sum_i[2] == 50:
            a_fog.append((feature_i, label_i,label_sum_i))
        else:
            other.append((feature_i, label_i,label_sum_i))
    random.shuffle(a_fog)
    random.shuffle(a_normal)
    index = []
    for i, (feature_i, label_i,label_sum_i) in tqdm.tqdm(enumerate(a_fog), total=len(a_fog), desc="Mixup Progress I"):
        for j,(feature_j, label_j, label_sum_j) in enumerate(a_normal):
            if j in index:
                continue
            if _mixup_function(label_sum_i, label_sum_j, cigma=cigma):
                feature_mixup, label_mixup = mix_up(feature_i, feature_j, label_i, label_j, alpha=alpha)
                mix_x.append(feature_mixup)
                mix_y.append(label_mixup)
                index.append(i)
                break
    index = []
    for i, (feature_i, label_i,label_sum_i) in tqdm.tqdm(enumerate(other), total=len(other), desc="Mixup Progress II"):
        for j,(feature_j, label_j, label_sum_j) in enumerate(other):
            if j in index or i == j:
                continue
            if _mixup_function(label_sum_i, label_sum_j, cigma=cigma):
                feature_mixup, label_mixup = mix_up(feature_i, feature_j, label_i, label_j, alpha=alpha)
                mix_x.append(feature_mixup)
                mix_y.append(label_mixup)
                index.append(i)
                index.append(j)
                break 
    random.shuffle(a_normal)
    for i in range(len(mix_x)):
        random_numbers = np.random.randint(0,len(a_normal))
        feature_j, label_j, label_sum_j = a_normal[random_numbers]
        mix_x.append(feature_j)
        mix_y.append(label_j)
            
    for i in range(len(a_fog)):
        feature_j, label_j, label_sum_j = a_fog[i]
        mix_x.append(feature_j)
        mix_y.append(label_j)

    mix_set = [(mix_x[i], mix_y[i]) for i in range(len(mix_x))]
    print(len(mix_set))
    return mix_set