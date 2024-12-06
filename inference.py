import torch
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")

def S2P_result(predictions, group_size=50):
    grouped_labels = []
    group_indices = []
    num_groups = len(predictions) // group_size
    for i in range(num_groups):
        group_start = i * group_size
        group_end = (i + 1) * group_size
        group = predictions[group_start: group_end]
        counts = np.bincount(group, minlength=3)
        group_label = np.argmax(counts)
        grouped_labels.append(group_label)
        group_indices.append((group_start, group_end))
    return grouped_labels, group_indices

def inference(model, config, test_loader):
    model.eval() 
    predictions = []
    true_labels = []
    grouped_true_labels = []
    with torch.no_grad():
        for inputs, labels, Lab in test_loader:
            inputs, labels, Lab = inputs.to(config.device).float(), labels.to(config.device), Lab.to(config.device)
            labels = torch.argmax(labels.permute(0, 2, 1), dim=1).view(-1,).long()
            predicted, _ = model(inputs)
            predicted = predicted.reshape(-1, 3)
            true_labels.extend(labels.cpu().numpy())
            predicted = torch.argmax(predicted, dim=1)
            predictions.extend(predicted.cpu().numpy())
            grouped_true_labels.extend(Lab.cpu().numpy())
            
    print("S2S")
    report = classification_report(true_labels, predictions, target_names=['Normal', 'Fog', 'Pre_Fog'],digits=4)
    print(report)
    print("Classification Report:")
    print("Confusion Matrix:")
    print(confusion_matrix(true_labels, predictions))
    grouped_predictions, _ = S2P_result(predictions, group_size=50)
    print("S2P")
    report = classification_report(grouped_true_labels, grouped_predictions, target_names=['Normal', 'Fog', 'Pre_Fog'],digits=4)
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(confusion_matrix(grouped_true_labels, grouped_predictions))

