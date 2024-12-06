import torch
import torch.nn.functional as F
import os
import tqdm

def Valid(model, val_loader, config, pre_train=True):
    model.eval()  
    running_loss = 0.0
    y_true,y_pred = [],[]
    with torch.no_grad():
        if pre_train:
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(config.device).float(), labels.to(config.device)
                labels = torch.argmax(labels.permute(0, 2, 1),dim=1).view(-1,).long()
                outputs,_= model(inputs)
                predicted= outputs.reshape(-1,2)
                valid_loss = F.cross_entropy(predicted, labels)
                running_loss +=  valid_loss.item()
                y_true.extend(labels.cpu().numpy())
                predicted = torch.argmax(predicted, dim=1)
                y_pred.extend(predicted.cpu().numpy())      
        else:  
            for inputs, labels, Lab in val_loader:
                inputs, labels,Lab = inputs.to(config.device).float(), labels.to(config.device), Lab.to(config.device)
                labels = torch.argmax(labels.permute(0, 2, 1),dim=1).view(-1,).long()
                outputs,_= model(inputs)
                predicted= outputs.reshape(-1,3)
                valid_loss = F.cross_entropy(predicted, labels)
                running_loss +=  valid_loss.item()
                y_true.extend(labels.cpu().numpy())
                predicted = torch.argmax(predicted, dim=1)
                y_pred.extend(predicted.cpu().numpy())      
    val_loss = running_loss / len(val_loader)
    return val_loss


def Pretrain(model, train_loader, valid_loader, optimizer, config, num_epochs):
    scheduler = config.SchedulerClass(optimizer, **config.scheduler_params)
    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0
        train_loader_tqdm = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")      
        for step, (inputs, labels) in enumerate(train_loader_tqdm, 1):
            inputs, labels = inputs.to(config.device).float(), labels.to(config.device).float() 
            optimizer.zero_grad() 
            predicted, _ = model(inputs) 
            labels = labels.view(-1, 2)
            predicted = predicted.view(-1, 2)
            predicted = F.softmax(predicted, dim=1)
            loss = F.kl_div(predicted.log(), labels, reduction='batchmean')
            loss.backward()  
            optimizer.step()  
            running_loss += loss.item()  
            train_loader_tqdm.set_postfix({'Train Loss': loss.item()})
        train_loss = running_loss / len(train_loader)  
        val_loss = Valid(model, valid_loader, config)
        scheduler.step(val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}")
    torch.save(model.state_dict(), config.pertrain_weight)


    


