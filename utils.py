#import library
import torch
import pdb

def train(model, train_dataset, val_dataset, optimizer, scheduler, \
            model_name= "model_weights/task4_model.pth", epochs= 4, patience=6):
    '''
    Creates the training pipeline by running training and validation steps
    '''
    model.train()
    loss_min = torch.tensor(1e10) #Min loss for comparison and saving best models
    patience_epochs =0 
    for epoch in range(epochs):    
        print(f"Epoch {epoch} training started.")
        train_epoch(model, train_dataset, optimizer)
        print(f"Epoch {epoch} validation started.")
        loss_val = val_epoch(model, val_dataset, epoch)
        scheduler.step(loss_val) #Scheduler changes learning rate based on criterion
        if loss_val<loss_min: #Model saved if min val loss obtained
            patience_epochs=0
            print("Model weights are saved.")
            loss_min = loss_val
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': loss_val,
            }, model_name)
        if patience_epochs>patience:
            break
        patience_epochs +=1
def train_epoch(model, train_dataset, optimizer):
    '''
    Training per epoch
    '''
    model.train()
    for idx, batch in enumerate(train_dataset):
        # pdb.set_trace()
        optimizer.zero_grad()
        loss = model.training_step(batch)
        loss.backward()
        optimizer.step() #Weights are updated
    

def val_epoch(model, val_dataset, epoch):
    '''
    Validation Step after each epoch
    '''
    model.eval()
    loss_val = 0
    for idx, batch in enumerate(val_dataset):
        metrics = model.validation_step(batch) #Gives the accuracy and Loss
        loss_val+=metrics['val_loss'] 
    return loss_val

def test_data(model, test_dataset):
    '''
    Gives the scores on the test set. 
    '''
    model.eval()
    model.eval()
    count = 0
    acc = 0
    for idx, batch in enumerate(test_dataset):
        metrics = model.validation_step(batch) #Gives the accuracy and Loss
        acc += len(batch)*metrics['val_acc']
        count+= len(batch)
    acc = acc/count
    return acc
    