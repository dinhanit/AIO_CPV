from ModelConfig import *
from tqdm import tqdm
from arg import *
class Best_Model:
    def __init__(self,model,loss_train=[],loss_test=[],f1_train=[],f1_test=[]):
        self.model = model
        self.loss_train = loss_train
        self.loss_test = loss_test
        self.f1_train = f1_train
        self.f1_test = f1_test
        
def Saved_Model(model):
    name_model = input('Enter Name Model: ')
    if name_model != "":
        path_model ="Model/"
        torch.save(model.model,path_model+name_model+'.pth')
        import json
        data = {
                'loss_train':model.loss_train,
                'loss_test':model.loss_test,
                'f1_train':model.f1_train,
                'f1_test':model.f1_test
                }
        path_log = "log/"
        json_file_path = path_log+name_model+'.json'
        with open(json_file_path, 'w') as json_file:
            json.dump(data, json_file)
    print('Finished Training')
    
loss_train = []
loss_test = []
f1_train = []
f1_test = []
worse=0
stop = 100
min_loss = 10
for epoch in range(EPOCHS):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    predictions_train = []
    true_labels_train = []
    for i, data in tqdm(enumerate(TRAINLOADER), desc='train'):
        inputs, labels = data
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs =  model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        predictions_train.extend(predicted.tolist())
        true_labels_train.extend(labels.tolist())
    if scheduler is not None:
        scheduler.step()  # Update learning rate with scheduler
    train_loss = running_loss / len(TRAINLOADER)
    train_f1 = f1_score(true_labels_train, predictions_train, average='weighted')

    loss_train.append(train_loss)
    f1_train.append(train_f1)

    # Evaluation on test set
    model.eval()  # Set the model to evaluation mode
    test_loss_val = 0.0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for data in TESTLOADER:
            inputs, labels = data
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            #outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss_val += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.tolist())
            true_labels.extend(labels.tolist())

    test_loss_val /= len(TESTLOADER)
    test_f1_val = f1_score(true_labels, predictions, average='weighted')

    if test_loss_val > min_loss:
        worse +=1
        print(worse)
        if worse >= stop:
            Saved_Model(cache)
            loss_test.append(test_loss_val)
            f1_test.append(test_f1_val)
            print(f'Epoch [{epoch + 1}/{EPOCHS}]  - Train Loss: {train_loss:.4f} - Train F1: {train_f1:.4f} - Test Loss: {test_loss_val:.4f} - Test F1: {test_f1_val:.4f}')
            break
    else:
        worse = 0
        cache = Best_Model(model,loss_train,loss_test,f1_train,f1_test)
        min_loss = test_loss_val
        
    loss_test.append(test_loss_val)
    f1_test.append(test_f1_val)
    print(f'Epoch [{epoch + 1}/{EPOCHS}]  - Train Loss: {train_loss:.4f} - Train F1: {train_f1:.4f} - Test Loss: {test_loss_val:.4f} - Test F1: {test_f1_val:.4f}')

Saved_Model(cache)
