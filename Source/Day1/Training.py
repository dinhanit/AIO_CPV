from ModelConfig import CustomNN
from LoadData import *
from sklearn.metrics import f1_score
import torch.nn as nn
import torch
import torch.optim as optim
from arg import *
import matplotlib.pyplot as plt

# activation = nn.Tanh()
activation = nn.ReLU()


model = CustomNN(input_size, hidden_layers, output_size, dropout_p, activation)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=Lr)
optimizer = optim.Adam(model.parameters(), lr=Lr)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=7)
# scheduler = None

loss_train = []
loss_test = []
f1_train = []
f1_test = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    predictions_train = []
    true_labels_train = []
    for i, data in enumerate(TRAINLOADER, 0):
        inputs, labels = data
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs =  model(inputs.view(-1, input_size))
        # outputs =  model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        predictions_train.extend(predicted.tolist())
        true_labels_train.extend(labels.tolist())
    if scheduler is not None:
        scheduler.step()
    train_loss = running_loss / len(TRAINLOADER)
    train_f1 = f1_score(true_labels_train, predictions_train, average='weighted')

    loss_train.append(train_loss)
    f1_train.append(train_f1)

    model.eval()
    test_loss_val = 0.0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for data in TESTLOADER:
            inputs, labels = data
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs.view(-1, input_size))
            #outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss_val += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.tolist())
            true_labels.extend(labels.tolist())

    test_loss_val /= len(TESTLOADER)
    test_f1_val = f1_score(true_labels, predictions, average='weighted')

    loss_test.append(test_loss_val)
    f1_test.append(test_f1_val)
    print(f'Epoch [{epoch + 1}/{EPOCHS}]  - Train Loss: {train_loss:.4f} - Train F1: {train_f1:.4f} - Test Loss: {test_loss_val:.4f} - Test F1: {test_f1_val:.4f}')


name_model = input('Enter Name Model: ')
if name_model != "":
    
    path_model ="Day1/Model/"
    # torch.save(model.state_dict(), path)
    torch.save(model,path_model+name_model+'.pth')

    import json
    data = {
            'loss_train':loss_train,
            'loss_test':loss_test,
            'f1_train':f1_train,
            'f1_test':f1_test
            }
    path_log = "Day1/log/"
    json_file_path = path_log+name_model+'.json'
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file)
print('Finished Training')