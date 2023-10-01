import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from BaseModel import *
from ProcessingData import train_data,test_data,num_classes
from arg import *
from lion_pytorch import Lion

train_data = train_data 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
# Create data loaders
TRAINLOADER = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
TESTLOADER = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


# Init Model
model = GoogleNet(num_classes=num_classes)
model.to(DEVICE)

# Loss
criterion = nn.CrossEntropyLoss()

# Optimizer and Scheduler
# optimizer = optim.SGD(model.parameters(), lr=Lr)# Choose the optimizer
# optimizer = optim.Adam(model.parameters(), lr=Lr,weight_decay=1e-4)
# 
optimizer = Lion(model.parameters(), lr=Lr, weight_decay=1e-4)

# scheduler = CosineAnnealingLR(optimizer, T_max=5)  # Choose the scheduler
scheduler = ExponentialLR(optimizer, gamma =0.95)  # Choose the scheduler
