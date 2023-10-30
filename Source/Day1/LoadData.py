from torch.utils.data import DataLoader
from DataSet import trainset,testset
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 200
BATCH_SIZE = 256  # You can adjust this based on your needs
# Create data loaders
TRAINLOADER = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
TESTLOADER = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)