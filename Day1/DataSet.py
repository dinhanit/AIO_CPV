import torchvision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as transforms

# Load FashionMNIST dataset

transform = torchvision.transforms.ToTensor()  # You can add more transformations if needed
trainset = torchvision.datasets.FashionMNIST(root='./Day1/data', train=True, download=False, transform=transform)
testset = torchvision.datasets.FashionMNIST(root='./Day1/data', train=False, download=False, transform=transform)

