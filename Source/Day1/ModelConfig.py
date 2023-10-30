import torch.nn as nn

class CustomNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout_p, activation):
        super(CustomNN, self).__init__()

        self.layers = nn.ModuleList()  

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_layers[0]))
        self.layers.append(activation)
        self.layers.append(nn.Dropout(p=dropout_p))

        # Hidden layers
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            self.layers.append(activation)
            self.layers.append(nn.Dropout(p=dropout_p))

        # Output layer
        self.layers.append(nn.Linear(hidden_layers[-1], output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(1, 32, kernel_size=3, padding=1))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(32, 64, kernel_size=3, padding=1))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.layers.append(nn.Conv2d(64, 128, kernel_size=3, padding=1))  
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(128 * 7 * 7, 128)) 
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(128, 10)) 

    def forward(self, x):
        for i,layer in enumerate(self.layers):
            x = layer(x)
            print(i,x.size())
        return x
