from ModelConfig import *

input_size = 784
hidden_layers = [256, 128]
output_size = 10
dropout_p = 0.2
activation = nn.Tanh()

model = CustomNN(input_size, hidden_layers, output_size, dropout_p, activation)
print(model)