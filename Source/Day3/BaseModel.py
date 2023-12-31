import torch
import torch.nn as nn

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        hidden_dim = in_channels * expand_ratio

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0, inverted_residual_setting=None):
        super(MobileNetV2, self).__init__()
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # Building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult)
        self.features = [nn.Sequential(
            nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )]
        # Building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
        # Building last several layers
        self.features.append(nn.Sequential(
            nn.Conv2d(input_channel, self.last_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.last_channel),
            nn.ReLU6(inplace=True)
        ))
        # Make it a nn.Module
        self.features = nn.Sequential(*self.features)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x




# class InceptionModule(nn.Module):
#     def __init__(self, in_channels, out1x1, reduce3x3, out3x3, reduce5x5, out5x5, out1x1pool):
#         super(InceptionModule, self).__init__()

#         # 1x1 convolution branch
#         self.branch1x1 = nn.Sequential(
#             nn.Conv2d(in_channels, out1x1, kernel_size=1),
#             nn.ReLU(inplace=True)
#         )

#         # 1x1 convolution followed by 3x3 convolution branch
#         self.branch3x3 = nn.Sequential(
#             nn.Conv2d(in_channels, reduce3x3, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(reduce3x3, out3x3, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )

#         # 1x1 convolution followed by 5x5 convolution branch
#         self.branch5x5 = nn.Sequential(
#             nn.Conv2d(in_channels, reduce5x5, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(reduce5x5, out5x5, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True)
#         )

#         # 3x3 max pooling followed by 1x1 convolution branch
#         self.branch1x1pool = nn.Sequential(
#             nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
#             nn.Conv2d(in_channels, out1x1pool, kernel_size=1),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         branch1x1 = self.branch1x1(x)
#         branch3x3 = self.branch3x3(x)
#         branch5x5 = self.branch5x5(x)
#         branch1x1pool = self.branch1x1pool(x)

#         outputs = [branch1x1, branch3x3, branch5x5, branch1x1pool]
#         return torch.cat(outputs, 1)

# class GoogleNet(nn.Module):
#     def __init__(self, num_classes=1000):
#         super(GoogleNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
#         self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
#         self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
#         self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
#         self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)

#         self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
#         self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
#         self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
#         self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
#         self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)

#         self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
#         self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.dropout = nn.Dropout(0.2)
#         self.fc = nn.Linear(1024, num_classes)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.maxpool1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.maxpool2(x)
#         x = self.inception3a(x)
#         x = self.inception3b(x)
#         x = self.maxpool3(x)
#         x = self.inception4a(x)
#         x = self.inception4b(x)
#         x = self.inception4c(x)
#         x = self.inception4d(x)
#         x = self.inception4e(x)
#         x = self.maxpool4(x)
#         x = self.inception5a(x)
#         x = self.inception5b(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.dropout(x)
#         x = self.fc(x)
#         return x

import torch
import torch.nn as nn

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out1x1, reduce3x3, out3x3, reduce5x5, out5x5, out1x1pool):
        super(InceptionModule, self).__init__()

        # 1x1 convolution branch
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out1x1, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # 1x1 convolution followed by 3x3 convolution branch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce3x3, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce3x3, out3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 1x1 convolution followed by 5x5 convolution branch
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce5x5, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce5x5, out5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )

        # 3x3 max pooling followed by 1x1 convolution branch
        self.branch1x1pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out1x1pool, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch1x1pool = self.branch1x1pool(x)

        outputs = [branch1x1, branch3x3, branch5x5, branch1x1pool]
        return torch.cat(outputs, 1)

class GoogleNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(GoogleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3a = InceptionModule(64, 32, 32, 64, 8, 16, 16)
        self.inception3b = InceptionModule(128, 64, 64, 128, 16, 32, 32)

        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionModule(256, 128, 128, 256, 32, 64, 64)
        self.inception4b = InceptionModule(512, 192, 96, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        
        # Calculate the input size for the FC layer dynamically
        self.fc_input_size = self._calculate_fc_input_size()

        self.fc = nn.Linear(self.fc_input_size, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def _calculate_fc_input_size(self):
        # This method calculates the input size for the FC layer dynamically
        x = torch.randn(1, 3, 224, 224)  # Input tensor with the same size as ImageNet images
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x.size(1)  # Return the



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out

class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet50, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(BasicBlock, 64, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

