import torch
import torch.nn as nn
from torchvision import models

class CalibResnet(nn.Module):
    def __init__(self, in_channels = 1, num_layer = 5):
        super().__init__()
        weights = models.ResNet34_Weights.IMAGENET1K_V1
        self.model = models.resnet34(weights=weights)
        self.num_layer = num_layer
        self.model.conv1 = nn.Conv2d(in_channels = in_channels, 
            out_channels = self.model.conv1.out_channels,
            kernel_size = self.model.conv1.kernel_size,
            stride = self.model.conv1.stride,
            padding = self.model.conv1.padding,
            bias = self.model.conv1.bias
            )

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        if self.num_layer >= 1:
            x = self.model.layer1(x)
        if self.num_layer >= 2:
            x = self.model.layer2(x)
        if self.num_layer >= 3:
            x = self.model.layer3(x)
        if self.num_layer >= 4:
            x = self.model.layer4(x)
        return x
        # x = self.model.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.model.fc(x)

class CalibNet(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.camera_model = CalibResnet(in_channels=3)
        self.lidar_model = CalibResnet(in_channels=1)
        self.conv1 = nn.Conv2d(in_channels = 1024, 
            out_channels = 1024, kernel_size=4)
        self.conv2 = nn.Conv2d(in_channels = 1024, 
            out_channels = 2048, kernel_size=4)
        self.avg1 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(2048, 6)

    def forward(self, x_img, x_depth):
        x_img = self.camera_model(x_img)
        x_depth = self.lidar_model(x_depth)
        x_out = torch.cat([x_img, x_depth], dim=1)
        x_out = self.conv1(x_out)
        x_out = self.conv2(x_out)
        x_out = self.avg1(x_out)
        x_out = x_out.flatten(1)
        x_out = self.fc1(x_out)
        return x_out

class RCalibNet(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.camera_model = CalibResnet(in_channels=3)
        self.lidar_model = CalibResnet(in_channels=1)
        
        self.conv1 = nn.Conv2d(in_channels = 1024, 
            out_channels = 1024, kernel_size=4)
        self.conv1_bn = nn.BatchNorm2d(1024)
        self.relu_1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels = 1024, 
            out_channels = 2048, kernel_size=4)
        self.conv2_bn = nn.BatchNorm2d(2048)
        self.relu_2 = nn.ReLU(inplace=True)

        self.avg1 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        self.n_layers = 3
        self.hidden_dim = 4096
        self.rnn = nn.RNN(input_size=2048, hidden_size=self.hidden_dim, batch_first=True, num_layers=self.n_layers)        
        self.dropout_1 = nn.Dropout1d(0.2)
        self.fc1 = nn.Linear(4096, 128)
        self.dropout_2 = nn.Dropout1d(0.2)
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x_img, x_depth):
        x_img = self.camera_model(x_img)
        x_depth = self.lidar_model(x_depth)
        x_out = torch.cat([x_img, x_depth], dim=1)
        
        x_out = self.conv1(x_out)
        x_out = self.conv1_bn(x_out)
        x_out = self.relu_1(x_out)
        
        x_out = self.conv2(x_out)
        x_out = self.conv2_bn(x_out)
        x_out = self.relu_2(x_out)

        x_out = self.avg1(x_out)
        x_out = x_out.flatten(1)
        B, C = x_out.shape
        x_out, self.hidden = self.rnn(x_out.view(B,1,C), self.hidden)
        x_out = self.dropout_1(x_out)
        x_out = self.fc1(x_out)
        x_out = self.dropout_2(x_out)
        x_out = self.fc2(x_out)
        return x_out.flatten(1)
    
    def reset_hidden(self, B):
        self.hidden = torch.zeros(self.n_layers, B, self.hidden_dim).to(self.fc1.weight.device)
        


if __name__ == "__main__":
    model = CalibNet()
    print(model(torch.rand(8,3,512,512), torch.rand(8,1,512,512)))
    modelrnn = RCalibNet()
    modelrnn.reset_hidden(8)
    print(modelrnn(torch.rand(8,3,512,512), torch.rand(8,1,512,512)))
