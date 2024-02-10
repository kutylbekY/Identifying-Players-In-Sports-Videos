import torch
import torch.nn as nn

class PlayerTrackingModel(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(PlayerTrackingModel, self).__init__()  # Correctly call the superclass constructor
       
        # Example architecture:
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # Adjust input size
        self.relu3 = nn.ReLU()
        
        self.fc2 = nn.Linear(512, num_classes)  # Adjust output size

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        
        x = self.fc1(x)
        x = self.relu3(x)
        
        x = self.fc2(x)
        return x
