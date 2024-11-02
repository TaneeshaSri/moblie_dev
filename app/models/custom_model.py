import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomMobileNetV3(nn.Module):
    def __init__(self, num_classes=80):
        super(CustomMobileNetV3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 40, kernel_size=3, stride=2, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True),
            nn.Conv2d(40, 40, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True),
            nn.Conv2d(40, 40, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True),
            nn.Conv2d(40, 40, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True),
            nn.Conv2d(40, 80, kernel_size=3, stride=2, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True),
            nn.Conv2d(80, 80, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True),
            nn.Conv2d(80, 80, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(80, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class TimeDistributedConv2D(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0):
        super(TimeDistributedConv2D, self).__init__()
        self.conv2d = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)
        self.batchnorm2d = nn.BatchNorm2d(output_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        x = x.view(batch_size * timesteps, C, H, W)
        
        x = self.conv2d(x)
        x = self.batchnorm2d(x)
        x = F.relu(x)
        x = self.pool(x)
        _, C, H, W = x.size()
        x = x.view(batch_size, timesteps, C, H, W)
        return x

class MobileNetTimeDistributed(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetTimeDistributed, self).__init__()

        # Time-distributed convolution
        self.conv = TimeDistributedConv2D(3, 32, (3, 3), stride=1, padding=1)

        # Custom MobileNetV3 architecture
        self.mobilenetv3 = CustomMobileNetV3(num_classes=80)  # Output 80 features

        # LSTM layer
        self.lstm = nn.LSTM(80, 128, batch_first=True)

        # Final classification layer
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        batch_size, C, H, W = x.size()  # Assuming input has 4 dimensions (batch_size, C, H, W)

        # Apply time-distributed convolution
        x = self.conv(x.unsqueeze(1))  # Add a time dimension at index 1

        # Reshape for MobileNetV3
        x = x.view(batch_size, -1, *x.shape[2:])  # Reshape to (batch_size, timesteps, C, H, W)

        # Apply MobileNetV3 features
        features = []
        for i in range(x.size(1)):  # Iterate over timesteps
            out = self.mobilenetv3(x[:, i, :, :, :])
            features.append(out)

        # Stack features
        features = torch.stack(features, dim=1)

        # LSTM input shape (batch_size, timesteps, num_features)
        lstm_out, _ = self.lstm(features)

        # Final classification
        x = self.fc(lstm_out[:, -1, :])  # Take the last timestep output of LSTM

        return x