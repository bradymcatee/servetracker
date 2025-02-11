import torch
import torch.nn as nn

class TrackNetModel(nn.Module):
    def __init__(self):
        super(TrackNetModel, self).__init__()
        # Encoder
        # First block
        self.conv1 = nn.Conv2d(9, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third block
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fourth block
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(512)
        
        # Decoder
        self.ups1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn13 = nn.BatchNorm2d(512)
        
        self.ups2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv14 = nn.Conv2d(512, 128, kernel_size=3, padding=1)
        self.bn14 = nn.BatchNorm2d(128)
        self.conv15 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn15 = nn.BatchNorm2d(128)
        
        self.ups3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv16 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn16 = nn.BatchNorm2d(64)
        self.conv17 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn17 = nn.BatchNorm2d(64)
        
        # Final convolution
        self.conv18 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.bn18 = nn.BatchNorm2d(256)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        # Encoder
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.pool3(x)
        
        x = self.relu(self.bn8(self.conv8(x)))
        x = self.relu(self.bn9(self.conv9(x)))
        x = self.relu(self.bn10(self.conv10(x)))
        
        # Decoder
        x = self.ups1(x)
        x = self.relu(self.bn11(self.conv11(x)))
        x = self.relu(self.bn12(self.conv12(x)))
        x = self.relu(self.bn13(self.conv13(x)))
        
        x = self.ups2(x)
        x = self.relu(self.bn14(self.conv14(x)))
        x = self.relu(self.bn15(self.conv15(x)))
        
        x = self.ups3(x)
        x = self.relu(self.bn16(self.conv16(x)))
        x = self.relu(self.bn17(self.conv17(x)))
        
        x = self.relu(self.bn18(self.conv18(x)))
        x = torch.softmax(x, dim=1)  # Apply softmax as per paper
        
        # Take the first channel as our heatmap
        x = x[:, 0:1, :, :]
        
        return x
