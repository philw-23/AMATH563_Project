import torch
import torchvision
import torch.nn as nn

class AlexNet(nn.Module):
    """
    Neural network model consisting of layers propsed by AlexNet paper.
    """
    def __init__(self, num_classes, drop_prob=0.5, batch_norm=True):
        """
        Define and allocate layers for this neural net.
        Args:
            num_classes (int): number of classes to predict with this model
        """
        super().__init__()
        
        # Define the layers
        self.net = self.create_net(drop_prob, batch_norm)
        
        self.classifier = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.BatchNorm1d(12544),
            nn.Linear(12544, 4096),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )
       
        # initialize bias
        self.init_bias(batch_norm)

    def create_net(self, drop_prob, batch_norm):
        
        if batch_norm:
            
            layers = [
                nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2), # Manual Bias Line
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1), # Manual Bias Line
                nn.BatchNorm2d(384),
                nn.ReLU(),
                nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1), # Manual Bias Line
                nn.ReLU(),
                nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            
        else:
            
            layers = [
                nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2), # Manual Bias Line
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1), # Manual Bias Line
                nn.ReLU(),
                nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1), # Manual Bias Line
                nn.ReLU(),
                nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2)    
            ]
            
        return nn.Sequential(*layers)
        
    def init_bias(self, batch_norm):
        # Initialize weights according to original paper
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
                
        if batch_norm:
            nn.init.constant_(self.net[4].bias, 1)
            nn.init.constant_(self.net[8].bias, 1)
            nn.init.constant_(self.net[11].bias, 1)

        else:
            nn.init.constant_(self.net[4].bias, 1)
            nn.init.constant_(self.net[10].bias, 1)
            nn.init.constant_(self.net[12].bias, 1)
            
        
    def forward(self, x):

        x = self.net(x)
        x = x.flatten(1) # Flatten starting at position 1
        
        return self.classifier(x)
