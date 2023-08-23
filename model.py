import torch
import torch.nn
import torch.nn.functional

class CrystalVision(torch.nn.Module):
    """A simple CNN for mineral classification."""
    def __init__(self, num_classes=7): 
        super(CrystalVision, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = torch.nn.Linear(64 * 32 * 32, 128)  # From preprocessing.py
        self.fc2 = torch.nn.Linear(128, num_classes)
        
    def forward(self, tensor):
        """
        Forward pass of the CrystalVision model.
        
        Parameters:
        - tensor (torch.Tensor): Input tensor representing a batch of images with shape [batch_size, channels, height, width].
        
        Returns:
        - torch.Tensor: Output tensor representing the class scores or probabilities for each image in the batch.
        """
        tensor = torch.nn.functional.relu(self.conv1(tensor))
        tensor = torch.nn.functional.max_pool2d(tensor, 2)
        tensor = torch.nn.functional.relu(self.conv2(tensor))
        tensor = torch.nn.functional.max_pool2d(tensor, 2)
        
        tensor = tensor.view(tensor.size(0), -1)
        tensor = torch.nn.functional.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor


model = CrystalVision()
