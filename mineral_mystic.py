import torch.nn

class MineralMystic(torch.nn.Module):
    def __init__(self):
        super(MineralMystic, self).__init__()

        # Model architecture inspired by Brandon Bennett Kaggle notebook
        # add link here
        self.conv1 = torch.nn.Conv2d(3, 48, 11, stride=3, padding=0)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(3, 1)
        
        self.conv2 = torch.nn.Conv2d(48, 128, 5, stride=1, padding=0)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(3, 1)

        self.conv3 = torch.nn.Conv2d(128, 128, 4, stride=1, padding=0)
        self.relu3 = torch.nn.ReLU()
        self.pool3 = torch.nn.MaxPool2d(4, 3)

        self.conv4 = torch.nn.Conv2d(128, 64, 3, stride=1, padding=0)
        self.relu4 = torch.nn.ReLU()
        self.pool4 = torch.nn.MaxPool2d(3, 3)

        self.flatten = torch.nn.Flatten()
        self.fc1     = torch.nn.Linear(6*6*64, 512)
        self.relu5   = torch.nn.ReLU()
        self.drop    = torch.nn.Dropout(p=0.3)
        self.fc2     = torch.nn.Linear(512, 7)
        self.softmax = torch.nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        x = self.conv1(x) #ouput 72X72
        x = self.relu1(x)
        x = self.pool1(x) #output 70X70

        x = self.conv2(x) #output 66X66
        x = self.relu2(x)
        x = self.pool2(x) #output 64X64

        x = self.conv3(x) #output 61X61
        x = self.relu3(x)
        x = self.pool3(x) #output 20X20

        x = self.conv4(x) #output 18X18
        x = self.relu4(x)
        x = self.pool4(x) #output 6X6

        x = self.flatten(x) #output 2306
        
        x = self.fc1(x) #output 512
        x = self.relu5(x)
        x = self.drop(x)
        x = self.fc2(x) #output 7
        x = self.softmax(x)
        
        return x
