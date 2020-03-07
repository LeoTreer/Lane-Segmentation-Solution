import torch.nn as nn


class Net(nn.Module):
    """
      most simple Model
      only one Linear layer 
    """
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        return output