import torch
import torch.nn as nn
import torch.nn.functional as nnf

class sebblock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(sebblock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class convulation_block(nn.Module):
    def __init__(self):
        super(convulation_block, self).__init__()
        self.convulation1 = nn.Conv2d(18, 256, 3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(256)
        
    def forward(self, state):
        state = state.view(-1, 18, 8, 8)
        state = nnf.relu(self.batchnorm1(self.convulation1(state)))
        return state

class residual_block(nn.Module):
    def __init__(self, channel_in=256, channel_out=256, stride=1, padding=1, reduction=16):
        super(residual_block, self).__init__()
        self.convulation1 = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(channel_out)
        self.convulation2 = nn.Conv2d(in_channels=channel_out, out_channels=channel_out, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(channel_out)
        self.se_block = sebblock(channel_out, reduction)
        
    def forward(self, info):
        residual_info = info
        output = nnf.relu(self.batchnorm1(self.convulation1(info)))
        output = self.batchnorm2(self.convulation2(output))
        output = self.se_block(output)
        output += residual_info
        output = nnf.relu(output)
        return output
        
class output_block(nn.Module):
    def __init__(self):
        super(output_block, self).__init__()
        self._calculate_value()
        self._find_policy()
        
    def _calculate_value(self):  
        self.convulation_result = nn.Conv2d(256, 1, kernel_size=1)
        self.batchnorm_result = nn.BatchNorm2d(1)
        self.connection1 = nn.Linear(64, 256)
        self.bathchnormcont = nn.BatchNorm1d(256)
        self.connection2 = nn.Linear(256, 1)
    
    def _find_policy(self):
        self.convulation_policy = nn.Conv2d(256, 32, kernel_size=1)
        self.batchnorm_policy = nn.BatchNorm2d(32)
        self.connection_policy = nn.Linear(2048, 4672)
    
    def forward(self, state):
        value_of_state = nnf.relu(self.batchnorm_result(self.convulation_result(state)))
        value_of_state = value_of_state.view(-1, 64)
        value_of_state = nnf.relu(self.bathchnormcont(self.connection1(value_of_state)))
        value_of_state = torch.tanh(self.connection2(value_of_state))
        
        policy_of_state = nnf.relu(self.batchnorm_policy(self.convulation_policy(state)))
        policy_of_state = policy_of_state.view(-1, 2048)
        policy_of_state = self.connection_policy(policy_of_state)
        
        return value_of_state, policy_of_state

class chess_neural_network(nn.Module):
    def __init__(self):   
        super(chess_neural_network, self).__init__()
        self.convulationblocks = convulation_block()
        self.residualblocks = nn.ModuleList([residual_block() for i in range(15)])
        self.outputblock = output_block()
        
    def forward(self, value):
        output = self.convulationblocks(value)
        for block in self.residualblocks:
            output = block(output)
        output = self.outputblock(output)
        return output
        
        
