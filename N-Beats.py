import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self,
                  input_size: int,
                  backcast_length: int,
                  forecast_length: int,
                  hidden_units: int,
                  hidden_layers: int):
        super(Block, self).__init__()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
    
        layers = []
        for _ in range(hidden_layers):
            layers.append(nn.Linear(input_size, hidden_units))
            layers.append(nn.ReLU())
            input_size = hidden_units
        #layers.append(nn.Linear( hidden_units, backcast_length + forecast_length))
        self.model = nn.Sequential(*layers)
    
    def forward(self,x):
        y = self.model(x)
        y_1d = y.squeeze()
        #backcast, forecast =torch.split(y_1d, [self.backcast_length, self.forecast_length], dim=0)
    
        return y_1d

class genericBlock(Block):
    def __init__(self, input_size: int, backcast_length: int, forecast_length: int, hidden_units: int, hidden_layers: int, theta: int):
        super(genericBlock, self).__init__(input_size, backcast_length, forecast_length, hidden_units, hidden_layers)

        self.relu = nn.ReLU()
        self.backCast_layer_1 = nn.Linear(hidden_units,theta)
        self.backCast_layer_2 = nn.Linear(theta, backcast_length)

        self.foreCast_layer_1 = nn.Linear(hidden_units, theta)
        self.foreCast_layer_2 = nn.Linear(theta, forecast_length)

    def forward(self, x):
        x = super(genericBlock, self).forward(x)
        print(x)
    
        x = torch.Tensor(torch.cat(x, dim=0))
        bc = self.backCast_layer_1(x)
        bc_relu = self.relu(bc)
        bc = self.backCast_layer_2(bc_relu)

        fc = self.foreCast_layer_1(x)
        fc_relu = self.relu(fc)
        fc = self.foreCast_layer_2(fc_relu)
        return bc, fc


class NbeatsStack(nn.Module):
    def __init__(self, 
                 input_size: int,
                 backcast_length: int,
                 forecast_length: int,
                 hidden_units: int,
                 hidden_layers: int,
                 num_blocks: int,
                 ):
        super(NbeatsStack, self).__init__()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length

        self.blocks = nn.ModuleList([Block(input_size, backcast_length, forecast_length, hidden_units, hidden_layers) for _ in range(num_blocks)])
        self.generic = genericBlock(input_size,backcast_length,forecast_length,hidden_units,hidden_layers,4)
    
    def forward(self, x):
        
        forecasts = torch.tensor([0 for _ in range(self.forecast_length)])
        for block in self.blocks:
            backcast, forecast = block(x)    
            x = x - backcast.squeeze()
            forecasts =  forecasts + forecast.squeeze()
        fc = forecasts
        return x, fc
        
class Nbeat(nn.Module):
    def __init__(self, 
                 input_size: int,
                 backcast_length: int,
                 forecast_length: int,
                 hidden_units: int,
                 hidden_layers: int,
                 num_blocks: int,
                 num_stacks: int
                 ):
        super(Nbeat, self).__init__()
        self.forecast_length = forecast_length    

        self.stacks = nn.ModuleList([NbeatsStack(input_size, backcast_length, forecast_length, hidden_units, hidden_layers, num_blocks) for _ in range(num_stacks)])

    def forward(self, x):
        backcasts = []
        fcs = torch.tensor([0 for _ in range(self.forecast_length)])
        for stack in self.stacks:
            backcast, forecast = stack(x)
            fcs = fcs + forecast
            x = backcast.squeeze()

       
        return backcast, fcs
