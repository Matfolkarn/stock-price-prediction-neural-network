import torch
import torch.nn as nn




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
        layers.append(nn.Linear_units, backcast_length + forecast_length)
        self.model = nn.Sequential(*layers)
    
    def forward(self,x):
        y = self.model(x)
        backcast, forecast =torch.split(y, [self.backcast_length, self.forecast_length], dim=1)
        return backcast, forecast
    
class NbeatsStack(nn.Module):
    def __init__(self, 
                 input_sixe: int,
                 backcast_length: int,
                 forecast_length: int,
                 hidden_units: int,
                 hidden_layers: int,
                 num_blocks: int,
                 ):
        super(NbeatsStack, self).__init__()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length

        self.blocks = nn.ModuleList([Block(input_sixe, backcast_length, forecast_length, hidden_units, hidden_layers) for _ in range(num_blocks)])
    
    def forward(self, x):
        backcasts = []
        forecasts = []
        for block in self.blocks:
            backcast, forecast = block(x)
            backcasts.append(backcast.unsqueeze(1))
            forecasts.append(forecast.unsqueeze(1))
            x = torch.cat([x, backcast.unsqueeze(1)], dim=2)

            backcasts = torch.cat(backcasts, dim=1)
            forecasts = torch.cat(forecasts, dim=1)
            return backcasts, forecasts