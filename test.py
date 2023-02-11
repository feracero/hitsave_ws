import torch.nn as nn
import torch

class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(NeuralNet).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.mlp = nn.Sequential(
            nn.Linear(self.input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, self.output_size)
        )

    def forward(self, x):
        return self.mlp(x)

class Runner():
    def __init__(self) -> None:
        pass

    def run(self):
        pass
        

if __name__=="__main__":
    runner = Runner()
    runner.run()