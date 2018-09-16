import argparse
import torch, torch.onnx
import torch.nn as nn
import constants as const
from model import DeepJ

class SoftmaxWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args):
        x, memory = self.module(*args)
        x = torch.softmax(x, dim=-1)
        return x, memory

def main():
    parser = argparse.ArgumentParser(description='Exports a model to ONNX format.')
    parser.add_argument('model', help='Path to model file')
    args = parser.parse_args()

    onnx_model_path = args.model.replace('.pt', '.onnx')

    print('Loading Pytorch model')
    model = DeepJ()
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model = SoftmaxWrapper(model)

    evt_input = torch.zeros((1,), dtype=torch.long)
    dummy_output, states = model(evt_input, None)
    
    print('Dummy output:', dummy_output)

    dummy_input = (evt_input, states)

    print('Exporting to ONNX format')
    torch.onnx.export(model, dummy_input, onnx_model_path, verbose=True)
    print('Done.')

if __name__ == '__main__':
    main()