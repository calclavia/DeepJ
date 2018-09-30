import argparse
import torch, torch.onnx
import torch.nn as nn
import constants as const
from model import DeepJ

def main():
    parser = argparse.ArgumentParser(description='Exports a model to ONNX format.')
    parser.add_argument('model', help='Path to model file')
    args = parser.parse_args()

    onnx_model_path = args.model.replace('.pt', '.onnx')

    print('Loading Pytorch model')
    model = DeepJ()
    model.load_state_dict(torch.load(args.model, map_location='cpu'))

    evt_input = torch.zeros((1,), dtype=torch.long)
    style_input = torch.zeros((1, model.style_units))
    dummy_output, states = model(evt_input, style_input)
    
    print('Dummy output:', dummy_output)

    dummy_input = (evt_input, style_input, states)

    print('Exporting to ONNX format')
    torch.onnx.export(model, dummy_input, onnx_model_path, verbose=True)
    print('Done.')

if __name__ == '__main__':
    main()