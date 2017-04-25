import torch
from music import NOTES_PER_BAR, NUM_CLASSES

def generate(model, num_bars=16):
    input = Variable(torch.zeros(NUM_CLASSES))
    for i in range(NOTES_PER_BAR * num_bars):
        out, states = model(input, states)

def main():
    model = DeepJ()
    generate(model)

if __name__ == '__main__':
    main()
