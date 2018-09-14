import argparse
import onnx
import numpy as np
from onnx_tf.backend import prepare

def main():
    parser = argparse.ArgumentParser(description='Exports a model from ONNX format to Tensorflow JS.')
    parser.add_argument('model', help='Path to model file')
    args = parser.parse_args()

    print('Loading model...')
    model = onnx.load(args.model)
    print(onnx.helper.printable_graph(model.graph))
    print('Preparing TF...')
    tf_rep = prepare(model)
    print(tf_rep)
    state_size = 1024
    x = np.zeros((1,), dtype=np.int64)
    memory = np.zeros((2, 1, state_size))
    # temperature = 1
    results = tf_rep.run((x, memory))
    print(results)
    
    # print(tf_rep)
    print('Done.')

if __name__ == '__main__':
    main()