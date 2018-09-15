import argparse
import onnx
import numpy as np
import tensorflow as tf
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

    print('ONNX output names:', tf_rep.outputs)
    print('TF output names:', [tf_rep.tensor_dict[output] for output in tf_rep.outputs])
    print('TF graph:', list(tf_rep.tensor_dict.items()))

    state_size = 1024
    x = np.zeros((1,), dtype=np.int64)
    memory = np.zeros((2, 1, state_size))
    # temperature = 1
    results = tf_rep.run((x, memory))

    print('Dummy output (TF):', results)

    tf_rep.export_graph('/tmp/model.pb')
    print('Done.')

if __name__ == '__main__':
    main()