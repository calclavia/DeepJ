import argparse
import onnx
import numpy as np
import tensorflow as tf
import constants as const
from onnx_tf.backend import prepare
from subprocess import call

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
    tf_outputs = [tf_rep.tensor_dict[output] for output in tf_rep.outputs]
    tf_outputs_names = [o.name.split(':')[0] for o in tf_outputs]
    print('TF output names:', tf_outputs_names)
    # print('TF graph:', list(tf_rep.tensor_dict.items()))

    state_size = const.NUM_UNITS
    x = np.zeros(tf_outputs[0].shape, dtype=np.int64)
    memory = np.zeros(tf_outputs[1].shape)
    results = tf_rep.run((x, memory))

    print('Dummy output (TF):', results)

    tf_rep.export_graph('/tmp/model.pb')
    print('Done.')

    print('Exporting to TFJS...')
    call(["tensorflowjs_converter", "/tmp/model.pb", "/tmp/tfjs", "--input_format", "tf_frozen_model", "--output_node_names", ','.join(tf_outputs_names)])
    call(["tar", "-czvf", "out/tfjs.tar.gz", "-C", "/tmp/tfjs/", "."])
    print('Done.')

if __name__ == '__main__':
    main()