import argparse
import keras
import keras.backend as K
import os
import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Attempts to export a keras trained inference or regression model to a tensorflow protocol buffer file")
    parser.add_argument("path_to_keras_model", type=str, help="path to the trained keras hdf5 model generated by calling model.save(filepath)")
    parser.add_argument("output_path", type=str, help="output path")
    args = parser.parse_args()

    input_binary = True
    K.set_learning_phase(0)

    path_to_keras_model_file = args.path_to_keras_model
    model_file_basename, file_extension = os.path.splitext(os.path.basename(path_to_keras_model_file))

    model = keras.models.load_model(path_to_keras_model_file)

    model_input = model.input.name.replace(':0', '')
    model_output = model.output.name.replace(':0', '')

    sess = K.get_session()

    # END OF keras specific code
    graph_def = sess.graph.as_graph_def()

    tf.train.Saver().save(sess, model_file_basename + '.ckpt')
    tf.train.write_graph(sess.graph.as_graph_def(), logdir='.', name=model_file_basename + '.binary.pb', as_text=not input_binary)

    # We save out the graph to disk, and then call the const conversion routine.
    checkpoint_state_name = model_file_basename + ".ckpt.index"
    input_graph_name = model_file_basename + ".binary.pb"
    output_graph_name = model_file_basename + ".pb"

    input_graph_path = os.path.join(".", input_graph_name)
    input_saver_def_path = ""
    input_checkpoint_path = os.path.join(".", model_file_basename + '.ckpt')

    output_node_names = model_output
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"

    output_graph_path = os.path.join(args.output_path, output_graph_name)
    clear_devices = False

    freeze_graph(input_graph_path, input_saver_def_path,
                 input_binary, input_checkpoint_path,
                 output_node_names, restore_op_name,
                 filename_tensor_name, output_graph_path,
                 clear_devices, "")

    print("Model loaded from: %s" % model_file_basename)
    print("Output written to: %s" % output_graph_path)
    print("Model input name : %s" % model_input)
    print("Model output name: %s" % model_output)
