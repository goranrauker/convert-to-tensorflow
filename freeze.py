import os
import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph


def freeze(tf_session, model_name, model_input_name, width, height, channels, model_output_name):

    input_binary = True
    graph_def = tf_session.graph.as_graph_def()

    tf.train.Saver().save(tf_session, model_name + '.ckpt')
    tf.train.write_graph(tf_session.graph.as_graph_def(), logdir='.', name=model_name + '.binary.pb', as_text=not input_binary)

    # We save out the graph to disk, and then call the const conversion routine.
    checkpoint_state_name = model_name + ".ckpt.index"
    input_graph_name = model_name + ".binary.pb"
    output_graph_name = model_name + ".pb"

    input_graph_path = os.path.join(".", input_graph_name)
    input_saver_def_path = ""
    input_checkpoint_path = os.path.join(".", model_name + '.ckpt')

    output_node_names = model_output_name
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"

    output_graph_path = os.path.join('.', output_graph_name)
    clear_devices = False

    freeze_graph(input_graph_path, input_saver_def_path,
                 input_binary, input_checkpoint_path,
                 output_node_names, restore_op_name,
                 filename_tensor_name, output_graph_path,
                 clear_devices, "")

    print("Model loaded from: %s" % model_name)
    print("Output written to: %s" % output_graph_path)
    print("Model input name : %s" % model_input_name)
    print("Model input size : %dx%dx%d (WxHxC)" % (width, height, channels))
    print("Model output name: %s" % model_output_name)
