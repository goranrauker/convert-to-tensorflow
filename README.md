# convert-to-tensorflow
Converts a variety of trained models to a frozen tensorflow protocol buffer file for use with the c++ tensorflow api.  C++ code is included for using the frozen models.

## Supported Architectures
This repo has been tested for convolutional regression and inference networks that contain a single input and a single ouput for image  models.

* Caffe model frozen using [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow)
* Keras models saved to hdf5 format (e.g. model.save('foo/bar.hdf5')

## Convert to TfLite
Likley want to run this part with a newer TF - perhaps 1.11 As documented here ...

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/toco/g3doc/python_api.md#api

```python
import tensorflow as tf

graph_def_file = "/path/to/frozen_graph.pb"
input_arrays = ["input"]
output_arrays = ["MobilenetV1/Predictions/Softmax"]

converter = tf.contrib.lite.TocoConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
```