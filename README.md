# convert-to-tensorflow
Converts a variety of trained models to a frozen tensorflow protocol buffer file for use with the c++ tensorflow api.  C++ code is included for using the frozen models.

## Supported Architectures
This repo has been tested for convolutional regression and inference networks that contain a single input and a single ouput for image  models.

* Caffe model frozen using [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow)
* Keras models saved to hdf5 format (e.g. model.save('foo/bar.hdf5')

## Using a Frozen Model
Coming soon ...