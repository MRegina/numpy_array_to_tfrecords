# numpy_array_to_tfrecords
TFRecord converter for numpy array data (e.g. 3D images for medical image processing)

numpy3d_to_tfrecords.py contains code for a TFRecords writer for Numpy array data specifically for 3D arrays with an additional channel dimension, however the interpret_npy_header function can be applied for any type of numpy array, so it is quite straightforward to rewrite this code for different dimensionality. Now it is implemented for numpy array shape [width,hight,depth,num_channels].

test_tfrecords.py can read in batches from the created TFRecords files and shows how to decode numpy arrays from the TFRecord data.

Example .npy data can be found in the input folder (3D images with one color channel, i.e grayscale), with train, validation and test .csv files containing filenames and labels. The output folder contains TFRecord files that should be generated. 

Code tested on Tensorflow-GPU 1.7.0 and Windows 10
