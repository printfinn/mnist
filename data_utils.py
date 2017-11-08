import os
import struct
import numpy as np

def loadMNIST(dataset="training", path=".", num_image=1):
	'''
	loads data from MNIST datasets
	Inputs: ubyte file of either image or label

	Returns: numpy matrix
	'''
	if dataset == "training":
		image_file = os.path.join(path, "dataset/train-images.idx3-ubyte")
		label_file = os.path.join(path, "dataset/train-labels.idx1-ubyte")

	if dataset == "test":
		image_file = os.path.join(path, "dataset/t10k-images.idx3-ubyte")
		label_file = os.path.join(path, "dataset/t10k-labels.idx1-ubyte")
	
	with open(image_file, 'br') as fd:  # b for binary, r for read only

		zero, zero, data_type_code, num_dimensions = struct.unpack('>BBBB', fd.read(4))

		# How this works:
		# idx file is C-like coded, to extract the datasets from it, follow the code
		# rule on http://yann.lecun.com/exdb/mnist/
		#
		# For example, the first 4 bytes of train-images.idx3-ubyte is 0x00 00 08 03, 
		# unpack these 4 bytes with Python Struct library to get 0 0 8 3. 
		# The parameter '>BBBB' indicates high endian, uchar, uchar, uchar, uchar(1 byte)
		# whereas '>I' indicates high endian, unsigned int (4 bytes)
		#
		# For more information see:
		# https://docs.python.org/3/library/struct.html#byte-order-size-and-alignment

		dataset_image_num, num_col, num_row = struct.unpack('>III', fd.read(12))
		
		image_buf = fd.read(num_image * num_col * num_row)
		
		image_data = np.frombuffer(image_buf, dtype=np.uint8).astype(np.float32)
		image_data = image_data.reshape(num_image, num_col, num_row)
	
	with open(label_file, 'br') as fd:
		magic = fd.read(4)
		zero, zero, data_type, num_dimensions = struct.unpack('>BBBB', magic)

		dataset_image_num = struct.unpack('>I', fd.read(4))

		label_buf = fd.read(num_image)

		label_data = np.frombuffer(label_buf, dtype=np.uint8).astype(np.int32)



	return image_data, label_data

