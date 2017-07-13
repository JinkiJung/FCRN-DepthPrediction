import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import time

import models

def predict(model_data_path, image_path):

    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1
    
    # Read image
    img = Image.open(image_path)
    img = img.resize([width,height], Image.ANTIALIAS)
    img.show()
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis = 0)


    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))
    
    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size)

    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')
        start = time.time()
        net.load(model_data_path, sess)      
        print('Finish loading - it took', time.time() - start, "seconds.")
        uninitialized_vars = []
        for var in tf.global_variables():
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:               
                uninitialized_vars.append(var)

        init_new_vars_op = tf.variables_initializer(uninitialized_vars)
        sess.run(init_new_vars_op)
        
        # Evalute the network for the given image
        start = time.time()
        pred = sess.run(net.get_output(), feed_dict={input_node: img})
        print('Finish prediction - it took', time.time() - start, "seconds.")

        # Plot result
        fig = plt.figure()
        ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
        fig.colorbar(ii)
        plt.show()

        return pred
        
                
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('/Users/jinkijung/Documents/Tensorflow/Test/Model/NYU_ResNet-UpProj.npy', help='Converted parameters for the model')
    parser.add_argument('/Users/jinkijung/Documents/Tensorflow/Test/Images', help='Directory of images to predict')
    #args = parser.parse_args()


    # Predict the image

    #print args.model_path, args.image_paths
    #pred = predict(args.model_path, args.image_paths)
    pred = predict('/Users/jinkijung/Documents/Tensorflow/Test/Model/NYU_ResNet-UpProj.npy','/Users/jinkijung/Documents/Tensorflow/Test/Images/20170616_080017.jpg')
    
    #os._exit(0)

if __name__ == '__main__':
    main()

        



