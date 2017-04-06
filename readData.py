import tensorflow as tf
import numpy as np
import os


train_path = '/home/quter/PycharmProjects/qunver/CatVsDog/data/train/'
def get_images(dir):
    cat_image = []
    cat_label = []
    dog_image = []
    dog_label = []
    for line in os.listdir(dir):
        name = line.strip().split('.')
        if name[0] == 'cat':
            cat_image.append(train_path + line)
            cat_label.append(0)
        else:
            dog_image.append(train_path + line)
            dog_label.append(1)
    image_list = np.hstack((cat_image,dog_image))
    label_list = np.hstack((cat_label,dog_label))
    temp = np.array([image_list,label_list]).transpose()
    np.random.shuffle(temp)
    image_list = temp[:,0]
    label_list = temp[:,1]
    label_list = [int(i) for i in label_list]
    return image_list,label_list
    

def get_batch(image,label,width,hight,batch_number,capacity):
    image = tf.cast(image,tf.string)
    label = tf.cast(label,tf.int32)
    input_queue = tf.train.slice_input_producer([image,label])
    label = input_queue[1]
    image_content = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_content,channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image,hight,width)
    image = tf.image.per_image_standardization(image)
    image_batch,label_batch = tf.train.batch([image,label],
                                             batch_size=batch_number,
                                             num_threads=64,
                                             capacity=capacity)
                                             
    label_batch = tf.reshape(label_batch,[batch_number])
    return image_batch,label_batch

# test
'''
import matplotlib.pyplot as plt
size = 5
capacity = 256
width = 200
hight = 200
image,label = get_images(train_path)
image_batch,label_batch = get_batch(image,label,width,hight,size,capacity)
with tf.Session() as sess:
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        while  i<1:
            img,lab = sess.run([image_batch,label_batch])
            for j in range(size):
                print('photo number is %d'%j)
                plt.imshow(img[j,:,:,:])
                plt.show()
            i+=1
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)
'''

