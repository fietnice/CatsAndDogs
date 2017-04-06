# -*- coding: utf-8 -*-
import tensorflow as tf
import readData
import model

batch_size = 20
number_classes = 2
learning_rate = 0.001
capacity = 256
width = 200
hight = 200
max_step = 5000
train_path = train_path = '/home/quter/PycharmProjects/qunver/CatVsDog/data/train/'
image,label = readData.get_images(train_path)
image_batch,label_batch = readData.get_batch(image,label,width,hight,batch_size,capacity)
train_logits = model.intference(image_batch,batch_size,number_classes)
train_loss = model.loss(train_logits,label_batch)
train_op = model.train(train_loss,learning_rate)
train_accuracy = model.evaluation(train_logits,label_batch)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)
try:
    for j in range(max_step):
        if coord.should_stop():
            break
        _,loss,accuracy = sess.run([train_op,train_loss,train_accuracy])
        if j%50==0:
            print('train step :%d,loss is :%.2f,accuracy is :%.2f'%(j,loss,accuracy))
except tf.errors.OutOfRangeError:
    print('done!')
finally:
    coord.request_stop()
coord.join(threads)
sess.close()

