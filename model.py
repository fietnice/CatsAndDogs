# -*- coding: utf-8 -*-
import tensorflow as tf

def intference(image,batch_size,n_classes):
    '''Build the model
    '''
    #conv1
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,3,32],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        bias = tf.get_variable('bias',shape=[32],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0.1))
        conv=tf.nn.conv2d(image,weights,strides=[1,1,1,1],padding='SAME')
        pre=tf.nn.bias_add(conv,bias)
        conv1=tf.nn.relu(pre,name=scope.name)
    with tf.variable_scope('pool1_norm1') as scope:
        pool1=tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool1')
        norm1=tf.nn.lrn(pool1,4,alpha=0.001/9.0,beta=0.75,name='norm1')
    
    #conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[5,5,32,64],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        bias = tf.get_variable('bias',shape=[64],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0.1))
        conv=tf.nn.conv2d(norm1,weights,strides=[1,1,1,1],padding='SAME')
        pre=tf.nn.bias_add(conv,bias)
        conv2=tf.nn.relu(pre,name=scope.name)
    with tf.variable_scope('pool2_norm2') as scope:
        pool2=tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool2')
        norm2=tf.nn.lrn(pool2,4,alpha=0.001/9.0,beta=0.75,name='norm2')
    
    #fc1
    with tf.variable_scope('fc1') as scope:
        reshape = tf.reshape(norm2,[batch_size,-1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',shape=[dim,128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        bias = tf.get_variable('bias',shape=[128],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu((tf.matmul(reshape,weights) + bias),name = scope.name)
        
    #fc2
    with tf.variable_scope('fc2') as scope:
        weights = tf.get_variable('weights',shape=[128,128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        bias = tf.get_variable('bias',shape=[128],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu((tf.matmul(fc1,weights) + bias),name = scope.name)
    
    #linear layer
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('weights',shape=[128,n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        bias = tf.get_variable('bias',shape=[n_classes],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0.1))
        linear_layer = tf.add(tf.matmul(fc2,weights),bias,name=scope.name)
    
    return linear_layer
    
def loss(logits,labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                       logits=logits,
                                                                       name='cross_entropy')
        loss = tf.reduce_mean(cross_entropy,name=scope.name)
    return loss

def train(loss,learning_rate):
    with tf.variable_scope('optimizer'):
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op=optimizer.minimize(loss)
    return train_op
    
def evaluation(logits,labels):
    correct = tf.nn.in_top_k(logits,labels,1)
    correct = tf.cast(correct,tf.float32)
    accuracy = tf.reduce_mean(correct)
    return accuracy