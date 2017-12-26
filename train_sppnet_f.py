
# coding: utf-8

# In[6]:


from spp_net_f import *
from spp_layer_f import *
import os
import time
import input_data
import tensorflow as tf
import numpy as np


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_size = 3060
batch_size = 4
max_epochs =2
num_class = 10
eval_frequency = 100
max_steps = 10000




# In[7]:


def train():
    global_step = tf.Variable(0, trainable=False)
    spp_net = SPPnet()
    spp_net.set_lr(0.0001, batch_size, train_size)
    
# load data
    print('load data')
    train_data=mnist.train.images
    train_label=mnist.train.labels
    test_data=mnist.test.images
    test_label=mnist.test.labels
    print("load done")
    num_class = 10

# train
    print('train')
        
    train_data = tf.placeholder("float", shape=[None, 784])
    train_data = tf.reshape(train_data, [-1,28,28,1])
    train_label = tf.placeholder("float", shape=[None, 10])
    
    
    logits = spp_net.inference(train_data, True, num_class)
    loss, accuracy = spp_net.loss(logits, train_label)
    opt, lr = spp_net.train(loss, global_step)
    print('train done')
    
# evaluation
#    eval_logits = spp_net.inference(valid_data, False, num_class)
#    eval_accuracy = spp_net.loss(eval_logits, valid_label)
    with tf.Session() as sess:
        saver = tf.train.Saver(tf.all_variables())
        init = tf.initialize_all_variables()
        coord = tf.train.Coordinator()
        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        
        start_time = time.time()
    #    print((FLAGS.max_epochs * train_size) // batch_size)
        for step in xrange(max_steps):
            batch = mnist.train.next_batch(batch_size)
            opt.eval(feed_dict={train_data:batch[0], train_label: batch[1], keep_prob: 1.0})
            loss_value=loss.eval(feed_dict={train_data:batch[0], train_label: batch[1], keep_prob: 1.0})
            accu = accuracy.eval(feed_dict={train_data:batch[0], train_label: batch[1], keep_prob: 1.0})
            if step % eval_frequency ==0:
                stop_time = time.time() - start_time
                start_time = time.time()
                print('epoch: %.2f , %.2f ms' % (step * batch_size /train_size,
                    1000 * stop_time / eval_frequency)) 
                print('train loss: %.3f' % loss_value) 
                print('train accu: %.2f%%' % accu)         
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    train()



