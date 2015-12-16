# coding=utf-8
import data
import cnn_init
import numpy
import tensorflow as tf

train_set,eval_set = data.packets_data()

train_set_num = len(train_set.payloads)
class_num = len(train_set.labels[0])
batch_size = 50  #define the size of one batch

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

def conv2d(x,W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='VALID')

def max_pooling_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

x = tf.placeholder("float", shape = [None,28,28,1])
y_ = tf.placeholder("float", shape = [None,class_num])

# 1st conv & pooling
W_conv1 = weight_variable([5,5,1,6])
b_conv1 = bias_variable([6])

h_conv1 = tf.nn.relu(conv2d(x,W_conv1) + b_conv1)
h_pool1 = max_pooling_2x2(h_conv1)

# 2nd conv & pooling
W_conv2 = weight_variable([5,5,6,12])
b_conv2 = bias_variable([12])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pooling_2x2(h_conv2)

# feature vector layer
fv_neuron_num = 100
W_fc1 = weight_variable([4*4*12, fv_neuron_num])
b_fc1 = bias_variable([fv_neuron_num])

h_pool2_flat = tf.reshape(h_pool2, [-1, 4*4*12]) #transformation into a row vector
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout -- to reduce overfitting??????
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#softmax layer
W_fc2 = weight_variable([fv_neuron_num, class_num])
b_fc2 = bias_variable([class_num])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
	for i in xrange(train_set_num/batch_size*20):
		batch_x, batch_y = train_set.next_batch(batch_size) # batch_x: batch_size*28*28*1
		#print "*********************************************"
		#print sess.run(tf.shape(h_conv1),feed_dict={x:batch_x})
		#print sess.run(tf.shape(h_pool1),feed_dict={x:batch_x})
		#print sess.run(tf.shape(h_pool2),feed_dict={x:batch_x})
		sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
		if i%50 == 0:
			print ("*************** train_step:"i" ***************")
			print sess.run(accuracy,feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
		
	print sess.run(accuracy, feed_dict={x: eval_set.payloads, y_: eval_set.labels, keep_prob: 1.0})

if __name__ == '__main__':
	#train_set,eval_set = data.packets_data()
	#batch_x, batch_y = train_set.next_batch(50)
	#batch_x = numpy.reshape(batch_x,[numpy.shape(batch_x)[0],28,28])

	layers = []
	layers.append(cnn_init.Layer('i',0,1))
	layers.append(cnn_init.Layer('c',5,6))
	layers.append(cnn_init.Layer('s',2,6))
	layers.append(cnn_init.Layer('c',5,12))
	layers.append(cnn_init.Layer('s',2,12))
	#net = cnn_init.cnn(layers,batch_x,batch_y)
	
