import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
from datetime import datetime

# Helper code for plotting and reading Burgers' input files
def read_me(*args):
    return np.hstack([np.loadtxt(x) for x in args])

def plot_it(v, i, flag = None, v_indices = np.array([0]), title = None, pltaxis = None):
    fig = plt.figure(figsize=(6,6)) #12,6
    plt.title(title,size=14)
    if np.sum(v_indices) == 0:
         vspace = np.linspace(0.0, 100.0, 1000)[:, None]
         plt.plot(vspace, v, lw=2)
    else:
         vspace = v_indices
         plt.plot(vspace, v, lw=2)
    if pltaxis != None:
         plt.axis(pltaxis)
    if flag == 'save':
        my_path = 'C:\\Users\\Tina\\Desktop\\FRG\\NNs\\Project\\figs\\'
        plt.savefig(my_path + datetime.now().strftime('%H.%M') + '_' + str(i) + '_test.png')  
    else:
        plt.show()

# Code here for importing data from file
#snaps =  read_me('burg/snaps_0p02_0p02_5.dat',
#                 'burg/snaps_0p02_0p02_1.dat',
#                 'burg/snaps_0p02_0p02_2p5.dat').T
				 
snaps =  read_me('burg/snaps_0p02_0p02_5.dat').T
#snaps = snaps[10:40,::5]
mu = np.array((5,1,2.5))
mu = np.array((5))
#(n_samp, n_x), n_mu = snaps.shape, mu.shape[0]
(n_samp, n_x), n_mu = snaps.shape, 1
n_t = int(n_samp/n_mu)

t = np.linspace(0.0, 500.0, n_t)
x = np.linspace(0.0, 100.0, n_x)
mu_vec = np.repeat(mu,n_t*n_x)
t_vec = np.tile(np.repeat(t,n_x),n_mu)
x_vec = np.tile(x,n_t*n_mu)
y = snaps.flatten().reshape((n_x*n_t*n_mu,1))
#input = np.vstack((x_vec,t_vec,mu_vec)).T
input = np.vstack((x_vec,t_vec)).T
#plot_it(np.hstack((snaps[:, [150]], snaps[:, [50]])),1)

# Make inputs noisy
#noisy_input = input + .2 * np.random.random_sample((input.shape)) - .1
noisy_input = input
output = y
# Scale to [0,1]
scaled_input_1 = np.divide((noisy_input-noisy_input.min()), (noisy_input.max()-noisy_input.min()))
scaled_output_1 = np.divide((output-output.min()), (output.max()-output.min()))
# Scale to [-1,1]
scaled_input_2 = (scaled_input_1*2)-1
scaled_output_2 = (scaled_output_1*2)-1
input_data = scaled_input_2
output_data = scaled_output_2

input_data = input
#output_data = y

# Build neural network with 3 hidden layers
n_samp, n_input = input_data.shape 
n_hidden = [10,10]
x_input = tf.placeholder("float", [None, n_input])
# Weights and biases from input to hidden layer 1
Wh1 = tf.Variable(tf.random_uniform((n_input, n_hidden[0]), -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
bh1 = tf.Variable(tf.zeros([n_hidden[0]]))
h1 = tf.nn.tanh(tf.matmul(x_input,Wh1) + bh1)
# Weights and biases from hidden layer 1 to hidden layer 2
Wh2 = tf.Variable(tf.random_uniform((n_hidden[0], n_hidden[1]), -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
bh2 = tf.Variable(tf.zeros([n_hidden[1]]))
h2 = tf.nn.tanh(tf.matmul(h1,Wh2) + bh2)
# Weights and biases from hidden layer 3 to output
#Wo = tf.transpose(Wh) # tied weights
Wo = tf.Variable(tf.random_uniform((n_hidden[1], 1), -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
bo = tf.Variable(tf.zeros([1]))
y = tf.matmul(h2,Wo) + bo
# Objective functions
y_ = tf.placeholder("float", [None,1])
generation_loss = tf.reduce_sum(tf.square(y_-y))
loss = tf.reduce_mean(generation_loss)
train_step = tf.train.AdamOptimizer().minimize(loss)

# Run autoencoder
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
n_rounds = 2000
batch_size = 1000
for i in range(n_rounds):
    sample = np.random.randint(n_samp, size=batch_size)
    batch_xs = input_data[sample][:]
    batch_ys = output_data[sample][:]
    sess.run(train_step, feed_dict={x_input: batch_xs, y_:batch_ys})
    if i % 100 == 0:
        #print(batch_xs.shape)
        print(i, sess.run(loss, feed_dict={x_input: batch_xs, y_:batch_ys})/batch_size)

print("Target:")
print(output_data)
print("Final activations:")
print(sess.run(y, feed_dict={x_input: input_data}))
print("Final weights (input => hidden layer)")
print(sess.run(Wh1))
print("Final biases (input => hidden layer)")
print(sess.run(bh1))
print("Final biases (hidden layer => output)")
print(sess.run(bo))
print("Final activations of hidden layer")
print(sess.run(h1, feed_dict={x_input: input_data}))

predictions = sess.run(y, feed_dict={x_input: input_data}).reshape((n_t*n_mu,n_x))
comparisons = output_data.reshape((n_t*n_mu,n_x))

numsnaps = snaps.shape[0]*.65
numplots = 10
for i in range(numplots): 
    x1 = predictions[[i*int(numsnaps/numplots)],:]
    x2 = comparisons[[i*int(numsnaps/numplots)],:]
    #plot_it(np.hstack((x2.T,x1.T)),int(numsnaps/numplots), v_indices = np.linspace(0.0, 100.0, 1000)[::5])
    plot_it(np.hstack((x2.T,x1.T)),int(numsnaps/numplots), v_indices = np.linspace(0.0, 100.0, 1000)[::1])
    
#sample = np.random.randint(n_samp, size=6)
#sess.run(generation_loss, feed_dict={x_input: input_data[sample], y_: output_data[sample]})
