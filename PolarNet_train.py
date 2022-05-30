import tensorflow as tf
from keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU, Concatenate, Flatten, Lambda
from keras.models import Model
from keras.callbacks import TensorBoard, Callback
import scipy.io as sio 
import numpy as np
import math
import time
from keras import backend as K 
tf.reset_default_graph()


envir = 'indoor' #'indoor' or 'outdoor'
# image params
img_height = 32
img_width = 32
img_channels = 2 
img_total = img_height*img_width*img_channels
# network params
residual_num = 2
encoded_dim = 256  #compress rate=1/4->dim.=512, compress rate=1/16->dim.=128, compress rate=1/32->dim.=64, compress rate=1/64->dim.=32

# Bulid the autoencoder model of PolarNet
def polar_network(a, b, encoded_dim):
    def add_common_layers(m):
        m = BatchNormalization()(m)
        m = LeakyReLU()(m)
        return m
    
    # Data IN - Polarization 1
    a = Conv2D(2, (8, 1), padding='same', data_format="channels_first")(a)
    a = add_common_layers(a)
    a = Conv2D(2, (1, 8), padding='same', data_format="channels_first")(a)
    a = add_common_layers(a)
    a = Conv2D(2, (8, 8), padding='same', data_format="channels_first")(a)
    
    # Data IN - Polarization 2
    b = Conv2D(2, (8, 1), padding='same', data_format="channels_first")(b)
    b = add_common_layers(b)
    b = Conv2D(2, (1, 8), padding='same', data_format="channels_first")(b)
    b = add_common_layers(b)
    b = Conv2D(2, (8, 8), padding='same', data_format="channels_first")(b)
    
    # Data concatenation
    c = Concatenate(axis=1)([a, b])
    
    # Chache concatenation point
    d = c
    
    # Encoder
    c = Conv2D(8, (3, 3), padding='same', data_format="channels_first")(c)
    c = add_common_layers(c)
    c = Conv2D(4, (3, 3), padding='same', data_format="channels_first")(c)
    c = add_common_layers(c)
    c = Concatenate(axis=1)([c, d])
    
    # Chache concatenation point
    f = c
    
    c = Conv2D(4, (3, 3), padding='same', data_format="channels_first")(c)
    c = add_common_layers(c)
    c = Concatenate(axis=1)([c, d])
    c = Concatenate(axis=1)([c, f])
    
    c = Flatten()(c)
    
    # Encoded word
    encoded = Dense(encoded_dim)(c)
    encoded = LeakyReLU()(encoded)
    
    # Decoder
    c = Dense(img_total*2)(encoded)
    c = LeakyReLU()(c)
    c = Reshape((img_channels*2, img_height, img_width,))(c)
    
    # Chache concatenation point
    d = c
    
    c = add_common_layers(c)
    c = Conv2D(8, (3, 3), padding='same', data_format="channels_first")(c)
    c = add_common_layers(c)
    c = Concatenate(axis=1)([c, d])
    
    # Chache concatenation point
    f = c
    
    c = Conv2D(4, (3, 3), padding='same', data_format="channels_first")(c)
    c = add_common_layers(c)
    c = Concatenate(axis=1)([c, d])
    c = Concatenate(axis=1)([c, f])
    c = Conv2D(4, (3, 3), padding='same', data_format="channels_first")(c)
    c = add_common_layers(c)
    c = Conv2D(4, (3, 3), padding='same', data_format="channels_first")(c)
    
    # Exits for f - first polarization, s - second polarization
    ext_f = Lambda(lambda c: tf.slice(c, (0,0, 0, 0), (-1,2, 32, 32)))(c)
    ext_s = Lambda(lambda c: tf.slice(c, (0,2, 0, 0), (-1,2, 32, 32)))(c)
    
    return [ext_f, ext_s]
    
image_tensor = Input(shape=(img_channels, img_height, img_width))
image_tensor_s = Input(shape=(img_channels, img_height, img_width))
[netw_out_f, netw_out_s] = polar_network(image_tensor, image_tensor_s, encoded_dim)
autoencoder = Model(inputs=[image_tensor, image_tensor_s], outputs=[netw_out_f, netw_out_s])
autoencoder.compile(optimizer='adam', loss='mse')
print(autoencoder.summary())

# Data loading
if envir == 'indoor':
    mat = sio.loadmat('data/DATA_Htrainin.mat') 
    x_train = mat['HT'] # array
    mat = sio.loadmat('data/DATA_Hvalin.mat')
    x_val = mat['HT'] # array
    mat = sio.loadmat('data/DATA_Htestin.mat')
    x_test = mat['HT'] # array

elif envir == 'outdoor':
    mat = sio.loadmat('data/DATA_Htrainout.mat') 
    x_train = mat['HT'] # array
    mat = sio.loadmat('data/DATA_Hvalout.mat')
    x_val = mat['HT'] # array
    mat = sio.loadmat('data/DATA_Htestout.mat')
    x_test = mat['HT'] # array

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')
x_train = np.reshape(x_train, (len(x_train), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format
x_train_f = x_train[0:50000, :, :, :]
x_train_s = x_train[50000:100000, :, :, :]
x_val = np.reshape(x_val, (len(x_val), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format
x_val_f = x_val[0:15000, :, :, :]
x_val_s = x_val[15000:30000, :, :, :]
x_test = np.reshape(x_test, (len(x_test), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format
x_test_f = x_test[0:10000, :, :, :]
x_test_s = x_test[10000:20000, :, :, :]

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses_train = []
        self.losses_val = []

    def on_batch_end(self, batch, logs={}):
        self.losses_train.append(logs.get('loss'))
        
    def on_epoch_end(self, epoch, logs={}):
        self.losses_val.append(logs.get('val_loss'))

history = LossHistory()
file = 'CsiNet_'+(envir)+'_dim'+str(encoded_dim)+time.strftime('_%m_%d')
path = 'result/TensorBoard_%s' %file

autoencoder.fit([x_train_f, x_train_s], [x_train_f, x_train_s],
                epochs=1000,
                batch_size=200,
                shuffle=True,
                validation_data=([x_val_f, x_val_s], [x_val_f, x_val_s]),
                callbacks=[history,
                           TensorBoard(log_dir = path)])

filename = 'result/trainloss_%s.csv'%file
loss_history = np.array(history.losses_train)
np.savetxt(filename, loss_history, delimiter=",")

filename = 'result/valloss_%s.csv'%file
loss_history = np.array(history.losses_val)
np.savetxt(filename, loss_history, delimiter=",")

#Testing data
tStart = time.time()
[x_hat_f, x_hat_s] = autoencoder.predict([x_test_f, x_test_s])
tEnd = time.time()
print ("It cost %f sec" % ((tEnd - tStart)/x_test_f.shape[0]))

# Calcaulating the NMSE and rho
if envir == 'indoor':
    mat = sio.loadmat('data/DATA_HtestFin_all.mat')
    X_test = mat['HF_all']# array

elif envir == 'outdoor':
    mat = sio.loadmat('data/DATA_HtestFout_all.mat')
    X_test = mat['HF_all']# array

#LAMBDA 1
X_test = np.reshape(X_test, (len(X_test), img_height, 125))
X_test_f = X_test[0:10000, :, :]
x_test_real_f = np.reshape(x_test_f[:, 0, :, :], (len(x_test_f), -1))
x_test_imag_f = np.reshape(x_test_f[:, 1, :, :], (len(x_test_f), -1))
x_test_C_f = x_test_real_f-0.5 + 1j*(x_test_imag_f-0.5)
x_hat_real_f = np.reshape(x_hat_f[:, 0, :, :], (len(x_hat_f), -1))
x_hat_imag_f = np.reshape(x_hat_f[:, 1, :, :], (len(x_hat_f), -1))
x_hat_C_f = x_hat_real_f-0.5 + 1j*(x_hat_imag_f-0.5)
x_hat_F_f = np.reshape(x_hat_C_f, (len(x_hat_C_f), img_height, img_width))
X_hat_f = np.fft.fft(np.concatenate((x_hat_F_f, np.zeros((len(x_hat_C_f), img_height, 257-img_width))), axis=2), axis=2)
X_hat_f = X_hat_f[:, :, 0:125]

#LAMBDA 2
X_test_s = X_test[10000:20000, :, :]
x_test_real_s = np.reshape(x_test_s[:, 0, :, :], (len(x_test_s), -1))
x_test_imag_s = np.reshape(x_test_s[:, 1, :, :], (len(x_test_s), -1))
x_test_C_s = x_test_real_s-0.5 + 1j*(x_test_imag_s-0.5)
x_hat_real_s = np.reshape(x_hat_s[:, 0, :, :], (len(x_hat_s), -1))
x_hat_imag_s = np.reshape(x_hat_s[:, 1, :, :], (len(x_hat_s), -1))
x_hat_C_s = x_hat_real_s-0.5 + 1j*(x_hat_imag_s-0.5)
x_hat_F_s = np.reshape(x_hat_C_s, (len(x_hat_C_s), img_height, img_width))
X_hat_s = np.fft.fft(np.concatenate((x_hat_F_s, np.zeros((len(x_hat_C_s), img_height, 257-img_width))), axis=2), axis=2)
X_hat_s = X_hat_s[:, :, 0:125]

#LAMBDA 1
n1_f = np.sqrt(np.sum(np.conj(X_test_f)*X_test_f, axis=1))
n1_f = n1_f.astype('float64')
n2_f = np.sqrt(np.sum(np.conj(X_hat_f)*X_hat_f, axis=1))
n2_f = n2_f.astype('float64')
aa_f = abs(np.sum(np.conj(X_test_f)*X_hat_f, axis=1))
rho_f = np.mean(aa_f/(n1_f*n2_f), axis=1)
X_hat_f = np.reshape(X_hat_f, (len(X_hat_f), -1))
X_test_f = np.reshape(X_test_f, (len(X_test_f), -1))
power_f = np.sum(abs(x_test_C_f)**2, axis=1)
power_d_f = np.sum(abs(X_hat_f)**2, axis=1)
mse_f = np.sum(abs(x_test_C_f-x_hat_C_f)**2, axis=1)

#LAMBDA 2
n1_s = np.sqrt(np.sum(np.conj(X_test_s)*X_test_s, axis=1))
n1_s = n1_s.astype('float64')
n2_s = np.sqrt(np.sum(np.conj(X_hat_s)*X_hat_s, axis=1))
n2_s = n2_s.astype('float64')
aa_s = abs(np.sum(np.conj(X_test_s)*X_hat_s, axis=1))
rho_s = np.mean(aa_s/(n1_s*n2_s), axis=1)
X_hat_s = np.reshape(X_hat_s, (len(X_hat_s), -1))
X_test_s = np.reshape(X_test_s, (len(X_test_s), -1))
power_s = np.sum(abs(x_test_C_s)**2, axis=1)
power_d_s = np.sum(abs(X_hat_s)**2, axis=1)
mse_s = np.sum(abs(x_test_C_s-x_hat_C_s)**2, axis=1)

print("In "+envir+" environment")
print("When dimension is", encoded_dim)
print("NMSE is ", 10*math.log10(np.mean(((mse_f+mse_s)/2)/((power_f+power_s)/2))))
print("Correlation is ", (np.mean(rho_f)+np.mean(rho_s))/2)
#filename = "result/decoded_%s.csv"%file
#x_hat1 = np.reshape(x_hat, (len(x_hat), -1))
#np.savetxt(filename, x_hat1, delimiter=",")
#filename = "result/rho_%s.csv"%file
#np.savetxt(filename, rho, delimiter=",")

# save
# serialize model to JSON
model_json = autoencoder.to_json()
outfile = "result/model_%s.json"%file
with open(outfile, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
outfile = "result/model_%s.h5"%file
autoencoder.save_weights(outfile)

import matplotlib.pyplot as plt
'''abs'''
n = 10
# LAMBDA 1
plt.figure(figsize=(20, 4))
for i in range(n):
    # display origoutal
    ax = plt.subplot(2, n, i + 1 )
    x_testplo = abs(x_test_f[i, 0, :, :]-0.5 + 1j*(x_test_f[i, 1, :, :]-0.5))
    plt.imshow(np.max(np.max(x_testplo))-x_testplo.T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()
    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    decoded_imgsplo = abs(x_hat_f[i, 0, :, :]-0.5 
                          + 1j*(x_hat_f[i, 1, :, :]-0.5))
    plt.imshow(np.max(np.max(decoded_imgsplo))-decoded_imgsplo.T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()
plt.show()

# LAMBDA 2
plt.figure(figsize=(20, 4))
for i in range(n):
    # display origoutal
    ax = plt.subplot(2, n, i + 1 )
    x_testplo = abs(x_test_s[i, 0, :, :]-0.5 + 1j*(x_test_s[i, 1, :, :]-0.5))
    plt.imshow(np.max(np.max(x_testplo))-x_testplo.T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()
    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    decoded_imgsplo = abs(x_hat_s[i, 0, :, :]-0.5 
                          + 1j*(x_hat_s[i, 1, :, :]-0.5))
    plt.imshow(np.max(np.max(decoded_imgsplo))-decoded_imgsplo.T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()
plt.show()