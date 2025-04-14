import numpy as np
from hdf5storage import loadmat
from sklearn.model_selection import train_test_split
from scipy import fft
import torch
import torch.nn.functional as F

def power_normalization(x_i, x_q):
    xcomplex = np.zeros([x_i.shape[0], x_i.shape[1]], dtype=complex)
    for i in range(x_i.shape[0]):
        max_power = np.sum((np.power(x_i[i,:],2) + np.power(x_q[i,:],2)))/x_i.shape[1]
        xcomplex[i,:] = (x_i[i,:] + 1j * x_q[i,:]) / np.power(max_power, 1/2)
    return xcomplex

def time_data(x):
    L = x.shape[1]
    x_i = x[:, :L//2]
    x_q = x[:, L//2:]
    xcomplex = power_normalization(x_i, x_q)

    x_out = np.concatenate((xcomplex.real, xcomplex.imag), axis=1)
    x_out = x_out.reshape(x_out.shape[0], 2, -1)
    return x_out

def freq_data(x, fft_point):
    L = x.shape[1]
    x_i = x[:, :L//2]
    x_q = x[:, L//2:]
    xcomplex = power_normalization(x_i, x_q)
    x_fft = fft.fft(xcomplex, xcomplex.shape[1], axis=1)
    x_out = np.concatenate((x_fft.real, x_fft.imag), axis=1)
    x_out = x_out.reshape(x_out.shape[0], 2, -1)

    return x_out
    
def read_data(rand_num, sr, snr, fft_point):
    file_name = f'../../dataset/TRdata_{sr}msps_{snr} dB.mat'
    
    data_mat = loadmat(file_name)
    x_data_all = data_mat['x']
    x_data = np.array(x_data_all)

    x_iq = time_data(x_data)
    x_iq = x_iq[:, :, np.newaxis, :] 

    x_fft = freq_data(x_data, fft_point)
    x_fft = x_fft[:, :, np.newaxis, :]

    y = data_mat['y']

    x_iq_data_all_, x_iq_test, y_data_all_, y_test = train_test_split(x_iq, y, test_size=0.3, random_state=rand_num)
    x_fft_data_all_, x_fft_test, _, _ = train_test_split(x_fft, y, test_size=0.3, random_state=rand_num)

    x_iq_train, x_iq_val, y_train, y_val = train_test_split(x_iq_data_all_, y_data_all_, test_size=0.3, random_state=rand_num)
    x_fft_train, x_fft_val, _, _ = train_test_split(x_fft_data_all_, y_data_all_, test_size=0.3, random_state=rand_num)
   
    return x_iq_train, x_iq_test, x_iq_val, x_fft_train, x_fft_test, x_fft_val, y_train, y_test, y_val
