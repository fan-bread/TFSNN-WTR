import numpy as np
from hdf5storage import loadmat
from sklearn.model_selection import train_test_split
from scipy import fft
import torch
import torch.nn.functional as F


#功率归一化（一般是对于IQ信号做，之后再FFT）对一条样本
def power_normalization(x_i, x_q):
    xcomplex = np.zeros([x_i.shape[0], x_i.shape[1]], dtype=complex)
    for i in range(x_i.shape[0]):
        max_power = np.sum((np.power(x_i[i,:],2) + np.power(x_q[i,:],2)))/x_i.shape[1]
        xcomplex[i,:] = (x_i[i,:] + 1j * x_q[i,:]) / np.power(max_power, 1/2)
    return xcomplex

#最大最小值归一化  对整个数据集
def min_max_normalization(x):
    x_min = np.min(x)
    x_max = np.max(x)
    x = (x - x_min)/(x_max - x_min)
    return x

def freq_data(x, fft_point):
    L = x.shape[1]
    x_i = x[:, :L//2]
    x_q = x[:, L//2:]
    xcomplex = power_normalization(x_i, x_q)#对单条样本功率归一化

    # 进行 FFT 转换
    x_fft = fft.fft(xcomplex, n=fft_point, axis=1)
    # x_fft = fft.fft(xcomplex, xcomplex.shape[1], axis=1)

    x_out = np.concatenate((x_fft.real, x_fft.imag), axis=1)
    x_out = x_out.reshape(x_out.shape[0], 2, -1)

    return x_out

def time_data(x):
    L = x.shape[1]
    x_i = x[:, :L//2]
    x_q = x[:, L//2:]
    xcomplex = power_normalization(x_i, x_q)

    x_out = np.concatenate((xcomplex.real, xcomplex.imag), axis=1)
    x_out = x_out.reshape(x_out.shape[0], 2, -1)
    return x_out

def read_data(rand_num, sr, snr, fft_point):
    # file_name = f'D:/multi_rat/dataset/Awgndata_six/dataset/TRdata/TRdata_{sr}msps_{snr} dB.mat'
    file_name = f'../../dataset/TRdata_{sr}msps_{snr} dB.mat'
    data_mat = loadmat(file_name)
    x_data_all = data_mat['x']
    x_data = np.array(x_data_all)

    x_iq = time_data(x_data)
    # x_iq = x_iq.swapaxes(1, 2) #输入LSTM，（bs,440,2)
    x_iq = x_iq[:, :, np.newaxis, :]  # data

    x_fft = freq_data(x_data, fft_point)
    x_fft = x_fft[:, :, np.newaxis, :]

    x_fft = torch.from_numpy(x_fft)
    x_fft = x_fft.float()
    x_fft = F.interpolate(x_fft, size=(1, 440), mode='bilinear', align_corners=False)

    y = data_mat['y']
    y = np.squeeze(y)#target

    # 分别为IQ和FFT数据划分数据集
    x_iq_data_all_, x_iq_test, y_data_all_, y_test = train_test_split(x_iq, y, test_size=0.3, random_state=rand_num)
    x_fft_data_all_, x_fft_test, _, _ = train_test_split(x_fft, y, test_size=0.3, random_state=rand_num)

    x_iq_train, x_iq_val, y_train, y_val = train_test_split(x_iq_data_all_, y_data_all_, test_size=0.3, random_state=rand_num)
    x_fft_train, x_fft_val, _, _ = train_test_split(x_fft_data_all_, y_data_all_, test_size=0.3, random_state=rand_num)

#最大最小值归一化 -> 一般是对整个数据集
    # data_all_ = min_max_normalization(data_all_)
    # x_test = min_max_normalization(x_test)

    # 返回IQ和FFT数据及共享的标签
    return x_iq_train, x_iq_test, x_iq_val, x_fft_train, x_fft_test, x_fft_val, y_train, y_test, y_val
# if __name__ == "__main__":
    # x_iq_train, x_iq_test, x_iq_val, x_fft_train, x_fft_test, x_fft_val, y_train, y_test, y_val = read_data(2023, 10, 0)