import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


from train_tools import *
from data_loader import *
from AFF import *

#hyperparameters
class Config:
    def __init__(self):
        self.seed = 2023 
        self.lr = 1e-3
        self.wd = 0
        self.sr = 10  
        self.snrs = range(-15, 26, 5) 
        self.bs = 64
        self.epoch = 200
        self.model_params = {
            'thresh': 1,
            'tau': 0.5,
            'gamma': 1,
            'dspike': True,
            'soft_reset': True,
        }


conf = Config()

setup_seed(conf.seed)

# model
model = SAFF(**conf.model_params)
model.cuda()

#optimizer
optim = torch.optim.Adam(model.parameters(), lr=conf.lr, weight_decay = conf.wd)

#loss
loss = nn.NLLLoss().cuda()


for snr in conf.snrs:
    print(f"The current SNR is {snr}dB")
    model = SAFF()
    model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=conf.lr, weight_decay=conf.wd)
    loss = nn.NLLLoss().cuda()
    scheduler = ReduceLROnPlateau(optim, 'min', factor=0.1, patience=30, min_lr=1e-6)

    x_iq_train, x_iq_test, x_iq_val, x_fft_train, x_fft_test, x_fft_val, y_train, y_test, y_val = read_data(conf.seed, conf.sr, snr, fft_point)
    train_dataset = TensorDataset(torch.Tensor(x_iq_train), x_fft_train, torch.Tensor(y_train))
    train_dataloader = DataLoader(train_dataset, batch_size=conf.bs, shuffle=True)
    val_dataset = TensorDataset(torch.Tensor(x_iq_val), x_fft_val, torch.Tensor(y_val))
    val_dataloader = DataLoader(val_dataset, batch_size=conf.bs, shuffle=True)
    print(f"The current samplingrate is {conf.sr}msps")
    train_and_evaluate(model,
                loss_function=loss,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                optimizer=optim,
                epochs=conf.epoch,
                scheduler=scheduler,
                save_path=f'model_weight/SAFF_fftpoint={fft_point}_seed={conf.seed}_lr={conf.lr}_wd={conf.wd}_sr={conf.sr}_snr={snr}_bs={conf.bs}_epoch={conf.epoch}_dspike={conf.model_params["dspike"]}.pth',
                loss_path=f'loss_acc/SAFF_fftpoint={fft_point}_fft=128_seed={conf.seed}_lr={conf.lr}_wd={conf.wd}_sr={conf.sr}_snr={snr}_bs={conf.bs}_epoch={conf.epoch}_dspike={conf.model_params["dspike"]}.mat'
                        )

for snr in conf.snrs:
    x_iq_train, x_iq_test, x_iq_val, x_fft_train, x_fft_test, x_fft_val, y_train, y_test, y_val = read_data(conf.seed, conf.sr, snr, fft_point)
    test_dataset = TensorDataset(torch.Tensor(x_iq_test), x_fft_test, torch.Tensor(y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    model = torch.load(f'model_weight/SAFF_fftpoint={fft_point}_seed={conf.seed}_lr={conf.lr}_wd={conf.wd}_sr={conf.sr}_snr={snr}_bs={conf.bs}_epoch={conf.epoch}_dspike={conf.model_params["dspike"]}.pth')
   
    # 打印测试结果
        print(f'Test results for sampling_rate {conf.sr}msps and SNR {snr}dB:')
        print(f'- Accuracy: {test_acc * 100:.2f}%')
