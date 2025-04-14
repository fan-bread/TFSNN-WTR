import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


from train_tools_dy_MI import *
from data_loader import *
# from channel_data_20dB import *
# from Spike_AFF import *
from AFF import *


def cm_plot_test(model, test_dataloader):
    model.eval()
    correct = 0
    target_pred = []
    target_real = []
    with torch.no_grad():
        for iq_data, fft_data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                iq_data, fft_data = iq_data.cuda(), fft_data.cuda()
                target = target.cuda()
            output = model(iq_data, fft_data)

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            target_pred.extend(pred.tolist())
            target_real.extend(target.tolist())

    return target_real, target_pred


#hyperparameters
class Config:
    def __init__(self):
        self.seed = 2023  # 用年份设置随机种子
        self.lr = 1e-3
        self.wd = 0
        self.sr = 10  # 采样率
        self.snrs = range(-15, 26, 5)  # 信噪比范围
        self.fft_points = [512]
        self.bs = 64
        self.epoch = 200
        self.model_params = {
            'thresh': 1,
            'tau': 0.5,
            'gamma': 1,
            'dspike': True,
            'soft_reset': False,
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

for fft_point in conf.fft_points:
    print(f"=====================当前 fft 点数是 {fft_point}=====================")
    # 循环信噪比
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
                    #打印模型和损失（损失保存是.mat方便写论文的时候绘制loss曲线
for fft_point in conf.fft_points:
    for snr in conf.snrs:
        x_iq_train, x_iq_test, x_iq_val, x_fft_train, x_fft_test, x_fft_val, y_train, y_test, y_val = read_data(conf.seed, conf.sr, snr, fft_point)
        test_dataset = TensorDataset(torch.Tensor(x_iq_test), x_fft_test, torch.Tensor(y_test))
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
        model = torch.load(f'model_weight/SAFF_fftpoint={fft_point}_seed={conf.seed}_lr={conf.lr}_wd={conf.wd}_sr={conf.sr}_snr={snr}_bs={conf.bs}_epoch={conf.epoch}_dspike={conf.model_params["dspike"]}.pth')
        # model = torch.load('model_weight/seed=2023_lr=0.0001_wd=0_sr=10_snr=30_bs=64_epoch=100_data_type=time_dspike=True')
    # # 参数量
    #     num_params = model.count_parameters()
    #     print(f"The model has {num_params:,} trainable parameters.")

    # acc和推理时间
        test_acc, avg_inference_time = test(model, test_dataloader)
    # 打印测试结果
        print(f'Test results for sampling_rate {conf.sr}msps, fft point {fft_point} and SNR {snr}dB:')
        print(f'- Accuracy: {test_acc * 100:.2f}%')
        # print(f'- Average inference time per sample: {avg_inference_time * 1000:.4f} ms')

        # # Calculate confusion matrix
        # target_real, target_pred = cm_plot_test(model, test_dataloader)
        # cm = confusion_matrix(target_real, target_pred)
        # print("Confusion matrix:")
        # print(cm)
        #
        # # Plot confusion matrix as a heatmap
        # plt.figure(figsize=(8, 6))
        # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(target_real), yticklabels=np.unique(target_real))
        # plt.xlabel('Predict')
        # plt.ylabel('True')
        # # plt.title('Confusion Matrix')
        # # plt.show()
        # plt.savefig(f'Visualization/confusion_matrix_sr={conf.sr}_snr={snr}.png')
        #
        # # Calculate classification report
        # report = classification_report(target_real, target_pred)
        # print("Classification report:")
        # print(report)

