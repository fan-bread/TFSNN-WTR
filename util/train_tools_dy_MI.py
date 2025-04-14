import time
import torch
import torch.nn.functional as F
import numpy as np
import random
from scipy.io import savemat

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def train(model, loss, train_dataloader, optimizer, epoch):
    model.train()
    correct = 0
    all_loss = 0
    for data_nn in train_dataloader:
        iq_data, fft_data, target = data_nn
        target = target.long()
        if torch.cuda.is_available():
            iq_data, fft_data = iq_data.cuda(), fft_data.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        output = model(iq_data, fft_data)
        output = F.log_softmax(output, dim=1)
        result_loss = loss(output, target)
        result_loss.backward()

        optimizer.step()
        all_loss += result_loss.item()*iq_data.size()[0]#data.size()[0]批次大小

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    print('Train Epoch: {} \tLoss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
        epoch,
        all_loss / len(train_dataloader.dataset),
        correct,
        len(train_dataloader.dataset),
        100.0 * correct / len(train_dataloader.dataset))
    )
    return all_loss / len(train_dataloader.dataset), 100.0 * correct / len(train_dataloader.dataset)

def evaluate(model, loss, val_dataloader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for iq_data, fft_data, target in val_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                iq_data, fft_data = iq_data.cuda(), fft_data.cuda()
                target = target.cuda()
            output = model(iq_data, fft_data)
            output = F.log_softmax(output, dim=1)
            test_loss += loss(output, target).item()*iq_data.size()[0]

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_dataloader.dataset)
    fmt = '\nValidation set: Loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            test_loss,
            correct,
            len(val_dataloader.dataset),
            100.0 * correct / len(val_dataloader.dataset),
        )
    )

    return test_loss, 100.0 * correct / len(val_dataloader.dataset)

def test(model, test_dataloader):
    model.eval()
    correct = 0
    total_inference_time = 0.0  # 记录总推理时间
    test_samples = len(test_dataloader.dataset)  # 测试集总样本数
    with torch.no_grad():
        for iq_data, fft_data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                iq_data, fft_data = iq_data.cuda(), fft_data.cuda()
                target = target.cuda()

            start_time = time.time()  # 开始计时

            output = model(iq_data, fft_data)

            end_time = time.time()
            # 累加推理时间
            total_inference_time += (end_time - start_time)

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    # 计算准确率
    accuracy = correct / test_samples

    # 计算平均推理时间（每条样本的推理时间）
    avg_inference_time = total_inference_time / test_samples

    return accuracy, avg_inference_time


def train_and_evaluate(model, loss_function, train_dataloader, val_dataloader, optimizer, epochs, save_path, loss_path, scheduler):
    current_min_test_loss = 100
    loss_acc = np.zeros([1, 4])
    epochs_no_improve = 0  # 用于追踪没有改善的epoch数量
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, loss_function, train_dataloader, optimizer, epoch)
        test_loss, test_acc = evaluate(model, loss_function, val_dataloader, epoch)
        # 更新学习率调度器
        scheduler.step(test_loss)
        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: Current learning rate = {current_lr}")

        if test_loss < current_min_test_loss:
            epochs_no_improve = 0  # 重置计数器
            print("The validation loss is improved from {} to {}, new model weight is saved.".format(
                current_min_test_loss, test_loss))
            current_min_test_loss = test_loss
            torch.save(model, save_path)
        else:
            epochs_no_improve += 1  # 增加没有改善的epoch计数
            print("The validation loss is not improved.")
        print("------------------------------------------------")
        loss_acc_each_epoch = np.reshape([train_loss, train_acc, test_loss, test_acc], [1, 4])
        loss_acc = np.concatenate((loss_acc, loss_acc_each_epoch), axis=0)
    savemat(loss_path, {'loss_acc': loss_acc})
    total_time = time.time() - start_time
    print(f"Total training time: {total_time}")