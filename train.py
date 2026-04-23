import torchvision
from torch.utils.tensorboard import SummaryWriter

from train_model import*
from torch import nn
from torch.utils.data import DataLoader
import time

#准备数据集
train_data = torchvision.datasets.CIFAR10(root="data",train=True,transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="data",train=False,transform=torchvision.transforms.ToTensor(),
                                          download=True)
#数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度为：{train_data_size}")
print(f"测试数据集的长度为：{test_data_size}")

#利用dataloader加载数据集
train_dataloader=DataLoader(train_data,batch_size=64)
test_dataloader=DataLoader(test_data,batch_size=64)

#创建网络模型
tudui=Tudui()
tudui=tudui.cuda()

#损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.cuda()

#优化器
learning_rate=0.01
optimizer = torch.optim.SGD(tudui.parameters(),lr=learning_rate)

#设置训练网络参数
total_train_step=0#记录训练次数
total_test_step=0#记录测试次数
epoch=10#训练轮数

#添加tensorboard
writer=SummaryWriter("logs_train")
start_time=time.time()

for i in range(epoch):
    print(f"------第{i+1}轮训练开始------")
    #训练步骤开始
    tudui.train()
    for data in train_dataloader:
        imgs,targets=data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = tudui(imgs)
        loss = loss_fn(outputs,targets)
        #优化器调优
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step=total_train_step+1
        if total_train_step%100 == 0:
            end_time = time.time()
            print(f"训练时间：{end_time-start_time}")
            print(f"训练次数{total_train_step}，{loss.item()}")
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    #测试步骤开始
    tudui.eval()
    total_test_loss=0
    total_accuracy=0#正确率
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = tudui(imgs)
            loss = loss_fn(outputs,targets)
            total_test_loss=total_test_loss+loss.item()
            accuracy=(outputs.argmax(1)==targets).sum()
            total_accuracy=total_accuracy+accuracy
    print(f"整体测试集loss为：{total_test_loss}")
    print(f"整体测试机正确率:{total_accuracy/test_data_size}")
    writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)
    writer.add_scalar("test_loss",  total_test_loss,total_test_step)
    total_test_step=total_test_step+1

    torch.save(tudui, "tudui.pth")
    print("模型已保存")

writer.close()
