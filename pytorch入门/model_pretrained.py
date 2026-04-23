import torch
import torchvision
from torch import nn

# train_data = torchvision.datasets.ImageNet("data_ImageNet",split="train",download=True,
#                                            transform=torchvision.transforms.ToTensor())

vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16_True = torchvision.models.vgg16(pretrained=True)
print(vgg16_True)

train_data = torchvision.datasets.CIFAR10("dataset",train=True,download=True,
                                         transform=torchvision.transforms.ToTensor())
#对现有模型进行修改
vgg16_True.classifier.add_module("add_linear",nn.Linear(1000,10))
print(vgg16_True)

#模型保存方式1(模型结构+参数)
torch.save(vgg16,"vgg16_method1.pth")
#模型加载方式1
model1 = torch.load("vgg16_method1.pth")
print(model1)

#保存方式2(模型参数)(官方推荐)
torch.save(vgg16.state_dict(),"vgg16_mothed2.pth")
#加载方式2
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)