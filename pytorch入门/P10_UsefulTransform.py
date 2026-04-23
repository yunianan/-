from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img=Image.open("hymenoptera_data/train/ants/9715481_b3cb4114ff.jpg")
print(img)

#totensor的使用（转化为tensor数据类型）
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("Totensor",img_tensor)
writer.close()

#归一化(可以加快梯度下降速度)
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([1,3,5],[3,2,1])
img_norm = trans_norm(img_tensor)
print(img_tensor[0][0][0])
writer.add_image("Normalize",img_norm,1)

#resize
print(img.size)
trans_resize = transforms.Resize((512,512))#注：这里是两个括号，一个括号会报错
#img是PIL数据类型，经过resize变换还是PIL数据类型
img_resize = trans_resize(img)
#转换为tensor数据类型
img_resize = trans_totensor(img_resize)
writer.add_image("Resize",img_resize,0)

#Compose - resize -2(相当于用compose把resize和totensor结合到一起)
trans_resize_2 = transforms.Resize((512))
trans_compose = transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize",img_resize_2,1)

#RandomCrop
trans_random = transforms.RandomCrop((5)) #这里数值太大可能会报错
trans_compose_2 = transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("Randomcrop",img_crop,i)




writer.close() 