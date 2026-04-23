import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# 1. 模型类
class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Con2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 2. 自动设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3.加载模型
model = torch.load("tudui.pth", map_location=device, weights_only=False)
model = model.to(device)
model.eval()

# 4. 图片处理
image = Image.open("part-00660-2127.jpg")  # 改成你的图片路径
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

image = transform(image).unsqueeze(0).to(device)  # 图片也放到GPU/CPU

# 5. 推理
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1))