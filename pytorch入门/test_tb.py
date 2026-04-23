from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
writer = SummaryWriter("logs")
image_path="hymenoptera_data/val/bees/72100438_73de9f17af.jpg"
img_PIL=Image.open(image_path)
img_array=np.array(img_PIL)

writer.add_image("train",img_array,2,dataformats="HWC")
#y=x
for i in range(100):
    writer.add_scalar("y=2x",2*i,i)

writer.close()