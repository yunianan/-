#超参数
import math
import os

import requests
import tiktoken
import torch
from torch import nn
import torch.nn.functional as F

batch_size = 4               #批次大小
context_length = 16          #决定模型一次能 “看到” 的文本长度
d_model = 64                 #模型维度
num_blocks = 8               #Transformer 块数量
num_heads = 4                #注意力头数
learning_rate = 1e-3         #学习率
dropout = 0.1                #模型训练时随机让一部分神经元失活的概率
max_iters = 500             #最大迭代次数
eval_interval = 50           #评估间隔：每训练多少步，就执行一次验证
eval_iters = 20              #评估迭代次数：每次评估时，连续跑多少个 batch 来计算验证损失
device = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)

# 获取数据集
if not os.path.exists('sales_textbook.txt'):
    url = 'https://hf-mirror.com/datasets/goendalf666/sales-textbook_for_convincing_and_selling/resolve/main/sales_textbook.txt'
    r = requests.get(url)
    with open('sales_textbook.txt', 'wb') as f:
        f.write(requests.get(url).content)

with open("sales_textbook.txt", 'r') as f:
    text = f.read()

#token化
encoding = tiktoken.get_encoding("cl100k_base")
tokenized_text = encoding.encode(text)
max_token_value = max(tokenized_text)

#切割数据训练和验证集
train_size = int(len(tokenized_text)*0.9)
train_data = tokenized_text[:train_size]
valid_data = tokenized_text[train_size:]

#创建前馈神经网络
class FeeedforwardNetWork(nn.Module):
    def __init__(self, d_model,d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_model*4)
        self.Relu  = nn.ReLU()
        self.linear2 = nn.Linear(d_model*4,d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.Relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


#缩放点积注意力(单头注意力机制)
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wq = nn.Linear(d_model,d_model)
        self.Wk = nn.Linear(d_model,d_model)
        self.Wv = nn.Linear(d_model,d_model)
        #应用mask
        self.register_buffer('mask',torch.tril(torch.ones(context_length,context_length)))

    def forward(self,x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        attention = Q @ K.transpose(-2,-1) / math.sqrt(d_model//num_heads)
        attention = attention.masked_fill(self.mask[:attention.size(-2), :attention.size(-1)]==0, float('-inf'))
        attention = F.softmax(attention,dim=-1)
        output = attention @ V
        return output


#多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([ScaledDotProductAttention() for _ in range(num_heads)])
        self.projection_layer = nn.Linear(d_model*num_heads,d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.projection_layer(out)
        out = self.dropout(out)
        return out

#Transformer块
class TransformerBlock(nn.Module):
    def __init__(self):
        super(TransformerBlock, self).__init__()
        self.multi_head_attention = MultiHeadAttention()
        self.feedforward_network = FeeedforwardNetWork(d_model,d_model*4)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self,x):
        x = x + self.multi_head_attention(self.layer_norm1(x))
        x = x + self.feedforward_network(self.layer_norm2(x))
        return x


#创建模型
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_lookup_table = nn.Embedding(max_token_value+1,d_model)
        self.transformer_blocks = nn.ModuleList([TransformerBlock() for _ in range(num_blocks)])
        self.model_out_linear_layer = nn.Linear(d_model,max_token_value+1)


    def forward(self,idx,targets = None):
        # 加入位置信息
        B,T= idx.shape
        position_encoding_lookup_table = torch.zeros(context_length, d_model,device=device)
        position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
        # apply the sine & cosine
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)
        position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)
        position_embedding = position_encoding_lookup_table[:T,:].to(device)
        x=self.embedding_lookup_table(idx)+position_embedding
        for block in self.transformer_blocks:
            x = block(x)
        #获取最后预测
        logits = self.model_out_linear_layer(x)

        if targets is  not None:
            B,T,C = logits.shape
            logits_reshape = logits.view(B*T,C)
            targets_reshaped = targets.view(B*T)
            #交叉熵损失
            loss = F.cross_entropy(logits_reshape,targets_reshaped)
        else:
            loss = None
        return logits,loss

    #生成方法（预测之后的词）
    def generate(self,idx,max_new_tokens=100):
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -context_length:]
            logits,loss = self.forward(idx_crop)
            logits_last_timestep = logits[:,-1,:]
            probs = F.softmax(logits_last_timestep,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
        return idx


#模型实例化
model = Model().to(device)

#从数据集中抽取一个批次（batch）的训练或验证数据
#模型的任务是：给定位置 t 的 token，预测位置 t+1 的 token。
def get_batch(split: str):
    data = train_data if split == 'train' else valid_data
    idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
    x = torch.stack([torch.tensor(data[idx:idx + context_length]) for idx in idxs]).to(device)
    y = torch.stack([torch.tensor(data[idx+1:idx + context_length + 1]) for idx in idxs]).to(device)
    return x, y


#用于估算模型在训练集和验证集上的平均损失
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()#将模型设置为评估模式
    for split in ['train', 'valid']:
        #初始化损失数组
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            #从当前数据集（训练或验证）中随机抽取一个 batch
            x_batch, y_batch = get_batch(split)
            #将批次数据传入模型
            logits, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()#将模型返回训练模式
    return out



# 创建优化器
#使用 AdamW 优化算法
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
tracked_losses = list()
for step in range(max_iters):
    #每 eval_iters 步（20 步）评估一次
    if step % eval_iters == 0 or step == max_iters - 1:
        losses = estimate_loss()
        tracked_losses.append(losses)
        print('Step:', step, 'Training Loss:', round(losses['train'].item(), 3), 'Validation Loss:',
              round(losses['valid'].item(), 3))

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


#保存
torch.save(model.state_dict(), "model.pth")

#评估模型
model.eval()
start = "mother"
start_ids = encoding.encode(start)
x = torch.tensor([start_ids], dtype=torch.long).to(device)
y = model.generate(x, max_new_tokens=50)#决定最后生成句子长度
print('---------')
print(encoding.decode(y[0].tolist()))
print('---------')