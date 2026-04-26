import os
import torch
from torch.utils.tensorboard import SummaryWriter
from model import Model

# ==================== 微调超参数 ====================
# 可以调整这些参数来优化微调效果
batch_size = 8  # 微调时的批次大小
context_length = 128  # 上下文长度（与预训练保持一致）
max_iters = 2000  # 微调迭代次数（通常比预训练少）
learning_rate = 5e-4  # 微调学习率（通常比预训练小）
eval_interval = 50  # 评估间隔
eval_iters = 20  # 评估时的迭代次数
device = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)

# TensorBoard 日志
writer = SummaryWriter("logs_finetuning")

# ==================== 准备微调数据 ====================
print("正在加载 webnovel_cn 数据集...")

# 读取新的训练数据（webnovel_cn 数据集）
# 假设你已经下载了数据并保存为 finetuning_data.txt
# 如果使用 ModelScope 数据集，需要先下载并处理
data_file = 'data/finetuning_data.txt'  # 微调数据文件路径

if not os.path.exists(data_file):
    print(f"警告: {data_file} 不存在，请先下载 webnovel_cn 数据集")
    print("可以从 https://modelscope.cn/datasets/AI-ModelScope/webnovel_cn 下载")
    # 这里使用示例数据，实际使用时请替换为真实数据
    with open('sales_textbook.txt', 'r', encoding='utf-8') as f:
        text = f.read()
else:
    with open(data_file, 'r', encoding='utf-8') as file:
        text = file.read()

print(f"数据集大小: {len(text)} 字符")

# 构建词汇表
vocab = sorted(list(set(text)))
vocab_size = len(vocab)
print(f"词汇表大小: {vocab_size}")

char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = {idx: char for char, idx in char2idx.items()}
encode = lambda x: [char2idx[char] for char in x]
decode = lambda idx_list: ''.join([idx2char[idx] for idx in idx_list])
tokenized_text = torch.tensor(encode(text), dtype=torch.long)

# 划分训练集和验证集
train_size = int(len(tokenized_text) * 0.8)
train_data = tokenized_text[:train_size]
val_data = tokenized_text[train_size:]

print(f"训练集大小: {train_size}, 验证集大小: {len(val_data)}")

# ==================== 加载预训练模型 ====================
print("正在加载预训练模型...")

# 注意：需要确保新数据的 vocab_size 与预训练模型一致
# 如果不一致，需要调整模型或重新训练
pretrained_vocab_size = 4055  # 根据之前训练的模型设置

# 如果词汇表大小不同，需要特殊处理
if vocab_size != pretrained_vocab_size:
    print(f"警告: 新数据词汇表大小 ({vocab_size}) 与预训练模型 ({pretrained_vocab_size}) 不一致")
    print("将使用预训练模型的词汇表大小")
    vocab_size = pretrained_vocab_size

# 创建模型并加载预训练权重
model = Model(max_token_value=vocab_size).to(device)

# 加载预训练模型权重
pretrained_model_path = 'model-scifi.pt'
if os.path.exists(pretrained_model_path):
    state_dict = torch.load(pretrained_model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"成功加载预训练模型: {pretrained_model_path}")
else:
    print(f"警告: 未找到预训练模型 {pretrained_model_path}，将从头开始训练")


# ==================== 数据批处理函数 ====================
def get_batch(split: str):
    """获取训练或验证批次"""
    data = train_data if split == 'train' else val_data
    idxs = torch.randint(low=0, high=(len(data) - context_length), size=(batch_size,))
    x = torch.stack([data[idx:idx + context_length] for idx in idxs]).to(device)
    y = torch.stack([data[idx + 1:idx + context_length + 1] for idx in idxs]).to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    """评估模型损失"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ==================== 开始微调 ====================
print("\n开始微调...")
print(f"设备: {device}")
print(f"批次大小: {batch_size}")
print(f"学习率: {learning_rate}")
print(f"迭代次数: {max_iters}\n")

# 创建优化器（可以使用更小的学习率）
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

tracked_losses = list()
for step in range(max_iters):
    # 定期评估
    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss()
        tracked_losses.append(losses)
        print(f'Step: {step:4d} | '
              f'Training Loss: {losses["train"].item():.4f} | '
              f'Validation Loss: {losses["val"].item():.4f}')

        # 记录到 TensorBoard
        writer.add_scalar('Finetuning Training Loss', losses['train'].item(), step)
        writer.add_scalar('Finetuning Validation Loss', losses['val'].item(), step)

    # 获取训练批次
    xb, yb = get_batch('train')

    # 前向传播
    logits, loss = model(xb, yb)

    # 反向传播
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ==================== 保存微调后的模型 ====================
output_model_path = 'model-finetuned.pt'
torch.save(model.state_dict(), output_model_path)
print(f"\n微调完成！模型已保存到: {output_model_path}")

writer.close()
print("TensorBoard 日志已关闭")
