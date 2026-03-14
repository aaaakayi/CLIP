import torch
import open_clip
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torch.optim as optim
from tqdm import tqdm

import torch
import numpy as np

def compute_recall(image_features, text_features, ks=(1, 5, 10)):
    """
    计算图像到文本和文本到图像的 R@K 指标。

    Args:
        image_features: 图像特征，形状为 (N, D)，可以是 torch.Tensor 或 numpy.ndarray。
        text_features: 文本特征，形状为 (N, D)，与图像特征一一对应。
        ks: 需要计算的 K 值列表，如 (1, 5, 10)。

    Returns:
        dict: 包含 i2t 和 t2i 的 R@K 结果，例如：
              {'i2t_R@1': 0.85, 'i2t_R@5': 0.95, 'i2t_R@10': 0.98,
               't2i_R@1': 0.84, 't2i_R@5': 0.94, 't2i_R@10': 0.97,
               'average_R@1': 0.845, ...}
    """
    # 转换为 torch.Tensor（如果需要）
    if not isinstance(image_features, torch.Tensor):
        image_features = torch.from_numpy(image_features)
    if not isinstance(text_features, torch.Tensor):
        text_features = torch.from_numpy(text_features)

    # 确保在相同设备上（CPU/GPU）
    device = image_features.device
    text_features = text_features.to(device)

    # 归一化特征（如果未归一化，则进行 L2 归一化）
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # 计算相似度矩阵 (N, N)
    sim_matrix = image_features @ text_features.T  # 点积

    n = sim_matrix.size(0)
    indices = torch.arange(n, device=device)

    # 图像到文本检索：对于每张图，找到相似度最高的文本索引
    i2t_results = {}
    for k in ks:
        # 获取每行前 k 个最大值的索引
        topk_indices = sim_matrix.topk(k, dim=1).indices  # (N, k)
        # 检查对角线元素（正确文本）是否出现在 top-k 中
        correct = (topk_indices == indices.unsqueeze(1)).any(dim=1)
        i2t_results[f'R@{k}'] = correct.float().mean().item()

    # 文本到图像检索：对于每句文本，找到相似度最高的图像索引
    t2i_results = {}
    for k in ks:
        topk_indices = sim_matrix.topk(k, dim=0).indices  # (k, N)，需要转置
        correct = (topk_indices == indices.unsqueeze(0).T).any(dim=0)
        t2i_results[f'R@{k}'] = correct.float().mean().item()

    # 合并结果
    results = {}
    for k in ks:
        results[f'i2t_R@{k}'] = i2t_results[f'R@{k}']
        results[f't2i_R@{k}'] = t2i_results[f'R@{k}']
        results[f'average_R@{k}'] = (i2t_results[f'R@{k}'] + t2i_results[f'R@{k}']) / 2

    return results

def count_parameters(model):
    """
    统计并打印模型的参数量。
    
    Args:
        model (nn.Module): PyTorch 模型
        
    Returns:
        tuple: (total_params, trainable_params) 单位：百万
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f} M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f} M)")
    
    return total_params / 1e6, trainable_params / 1e6


def count_parameters_per_module(model):
    """打印模型中每个子模块的参数量"""
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"{name}: {params:,} parameters ({params/1e6:.2f} M)")

# -------------------- 配置 --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 数据路径（请根据实际情况修改）
image_dir = './Data/train_images'
text_dir = './Data/train_texts'

# 训练超参数
batch_size = 32          # 可根据显存调整
epochs = 20               # 微调轮次
lr = 1e-5                  # 微调学习率（通常比从头训练小）
weight_decay = 0.001       # 权重衰减
grad_clip = 1.0            # 梯度裁剪
save_path = './finetuned_clip.pth'  # 模型保存路径

# -------------------- 模型与预处理 --------------------
model_name = 'ViT-B-32'
pretrained='./clip_weights/open_clip_pytorch_model.bin'  # 本地文件路径
model, _, preprocess_train = open_clip.create_model_and_transforms(
    model_name,
    pretrained=pretrained,
    device=device
)
tokenizer = open_clip.get_tokenizer(model_name)

# -------------------- 打印整个模型 --------------------
print(model)

# 打印视觉部分
print("\n=== Visual Encoder ===")
print(model.visual)

# 打印文本部分
print("\n=== Text Encoder ===")
print(model.transformer)  # 注意：open_clip 中文本 transformer 直接是 model.transformer
print(model.token_embedding)
print(model.ln_final)

# 打印所有参数的名称和形状（便于自定义模型时核对）
print("\n=== All parameters ===")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")


tokens = tokenizer("a sample text")
print(tokens)  # 应输出一个形状为 [1, 77] 的张量
print(tokens.shape)  # 应输出 torch.Size([1, 77])


# 1. 加载并预处理图片
image_path = './Data/small_test/s0000000.jpg'  # 替换为你的图片路径
image = Image.open(image_path).convert('RGB')
image_input = preprocess_train(image).unsqueeze(0).to(device)

# 2. 候选文本列表（这里用几个示例，实际可以替换为你的全部描述）
candidate_texts = [
    'A basketball player.',                                                                # 准确描述
    'A football player.',                                                                  # 无关
    'A cat sits on a windowsill.',                                                         # 无关
    'A car is parked on the street.',                                                      # 无关
    'A delicious pizza with pepperoni.',                                                   # 无关
    'A person reading a book in a library.',                                               # 无关
    'A beautiful mountain landscape at sunset.'                                            # 无关
]

# 对文本进行 tokenize
text_tokens = tokenizer(candidate_texts).to(device)

# 3. 计算图像和文本的特征
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_tokens)

    # 归一化（模型可能已经归一化，但显式做一下更保险）
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # 计算相似度（点积）
    similarities = (image_features @ text_features.T).squeeze(0)  # 形状 [num_texts]

# 4. 找出最相似的几个描述
top_k = 3
top_indices = similarities.topk(top_k).indices
print(f"Top {top_k} matching descriptions:")
for i, idx in enumerate(top_indices):
    print(f"{i+1}. {candidate_texts[idx]} (score: {similarities[idx].item():.4f})")