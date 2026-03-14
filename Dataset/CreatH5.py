import torch.nn as nn
import h5py
from PIL import Image
import os
import h5py
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoTokenizer
import open_clip

class ImageTextToHDF5:
    def __init__(self, image_dir, text_dir, output_h5_path, transform=None, max_len=77):
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.output_h5 = output_h5_path
        self.transform = transform
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32', context_length=max_len)
        self.max_len = max_len

        # 获取所有文件名（不含后缀）
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.base_names = [os.path.splitext(f)[0] for f in image_files]

    def generate(self):
        """遍历所有样本，写入 HDF5 文件"""
        with h5py.File(self.output_h5, 'w') as f:
            # 每个图片5条描述
            total_samples = len(self.base_names) * 5

            # 创建图像数据集（固定形状）
            img_dataset = f.create_dataset(
                'images',
                shape=(total_samples, 3, 224, 224),
                dtype=np.float32,
                chunks=True,
                #compression='gzip'
            )
            # 创建文本 ID 数据集（每个样本固定长度）
            input_ids_dataset = f.create_dataset(
                'input_ids',
                shape=(total_samples, self.max_len),
                dtype=np.int32,
                chunks=True,
                #compression='gzip'
            )
            # 创建 attention mask 数据集
            mask_dataset = f.create_dataset(
                'attention_mask',
                shape=(total_samples, self.max_len),
                dtype=np.int32,
                chunks=True,
                #compression='gzip'
            )

            # 遍历每个基础文件名
            idx = 0
            for base in tqdm(self.base_names):
                # 1. 处理图像
                img_path = os.path.join(self.image_dir, base + '.jpg')  # 假设后缀为 .jpg
                try:
                    image = Image.open(img_path).convert('RGB')
                except FileNotFoundError:
                    # 尝试其他后缀
                    img_path = os.path.join(self.image_dir, base + '.png')
                    image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)  # 得到 [C, H, W] tensor
                image_np = image.numpy()  # 转为 numpy

                # 2. 读取文本文件（5行）
                text_path = os.path.join(self.text_dir, base + '.txt')
                with open(text_path, 'r', encoding='utf-8') as f_txt:
                    lines = [line.strip() for line in f_txt if line.strip()][:5]  # 取前5行非空

                # 3. 对每一行进行编码，写入 H5
                for line in lines:
                    tokens = self.tokenizer(line)                     # [1, max_len]
                    input_ids = tokens[0].numpy().astype(np.int32)     # [max_len]
                    attention_mask = (input_ids != 0).astype(np.int32) # 0 为 pad token id

                    # 写入 H5
                    img_dataset[idx] = image_np
                    input_ids_dataset[idx] = input_ids
                    mask_dataset[idx] = attention_mask
                    idx += 1

            # 确保写入了正确数量的样本
            assert idx == total_samples, f"Expected {total_samples}, wrote {idx}"

if __name__ == "__main__":
    image_dir = './Data/test'
    text_dir = './Data/test'
    output_h5_path = './Data/test.h5'

    if os.path.exists(output_h5_path):
        # 文件已存在，直接读取并打印 keys
        with h5py.File(output_h5_path, 'r') as f:
            print("文件中的数据集:", list(f.keys()))
            print(len(f['images']))
            print(len(f['input_ids']))
            print(f['images'][0].shape)  # 应该是 (3, 224, 224)
            print(f['input_ids'][0].shape)  # 应该是 (max_len,)
            print(f['attention_mask'][0].shape)  # 应该是 (max_len,)
            
        model_name = 'ViT-B-32'
        tokenizer = open_clip.get_tokenizer(model_name)
        with h5py.File(output_h5_path, 'r') as f:
            # 查看第一张图片的五条文本
            for i in range(5):
                input_ids = f['input_ids'][i]
                text = tokenizer.decode(input_ids, skip_special_tokens=True)
                print(f"样本 {i}: {text}")
    else:
        # 文件不存在，生成
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        generator = ImageTextToHDF5(image_dir, text_dir, output_h5_path, transform)
        generator.generate()
        print("HDF5 文件生成完毕！")
