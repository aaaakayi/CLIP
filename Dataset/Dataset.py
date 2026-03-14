from torch.utils.data import DataLoader, Dataset
import h5py
import torch

class ImageTextDataset(Dataset):
    """
    根据 CreatH5.py 中构建的 h5 文件创建数据集
    h5 结构为:
        'images' : 图像数据
        'input_ids' : 图像描述的 encoding，每个图像有五个句子描述
        'attention_mask' : 相关 encoding 的 mask
    """
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.h5_file = None  # 延迟打开，避免 pickle
        self.transform = transform
    def __len__(self):
        # 临时打开文件获取长度，然后关闭
        with h5py.File(self.h5_path, 'r') as f:
            return len(f['images'])

    def __getitem__(self, idx):
        # 每个 worker 进程第一次调用时打开文件，并缓存
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
        image = self.h5_file['images'][idx]
        input_ids = self.h5_file['input_ids'][idx]
        attention_mask = self.h5_file['attention_mask'][idx]

        # 转为 tensor（假设 H5 中存储的是 uint8，值范围 0-255）
        image = torch.from_numpy(image).float() / 255.0  # 归一化到 [0,1]
        input_ids = torch.from_numpy(input_ids).long()
        attention_mask = torch.from_numpy(attention_mask).long()

        if self.transform:
            image = self.transform(image)   # 应用数据增强

        return image, input_ids, attention_mask

if __name__ == "__main__":
    output_h5_path = './Data/h5_3.h5'

    # 创建 Dataset 实例（只传入路径）
    train_dataset = ImageTextDataset(output_h5_path)

    # 创建 DataLoader，设置 num_workers>0
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    # 测试一个 batch
    for images, input_ids, attention_mask in train_loader:
        print(images.shape)          # [batch, 3, 224, 224]
        print(input_ids.shape)       # [batch, seq_len]
        print(attention_mask.shape)  # [batch, seq_len]
        break