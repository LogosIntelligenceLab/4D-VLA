import torch
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class SimpleFeatureExtractor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.resnet50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet50 = torch.nn.Sequential(*list(self.resnet50.children())[:-1])
        self.resnet50.eval()


    def extract_image_feature(self, image_batch):
        """ 提取单张图像的特征向量 """
        if isinstance(image_batch, np.ndarray):
            image_batch = torch.tensor(image_batch, dtype=torch.float32)

        # 2. 变换数据，调整通道顺序 (B, H, W, C) -> (B, C, H, W)
        image_batch = image_batch.permute(0, 3, 1, 2) / 255.0  # 归一化到 [0,1] 范围

        # 3. 进行 transform 处理
        image_batch = self.transform(image_batch).to(self.resnet50[0].weight.device)

        # 4. 提取特征
        with torch.no_grad():
            feature = self.resnet50(image_batch)  # (B, 2048, 1, 1)

        # 5. 变成 (B, 2048)
        feature = feature.squeeze(-1).squeeze(-1)

        # 6. 归一化
        feature = feature / torch.norm(feature, dim=1, keepdim=True)

        return feature.numpy()  # 返回 NumPy 数组
    


class SimpleMemoryBank:
    def __init__(self, max_size=5, window_size=20, max_interval=4, min_interval=2, similarity_threshold=0.9):
        self.max_size = max_size  # 窗口内最多存的帧数
        self.window_size = window_size  # 时间窗口长度
        self.similarity_threshold = similarity_threshold
        self.max_interval = max_interval  # 最大帧间隔
        self.min_interval = min_interval  # 最小帧间隔
        self.memory = []  # 存储特征
        self.memory_fid = []  # 存储帧号
        self.feature_extractor = SimpleFeatureExtractor()

    def add(self, image, fid, from_img=True):
        """ 
        添加新特征，保持时序 & 仅对比最后一帧 
        - 保证时间窗口内最多存 `max_size=5` 帧
        - `fid` 超出 `window_size=20` 时，删除最旧帧
        """
        if from_img:
            new_feature = self.feature_extractor.extract_image_feature(image)
        else:
            new_feature = image
        new_feature = np.array(new_feature).astype(np.float32)

        # **清理超出时间窗口的帧**
        while len(self.memory_fid) > 0 and (fid - self.memory_fid[0] > self.window_size):
            self.memory.pop(0)
            self.memory_fid.pop(0)

        if len(self.memory) == 0:
            # **Memory Bank 为空，直接添加**
            self.memory.append(new_feature)
            self.memory_fid.append(fid)
            return
        
        # **计算新帧和最后一帧的余弦相似度**
        last_feature = self.memory[-1]
        cos_sim = np.dot(last_feature, new_feature) / (np.linalg.norm(last_feature) * np.linalg.norm(new_feature))

        if cos_sim > self.similarity_threshold:
            if (len(self.memory) < self.max_size):  # and ((fid - self.memory_fid[-1]) > self.min_internval):
                self.memory.append(new_feature)
                self.memory_fid.append(fid)
            # elif (fid - self.memory_fid[-1]) > self.max_interval:
            #     self.memory.append(new_feature)
            #     self.memory_fid.append(fid)
            #     self.memory.pop(0)
            #     self.memory_fid.pop(0)
            else:
                # self.memory[-1] = new_feature
                # self.memory_fid[-1] = fid
                pass
        else:
            # **相似度低 -> 追加到末尾**
            if ((fid - self.memory_fid[-1]) >= self.min_interval):
                if (len(self.memory) >= self.max_size):
                    self.memory.pop(0)  # 删除最旧帧
                    self.memory_fid.pop(0)
                self.memory.append(new_feature)
                self.memory_fid.append(fid)

    def get_memory(self):
        """ 返回 Memory Bank 当前存储的所有特征 """
        return np.array(self.memory)
    
    def get_length(self):
        return len(self.memory)
    
    def get_fid(self):
        return self.memory_fid
    
    def clear(self):
        self.memory = []
        self.memory_fid = []