import time
import torch
from PIL import ImageOps
from torch import nn
from utils1.data_gen import data_transforms
from nets.models import resnet101


class HParams:
    def __init__(self):
        self.pretrained = False         #False表示直接用预训练模型，不进行重新训练模型
        self.use_se = True


class Insightface(object):

    _defaults = {

        # insight-face预训练模型路径
        "model_path": 'model_data/insight-face-v3.pt',

        # 运行设备类型('cuda' or 'cpu')
        "device_type": 'cpu'            #电脑没有独显，这里我使用cpu
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # 初始化Insightface

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.generate()

    # 载入模型

    def generate(self):
        # 运行设备
        self.device = torch.device(self.device_type)
        start = time.time()
        # 主干网络配置
        self.net = resnet101(HParams())
        # 加载网络
        self.net.load_state_dict(torch.load(self.model_path,map_location=self.device))
        print('{0} model, and classes loaded in {1:.2f}.'.format(
            self.model_path, time.time() - start))
        self.net = nn.DataParallel(self.net)
        self.net = self.net.to(self.device)
        self.net.eval()


    # 将图像转移至模型运行设备

    def send_image_to_device(self, img, transformer=data_transforms['val'], flip=False):
        if flip:
            img = ImageOps.flip(img)
        img = transformer(img)
        return img.to(self.device)

    # 获取单个图像特征

    def get_faces_features_in_single_pic(self, image):
        # 输入图像规模 3x112x112
        feature = None
        with torch.no_grad():
            imgs = torch.zeros(
                [1, 3, 112, 112], dtype=torch.float,  device=self.device)
            imgs[0] = image
            features = self.net(imgs.to(self.device))
            features = features.cpu().numpy()
            feature = features[0]
        return feature


