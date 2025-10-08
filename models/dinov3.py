# pip install torchao
import os

import torch
from transformers import TorchAoConfig, AutoImageProcessor, AutoModel, AutoConfig, pipeline
#from torchao.quantization import Int4WeightOnlyConfig
from transformers.image_utils import load_image
from threading import Lock

class DINOv3ModelLoader:
    """
    全局单例模型加载器，支持懒加载
    """
    _instance = None
    _model = None
    _processor = None
    _lock = Lock()  # 线程安全

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self, model_path: str = None):
        """
        懒加载模型，只在第一次调用时初始化
        """
        #model_path = "D:\\work\\filedetectDemo\\models\\Qwen3-Embedding-0.6B"
        model_path = "F:\\Model\\dino\\dinov3-vitl16-pretrain-lvd1689m"
        if self._model is None:
            with self._lock:
                if self._model is None:  # double-checked locking
                    model_path = model_path or os.getenv("MODEL_PATH", "/app/models/paraphrase-multilingual-MiniLM-L12-v2")
                    self._model = AutoModel.from_pretrained(
                        model_path,
                        dtype=torch.bfloat16,
                        device_map="auto",

                        # quantization_config=quantization_config
                    )
                    self._processor = AutoImageProcessor.from_pretrained(model_path)

        return self._model

    def forward(self, img_path):

        image = load_image(img_path)
        inputs = self._processor(images=image, return_tensors="pt").to(self._model.device)
        with torch.inference_mode():
            outputs = self._model(**inputs, output_hidden_states=True)
        return outputs

#
# dinov3_model = DINOv3ModelLoader()
# dinov3_model.load_model()
#
#
# url = "F:\\项目数据\\中石化比赛\\以图找图工程图纸智能识别\\初赛发布\\图例\\wall2.png"
# url = "F:\\项目数据\\中石化比赛\\以图找图工程图纸智能识别\\初赛发布\\验证集图片\\dx9.jpg"
# outputs = dinov3_model.forward(img_path=url)
# hidden_states = outputs.hidden_states
# pooled_output = outputs.pooler_output
# last_hidden_state = outputs.last_hidden_state
# print("Pooled output shape:", pooled_output.shape)
# print("last_hidden_state: ",last_hidden_state.shape)
# config = AutoConfig.from_pretrained("F:\\Model\\dino\\dinov3-vitl16-pretrain-lvd1689m")
# print(config)



# 1. 加载模型（使用 torch.hub）
REPO_DIR = "D:\\work\\submission\\dinov3"
WEIGHTS_PATH = "F:\\Model\\dino\\dinov3-vitl16-pretrain-lvd1689m"
url = "F:\\项目数据\\中石化比赛\\以图找图工程图纸智能识别\\初赛发布\\图例\\wall2.png"
#url = "F:\\项目数据\\中石化比赛\\以图找图工程图纸智能识别\\初赛发布\\验证集图片\\dx9.jpg"

REPO_DIR = "D:\\work\\submission\\dinov3"  # 你克隆的官方 repo 路径
WEIGHTS_PATH = "F:\\Model\\dino\\dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"  # 你转换后的 .pth

# 加载模型（不通过 weights=路径，而是手动加载）
model = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights=WEIGHTS_PATH)
state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
model.load_state_dict(state_dict, strict=True)
model.eval().cuda()