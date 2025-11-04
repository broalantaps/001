import torch.nn as nn
from qwen3.modeling_qwen3_vl import Qwen3VLModel
from qwen3.configuration_qwen3_vl import Qwen3VLConfig
from qwen3.modular_qwen3_vl import Qwen3VLForConditionalGeneration
from transformers import Trainer, TrainingArguments



class MLLMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Qwen3VLModel(config)