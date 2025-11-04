import torch.nn as nn
import torch
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig as PreTrainedConfig
from qwen3.modeling_qwen3_vl import Qwen3VLModel, Qwen3VLForConditionalGeneration
from transformers.activations import ACT2FN

gelu = ACT2FN["gelu"]


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_state):
        input_type = hidden_state.dtype
        hidden_state = hidden_state.to(torch.float32)
        variance = hidden_state.pow(2).mean(-1, keepdim=True)
        hidden_state = hidden_state * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_state
    
class Connector(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int
    ):
        super(Connector, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.RMSNorm = RMSNorm(in_dim)
        
        self.dense_in = nn.Linear(in_dim, out_dim)
        self.dense_out = nn.Linear(out_dim, out_dim)
        
        self.print_trainable_parameters()
        
    def print_trainable_parameters(self):
        trainable_param = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_param += param.numel()
        print(f"Converter trainable parameters: {trainable_param}, All parameters: {all_param}")
        
    def forward(
        self, 
        embeddings: torch.Tensor
    ):
        embeddings = self.RMSNorm(embeddings)
        embeddings = embeddings.to(torch.bfloat16)  
        x = self.dense_in(embeddings)
        x = self.dense_out(gelu(x))
        x = x.to(torch.float32)
        return x


class MLLMModelConfig(PreTrainedConfig):
    """配置类 for MLLMModel"""
    model_type = "mllm_model"
    
    def __init__(
        self,
        compressor_model_name_or_path: str = "Qwen/Qwen3-VL-2B-Instruct",
        decoder_model_name_or_path: str = "Qwen/Qwen3-VL-8B-Instruct",
        compressor_config: dict = None,
        decoder_config: dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.compressor_model_name_or_path = compressor_model_name_or_path
        self.decoder_model_name_or_path = decoder_model_name_or_path
        
        # 保存子模型的配置
        if compressor_config is not None:
            self.compressor_config = compressor_config
        if decoder_config is not None:
            self.decoder_config = decoder_config


class MLLMModel(PreTrainedModel):
    """支持from_pretrained的MLLMModel类"""
    config_class = MLLMModelConfig
    base_model_prefix = "mllm"
    
    def __init__(self, config: MLLMModelConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        
        # 从kwargs中提取用于子模型的参数，排除config等不应该传递的参数
        submodel_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['config', 'state_dict', 'cache_dir', 'force_download']}
        
        # 加载compressor模型
        self.compressor = Qwen3VLModel.from_pretrained(
            config.compressor_model_name_or_path,
            **submodel_kwargs
        )
        
        # 加载decoder模型
        self.decoder = Qwen3VLForConditionalGeneration.from_pretrained(
            config.decoder_model_name_or_path,
            **submodel_kwargs
        )
        
        # 将子模型的配置保存到MLLMModelConfig中
        config.compressor_config = self.compressor.config.to_dict()
        config.decoder_config = self.decoder.config.to_dict()
        
        # 创建converter - 注意：Qwen3VLConfig的hidden_size在text_config中
        # compressor的hidden_size来自text_config
        compressor_hidden_size = self.compressor.config.text_config.hidden_size
        decoder_hidden_size = self.decoder.config.text_config.hidden_size
        
        self.coverter = Connector(
            in_dim=compressor_hidden_size,
            out_dim=decoder_hidden_size
        )
        
        # 初始化权重
        self.post_init()
    
    def _init_weights(self, module):
        """初始化权重 - 子模型已经加载了预训练权重，这里不需要额外初始化"""
        pass


if __name__ == "__main__":
    # 方式1: 使用配置创建模型
    config = MLLMModelConfig(
        compressor_model_name_or_path="Qwen/Qwen3-VL-2B-Instruct",
        decoder_model_name_or_path="Qwen/Qwen3-VL-8B-Instruct",
    )
    model = MLLMModel.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",  # 如果存在配置文件，会从这里加载
        config=config,  # 显式传入配置
    )
    
    # 保存更新后的配置（包含子模型配置）
    model.config.save_pretrained("mllm_model")
    # model.save_pretrained("mllm_model")
    # 方式2: 直接传入配置参数（如果模型路径不存在配置文件）
    # model = MLLMModel.from_pretrained(
    #     "Qwen/Qwen3-VL-2B-Instruct",
    #     compressor_model_name_or_path="Qwen/Qwen3-VL-2B-Instruct",
    #     decoder_model_name_or_path="Qwen/Qwen3-VL-8B-Instruct",
    # )
    
    print(model)
    # 统计参数量
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    print(f"Total parameters: {total_params}")