from dataclasses import dataclass, field


@dataclass
class MambaConfig:

    d_model: int = 4
    d_intermediate: int = 0
    n_layer: int = 64
    vocab_size: int = 4  # 这里决定输出的序列长度，torch.size([16, 512, 1])对输出效果有影响？
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 1 #调节输入张量的形状
    tie_embeddings: bool = True
