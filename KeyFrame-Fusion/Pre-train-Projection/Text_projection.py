import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel
from g2p_en import G2p

# ═══════════ 音素词表定义 ═══════════
CMU_PHONEMES = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY',
    'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY',
    'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW', 'OY', 'P',
    'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW',
    'V', 'W', 'Y', 'Z', 'ZH'
]
PHONE_TO_ID = {p: i + 1 for i, p in enumerate(sorted(CMU_PHONEMES))}
PHONE_VOCAB_SIZE = len(PHONE_TO_ID) + 1  # 40 (含有 CTC Blank / Padding)


class SinusoidalPositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, dim: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, :x.size(1), :]


class EnrollmentTextPipelineModule(nn.Module):
    """
    符合 KFC-KWS 论文标准的 Enrollment Text 完整模块
    输入：自然语言文本列表（如 ["open the door", "turn on the light"]）
    输出：
        - E_t_s: [B, T_text, D]  语义文本特征
        - E_p_s: [B, T_phone, D] 声学音素特征
    """
    def __init__(
        self,
        dim: int = 128,
        bert_model_name: str = "distilbert-base-uncased",
        phone_emb_dim: int = 64,
        max_len: int = 500
    ):
        super().__init__()
        self.dim = dim

        # 1. 预处理解析器
        self.tokenizer = DistilBertTokenizer.from_pretrained(bert_model_name)
        self.g2p = G2p()

        # 2. Text 语义编码分支
        self.text_encoder = DistilBertModel.from_pretrained(bert_model_name)
        self.text_encoder.eval()
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        text_hidden_size = self.text_encoder.config.hidden_size  # 768
        self.text_proj = nn.Linear(text_hidden_size, dim)
        self.text_modality_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        # 3. Phoneme 音素编码分支
        self.phone_emb = nn.Embedding(PHONE_VOCAB_SIZE, phone_emb_dim, padding_idx=0)
        self.phone_proj = nn.Linear(phone_emb_dim, dim)
        self.phone_modality_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        # 4. 共享位置编码
        self.pos_encoder = SinusoidalPositionalEncoding(dim=dim, max_len=max_len)

    def _text_to_phone_ids(self, text: str) -> list[int]:
        """内部 G2P 转换与清洗"""
        raw_phonemes = self.g2p(text)
        clean_ids = []
        for p in raw_phonemes:
            clean_p = re.sub(r'\d+', '', p).strip()  # 去掉重音标记
            if clean_p in PHONE_TO_ID:
                clean_ids.append(PHONE_TO_ID[clean_p])
        return clean_ids

    def forward(self, texts: list[str], device: torch.device = None):
        if device is None:
            device = next(self.parameters()).device

        # ─────────────────────────────────────────────────────────────
        # Step 1: 文本语义特征处理 (DistilBERT -> Projection)
        # ─────────────────────────────────────────────────────────────
        text_inputs = self.tokenizer(texts, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            bert_out = self.text_encoder(**text_inputs)
            raw_text_feat = bert_out.last_hidden_state  # [B, T_t, 768]

        proj_text = self.text_proj(raw_text_feat)  # [B, T_t, D]
        pos_text = self.pos_encoder(proj_text)
        
        # 相加并归一化 -> 得到 E_t_s
        E_t_s = proj_text + pos_text + self.text_modality_embed
        E_t_s = F.normalize(E_t_s, p=2, dim=-1)

        # ─────────────────────────────────────────────────────────────
        # Step 2: 音素特征处理 (G2P -> Embedding -> Projection)
        # ─────────────────────────────────────────────────────────────
        batch_phone_ids = [self._text_to_phone_ids(t) for t in texts]
        max_p_len = max(len(seq) for seq in batch_phone_ids)
        
        # 动态 Padding 补全
        padded_phones = [seq + [0] * (max_p_len - len(seq)) for seq in batch_phone_ids]
        phone_tensor = torch.tensor(padded_phones, dtype=torch.long, device=device)

        raw_phone_feat = self.phone_emb(phone_tensor)  # [B, T_p, 64]
        proj_phone = self.phone_proj(raw_phone_feat)  # [B, T_p, D]
        pos_phone = self.pos_encoder(proj_phone)

        # 相加并归一化 -> 得到 E_p_s
        E_p_s = proj_phone + pos_phone + self.phone_modality_embed
        E_p_s = F.normalize(E_p_s, p=2, dim=-1)

        return E_t_s, E_p_s


# ═══════════ 单元运行测试 ═══════════
if __name__ == "__main__":
    # 实例化端到端模块
    pipeline = EnrollmentTextPipelineModule(dim=128)

    # 模拟真实输入字符串列表
    input_texts = [
        "turn off the lights in the room",
        "open the front door"
    ]

    # 直接传入文本列表
    E_t_s, E_p_s = pipeline(input_texts)

    print("输入文本:", input_texts)
    print("输出 E_t_s (语义特征) 维度:", E_t_s.shape)  # [2, T_text, 128]
    print("输出 E_p_s (音素特征) 维度:", E_p_s.shape)  # [2, T_phone, 128]