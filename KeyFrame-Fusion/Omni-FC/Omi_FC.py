import torch
import torch.nn as nn

class QbyOmniSequenceClassifier(nn.Module):
    """
    QbyOmni 分支的逐 Token/逐帧 FC 判别头
    对应图 2 中面向文本 (L_text) 和音素 (L_phon) 的分类器[cite: 3]
    """
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        # 逐位置做特征映射
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)  # 输出每个位置匹配概率
        )

    def forward(self, F_mc: torch.Tensor, target_masks: torch.Tensor = None, labels: torch.Tensor = None):
        """
        参数:
            F_mc: 特征矩阵 [B, T_sm, D] (例如 F_t_c 或 F_p_c)[cite: 3]
            target_masks: 序列 Mask 掩码 [B, T_sm] (忽略 Padding 位置)[cite: 3]
            labels: 逐位置标签 [B, T_sm][cite: 3]
        """
        # 1. 对序列上的每一个 Token 计算 Logit
        logits = self.fc(F_mc).squeeze(-1)  # [B, T_sm]
        probs = torch.sigmoid(logits)        # [B, T_sm]

        # 2. 计算 Masked BCE Loss (对应公式 8 L_s^m)[cite: 3]
        loss = None
        if labels is not None and target_masks is not None:
            criterion = nn.BCEWithLogitsLoss(reduction='none')
            raw_loss = criterion(logits, labels.float())
            
            # 仅计算有效填充位置的损失[cite: 3]
            masked_loss = raw_loss * target_masks.float()
            loss = masked_loss.sum() / (target_masks.sum() + 1e-8)

        return probs, loss


# ---------------- 单元测试 ----------------
if __name__ == "__main__":
    B, D = 4, 128
    
    # 模拟文本序列与音素序列[cite: 3]
    T_text, T_phon = 5, 10
    F_tc = torch.randn(B, T_text, D)
    F_pc = torch.randn(B, T_phon, D)

    # 模拟 Mask 与逐位置标签
    text_mask = torch.tensor([[1, 1, 1, 0, 0]] * B)
    text_labels = torch.randint(0, 2, (B, T_text)).float()

    phon_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0]] * B)
    phon_labels = torch.randint(0, 2, (B, T_phon)).float()

    # 实例化分类器
    text_classifier = QbyOmniSequenceClassifier(embed_dim=D)
    phon_classifier = QbyOmniSequenceClassifier(embed_dim=D)

    text_probs, loss_text = text_classifier(F_tc, text_mask, text_labels)
    phon_probs, loss_phon = phon_classifier(F_pc, phon_mask, phon_labels)

    print("QbyOmni Text 概率维度:", text_probs.shape)   # [4, 5]
    print("QbyOmni Text Loss (L_text):", loss_text.item() if loss_text is not None else "N/A")
    print("QbyOmni Phoneme 概率维度:", phon_probs.shape) # [4, 10]
    print("QbyOmni Phoneme Loss (L_phon):", loss_phon.item() if loss_phon is not None else "N/A")