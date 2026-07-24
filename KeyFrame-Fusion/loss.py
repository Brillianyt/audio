import torch
import torch.nn as nn
import torch.nn.functional as F

class KFCKWSLoss(nn.Module):
    """
    KFC-KWS 论文 Section 2.3 复合损失函数实现[cite: 3]
    包含：L_u (整句级), L_s_p (音素级), L_s_t (文本级), L_c (CTC 损失)[cite: 3]
    """
    def __init__(self, lambda_ctc: float = 0.2, blank_idx: int = 0):
        super().__init__()
        self.lambda_ctc = lambda_ctc
        self.blank_idx = blank_idx
        
        # 交叉熵损失函数
        self.bce_utt = nn.BCEWithLogitsLoss()
        self.bce_seq = nn.BCEWithLogitsLoss(reduction='none')
        self.ctc_loss_fn = nn.CTCLoss(blank=self.blank_idx, zero_infinity=True)

    def forward(
        self,
        # 1. QbyKeyframe 预测 Logits 与整句标签
        logit_u: torch.Tensor,             # [B, 1][cite: 3]
        label_u: torch.Tensor,             # [B, 1][cite: 3]
        
        # 2. QbyOmni 序列预测 Logits、掩码与标签
        logits_p: torch.Tensor,            # [B, T_sp][cite: 3]
        mask_p: torch.Tensor,              # [B, T_sp][cite: 3]
        labels_p: torch.Tensor,            # [B, T_sp][cite: 3]
        
        logits_t: torch.Tensor,            # [B, T_st][cite: 3]
        mask_t: torch.Tensor,              # [B, T_st][cite: 3]
        labels_t: torch.Tensor,            # [B, T_st][cite: 3]
        
        # 3. CTC 预测分布与真实音素序列 (可选)
        ctc_logits_q: torch.Tensor = None, # [T_q, B, Num_Classes][cite: 3]
        targets_q: torch.Tensor = None,    # [B, S_q][cite: 3]
        input_lens_q: torch.Tensor = None, # [B]
        target_lens_q: torch.Tensor = None,# [B]
        
        ctc_logits_s: torch.Tensor = None, # [T_s, B, Num_Classes][cite: 3]
        targets_s: torch.Tensor = None,    # [B, S_s][cite: 3]
        input_lens_s: torch.Tensor = None, # [B]
        target_lens_s: torch.Tensor = None # [B]
    ) -> dict:
        
        # ==================== 1. 整句级别损失 L_u (公式 7) ====================[cite: 3]
        L_u = self.bce_utt(logit_u, label_u.float())

        # ==================== 2. 序列级别损失 L_s^p 和 L_s^t (公式 8) ====================[cite: 3]
        # 音素序列损失 L_s^p
        loss_p_raw = self.bce_seq(logits_p, labels_p.float())
        L_s_p = (loss_p_raw * mask_p.float()).sum() / (mask_p.float().sum() + 1e-8)

        # 文本序列损失 L_s^t
        loss_t_raw = self.bce_seq(logits_t, labels_t.float())
        L_s_t = (loss_t_raw * mask_t.float()).sum() / (mask_t.float().sum() + 1e-8)

        # ==================== 3. CTC 损失 L_c (公式 9) ====================[cite: 3]
        L_c = torch.tensor(0.0, device=logit_u.device)
        if ctc_logits_q is not None and targets_q is not None:
            # Query 音频 CTC Loss[cite: 3]
            log_probs_q = F.log_softmax(ctc_logits_q, dim=-1)
            loss_ctc_q = self.ctc_loss_fn(log_probs_q, targets_q, input_lens_q, target_lens_q)
            
            # Enroll 音频 CTC Loss[cite: 3]
            if ctc_logits_s is not None and targets_s is not None:
                log_probs_s = F.log_softmax(ctc_logits_s, dim=-1)
                loss_ctc_s = self.ctc_loss_fn(log_probs_s, targets_s, input_lens_s, target_lens_s)
            else:
                loss_ctc_s = 0.0

            L_c = loss_ctc_q + loss_ctc_s

        # ==================== 4. 复合损失计算 (公式 10) ====================[cite: 3]
        L_total = L_u + L_s_p + L_s_t + self.lambda_ctc * L_c

        return {
            "loss_total": L_total,
            "loss_utt": L_u,
            "loss_phoneme_seq": L_s_p,
            "loss_text_seq": L_s_t,
            "loss_ctc": L_c
        }


# ═══════════ 单元运行测试 ═══════════
if __name__ == "__main__":
    B = 4
    T_q, T_s = 100, 80      # 音频时间帧数
    T_sp, T_st = 10, 5      # 音素和文本的 Token 数量
    vocab_size = 65         # 音素词表大小 + 1 (blank)[cite: 3]

    # 1. 模拟网络前向输出
    logit_u  = torch.randn(B, 1)                  # QbyKeyframe 分类 logit[cite: 3]
    logits_p = torch.randn(B, T_sp)               # QbyOmni 音素 logit[cite: 3]
    logits_t = torch.randn(B, T_st)               # QbyOmni 文本 logit[cite: 3]

    ctc_logits_q = torch.randn(T_q, B, vocab_size) # CTC 模块的 Query 输出[cite: 3]
    ctc_logits_s = torch.randn(T_s, B, vocab_size) # CTC 模块的 Enroll 输出[cite: 3]

    # 2. 模拟标签与掩码
    label_u   = torch.randint(0, 2, (B, 1))
    
    mask_p    = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0]] * B)
    labels_p  = torch.randint(0, 2, (B, T_sp))

    mask_t    = torch.tensor([[1, 1, 1, 0, 0]] * B)
    labels_t  = torch.randint(0, 2, (B, T_st))

    targets_q = torch.randint(1, vocab_size, (B, 8))  # 真实音素序列[cite: 3]
    targets_s = torch.randint(1, vocab_size, (B, 6))

    input_lens_q  = torch.full((B,), T_q, dtype=torch.long)
    target_lens_q = torch.full((B,), 8, dtype=torch.long)
    
    input_lens_s  = torch.full((B,), T_s, dtype=torch.long)
    target_lens_s = torch.full((B,), 6, dtype=torch.long)

    # 3. 计算 Loss
    criterion = KFCKWSLoss(lambda_ctc=0.2)[cite: 3]
    loss_dict = criterion(
        logit_u, label_u,
        logits_p, mask_p, labels_p,
        logits_t, mask_t, labels_t,
        ctc_logits_q, targets_q, input_lens_q, target_lens_q,
        ctc_logits_s, targets_s, input_lens_s, target_lens_s
    )

    print("=== 损失计算结果 ===")
    print(f"总损失 (L_total): {loss_dict['loss_total'].item():.4f}")
    print(f"整句损失 (L_u)    : {loss_dict['loss_utt'].item():.4f}")
    print(f"音素序列损失 (L_s_p): {loss_dict['loss_phoneme_seq'].item():.4f}")
    print(f"文本序列损失 (L_s_t): {loss_dict['loss_text_seq'].item():.4f}")
    print(f"CTC 损失 (L_c)     : {loss_dict['loss_ctc'].item():.4f}")