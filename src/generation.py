import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from utils.evo.tokenization import Vocab

class UTRGenerator:
    """
    针对 3'UTR 优化的迭代取消掩码生成器。
    支持随机种子启动、多项式采样以及基本的统计评估。
    """
    def __init__(self, model, vocab: Vocab, device="cuda"):
        self.model = model.to(device)
        self.model.eval()
        self.vocab = vocab
        self.device = device
        
        # 识别有效的 RNA 碱基
        self.rna_tokens = ["A", "C", "G", "U"]
        self.rna_ids = [vocab.index(t) for t in self.rna_tokens if t in vocab.tokens]

    @torch.no_grad()
    def _unmask_step(self, input_ids, num_to_unmask, temperature=1.0):
        """执行单步迭代，利用模型置信度解开一部分 MASK"""
        outputs = self.model(input_ids)
        logits = outputs["logits"]
        
        # 屏蔽特殊 token，仅保留 A, C, G, U 的概率空间
        mask_filter = torch.full(logits.shape, float('-inf'), device=self.device)
        mask_filter[:, :, self.rna_ids] = 0
        filtered_logits = (logits + mask_filter) / temperature
        probs = F.softmax(filtered_logits, dim=-1)
        
        # 定位剩余的 MASK 位置
        current_masks = (input_ids == self.vocab.mask_idx).nonzero(as_tuple=True)
        if current_masks[1].numel() == 0:
            return input_ids
            
        # 多项式采样 (Multinomial Sampling) 增加生成多样性
        p = probs[current_masks[0], current_masks[1]] 
        predictions_sampled = torch.multinomial(p, 1).squeeze(-1)
        confidences_sampled = p[torch.arange(len(predictions_sampled)), predictions_sampled]

        # 选出置信度最高的 N 个进行填充
        num_to_unmask = min(num_to_unmask, current_masks[1].numel())
        _, topk_indices = torch.topk(confidences_sampled, k=num_to_unmask)
        
        for idx in topk_indices:
            row, col = current_masks[0][idx], current_masks[1][idx]
            input_ids[row, col] = predictions_sampled[idx]
            
        return input_ids

    @torch.no_grad()
    def generate(self, length=100, steps=20, temperature=1.2, seed_ratio=0.1):
        """
        全序列生成循环。
        length: 目标序列长度
        steps: 迭代总步数
        temperature: 采样温度，越高越多样
        seed_ratio: 初始随机种子的比例，用于打破全 U 退化
        """
        # 1. 序列初始化
        input_ids = torch.full((1, length), self.vocab.mask_idx).to(self.device)
        if seed_ratio > 0:
            num_seeds = int(length * seed_ratio)
            seed_indices = random.sample(range(length), num_seeds)
            for idx in seed_indices:
                input_ids[0, idx] = random.choice(self.rna_ids)

        # 2. 添加特殊 token [CLS]...[EOS]
        cls_id = torch.tensor([[self.vocab.bos_idx]]).to(self.device)
        eos_id = torch.tensor([[self.vocab.eos_idx]]).to(self.device)
        input_ids = torch.cat([cls_id, input_ids, eos_id], dim=1)
        
        # 3. 迭代循环
        for s in range(steps):
            mask_indices = (input_ids == self.vocab.mask_idx).nonzero(as_tuple=True)
            current_mask_count = mask_indices[1].numel()
            if current_mask_count == 0:
                break
            
            # 线性分配每步解开的 token 数量
            num_to_unmask = (current_mask_count + (steps - s - 1)) // (steps - s)
            input_ids = self._unmask_step(input_ids, num_to_unmask, temperature)
            
        # 4. 解码返回
        return self.vocab.decode(input_ids[0, 1:-1])

def get_basic_stats(seq):
    """计算序列的基本生物学统计指标"""
    if not seq: return {}
    stats = {
        "length": len(seq),
        "GC_content": (seq.count('G') + seq.count('C')) / len(seq) * 100,
        "base_counts": {base: seq.count(base) for base in "AGCU"}
    }
    return stats
