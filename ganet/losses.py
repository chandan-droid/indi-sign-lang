import torch
import torch.nn.functional as F

from .models.heads import length_mask


def ctc_loss(logits, in_len, targets, tgt_len):
    """logits [B, T, V]; targets: 1-D concatenated label ids (blank = 0 is never a target)."""
    lp = logits.float().log_softmax(-1).transpose(0, 1)             # CTC in float32
    return F.ctc_loss(lp, targets.cpu().int(), in_len.cpu().int(), tgt_len.cpu().int(),
                      blank=0, reduction='mean', zero_infinity=True)


def seq_kd(student, teacher, lens, T=8.0):
    """VAC-style distillation: KL(teacher || student) with temperature, teacher detached."""
    m = length_mask(lens, student.size(1)).float()
    logp_s = F.log_softmax(student.float() / T, -1)
    p_t = F.softmax(teacher.detach().float() / T, -1)
    kl = (p_t * (torch.log(p_t.clamp_min(1e-8)) - logp_s)).sum(-1)
    return (kl * m).sum() / m.sum().clamp_min(1.0) * T * T


def total_loss(out, batch, w):
    """L_total = L_CTC + lambda_GLP * L_GLP + alpha * L_VAC(ctc) + beta * L_KD."""
    dev = out['seq_logits'].device
    parts = {'ctc': ctc_loss(out['seq_logits'], out['out_len'], batch['labels'], batch['label_len'])}
    loss = parts['ctc']
    if 'glp_logits' in out and w.get('glp', 0) > 0:
        parts['glp'] = F.cross_entropy(out['glp_logits'].float(), batch['count_bin'].to(dev))
        loss = loss + w['glp'] * parts['glp']
    if 'vac_logits' in out and w.get('vac', 0) > 0:
        parts['vac'] = ctc_loss(out['vac_logits'], out['vac_len'], batch['labels'], batch['label_len'])
        loss = loss + w['vac'] * parts['vac']
        if w.get('kd', 0) > 0 and out['vac_logits'].size(1) == out['seq_logits'].size(1):
            parts['kd'] = seq_kd(out['vac_logits'], out['seq_logits'], out['out_len'], w.get('kd_T', 8.0))
            loss = loss + w['kd'] * parts['kd']
    return loss, {k: float(v.detach()) for k, v in parts.items()}
