import torch

from .metrics import greedy_decode, prefix_beam_search, wer_report


@torch.no_grad()
def run_inference(model, loader, vocab, device, beam=1, amp=True):
    model.eval()
    ids, refs, hyps = [], [], []
    for batch in loader:
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp and device.type == 'cuda'):
            out = model(batch['video'].to(device), batch['vlen'].to(device))
        probs = out['seq_logits'].float().softmax(-1).cpu().numpy()
        olen = out['out_len'].cpu().tolist()
        for b in range(probs.shape[0]):
            seq = (greedy_decode(probs[b], olen[b]) if beam <= 1
                   else prefix_beam_search(probs[b], olen[b], beam=beam))
            hyps.append(vocab.decode(seq))
            refs.append(batch['ref'][b])
            ids.append(batch['id'][b])
    return ids, refs, hyps


def evaluate(model, loader, vocab, device, beam=1, amp=True):
    ids, refs, hyps = run_inference(model, loader, vocab, device, beam, amp)
    rep = wer_report(refs, hyps)
    rep['predictions'] = [dict(id=i, ref=' '.join(r), hyp=' '.join(h)) for i, r, h in zip(ids, refs, hyps)]
    return rep
