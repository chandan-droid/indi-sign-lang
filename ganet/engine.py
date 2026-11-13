import torch

from .metrics import greedy_decode, prefix_beam_search, wer_report


@torch.no_grad()
def run_inference(model, loader, vocab, device, beam=1, amp=True):
    model.eval()
    ids, refs, hyps = [], [], []
    # Normalize device to torch.device object
    dev = torch.device(device) if isinstance(device, str) else device
    is_cuda = (dev.type == 'cuda')

    for batch in loader:
        if is_cuda and amp:
            try:
                autocast_ctx = torch.amp.autocast('cuda', dtype=torch.float16)
            except (AttributeError, TypeError):
                autocast_ctx = torch.cuda.amp.autocast()
        else:
            class DummyCtx:
                def __enter__(self): pass
                def __exit__(self, *args): pass
            autocast_ctx = DummyCtx()

        with autocast_ctx:
            out = model(batch['video'].to(dev), batch['vlen'].to(dev))
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
