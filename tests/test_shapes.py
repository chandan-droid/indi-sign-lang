"""Needs torch. Run on your machine first:  python tests/test_shapes.py
Checks shapes, gradient flow, GAM identity at init, and checkpoint key compatibility."""
import sys
sys.path.insert(0, '.')
import torch
from ganet.models.gcmm import GCMM
from ganet.models.gam import GAM
from ganet.models.visual import VisualStage1
from ganet.models.ganet import GANet
from ganet.models.include_model import IncludeClassifier
from ganet.losses import total_loss
from ganet.utils import load_pretrained_visual

B, T, H = 2, 12, 224
x = torch.randn(B, T, 3, H, H)

g = GCMM()(x.transpose(1, 2))
assert [tuple(f.shape[1:]) for f in g] == [(16, T, 56, 56), (32, T, 28, 28), (64, T, 14, 14), (128, T, 7, 7)]

gam = GAM(64, 16).eval()
main, glob = torch.randn(B, 64, T, 8, 8), torch.randn(B, 16, T, 8, 8)
assert torch.allclose(gam(main, glob), main, atol=1e-6), 'GAM must start as identity'

V = 20
model = GANet(V, glp_bins=12, vac=True, vac_tap='conv', imagenet=False)
vlen = torch.tensor([T, T - 2])
out = model(x, vlen)
L = T // 4
assert out['seq_logits'].shape == (B, L, V) and out['out_len'].tolist() == [L, (T - 2) // 4]
assert out['glp_logits'].shape == (B, 12) and out['vac_logits'].shape == (B, L, V)
batch = dict(labels=torch.tensor([1, 2, 3, 4], dtype=torch.int32), label_len=torch.tensor([2, 2], dtype=torch.int32),
             count_bin=torch.tensor([1, 1]))
loss, parts = total_loss(out, batch, dict(glp=1.0, vac=1.0, kd=1.0))
loss.backward()
no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
print('parameters without gradient:', no_grad[:5], len(no_grad))
print('loss parts:', parts)

# pre-training -> fine-tuning key compatibility (modified architecture)
import os, tempfile
tmp_pt = os.path.join(tempfile.gettempdir(), 'inc_test.pt')
inc = IncludeClassifier(10, imagenet=False)
torch.save(dict(visual=inc.visual.state_dict()), tmp_pt)
fresh = GANet(V, glp_tap='temporal', vac=True, vac_tap='visual', imagenet=False)
load_pretrained_visual(fresh.visual, tmp_pt, transfer_gam=True)
k = 'layers.0.0.conv1.weight'
assert torch.equal(fresh.visual.state_dict()[k], inc.visual.state_dict()[k])

# modified architecture shapes & loss test (GLP from BiLSTM + VAC from Stage 1 Visual)
out_mod = fresh(x, vlen)
assert out_mod['seq_logits'].shape == (B, L, V)
assert out_mod['glp_logits'].shape == (B, 12)
assert out_mod['vac_logits'].shape == (B, T, V)  # tapped directly from Stage 1
loss_mod, parts_mod = total_loss(out_mod, batch, dict(glp=1.0, vac=1.0, kd=0.0))
loss_mod.backward()
print('modified architecture loss parts:', parts_mod)
if os.path.exists(tmp_pt):
    os.remove(tmp_pt)
print('shape tests passed')
