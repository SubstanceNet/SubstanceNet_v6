"""SubstanceNet v4 — Recognition Demo"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch, torch.nn.functional as F
from collections import defaultdict

print()
print('  SubstanceNet v4 — Recognition Demo')
print('  ===================================')
print('  Encode → Store → Recognize (no backprop)')
print()

from src.model.substance_net import SubstanceNet
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SubstanceNet(num_classes=10).to(device).eval()

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(data_dir, train=True, download=False, transform=transform)
test_data = datasets.MNIST(data_dir, train=False, transform=transform)

def get_feat(images):
    with torch.no_grad():
        out = model(images.to(device), mode='image')
        return torch.cat([out['amplitude'], out['phase']], dim=-1).mean(dim=1)

for N in [5, 20, 100]:
    memory = []
    cc = defaultdict(int)
    for img, label in train_data:
        if cc[label] < N:
            memory.append((get_feat(img.unsqueeze(0)).squeeze(0), label))
            cc[label] += 1
        if len(cc) == 10 and all(v >= N for v in cc.values()):
            break

    mem_feats = torch.stack([m[0] for m in memory])
    mem_labels = [m[1] for m in memory]

    correct, total = 0, 0
    loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)
    for images, labels in loader:
        if total >= 1024: break
        feats = get_feat(images)
        sims = F.cosine_similarity(feats.unsqueeze(1), mem_feats.unsqueeze(0), dim=2)
        topk_vals, topk_idx = sims.topk(5, dim=1)
        for i in range(feats.shape[0]):
            if total >= 1024: break
            votes = defaultdict(float)
            for j in range(5):
                votes[mem_labels[topk_idx[i,j].item()]] += topk_vals[i,j].item()
            if max(votes, key=votes.get) == labels[i].item():
                correct += 1
            total += 1

    print(f'  {N:>3}-shot: {correct/total:.1%} ({correct}/{total} correct, {N*10} memories)')

print()
print('  No backprop. No optimizer. No loss function.')
print('  Just: see → remember → recognize.')
print()
