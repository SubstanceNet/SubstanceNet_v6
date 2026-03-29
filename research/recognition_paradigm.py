"""
SubstanceNet v4 — Recognition Paradigm Prototype
=================================================
Instead of: train(backprop, 50000 images) → classify(logits)
We test:    encode(few examples) → store(hippocampus) → recognize(similarity)

This is inspired by:
- Complementary Learning Systems (McClelland et al., 1995; Kumaran et al., 2016)
- Ostensive definitions (Dubovykov) — identity through comparison with base
- Biological recognition: see once → remember → recognize later

The model's V1→V4 pipeline extracts features WITHOUT training.
Hippocampus stores episodes WITH class labels.
Recognition = find most similar stored episode.
"""
import sys
sys.path.insert(0, '/media/ssd2/ai_projects/SubstanceNet_v4')

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from src.model.substance_net import SubstanceNet
from collections import defaultdict


def extract_abstract(model, images, device):
    """Run images through V1→V4 pipeline, return abstract representations."""
    with torch.no_grad():
        out = model(images.to(device), mode='image')
    # Use amplitude+phase features (separation ratio ~50) 
    # instead of abstract (separation ratio ~1.6)
    features = torch.cat([out['amplitude'], out['phase']], dim=-1)
    # Pool spatial dimension: [B, 9, 128] -> [B, 128]
    features_pooled = features.mean(dim=1)
    return features_pooled, out['amplitude_c']


def store_examples(model, examples_per_class, train_data, device):
    """
    Phase 1: LEARNING — show few examples, store in hippocampus.
    Like a child seeing 5 cats and 5 dogs.
    """
    # Collect N examples per class
    class_examples = defaultdict(list)
    for img, label in train_data:
        if len(class_examples[label]) < examples_per_class:
            class_examples[label].append(img)
        if all(len(v) >= examples_per_class for v in class_examples.values()):
            break

    total_stored = 0
    for label, images in class_examples.items():
        batch = torch.stack(images)
        abstract, amp_c = extract_abstract(model, batch, device)

        # Store each example as episode with class label
        for i in range(len(images)):
            model.hippocampus.encode_and_store(
                x=abstract[i:i+1],
                consciousness_state=amp_c[i:i+1],
                task_type=f'class_{label}',
                metrics={'class_label': label}
            )
            total_stored += 1

    print(f'  Stored {total_stored} episodes '
          f'({examples_per_class} per class, {len(class_examples)} classes)')
    return total_stored


def recognize(model, test_images, device, top_k=3):
    """
    Phase 2: RECOGNITION — find most similar stored episode.
    Like seeing a new cat and thinking "this looks like the cat I saw before".
    """
    abstract, amp_c = extract_abstract(model, test_images, device)
    predictions = []

    for i in range(min(abstract.shape[0], test_images.shape[0])):
        query = abstract[i:i+1]
        consciousness = amp_c[i:i+1]

        # Find similar episodes in hippocampus
        similar = model.hippocampus.retrieve_similar(
            query, consciousness, top_k=top_k)

        if not similar:
            predictions.append(-1)
            continue

        # Vote: most common class among similar episodes
        votes = defaultdict(float)
        for episode in similar:
            label = episode['metrics'].get('class_label', -1)
            # Weight by decay factor (fresher memories count more)
            votes[label] += episode['decay_factor']

        predicted = max(votes, key=votes.get)
        predictions.append(predicted)

    return predictions


def run_experiment(examples_per_class=5, test_size=500, in_channels=1):
    """Run complete encode→store→recognize experiment."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model — NO TRAINING, random weights
    # V1→V4 extracts features with biological Gabor filters (not learned)
    model = SubstanceNet(
        num_classes=10, in_channels=in_channels, abstract_dim=3
    ).to(device)
    model.eval()

    # Dataset
    if in_channels == 1:
        transform = transforms.Compose([transforms.ToTensor()])
        dataset_cls = datasets.MNIST
        dataset_name = 'MNIST'
    else:
        transform = transforms.Compose([transforms.ToTensor()])
        dataset_cls = datasets.CIFAR10
        dataset_name = 'CIFAR-10'

    train_data = dataset_cls('data', train=True, download=False, transform=transform)
    test_data = dataset_cls('data', train=False, transform=transform)

    print(f'\n{"="*60}')
    print(f'Recognition Paradigm: {dataset_name}')
    print(f'Examples per class: {examples_per_class}')
    print(f'Model: UNTRAINED (random weights + biological Gabor filters)')
    print(f'{"="*60}')

    # Phase 1: Store examples
    print(f'\nPhase 1: LEARNING (encoding {examples_per_class} examples/class)')
    store_examples(model, examples_per_class, train_data, device)

    # Phase 2: Recognition
    print(f'\nPhase 2: RECOGNITION (testing on {test_size} images)')
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=32, shuffle=True)

    all_preds = []
    all_labels = []
    tested = 0

    for images, labels in test_loader:
        if tested >= test_size:
            break
        preds = recognize(model, images, device, top_k=5)
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
        tested += len(images)

    # Results
    all_preds = np.array(all_preds[:test_size])
    all_labels = np.array(all_labels[:test_size])
    accuracy = (all_preds == all_labels).mean()

    # Consciousness metrics
    with torch.no_grad():
        sample_img = next(iter(test_loader))[0][:4].to(device)
        out = model(sample_img, mode='image')
        m = model.get_consciousness_metrics(out)

    print(f'\n{"="*60}')
    print(f'RESULTS: {dataset_name} Recognition')
    print(f'{"="*60}')
    print(f'  Examples per class:  {examples_per_class}')
    print(f'  Test size:           {test_size}')
    print(f'  Recognition acc:     {accuracy:.4f}')
    print(f'  Random baseline:     {1/10:.4f}')
    print(f'  R (consciousness):   {m["reflexivity_score"]:.4f}')
    print(f'  Memory episodes:     {model.hippocampus.get_state()["total_episodes"]}')
    print(f'  Model trained:       NO (biological features only)')

    # Per-class accuracy
    print(f'\n  Per-class accuracy:')
    for c in range(10):
        mask = all_labels == c
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == c).mean()
            print(f'    Class {c}: {class_acc:.4f} ({mask.sum()} samples)')

    return accuracy


if __name__ == '__main__':
    print('='*60)
    print('SubstanceNet v4 — Recognition Paradigm')
    print('encode → store → recognize (NO backpropagation)')
    print('='*60)

    # Test with different numbers of examples
    for n in [1, 5, 10, 20]:
        acc = run_experiment(examples_per_class=n, test_size=500, in_channels=1)
        print(f'\n  >>> {n}-shot accuracy: {acc:.4f}\n')
