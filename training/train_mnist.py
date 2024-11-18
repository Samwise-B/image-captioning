from pathlib import Path
import sys
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import torch
import torchvision

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir))

from sets.mnist import Combine
from models.combined import DoubleTrouble

full_dataset = torchvision.datasets.MNIST(root=".", download=True)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

# Randomly split the dataset
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_dataset = Combine(train_dataset)
# Create DataLoader with the custom collate function
train_loader = DataLoader(
    train_dataset, batch_size=24, shuffle=True, collate_fn=Combine.collate_fn
)

test_dataset = Combine(test_dataset)
test_loader = DataLoader(
    test_dataset, batch_size=24, shuffle=True, collate_fn=Combine.collate_fn
)

model = DoubleTrouble(
    vocab_size=12,
    patch_size=196,
    word_embed_dim=6,
    img_embed_dim=98,
    ff_dim_decoder=4 * 98,
    context_size=6,
    num_patches=16,
    num_layers_encoder=1,
    num_layers_decoder=1,
    num_heads_encoder=98 / 7,
    num_heads_decoder=2,
    ff_dim_encoder=4 * 98,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

wandb.init(project="encoder-decoder", name="mnist-multi-head-3-layers-with-val")
running_loss = []
running_accuracy = []
for _ in range(5):
    for i, (patches, tokens, target) in enumerate(tqdm(train_loader, desc="Training")):
        optimizer.zero_grad()
        pred = model(tokens, patches)
        loss = criterion(pred.view(-1, pred.size(-1)), target.view(-1))
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())

        correct = (
            (torch.argmax(pred.view(-1, pred.size(-1)), dim=1) == target.view(-1))
            .sum()
            .item()
        )  # Count correct predictions
        total = target.view(-1).size(0)  # Total number of predictions
        accuracy = correct / total * 100
        running_accuracy.append(accuracy)

        if (i + 1) % 100 == 0:
            # print("", end="\r")
            # print(f"loss: {sum(running_loss) / 100}", end="\r")
            wandb.log(
                {
                    "loss-100": sum(running_loss) / 100,
                    "accuracy-100": sum(running_accuracy) / 100,
                }
            )
            running_loss = []
            running_accuracy = []
    with torch.inference_mode():
        val_accuracy = []
        for i, (patches, tokens, target) in enumerate(
            tqdm(test_loader, desc="Validation")
        ):
            pred = model(tokens, patches)
            correct = (
                (torch.argmax(pred.view(-1, pred.size(-1)), dim=1) == target.view(-1))
                .sum()
                .item()
            )  # Count correct predictions
            total = target.view(-1).size(0)  # Total number of predictions
            accuracy = correct / total * 100
            val_accuracy.append(accuracy)
            if (i + 1) % 100 == 0:
                # print("", end="\r")
                # print(f"loss: {sum(running_loss) / 100}", end="\r")
                wandb.log(
                    {
                        "val-accuracy-100": sum(val_accuracy) / 100,
                    }
                )
                val_accuracy = []


wandb.finish()
