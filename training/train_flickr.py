from pathlib import Path
import sys
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import torch
import torchvision

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir))

from sets.flickr import Flickr
from models.combined import DoubleTrouble

torch.manual_seed(42)
model_dir = repo_dir / "weights/"

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(
            256
        ),  # Resize shorter side to 256 and keep aspect ratio
        torchvision.transforms.CenterCrop(256),  # Optionally crop the center to 256x256
        torchvision.transforms.ToTensor(),
    ]
)

train_dataset = Flickr("train", num_rows=128)
# Create DataLoader with the custom collate function
train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, collate_fn=Flickr.collate_fn
)

val_dataset = Flickr("val", num_rows=128)
val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=Flickr.collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Available device is {device}")

patch_size = (16**2) * 3
args = {
    "vocab_size": train_dataset.vocab_size,
    "patch_size": patch_size,
    "word_embed_dim": train_dataset.vocab_size // 8 + 1,
    "img_embed_dim": patch_size,
    "ff_dim_decoder": 2 * (train_dataset.vocab_size // 8),
    "num_patches": 196,
    "num_layers_encoder": 1,
    "num_layers_decoder": 1,
    "num_heads_encoder": 1,
    "num_heads_decoder": 1,
    "ff_dim_encoder": 2 * (train_dataset.vocab_size // 8),
}

model = DoubleTrouble(**args)
model.to(device)

print(
    f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

model_name = "flickr-vit-100rows"
wandb.init(project="image-captioning", name=model_name, config=args)
running_loss = []
running_accuracy = []
for epoch in range(100):
    for i, (patches, tokens, target, cap_lens) in enumerate(
        tqdm(train_loader, desc="Training")
    ):
        patches = patches.to(device)
        tokens = tokens.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        pred = model(tokens, patches)
        pred = torch.cat([x[: cap_lens[i]] for i, x in enumerate(pred)], dim=0)

        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())

        correct = (
            (torch.argmax(pred, dim=1) == target).sum().item()
        )  # Count correct predictions
        total = target.size(0)  # Total number of predictions
        accuracy = correct / total
        running_accuracy.append(accuracy)

        if (i + 1) % 5000 == 0:
            torch.save(model.state_dict(), model_dir / f"{model_name}-e{epoch}-{i}.pt")
            wandb.save(model_dir / f"{model_name}-e{epoch}-{i}.pt", base_path="weights")

    val_loss = []
    val_accuracy = []
    # evaluation
    with torch.inference_mode():
        for i, (patches, tokens, target, cap_lens) in enumerate(
            tqdm(val_loader, desc="Validation")
        ):
            patches = patches.to(device)
            tokens = tokens.to(device)
            target = target.to(device)

            pred = model(tokens, patches)
            pred = torch.cat([x[: cap_lens[i]] for i, x in enumerate(pred)], dim=0)

            loss = criterion(pred, target)
            val_loss.append(loss.item())

            correct = (
                (torch.argmax(pred, dim=1) == target).sum().item()
            )  # Count correct predictions
            total = target.size(0)  # Total number of predictions
            accuracy = correct / total
            val_accuracy.append(accuracy)

    # log data
    wandb.log(
        {
            "loss": sum(running_loss) / train_dataset.__len__(),
            "precision": sum(running_accuracy) / train_dataset.__len__(),
            "val-loss": sum(val_loss) / val_dataset.__len__(),
            "val-precision": sum(val_accuracy) / val_dataset.__len__(),
        }
    )
    running_loss = []
    running_accuracy = []
    val_loss = []
    val_accuracy = []

torch.save(model.state_dict(), model_dir / f"{model_name}-e{epoch}-{i}.pt")
wandb.save(model_dir / f"{model_name}-e{epoch}-{i}.pt", base_path="weights")
wandb.finish()
