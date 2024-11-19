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

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(
            256
        ),  # Resize shorter side to 256 and keep aspect ratio
        torchvision.transforms.CenterCrop(256),  # Optionally crop the center to 256x256
        torchvision.transforms.ToTensor(),
    ]
)

train_dataset = Flickr("train", num_rows=-1, transform=transform)
# Create DataLoader with the custom collate function
train_loader = DataLoader(
    train_dataset, batch_size=128, shuffle=True, collate_fn=Flickr.collate_fn
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Available device is {device}")
patch_size = (16**2) * 3
model = DoubleTrouble(
    vocab_size=train_dataset.vocab_size,
    patch_size=patch_size,
    word_embed_dim=train_dataset.vocab_size // 8 + 1,
    img_embed_dim=patch_size,
    ff_dim_decoder=2 * (train_dataset.vocab_size // 8),
    num_patches=196,
    num_layers_encoder=1,
    num_layers_decoder=1,
    num_heads_encoder=1,
    num_heads_decoder=1,
    ff_dim_encoder=2 * (train_dataset.vocab_size // 8),
)

model.to(device)

print(
    f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

wandb.init(project="image-captioning", name="flickr-vit-100rows")
running_loss = []
running_accuracy = []
for _ in range(1000):
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
            (torch.argmax(pred.view(-1, pred.size(-1)), dim=1) == target.view(-1))
            .sum()
            .item()
        )  # Count correct predictions
        total = target.view(-1).size(0)  # Total number of predictions
        accuracy = correct / total
        running_accuracy.append(accuracy)

        # print("", end="\r")
        # print(f"loss: {sum(running_loss) / 100}", end="\r")
        # if (i+1) % 100 == 0:
        wandb.log(
            {
                "loss-100": sum(running_loss),
                "accuracy-100": sum(running_accuracy),
            }
        )
        running_loss = []
        running_accuracy = []
