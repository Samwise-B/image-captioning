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

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(
            256
        ),  # Resize shorter side to 256 and keep aspect ratio
        torchvision.transforms.CenterCrop(256),  # Optionally crop the center to 256x256
        torchvision.transforms.ToTensor(),
    ]
)

train_dataset = Flickr("train", transform=transform)
# Create DataLoader with the custom collate function
train_loader = DataLoader(
    train_dataset, batch_size=1, shuffle=True, collate_fn=Flickr.collate_fn
)

model = DoubleTrouble(
    vocab_size=train_dataset.vocab_size,
    patch_size=16**2,
    word_embed_dim=train_dataset.vocab_size // 2 + 1,
    img_embed_dim=(16**2) // 2,
    ff_dim_decoder=4 * (train_dataset.vocab_size // 2),
    context_size=100,
    num_patches=768,
    num_layers_encoder=1,
    num_layers_decoder=1,
    num_heads_encoder=1,
    num_heads_decoder=1,
    ff_dim_encoder=4 * (train_dataset.vocab_size // 2),
)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

wandb.init(project="encoder-decoder", name="flickr-multi-head")
running_loss = []
running_accuracy = []
for _ in range(5):
    for i, (patches, tokens, target, cap_lens) in enumerate(
        tqdm(train_loader, desc="Training")
    ):
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
