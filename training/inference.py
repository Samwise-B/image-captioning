import torch
from tqdm import tqdm
from pathlib import Path
import sys
from torchvision.transforms import ToPILImage
from PIL import Image
import random
import wandb
import torch.nn.functional as F

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir))

from models.gpt_transformer import DoubleTrouble

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_inference(model, ds, i, temperature=1):
    tokeniser = ds.tokeniser
    if ds.gpt:
        bos_token = tokeniser.bos_token_id
        eos_token = tokeniser.eos_token_id
    else:
        bos_token = tokeniser.bos_id()
        eos_token = tokeniser.eos_id()

    with torch.inference_mode():
        # idx = random.randint(0, ds.__len__())
        patches, _, target = ds[i]
        patches = patches.to(device)
        inpt_text = ""
        inpt = torch.tensor([bos_token], device=device).unsqueeze(0)
        while inpt.shape[-1] <= target.shape[-1] and inpt[0, -1] != eos_token:
            next_pred = model(inpt, patches)
            logits = next_pred[:, -1, :]
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            inpt = torch.cat([inpt, next_token], dim=-1)

        # print(f"Target: {tokeniser.decode(target, skip_special_tokens=True)}")
        # print(f"Prediction: {tokeniser.decode(inpt.squeeze(), skip_special_tokens=True)}")
        if ds.gpt:
            return tokeniser.decode(
                inpt.squeeze(), skip_special_tokens=True
            ), tokeniser.decode(target, skip_special_tokens=True)
        else:
            return tokeniser.decode(inpt.squeeze().tolist()), tokeniser.decode(target.tolist())


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from sets.flickr import Flickr

    val_dataset = Flickr("val", num_rows=10)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=True, collate_fn=Flickr.collate_fn
    )

    model = DoubleTrouble()
    model = model.to(device)
    # patches, inpt, targ, img = val_dataset
    out, targ = run_inference(model, val_dataset, 0)
    pass
