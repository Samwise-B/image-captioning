import torch
from tqdm import tqdm
from pathlib import Path
import sys
from torchvision.transforms import ToPILImage
from PIL import Image
import random
import wandb

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir))

from models.gpt_transformer import DoubleTrouble

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_inference(model, ds, i):
    tokeniser = ds.tokeniser
    bos_token = tokeniser.bos_token_id
    eos_token = tokeniser.eos_token_id

    with torch.inference_mode():
        # idx = random.randint(0, ds.__len__())
        patches, _, target = ds[i]
        patches = patches.to(device)
        inpt_text = ""
        inpt = torch.tensor([bos_token], device=device).unsqueeze(0)
        while inpt.shape[-1] <= target.shape[-1] and inpt[0, -1] != eos_token:
            next_pred = model(inpt, patches)
            next_tokens = torch.argmax(next_pred, dim=-1)

            next_char = tokeniser.decode(next_tokens[0, -1])
            inpt_text += next_char
            inpt = torch.cat([inpt, next_tokens[0, -1:].unsqueeze(0)], dim=1)
            # print(inpt_text)

        if not wandb:
            print(f"Target: {target}")
            print(f"Prediction: {inpt_text}")
            # img = patches.cpu().clone()
            img[0].show()

        return inpt_text, tokeniser.decode(target), img


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from sets.flickr import Flickr

    val_dataset = Flickr("val", num_rows=10)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=True, collate_fn=Flickr.collate_fn
    )

    model = DoubleTrouble()
    # patches, inpt, targ, img = val_dataset
    out, targ, img = run_inference(model, val_dataset)
    pass
