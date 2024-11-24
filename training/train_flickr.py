from pathlib import Path
import sys
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import torch
import torchvision
import random
import logging
import nltk


repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir))

from sets.flickr import Flickr
from models.gpt_transformer import DoubleTrouble
from models.transformer import Transformer
from training.inference import run_inference
from utils.metrics import compute_bleu, compute_meteor, compute_rouge

torch.manual_seed(42)
model_dir = repo_dir / "weights/"



logging.basicConfig(
    filename="training.log",  # File to log to
    filemode="w",  # Overwrite file each time; use "a" to append
    level=logging.INFO,  # Log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
)

def main():
    nltk.download('wordnet')
    nltk.download('omw')

    batch_size = 24
    subset_size = -1
    train_dataset = Flickr("train", num_rows=subset_size, gpt=False)
    # Create DataLoader with the custom collate function
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=Flickr.collate_fn
    )

    val_dataset = Flickr("val", num_rows=subset_size, gpt=False)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=Flickr.collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Available device is {device}")

    patch_size = (16**2) * 3
    args = {
        "vocab_size": train_dataset.vocab_size,
        "patch_size": patch_size,
        "word_embed_dim": patch_size,
        "img_embed_dim": patch_size,
        "ff_dim_decoder": 4 * patch_size,
        "num_patches": 196,
        "num_layers_encoder": 1,
        "num_layers_decoder": 12,
        "num_heads_encoder": 1,
        "num_heads_decoder": 12,
        "ff_dim_encoder": 4 * patch_size,
    }

    #model = DoubleTrouble()
    model = Transformer(**args)
    model.to(device)

    print(
        f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    model_name = "flickr-vit-transformer-full"
    num_epochs = 100
    args["num_epochs"] = num_epochs
    args["batch_size"] = batch_size
    args["subset_size"] = subset_size
    save_counter = 0


    wandb.init(project="image-captioning", name=model_name, config=args)
    running_loss = []
    for epoch in range(num_epochs):
        for i, (patches, tokens, target, cap_lens) in enumerate(
            tqdm(train_loader, desc=f"Training {epoch}")
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


            # if (i + 1) % (train_dataset.__len__() / 10) == 0:
            if (i + 1) % 1 == 0:
                torch.save(model.state_dict(), model_dir / f"{model_name}-{save_counter % 5}.pt")
                wandb.save(str(model_dir / f"{model_name}-{save_counter % 5}.pt"), base_path=str(model_dir))
                save_counter += 1

                for step in range(5):
                    o, t = run_inference(model, train_loader.dataset, step)
                    logging.info(f"\nPrediction: {o}. \nGround Truth: {t}")

                # clear training batch from memory
                del patches, tokens, target, pred, loss
                torch.cuda.empty_cache()

                val_loss, bleu_scores, meteor_scores, rouge_scores = [], [], [], []
                # evaluation
                with torch.inference_mode():
                    for i, (patches, tokens, target, cap_lens) in enumerate(
                        tqdm(val_loader, desc="Validation")
                    ):
                        patches = patches.to(device)
                        tokens = tokens.to(device)
                        target = target.to(device)

                        pred = model(tokens, patches)
                        pred = torch.cat(
                            [x[: cap_lens[i]] for i, x in enumerate(pred)], dim=0
                        )

                        loss = criterion(pred, target)
                        val_loss.append(loss.item())
                        
                        pred_tokens = torch.argmax(pred, dim=1)
                        pred_cap, targ_cap = model.get_captions(pred_tokens, target, train_dataset.tokeniser)

                        bleu_scores.append(compute_bleu(targ_cap, pred_cap)[3])  # BLEU-4
                        meteor_scores.append(compute_meteor([targ_cap.split()], pred_cap.split()))
                        rouge_scores.append(compute_rouge(targ_cap, pred_cap)['rougeL'].fmeasure)

                # log data
                wandb.log(
                    {
                        "loss": sum(running_loss) / len(running_loss),
                        "val-loss": sum(val_loss) / len(val_loss),
                        "bleu_score (4gram)": sum(bleu_scores) / len(bleu_scores),
                        "meteor_score": sum(meteor_scores) / len(meteor_scores),
                        "rouge_scores": sum(rouge_scores) / len(rouge_scores)
                    }
                )
                running_loss = []
                val_loss = []

    torch.save(model.state_dict(), model_dir / f"{model_name}-final.pt")
    wandb.save(model_dir / f"{model_name}-final.pt", base_path="weights")
    wandb.finish()

if __name__ == "__main__":
    main()