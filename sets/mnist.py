import torch
import torchvision
import random
import PIL


class Combine(torch.utils.data.Dataset):
    def __init__(self, ds, window_size: int = 14):
        super().__init__()
        self.window_size = window_size
        self.tf = torchvision.transforms.ToTensor()
        self.ds = ds
        self.tokeniser = {"<s>": 10, "</s>": 11}
        for token in range(10):
            self.tokeniser[token] = token + 2
        self.ln = len(self.ds)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        idx = random.sample(range(self.__len__()), 4)
        store = []
        label = []

        for i in idx:
            x, y = self.ds[i]
            store.append(x)
            label.append(y)

        img = PIL.Image.new("L", (56, 56))
        img.paste(store[0], (0, 0))
        img.paste(store[1], (28, 0))
        img.paste(store[2], (0, 28))
        img.paste(store[3], (28, 28))
        patches = self.get_patches(img)
        enc_label = self.encode_label(label)
        return patches, enc_label[0][:-1].unsqueeze(0), enc_label[0][1:].unsqueeze(0)

    def collate_fn(batch):
        patches, tokens, targets = zip(*batch)
        # Concatenate images and labels
        images = torch.cat(patches, dim=0)
        inpts = torch.cat(tokens, dim=0)
        targets = torch.cat(targets, dim=0)

        return images, inpts, targets

    def encode_label(self, label):
        return torch.tensor(
            [[self.tokeniser["<s>"]] + label + [self.tokeniser["</s>"]]]
        )

    def get_patches(self, img):

        batch_size, height, width = self.tf(img).shape

        img = self.tf(img)
        # Check if image dimensions are divisible by window_size
        assert (
            height % self.window_size == 0 and width % self.window_size == 0
        ), "Height and width must be divisible by the window size."

        # Use unfold to extract patches
        patches = img.unfold(1, self.window_size, self.window_size).unfold(
            2, self.window_size, self.window_size
        )
        # Shape: (batch_size, num_patches_h, num_patches_w, window_size, window_size)

        patches = patches.reshape(batch_size, -1, self.window_size * self.window_size)
        # Shape: (batch_size, num_windows, channels * window_size * window_size)

        return patches


if __name__ == "__main__":
    ds = Combine()
    img, label, target = ds[0]
    print(label)
