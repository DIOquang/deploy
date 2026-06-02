import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from collections import Counter
from tqdm import tqdm

# ==============================================================================
# 1. Cấu hình
# ==============================================================================
BASE_PATH   = "/teamspace/studios/this_studio/hf_dataset"
IMAGE_DIR   = os.path.join(BASE_PATH, "images")
METADATA_PATH = os.path.join(BASE_PATH, "metadata.jsonl")
OUTPUT_DIR  = "/teamspace/studios/this_studio/cvae_results"

LATENT_DIM  = 128
COND_DIM    = 128
MAX_SEQ_LEN = 15
BATCH_SIZE  = 64
EPOCHS      = 60
LR          = 1e-3
BETA        = 0.5          # KL weight

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==============================================================================
# 2. Đọc metadata & xây Vocabulary
# ==============================================================================
data_records = []
all_words    = []

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        img_name = os.path.basename(item.get("file_name", ""))
        prompt   = item.get("text", "").lower()
        words    = re.findall(r"\b\w+\b", prompt)
        all_words.extend(words)
        data_records.append({
            "img_path": os.path.join(IMAGE_DIR, img_name),
            "prompt":   prompt,
            "words":    words,
        })

word_counts = Counter(all_words)
vocab       = {word: i + 1 for i, (word, _) in enumerate(word_counts.items())}  # 0 = padding
VOCAB_SIZE  = len(vocab) + 1
print(f"Vocabulary size: {VOCAB_SIZE}  |  Dataset size: {len(data_records)}")


# ==============================================================================
# 3. Dataset
# ==============================================================================
class GameIconDataset(Dataset):
    def __init__(self, records, vocab, transform=None):
        self.records   = records
        self.vocab     = vocab
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        img    = Image.open(record["img_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        tokens  = [self.vocab.get(w, 0) for w in record["words"]][:MAX_SEQ_LEN]
        tokens += [0] * (MAX_SEQ_LEN - len(tokens))
        return img, torch.tensor(tokens, dtype=torch.long)


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),          # scale [0, 1]
])

dataset    = GameIconDataset(data_records, vocab, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


# ==============================================================================
# 4. Kiến trúc cVAE (khớp với cvae_best.pth)
# ==============================================================================
class TextEncoder(nn.Module):
    """Embedding đơn giản → mean pooling."""
    def __init__(self, vocab_size, embed_dim=COND_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        embedded = self.embedding(x)       # (B, seq_len, embed_dim)
        return embedded.mean(dim=1)        # (B, embed_dim)


class cVAE(nn.Module):
    def __init__(self, vocab_size, latent_dim=LATENT_DIM, cond_dim=COND_DIM):
        super().__init__()
        self.latent_dim = latent_dim

        # Text encoder
        self.text_encoder = TextEncoder(vocab_size, cond_dim)

        # Encoder: 3×128×128 → 256×8×8
        self.enc_conv = nn.Sequential(
            nn.Conv2d(3,   32,  4, 2, 1), nn.ReLU(),   # 64×64
            nn.Conv2d(32,  64,  4, 2, 1), nn.ReLU(),   # 32×32
            nn.Conv2d(64,  128, 4, 2, 1), nn.ReLU(),   # 16×16
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),   #  8×8
        )
        self.flatten_size = 256 * 8 * 8   # 16 384

        self.fc_mu     = nn.Linear(self.flatten_size + cond_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size + cond_dim, latent_dim)

        # Decoder: (z + cond) → 3×128×128
        self.fc_dec = nn.Linear(latent_dim + cond_dim, self.flatten_size)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),      # 16×16
            nn.ConvTranspose2d(128,  64, 4, 2, 1), nn.ReLU(),      # 32×32
            nn.ConvTranspose2d( 64,  32, 4, 2, 1), nn.ReLU(),      # 64×64
            nn.ConvTranspose2d( 32,   3, 4, 2, 1), nn.Sigmoid(),   # 128×128
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c_text):
        c_vec   = self.text_encoder(c_text)          # (B, cond_dim)
        x_enc   = self.enc_conv(x)
        x_flat  = x_enc.view(x_enc.size(0), -1)
        enc_in  = torch.cat([x_flat, c_vec], dim=1)
        mu      = self.fc_mu(enc_in)
        logvar  = self.fc_logvar(enc_in)
        z       = self.reparameterize(mu, logvar)
        dec_in  = torch.cat([z, c_vec], dim=1)
        x_dec   = self.fc_dec(dec_in).view(-1, 256, 8, 8)
        out     = self.dec_conv(x_dec)
        return out, mu, logvar

    @torch.no_grad()
    def generate(self, tokens_tensor):
        """Sinh ảnh từ token tensor (B, MAX_SEQ_LEN)."""
        self.eval()
        c_vec   = self.text_encoder(tokens_tensor)
        z       = torch.randn(tokens_tensor.size(0), self.latent_dim, device=tokens_tensor.device)
        dec_in  = torch.cat([z, c_vec], dim=1)
        x_dec   = self.fc_dec(dec_in).view(-1, 256, 8, 8)
        return self.dec_conv(x_dec)


# ==============================================================================
# 5. Loss Function
# ==============================================================================
def cvae_loss_fn(recon_x, x, mu, logvar, beta=BETA):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
    kl_loss    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss


# ==============================================================================
# 6. Helper: tokenize prompt thành tensor
# ==============================================================================
def tokenize(prompt, vocab, max_len=MAX_SEQ_LEN):
    words  = re.findall(r"\b\w+\b", prompt.lower())
    tokens = [vocab.get(w, 0) for w in words][:max_len]
    tokens += [0] * (max_len - len(tokens))
    return tokens


# ==============================================================================
# 7. Training
# ==============================================================================
if __name__ == "__main__":
    VAL_PROMPTS = [
        "a diamond sword, game icon, glowing cyan aura",
        "a gold potion, game icon, radiant yellow aura",
        "a wood spellbook, game icon, warm brown",
        "a copper axe, game icon, orange-brown copper",
    ]

    model     = cVAE(vocab_size=VOCAB_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_loss  = float("inf")
    best_path  = os.path.join(OUTPUT_DIR, "cvae_best.pth")
    val_dir    = os.path.join(OUTPUT_DIR, "val_images")
    os.makedirs(val_dir, exist_ok=True)

    print("Bắt đầu Training cVAE...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, text_tokens in pbar:
            images      = images.to(device)
            text_tokens = text_tokens.to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(images, text_tokens)
            loss = cvae_loss_fn(recon, images, mu, logvar)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item()/len(images):.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"→ Epoch [{epoch+1}/{EPOCHS}] | Avg Loss: {avg_loss:.4f}")

        # Lưu best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_path)
            print(f"  ✅ Saved best model (loss={best_loss:.4f})")

        # Validation mỗi 5 epoch
        if (epoch + 1) % 5 == 0:
            model.eval()
            print(f"  [Val] Sinh ảnh tại Epoch {epoch+1}...")
            for i, prompt in enumerate(VAL_PROMPTS):
                tokens = tokenize(prompt, vocab)
                t      = torch.tensor([tokens], dtype=torch.long, device=device)
                img    = model.generate(t)
                fname  = os.path.join(val_dir, f"val_epoch{epoch+1:02d}_p{i}.png")
                save_image(img, fname)
            model.train()

    print(f"\n✅ Training hoàn tất! Best model: {best_path}")
