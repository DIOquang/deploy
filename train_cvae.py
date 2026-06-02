import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import CSVLogger
from PIL import Image
from transformers import AutoTokenizer, AutoModel

# ==============================================================================
# 1. Dataset & DataLoader
# ==============================================================================
class TextImageDataset(Dataset):
    def __init__(self, metadata_path, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.data = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_name = os.path.basename(item['file_name'])
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        text_prompt = item.get('text', '')
        return image, text_prompt


def get_dataloader(metadata_path, image_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    dataset = TextImageDataset(metadata_path, image_dir, transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, drop_last=True, pin_memory=True
    )
    return dataloader


# ==============================================================================
# 2. cVAE Architecture
# ==============================================================================
LATENT_DIM    = 256
TEXT_EMBED_DIM = 768  # DistilBERT hidden size


class ResBlock(nn.Module):
    """Residual block với GroupNorm để training ổn định hơn."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    """
    Encoder: ảnh 3×128×128 + text_embed → μ và log_σ² (dim=LATENT_DIM)
    Kiến trúc: Conv2D downsampling + text projection được inject vào latent
    """
    def __init__(self, latent_dim=LATENT_DIM, text_embed_dim=TEXT_EMBED_DIM):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_embed_dim, 256),
            nn.SiLU(),
        )

        # Downsampling: 128 → 64 → 32 → 16 → 8
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),      # 64×64
            nn.SiLU(),
            ResBlock(64),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),    # 32×32
            nn.SiLU(),
            ResBlock(128),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),   # 16×16
            nn.SiLU(),
            ResBlock(256),

            nn.Conv2d(256, 512, 3, stride=2, padding=1),   # 8×8
            nn.SiLU(),
            ResBlock(512),
        )

        # Flatten 512×8×8 = 32768, concat với text_proj (256) → fc
        flat_dim = 512 * 8 * 8 + 256
        self.fc_mu     = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, img, text_embed):
        feat = self.enc(img)
        feat = feat.flatten(1)
        text_feat = self.text_proj(text_embed)
        combined = torch.cat([feat, text_feat], dim=1)
        mu     = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder: z (LATENT_DIM) + text_embed → ảnh 3×128×128
    Kiến trúc: Linear projection → reshape → ConvTranspose2D upsampling
    """
    def __init__(self, latent_dim=LATENT_DIM, text_embed_dim=TEXT_EMBED_DIM):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_embed_dim, 256),
            nn.SiLU(),
        )

        # Kết hợp z + text → project lên spatial feature map
        self.fc = nn.Linear(latent_dim + 256, 512 * 8 * 8)

        # Upsampling: 8 → 16 → 32 → 64 → 128
        self.dec = nn.Sequential(
            ResBlock(512),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 16×16
            nn.SiLU(),
            ResBlock(256),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 32×32
            nn.SiLU(),
            ResBlock(128),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 64×64
            nn.SiLU(),
            ResBlock(64),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 128×128
            nn.SiLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, text_embed):
        text_feat = self.text_proj(text_embed)
        combined  = torch.cat([z, text_feat], dim=1)
        out = self.fc(combined)
        out = out.view(-1, 512, 8, 8)
        img = self.dec(out)
        return img


# ==============================================================================
# 3. Lightning Module
# ==============================================================================
class cVAELightning(pl.LightningModule):
    """
    Conditional VAE Lightning Module.

    Loss = Reconstruction (MSE) + β × KL Divergence
    β tăng dần từ 0 → 1 trong 5 epoch đầu (β-annealing)
    để tránh posterior collapse.
    """
    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        lr: float = 1e-4,
        beta_max: float = 1.0,
        warmup_epochs: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

        # Frozen text encoder (DistilBERT)
        self.tokenizer    = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.text_encoder = AutoModel.from_pretrained("distilbert-base-uncased")
        self.text_encoder.eval()
        for p in self.text_encoder.parameters():
            p.requires_grad = False

    # ── Text Encoding ─────────────────────────────────────────────────────────
    def get_text_embeddings(self, texts):
        inputs = self.tokenizer(
            texts, padding=True, truncation=True,
            return_tensors="pt", max_length=77
        ).to(self.device)
        with torch.no_grad():
            out = self.text_encoder(**inputs)
        return out.last_hidden_state[:, 0, :]  # [CLS] token

    # ── Reparameterization Trick ──────────────────────────────────────────────
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # Deterministic during inference

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, img, text_embed):
        mu, logvar = self.encoder(img, text_embed)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decoder(z, text_embed)
        return recon, mu, logvar

    # ── Inference (text-only, sample từ prior) ────────────────────────────────
    @torch.no_grad()
    def generate(self, texts):
        text_embed = self.get_text_embeddings(texts)
        z = torch.randn(len(texts), self.hparams.latent_dim, device=self.device)
        imgs = self.decoder(z, text_embed)
        imgs = imgs * 0.5 + 0.5  # [-1,1] → [0,1]
        return imgs

    # ── Training Step ─────────────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        imgs, texts = batch
        text_embeds = self.get_text_embeddings(texts)

        recon, mu, logvar = self(imgs, text_embeds)

        # Reconstruction loss (MSE trên ảnh đã normalize [-1,1])
        recon_loss = F.mse_loss(recon, imgs, reduction='mean')

        # KL Divergence: -0.5 * sum(1 + logvar - μ² - e^logvar)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # β-annealing: tăng dần β trong warmup_epochs đầu
        current_epoch = self.current_epoch
        beta = min(
            self.hparams.beta_max,
            self.hparams.beta_max * current_epoch / max(1, self.hparams.warmup_epochs)
        )

        loss = recon_loss + beta * kl_loss

        self.log('train_loss',  loss,       prog_bar=True,  on_step=True, on_epoch=True)
        self.log('recon_loss',  recon_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log('kl_loss',     kl_loss,    prog_bar=False, on_step=False, on_epoch=True)
        self.log('beta',        beta,       prog_bar=True,  on_step=False, on_epoch=True)
        return loss

    # ── Optimizer ─────────────────────────────────────────────────────────────
    def configure_optimizers(self):
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        opt = torch.optim.AdamW(params, lr=self.hparams.lr, weight_decay=1e-5)

        # Cosine Annealing LR
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=30, eta_min=1e-6
        )
        return [opt], [scheduler]


# ==============================================================================
# 4. Callback sinh ảnh Validation
# ==============================================================================
class cVAEGenerateCallback(Callback):
    def __init__(self, prompts, output_dir="cvae_val_results", every_n_epochs=5):
        super().__init__()
        self.prompts     = prompts
        self.output_dir  = output_dir
        self.every_n_epochs = every_n_epochs
        os.makedirs(output_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.every_n_epochs == 0:
            print(f"\n[Epoch {epoch}] Sinh ảnh validation cVAE...")
            imgs = pl_module.generate(self.prompts)
            for i, (img_tensor, prompt) in enumerate(zip(imgs, self.prompts)):
                fname = f"cvae_val_epoch_{epoch:02d}_prompt_{i}.png"
                save_image(img_tensor, os.path.join(self.output_dir, fname))
                print(f"  ✔ Saved: {fname}  \"{prompt}\"")


# ==============================================================================
# 5. Hàm sinh ảnh sau khi train xong
# ==============================================================================
def generate_images_post_train(model, output_dir="cvae_results", n=500):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    sample_prompts = [
        "a diamond sword, game icon, glowing cyan aura",
        "a gold potion, game icon, radiant yellow aura",
        "a wood spellbook, game icon, warm brown",
        "a copper axe, game icon, orange-brown copper",
    ]
    print(f"\nBắt đầu sinh {n} ảnh tại {output_dir}...")
    for i in range(n):
        prompt = sample_prompts[i % len(sample_prompts)]
        imgs = model.generate([prompt])
        save_image(imgs[0], os.path.join(output_dir, f"cvae_gen_{i:04d}.png"))
    print(f"✅ Hoàn thành sinh {n} ảnh cVAE!")


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == '__main__':
    METADATA_PATH = "/teamspace/studios/this_studio/hf_dataset/metadata.jsonl"
    IMAGE_DIR     = "/teamspace/studios/this_studio/hf_dataset/images"
    CVAE_CKPT_DIR = "/teamspace/studios/this_studio/cvae_checkpoints"
    CVAE_RESULT   = "/teamspace/studios/this_studio/cvae_results"

    VAL_PROMPTS = [
        "a diamond sword, game icon, glowing cyan aura",
        "a gold potion, game icon, radiant yellow aura",
        "a wood spellbook, game icon, warm brown",
        "a copper axe, game icon, orange-brown copper",
    ]

    dataloader = get_dataloader(METADATA_PATH, IMAGE_DIR, batch_size=32)
    model = cVAELightning(latent_dim=LATENT_DIM, lr=1e-4, beta_max=1.0, warmup_epochs=5)

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=CVAE_CKPT_DIR,
        filename="cvae-{epoch:02d}-{train_loss:.4f}",
        every_n_epochs=5,
        save_top_k=-1,
        save_last=True,
    )
    image_cb = cVAEGenerateCallback(
        prompts=VAL_PROMPTS,
        output_dir="./cvae_val_results",
        every_n_epochs=5,
    )
    csv_logger = CSVLogger(save_dir="./", name="cvae_training_logs")

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",
        devices=1,
        precision="16-mixed",     # fp16 mixed precision
        callbacks=[checkpoint_cb, image_cb],
        logger=csv_logger,
        log_every_n_steps=10,
    )

    print("Bắt đầu Training cVAE...")
    trainer.fit(model, dataloader)

    print("\nTraining hoàn tất! Lưu final model...")
    # Lưu final checkpoint
    final_ckpt = os.path.join(CVAE_CKPT_DIR, "final.ckpt")
    trainer.save_checkpoint(final_ckpt)
    print(f"✅ Final checkpoint lưu tại: {final_ckpt}")

    # Sinh ảnh test
    best_path = trainer.checkpoint_callback.last_model_path
    if best_path and os.path.exists(best_path):
        model = cVAELightning.load_from_checkpoint(best_path)

    model.to("cuda" if torch.cuda.is_available() else "cpu")
    generate_images_post_train(model, output_dir=CVAE_RESULT, n=500)
