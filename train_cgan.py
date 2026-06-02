import os
# Tắt cảnh báo kẹt đa luồng của Tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import CSVLogger  # <-- Đã thêm thư viện lưu CSV
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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    return dataloader

# ==============================================================================
# 2. Generator & Discriminator Models
# ==============================================================================
class Generator(nn.Module):
    def __init__(self, latent_dim=100, text_embed_dim=768, img_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.text_proj = nn.Linear(text_embed_dim, 128)
        
        self.init_size = 128 // 16 # 8x8
        self.l1 = nn.Sequential(nn.Linear(latent_dim + 128, 512 * self.init_size ** 2))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor=2), # 16x16
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2), # 32x32
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2), # 64x64
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2), # 128x128
            nn.Conv2d(64, img_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, text_embeddings):
        c = F.leaky_relu(self.text_proj(text_embeddings), 0.2)
        gen_input = torch.cat((noise, c), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, text_embed_dim=768, img_channels=3):
        super().__init__()
        self.text_proj = nn.Linear(text_embed_dim, 128)
        
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(img_channels + 1, 64, bn=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
        )
        self.adv_layer = nn.Sequential(nn.Linear(512 * 8 * 8, 1), nn.Sigmoid())

    def forward(self, img, text_embeddings):
        c = F.leaky_relu(self.text_proj(text_embeddings), 0.2)
        c_expanded = c.view(-1, 128, 1, 1).expand(-1, 128, 128, 128)
        c_mask = c_expanded.mean(dim=1, keepdim=True)
        d_in = torch.cat((img, c_mask), 1)
        out = self.model(d_in)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

# ==============================================================================
# 3. Lightning Module (Sử dụng TTUR: LR của D chậm hơn G)
# ==============================================================================
class cGANLightning(pl.LightningModule):
    def __init__(self, latent_dim=100, lr_g=0.0002, lr_d=0.00005, b1=0.5, b2=0.999):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        self.generator = Generator(latent_dim=self.hparams.latent_dim)
        self.discriminator = Discriminator()
        
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.text_encoder = AutoModel.from_pretrained("distilbert-base-uncased")
        self.text_encoder.eval()
        for param in self.text_encoder.parameters():
            param.requires_grad = False 

        self.adversarial_loss = nn.BCELoss()

    def forward(self, z, text_embeddings):
        return self.generator(z, text_embeddings)
        
    def get_text_embeddings(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=77).to(self.device)
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
        return outputs.last_hidden_state[:, 0, :]

    def training_step(self, batch, batch_idx):
        imgs, texts = batch
        text_embeds = self.get_text_embeddings(texts)
        
        opt_g, opt_d = self.optimizers()
        valid = torch.ones(imgs.size(0), 1, device=self.device)
        fake = torch.zeros(imgs.size(0), 1, device=self.device)
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim, device=self.device)

        # Train G
        self.toggle_optimizer(opt_g)
        gen_imgs = self(z, text_embeds)
        g_loss = self.adversarial_loss(self.discriminator(gen_imgs, text_embeds), valid)
        self.manual_backward(g_loss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        # Train D
        self.toggle_optimizer(opt_d)
        real_loss = self.adversarial_loss(self.discriminator(imgs, text_embeds), valid)
        fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach(), text_embeds), fake)
        d_loss = (real_loss + fake_loss) / 2
        self.manual_backward(d_loss)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

        # Log metrics (sẽ tự động ghi vào CSVLogger)
        self.log('g_loss', g_loss, prog_bar=True)
        self.log('d_loss', d_loss, prog_bar=True)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr_g, betas=(self.hparams.b1, self.hparams.b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr_d, betas=(self.hparams.b1, self.hparams.b2))
        return [opt_g, opt_d], []

# ==============================================================================
# 4. Callback sinh ảnh Validation 
# ==============================================================================
class GenerateImagesCallback(Callback):
    def __init__(self, prompts, output_dir="val_results", every_n_epochs=5):
        super().__init__()
        self.prompts = prompts
        self.output_dir = output_dir
        self.every_n_epochs = every_n_epochs
        os.makedirs(self.output_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1 
        if epoch % self.every_n_epochs == 0:
            print(f"\n[Epoch {epoch}] Sinh ảnh validation cGAN...")
            pl_module.eval() 
            with torch.no_grad():
                for i, prompt in enumerate(self.prompts):
                    text_embed = pl_module.get_text_embeddings([prompt])
                    z = torch.randn(1, pl_module.hparams.latent_dim, device=pl_module.device)
                    gen_img = pl_module(z, text_embed)
                    gen_img = gen_img * 0.5 + 0.5 
                    img_name = f"cgan_val_epoch_{epoch:02d}_prompt_{i}.png"
                    save_image(gen_img, os.path.join(self.output_dir, img_name))
            pl_module.train() 

# ==============================================================================
# 5. Hàm sinh 500 ảnh sau khi train
# ==============================================================================
def generate_500_images(model, output_dir="cgan_results"):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    sample_prompts = [
        "a diamond sword, game icon, glowing cyan aura",
        "a gold potion, game icon, radiant yellow aura",
        "a wood spellbook, game icon, warm brown",
        "a copper axe, game icon, orange-brown copper"
    ]
    print(f"\nBắt đầu sinh 500 ảnh tại {output_dir}...")
    with torch.no_grad():
        for i in range(500):
            prompt = sample_prompts[i % len(sample_prompts)]
            z = torch.randn(1, model.hparams.latent_dim, device=model.device)
            text_embed = model.get_text_embeddings([prompt])
            gen_img = model(z, text_embed)
            gen_img = gen_img * 0.5 + 0.5 
            save_image(gen_img, os.path.join(output_dir, f"cgan_gen_{i:04d}.png"))
    print("Hoàn thành sinh ảnh cGAN!")

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == '__main__':
    METADATA_PATH = "/teamspace/studios/this_studio/hf_dataset/metadata.jsonl"
    IMAGE_DIR = "/teamspace/studios/this_studio/hf_dataset/images"
    
    VAL_PROMPTS = [
        "a diamond sword, game icon, glowing cyan aura",
        "a gold potion, game icon, radiant yellow aura",
        "a wood spellbook, game icon, warm brown",
        "a copper axe, game icon, orange-brown copper",
    ]
    
    dataloader = get_dataloader(METADATA_PATH, IMAGE_DIR, batch_size=32)
    model = cGANLightning(latent_dim=100)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="./cgan_checkpoints",
        filename="cgan-{epoch:02d}", 
        every_n_epochs=5,
        save_top_k=-1, 
        save_last=True 
    )
    
    image_callback = GenerateImagesCallback(
        prompts=VAL_PROMPTS, 
        output_dir="./val_results", 
        every_n_epochs=5
    )
    
    # <-- KHỞI TẠO CSV LOGGER Ở ĐÂY -->
    csv_logger = CSVLogger(save_dir="./", name="cgan_training_logs")
    
    trainer = pl.Trainer(
        max_epochs=30, 
        accelerator="auto", 
        devices=1,
        callbacks=[checkpoint_callback, image_callback],
        logger=csv_logger, # <-- KẾT NỐI LOGGER VÀO TRAINER
    )
    
    print("Bắt đầu Training cGAN...")
    trainer.fit(model, dataloader)
    
    print("\nTraining hoàn tất! Chạy sinh 500 ảnh test...")
    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        model = cGANLightning.load_from_checkpoint(best_model_path)
    
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    generate_500_images(model, output_dir="/teamspace/studios/this_studio/cgan_results")