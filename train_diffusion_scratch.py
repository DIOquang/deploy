# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Train Text-to-Image Diffusion From Scratch — Lightning.ai Studio        ║
# ║  - Resolution: 128x128                                                   ║
# ║  - Model: UNet2DConditionModel (from scratch)                            ║
# ║  - Text Encoder: Frozen CLIP (pretrained)                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import subprocess, sys

# Install required packages (run once per env)
pkgs = [
    "diffusers==0.30.3",
    "huggingface_hub==0.24.7",
    "transformers==4.41.2",
    "accelerate==0.30.1",
    "safetensors",
    "Pillow",
    "tqdm",
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=True)
print("✅ Packages installed")

import os, json, random, math, torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm

from diffusers import (
    UNet2DConditionModel,
    DDPMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer

# ── 1. Cấu hình ─────────────────────────────────────────────────────────────
DATASET_ROOT = "/teamspace/studios/this_studio/hf_dataset"
OUTPUT_DIR   = "/teamspace/studios/this_studio/diffusion-from-scratch"
TEXT_ENC_ID  = "openai/clip-vit-large-patch14" # Pretrained CLIP model

RESOLUTION   = 128
BATCH_SIZE   = 32             # Chiến lược L4: Batch 32 an toàn không lo OOM
GRAD_ACCUM   = 4              # Gộp gradient 4 lần để giữ nguyên Effective Batch = 128
MAX_STEPS    = 10_000
WARMUP_STEPS = 1_000
LEARNING_RATE = 1e-4
MIXED_PREC   = "bf16"
SEED         = 42
SAVE_EVERY   = 5_000
VALIDATE_EVERY = 1_000
LOG_LOSS_EVERY = 100

VAL_PROMPTS = [
    "a diamond sword, game icon, glowing cyan aura",
    "a gold potion, game icon, radiant yellow aura",
    "a wood spellbook, game icon, warm brown",
    "a copper axe, game icon, orange-brown copper",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.bfloat16 if MIXED_PREC == "bf16" else torch.float16

# Bật TF32 (TensorFloat-32) - Tính năng cực mạnh trên L4/Ada Lovelace giúp tăng tốc matrix multiplication
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(f"✅ Device: {device} | Precision: {MIXED_PREC} | TF32: Enabled")


# ── 2. Dataset & DataLoader ────────────────────────────────────────────────
meta_file = Path(DATASET_ROOT) / "metadata.jsonl"

records = []
if meta_file.exists():
    with open(meta_file, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            path = Path(DATASET_ROOT) / rec["file_name"]
            if path.exists():
                records.append({"image_path": str(path), "text": rec["text"]})
print(f"✅ Dataset: {len(records):,} images found.")

tokenizer = CLIPTokenizer.from_pretrained(TEXT_ENC_ID)

class ScratchIconDataset(Dataset):
    def __init__(self, records, tokenizer, resolution):
        self.records = records
        self.tokenizer = tokenizer
        self.transform = T.Compose([
            T.Resize((resolution, resolution), interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec["image_path"])
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (18, 16, 28))
            bg.paste(img, mask=img.split()[3])
            img = bg
        else:
            img = img.convert("RGB")

        pixel_values = self.transform(img)
        tokens = self.tokenizer(
            rec["text"],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": tokens.input_ids.squeeze(0),
        }

if len(records) > 0:
    train_ds = ScratchIconDataset(records, tokenizer, resolution=RESOLUTION)
    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=8, pin_memory=True, drop_last=True  # L4 Studio thường có CPU mạnh (8-16 cores)
    )
    print(f"✅ DataLoader: eff. batch = {BATCH_SIZE*GRAD_ACCUM}")


# ── 3. Khởi tạo Models ─────────────────────────────────────────────────────
print("Loading Text Encoder (Frozen)...")
text_enc = CLIPTextModel.from_pretrained(TEXT_ENC_ID).to(device, dtype)
text_enc.requires_grad_(False)

print("Initializing UNet from scratch...")
# Architecture optimized for 128x128 Pixel-space processing
unet = UNet2DConditionModel(
    sample_size=RESOLUTION,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 256, 384, 512),
    down_block_types=(
        "DownBlock2D",           # 128x128 -> 64x64
        "CrossAttnDownBlock2D",  # 64x64 -> 32x32
        "CrossAttnDownBlock2D",  # 32x32 -> 16x16
        "CrossAttnDownBlock2D",  # 16x16 -> 16x16 (bottom)
    ),
    up_block_types=(
        "CrossAttnUpBlock2D",    # 16x16 -> 32x32
        "CrossAttnUpBlock2D",    # 32x32 -> 64x64
        "CrossAttnUpBlock2D",    # 64x64 -> 128x128
        "UpBlock2D",             # 128x128 -> 128x128
    ),
    cross_attention_dim=text_enc.config.hidden_size, # 768 for CLIP ViT-L
).to(device)

unet.enable_gradient_checkpointing()
print("Using native PyTorch 2.0 SDPA for acceleration.")
# unet = torch.compile(unet)  # Tạm tắt torch.compile vì gây lỗi CUDA Graphs với Diffusers

noise_sch = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")

trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
print(f"✅ UNet initialized: {trainable_params/1e6:.2f} M params")


# ── 4. Optimizer & Scheduler ───────────────────────────────────────────────
optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

def lr_lambda(step):
    if step < WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

lr_scheduler = LambdaLR(optimizer, lr_lambda)


# ── 5. Hàm Validation (Sinh ảnh) ───────────────────────────────────────────
@torch.no_grad()
def run_validation(step):
    print(f"\n  ── Validation @ step {step} ──")
    
    # Cài đặt inference Pipeline trực tiếp cho pixel-space 
    unet.eval()
    val_dir = Path(OUTPUT_DIR) / f"val_step_{step:06d}"
    val_dir.mkdir(exist_ok=True)
    
    val_sch = DDPMScheduler.from_config(noise_sch.config)
    val_sch.set_timesteps(50)
    
    for i, prompt in enumerate(VAL_PROMPTS):
        tokens = tokenizer(
            prompt, padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        ).to(device)
        
        encoder_hidden_states = text_enc(tokens.input_ids)[0]
        
        # Classifier-free guidance rỗng
        uncond_tokens = tokenizer(
            "", padding="max_length", max_length=tokenizer.model_max_length,
            return_tensors="pt"
        ).to(device)
        uncond_hidden = text_enc(uncond_tokens.input_ids)[0]
        
        latents = torch.randn(
            (1, unet.config.in_channels, RESOLUTION, RESOLUTION), 
            device=device, dtype=dtype
        )
        
        for t in val_sch.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = val_sch.scale_model_input(latent_model_input, t)
            
            with torch.amp.autocast("cuda", dtype=dtype):
                noise_pred = unet(
                    latent_model_input, t, 
                    encoder_hidden_states=torch.cat([uncond_hidden, encoder_hidden_states])
                ).sample
                
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
            
            latents = val_sch.step(noise_pred, t, latents).prev_sample
            
        # Hậu xử lý về ảnh RGB
        image = (latents / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
        image = (image * 255).round().astype("uint8")
        pil_img = Image.fromarray(image)
        
        out = val_dir / f"val_{i:02d}.png"
        pil_img.save(out)
        print(f"    ✔ {out.name}  \"{prompt}\"")
        
    unet.train()
    print()


# ── 6. Vòng lặp Training ───────────────────────────────────────────────────
if len(records) > 0:
    use_scaler = (MIXED_PREC == "fp16")
    scaler = GradScaler(enabled=use_scaler)
    amp_dtype = torch.bfloat16 if MIXED_PREC == "bf16" else torch.float16

    global_step = 0
    total_loss = 0.0
    unet.train()

    print(f"\n{'='*62}\n  Training start  —  {MAX_STEPS:,} steps\n{'='*62}\n")
    
    # Tạo file log loss
    loss_file = Path(OUTPUT_DIR) / "loss_log.csv"
    with open(loss_file, "w") as f:
        f.write("step,loss,learning_rate\n")
        
    pbar = tqdm(total=MAX_STEPS, desc="Training")

    while global_step < MAX_STEPS:
        for batch in train_dl:
            if global_step >= MAX_STEPS:
                break
                
            pixel_values = batch["pixel_values"].to(device, dtype=amp_dtype)
            input_ids = batch["input_ids"].to(device)

            noise = torch.randn_like(pixel_values)
            timesteps = torch.randint(
                0, noise_sch.config.num_train_timesteps,
                (pixel_values.shape[0],), device=device
            ).long()
            noisy_images = noise_sch.add_noise(pixel_values, noise, timesteps)

            with torch.no_grad():
                encoder_hidden_states = text_enc(input_ids)[0]
                
            # CFG Dropout (10% unconditioned)
            if random.random() < 0.1:
                uncond_tokens = tokenizer([""] * input_ids.shape[0], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").to(device)
                encoder_hidden_states = text_enc(uncond_tokens.input_ids)[0]

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                noise_pred = unet(noisy_images, timesteps, encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float()) / GRAD_ACCUM

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * GRAD_ACCUM

            if (global_step + 1) % GRAD_ACCUM == 0:
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                    
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            pbar.update(1)

            if global_step % LOG_LOSS_EVERY == 0:
                avg = total_loss / LOG_LOSS_EVERY
                lr_u = optimizer.param_groups[0]["lr"]
                pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr_u:.2e}")
                
                # Lưu loss vào file
                with open(loss_file, "a") as f:
                    f.write(f"{global_step},{avg:.6f},{lr_u:.6e}\n")
                    
                total_loss = 0.0

            if global_step % VALIDATE_EVERY == 0:
                run_validation(global_step)

            if global_step % SAVE_EVERY == 0:
                ckpt = Path(OUTPUT_DIR) / f"checkpoint-{global_step:06d}"
                unet.save_pretrained(str(ckpt))
                print(f"  💾 Saved UNet to {ckpt}")

    pbar.close()
    
    # Save final model
    final = Path(OUTPUT_DIR) / "final_model"
    unet.save_pretrained(str(final))
    print(f"\n✅ Training complete! Final model saved to {final}")
else:
    print("❌ Cannot train because dataset records were not found.")
