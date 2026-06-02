# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  RESUME TRAINING: Text-to-Image Diffusion — Lightning.ai Studio          ║
# ║  - Continuing for 5,000 more steps (Step 10,000 -> 15,000)               ║
# ║  - Feature: Auto-save & auto-load Optimizer/Scheduler states             ║
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
print("✅ Packages checked/installed")

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
TEXT_ENC_ID  = "openai/clip-vit-large-patch14"

# ⚠️ QUAN TRỌNG: Trỏ đến thư mục chứa model bạn vừa train xong 10,000 step
# Nếu bạn lưu tên khác, hãy sửa lại đường dẫn này.
RESUME_FROM_CKPT = f"{OUTPUT_DIR}/final_model" 

RESOLUTION   = 128
BATCH_SIZE   = 32             
GRAD_ACCUM   = 4              
MAX_STEPS    = 5_000          # Số step muốn train THÊM trong lần chạy này
WARMUP_STEPS = 500            # Warmup ngắn để làm nóng optimizer
LEARNING_RATE = 5e-5          # LR giảm so với lúc bắt đầu (1e-4) vì model đã hội tụ một phần
MIXED_PREC   = "bf16"
SEED         = 42
SAVE_EVERY   = 2_500          # Lưu checkpoint mỗi 2500 step
VALIDATE_EVERY = 1_000
LOG_LOSS_EVERY = 100

INITIAL_GLOBAL_STEP = 10_000  # Cột mốc bắt đầu (để ghi log và đặt tên file đúng số)

VAL_PROMPTS = [
    "a diamond sword, game icon, glowing cyan aura",
    "a gold potion, game icon, radiant yellow aura",
    "a wood spellbook, game icon, warm brown",
    "a copper axe, game icon, orange-brown copper",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.bfloat16 if MIXED_PREC == "bf16" else torch.float16

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
        num_workers=8, pin_memory=True, drop_last=True
    )
    print(f"✅ DataLoader: eff. batch = {BATCH_SIZE*GRAD_ACCUM}")


# ── 3. Khởi tạo & Nạp Model cũ ─────────────────────────────────────────────
print("Loading Text Encoder (Frozen)...")
text_enc = CLIPTextModel.from_pretrained(TEXT_ENC_ID).to(device, dtype)
text_enc.requires_grad_(False)

print(f"Loading UNet from checkpoint: {RESUME_FROM_CKPT}...")
unet = UNet2DConditionModel.from_pretrained(RESUME_FROM_CKPT).to(device)
unet.enable_gradient_checkpointing()

noise_sch = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
print(f"✅ UNet loaded successfully: {trainable_params/1e6:.2f} M params")


# ── 4. Khởi tạo & Phục hồi Optimizer / Scheduler ───────────────────────────
use_scaler = (MIXED_PREC == "fp16")
scaler = GradScaler(enabled=use_scaler)
optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

def lr_lambda(step):
    if step < WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

lr_scheduler = LambdaLR(optimizer, lr_lambda)

# Logic tự động load trạng thái (nếu có)
resume_dir = Path(RESUME_FROM_CKPT)
if (resume_dir / "optimizer.bin").exists():
    optimizer.load_state_dict(torch.load(resume_dir / "optimizer.bin", map_location=device))
    print("✅ Restored Optimizer state (Momentum recovered!)")
else:
    print("ℹ️ No optimizer state found in checkpoint. Starting optimizer fresh.")

if use_scaler and (resume_dir / "scaler.bin").exists():
    scaler.load_state_dict(torch.load(resume_dir / "scaler.bin"))
    print("✅ Restored GradScaler state.")


# ── 5. Hàm Validation ──────────────────────────────────────────────────────
@torch.no_grad()
def run_validation(step):
    real_step = step + INITIAL_GLOBAL_STEP
    print(f"\n  ── Validation @ step {real_step} ──")
    
    unet.eval()
    val_dir = Path(OUTPUT_DIR) / f"val_step_{real_step:06d}"
    val_dir.mkdir(exist_ok=True)
    
    val_sch = DDPMScheduler.from_config(noise_sch.config)
    val_sch.set_timesteps(50)
    
    for i, prompt in enumerate(VAL_PROMPTS):
        tokens = tokenizer(
            prompt, padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        ).to(device)
        
        encoder_hidden_states = text_enc(tokens.input_ids)[0]
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
    amp_dtype = torch.bfloat16 if MIXED_PREC == "bf16" else torch.float16

    global_step = 0
    total_loss = 0.0
    unet.train()

    print(f"\n{'='*62}\n  Resuming training for {MAX_STEPS:,} steps (Target: {INITIAL_GLOBAL_STEP + MAX_STEPS:,})\n{'='*62}\n")
    
    loss_file = Path(OUTPUT_DIR) / "loss_log.csv"
    with open(loss_file, "a") as f:
        f.write("\n# Resumed Training Session\n")
        
    pbar = tqdm(total=MAX_STEPS, desc="Resumed Training")

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
                
                with open(loss_file, "a") as f:
                    f.write(f"{global_step + INITIAL_GLOBAL_STEP},{avg:.6f},{lr_u:.6e}\n")
                total_loss = 0.0

            if global_step % VALIDATE_EVERY == 0:
                run_validation(global_step)

            if global_step % SAVE_EVERY == 0:
                real_step = global_step + INITIAL_GLOBAL_STEP
                ckpt = Path(OUTPUT_DIR) / f"checkpoint-{real_step:06d}"
                
                # Lưu model weights
                unet.save_pretrained(str(ckpt))
                
                # LƯU THÊM TRẠNG THÁI OPTIMIZER
                torch.save(optimizer.state_dict(), ckpt / "optimizer.bin")
                torch.save(lr_scheduler.state_dict(), ckpt / "scheduler.bin")
                if use_scaler:
                    torch.save(scaler.state_dict(), ckpt / "scaler.bin")
                    
                print(f"  💾 Saved UNet & Training States to {ckpt}")

    pbar.close()
    
    # Save final model
    final = Path(OUTPUT_DIR) / "final_model_resumed_15k"
    unet.save_pretrained(str(final))
    
    # LƯU THÊM TRẠNG THÁI OPTIMIZER CHO BẢN FINAL
    torch.save(optimizer.state_dict(), final / "optimizer.bin")
    torch.save(lr_scheduler.state_dict(), final / "scheduler.bin")
    if use_scaler:
        torch.save(scaler.state_dict(), final / "scaler.bin")
        
    print(f"\n✅ Training complete! Final model & states saved to {final}")
else:
    print("❌ Cannot train because dataset records were not found.")