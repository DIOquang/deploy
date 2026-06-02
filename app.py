import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer, AutoModel
from PIL import Image
import numpy as np
import time
import traceback

# ── 1. Cấu hình & Đường dẫn ────────────────────────────────────────────────
DIFFUSION_MODEL_DIR = "/teamspace/studios/this_studio/diffusion-from-scratch/final_model"
CGAN_CKPT_PATH      = "/teamspace/studios/this_studio/cgan_checkpoints/last.ckpt"
CVAE_CKPT_PATH      = "/teamspace/studios/this_studio/cvae_checkpoints/final.ckpt"

TEXT_ENC_ID = "openai/clip-vit-large-patch14"
RESOLUTION  = 128
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fp16_dtype  = torch.float16

print(f"🖥️  Device: {device}")

# ── 2. Định nghĩa lại kiến trúc cGAN & cVAE (để load checkpoint) ──────────

# -------- cGAN --------
class CGANGenerator(nn.Module):
    def __init__(self, latent_dim=100, text_embed_dim=768, img_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.text_proj  = nn.Linear(text_embed_dim, 128)
        self.init_size  = 128 // 16
        self.l1 = nn.Sequential(nn.Linear(latent_dim + 128, 512 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, 3, stride=1, padding=1), nn.BatchNorm2d(256, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),  nn.BatchNorm2d(64, 0.8),  nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, img_channels, 3, stride=1, padding=1), nn.Tanh(),
        )

    def forward(self, noise, text_embeddings):
        c = F.leaky_relu(self.text_proj(text_embeddings), 0.2)
        gen_input = torch.cat((noise, c), -1)
        out = self.l1(gen_input).view(out.shape[0], 512, self.init_size, self.init_size) if False else \
              self.l1(gen_input)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        return self.conv_blocks(out)


# -------- cVAE (Decoder only cần thiết cho inference) --------
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
    def forward(self, x): return x + self.block(x)


class CVAEDecoder(nn.Module):
    def __init__(self, latent_dim=256, text_embed_dim=768):
        super().__init__()
        self.text_proj = nn.Sequential(nn.Linear(text_embed_dim, 256), nn.SiLU())
        self.fc = nn.Linear(latent_dim + 256, 512 * 8 * 8)
        self.dec = nn.Sequential(
            ResBlock(512),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.SiLU(), ResBlock(256),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.SiLU(), ResBlock(128),
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1), nn.SiLU(), ResBlock(64),
            nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 3, 3, padding=1), nn.Tanh(),
        )
    def forward(self, z, text_embed):
        t = self.text_proj(text_embed)
        out = self.fc(torch.cat([z, t], dim=1)).view(-1, 512, 8, 8)
        return self.dec(out)


# ── 3. Load Models ──────────────────────────────────────────────────────────
print("Đang tải models...")

# 3a. Diffusion
diffusion_error = None
try:
    clip_tokenizer = CLIPTokenizer.from_pretrained(TEXT_ENC_ID)
    clip_text_enc  = CLIPTextModel.from_pretrained(TEXT_ENC_ID).to(device, fp16_dtype)
    clip_text_enc.eval()
    unet = UNet2DConditionModel.from_pretrained(DIFFUSION_MODEL_DIR).to(device, fp16_dtype)
    unet.eval()
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    print("✅ Diffusion model loaded.")
except Exception as e:
    diffusion_error = traceback.format_exc()
    print(f"⚠️  Diffusion load failed: {e}")
    unet = clip_text_enc = clip_tokenizer = scheduler = None

# 3b. cGAN
cgan_error = None
cgan_generator = None
try:
    # Load Lightning checkpoint
    import pytorch_lightning as pl
    from train_cgan import cGANLightning
    cgan_module = cGANLightning.load_from_checkpoint(CGAN_CKPT_PATH, map_location=device)
    cgan_generator  = cgan_module.generator.to(device).eval()
    cgan_tokenizer  = cgan_module.tokenizer
    cgan_text_enc   = cgan_module.text_encoder.to(device).eval()
    print("✅ cGAN model loaded.")
except Exception as e:
    cgan_error = traceback.format_exc()
    print(f"⚠️  cGAN load failed: {e}")

# 3c. cVAE
cvae_error = None
cvae_decoder = None
try:
    from train_cvae import cVAELightning
    cvae_module  = cVAELightning.load_from_checkpoint(CVAE_CKPT_PATH, map_location=device)
    cvae_decoder = cvae_module.decoder.to(device).eval()
    cvae_text_enc_module = cvae_module.text_encoder.to(device).eval()
    cvae_tokenizer_obj   = cvae_module.tokenizer
    print("✅ cVAE model loaded.")
except Exception as e:
    cvae_error = traceback.format_exc()
    print(f"⚠️  cVAE load failed: {e}")

print("🚀 Khởi tạo giao diện...")


# ── 4. Helper: encode text cho DistilBERT ──────────────────────────────────
def encode_distilbert(tokenizer_obj, text_enc_model, prompt, device):
    inputs = tokenizer_obj(
        [prompt], padding=True, truncation=True,
        return_tensors="pt", max_length=77
    ).to(device)
    with torch.no_grad():
        out = text_enc_model(**inputs)
    return out.last_hidden_state[:, 0, :]


# ── 5. Inference Functions ──────────────────────────────────────────────────

@torch.no_grad()
def run_diffusion(prompt, num_steps, guidance_scale, seed):
    """Sinh ảnh bằng Diffusion Model. Trả về (PIL.Image, float, str|None)."""
    if unet is None:
        return None, 0.0, f"❌ Diffusion chưa load được model:\n{diffusion_error}"

    t0 = time.time()
    if seed != -1:
        torch.manual_seed(int(seed))

    scheduler.set_timesteps(num_steps)

    tokens = clip_tokenizer(
        prompt, padding="max_length", max_length=clip_tokenizer.model_max_length,
        truncation=True, return_tensors="pt"
    ).to(device)
    enc_hidden = clip_text_enc(tokens.input_ids)[0]

    uncond_tokens = clip_tokenizer(
        "", padding="max_length", max_length=clip_tokenizer.model_max_length,
        return_tensors="pt"
    ).to(device)
    uncond_hidden = clip_text_enc(uncond_tokens.input_ids)[0]

    latents = torch.randn(
        (1, unet.config.in_channels, RESOLUTION, RESOLUTION),
        device=device, dtype=fp16_dtype
    )

    for t in scheduler.timesteps:
        latent_input = torch.cat([latents] * 2)
        latent_input = scheduler.scale_model_input(latent_input, t)
        with torch.amp.autocast("cuda", dtype=fp16_dtype):
            noise_pred = unet(
                latent_input, t,
                encoder_hidden_states=torch.cat([uncond_hidden, enc_hidden])
            ).sample
        noise_uncond, noise_cond = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    img = (latents / 2 + 0.5).clamp(0, 1)
    img = img.cpu().permute(0, 2, 3, 1).float().numpy()[0]
    img = (img * 255).round().astype("uint8")
    elapsed = time.time() - t0
    return Image.fromarray(img), elapsed, None


@torch.no_grad()
def run_cgan(prompt, seed):
    """Sinh ảnh bằng cGAN. Trả về (PIL.Image, float, str|None)."""
    if cgan_generator is None:
        return None, 0.0, f"❌ cGAN chưa load được model:\n{cgan_error}"

    t0 = time.time()
    if seed != -1:
        torch.manual_seed(int(seed))

    text_embed = encode_distilbert(cgan_tokenizer, cgan_text_enc, prompt, device)
    z = torch.randn(1, 100, device=device)
    gen_img = cgan_generator(z, text_embed)
    gen_img = (gen_img * 0.5 + 0.5).clamp(0, 1)

    img = gen_img.cpu().permute(0, 2, 3, 1).float().numpy()[0]
    img = (img * 255).round().astype("uint8")
    elapsed = time.time() - t0
    return Image.fromarray(img), elapsed, None


@torch.no_grad()
def run_cvae(prompt, seed):
    """Sinh ảnh bằng cVAE. Trả về (PIL.Image, float, str|None)."""
    if cvae_decoder is None:
        return None, 0.0, f"❌ cVAE chưa load được model:\n{cvae_error}"

    t0 = time.time()
    if seed != -1:
        torch.manual_seed(int(seed))

    text_embed = encode_distilbert(cvae_tokenizer_obj, cvae_text_enc_module, prompt, device)
    z = torch.randn(1, 256, device=device)
    gen_img = cvae_decoder(z, text_embed)
    gen_img = (gen_img * 0.5 + 0.5).clamp(0, 1)

    img = gen_img.cpu().permute(0, 2, 3, 1).float().numpy()[0]
    img = (img * 255).round().astype("uint8")
    elapsed = time.time() - t0
    return Image.fromarray(img), elapsed, None


# ── 6. Hàm chính gọi cả 3 model ────────────────────────────────────────────
def generate_all(prompt, num_steps, guidance_scale, seed):
    """
    Gọi 3 model song song (tuần tự) và trả về:
    img_diff, time_diff, err_diff,
    img_cgan, time_cgan, err_cgan,
    img_cvae, time_cvae, err_cvae
    """
    img_diff, t_diff, err_diff = run_diffusion(prompt, num_steps, guidance_scale, seed)
    img_cgan, t_cgan, err_cgan = run_cgan(prompt, seed)
    img_cvae, t_cvae, err_cvae = run_cvae(prompt, seed)

    # Format thời gian
    label_diff = f"⏱️ {t_diff:.2f}s" if err_diff is None else f"⚠️ Lỗi"
    label_cgan = f"⏱️ {t_cgan:.2f}s" if err_cgan is None else f"⚠️ Lỗi"
    label_cvae = f"⏱️ {t_cvae:.2f}s" if err_cvae is None else f"⚠️ Lỗi"

    # Error message tổng hợp
    errors = []
    if err_diff: errors.append(f"**Diffusion:**\n```\n{err_diff}\n```")
    if err_cgan: errors.append(f"**cGAN:**\n```\n{err_cgan}\n```")
    if err_cvae: errors.append(f"**cVAE:**\n```\n{err_cvae}\n```")
    error_msg = "\n\n".join(errors) if errors else ""

    return (
        img_diff, label_diff,
        img_cgan, label_cgan,
        img_cvae, label_cvae,
        error_msg,
    )


# ── 7. Giao diện Gradio ─────────────────────────────────────────────────────
css = """
.model-col { border: 1px solid #2d2d2d; border-radius: 12px; padding: 12px; background: #1a1a2e; }
.time-badge { text-align: center; font-size: 1.1em; font-weight: bold; color: #00d4ff; margin-top: 6px; }
.error-box textarea { font-family: monospace; font-size: 0.82em; color: #ff6b6b; background: #1a1a1a; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), css=css) as demo:
    gr.Markdown("""
    # 🎨 Text-to-Image: So sánh 3 Model Deep Learning
    **Diffusion (UNet)** &nbsp;|&nbsp; **cGAN (Conditional GAN)** &nbsp;|&nbsp; **cVAE (Conditional VAE)**
    
    Nhập prompt → Sinh ảnh từ cả 3 model và so sánh kết quả + thời gian.
    """)

    # ── Input ──────────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=3):
            prompt_input = gr.Textbox(
                label="📝 Prompt (Mô tả ảnh)",
                placeholder="VD: a diamond sword, game icon, glowing cyan aura...",
                lines=2,
            )
        with gr.Column(scale=1):
            seed_input = gr.Number(label="🎲 Seed (-1 = ngẫu nhiên)", value=-1, precision=0)

    with gr.Row():
        num_steps_input = gr.Slider(
            minimum=10, maximum=100, value=50, step=1,
            label="🔁 Số bước khử nhiễu (chỉ cho Diffusion)"
        )
        guidance_input = gr.Slider(
            minimum=1.0, maximum=15.0, value=7.5, step=0.5,
            label="🎯 CFG Scale (chỉ cho Diffusion)"
        )

    generate_btn = gr.Button("🚀 Sinh Ảnh (Cả 3 Model)", variant="primary", size="lg")

    gr.Markdown("---")

    # ── Output: 3 cột ──────────────────────────────────────────────────────
    with gr.Row(equal_height=True):
        with gr.Column(elem_classes="model-col"):
            gr.Markdown("### 🌊 Diffusion Model")
            diff_img   = gr.Image(label="Kết quả", type="pil", interactive=False, height=256)
            diff_time  = gr.Textbox(label="⏱️ Thời gian sinh", interactive=False, elem_classes="time-badge")

        with gr.Column(elem_classes="model-col"):
            gr.Markdown("### ⚡ cGAN")
            cgan_img   = gr.Image(label="Kết quả", type="pil", interactive=False, height=256)
            cgan_time  = gr.Textbox(label="⏱️ Thời gian sinh", interactive=False, elem_classes="time-badge")

        with gr.Column(elem_classes="model-col"):
            gr.Markdown("### 🧬 cVAE")
            cvae_img   = gr.Image(label="Kết quả", type="pil", interactive=False, height=256)
            cvae_time  = gr.Textbox(label="⏱️ Thời gian sinh", interactive=False, elem_classes="time-badge")

    # ── Error box ──────────────────────────────────────────────────────────
    error_output = gr.Markdown(
        label="⚠️ Lỗi chi tiết (nếu có)",
        visible=True,
        value="",
    )

    # ── Kết nối button ─────────────────────────────────────────────────────
    generate_btn.click(
        fn=generate_all,
        inputs=[prompt_input, num_steps_input, guidance_input, seed_input],
        outputs=[
            diff_img, diff_time,
            cgan_img, cgan_time,
            cvae_img, cvae_time,
            error_output,
        ],
    )

    # ── Prompt gợi ý ───────────────────────────────────────────────────────
    gr.Examples(
        examples=[
            ["a diamond sword, game icon, glowing cyan aura",    50, 7.5, 42],
            ["a gold potion, game icon, radiant yellow aura",    50, 7.5, 42],
            ["a wood spellbook, game icon, warm brown texture",  50, 7.5, 42],
            ["a copper axe, game icon, orange-brown copper ore", 50, 7.5, 42],
        ],
        inputs=[prompt_input, num_steps_input, guidance_input, seed_input],
        label="💡 Prompt mẫu (Click để dùng)",
    )

    gr.Markdown("""
    ---
    **Ghi chú:**
    - Diffusion dùng **CLIP** + UNet tùy chỉnh, mất nhiều thời gian hơn (nhiều bước khử nhiễu).
    - cGAN & cVAE dùng **DistilBERT** làm text encoder, sinh ảnh nhanh hơn (1 forward pass).
    - Nếu model chưa được train, sẽ hiển thị lỗi chi tiết bên dưới.
    """)


# ── 8. Chạy server ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(share=True)