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
import json
import re
import os
from collections import Counter

# ── 1. Cấu hình đường dẫn ────────────────────────────────────────────────────
DIFFUSION_MODEL_DIR = "/teamspace/studios/this_studio/diffusion-from-scratch/final_model"
METADATA_PATH       = "/teamspace/studios/this_studio/hf_dataset/metadata.jsonl"
CVAE_CKPT_PATH      = os.path.join(os.path.dirname(__file__), "cvae_best.pth")

# Thử load checkpoint cGAN từ epoch sớm nhất (tránh mode collapse ở epoch 30)
# Lightning lưu theo format: cgan-epoch=04.ckpt (0-indexed: epoch=04 = training epoch 5)
CGAN_CKPT_BASE = "/teamspace/studios/this_studio/cgan_checkpoints"
CGAN_CKPT_PRIORITY = [
    os.path.join(CGAN_CKPT_BASE, "cgan-epoch=04.ckpt"),   # epoch 5  — tốt nhất
    os.path.join(CGAN_CKPT_BASE, "cgan-epoch=09.ckpt"),   # epoch 10
    os.path.join(CGAN_CKPT_BASE, "cgan-epoch=14.ckpt"),   # epoch 15
    os.path.join(CGAN_CKPT_BASE, "cgan-epoch=19.ckpt"),   # epoch 20
    os.path.join(CGAN_CKPT_BASE, "last.ckpt"),            # epoch 30 — mode collapse
]

TEXT_ENC_ID = "openai/clip-vit-large-patch14"
RESOLUTION  = 128
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fp16_dtype  = torch.float16

MAX_SEQ_LEN = 15

print(f"🖥️  Device: {device}")


# ── 2. Xây Vocabulary cho cVAE (từ metadata) ─────────────────────────────────
def build_vocab(metadata_path):
    all_words = []
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                item   = json.loads(line)
                prompt = item.get("text", "").lower()
                all_words.extend(re.findall(r"\b\w+\b", prompt))
        word_counts = Counter(all_words)
        vocab = {word: i + 1 for i, (word, _) in enumerate(word_counts.items())}
        return vocab, len(vocab) + 1   # vocab_size bao gồm padding=0
    except Exception as e:
        print(f"⚠️  Không đọc được metadata để build vocab: {e}")
        return {}, 68   # fallback: dùng đúng vocab_size từ checkpoint

vocab, VOCAB_SIZE = build_vocab(METADATA_PATH)
print(f"📖 Vocab size: {VOCAB_SIZE}")


def tokenize_prompt(prompt, vocab, max_len=MAX_SEQ_LEN):
    words  = re.findall(r"\b\w+\b", prompt.lower())
    tokens = [vocab.get(w, 0) for w in words][:max_len]
    tokens += [0] * (max_len - len(tokens))
    return tokens


# ── 3. Định nghĩa kiến trúc models ───────────────────────────────────────────

# -------- cGAN --------
class CGANGenerator(nn.Module):
    def __init__(self, latent_dim=100, text_embed_dim=768, img_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.text_proj  = nn.Linear(text_embed_dim, 128)
        self.init_size  = 128 // 16   # 8
        self.l1 = nn.Sequential(nn.Linear(latent_dim + 128, 512 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, 3, 1, 1), nn.BatchNorm2d(256, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128,  64, 3, 1, 1), nn.BatchNorm2d( 64, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d( 64, img_channels, 3, 1, 1), nn.Tanh(),
        )

    def forward(self, noise, text_embeddings):
        c   = F.leaky_relu(self.text_proj(text_embeddings), 0.2)
        out = self.l1(torch.cat((noise, c), -1))
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        return self.conv_blocks(out)


# -------- cVAE (khớp với cvae_best.pth từ notebook) --------
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.embedding(x).mean(dim=1)


class cVAE(nn.Module):
    def __init__(self, vocab_size, latent_dim=128, cond_dim=128):
        super().__init__()
        self.latent_dim   = latent_dim
        self.text_encoder = TextEncoder(vocab_size, cond_dim)
        self.enc_conv = nn.Sequential(
            nn.Conv2d(  3,  32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d( 32,  64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d( 64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
        )
        flat = 256 * 8 * 8
        self.fc_mu     = nn.Linear(flat + cond_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat + cond_dim, latent_dim)
        self.fc_dec    = nn.Linear(latent_dim + cond_dim, flat)
        self.dec_conv  = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128,  64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d( 64,  32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d( 32,   3, 4, 2, 1), nn.Sigmoid(),
        )

    @torch.no_grad()
    def generate(self, tokens_tensor):
        c_vec  = self.text_encoder(tokens_tensor)
        z      = torch.randn(tokens_tensor.size(0), self.latent_dim, device=tokens_tensor.device)
        dec_in = torch.cat([z, c_vec], dim=1)
        x_dec  = self.fc_dec(dec_in).view(-1, 256, 8, 8)
        return self.dec_conv(x_dec)   # output: [0, 1]


# ── 4. Load Models ────────────────────────────────────────────────────────────
print("Đang tải models...")

# 4a. Diffusion
diffusion_error = None
unet = clip_text_enc = clip_tokenizer = scheduler = None
try:
    clip_tokenizer = CLIPTokenizer.from_pretrained(TEXT_ENC_ID)
    clip_text_enc  = CLIPTextModel.from_pretrained(TEXT_ENC_ID).to(device, fp16_dtype)
    clip_text_enc.eval()
    unet      = UNet2DConditionModel.from_pretrained(DIFFUSION_MODEL_DIR).to(device, fp16_dtype)
    unet.eval()
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    print("✅ Diffusion model loaded.")
except Exception:
    diffusion_error = traceback.format_exc()
    print(f"⚠️  Diffusion load failed.")

# 4b. cGAN — thử từng checkpoint theo thứ tự ưu tiên (tránh mode collapse)
cgan_error          = None
cgan_generator      = cgan_tokenizer = cgan_text_enc = None
cgan_loaded_ckpt    = None
try:
    from train_cgan import cGANLightning
    loaded = False
    errors = []
    for ckpt_path in CGAN_CKPT_PRIORITY:
        if not os.path.exists(ckpt_path):
            errors.append(f"  Not found: {ckpt_path}")
            continue
        try:
            cgan_module    = cGANLightning.load_from_checkpoint(ckpt_path, map_location=device)
            cgan_generator = cgan_module.generator.to(device).eval()
            cgan_tokenizer = cgan_module.tokenizer
            cgan_text_enc  = cgan_module.text_encoder.to(device).eval()
            cgan_loaded_ckpt = os.path.basename(ckpt_path)
            print(f"✅ cGAN loaded from: {cgan_loaded_ckpt}")
            loaded = True
            break
        except Exception as e:
            errors.append(f"  Failed {ckpt_path}: {e}")
    if not loaded:
        raise RuntimeError("Không tìm thấy checkpoint cGAN nào.\n" + "\n".join(errors))
except Exception:
    cgan_error = traceback.format_exc()
    print(f"⚠️  cGAN load failed.")

# 4c. cVAE (state dict từ notebook)
cvae_error = None
cvae_model = None
try:
    cvae_model = cVAE(vocab_size=VOCAB_SIZE).to(device)
    state = torch.load(CVAE_CKPT_PATH, map_location=device, weights_only=False)
    cvae_model.load_state_dict(state)
    cvae_model.eval()
    print("✅ cVAE model loaded.")
except Exception:
    cvae_error = traceback.format_exc()
    print(f"⚠️  cVAE load failed.")

print("🚀 Khởi tạo giao diện...")


# ── 5. Helper: encode text cho DistilBERT (dùng cho cGAN) ────────────────────
def encode_distilbert(tokenizer_obj, text_enc_model, prompt):
    inputs = tokenizer_obj(
        [prompt], padding=True, truncation=True,
        return_tensors="pt", max_length=77
    ).to(device)
    with torch.no_grad():
        out = text_enc_model(**inputs)
    return out.last_hidden_state[:, 0, :]


# ── 6. Inference Functions ────────────────────────────────────────────────────

@torch.no_grad()
def run_diffusion(prompt, num_steps, guidance_scale, seed):
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

    uncond = clip_tokenizer(
        "", padding="max_length", max_length=clip_tokenizer.model_max_length,
        return_tensors="pt"
    ).to(device)
    uncond_hidden = clip_text_enc(uncond.input_ids)[0]

    latents = torch.randn(
        (1, unet.config.in_channels, RESOLUTION, RESOLUTION),
        device=device, dtype=fp16_dtype
    )
    for t in scheduler.timesteps:
        inp  = scheduler.scale_model_input(torch.cat([latents] * 2), t)
        with torch.amp.autocast("cuda", dtype=fp16_dtype):
            pred = unet(inp, t, encoder_hidden_states=torch.cat([uncond_hidden, enc_hidden])).sample
        uncond_p, cond_p = pred.chunk(2)
        noise = uncond_p + guidance_scale * (cond_p - uncond_p)
        latents = scheduler.step(noise, t, latents).prev_sample

    img = (latents / 2 + 0.5).clamp(0, 1)
    img = (img.cpu().permute(0, 2, 3, 1).float().numpy()[0] * 255).round().astype("uint8")
    return Image.fromarray(img), time.time() - t0, None


@torch.no_grad()
def run_cgan(prompt, seed):
    if cgan_generator is None:
        return None, 0.0, f"❌ cGAN chưa load được model:\n{cgan_error}"
    t0 = time.time()
    if seed != -1:
        torch.manual_seed(int(seed))

    text_embed = encode_distilbert(cgan_tokenizer, cgan_text_enc, prompt)
    z          = torch.randn(1, 100, device=device)
    gen_img    = cgan_generator(z, text_embed)
    gen_img    = (gen_img * 0.5 + 0.5).clamp(0, 1)
    img = (gen_img.cpu().permute(0, 2, 3, 1).float().numpy()[0] * 255).round().astype("uint8")
    return Image.fromarray(img), time.time() - t0, None


@torch.no_grad()
def run_cvae(prompt, seed):
    if cvae_model is None:
        return None, 0.0, f"❌ cVAE chưa load được model:\n{cvae_error}"
    t0 = time.time()
    if seed != -1:
        torch.manual_seed(int(seed))

    if not vocab:
        return None, 0.0, "❌ cVAE: Không build được vocab (thiếu metadata.jsonl)."

    tokens = tokenize_prompt(prompt, vocab)
    t_in   = torch.tensor([tokens], dtype=torch.long, device=device)
    gen    = cvae_model.generate(t_in)   # output đã [0,1]
    img = (gen.cpu().permute(0, 2, 3, 1).float().numpy()[0] * 255).round().astype("uint8")
    return Image.fromarray(img), time.time() - t0, None


# ── 7. Hàm chính: chạy cả 3 model ────────────────────────────────────────────
def generate_all(prompt, num_steps, guidance_scale, seed):
    img_diff, t_diff, err_diff = run_diffusion(prompt, num_steps, guidance_scale, seed)
    img_cgan, t_cgan, err_cgan = run_cgan(prompt, seed)
    img_cvae, t_cvae, err_cvae = run_cvae(prompt, seed)

    label_diff = f"⏱️ {t_diff:.2f}s" if err_diff is None else "⚠️ Lỗi"
    label_cgan = f"⏱️ {t_cgan:.2f}s" if err_cgan is None else "⚠️ Lỗi"
    label_cvae = f"⏱️ {t_cvae:.2f}s" if err_cvae is None else "⚠️ Lỗi"

    errors = []
    if err_diff: errors.append(f"**Diffusion:**\n```\n{err_diff}\n```")
    if err_cgan: errors.append(f"**cGAN:**\n```\n{err_cgan}\n```")
    if err_cvae: errors.append(f"**cVAE:**\n```\n{err_cvae}\n```")
    error_md = "\n\n".join(errors)

    return img_diff, label_diff, img_cgan, label_cgan, img_cvae, label_cvae, error_md


# ── 8. Giao diện Gradio ───────────────────────────────────────────────────────
css = """
.model-col { border: 1px solid #2d2d2d; border-radius: 12px; padding: 12px; background: #1a1a2e; }
.time-badge textarea { text-align: center; font-size: 1.1em; font-weight: bold; color: #00d4ff; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), css=css) as demo:
    gr.Markdown("""
    # 🎨 Text-to-Image: So sánh 3 Model Deep Learning
    **Diffusion (UNet)** &nbsp;|&nbsp; **cGAN (Conditional GAN)** &nbsp;|&nbsp; **cVAE (Conditional VAE)**

    Nhập prompt → Sinh ảnh từ cả 3 model và so sánh kết quả + thời gian.
    """)

    with gr.Row():
        with gr.Column(scale=3):
            prompt_input = gr.Textbox(
                label="📝 Prompt (Mô tả ảnh)",
                placeholder="VD: a diamond sword, game icon, glowing cyan aura...",
                lines=2,
            )
        with gr.Column(scale=1):
            seed_input = gr.Number(label="🎲 Seed (-1 = ngẫu nhiên)", value=42, precision=0)

    with gr.Row():
        num_steps_input = gr.Slider(10, 100, value=50, step=1,
                                    label="🔁 Số bước khử nhiễu (Diffusion)")
        guidance_input  = gr.Slider(1.0, 15.0, value=7.5, step=0.5,
                                    label="🎯 CFG Scale (Diffusion)")

    generate_btn = gr.Button("🚀 Sinh Ảnh (Cả 3 Model)", variant="primary", size="lg")
    gr.Markdown("---")

    # 3 cột kết quả
    with gr.Row(equal_height=True):
        with gr.Column(elem_classes="model-col"):
            gr.Markdown("### 🌊 Diffusion Model")
            diff_img  = gr.Image(label="Kết quả", type="pil", interactive=False, height=256)
            diff_time = gr.Textbox(label="⏱️ Thời gian sinh", interactive=False,
                                   elem_classes="time-badge")

        with gr.Column(elem_classes="model-col"):
            cgan_ckpt_name = cgan_loaded_ckpt or "Không load được"
            gr.Markdown(f"### ⚡ cGAN\n`checkpoint: {cgan_ckpt_name}`")
            cgan_img  = gr.Image(label="Kết quả", type="pil", interactive=False, height=256)
            cgan_time = gr.Textbox(label="⏱️ Thời gian sinh", interactive=False,
                                   elem_classes="time-badge")

        with gr.Column(elem_classes="model-col"):
            gr.Markdown("### 🧬 cVAE")
            cvae_img  = gr.Image(label="Kết quả", type="pil", interactive=False, height=256)
            cvae_time = gr.Textbox(label="⏱️ Thời gian sinh", interactive=False,
                                   elem_classes="time-badge")

    error_output = gr.Markdown(label="⚠️ Lỗi chi tiết (nếu có)", value="")

    generate_btn.click(
        fn=generate_all,
        inputs=[prompt_input, num_steps_input, guidance_input, seed_input],
        outputs=[diff_img, diff_time, cgan_img, cgan_time, cvae_img, cvae_time, error_output],
    )

    gr.Examples(
        examples=[
            ["a diamond sword, game icon, glowing cyan aura",    50, 7.5, 42],
            ["a gold potion, game icon, radiant yellow aura",    50, 7.5, 42],
            ["a wood spellbook, game icon, warm brown texture",  50, 7.5, 42],
            ["a copper axe, game icon, orange-brown copper ore", 50, 7.5, 42],
        ],
        inputs=[prompt_input, num_steps_input, guidance_input, seed_input],
        label="💡 Prompt mẫu",
    )

    gr.Markdown("""
    ---
    **Ghi chú:**
    - **Diffusion** dùng CLIP + UNet tùy chỉnh — chậm hơn (nhiều bước denoising).
    - **cGAN** dùng DistilBERT — sinh ảnh nhanh (1 forward pass). ⚠️ GAN hay bị *mode collapse* ở epoch muộn, app tự ưu tiên checkpoint epoch sớm (5→10).
    - **cVAE** dùng Embedding đơn giản (vocab từ dataset) — sinh ảnh nhanh nhất.
    """)


# ── 9. Chạy ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(share=True)