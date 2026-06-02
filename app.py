import gradio as gr
import torch
from diffusers import UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import numpy as np

# ── 1. Cấu hình & Đường dẫn ───────────────────────────────────────────────
MODEL_DIR    = "/teamspace/studios/this_studio/diffusion-from-scratch/final_model"
TEXT_ENC_ID  = "openai/clip-vit-large-patch14"
RESOLUTION   = 128
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype        = torch.float16 # Dùng fp16 cho inference để tiết kiệm VRAM và tăng tốc

print("Đang tải các models...")

# ── 2. Load Tokenizer & Text Encoder (Pretrained) ────────────────────────
tokenizer = CLIPTokenizer.from_pretrained(TEXT_ENC_ID)
text_enc = CLIPTextModel.from_pretrained(TEXT_ENC_ID).to(device, dtype)
text_enc.eval()

# ── 3. Load UNet (Mô hình bạn vừa train) ──────────────────────────────────
unet = UNet2DConditionModel.from_pretrained(MODEL_DIR).to(device, dtype)
unet.eval()

# ── 4. Scheduler ─────────────────────────────────────────────────────────
scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")

print("✅ Tải models thành công! Đang khởi tạo giao diện...")

# ── 5. Hàm Sinh Ảnh (Inference Pipeline) ─────────────────────────────────
@torch.no_grad()
def generate_image(prompt, num_steps, guidance_scale, seed):
    # Set seed nếu có
    if seed != -1:
        torch.manual_seed(seed)
    
    scheduler.set_timesteps(num_steps)
    
    # 5.1. Encode Text (Prompt của người dùng)
    tokens = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt"
    ).to(device)
    encoder_hidden_states = text_enc(tokens.input_ids)[0]
    
    # 5.2. Encode Text (Unconditional cho Classifier-Free Guidance)
    uncond_tokens = tokenizer(
        "", padding="max_length", max_length=tokenizer.model_max_length,
        return_tensors="pt"
    ).to(device)
    uncond_hidden = text_enc(uncond_tokens.input_ids)[0]
    
    # 5.3. Khởi tạo nhiễu ngẫu nhiên (Noise)
    latents = torch.randn(
        (1, unet.config.in_channels, RESOLUTION, RESOLUTION), 
        device=device, dtype=dtype
    )
    
    # 5.4. Vòng lặp khử nhiễu (Denoising Loop)
    for t in scheduler.timesteps:
        # Nhân đôi latent để đưa vào mô hình 1 lần (cho cả cond và uncond)
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
        with torch.amp.autocast("cuda", dtype=dtype):
            noise_pred = unet(
                latent_model_input, t, 
                encoder_hidden_states=torch.cat([uncond_hidden, encoder_hidden_states])
            ).sample
            
        # Tách kết quả và áp dụng Classifier-Free Guidance (CFG)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Bước lùi về thời điểm t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        
    # 5.5. Giải mã về ảnh RGB
    image = (latents / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
    image = (image * 255).round().astype("uint8")
    
    return Image.fromarray(image)

# ── 6. Giao diện Gradio ──────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 Demo Text-to-Image (Tự build từ con số 0)")
    gr.Markdown("Nhập miêu tả (prompt) để sinh ảnh game icon kích thước 128x128.")
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(label="Prompt (Mô tả ảnh)", placeholder="VD: a diamond sword, game icon, glowing cyan aura...")
            
            with gr.Row():
                num_steps = gr.Slider(minimum=10, maximum=100, value=50, step=1, label="Số bước khử nhiễu (Steps)")
                guidance_scale = gr.Slider(minimum=1.0, maximum=15.0, value=7.5, step=0.5, label="Mức độ bám sát Prompt (CFG Scale)")
            
            seed = gr.Number(label="Seed (để ngẫu nhiên thì nhập -1)", value=-1, precision=0)
            btn = gr.Button("🚀 Sinh Ảnh", variant="primary")
            
        with gr.Column(scale=1):
            result_image = gr.Image(label="Kết quả sinh ra", type="pil", interactive=False)
            
    btn.click(
        fn=generate_image,
        inputs=[prompt, num_steps, guidance_scale, seed],
        outputs=[result_image]
    )
    
    # Gợi ý vài prompt mẫu
    gr.Examples(
        examples=[
            ["a diamond sword, game icon, glowing cyan aura", 50, 7.5, -1],
            ["a gold potion, game icon, radiant yellow aura", 50, 7.5, -1],
            ["a wood spellbook, game icon, warm brown", 50, 7.5, -1]
        ],
        inputs=[prompt, num_steps, guidance_scale, seed]
    )

# Chạy server
if __name__ == "__main__":
    demo.launch(share=True) # share=True sẽ tạo một link public (xxx.gradio.live) để bạn truy cập từ điện thoại/máy tính ngoài