import pytest
import torch
from diffusers import SanaPAGPipeline, SanaPipeline

from nunchakukp import NunchakukpSanaTransformer2DModel
from nunchakukp.utils import get_precision, is_turing


@pytest.mark.skipif(is_turing() or get_precision() == "fp4", reason="Skip tests due to Turing GPUs")
def test_sana():
    transformer = NunchakukpSanaTransformer2DModel.from_pretrained("mit-han-lab/svdq-int4-sana-1600m")
    pipe = SanaPipeline.from_pretrained(
        "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
        transformer=transformer,
        variant="bf16",
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    pipe.vae.to(torch.bfloat16)
    pipe.text_encoder.to(torch.bfloat16)

    prompt = "A cute 🐼 eating 🎋, ink drawing style"
    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=4.5,
        num_inference_steps=20,
        generator=torch.Generator().manual_seed(42),
    ).images[0]

    image.save("sana_1600m.png")


@pytest.mark.skipif(is_turing() or get_precision() == "fp4", reason="Skip tests due to Turing GPUs")
def test_sana_pag():
    transformer = NunchakukpSanaTransformer2DModel.from_pretrained("mit-han-lab/svdq-int4-sana-1600m", pag_layers=8)
    pipe = SanaPAGPipeline.from_pretrained(
        "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
        transformer=transformer,
        variant="bf16",
        torch_dtype=torch.bfloat16,
        pag_applied_layers="transformer_blocks.8",
    ).to("cuda")
    pipe._set_pag_attn_processor = lambda *args, **kwargs: None

    pipe.text_encoder.to(torch.bfloat16)
    pipe.vae.to(torch.bfloat16)

    image = pipe(
        prompt="A cute 🐼 eating 🎋, ink drawing style",
        height=1024,
        width=1024,
        guidance_scale=5.0,
        pag_scale=2.0,
        num_inference_steps=20,
    ).images[0]
    image.save("sana_1600m_pag.png")
