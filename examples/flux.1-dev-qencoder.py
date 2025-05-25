import torch
from diffusers import FluxPipeline

from nunchakukp import NunchakukpFluxTransformer2dModel, NunchakukpT5EncoderModel
from nunchakukp.utils import get_precision

precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
transformer = NunchakukpFluxTransformer2dModel.from_pretrained(f"mit-han-lab/svdq-{precision}-flux.1-dev")
text_encoder_2 = NunchakukpT5EncoderModel.from_pretrained("mit-han-lab/svdq-flux.1-t5")
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    text_encoder_2=text_encoder_2,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
).to("cuda")
image = pipeline("A cat holding a sign that says hello world", num_inference_steps=50, guidance_scale=3.5).images[0]
image.save(f"flux.1-dev-{precision}.png")
