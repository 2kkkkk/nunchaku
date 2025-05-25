import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image

from nunchakukp import NunchakukpFluxTransformer2dModel
from nunchakukp.utils import get_precision

image = load_image("https://huggingface.co/mit-han-lab/svdq-int4-flux.1-fill-dev/resolve/main/example.png")
mask = load_image("https://huggingface.co/mit-han-lab/svdq-int4-flux.1-fill-dev/resolve/main/mask.png")

precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
transformer = NunchakukpFluxTransformer2dModel.from_pretrained(f"mit-han-lab/svdq-{precision}-flux.1-fill-dev")
pipe = FluxFillPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Fill-dev", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")
image = pipe(
    prompt="A wooden basket of a cat.",
    image=image,
    mask_image=mask,
    height=1024,
    width=1024,
    guidance_scale=30,
    num_inference_steps=50,
    max_sequence_length=512,
).images[0]
image.save(f"flux.1-fill-dev-{precision}.png")
