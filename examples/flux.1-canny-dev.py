import torch
from controlnet_aux import CannyDetector
from diffusers import FluxControlPipeline
from diffusers.utils import load_image

from nunchakukp import NunchakukpFluxTransformer2dModel
from nunchakukp.utils import get_precision

precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
transformer = NunchakukpFluxTransformer2dModel.from_pretrained(f"mit-han-lab/svdq-{precision}-flux.1-canny-dev")
pipe = FluxControlPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Canny-dev", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")

prompt = (
    "A robot made of exotic candies and chocolates of different kinds. "
    "The background is filled with confetti and celebratory gifts."
)
control_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png")

processor = CannyDetector()
control_image = processor(
    control_image, low_threshold=50, high_threshold=200, detect_resolution=1024, image_resolution=1024
)

image = pipe(
    prompt=prompt, control_image=control_image, height=1024, width=1024, num_inference_steps=50, guidance_scale=30.0
).images[0]
image.save(f"flux.1-canny-dev-{precision}.png")
