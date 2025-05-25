import torch
from diffusers import FluxPipeline, FluxPriorReduxPipeline
from diffusers.utils import load_image

from nunchakukp import NunchakukpFluxTransformer2dModel
from nunchakukp.utils import get_precision

precision = get_precision()
pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Redux-dev", torch_dtype=torch.bfloat16
).to("cuda")
transformer = NunchakukpFluxTransformer2dModel.from_pretrained(f"mit-han-lab/svdq-{precision}-flux.1-dev")
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    text_encoder=None,
    text_encoder_2=None,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
).to("cuda")

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png")
pipe_prior_output = pipe_prior_redux(image)
images = pipe(guidance_scale=2.5, num_inference_steps=50, **pipe_prior_output).images
images[0].save(f"flux.1-redux-dev-{precision}.png")
