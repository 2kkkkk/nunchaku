import os

import torch
from controlnet_aux import CannyDetector
from diffusers import FluxControlPipeline, FluxFillPipeline, FluxPipeline, FluxPriorReduxPipeline
from diffusers.utils import load_image
from image_gen_aux import DepthPreprocessor
from tqdm import tqdm

import nunchakukp
from nunchakukp import NunchakukpFluxTransformer2dModel, NunchakukpT5EncoderModel
from nunchakukp.lora.flux.compose import compose_lora
from ..data import download_hf_dataset, get_dataset
from ..utils import already_generate, compute_lpips, hash_str_to_int

ORIGINAL_REPO_MAP = {
    "flux.1-schnell": "black-forest-labs/FLUX.1-schnell",
    "flux.1-dev": "black-forest-labs/FLUX.1-dev",
    "shuttle-jaguar": "shuttleai/shuttle-jaguar",
    "flux.1-canny-dev": "black-forest-labs/FLUX.1-Canny-dev",
    "flux.1-depth-dev": "black-forest-labs/FLUX.1-Depth-dev",
    "flux.1-fill-dev": "black-forest-labs/FLUX.1-Fill-dev",
}

NUNCHAKUKP_REPO_PATTERN_MAP = {
    "flux.1-schnell": "mit-han-lab/svdq-{precision}-flux.1-schnell",
    "flux.1-dev": "mit-han-lab/svdq-{precision}-flux.1-dev",
    "shuttle-jaguar": "mit-han-lab/svdq-{precision}-shuttle-jaguar",
    "flux.1-canny-dev": "mit-han-lab/svdq-{precision}-flux.1-canny-dev",
    "flux.1-depth-dev": "mit-han-lab/svdq-{precision}-flux.1-depth-dev",
    "flux.1-fill-dev": "mit-han-lab/svdq-{precision}-flux.1-fill-dev",
}

LORA_PATH_MAP = {
    "hypersd8": "ByteDance/Hyper-SD/Hyper-FLUX.1-dev-8steps-lora.safetensors",
    "turbo8": "alimama-creative/FLUX.1-Turbo-Alpha/diffusion_pytorch_model.safetensors",
    "realism": "XLabs-AI/flux-RealismLora/lora.safetensors",
    "ghibsky": "aleksa-codes/flux-ghibsky-illustration/lora.safetensors",
    "anime": "alvdansen/sonny-anime-fixed/araminta_k_sonnyanime_fluxd_fixed.safetensors",
    "sketch": "Shakker-Labs/FLUX.1-dev-LoRA-Children-Simple-Sketch/FLUX-dev-lora-children-simple-sketch.safetensors",
    "yarn": "linoyts/yarn_art_Flux_LoRA/pytorch_lora_weights.safetensors",
    "haunted_linework": "alvdansen/haunted_linework_flux/hauntedlinework_flux_araminta_k.safetensors",
    "canny": "black-forest-labs/FLUX.1-Canny-dev-lora/flux1-canny-dev-lora.safetensors",
    "depth": "black-forest-labs/FLUX.1-Depth-dev-lora/flux1-depth-dev-lora.safetensors",
}


def run_pipeline(dataset, task: str, pipeline: FluxPipeline, save_dir: str, forward_kwargs: dict = {}):
    os.makedirs(save_dir, exist_ok=True)
    pipeline.set_progress_bar_config(desc="Sampling", leave=False, dynamic_ncols=True, position=1)

    if task == "canny":
        processor = CannyDetector()
    elif task == "depth":
        processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
    elif task == "redux":
        processor = FluxPriorReduxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Redux-dev", torch_dtype=torch.bfloat16
        ).to("cuda")
    else:
        assert task in ["t2i", "fill"]
        processor = None

    for row in tqdm(dataset):
        filename = row["filename"]
        prompt = row["prompt"]

        _forward_kwargs = {k: v for k, v in forward_kwargs.items()}

        if task == "canny":
            assert forward_kwargs.get("height", 1024) == 1024
            assert forward_kwargs.get("width", 1024) == 1024
            control_image = load_image(row["canny_image_path"])
            control_image = processor(
                control_image,
                low_threshold=50,
                high_threshold=200,
                detect_resolution=1024,
                image_resolution=1024,
            )
            _forward_kwargs["control_image"] = control_image
        elif task == "depth":
            control_image = load_image(row["depth_image_path"])
            control_image = processor(control_image)[0].convert("RGB")
            _forward_kwargs["control_image"] = control_image
        elif task == "fill":
            image = load_image(row["image_path"])
            mask_image = load_image(row["mask_image_path"])
            _forward_kwargs["image"] = image
            _forward_kwargs["mask_image"] = mask_image
        elif task == "redux":
            image = load_image(row["image_path"])
            _forward_kwargs.update(processor(image))

        seed = hash_str_to_int(filename)
        if task == "redux":
            image = pipeline(generator=torch.Generator().manual_seed(seed), **_forward_kwargs).images[0]
        else:
            image = pipeline(prompt, generator=torch.Generator().manual_seed(seed), **_forward_kwargs).images[0]
        image.save(os.path.join(save_dir, f"{filename}.png"))
        torch.cuda.empty_cache()


def run_test(
    precision: str = "int4",
    model_name: str = "flux.1-schnell",
    dataset_name: str = "MJHQ",
    task: str = "t2i",
    dtype: str | torch.dtype = torch.bfloat16,  # the full precision dtype
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 4,
    guidance_scale: float = 3.5,
    use_qencoder: bool = False,
    attention_impl: str = "flashattn2",  # "flashattn2" or "nunchakukp-fp16"
    cpu_offload: bool = False,
    cache_threshold: float = 0,
    lora_names: str | list[str] | None = None,
    lora_strengths: float | list[float] = 1.0,
    max_dataset_size: int = 20,
    i2f_mode: str | None = None,
    expected_lpips: float = 0.5,
):
    if isinstance(dtype, str):
        dtype_str = dtype
        if dtype == "bf16":
            dtype = torch.bfloat16
        else:
            assert dtype == "fp16"
            dtype = torch.float16
    else:
        if dtype == torch.bfloat16:
            dtype_str = "bf16"
        else:
            assert dtype == torch.float16
            dtype_str = "fp16"

    dataset = get_dataset(name=dataset_name, max_dataset_size=max_dataset_size)
    model_id_16bit = ORIGINAL_REPO_MAP[model_name]

    folder_name = f"w{width}h{height}t{num_inference_steps}g{guidance_scale}"

    if lora_names is None:
        lora_names = []
    elif isinstance(lora_names, str):
        lora_names = [lora_names]

    if len(lora_names) > 0:
        if isinstance(lora_strengths, (int, float)):
            lora_strengths = [lora_strengths]
        assert len(lora_names) == len(lora_strengths)

        for lora_name, lora_strength in zip(lora_names, lora_strengths):
            folder_name += f"-{lora_name}_{lora_strength}"

    if not os.path.exists(os.path.join("test_results", "ref")):
        ref_root = download_hf_dataset(local_dir=os.path.join("test_results", "ref"))
    else:
        ref_root = os.path.join("test_results", "ref")
    save_dir_16bit = os.path.join(ref_root, dtype_str, model_name, folder_name)

    if task in ["t2i", "redux"]:
        pipeline_cls = FluxPipeline
    elif task in ["canny", "depth"]:
        pipeline_cls = FluxControlPipeline
    elif task == "fill":
        pipeline_cls = FluxFillPipeline
    else:
        raise NotImplementedError(f"Unknown task {task}!")

    if not already_generate(save_dir_16bit, max_dataset_size):
        pipeline_init_kwargs = {"text_encoder": None, "text_encoder2": None} if task == "redux" else {}
        pipeline = pipeline_cls.from_pretrained(model_id_16bit, torch_dtype=dtype, **pipeline_init_kwargs)
        pipeline = pipeline.to("cuda")

        if len(lora_names) > 0:
            for i, (lora_name, lora_strength) in enumerate(zip(lora_names, lora_strengths)):
                lora_path = LORA_PATH_MAP[lora_name]
                pipeline.load_lora_weights(
                    os.path.dirname(lora_path), weight_name=os.path.basename(lora_path), adapter_name=f"lora_{i}"
                )
            pipeline.set_adapters([f"lora_{i}" for i in range(len(lora_names))], lora_strengths)

        run_pipeline(
            dataset=dataset,
            task=task,
            pipeline=pipeline,
            save_dir=save_dir_16bit,
            forward_kwargs={
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
            },
        )
        del pipeline
        # release the gpu memory
        torch.cuda.empty_cache()

    precision_str = precision
    if use_qencoder:
        precision_str += "-qe"
    if attention_impl == "flashattn2":
        precision_str += "-fa2"
    else:
        assert attention_impl == "nunchakukp-fp16"
        precision_str += "-nfp16"
    if cpu_offload:
        precision_str += "-co"
    if cache_threshold > 0:
        precision_str += f"-cache{cache_threshold}"
    if i2f_mode is not None:
        precision_str += f"-i2f{i2f_mode}"

    save_dir_4bit = os.path.join("test_results", dtype_str, precision_str, model_name, folder_name)
    if not already_generate(save_dir_4bit, max_dataset_size):
        pipeline_init_kwargs = {}
        model_id_4bit = NUNCHAKUKP_REPO_PATTERN_MAP[model_name].format(precision=precision)

        if i2f_mode is not None:
            nunchakukp._C.utils.set_faster_i2f_mode(i2f_mode)

        transformer = NunchakukpFluxTransformer2dModel.from_pretrained(
            model_id_4bit, offload=cpu_offload, torch_dtype=dtype
        )
        transformer.set_attention_impl(attention_impl)

        if len(lora_names) > 0:
            if len(lora_names) == 1:  # directly load the lora
                lora_path = LORA_PATH_MAP[lora_names[0]]
                lora_strength = lora_strengths[0]
                transformer.update_lora_params(lora_path)
                transformer.set_lora_strength(lora_strength)
            else:
                composed_lora = compose_lora(
                    [
                        (LORA_PATH_MAP[lora_name], lora_strength)
                        for lora_name, lora_strength in zip(lora_names, lora_strengths)
                    ]
                )
                transformer.update_lora_params(composed_lora)

        pipeline_init_kwargs["transformer"] = transformer
        if task == "redux":
            pipeline_init_kwargs.update({"text_encoder": None, "text_encoder_2": None})
        elif use_qencoder:
            text_encoder_2 = NunchakukpT5EncoderModel.from_pretrained("mit-han-lab/svdq-flux.1-t5")
            pipeline_init_kwargs["text_encoder_2"] = text_encoder_2
        pipeline = pipeline_cls.from_pretrained(model_id_16bit, torch_dtype=dtype, **pipeline_init_kwargs)
        if cpu_offload:
            pipeline.enable_sequential_cpu_offload()
        else:
            pipeline = pipeline.to("cuda")
        run_pipeline(
            dataset=dataset,
            task=task,
            pipeline=pipeline,
            save_dir=save_dir_4bit,
            forward_kwargs={
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
            },
        )
        del transformer
        del pipeline
        # release the gpu memory
        torch.cuda.empty_cache()
    lpips = compute_lpips(save_dir_16bit, save_dir_4bit)
    print(f"lpips: {lpips}")
    assert lpips < expected_lpips * 1.05
