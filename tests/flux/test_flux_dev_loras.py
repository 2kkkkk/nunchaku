import pytest

from nunchakukp.utils import get_precision, is_turing
from .utils import run_test


@pytest.mark.skipif(is_turing(), reason="Skip tests due to Turing GPUs")
@pytest.mark.parametrize(
    "num_inference_steps,lora_name,lora_strength,cpu_offload,expected_lpips",
    [
        (25, "realism", 0.9, True, 0.178),
        (25, "ghibsky", 1, False, 0.164),
        (28, "anime", 1, False, 0.284),
        (24, "sketch", 1, True, 0.223),
        (28, "yarn", 1, False, 0.211),
        (25, "haunted_linework", 1, True, 0.317),
    ],
)
def test_flux_dev_loras(num_inference_steps, lora_name, lora_strength, cpu_offload, expected_lpips):
    run_test(
        precision=get_precision(),
        model_name="flux.1-dev",
        dataset_name=lora_name,
        height=1024,
        width=1024,
        num_inference_steps=num_inference_steps,
        guidance_scale=3.5,
        use_qencoder=False,
        cpu_offload=cpu_offload,
        lora_names=lora_name,
        lora_strengths=lora_strength,
        cache_threshold=0,
        expected_lpips=expected_lpips,
    )


@pytest.mark.skipif(is_turing(), reason="Skip tests due to Turing GPUs")
def test_flux_dev_hypersd8_1536x2048():
    run_test(
        precision=get_precision(),
        model_name="flux.1-dev",
        dataset_name="MJHQ",
        height=1536,
        width=2048,
        num_inference_steps=8,
        guidance_scale=3.5,
        use_qencoder=False,
        attention_impl="nunchakukp-fp16",
        cpu_offload=True,
        lora_names="hypersd8",
        lora_strengths=0.125,
        cache_threshold=0,
        expected_lpips=0.291,
    )


@pytest.mark.skipif(is_turing(), reason="Skip tests due to Turing GPUs")
def test_flux_dev_turbo8_2048x2048():
    run_test(
        precision=get_precision(),
        model_name="flux.1-dev",
        dataset_name="MJHQ",
        height=2048,
        width=2048,
        num_inference_steps=8,
        guidance_scale=3.5,
        use_qencoder=False,
        attention_impl="nunchakukp-fp16",
        cpu_offload=True,
        lora_names="turbo8",
        lora_strengths=1,
        cache_threshold=0,
        expected_lpips=0.189,
    )


# lora composition
@pytest.mark.skipif(is_turing(), reason="Skip tests due to Turing GPUs")
def test_flux_dev_turbo8_yarn_2048x1024():
    run_test(
        precision=get_precision(),
        model_name="flux.1-dev",
        dataset_name="yarn",
        height=2048,
        width=1024,
        num_inference_steps=8,
        guidance_scale=3.5,
        use_qencoder=False,
        cpu_offload=True,
        lora_names=["turbo8", "yarn"],
        lora_strengths=[1, 1],
        cache_threshold=0,
        expected_lpips=0.252,
    )


# large rank loras
@pytest.mark.skipif(is_turing(), reason="Skip tests due to Turing GPUs")
def test_flux_dev_turbo8_yarn_1024x1024():
    run_test(
        precision=get_precision(),
        model_name="flux.1-dev",
        dataset_name="ghibsky",
        height=1024,
        width=1024,
        num_inference_steps=8,
        guidance_scale=3.5,
        use_qencoder=False,
        cpu_offload=True,
        lora_names=["realism", "ghibsky", "anime", "sketch", "yarn", "haunted_linework", "turbo8"],
        lora_strengths=[0, 1, 0, 0, 0, 0, 1],
        cache_threshold=0,
        expected_lpips=0.44,
    )
