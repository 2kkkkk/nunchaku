import pytest
import torch

from nunchakukp.utils import get_precision, is_turing
from .utils import run_test


@pytest.mark.skipif(is_turing(), reason="Skip tests due to Turing GPUs")
def test_flux_canny_dev():
    run_test(
        precision=get_precision(),
        model_name="flux.1-canny-dev",
        dataset_name="MJHQ-control",
        task="canny",
        dtype=torch.bfloat16,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=30,
        attention_impl="nunchakukp-fp16",
        cpu_offload=False,
        cache_threshold=0,
        expected_lpips=0.103 if get_precision() == "int4" else 0.164,
    )


@pytest.mark.skipif(is_turing(), reason="Skip tests due to Turing GPUs")
def test_flux_depth_dev():
    run_test(
        precision=get_precision(),
        model_name="flux.1-depth-dev",
        dataset_name="MJHQ-control",
        task="depth",
        dtype=torch.bfloat16,
        height=1024,
        width=1024,
        num_inference_steps=30,
        guidance_scale=10,
        attention_impl="nunchakukp-fp16",
        cpu_offload=False,
        cache_threshold=0,
        expected_lpips=0.170 if get_precision() == "int4" else 0.120,
    )


@pytest.mark.skipif(is_turing(), reason="Skip tests due to Turing GPUs")
def test_flux_fill_dev():
    run_test(
        precision=get_precision(),
        model_name="flux.1-fill-dev",
        dataset_name="MJHQ-control",
        task="fill",
        dtype=torch.bfloat16,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=30,
        attention_impl="nunchakukp-fp16",
        cpu_offload=False,
        cache_threshold=0,
        expected_lpips=0.045,
    )


@pytest.mark.skipif(is_turing(), reason="Skip tests due to Turing GPUs")
def test_flux_dev_canny_lora():
    run_test(
        precision=get_precision(),
        model_name="flux.1-dev",
        dataset_name="MJHQ-control",
        task="canny",
        dtype=torch.bfloat16,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=30,
        attention_impl="nunchakukp-fp16",
        cpu_offload=False,
        lora_names="canny",
        lora_strengths=0.85,
        cache_threshold=0,
        expected_lpips=0.103,
    )


@pytest.mark.skipif(is_turing(), reason="Skip tests due to Turing GPUs")
def test_flux_dev_depth_lora():
    run_test(
        precision=get_precision(),
        model_name="flux.1-dev",
        dataset_name="MJHQ-control",
        task="depth",
        dtype=torch.bfloat16,
        height=1024,
        width=1024,
        num_inference_steps=30,
        guidance_scale=10,
        attention_impl="nunchakukp-fp16",
        cpu_offload=False,
        cache_threshold=0,
        lora_names="depth",
        lora_strengths=0.85,
        expected_lpips=0.163,
    )


@pytest.mark.skipif(is_turing(), reason="Skip tests due to Turing GPUs")
def test_flux_fill_dev_turbo():
    run_test(
        precision=get_precision(),
        model_name="flux.1-fill-dev",
        dataset_name="MJHQ-control",
        task="fill",
        dtype=torch.bfloat16,
        height=1024,
        width=1024,
        num_inference_steps=8,
        guidance_scale=30,
        attention_impl="nunchakukp-fp16",
        cpu_offload=False,
        cache_threshold=0,
        lora_names="turbo8",
        lora_strengths=1,
        expected_lpips=0.048,
    )


@pytest.mark.skipif(is_turing(), reason="Skip tests due to Turing GPUs")
def test_flux_dev_redux():
    run_test(
        precision=get_precision(),
        model_name="flux.1-dev",
        dataset_name="MJHQ-control",
        task="redux",
        dtype=torch.bfloat16,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=2.5,
        attention_impl="nunchakukp-fp16",
        cpu_offload=False,
        cache_threshold=0,
        expected_lpips=0.198 if get_precision() == "int4" else 0.55,  # redux seems to generate different images on 5090
    )
