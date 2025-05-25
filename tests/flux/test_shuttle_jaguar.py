import pytest

from .utils import run_test
from nunchakukp.utils import get_precision, is_turing


@pytest.mark.skipif(is_turing(), reason="Skip tests due to Turing GPUs")
@pytest.mark.parametrize(
    "height,width,attention_impl,cpu_offload,expected_lpips",
    [(1024, 1024, "flashattn2", False, 0.25), (2048, 512, "nunchakukp-fp16", False, 0.25)],
)
def test_shuttle_jaguar(height: int, width: int, attention_impl: str, cpu_offload: bool, expected_lpips: float):
    run_test(
        precision=get_precision(),
        model_name="shuttle-jaguar",
        height=height,
        width=width,
        attention_impl=attention_impl,
        cpu_offload=cpu_offload,
        expected_lpips=expected_lpips,
    )
