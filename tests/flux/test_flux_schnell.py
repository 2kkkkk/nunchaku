import pytest

from nunchakukp.utils import get_precision, is_turing
from .utils import run_test


@pytest.mark.skipif(is_turing(), reason="Skip tests due to Turing GPUs")
@pytest.mark.parametrize(
    "height,width,attention_impl,cpu_offload,expected_lpips",
    [
        (1024, 1024, "flashattn2", False, 0.250),
        (1024, 1024, "nunchakukp-fp16", False, 0.255),
        (1024, 1024, "flashattn2", True, 0.250),
        (1920, 1080, "nunchakukp-fp16", False, 0.253),
        (2048, 2048, "flashattn2", True, 0.274),
    ],
)
def test_int4_schnell(height: int, width: int, attention_impl: str, cpu_offload: bool, expected_lpips: float):
    run_test(
        precision=get_precision(),
        height=height,
        width=width,
        attention_impl=attention_impl,
        cpu_offload=cpu_offload,
        expected_lpips=expected_lpips,
    )
