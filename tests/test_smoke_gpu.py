import torch
import pytest

from cryptomvp.utils.gpu import CudaUnavailableError, require_cuda


def test_require_cuda():
    if torch.cuda.is_available():
        device = require_cuda()
        assert device.type == "cuda"
    else:
        with pytest.raises(CudaUnavailableError):
            require_cuda()