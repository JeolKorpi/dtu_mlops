from . import _PATH_DATA
import torch
from model import JoelsAwesomeModel
import pytest

def test_model():
    """implement at least a test that checks for a given 
    input with shape X that the output of the model has shape Y.
    """

    dummy_input = torch.randn(1, 1, 28, 28)
    model = JoelsAwesomeModel()
    output = model(dummy_input)

    assert dummy_input.shape == (1,1,28,28), "Input shape does not match - should be [1, 28, 28]"
    assert output.shape == (1,10), "Output shape does not match - should be 10"

def test_error_on_wrong_shape():
    model = JoelsAwesomeModel()
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(torch.randn(1,2,3)), "ValueError does not recognize incorrect input dimensionality"
    with pytest.raises(ValueError, match=r'Expected each sample to have shape \[1, 28, 28\]'):
        model(torch.randn(1,1,28,29)), "ValueError does not recognize incorrect input shape"

@pytest.mark.parametrize("batch_size", [32, 64])
def test_parametrize(batch_size: int) -> None:
    model = JoelsAwesomeModel()
    x = torch.randn(batch_size, 1, 28, 28)
    y = model(x)
    assert y.shape == (batch_size, 10)