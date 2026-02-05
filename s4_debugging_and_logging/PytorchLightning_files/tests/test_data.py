from . import _PATH_DATA
import torch
import pytest
import os

@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data():
    """dataset = MNIST(...)
    assert len(dataset) == N_train for training and N_test for test.
    assert that each datapoint has shape [1,28,28] or [784] depending 
    on how you choose to format.
    assert that all labels are represented."""

    train_images = torch.load(f"{_PATH_DATA}/train_images.pt")
    train_target = torch.load(f"{_PATH_DATA}/train_target.pt")
    test_images = torch.load(f"{_PATH_DATA}/test_images.pt")
    test_target = torch.load(f"{_PATH_DATA}/test_target.pt")
    N_train = 50000
    N_test = 5000
    
    assert len(train_images) == N_train, "Train set did not have the correct number of samples"
    assert len(test_images) == N_test, "Test set did not have the correct number of samples"
    
    assert test_images.shape[1:] == (1,28,28), "Input shape incorrect size"

    unique_labels_train = set(train_target.tolist())
    assert all(i in unique_labels_train for i in range(10)), "All labels were not represented in train set"

    unique_labels_test = set(test_target.tolist())
    assert all(i in unique_labels_test for i in range(10)), "All labels were not represented in test set"