
import os
import pytest

resources_dir =  os.path.dirname(__file__) + os.sep + os.pardir + "/resources"

@pytest.fixture
def bknd_depths_100():
    return os.path.join(resources_dir, "bknd_depths_100.bed")
