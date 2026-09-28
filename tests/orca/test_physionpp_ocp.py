from pathlib import Path
import torch

def test_recipe_files_exist():
    # Paths updated by the makeup/ reorganization; this test used to live at
    # recipe/orca/tests/ and resolve relative to recipe/orca/.
    root = Path(__file__).resolve().parents[2]
    assert (root / "src/core/orca_model.py").is_file()
    assert (root / "src/experiments/orca_physionpp_oracle_ocp_seed239/cache_latents.py").is_file()

def test_feature_concat_shape():
    c=torch.zeros(2560); f=torch.ones(2560)
    assert torch.cat([c,f]).shape==(5120,)
