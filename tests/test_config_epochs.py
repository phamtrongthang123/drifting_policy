"""
Test: All drifting configs have num_epochs <= 300.

MOTIVATION:
  The spec explicitly requires num_epochs ~ 200-300 because DDPM (the baseline)
  converges within ~150 epochs. The drifting model should match DDPM speed.
  - Can: was 300 (correct)
  - Lift: was 500 → fixed to 300
  - PushT: was 3050 (inherited from original Diffusion Policy) → fixed to 300

  Running 3050 epochs wastes ~10x compute vs. the needed 300 epochs.

LESSONS LEARNED:
  - "num_epochs: 3050" in PushT was inherited from the original Diffusion Policy
    repo and is NOT appropriate for the drifting model or even for DDPM.
  - DDPM hits 0.96 on Can at epoch 50 and 0.98 at epoch 150.
  - DDPM hits 1.00 on Lift by epoch 50.
  - Therefore 300 epochs is sufficient for all tasks.
  - These tests are regression guards to prevent accidentally re-inflating epochs.
"""

import pathlib
import yaml
import pytest

ROOT = pathlib.Path(__file__).parent.parent
MAX_EPOCHS = 300


def load_yaml(name):
    with open(ROOT / name) as f:
        return yaml.safe_load(f)


class TestConfigEpochs:

    def test_can_image_num_epochs(self):
        """drifting_can_image.yaml must have num_epochs <= 300."""
        cfg = load_yaml("drifting_can_image.yaml")
        n = cfg["training"]["num_epochs"]
        assert n <= MAX_EPOCHS, (
            f"drifting_can_image.yaml num_epochs={n} exceeds {MAX_EPOCHS}. "
            f"DDPM gets 0.98 at epoch 150; no reason to run longer."
        )
        assert n >= 200, (
            f"drifting_can_image.yaml num_epochs={n} < 200. "
            f"Need enough epochs for convergence (DDPM: ~150 epochs)."
        )

    def test_lift_image_num_epochs(self):
        """drifting_lift_image.yaml must have num_epochs <= 300."""
        cfg = load_yaml("drifting_lift_image.yaml")
        n = cfg["training"]["num_epochs"]
        assert n <= MAX_EPOCHS, (
            f"drifting_lift_image.yaml num_epochs={n} exceeds {MAX_EPOCHS}. "
            f"Was 500 before fix. DDPM hits 1.00 on Lift by epoch 50."
        )
        assert n >= 200, (
            f"drifting_lift_image.yaml num_epochs={n} < 200."
        )

    def test_pusht_image_num_epochs(self):
        """drifting_pusht_image.yaml must have num_epochs <= 300."""
        cfg = load_yaml("drifting_pusht_image.yaml")
        n = cfg["training"]["num_epochs"]
        assert n <= MAX_EPOCHS, (
            f"drifting_pusht_image.yaml num_epochs={n} exceeds {MAX_EPOCHS}. "
            f"Was 3050 (inherited from original Diffusion Policy) — ~10x compute waste."
        )
        assert n >= 200, (
            f"drifting_pusht_image.yaml num_epochs={n} < 200."
        )

    def test_all_configs_consistent(self):
        """All three configs should have the same num_epochs for fairness."""
        can = load_yaml("drifting_can_image.yaml")["training"]["num_epochs"]
        lift = load_yaml("drifting_lift_image.yaml")["training"]["num_epochs"]
        pusht = load_yaml("drifting_pusht_image.yaml")["training"]["num_epochs"]
        assert can == lift == pusht, (
            f"num_epochs mismatch: can={can}, lift={lift}, pusht={pusht}. "
            f"All tasks should train for the same number of epochs for fair comparison."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
