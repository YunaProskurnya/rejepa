import os
from pathlib import Path
import runpy
import sys
from unittest.mock import MagicMock, patch


if __name__ == "__main__":
    launcher = Path(__file__).resolve().parents[1] / "scripts/train.py"
    sys.argv[0] = str(launcher)
    with patch.dict(os.environ, {"COMET_API_KEY": "test-key"}), patch.dict(
        sys.modules,
        {"comet_ml": MagicMock(), "comet_ml.integration.pytorch": MagicMock()},
    ):
        runpy.run_path(str(launcher), run_name="__main__")
