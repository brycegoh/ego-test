"""The GPU stages are first-class (no NotImplementedError gating), but importing the CPU
path must still never pull in torch or a GPU model library -- the GPU stage modules are
imported only when their backend is selected. Also byte-compile the GPU stage modules so
typos are caught even though their deps can't be installed in CI/CPU.
"""

import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src" / "egodex_robot"
_GPU_MODULES = ["stages/hamer.py", "stages/sam3d.py", "stages/segment.py", "stages/graspgen.py"]


def test_cpu_path_imports_without_torch():
    """Importing the package, CLI, pipeline, and CPU stages must not import torch et al."""
    code = (
        "import sys;"
        "import egodex_robot, egodex_robot.cli, egodex_robot.pipeline;"
        "import egodex_robot.stages.grasp, egodex_robot.stages.render, egodex_robot.stages.overlay;"
        "heavy=[m for m in ('torch','hamer','grasp_gen','sam2','detectron2','vitpose_model') if m in sys.modules];"
        "assert not heavy, heavy;"
        "print('clean')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "clean" in out.stdout


@pytest.mark.parametrize("rel", _GPU_MODULES)
def test_gpu_stage_modules_compile(rel):
    py_compile.compile(str(_SRC / rel), doraise=True)


@pytest.mark.parametrize("rel", _GPU_MODULES)
def test_gpu_stage_modules_are_ungated(rel):
    """No NotImplementedError gating should remain in the GPU stage modules."""
    text = (_SRC / rel).read_text()
    assert "NotImplementedError" not in text
