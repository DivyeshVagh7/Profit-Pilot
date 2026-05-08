from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "kaggle_p100_5m_profit_pilot_training.ipynb"


def _notebook_source_text() -> str:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_kaggle_notebook_is_branded_for_t4x2_outputs() -> None:
    source = _notebook_source_text()
    lower_source = source.lower()

    assert "Kaggle T4 x2" in source
    assert "GPU T4 x2" in source
    assert "kaggle_t4x2_5m_cash_allocation" in source
    assert "ppo_5m_t4x2_cash_multi_asset" in source
    assert "p100" not in lower_source


def test_kaggle_notebook_checks_dual_t4_runtime_and_uses_cuda0() -> None:
    source = _notebook_source_text()

    assert "EXPECTED_T4_GPU_COUNT" in source
    assert "torch.cuda.device_count()" in source
    assert "torch.cuda.get_device_name(gpu_index)" in source
    assert "CUDA_TRAIN_DEVICE = 'cuda:0'" in source
    assert "PROFIT_PILOT_CUDA_TRAIN_DEVICE" in source


def test_kaggle_notebook_parallelizes_rollout_envs_for_faster_wall_clock() -> None:
    source = _notebook_source_text()

    assert "SubprocVecEnv" in source
    assert "TRAIN_N_ENVS" in source
    assert "SINGLE_T4_BASELINE_N_ENVS" in source
    assert "benchmark_vec_env_throughput" in source
    assert "start_method='fork'" in source
    assert "'n_envs': TRAIN_N_ENVS" in source


def test_kaggle_notebook_can_launch_distributed_feedforward_ppo() -> None:
    source = _notebook_source_text()

    assert "RUN_DISTRIBUTED_FEEDFORWARD_PPO" in source
    assert "torch.distributed.run" in source
    assert "--nproc_per_node=2" in source
    assert "profit_pilot.train.distributed_ppo" in source
