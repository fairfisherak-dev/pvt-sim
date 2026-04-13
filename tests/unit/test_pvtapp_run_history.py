"""Run-history artifact helpers for backend-driven reruns."""

from __future__ import annotations

from pathlib import Path

from pvtapp.job_runner import (
    build_rerun_config,
    load_run_config,
    rerun_saved_run,
    run_calculation,
)
from pvtapp.schemas import RunConfig, RunStatus


def _pt_flash_config() -> RunConfig:
    return RunConfig.model_validate(
        {
            "run_id": "seed-run",
            "run_name": "PT Flash seed",
            "composition": {
                "components": [
                    {"component_id": "C1", "mole_fraction": 0.5},
                    {"component_id": "C10", "mole_fraction": 0.5},
                ]
            },
            "calculation_type": "pt_flash",
            "eos_type": "peng_robinson",
            "pt_flash_config": {
                "pressure_pa": 5.0e6,
                "temperature_k": 350.0,
            },
        }
    )


def test_load_run_config_reads_persisted_config(tmp_path: Path) -> None:
    config = _pt_flash_config()

    seed_result = run_calculation(
        config=config,
        output_dir=tmp_path,
        write_artifacts=True,
    )

    run_dirs = sorted(path for path in tmp_path.iterdir() if path.is_dir())
    assert len(run_dirs) == 1

    loaded = load_run_config(run_dirs[0])

    assert loaded is not None
    assert loaded.model_dump(mode="json") == seed_result.config.model_dump(mode="json")


def test_build_rerun_config_clears_run_id_and_can_override_name() -> None:
    config = _pt_flash_config()

    rerun_config = build_rerun_config(config, run_name="PT Flash rerun")

    assert rerun_config.run_id is None
    assert rerun_config.run_name == "PT Flash rerun"
    assert rerun_config.calculation_type == config.calculation_type
    assert rerun_config.composition.model_dump(mode="json") == config.composition.model_dump(mode="json")


def test_rerun_saved_run_replays_artifact_config_with_fresh_identity(tmp_path: Path) -> None:
    config = _pt_flash_config()

    seed_result = run_calculation(
        config=config,
        output_dir=tmp_path,
        write_artifacts=True,
    )
    seed_run_dir = next(path for path in tmp_path.iterdir() if path.is_dir())

    rerun_result = rerun_saved_run(
        seed_run_dir,
        output_dir=tmp_path,
        write_artifacts=True,
        run_name="PT Flash rerun",
    )

    run_dirs = sorted(path for path in tmp_path.iterdir() if path.is_dir())
    assert len(run_dirs) == 2

    assert seed_result.status == RunStatus.COMPLETED
    assert rerun_result.status == RunStatus.COMPLETED
    assert rerun_result.run_id != seed_result.run_id
    assert rerun_result.run_name == "PT Flash rerun"
    assert rerun_result.config.run_id is None
    assert rerun_result.config.run_name == "PT Flash rerun"
    assert rerun_result.pt_flash_result is not None
    assert seed_result.pt_flash_result is not None
    assert rerun_result.pt_flash_result.phase == seed_result.pt_flash_result.phase
    assert rerun_result.pt_flash_result.vapor_fraction == seed_result.pt_flash_result.vapor_fraction
