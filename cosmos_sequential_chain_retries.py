#!/usr/bin/env python3
"""
Multi-stage kitchen-roboarm chains with per-stage retries and MuJoCo state checkpoints.

Motivation:
  Outer runs (e.g. 10 episodes), each episode = N stages with the same uninterrupted sim.
  If stage k fails after up to ``stage_retries`` attempts, optionally resample a new seed only
  for that stage retry while rewinding physics to the checkpoint *before* that stage began.

Artifacts (under cfg.local_log_dir, typically …/sequential_retries/<experiment_id>/):
  run_<i>/
    run_meta.json
    env_initial_reset.json             полный get_ep_meta() + cfg после первого env.reset()
    ENV_EVAL-*.txt                     (via setup_logging for this run)
    stage_<k>/retry_<j>/
      attempt_meta.json                (+ instruction_lang, layout/style id)
      attempt_scene.json               ep_meta (object_cfgs, fixtures, lang, …), cfg, объекты в сцене
      *.mp4 (rollout + optional future_pred) в этой же папке
    run_XXX/full_run_episode--*.mp4   склейка всех стадий + плавный «домой» 1s между стадиями (без future overlay)
      turnoff_stage_debug.log          (chain4 st2 / chain6 st3 «press stop»: JSONL по env.step + summary)

Env:
  COSMOS_SEQ_CHAIN_RUNS       outer loop count (default 10)
  COSMOS_SEQ_STAGE_RETRIES   max tries per stage (default 3)
  COSMOS_SEQ_MASTER_SEED     optional int for np.random.default_rng ([0,max) for base seeds)
  COSMOS_SEQ_SAVE_MUJOCO_CHECKPOINTS  если 1/true: в ``run_NNN/`` пишутся ``mujoco_checkpoint__after_reset_chain_stage_00.npz``, ``mujoco_checkpoint__after_stageKK_success_chain_stage_JJ.npz``, ``…_terminal`` (+ ``outer_episode_idx`` для ``create_robocasa_env(..., episode_idx=…)``); алиасы: chain4 после start — ``chain4_after_start_success.npz``; chain6 — ``chain6_after_start_success.npz`` (перед стадией stop)
  COSMOS_SEQ_HOTSTART_NPZ           путь к .npz (как после успешной стадии K); ``COSMOS_SEQ_HOTSTART_FIRST_STAGE`` = индекс первой стадии цикла (напр. 3 = сразу TurnOff)
  COSMOS_SEQ_HOTSTART_ALL_RUNS      если 1: каждый outer run начинается с hotstart (иначе только ``run_000``)
  COSMOS_SEQ_CONSOLE_LOG     если 0/false/no — не дублировать ключевые строки в stderr контейнера (по умолчанию: дублировать)

Invoked like cosmos_sequential_one_reset_chain.py (same PolicyEvalConfig / docker bash block).

Note: do not use ``from __future__ import annotations`` here — draccus needs a real PolicyEvalConfig
type on the decorated ``main(cfg: PolicyEvalConfig)`` parameter.
"""

import dataclasses
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import draccus

import robocasa  # noqa: F401
from robocasa.environments.kitchen.custom import kitchen_roboarm_cosmos_chain3 as _k_chain3

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_model,
    get_planning_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)
from cosmos_policy.experiments.robot.robocasa.robocasa_utils import save_rollout_video
from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import (
    TASK_MAX_STEPS,
    PolicyEvalConfig,
    create_robocasa_env,
    prepare_observation,
    run_episode,
    validate_config,
)
from cosmos_policy.utils.utils import set_seed_everywhere

# Re-use rollout saving from existing entrypoint (same filenames / layout conventions).
from cosmos_sequential_one_reset_chain import _save_stage_rollouts, resolve_chain_stages

from cosmos_policy.experiments.robot.robot_utils import log_message, setup_logging


def _env_snapshot(env: Any) -> dict[str, Any]:
    flat = np.array(env.sim.get_state().flatten(), dtype=np.float64)
    chain_stage = int(getattr(env, "chain_stage"))
    out: dict[str, Any] = {"sim_flat": flat, "chain_stage": chain_stage}
    mw = getattr(env, "microwave", None)
    if mw is not None and hasattr(mw, "_turned_on"):
        out["microwave_turned_on"] = bool(mw._turned_on)
    return out


def _snapshot_for_checkpoint(env: Any, outer_episode_idx: int) -> dict[str, Any]:
    """MuJoCo snapshot + metadata for .npz reload (layout scene index = outer run index)."""
    snap = _env_snapshot(env)
    snap["outer_episode_idx"] = int(outer_episode_idx)
    return snap


def _read_hotstart_npz(npz_path: str) -> tuple[dict[str, Any], int | None, int | None]:
    """Load ``sim_flat`` + ``chain_stage`` (+ optional ``run_seed_layout``, ``outer_episode_idx``) from saved .npz."""
    zf = np.load(npz_path, allow_pickle=False)
    snap: dict[str, Any] = {
        "sim_flat": np.asarray(zf["sim_flat"], dtype=np.float64).copy(),
        "chain_stage": int(np.asarray(zf["chain_stage"]).reshape(())),
    }
    if "microwave_turned_on" in zf.files:
        snap["microwave_turned_on"] = bool(np.asarray(zf["microwave_turned_on"]).reshape(()))
    rs = int(np.asarray(zf["run_seed_layout"]).reshape(())) if "run_seed_layout" in zf.files else None
    ep = int(np.asarray(zf["outer_episode_idx"]).reshape(())) if "outer_episode_idx" in zf.files else None
    return snap, rs, ep


def _env_restore(env: Any, snap: dict[str, Any]) -> None:
    sim = env.sim
    flat = snap["sim_flat"]
    try:
        sim.set_state_from_flattened(flat)
    except AttributeError as e:
        raise RuntimeError(
            "Simulator lacks set_state_from_flattened; cannot rewind physics for retries."
        ) from e
    env._chain_stage = int(snap["chain_stage"])  # noqa: SLF001
    sim.forward()
    # Extra forwards: after flat restore, hinge/contact geom poses settle; avoids fixture.update_state edge cases.
    sim.forward()
    sim.forward()
    # Microwave._turned_on is Python-side and NOT in sim_flat; persist it in the snapshot so retries match
    # the checkpoint (otherwise a failed «start» leaves True and the next retry instant-successes).
    mw = getattr(env, "microwave", None)
    if mw is not None and hasattr(mw, "_turned_on"):
        if "microwave_turned_on" in snap:
            mw._turned_on = bool(snap["microwave_turned_on"])
        else:
            mw._turned_on = False
    if hasattr(env, "update_state"):
        env.update_state()


def _mujoco_checkpoint_save_enabled() -> bool:
    return os.environ.get("COSMOS_SEQ_SAVE_MUJOCO_CHECKPOINTS", "").strip().lower() in ("1", "true", "yes")


def _save_mujoco_checkpoint_npz(
    run_dir: Path,
    checkpoint: dict[str, Any],
    lf_run: Any,
    tag: str,
    *,
    run_seed_layout: int | None = None,
) -> None:
    """Write sim_flat + chain_stage (+ microwave_turned_on) for hot-start / offline replay."""
    if not _mujoco_checkpoint_save_enabled():
        return
    outp = run_dir / f"mujoco_checkpoint__{tag}.npz"
    ck = checkpoint
    arrays: dict[str, Any] = {
        "sim_flat": np.asarray(ck["sim_flat"], dtype=np.float64),
        "chain_stage": np.int32(int(ck["chain_stage"])),
    }
    if "microwave_turned_on" in ck:
        arrays["microwave_turned_on"] = np.asarray(bool(ck["microwave_turned_on"]), dtype=np.bool_)
    if run_seed_layout is not None:
        arrays["run_seed_layout"] = np.int32(int(run_seed_layout))
    if ck.get("outer_episode_idx") is not None:
        arrays["outer_episode_idx"] = np.int32(int(ck["outer_episode_idx"]))
    np.savez_compressed(outp, **arrays)
    _orc_log(lf_run, f"saved mujoco checkpoint {outp.name} ({int(arrays['sim_flat'].nbytes)} B sim_flat)", stderr_too=False)


def _attempt_dir(run_dir: Path, stage_idx: int, retry_idx: int) -> Path:
    return run_dir / f"stage_{stage_idx:02d}" / f"retry_{retry_idx:02d}"


def _jsonable(x: Any) -> Any:
    """Привести значение к виду, годному для json.dumps (numpy, tuple, неизвестные типы)."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def _policy_cfg_snapshot(cfg: PolicyEvalConfig) -> dict[str, Any]:
    """Снимок PolicyEvalConfig полями dataclass (для отладки ранов)."""
    out: dict[str, Any] = {}
    try:
        for f in dataclasses.fields(cfg):
            if f.name.startswith("_"):
                continue
            out[f.name] = _jsonable(getattr(cfg, f.name))
    except TypeError:
        out["error"] = "not_a_dataclass"
    return out


def _write_run_initial_scene(run_dir: Path, env: Any, *, run_seed_base: int, cfg: PolicyEvalConfig) -> None:
    payload = {
        "written_utc": _utc_now_compact(),
        "run_seed_layout": run_seed_base,
        "chain_stage_after_reset": int(getattr(env, "chain_stage")),
        "ep_meta_after_first_reset": _jsonable(env.get_ep_meta()),
        "policy_eval_config": _policy_cfg_snapshot(cfg),
    }
    (run_dir / "env_initial_reset.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _write_attempt_scene_bundle(
    adir: Path,
    env: Any,
    *,
    stage_idx: int,
    horizon_name: str,
    retry_idx: int,
    irun: int,
    attempt_seed: int,
    run_seed_base: int,
    cfg_policy: PolicyEvalConfig,
) -> None:
    ep_meta = env.get_ep_meta()
    bundle: dict[str, Any] = {
        "written_utc": _utc_now_compact(),
        "outer_run_index": irun,
        "run_seed_layout": run_seed_base,
        "stage_idx": stage_idx,
        "horizon_name": horizon_name,
        "retry_idx": retry_idx,
        "attempt_seed_policy": attempt_seed,
        "chain_stage": int(getattr(env, "chain_stage")),
        "instruction_lang": ep_meta.get("lang"),
        "ep_meta": _jsonable(ep_meta),
        "policy_eval_config": _policy_cfg_snapshot(cfg_policy),
    }
    objs = getattr(env, "objects", None)
    if isinstance(objs, dict):
        rt: dict[str, Any] = {}
        for k, o in objs.items():
            ent: dict[str, Any] = {"repr": repr(o)[:800]}
            for attr in ("name", "root_body", "mjcf_model"):
                if hasattr(o, attr):
                    try:
                        ent[attr] = str(getattr(o, attr))
                    except Exception:
                        ent[attr] = "<err>"
            rt[str(k)] = ent
        bundle["objects_runtime"] = rt
    (adir / "attempt_scene.json").write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _console_dup_enabled() -> bool:
    raw = os.environ.get("COSMOS_SEQ_CONSOLE_LOG", "1").strip().lower()
    return raw not in ("0", "false", "no", "")


def _orc_log(lf_run: Any, msg: str, *, stderr_too: bool = True, flush: bool = True) -> None:
    """Понятные строки в ENV_EVAL (+ опционально stderr для docker attach)."""
    line = f"[chain_retry {_utc_now_compact()}] {msg}"
    log_message(line, lf_run)
    try:
        if flush and hasattr(lf_run, "flush"):
            lf_run.flush()
    except Exception:
        pass
    if stderr_too and _console_dup_enabled():
        print(line, file=sys.stderr, flush=True)


def _lf_flush(lf_run: Any) -> None:
    try:
        if hasattr(lf_run, "flush"):
            lf_run.flush()
    except Exception:
        pass


def _run_episode_with_heartbeat(lf_run: Any, label: str, fn: Any) -> Any:
    """Пока run_episode блокируется на T5/CUDA/диффузии, пишет HEARTBEAT в лог каждые N сек."""
    interval = float(os.environ.get("COSMOS_SEQ_HEARTBEAT_SEC", "20"))
    if interval <= 0:
        return fn()
    stop_evt = threading.Event()
    t0 = time.perf_counter()
    ticks: list[int] = [0]

    def _beat() -> None:
        while not stop_evt.wait(interval):
            ticks[0] += 1
            dt = time.perf_counter() - t0
            _orc_log(
                lf_run,
                f"HEARTBEAT tick={ticks[0]} elapsed_s≈{dt:.1f}: всё ещё внутри `{label}` "
                "(первый T5/transformers/Cosmos forward часто 1–15+ мин; дальше обычно быстрее).",
                stderr_too=True,
            )

    th = threading.Thread(target=_beat, name="cosmos_seq_hb", daemon=True)
    th.start()
    try:
        return fn()
    finally:
        stop_evt.set()
        th.join(timeout=min(interval + 2.0, 30.0))


def main_with_cfg(cfg: PolicyEvalConfig) -> int:
    num_stages, stage_horizon_names = resolve_chain_stages(cfg.task_name)

    hot_npz = os.environ.get("COSMOS_SEQ_HOTSTART_NPZ", "").strip()
    hot_first_raw = os.environ.get("COSMOS_SEQ_HOTSTART_FIRST_STAGE", "").strip()
    hot_first: int | None = int(hot_first_raw) if hot_first_raw != "" else None
    hot_all = os.environ.get("COSMOS_SEQ_HOTSTART_ALL_RUNS", "").strip().lower() in ("1", "true", "yes")
    hot_snap: dict[str, Any] | None = None
    hot_rs: int | None = None
    hot_ep: int | None = None
    if hot_npz:
        if hot_first is None:
            print(
                "cosmos_sequential_chain_retries: set COSMOS_SEQ_HOTSTART_FIRST_STAGE (e.g. 3) with COSMOS_SEQ_HOTSTART_NPZ",
                file=sys.stderr,
            )
            return 2
        hot_snap, hot_rs, hot_ep = _read_hotstart_npz(hot_npz)

    chain_runs = int(os.environ.get("COSMOS_SEQ_CHAIN_RUNS", "10"))
    stage_retries = int(os.environ.get("COSMOS_SEQ_STAGE_RETRIES", "3"))
    master_seed_raw = os.environ.get("COSMOS_SEQ_MASTER_SEED", "").strip()
    rng = np.random.default_rng(None if master_seed_raw == "" else int(master_seed_raw))

    if cfg.deterministic:
        os.environ["DETERMINISTIC"] = "True"
    else:
        os.environ.pop("DETERMINISTIC", None)

    validate_config(cfg)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)

    model, cosmos_config = get_model(cfg)
    assert cfg.chunk_size == cosmos_config.dataloader_train.dataset.chunk_size, (
        f"chunk_size mismatch: cfg={cfg.chunk_size} train={cosmos_config.dataloader_train.dataset.chunk_size}"
    )
    if cfg.planning_model_ckpt_path != "":
        planning_model, _ = get_planning_model(cfg)
    else:
        planning_model = None

    base_log_path = Path(cfg.local_log_dir)
    base_log_path.mkdir(parents=True, exist_ok=True)
    manifest_path = base_log_path / "experiment_manifest.json"

    manifest: dict[str, Any] = {
        "experiment_root": str(base_log_path),
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "chain_runs": chain_runs,
        "stage_retries": stage_retries,
        "num_stages": num_stages,
        "stage_horizon_names": list(stage_horizon_names),
        "policy_cfg_seed_field": cfg.seed,
        "layout_and_style_ids": cfg.layout_and_style_ids,
        "hotstart_npz": hot_npz or None,
        "hotstart_first_stage": hot_first,
        "hotstart_all_runs": bool(hot_all),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Единый «шпаргалочный» лог на весь эксперимент в корень (дополнительно к per-run ENV_EVAL).
    orc_summ = base_log_path / "orchestrator_timeline.txt"
    with orc_summ.open("w", encoding="utf-8") as otl:
        otl.write(
            "Оркестратор cosmos_sequential_chain_retries\n"
            "- Подробный текст по каждому run также в run_NNN/ENV_EVAL-*.txt.\n"
            "- Этот файл: краткая шкала времени.\n\n"
        )

    def _timeline(msg: str) -> None:
        line = f"{_utc_now_compact()}  {msg}\n"
        with orc_summ.open("a", encoding="utf-8") as otl:
            otl.write(line)

    def _append_rollout_frame_from_env_robot_obs(
        env: Any,
        cfg0: PolicyEvalConfig,
        rp: list,
        rs: list,
        rw: list,
    ) -> None:
        """After arm-home ``sim.forward`` substeps, refresh cached obs and append RGB (same layout as run_episode)."""
        try:
            obs = env._get_observations()
        except Exception:
            return
        observation = prepare_observation(obs, cfg0.flip_images)
        pi, si, wi = observation["primary_image"], observation["secondary_image"], observation["wrist_image"]
        if pi is not None:
            rp.append(np.array(pi, copy=True, order="C"))
        if si is not None:
            rs.append(np.array(si, copy=True, order="C"))
        if wi is not None:
            rw.append(np.array(wi, copy=True, order="C"))

    global_outcomes: list[dict[str, Any]] = []

    if _console_dup_enabled():
        print(
            f"[chain_retry {_utc_now_compact()}] Model loaded. Outer runs={chain_runs}, "
            f"retries/stage≤{stage_retries}, stages={[f'{i}:{n}' for i, n in enumerate(stage_horizon_names)]}, "
            f"layout_ids={cfg.layout_and_style_ids!r}",
            file=sys.stderr,
            flush=True,
        )
    _timeline(
        f"model_loaded chain_runs={chain_runs} stage_retries_cap={stage_retries} "
        f"layout_ids={cfg.layout_and_style_ids!r}"
    )

    for irun in range(chain_runs):
        use_hot_now = hot_snap is not None and (irun == 0 or hot_all)
        if use_hot_now:
            run_seed_base = int(hot_rs) if hot_rs is not None else int(rng.integers(0, 2**31 - 1))
            ep_idx = int(hot_ep) if hot_ep is not None else irun
            stage0 = int(hot_first)  # type: ignore[arg-type]
        else:
            run_seed_base = int(rng.integers(0, 2**31 - 1))
            ep_idx = irun
            stage0 = 0

        run_dir = base_log_path / f"run_{irun:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_rollout_primary: list = []
        run_rollout_secondary: list = []
        run_rollout_wrist: list = []

        set_seed_everywhere(cfg.seed)

        lf_run, lf_path_run, _ = setup_logging(
            cfg=cfg,
            task_identifier=cfg.task_name,
            log_dir=str(run_dir),
            run_id_note=(cfg.run_id_note or "") + f"-retryRun{irun:03d}-baseSeed{run_seed_base}",
            use_wandb=cfg.use_wandb,
            wandb_entity=cfg.wandb_entity,
            wandb_project=cfg.wandb_project,
        )
        _orc_log(
            lf_run,
            f"===== outer_run {irun + 1}/{chain_runs} | env_spawn_seed≈{run_seed_base} | "
            f"log={lf_path_run} | artifacts={run_dir} =====",
            stderr_too=True,
        )
        _timeline(f"run_{irun:03d}_start spawn_seed≈{run_seed_base}")

        t_env = time.perf_counter()
        env, _ = create_robocasa_env(cfg, seed=run_seed_base, episode_idx=ep_idx)
        if cfg.deterministic_reset:
            rs = cfg.deterministic_reset_seed if cfg.deterministic_reset_seed is not None else cfg.seed
            set_seed_everywhere(rs)
        else:
            _orc_log(lf_run, f"deterministic_reset=False (env stochasticity может отличаться от ожиданий).")

        env.reset()
        if use_hot_now:
            assert hot_snap is not None and hot_first is not None
            _env_restore(env, hot_snap)
            if int(env.chain_stage) != int(hot_first):
                raise RuntimeError(
                    f"hotstart: env.chain_stage={env.chain_stage} != COSMOS_SEQ_HOTSTART_FIRST_STAGE={hot_first} "
                    f"(npz={hot_npz!r})"
                )
            _orc_log(
                lf_run,
                f"hotstart MuJoCo from {hot_npz!r} | first_stage={stage0} | layout_seed={run_seed_base} | "
                f"episode_idx={ep_idx} | chain_stage={int(getattr(env, 'chain_stage'))}",
                stderr_too=True,
            )
        checkpoint_before_stage = _snapshot_for_checkpoint(env, irun)
        try:
            _write_run_initial_scene(run_dir, env, run_seed_base=run_seed_base, cfg=cfg)
        except Exception as e:
            _orc_log(lf_run, f"warn: env_initial_reset.json не записан: {e!r}", stderr_too=False)
        _orc_log(
            lf_run,
            f"env.reset OK | create_robocasa_wall_s={time.perf_counter() - t_env:.2f} | "
            f"checkpoint.chain_stage={checkpoint_before_stage['chain_stage']} | "
            f"Mujoco_flat_state_bytes≈{int(checkpoint_before_stage['sim_flat'].nbytes)}",
            stderr_too=True,
        )
        _ck0_tag = f"after_hotstart_chain_stage_{stage0:02d}" if use_hot_now else "after_reset_chain_stage_00"
        _save_mujoco_checkpoint_npz(
            run_dir,
            checkpoint_before_stage,
            lf_run,
            _ck0_tag,
            run_seed_layout=run_seed_base,
        )

        run_ok = True
        stage_records: list[dict[str, Any]] = []

        for stage_idx in range(stage0, num_stages):
            assert env.chain_stage == stage_idx
            horizon_name = stage_horizon_names[stage_idx]
            max_steps_horizon = TASK_MAX_STEPS.get(horizon_name, 500)
            succeeded = False
            attempt_metas: list[dict[str, Any]] = []

            for retry_idx in range(stage_retries):
                _orc_log(
                    lf_run,
                    f"-- stage={stage_idx} ({horizon_name}) retry={retry_idx}/{stage_retries - 1} | "
                    f"checkpoint.chain_stage={checkpoint_before_stage['chain_stage']} | restoring sim…",
                    stderr_too=(retry_idx == 0),
                )
                tr0 = time.perf_counter()
                _env_restore(env, checkpoint_before_stage)
                _orc_log(
                    lf_run,
                    f"Mujoco restore OK in {(time.perf_counter() - tr0) * 1000:.1f} ms (same physics as checkpoint).",
                    stderr_too=False,
                )

                attempt_seed = int(rng.integers(0, 2**31 - 1))
                cfg_attempt = dataclasses.replace(cfg, seed=attempt_seed)
                set_seed_everywhere(attempt_seed)

                adir = _attempt_dir(run_dir, stage_idx, retry_idx)
                adir.mkdir(parents=True, exist_ok=True)
                cfg_logged = dataclasses.replace(cfg_attempt, local_log_dir=str(adir))

                task_description = env.get_ep_meta()["lang"]
                try:
                    _write_attempt_scene_bundle(
                        adir,
                        env,
                        stage_idx=stage_idx,
                        horizon_name=horizon_name,
                        retry_idx=retry_idx,
                        irun=irun,
                        attempt_seed=attempt_seed,
                        run_seed_base=run_seed_base,
                        cfg_policy=cfg_attempt,
                    )
                except Exception as e:
                    _orc_log(lf_run, f"warn: attempt_scene.json не записан: {e!r}", stderr_too=False)
                ep_log_idx = irun * num_stages + stage_idx
                _orc_log(
                    lf_run,
                    f"attemptdir={adir} | cfg.seed(политики)={attempt_seed} | "
                    f'instruction="{task_description}" | horizon_name={horizon_name} | '
                    f"max_steps≈{max_steps_horizon} (внутри run_episode ~10 warmup step нулём)",
                    stderr_too=True,
                )
                _orc_log(
                    lf_run,
                    "Дальше может долго не печататься: считаются T5/диффузия/query на первые чанки — "
                    "смотри строки ниже из run_robocasa_eval («Query», «Success», «Episode … SUCCESS/FAIL»). "
                    "Прогресс симулятора дублируется в stderr как `print(t: …)` если не отключено уровнем лога.",
                    stderr_too=True,
                )
                _timeline(
                    f"run_{irun:03d} stage_{stage_idx:02d} retry_{retry_idx:02d} start policy_seed={attempt_seed}"
                )

                t_ep = time.perf_counter()
                hb_label = (
                    f"run_episode stage={stage_idx}/{num_stages - 1} retry={retry_idx} "
                    f"horizon={horizon_name} irun={irun}"
                )

                def _do_episode() -> Any:
                    return run_episode(
                        cfg_logged,
                        env,
                        task_description,
                        model,
                        planning_model,
                        dataset_stats,
                        None,
                        irun,
                        lf_run,
                        episode_log_index=ep_log_idx,
                        task_name_for_horizon=horizon_name,
                    )

                env._chain4_turnoff_debug_fp = None
                # Must match robocasa ``_CHAIN_TURNOFF_DEBUG_STAGE_BY_CLASS`` (TurnOffMicrowave horizon index).
                _turnoff_debug_stage_idx = {
                    "PnPRoboarmCosmosChain4MicrowaveCloseOnOffOpen": 2,
                    "PnPRoboarmCosmosChain6PotatoMwPlate": 3,
                }.get(cfg.task_name)
                if _turnoff_debug_stage_idx is not None and stage_idx == _turnoff_debug_stage_idx:
                    _tdbg = adir / "turnoff_stage_debug.log"
                    env._chain4_turnoff_debug_fp = _tdbg.open("w", encoding="utf-8")
                    env._chain4_turnoff_min_eef = float("inf")
                    env._chain4_turnoff_min_finger_geom = float("inf")
                    env._chain4_turnoff_min_best = float("inf")
                    env._chain4_turnoff_contact_ever = False
                try:
                    (
                        success,
                        ep_len,
                        replay_primary_images,
                        replay_secondary_images,
                        replay_wrist_images,
                        future_image_predictions_list,
                        _collected,
                    ) = _run_episode_with_heartbeat(lf_run, hb_label, _do_episode)
                finally:
                    _dfp = getattr(env, "_chain4_turnoff_debug_fp", None)
                    if _dfp is not None:
                        _dfp.flush()
                        _dfp.close()
                        delattr(env, "_chain4_turnoff_debug_fp")
                ep_wall = time.perf_counter() - t_ep
                _orc_log(
                    lf_run,
                    f"run_episode завершился: SUCCESS={success} length={ep_len} wall_time_s≈{ep_wall:.2f}",
                    stderr_too=True,
                )
                _lf_flush(lf_run)

                _k_chain3.write_chain4_turnoff_stage_debug_log(adir, env, bool(success))

                _save_stage_rollouts(
                    cfg_logged,
                    stage_idx,
                    irun,
                    success,
                    task_description,
                    replay_primary_images,
                    replay_secondary_images,
                    replay_wrist_images,
                    future_image_predictions_list,
                    lf_run,
                )
                _orc_log(lf_run, f"видеокадры сохранены в {adir.resolve()}/")
                _lf_flush(lf_run)

                _em = env.get_ep_meta()
                am = {
                    "stage_idx": stage_idx,
                    "retry_idx": retry_idx,
                    "attempt_seed": attempt_seed,
                    "success": bool(success),
                    "episode_length": int(ep_len),
                    "attempt_dir": str(adir.relative_to(run_dir)),
                    "instruction_lang": task_description,
                    "layout_id": _em.get("layout_id"),
                    "style_id": _em.get("style_id"),
                    "horizon_name": horizon_name,
                }
                (adir / "attempt_meta.json").write_text(json.dumps(am, indent=2))
                attempt_metas.append(am)

                if success:
                    succeeded = True
                    run_rollout_primary.extend(replay_primary_images)
                    run_rollout_secondary.extend(replay_secondary_images)
                    run_rollout_wrist.extend(replay_wrist_images)
                    if stage_idx < num_stages - 1:
                        try:
                            env._chain_arm_home_capture_cb = lambda e: _append_rollout_frame_from_env_robot_obs(
                                e, cfg, run_rollout_primary, run_rollout_secondary, run_rollout_wrist
                            )
                            env.advance_chain_stage()
                        finally:
                            if hasattr(env, "_chain_arm_home_capture_cb"):
                                delattr(env, "_chain_arm_home_capture_cb")
                        checkpoint_before_stage = _snapshot_for_checkpoint(env, irun)
                        _orc_log(
                            lf_run,
                            f"стадия {stage_idx} SUCCESS → advance_chain_stage OK | "
                            f"новый checkpoint.chain_stage={checkpoint_before_stage['chain_stage']}",
                            stderr_too=True,
                        )
                        ncs = int(checkpoint_before_stage["chain_stage"])
                        _save_mujoco_checkpoint_npz(
                            run_dir,
                            checkpoint_before_stage,
                            lf_run,
                            f"after_stage{stage_idx:02d}_success_chain_stage_{ncs:02d}",
                            run_seed_layout=run_seed_base,
                        )
                        if cfg.task_name == "PnPRoboarmCosmosChain4MicrowaveCloseOnOffOpen" and stage_idx == 1:
                            outp2 = run_dir / "chain4_after_start_success.npz"
                            ck = checkpoint_before_stage
                            np.savez_compressed(
                                outp2,
                                sim_flat=np.asarray(ck["sim_flat"], dtype=np.float64),
                                chain_stage=np.int32(int(ck["chain_stage"])),
                                microwave_turned_on=np.asarray(
                                    bool(ck.get("microwave_turned_on", True)), dtype=np.bool_
                                ),
                                run_seed_layout=np.int32(int(run_seed_base)),
                                outer_episode_idx=np.int32(int(irun)),
                            )
                            _orc_log(
                                lf_run,
                                f"saved alias for stop-only script: {outp2.name}",
                                stderr_too=True,
                            )
                        if cfg.task_name == "PnPRoboarmCosmosChain6PotatoMwPlate" and stage_idx == 2:
                            outp2 = run_dir / "chain6_after_start_success.npz"
                            ck = checkpoint_before_stage
                            np.savez_compressed(
                                outp2,
                                sim_flat=np.asarray(ck["sim_flat"], dtype=np.float64),
                                chain_stage=np.int32(int(ck["chain_stage"])),
                                microwave_turned_on=np.asarray(
                                    bool(ck.get("microwave_turned_on", True)), dtype=np.bool_
                                ),
                                run_seed_layout=np.int32(int(run_seed_base)),
                                outer_episode_idx=np.int32(int(irun)),
                            )
                            _orc_log(
                                lf_run,
                                f"saved alias for chain6 stop-only script: {outp2.name}",
                                stderr_too=True,
                            )
                    else:
                        _orc_log(lf_run, "последняя стадия SUCCESS — финиш цепочки.", stderr_too=True)
                        snap_fin = _snapshot_for_checkpoint(env, irun)
                        _save_mujoco_checkpoint_npz(
                            run_dir,
                            snap_fin,
                            lf_run,
                            f"after_stage{stage_idx:02d}_success_terminal",
                            run_seed_layout=run_seed_base,
                        )
                    break
                else:
                    _orc_log(
                        lf_run,
                        f"стадия {stage_idx}: попытка {retry_idx} провалена ({stage_retries - retry_idx - 1} попытки остаются или конец retry).",
                        stderr_too=True,
                    )

            stage_records.append({"stage_idx": stage_idx, "succeeded": succeeded, "attempts": attempt_metas})
            _orc_log(lf_run, f"итог стадии {stage_idx} ({horizon_name}): success={succeeded}", stderr_too=True)
            if not succeeded:
                run_ok = False
                log_message(f"Run {irun}: abort after stage {stage_idx} exhaustion.", lf_run)
                break

        if run_ok and run_rollout_primary:
            try:
                save_rollout_video(
                    run_rollout_primary,
                    run_rollout_secondary,
                    run_rollout_wrist,
                    irun,
                    success=True,
                    task_description="full_run_episode_including_arm_home",
                    rollout_data_dir=str(run_dir),
                    log_file=lf_run,
                )
            except Exception as e:
                _orc_log(lf_run, f"warn: full-run video not saved: {e!r}", stderr_too=False)

        env.close()
        run_summary = {
            "irun": irun,
            "run_seed_layout": run_seed_base,
            "run_all_stages_success": run_ok,
            "stages": stage_records,
            "hotstart_npz": hot_npz if use_hot_now else None,
            "hotstart_first_stage": int(hot_first) if use_hot_now and hot_first is not None else None,
            "episode_idx": ep_idx,
        }
        (run_dir / "run_meta.json").write_text(json.dumps(run_summary, indent=2))
        _orc_log(
            lf_run,
            f"outer_run {irun} завершён: all_stages_OK={run_ok} | см. также run_meta.json",
            stderr_too=True,
        )
        _timeline(f"run_{irun:03d}_end all_segments_success={run_ok}")

        lf_run.close()
        global_outcomes.append(run_summary)

    manifest["finished_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest["runs"] = global_outcomes
    manifest["fraction_runs_full_success"] = float(
        np.mean([1.0 if r["run_all_stages_success"] else 0.0 for r in global_outcomes])
    )
    manifest["orchestrator_timeline_txt"] = str(base_log_path / "orchestrator_timeline.txt")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(
        f"\n[chain_retry {_utc_now_compact()}] Done. Manifest: {manifest_path}\n"
        f"Timeline (кратко по времени): {manifest['orchestrator_timeline_txt']}",
        flush=True,
        file=sys.stderr,
    )

    overall = bool(global_outcomes) and all(r["run_all_stages_success"] for r in global_outcomes)
    return 0 if overall else 1


@draccus.wrap()
def main(cfg: PolicyEvalConfig) -> int:
    try:
        resolve_chain_stages(cfg.task_name)
    except ValueError as e:
        print(f"cosmos_sequential_chain_retries.py: {e}", file=sys.stderr)
        return 2

    if cfg.deterministic:
        os.environ["DETERMINISTIC"] = "True"

    validate_config(cfg)
    set_seed_everywhere(cfg.seed)
    Path(cfg.local_log_dir).mkdir(parents=True, exist_ok=True)
    return main_with_cfg(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
