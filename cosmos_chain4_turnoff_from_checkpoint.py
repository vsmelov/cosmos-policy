#!/usr/bin/env python3
"""
Load ``chain4_after_start_success.npz`` (sim_flat + chain_stage + microwave_turned_on) and run **only**
the ``TurnOffMicrowave`` horizon with Cosmos (same ``PolicyEvalConfig`` / docker stack as retries).

**Generate the .npz once** (after a successful «press start» in chain4 retries)::

  COSMOS_SEQ_SAVE_MUJOCO_CHECKPOINTS=1 ./scripts/cosmos_sequential_chain/run_microwave_close_on_off_open_3runs.sh

Then copy ``run_000/chain4_after_start_success.npz`` from that experiment into a path visible in the container
(typically under ``/data/logs/...`` or mount).

**Run stop-only** (host)::

  export COSMOS_CHAIN4_CHECKPOINT_NPZ=/data/logs/sequential_retries/<exp>/run_000/chain4_after_start_success.npz
  export COSMOS_CHAIN4_CHECKPOINT_LAYOUT_SEED=<same as run_meta.json run_seed_layout for that run>
  ./scripts/cosmos_sequential_chain/run_chain4_stop_from_checkpoint.sh

``COSMOS_CHAIN4_CHECKPOINT_LAYOUT_SEED`` must match ``create_robocasa_env(..., seed=...)`` used when the checkpoint
was recorded, otherwise ``reset()`` layout can diverge before ``set_state_from_flattened``.
"""

import os
import sys

import draccus
import numpy as np
import robocasa  # noqa: F401
import robocasa.environments.kitchen.custom.kitchen_roboarm_cosmos_chain3  # noqa: F401

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_model,
    get_planning_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)
from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import (
    PolicyEvalConfig,
    create_robocasa_env,
    run_episode,
    validate_config,
)
from cosmos_policy.experiments.robot.robot_utils import setup_logging
from cosmos_policy.utils.utils import set_seed_everywhere
from cosmos_sequential_chain_retries import _env_restore
from cosmos_sequential_one_reset_chain import _save_stage_rollouts


@draccus.wrap()
def main(cfg: PolicyEvalConfig) -> int:
    npz_path = os.environ.get("COSMOS_CHAIN4_CHECKPOINT_NPZ", "").strip()
    if not npz_path:
        print(
            "cosmos_chain4_turnoff_from_checkpoint.py: set COSMOS_CHAIN4_CHECKPOINT_NPZ=/path/to/chain4_after_start_success.npz",
            file=sys.stderr,
        )
        return 2
    if cfg.task_name != "PnPRoboarmCosmosChain4MicrowaveCloseOnOffOpen":
        print(
            "cosmos_chain4_turnoff_from_checkpoint.py: --task_name must be PnPRoboarmCosmosChain4MicrowaveCloseOnOffOpen",
            file=sys.stderr,
        )
        return 2

    if cfg.deterministic:
        os.environ["DETERMINISTIC"] = "True"
    else:
        os.environ.pop("DETERMINISTIC", None)

    validate_config(cfg)
    set_seed_everywhere(cfg.seed)
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

    zf = np.load(npz_path, allow_pickle=False)
    sim_flat = np.asarray(zf["sim_flat"], dtype=np.float64).copy()
    chain_stage = int(np.asarray(zf["chain_stage"]).reshape(()))
    snap = {"sim_flat": sim_flat, "chain_stage": chain_stage}
    if "microwave_turned_on" in zf.files:
        snap["microwave_turned_on"] = bool(np.asarray(zf["microwave_turned_on"]).reshape(()))

    layout_seed_raw = os.environ.get("COSMOS_CHAIN4_CHECKPOINT_LAYOUT_SEED", "").strip()
    if layout_seed_raw:
        layout_seed = int(layout_seed_raw)
    elif "run_seed_layout" in zf.files:
        layout_seed = int(np.asarray(zf["run_seed_layout"]).reshape(()))
    else:
        layout_seed = int(cfg.seed)
    env, _ = create_robocasa_env(cfg, seed=layout_seed, episode_idx=0)
    if cfg.deterministic_reset:
        rs = cfg.deterministic_reset_seed if cfg.deterministic_reset_seed is not None else cfg.seed
        set_seed_everywhere(rs)
    env.reset()
    _env_restore(env, snap)

    os.makedirs(cfg.local_log_dir, exist_ok=True)
    lf, _, _ = setup_logging(
        cfg=cfg,
        task_identifier=cfg.task_name,
        log_dir=cfg.local_log_dir,
        run_id_note=(cfg.run_id_note or "") + "-turnoffFromNpz",
        use_wandb=cfg.use_wandb,
        wandb_entity=cfg.wandb_entity,
        wandb_project=cfg.wandb_project,
    )
    task_description = env.get_ep_meta()["lang"]
    if chain_stage != 2:
        print(
            f"cosmos_chain4_turnoff_from_checkpoint.py: warning: chain_stage in npz is {chain_stage} (expected 2).",
            file=sys.stderr,
        )

    success, ep_len, rpi, rsi, rwi, fut, _col = run_episode(
        cfg,
        env,
        task_description,
        model,
        planning_model,
        dataset_stats,
        None,
        0,
        lf,
        episode_log_index=0,
        task_name_for_horizon="TurnOffMicrowave",
    )
    _save_stage_rollouts(cfg, 2, 0, success, task_description, rpi, rsi, rwi, fut, lf)
    env.close()
    lf.close()
    print(f"cosmos_chain4_turnoff_from_checkpoint: success={success} episode_length={ep_len}", flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
