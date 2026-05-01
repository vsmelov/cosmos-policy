#!/usr/bin/env python3
"""
Load a MuJoCo checkpoint (.npz) taken **after** successful «press start» in
``PnPRoboarmCosmosChain6PotatoMwPlate`` (``chain_stage`` = 3) and run only the
``TurnOffMicrowave`` horizon.

Generate the .npz (host)::

  COSMOS_SEQ_SAVE_MUJOCO_CHECKPOINTS=1 ./scripts/cosmos_sequential_chain/run_chain6_potato_mw_plate.sh

Then use ``run_NNN/chain6_after_start_success.npz`` or
``mujoco_checkpoint__after_stage02_success_chain_stage_03.npz`` from the same run.

**Host**::

  export COSMOS_CHAIN6_CHECKPOINT_NPZ=.../run_004/chain6_after_start_success.npz
  export COSMOS_CHAIN6_CHECKPOINT_LAYOUT_SEED=285337752   # run_meta.json run_seed_layout
  # optional if npz lacks outer_episode_idx: same as run_meta irun
  export COSMOS_CHAIN6_CHECKPOINT_EPISODE_IDX=4
  ./scripts/cosmos_sequential_chain/run_chain6_stop_from_checkpoint.sh

``COSMOS_CHAIN6_CHECKPOINT_LAYOUT_SEED`` must match ``create_robocasa_env(..., seed=...)`` for that run.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import draccus
import numpy as np
import robocasa  # noqa: F401
import robocasa.environments.kitchen.custom.kitchen_roboarm_cosmos_chain3 as _k_chain3  # noqa: F401

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
    npz_path = os.environ.get("COSMOS_CHAIN6_CHECKPOINT_NPZ", "").strip()
    if not npz_path:
        print(
            "cosmos_chain6_turnoff_from_checkpoint.py: set COSMOS_CHAIN6_CHECKPOINT_NPZ=/path/to/chain6_after_start_success.npz",
            file=sys.stderr,
        )
        return 2
    if cfg.task_name != "PnPRoboarmCosmosChain6PotatoMwPlate":
        print(
            "cosmos_chain6_turnoff_from_checkpoint.py: --task_name must be PnPRoboarmCosmosChain6PotatoMwPlate",
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
    snap: dict = {"sim_flat": sim_flat, "chain_stage": chain_stage}
    if "microwave_turned_on" in zf.files:
        snap["microwave_turned_on"] = bool(np.asarray(zf["microwave_turned_on"]).reshape(()))

    layout_seed_raw = os.environ.get("COSMOS_CHAIN6_CHECKPOINT_LAYOUT_SEED", "").strip()
    if layout_seed_raw:
        layout_seed = int(layout_seed_raw)
    elif "run_seed_layout" in zf.files:
        layout_seed = int(np.asarray(zf["run_seed_layout"]).reshape(()))
    else:
        layout_seed = int(cfg.seed)

    ep_raw = os.environ.get("COSMOS_CHAIN6_CHECKPOINT_EPISODE_IDX", "").strip()
    if ep_raw:
        episode_idx = int(ep_raw)
    elif "outer_episode_idx" in zf.files:
        episode_idx = int(np.asarray(zf["outer_episode_idx"]).reshape(()))
    else:
        episode_idx = 0

    env, _ = create_robocasa_env(cfg, seed=layout_seed, episode_idx=episode_idx)
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
        run_id_note=(cfg.run_id_note or "") + "-chain6TurnoffFromNpz",
        use_wandb=cfg.use_wandb,
        wandb_entity=cfg.wandb_entity,
        wandb_project=cfg.wandb_project,
    )
    task_description = env.get_ep_meta()["lang"]
    if chain_stage != 3:
        print(
            f"cosmos_chain6_turnoff_from_checkpoint.py: warning: chain_stage in npz is {chain_stage} (expected 3).",
            file=sys.stderr,
        )

    log_dir = Path(cfg.local_log_dir)
    tdbg = log_dir / "turnoff_stage_debug.log"
    env._chain4_turnoff_debug_fp = tdbg.open("w", encoding="utf-8")
    env._chain4_turnoff_min_eef = float("inf")
    env._chain4_turnoff_min_finger_geom = float("inf")
    env._chain4_turnoff_min_best = float("inf")
    env._chain4_turnoff_contact_ever = False
    try:
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
    finally:
        _dfp = getattr(env, "_chain4_turnoff_debug_fp", None)
        if _dfp is not None:
            _dfp.flush()
            _dfp.close()
            delattr(env, "_chain4_turnoff_debug_fp")

    _k_chain3.write_chain4_turnoff_stage_debug_log(log_dir, env, bool(success))
    (log_dir / "chain6_stop_from_npz_meta.json").write_text(
        json.dumps(
            {
                "npz": npz_path,
                "layout_seed": layout_seed,
                "episode_idx": episode_idx,
                "chain_stage_in_npz": chain_stage,
                "success": bool(success),
                "episode_length": int(ep_len),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _save_stage_rollouts(cfg, 3, 0, success, task_description, rpi, rsi, rwi, fut, lf)
    env.close()
    lf.close()
    print(f"cosmos_chain6_turnoff_from_checkpoint: success={success} episode_length={ep_len}", flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
