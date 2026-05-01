#!/usr/bin/env python3
"""
One MuJoCo env: multi-stage kitchen-roboarm chains, **no env.reset()** between stages.

Supported ``--task_name`` values are registered env classes that define
``CHAIN_STAGE_HORIZON_NAMES`` (e.g. ``PnPRoboarmCosmosChain3``).

Invoked inside the cosmos-robocasa container (same deps as run_robocasa_eval).
Host: ``scripts/cosmos_sequential_chain/run_one_reset_chain.sh`` (compose mounts this file to
``/opt/cosmos-policy/cosmos_sequential_one_reset_chain.py``).
"""

import os
import sys

import draccus

# Register RoboCasa envs (including PnPRoboarmCosmosChain3) before robosuite.make.
import robocasa  # noqa: F401
import robocasa.environments.kitchen.custom.kitchen_roboarm_cosmos_chain3  # noqa: F401  # register env

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_model,
    get_planning_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)
from cosmos_policy.experiments.robot.robocasa.robocasa_utils import (
    save_rollout_video,
    save_rollout_video_with_future_image_predictions,
)
from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import (
    PolicyEvalConfig,
    create_robocasa_env,
    run_episode,
    validate_config,
)
from cosmos_policy.experiments.robot.robot_utils import DATE_TIME, log_message, setup_logging
from cosmos_policy.utils.utils import set_seed_everywhere


def resolve_chain_stages(task_name: str) -> tuple[int, tuple[str, ...]]:
    """Return (num_stages, horizon names per stage) for a registered chain env."""
    from robocasa.environments.kitchen.kitchen import REGISTERED_KITCHEN_ENVS

    cls = REGISTERED_KITCHEN_ENVS.get(task_name)
    if cls is None:
        raise ValueError(f"Unknown kitchen env {task_name!r}")
    if not hasattr(cls, "CHAIN_STAGE_HORIZON_NAMES"):
        raise ValueError(
            f"Task {task_name!r} has no CHAIN_STAGE_HORIZON_NAMES; only multi-stage chain envs are supported."
        )
    names = tuple(getattr(cls, "CHAIN_STAGE_HORIZON_NAMES"))
    return len(names), names


def _save_stage_rollouts(
    cfg: PolicyEvalConfig,
    stage_idx: int,
    trial_idx: int,
    success: bool,
    task_description: str,
    replay_primary_images,
    replay_secondary_images,
    replay_wrist_images,
    future_image_predictions_list,
    log_file,
):
    slug = f"{stage_idx + 1:03d}_{cfg.task_name}--stage{stage_idx:02d}--{DATE_TIME}"
    rollout_data_dir = os.path.join(cfg.local_log_dir, "rollout_data", slug)
    os.makedirs(rollout_data_dir, exist_ok=True)
    save_rollout_video(
        replay_primary_images,
        replay_secondary_images,
        replay_wrist_images,
        trial_idx,
        success=success,
        task_description=task_description,
        rollout_data_dir=rollout_data_dir,
        log_file=log_file,
    )
    if len(future_image_predictions_list) > 0:
        fp = fs = fw = None
        if (
            "future_image" in future_image_predictions_list[0]
            and future_image_predictions_list[0]["future_image"] is not None
        ):
            fp = [x["future_image"] for x in future_image_predictions_list]
        if (
            "future_image2" in future_image_predictions_list[0]
            and future_image_predictions_list[0]["future_image2"] is not None
        ):
            fs = [x["future_image2"] for x in future_image_predictions_list]
        if (
            "future_wrist_image" in future_image_predictions_list[0]
            and future_image_predictions_list[0]["future_wrist_image"] is not None
        ):
            fw = [x["future_wrist_image"] for x in future_image_predictions_list]
        if fp is not None and fs is not None and fw is not None:
            save_rollout_video_with_future_image_predictions(
                replay_primary_images,
                replay_secondary_images,
                replay_wrist_images,
                trial_idx,
                success=success,
                task_description=task_description,
                rollout_data_dir=rollout_data_dir,
                chunk_size=cfg.chunk_size,
                num_open_loop_steps=cfg.num_open_loop_steps,
                future_primary_image_predictions=fp,
                future_secondary_image_predictions=fs,
                future_wrist_image_predictions=fw,
                show_diff=False,
                log_file=log_file,
                show_timestep=True,
            )


@draccus.wrap()
def main(cfg: PolicyEvalConfig) -> int:
    try:
        num_stages, stage_horizon_names = resolve_chain_stages(cfg.task_name)
    except ValueError as e:
        print(f"cosmos_sequential_one_reset_chain.py: {e}", file=sys.stderr)
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

    log_file, local_log_filepath, _run_id = setup_logging(
        cfg=cfg,
        task_identifier=cfg.task_name,
        log_dir=cfg.local_log_dir,
        run_id_note=cfg.run_id_note,
        use_wandb=cfg.use_wandb,
        wandb_entity=cfg.wandb_entity,
        wandb_project=cfg.wandb_project,
    )
    log_message(
        "\nSequential one-reset chain: one env.reset() at episode start only; "
        "between stages only advance_chain_stage() (same MuJoCo state).",
        log_file,
    )
    log_message(f"Eval config: {cfg}", log_file)

    all_trial_ok: list[bool] = []
    for trial_idx in range(cfg.num_trials_per_task):
        log_message(f"\n=== Chain trial {trial_idx + 1}/{cfg.num_trials_per_task} ===", log_file)
        if cfg.deterministic or cfg.deterministic_reset:
            seed = cfg.seed * trial_idx * 256
        else:
            seed = None
        env, _ = create_robocasa_env(cfg, seed=seed, episode_idx=trial_idx)
        if cfg.deterministic_reset:
            rs = cfg.deterministic_reset_seed if cfg.deterministic_reset_seed is not None else cfg.seed
            set_seed_everywhere(rs)
        env.reset()
        trial_ok = True
        for stage_idx in range(num_stages):
            assert env.chain_stage == stage_idx, (env.chain_stage, stage_idx)
            task_description = env.get_ep_meta()["lang"]
            log_message(f"\n--- Stage {stage_idx + 1}/{num_stages} lang={task_description!r} ---", log_file)
            ep_log_idx = trial_idx * num_stages + stage_idx
            horizon_name = stage_horizon_names[stage_idx]
            (
                success,
                _length,
                replay_primary_images,
                replay_secondary_images,
                replay_wrist_images,
                future_image_predictions_list,
                _collected,
            ) = run_episode(
                cfg,
                env,
                task_description,
                model,
                planning_model,
                dataset_stats,
                None,
                trial_idx,
                log_file,
                episode_log_index=ep_log_idx,
                task_name_for_horizon=horizon_name,
            )
            _save_stage_rollouts(
                cfg,
                stage_idx,
                trial_idx,
                success,
                task_description,
                replay_primary_images,
                replay_secondary_images,
                replay_wrist_images,
                future_image_predictions_list,
                log_file,
            )
            if not success:
                trial_ok = False
                log_message(f"Stage {stage_idx} failed; stopping trial (no further resets).", log_file)
                break
            if stage_idx < num_stages - 1:
                env.advance_chain_stage()
        env.close()
        all_trial_ok.append(trial_ok)
        log_message(f"Chain trial {trial_idx} all_segments_success={trial_ok}", log_file)

    overall = bool(all_trial_ok) and all(all_trial_ok)
    log_message("\n" + "=" * 80, log_file)
    log_message("FINAL: sequential one-reset chain", log_file)
    log_message(f"  trials all success: {overall}", log_file)
    log_message(f"  per-trial: {all_trial_ok}", log_file)
    log_message(f"Results saved to: {local_log_filepath}", log_file)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
