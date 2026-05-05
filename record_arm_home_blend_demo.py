#!/usr/bin/env python3
"""
Record a short MP4: arm moves from pose A (deliberate offset from episode-init qpos) to pose B
(the chain «0» pose = snapshot taken at env.reset), using the same blend as between chain stages.

No policy, no Cosmos checkpoint — only MuJoCo forward kinematics + offscreen cameras.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import imageio
import numpy as np

import robocasa  # noqa: F401
import robosuite  # noqa: F401

from robocasa.environments.kitchen.custom.kitchen_roboarm_cosmos_chain3 import (
    _restore_chain_init_arm_gripper_smooth,
)

from cosmos_policy.experiments.robot.robocasa.robocasa_utils import (
    _overlay_frame_caption_top_right,
    _resize_uint8_hwc,
)
from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import (
    PolicyEvalConfig,
    create_robocasa_env,
    prepare_observation,
    render_robocasa_eval_camera_triplet,
)


def _parse_delta(s: str) -> np.ndarray:
    parts = [float(x.strip()) for x in s.split(",") if x.strip()]
    return np.asarray(parts, dtype=np.float64)


def _append_frame(env, cfg: PolicyEvalConfig, rp: list, rs: list, rw: list) -> None:
    pi, si, wi = render_robocasa_eval_camera_triplet(
        env, env_img_res=int(cfg.env_img_res), flip_images=bool(cfg.flip_images)
    )
    if pi is None or si is None or wi is None:
        obs = env._get_observations()
        o = prepare_observation(obs, cfg.flip_images)
        pi, si, wi = o.get("primary_image"), o.get("secondary_image"), o.get("wrist_image")
    for im, buf in ((pi, rp), (si, rs), (wi, rw)):
        if im is not None:
            buf.append(np.asarray(im, copy=True, order="C"))


def main() -> int:
    p = argparse.ArgumentParser(description="MP4: arm A→init (no AI).")
    p.add_argument("--task_name", type=str, default="PnPRoboarmCosmosChain6PotatoMwPlate")
    p.add_argument("--layout_and_style_ids", type=str, default="((5,0),)")
    p.add_argument("--seed", type=int, default=15251)
    p.add_argument("--episode_idx", type=int, default=0)
    p.add_argument(
        "--out",
        type=str,
        default="/data/logs/arm_home_blend_demo.mp4",
        help="Path inside container (host: data/cosmos_robocasa/logs/... if using compose mounts).",
    )
    p.add_argument(
        "--joint_delta",
        type=str,
        default="0.55,-0.45,0.65,-0.55,0.5,-0.35,0.2",
        help="Comma-separated radians added to init arm qpos to form pose A (truncated/padded to arm DoF).",
    )
    p.add_argument("--hold_frames", type=int, default=8, help="How many identical frames to hold at pose A before blending.")
    p.add_argument("--duration_s", type=float, default=-1.0, help="Blend duration; <0 uses CHAIN_ARM_HOME_DURATION_S from env module.")
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    dur = None if args.duration_s < 0 else float(args.duration_s)

    cfg = PolicyEvalConfig(
        task_name=args.task_name,
        layout_and_style_ids=args.layout_and_style_ids,
        unnormalize_actions=False,
        normalize_proprio=False,
        flip_images=True,
    )
    env, _ = create_robocasa_env(cfg, seed=int(args.seed), episode_idx=int(args.episode_idx))
    env.reset()

    robot = env.robots[0]
    arm_i = np.asarray(robot._ref_arm_joint_pos_indexes, dtype=int)
    delta = _parse_delta(args.joint_delta)
    if delta.size < arm_i.size:
        delta = np.pad(delta, (0, arm_i.size - delta.size))
    else:
        delta = delta[: arm_i.size]

    # Pose A: episode-init arm + delta (same object layout; only arm moved in qpos).
    env.sim.data.qpos[arm_i] = np.asarray(env._chain_init_arm_qpos, dtype=np.float64) + delta
    arm_vi = np.asarray(getattr(robot, "_ref_arm_joint_vel_indexes", []), dtype=int)
    if arm_vi.size:
        env.sim.data.qvel[arm_vi] = 0.0
    env.sim.forward()

    rp: list = []
    rs: list = []
    rw: list = []
    cap_lbl_a = "POSE_A (init+delta)"
    cap_blend = "BLEND A→init (no policy)"
    for _ in range(max(0, int(args.hold_frames))):
        _append_frame(env, cfg, rp, rs, rw)

    n_before = len(rp)

    def _cb(e):
        _append_frame(e, cfg, rp, rs, rw)

    env._chain_arm_home_capture_cb = _cb
    try:
        _restore_chain_init_arm_gripper_smooth(env, duration_s=dur)
    finally:
        if hasattr(env, "_chain_arm_home_capture_cb"):
            delattr(env, "_chain_arm_home_capture_cb")

    n_blend = len(rp) - n_before
    captions = [cap_lbl_a] * n_before + [cap_blend] * max(n_blend, 0)
    if len(captions) != len(rp):
        captions = None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # save_rollout_video builds its own filename with DATE_TIME; write a second clip with fixed name via imageio
    panel_h = panel_w = 0
    for a, b, c in zip(rp, rs, rw):
        for im in (a, b, c):
            u = np.asarray(im)
            panel_h = max(panel_h, int(u.shape[0]))
            panel_w = max(panel_w, int(u.shape[1]))
    if panel_h == 0 or panel_w == 0:
        panel_h = panel_w = 224

    w = imageio.get_writer(str(out_path), fps=int(args.fps))
    try:
        for i, (a, b, c) in enumerate(zip(rp, rs, rw)):
            row = np.concatenate(
                [
                    _resize_uint8_hwc(a, panel_w, panel_h),
                    _resize_uint8_hwc(b, panel_w, panel_h),
                    _resize_uint8_hwc(c, panel_w, panel_h),
                ],
                axis=1,
            )
            if captions is not None:
                row = _overlay_frame_caption_top_right(row, captions[i])
            w.append_data(np.ascontiguousarray(row))
    finally:
        w.close()

    env.close()
    print(f"Wrote {len(rp)} frames to {out_path.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
