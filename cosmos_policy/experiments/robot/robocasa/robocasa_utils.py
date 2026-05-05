# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils for evaluating policies in RoboCasa simulation environments."""

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cosmos_policy.experiments.robot.robot_utils import DATE_TIME


def _numpy_image_to_uint8_hwc(x: np.ndarray) -> np.ndarray:
    """Coerce simulator or model outputs to contiguous uint8 HWC RGB for PIL / video."""
    a = np.asarray(x)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    elif a.ndim == 3 and a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    if a.dtype != np.uint8:
        f = a.astype(np.float32)
        mn = float(f.min()) if f.size else 0.0
        mx = float(f.max()) if f.size else 0.0
        # Model heads often emit float32 in [0, 1] or [-1, 1]; direct uint8 cast would clip to black.
        if mx <= 1.0 + 1e-3 and mn >= -1.0 - 1e-3 and mn < -1e-3:
            f = (np.clip(f, -1.0, 1.0) + 1.0) * 0.5 * 255.0
        elif mx <= 1.0 + 1e-3:
            f = np.clip(f, 0.0, 1.0) * 255.0
        else:
            f = np.clip(f, 0.0, 255.0)
        a = np.rint(f).astype(np.uint8)
    if a.ndim == 3 and a.shape[2] > 3:
        a = a[..., :3]
    return np.ascontiguousarray(a)


def _resize_uint8_hwc(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    u = _numpy_image_to_uint8_hwc(img)
    pil_img = Image.fromarray(u)
    if pil_img.size != (target_w, target_h):
        pil_img = pil_img.resize((target_w, target_h), Image.LANCZOS)
    return np.asarray(pil_img)


def _overlay_frame_caption_top_right(img: np.ndarray, caption: str, font_px: int = 12) -> np.ndarray:
    """Dark label top-right on RGB uint8 HWC (in-place safe: returns new array)."""
    pil = Image.fromarray(np.asarray(img, dtype=np.uint8, order="C"))
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_px)
    except OSError:
        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", font_px)
        except OSError:
            font = ImageFont.load_default()
    text = (caption or "").replace("\n", " ").strip()
    if len(text) > 110:
        text = text[:107] + "..."
    if not text:
        return np.asarray(pil)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    w, h = pil.size
    pad = 4
    bx0 = max(0, w - tw - pad * 3)
    by0 = pad
    bx1 = w - pad
    by1 = by0 + th + pad * 2
    draw.rectangle((bx0, by0, bx1, by1), fill=(0, 0, 0))
    draw.text((bx0 + pad, by0 + pad), text, font=font, fill=(255, 255, 255))
    return np.asarray(pil)


def save_rollout_video(
    rollout_primary_images,
    rollout_secondary_images,
    rollout_wrist_images,
    idx,
    success,
    task_description,
    rollout_data_dir,
    log_file=None,
    frame_captions=None,
):
    """Saves an MP4 replay of an episode with all three camera views."""
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:40]
    mp4_path = (
        f"{rollout_data_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    )
    video_writer = imageio.get_writer(mp4_path, fps=30)

    # One panel size for the whole clip: varying H/W between timesteps breaks many MP4 writers.
    panel_h = 0
    panel_w = 0
    for primary_img, secondary_img, wrist_img in zip(
        rollout_primary_images, rollout_secondary_images, rollout_wrist_images
    ):
        for img in (primary_img, secondary_img, wrist_img):
            u = _numpy_image_to_uint8_hwc(img)
            panel_h = max(panel_h, int(u.shape[0]))
            panel_w = max(panel_w, int(u.shape[1]))
    if panel_h == 0 or panel_w == 0:
        panel_h, panel_w = 224, 224

    # Concatenate all three camera views horizontally: primary (left) | secondary | wrist.
    n_frames = len(rollout_primary_images)
    if frame_captions is not None and len(frame_captions) != n_frames:
        frame_captions = None
    for i, (primary_img, secondary_img, wrist_img) in enumerate(
        zip(rollout_primary_images, rollout_secondary_images, rollout_wrist_images)
    ):
        p = _resize_uint8_hwc(_numpy_image_to_uint8_hwc(primary_img), panel_w, panel_h)
        s = _resize_uint8_hwc(_numpy_image_to_uint8_hwc(secondary_img), panel_w, panel_h)
        w = _resize_uint8_hwc(_numpy_image_to_uint8_hwc(wrist_img), panel_w, panel_h)
        combined_img = np.ascontiguousarray(np.concatenate([p, s, w], axis=1))
        if frame_captions is not None:
            combined_img = _overlay_frame_caption_top_right(combined_img, str(frame_captions[i]))
        video_writer.append_data(combined_img)

    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def save_rollout_video_with_future_image_predictions(
    rollout_primary_images,
    rollout_secondary_images,
    rollout_wrist_images,
    idx,
    success,
    task_description,
    rollout_data_dir,
    chunk_size,
    num_open_loop_steps,
    future_primary_image_predictions=None,
    future_secondary_image_predictions=None,
    future_wrist_image_predictions=None,
    show_diff=False,
    log_file=None,
    show_timestep=False,
    timestep=0,
    frame_captions=None,
):
    """Saves an MP4 replay of an episode with 2 rows and 3 columns:
    Top row: current wrist, current primary, current secondary images
    Bottom row: future wrist, future primary, future secondary predictions.

    For RoboCasa, we have three camera views:
    - Wrist (eye-in-hand)
    - Primary (left third-person)
    - Secondary (right third-person)

    Args:
        rollout_primary_images: List of primary (left) camera images
        rollout_secondary_images: List of secondary (right) camera images
        rollout_wrist_images: List of wrist camera images
        idx: Episode index
        success: Whether the episode was successful
        task_description: Description of the task
        chunk_size: Number of timesteps for future prediction
        num_open_loop_steps: Number of open loop steps
        future_primary_image_predictions: List of predicted future primary images
        future_secondary_image_predictions: List of predicted future secondary images
        future_wrist_image_predictions: List of predicted future wrist images
        show_diff: If True, show difference images (ignored in this version)
        log_file: Optional file for logging
        show_timestep: If True, show the timestep on the video
        timestep: The current timestep
    """
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_data_dir}/{DATE_TIME}--with_future_img--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)

    # Ensure future prediction lists have at least one element
    if not future_wrist_image_predictions:
        raise ValueError("future_wrist_image_predictions must have at least one element")
    if not future_primary_image_predictions:
        raise ValueError("future_primary_image_predictions must have at least one element")
    if not future_secondary_image_predictions:
        raise ValueError("future_secondary_image_predictions must have at least one element")

    # Reference size from future frame (uint8 — float predictions were written into uint8 buffers as black)
    ref0 = _numpy_image_to_uint8_hwc(future_primary_image_predictions[0])
    target_h, target_w = ref0.shape[0], ref0.shape[1]

    # Define text parameters
    text_height = 60 if show_timestep else 30  # Height for text area (increased if showing timestep)
    font_size = 16

    # Define column labels
    column_labels = ["wrist image", "primary image (left)", "secondary image (right)"]

    n_frames_fut = len(rollout_primary_images)
    if frame_captions is not None and len(frame_captions) != n_frames_fut:
        frame_captions = None

    for i, (primary_img, secondary_img, wrist_img) in enumerate(
        zip(rollout_primary_images, rollout_secondary_images, rollout_wrist_images)
    ):
        # Process current images - resize to match future prediction dimensions
        current_images_to_process = [wrist_img, primary_img, secondary_img]
        processed_current_images = []

        for current_img in current_images_to_process:
            processed_current_images.append(_resize_uint8_hwc(current_img, target_w, target_h))

        # Unpack processed current images
        wrist_img_resized, primary_img_resized, secondary_img_resized = processed_current_images

        # Determine which future prediction images to use
        future_idx = i // max(int(num_open_loop_steps), 1)
        future_wrist_idx = min(future_idx, len(future_wrist_image_predictions) - 1)
        future_primary_idx = min(future_idx, len(future_primary_image_predictions) - 1)
        future_secondary_idx = min(future_idx, len(future_secondary_image_predictions) - 1)

        future_wrist_img = _resize_uint8_hwc(
            future_wrist_image_predictions[future_wrist_idx], target_w, target_h
        )
        future_primary_img = _resize_uint8_hwc(
            future_primary_image_predictions[future_primary_idx], target_w, target_h
        )
        future_secondary_img = _resize_uint8_hwc(
            future_secondary_image_predictions[future_secondary_idx], target_w, target_h
        )

        # Create a combined image with 2 rows and 3 columns (always RGB for vstack with text banner)
        combined_img = np.zeros((target_h * 2, target_w * 3, 3), dtype=np.uint8)

        # Top row: current images (wrist, primary, secondary)
        combined_img[:target_h, :target_w, :] = wrist_img_resized
        combined_img[:target_h, target_w : target_w * 2, :] = primary_img_resized
        combined_img[:target_h, target_w * 2 : target_w * 3, :] = secondary_img_resized

        # Bottom row: future predictions (wrist, primary, secondary)
        combined_img[target_h:, :target_w, :] = future_wrist_img
        combined_img[target_h:, target_w : target_w * 2, :] = future_primary_img
        combined_img[target_h:, target_w * 2 : target_w * 3, :] = future_secondary_img

        if frame_captions is not None:
            combined_img = _overlay_frame_caption_top_right(combined_img, str(frame_captions[i]), font_px=11)

        # Create a blank area for text (white background)
        text_area = np.ones((text_height, target_w * 3, 3), dtype=np.uint8) * 255

        # Convert numpy array to PIL Image for text drawing
        text_img = Image.fromarray(text_area)
        draw = ImageDraw.Draw(text_img)

        # Try to use a standard font, fall back to default if not available
        try:
            font = ImageFont.truetype("Arial", font_size)
        except IOError:
            try:
                font = ImageFont.truetype("DejaVuSans", font_size)
            except IOError:
                try:
                    font = ImageFont.truetype("Verdana", font_size)
                except IOError:
                    font = ImageFont.load_default()

        # Add timestep if requested
        if show_timestep:
            timestep_text = f"t = {i}"
            timestep_width = draw.textlength(timestep_text, font=font)
            # Draw timestep centered at the top
            draw.text(((target_w * 3 - timestep_width) // 2, 2), timestep_text, font=font, fill=(0, 0, 0))

        # Add column labels
        label_y_pos = 32 if show_timestep else 8  # Adjust y position based on whether timestep is shown
        for col_idx, label in enumerate(column_labels):
            # Calculate center position for each column
            x_pos = col_idx * target_w + target_w // 2

            # Draw text centered in each column
            text_width = draw.textlength(label, font=font)
            draw.text((x_pos - text_width // 2, label_y_pos), label, font=font, fill=(0, 0, 0))

        # Convert back to numpy array
        text_area = np.array(text_img)

        # Combine text area and images
        final_frame = np.vstack((text_area, combined_img))

        video_writer.append_data(final_frame)

    video_writer.close()
    print(f"Saved rollout MP4 with future predictions at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 with future predictions at path {mp4_path}\n")
    return mp4_path


def save_rollout_video_with_future_image_predictions_and_gt(
    rollout_primary_images,
    rollout_secondary_images,
    rollout_wrist_images,
    idx,
    success,
    task_description,
    rollout_data_dir,
    chunk_size,
    num_open_loop_steps,
    future_primary_image_predictions=None,
    future_secondary_image_predictions=None,
    future_wrist_image_predictions=None,
    gt_future_primary_image_predictions=None,
    gt_future_secondary_image_predictions=None,
    gt_future_wrist_image_predictions=None,
    show_diff=True,
    log_file=None,
    show_timestep=False,
    timestep=0,
):
    """Saves an MP4 replay of an episode with 2 rows and 3 columns:
    Top row: current wrist, current primary, current secondary images
    Bottom row: future wrist, future primary, future secondary predictions.

    For RoboCasa, we have three camera views:
    - Wrist (eye-in-hand)
    - Primary (left third-person)
    - Secondary (right third-person)

    Args:
        rollout_primary_images: List of primary (left) camera images
        rollout_secondary_images: List of secondary (right) camera images
        rollout_wrist_images: List of wrist camera images
        idx: Episode index
        success: Whether the episode was successful
        task_description: Description of the task
        chunk_size: Number of timesteps for future prediction
        num_open_loop_steps: Number of open loop steps
        future_primary_image_predictions: List of predicted future primary images
        future_secondary_image_predictions: List of predicted future secondary images
        future_wrist_image_predictions: List of predicted future wrist images
        show_diff: If True, show difference images (ignored in this version)
        log_file: Optional file for logging
        show_timestep: If True, show the timestep on the video
        timestep: The current timestep
    """
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_data_dir}/{DATE_TIME}--with_future_img--episode={idx}--success={success}--task={processed_task_description}--gt.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)

    # Ensure future prediction lists have at least one element
    if not future_wrist_image_predictions:
        raise ValueError("future_wrist_image_predictions must have at least one element")
    if not future_primary_image_predictions:
        raise ValueError("future_primary_image_predictions must have at least one element")
    if not future_secondary_image_predictions:
        raise ValueError("future_secondary_image_predictions must have at least one element")

    ref0 = _numpy_image_to_uint8_hwc(future_primary_image_predictions[0])
    target_h, target_w = ref0.shape[0], ref0.shape[1]

    # Define text parameters
    text_height = 60 if show_timestep else 30  # Height for text area (increased if showing timestep)
    font_size = 16

    # Define column labels
    column_labels = ["wrist image", "primary image (left)", "secondary image (right)"]

    # Center-crop the ground-truth future images to match the future prediction images
    def center_crop(img, img_size):
        import torch
        import torchvision.transforms.functional as F

        img_tensor = torch.from_numpy(img.copy()).permute(2, 0, 1)
        crop_size = int(img_size * 0.9**0.5)  # Square root because we're dealing with area
        img_crop = F.center_crop(img_tensor, crop_size)
        img_resized = F.resize(img_crop, [img_size, img_size], antialias=True)
        return img_resized.numpy().transpose(1, 2, 0)

    gt_future_wrist_images = []
    gt_future_primary_images = []
    gt_future_secondary_images = []
    for gt_future_wrist_img, gt_future_primary_img, gt_future_secondary_img in zip(
        gt_future_wrist_image_predictions, gt_future_primary_image_predictions, gt_future_secondary_image_predictions
    ):
        gt_future_wrist_img = center_crop(gt_future_wrist_img, target_h)
        gt_future_primary_img = center_crop(gt_future_primary_img, target_h)
        gt_future_secondary_img = center_crop(gt_future_secondary_img, target_h)
        gt_future_wrist_images.append(gt_future_wrist_img)
        gt_future_primary_images.append(gt_future_primary_img)
        gt_future_secondary_images.append(gt_future_secondary_img)
    gt_future_wrist_image_predictions = gt_future_wrist_images
    gt_future_primary_image_predictions = gt_future_primary_images
    gt_future_secondary_image_predictions = gt_future_secondary_images

    for i, (primary_img, secondary_img, wrist_img) in enumerate(
        zip(rollout_primary_images, rollout_secondary_images, rollout_wrist_images)
    ):
        # Process current images - resize to match future prediction dimensions
        current_images_to_process = [wrist_img, primary_img, secondary_img]
        processed_current_images = []

        for current_img in current_images_to_process:
            processed_current_images.append(_resize_uint8_hwc(current_img, target_w, target_h))

        # Unpack processed current images
        wrist_img_resized, primary_img_resized, secondary_img_resized = processed_current_images

        # Determine which future prediction images to use
        future_idx = i // max(int(num_open_loop_steps), 1)
        future_wrist_idx = min(future_idx, len(future_wrist_image_predictions) - 1)
        future_primary_idx = min(future_idx, len(future_primary_image_predictions) - 1)
        future_secondary_idx = min(future_idx, len(future_secondary_image_predictions) - 1)

        future_wrist_img = _resize_uint8_hwc(
            future_wrist_image_predictions[future_wrist_idx], target_w, target_h
        )
        future_primary_img = _resize_uint8_hwc(
            future_primary_image_predictions[future_primary_idx], target_w, target_h
        )
        future_secondary_img = _resize_uint8_hwc(
            future_secondary_image_predictions[future_secondary_idx], target_w, target_h
        )

        gt_future_wrist_img = _resize_uint8_hwc(
            gt_future_wrist_image_predictions[future_wrist_idx], target_w, target_h
        )
        gt_future_primary_img = _resize_uint8_hwc(
            gt_future_primary_image_predictions[future_primary_idx], target_w, target_h
        )
        gt_future_secondary_img = _resize_uint8_hwc(
            gt_future_secondary_image_predictions[future_secondary_idx], target_w, target_h
        )

        # Compute difference images if show_diff is True
        if show_diff:
            # Compute difference between future primary image and ground-truth future primary image
            primary_diff = np.abs(future_primary_img.astype(np.float32) - gt_future_primary_img.astype(np.float32))
            primary_diff = np.clip(primary_diff, 0, 255).astype(np.uint8)
            # Compute difference between future wrist image and ground-truth future wrist image
            wrist_diff = np.abs(future_wrist_img.astype(np.float32) - gt_future_wrist_img.astype(np.float32))
            wrist_diff = np.clip(wrist_diff, 0, 255).astype(np.uint8)
            # Compute difference between future secondary image and ground-truth future secondary image
            secondary_diff = np.abs(
                future_secondary_img.astype(np.float32) - gt_future_secondary_img.astype(np.float32)
            )
            secondary_diff = np.clip(secondary_diff, 0, 255).astype(np.uint8)
        else:
            z = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            wrist_diff = primary_diff = secondary_diff = z

        # Create a combined image with 4 rows and 3 columns
        combined_img = np.zeros((target_h * 4, target_w * 3, 3), dtype=np.uint8)

        # Top row: current images (wrist, primary, secondary)
        combined_img[:target_h, :target_w, :] = wrist_img_resized
        combined_img[:target_h, target_w : target_w * 2, :] = primary_img_resized
        combined_img[:target_h, target_w * 2 : target_w * 3, :] = secondary_img_resized
        # Second row: future predictions (wrist, primary, secondary)
        combined_img[target_h : target_h * 2, :target_w, :] = future_wrist_img
        combined_img[target_h : target_h * 2, target_w : target_w * 2, :] = future_primary_img
        combined_img[target_h : target_h * 2, target_w * 2 : target_w * 3, :] = future_secondary_img
        # Third row: ground-truth future images (wrist, primary, secondary)
        combined_img[target_h * 2 : target_h * 3, :target_w, :] = gt_future_wrist_img
        combined_img[target_h * 2 : target_h * 3, target_w : target_w * 2, :] = gt_future_primary_img
        combined_img[target_h * 2 : target_h * 3, target_w * 2 : target_w * 3, :] = gt_future_secondary_img
        # Fourth row: difference images (wrist, primary, secondary)
        combined_img[target_h * 3 : target_h * 4, :target_w, :] = wrist_diff
        combined_img[target_h * 3 : target_h * 4, target_w : target_w * 2, :] = primary_diff
        combined_img[target_h * 3 : target_h * 4, target_w * 2 : target_w * 3, :] = secondary_diff
        # Create a blank area for text (white background)
        text_area = np.ones((text_height, target_w * 3, 3), dtype=np.uint8) * 255

        # Convert numpy array to PIL Image for text drawing
        text_img = Image.fromarray(text_area)
        draw = ImageDraw.Draw(text_img)

        # Try to use a standard font, fall back to default if not available
        try:
            font = ImageFont.truetype("Arial", font_size)
        except IOError:
            try:
                font = ImageFont.truetype("DejaVuSans", font_size)
            except IOError:
                try:
                    font = ImageFont.truetype("Verdana", font_size)
                except IOError:
                    font = ImageFont.load_default()

        # Add timestep if requested
        if show_timestep:
            timestep_text = f"t = {i}"
            timestep_width = draw.textlength(timestep_text, font=font)
            # Draw timestep centered at the top
            draw.text(((target_w * 3 - timestep_width) // 2, 2), timestep_text, font=font, fill=(0, 0, 0))

        # Add column labels
        label_y_pos = 32 if show_timestep else 8  # Adjust y position based on whether timestep is shown
        for col_idx, label in enumerate(column_labels):
            # Calculate center position for each column
            x_pos = col_idx * target_w + target_w // 2

            # Draw text centered in each column
            text_width = draw.textlength(label, font=font)
            draw.text((x_pos - text_width // 2, label_y_pos), label, font=font, fill=(0, 0, 0))

        # Convert back to numpy array
        text_area = np.array(text_img)

        # Combine text area and images
        final_frame = np.vstack((text_area, combined_img))

        video_writer.append_data(final_frame)

    video_writer.close()
    print(f"Saved rollout MP4 with future predictions and ground-truth future images at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 with future predictions and ground-truth future images at path {mp4_path}\n")
    return mp4_path
