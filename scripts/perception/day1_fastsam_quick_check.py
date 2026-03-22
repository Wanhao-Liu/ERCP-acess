"""Day 1 perception chain validation: FastSAM center-point prompt on 15 phantom frames."""

import os
import sys

sys.path.insert(0, "/data/ERCP/ercp_access")

import numpy as np
import cv2

from src.data.phantom_dataset import PhantomEpisode

EPISODE_IDS = [1, 11, 40]
PERCENTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
IMG_W, IMG_H = 1920, 1080
CX_PROMPT, CY_PROMPT = 960, 540
OUT_DIR = "/data/ERCP/ercp_access/outputs/perception/day1_fastsam"


def select_frame_indices(n_frames):
    return [int(p * (n_frames - 1)) for p in PERCENTILES]


def best_mask_by_centroid(masks_tensor):
    """Return (mask_np, cx, cy, area) for mask whose centroid is closest to image center."""
    best = None
    best_dist = float("inf")
    for i in range(masks_tensor.shape[0]):
        mask_np = masks_tensor[i].cpu().numpy().astype(np.uint8)
        y_coords, x_coords = np.where(mask_np > 0)
        if len(x_coords) == 0:
            continue
        cx = float(np.mean(x_coords))
        cy = float(np.mean(y_coords))
        area = len(x_coords)
        dist = (cx - CX_PROMPT) ** 2 + (cy - CY_PROMPT) ** 2
        if dist < best_dist:
            best_dist = dist
            best = (mask_np, cx, cy, area)
    return best


def save_overlay(rgb_chw, mask_np, cx, cy, out_path):
    img_bgr = cv2.cvtColor(
        (np.transpose(rgb_chw, (1, 2, 0)) * 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )
    overlay = img_bgr.copy()
    overlay[mask_np > 0] = [0, 200, 0]
    result_img = cv2.addWeighted(img_bgr, 0.6, overlay, 0.4, 0)
    cv2.circle(result_img, (int(cx), int(cy)), 15, (0, 0, 255), -1)
    cv2.imwrite(out_path, result_img)


def save_raw(rgb_chw, out_path):
    img_bgr = cv2.cvtColor(
        (np.transpose(rgb_chw, (1, 2, 0)) * 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )
    cv2.imwrite(out_path, img_bgr)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    try:
        from ultralytics import FastSAM
        model = FastSAM("FastSAM-s.pt")
        model_ok = True
    except Exception as e:
        print(f"[ERROR] FastSAM load failed: {e}")
        model_ok = False

    header = f"{'episode':>8} {'frame_idx':>10} {'detected':>9} {'cx_norm':>9} {'cy_norm':>9} {'area_norm':>10} {'conf':>6}"
    print(header)
    print("-" * len(header))

    for ep_id in EPISODE_IDS:
        ep = PhantomEpisode(ep_id)
        n_frames = ep.n_frames
        indices = select_frame_indices(n_frames)

        for frame_idx in indices:
            rgb_chw = ep.get_frame(frame_idx)
            out_path = os.path.join(OUT_DIR, f"ep{ep_id:03d}_frame{frame_idx:05d}.jpg")

            detected = False
            cx_norm = cy_norm = area_norm = conf_val = float("nan")

            if not model_ok:
                save_raw(rgb_chw, out_path)
                print(f"{ep_id:>8} {frame_idx:>10} {'N':>9} {'nan':>9} {'nan':>9} {'nan':>10} {'nan':>6}")
                continue

            try:
                img_hwc_uint8 = (np.transpose(rgb_chw, (1, 2, 0)) * 255).astype(np.uint8)

                results = model(
                    img_hwc_uint8,
                    device="cuda",
                    retina_masks=True,
                    conf=0.4,
                    iou=0.9,
                    verbose=False,
                    points=[[CX_PROMPT, CY_PROMPT]],
                    labels=[1],
                )

                if results[0].masks is not None:
                    masks = results[0].masks.data
                    result = best_mask_by_centroid(masks)
                    if result is not None:
                        mask_np, cx, cy, area = result
                        cx_norm = (cx - CX_PROMPT) / CX_PROMPT
                        cy_norm = (cy - CY_PROMPT) / CY_PROMPT
                        area_norm = area / (IMG_W * IMG_H)
                        conf_scores = results[0].boxes.conf if results[0].boxes is not None else None
                        conf_val = float(conf_scores.mean()) if conf_scores is not None and len(conf_scores) > 0 else float("nan")
                        detected = True
                        save_overlay(rgb_chw, mask_np, cx, cy, out_path)
                    else:
                        save_raw(rgb_chw, out_path)
                else:
                    save_raw(rgb_chw, out_path)

            except Exception as e:
                print(f"[ERROR] ep{ep_id} frame{frame_idx}: {e}")
                save_raw(rgb_chw, out_path)

            det_str = "Y" if detected else "N"
            cx_s = f"{cx_norm:.4f}" if detected else "nan"
            cy_s = f"{cy_norm:.4f}" if detected else "nan"
            ar_s = f"{area_norm:.5f}" if detected else "nan"
            cf_s = f"{conf_val:.3f}" if detected and not np.isnan(conf_val) else "nan"
            print(f"{ep_id:>8} {frame_idx:>10} {det_str:>9} {cx_s:>9} {cy_s:>9} {ar_s:>10} {cf_s:>6}")

        ep.close()

    print(f"\nOutput images saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
