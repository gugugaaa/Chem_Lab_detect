import argparse
import os
import sys

import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debug.score_pipeline import ScorerPipeline
from safety_detect.safety_detector import SafetyDetector
from utils.draw_hand import draw_landmarks
from utils.draw_keypoints import draw_keypoints
from utils.draw_safety import draw_safety
from vessel_detect.vessel_cascade import VesselCascadeDetector


vessel_keypoint_colors = [
    (130, 76, 130),
    (180, 82, 120),
    (211, 160, 221),
    (148, 87, 235),
    (204, 153, 255),
]


def build_detector(name):
    if name == "safety":
        return SafetyDetector()
    if name == "vessel":
        return VesselCascadeDetector()
    if name == "score":
        return ScorerPipeline()
    raise ValueError(f"Unsupported detector: {name}")


def resize_by_width(frame, width):
    if width <= 0:
        return frame
    height, current_width = frame.shape[:2]
    scale = width / current_width
    return cv2.resize(frame, (width, int(height * scale)))


def draw_cached_result(frame, detector_name, detection_info):
    if not detection_info:
        return frame

    if detector_name == "safety":
        return draw_safety(frame, detection_info, show_status=True, show_debug=False)

    if detector_name == "vessel":
        return draw_keypoints(
            frame,
            detection_info,
            keypoint_colors=vessel_keypoint_colors,
            show_names=False,
            draw_bbox=True,
        )

    if detector_name == "score":
        vessel_info = detection_info.get("vessel_info", {})
        gesture_info = detection_info.get("gesture_info", {})
        score_result = detection_info.get("score_result")

        frame = draw_keypoints(
            frame,
            vessel_info,
            keypoint_colors=vessel_keypoint_colors,
            show_names=False,
            draw_bbox=True,
        )
        for hand in gesture_info.get("hands", []):
            points = [(kpt["x"], kpt["y"]) for kpt in hand.get("keypoints", [])]
            frame = draw_landmarks(frame, points)
        if score_result:
            text = f'Score: {score_result["score"]:.1f} ({score_result["operation"]})'
            cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)
        return frame

    return frame


def run_video_sampling(
    source,
    detector_name,
    output=None,
    frame_interval=5,
    width=320,
    display=True,
):
    """
    Edge-friendly video inference: only run detector every N frames.
    Skipped frames reuse the latest annotated inference result.
    """
    cap = cv2.VideoCapture(0 if source == "camera" else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    detector = build_detector(detector_name)
    writer = None
    frame_idx = 0
    last_info = None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = resize_by_width(frame, width)

            should_detect = frame_idx % frame_interval == 0 or last_info is None
            if should_detect:
                processed_frame, last_info = detector.detect_frame(frame)
            else:
                processed_frame = draw_cached_result(frame.copy(), detector_name, last_info)
                cv2.putText(
                    processed_frame,
                    f"skip infer, reuse frame {frame_idx - frame_idx % frame_interval}",
                    (10, processed_frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            if writer is None and output:
                h, w = processed_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output, fourcc, fps, (w, h))

            if writer:
                writer.write(processed_frame)

            if display:
                cv2.imshow("Frame Sampling Detection", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if should_detect:
                print(f"frame={frame_idx}, info={last_info}")

            frame_idx += 1
    finally:
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Video frame sampling detection demo.")
    parser.add_argument("--source", default="camera", help="Video path, stream url, or 'camera'.")
    parser.add_argument(
        "--detector",
        default="safety",
        choices=["safety", "vessel", "score"],
        help="Detection pipeline to run on sampled frames.",
    )
    parser.add_argument("--output", default=None, help="Optional output mp4 path.")
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=5,
        help="Run inference every N frames. Larger values reduce edge-device load.",
    )
    parser.add_argument("--width", type=int, default=320, help="Resize input width before detection.")
    parser.add_argument("--no-display", action="store_true", help="Disable preview window.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_video_sampling(
        source=args.source,
        detector_name=args.detector,
        output=args.output,
        frame_interval=max(1, args.frame_interval),
        width=args.width,
        display=not args.no_display,
    )
