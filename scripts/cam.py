import torch
import cv2
import os
import argparse
from ultralytics import YOLO

def run_detection(model_path, source, conf=0.5, save_output=True):
    """
    Run YOLOv8 detection on image, video, or webcam.
    Args:
        model_path: Path to your trained YOLOv8 model (.pt file)
        source: Path to input (image/video) or integer (webcam)
        conf: Confidence threshold for detections
        save_output: Whether to save annotated output

        script: python scripts/cam.py --model scripts/best.pt --source 0
        0 for webcam
        for media provide the path in place of 0
    """
    # Check CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device.upper()}")

    # Load model
    model = YOLO(r"C:\Users\KARAN\Desktop\College\projects\MINI Project Sessions\Tensorflow\models\best.pt")
    model.to(device)

    # Run detection
    print(f"[INFO] Running detection on: {source}")
    results = model.predict(
        source=source,
        conf=conf,
        show=True,           # Shows live output (useful for webcam)
        save=save_output,    # Saves annotated image/video
        device=device,
        stream=False
    )

    # Get save directory
    if save_output:
        save_dir = os.path.join("runs", "detect", "results")
        os.makedirs(save_dir, exist_ok=True)
        print(f"[INFO] Results saved in: {save_dir}")

    print("\n✅ Detection completed successfully!")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Media Detection Script")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLOv8 model (.pt)")
    parser.add_argument("--source", type=str, required=True,
                        help="Media source: image/video path or '0' for webcam")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument("--nosave", action="store_true", help="Do not save output results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    source = 0 if args.source == "0" else args.source
    run_detection(args.model, source, conf=args.conf, save_output=not args.nosave)
