import os
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER

def on_epoch_end(trainer):
    """
    Callback function to compute and save confusion matrix after each epoch.
    """
    LOGGER.info(f"\n📊 Generating confusion matrix for epoch {trainer.epoch + 1}...")
    
    # Run validation silently on the current model weights
    metrics = trainer.model.val(data=trainer.args.data, split='val', save_json=False, plots=False, verbose=False)
    
    # Access the confusion matrix
    cm = metrics.confusion_matrix
    
    # Plot and save confusion matrix
    save_dir = os.path.join(trainer.save_dir, f"confusion_matrix_epoch_{trainer.epoch + 1}.png")
    cm.plot(save_dir=save_dir, names=trainer.data['names'])
    
    LOGGER.info(f"✅ Confusion matrix saved: {save_dir}\n")


def main():
    # Check for GPU
    if torch.cuda.is_available():
        print(f"[GPU] Using device: {torch.cuda.get_device_name(0)}")
    else:
        print("[CPU] CUDA not available. Training will run on CPU.")

    # Paths
    model_path = "yolov8l.pt"
    data_yaml = os.path.join("configs", "data.yaml")

    # Initialize YOLO model
    model = YOLO(model_path)

    # Register callback
    model.add_callback("on_epoch_end", on_epoch_end)

    # Start training
    model.train(
        data=data_yaml,
        epochs=40,
        batch=8,
        imgsz=640,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=4,
        project="runs/detect",
        name="weapons_yolov8l",
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        patience=50,
        augment=True,
        verbose=True,
        deterministic=True,
        save=True,
        plots=True,
    )

    print("\n✅ Training complete! Model weights saved under 'runs/detect/weapons_yolov8l'")


if __name__ == "__main__":
    main()
