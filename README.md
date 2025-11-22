# Weapon Detection + Tracker + SMS Alert

Quick scaffold that:
- Fine-tunes a Faster R-CNN (ResNet50 FPN) on COCO-style weapon dataset using PyTorch.
<<<<<<< HEAD
- Runs webcam inference, applies SORT tracker, counts consecutive detections for the same tracked object, and triggers an SMS alert after 5 consecutive frames.
- Uses Twilio for SMS (you must supply account credentials).

## Structure
- `train.py` - training script (torchvision Faster R-CNN).
- `detect_webcam.py` - webcam inference + tracking + SMS trigger.
- `sort.py` - lightweight SORT tracker implementation.
- `twilio_sms.py` - wrapper to send SMS via Twilio.
- `utils.py` - helper functions.
=======
- Runs webcam inference, applies SORT tracker, counts consecutive detections for the same tracked object, and triggers an SMS alert after 5 cnsecutive frames.
- Uses Twilio for SMS (you must supply account credentials).

## Structure
- `train.py` - training script.
- `cam.py` - webcam inference.
>>>>>>> f2e5edf (Better performance of model)
- `requirements.txt` - Python deps.
- `configs/` - example config.

## Before you run
<<<<<<< HEAD
1. Place your COCO-style dataset ZIP at `data/dataset.zip` and extract to `data/coco/` with `annotations/` and `images/`.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Edit `configs/config.json`:
   - `data_root`: path to dataset root with `train2017/`, `val2017/`, and `annotations/instances_train.json`.
   - `twilio` section: set `account_sid`, `auth_token`, `from_number`. The script will ask for `to_number` (phone number) at runtime.
4. Training will use CUDA if available. Set `--device cuda` when running training or inference.

## Notes & design choices
- I chose Faster R-CNN (torchvision) rather than YOLO because it integrates cleanly with COCO-format annotations and is straightforward to fine-tune on GPUs. It's robust for small-object detection like handguns/knives when fine-tuned with enough data.
- For temporal "confirmations", SORT tracker provides stable IDs across frames; we count repeated detections of the same tracked ID across consecutive frames and trigger SMS when the count reaches 5.
- Twilio is used for SMS; you will need an account and a valid `from_number`.
=======
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. [Optional] If you want to train your own model with a database, you can run the "train.py" script with proper path declaration and adjusting numerical values.
   ```
   python train.py
   ```

3. start detection through webcam by the use of "cam.py".
   ```
   python scripts/cam.py --model scripts/best.pt --source 0
   ```
>>>>>>> f2e5edf (Better performance of model)

## Useful commands
Train:
```
<<<<<<< HEAD
python train.py --config configs/config.json --epochs 10 --batch-size 4 --device cuda
=======
python train.py
>>>>>>> f2e5edf (Better performance of model)
```

Run webcam:
```
<<<<<<< HEAD
python detect_webcam.py --config configs/config.json --device cuda
=======
python scripts/cam.py --model scripts/best.pt --source 0
>>>>>>> f2e5edf (Better performance of model)
```

