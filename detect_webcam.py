import argparse, time
import cv2
import torch
import numpy as np
from utils import get_model_instance, load_config
from sort import Sort
from twilio_sms import send_sms

# python detect_webcam.py --config configs/config.json --device cuda

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--score-thresh', type=float, default=0.6)
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)
    class_names = cfg.get('class_names', ['__background__','knife','handgun'])
    num_classes = cfg.get('num_classes', len(class_names))
    device = torch.device(args.device if torch.cuda.is_available() and args.device=='cuda' else 'cpu')

    model = get_model_instance(num_classes)
    # Attempt to load latest checkpoint if exists
    try:
        model.load_state_dict(torch.load('checkpoint_epoch_last.pth', map_location=device))
        print('Loaded checkpoint_epoch_last.pth')
    except Exception:
        print('No checkpoint found or failed to load (continuing with pretrained backbone).')
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(0)
    tracker = Sort(max_age=10, min_hits=1, iou_threshold=0.3)
    consecutive = {}  # track_id -> consecutive frames seen
    alerted = set()

    print('Enter destination phone number (E.164 format, e.g. +919XXXXXXXXX):')
    to_number = input().strip()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img / 255.).permute(2,0,1).float().to(device).unsqueeze(0)
        # run model
        with torch.no_grad():
            outputs = model(img_tensor)[0]
        boxes = outputs['boxes'].cpu().numpy()
        scores = outputs['scores'].cpu().numpy()
        labels = outputs['labels'].cpu().numpy()
        dets = []
        for (b,s,l) in zip(boxes, scores, labels):
            if s < args.score_thresh: 
                continue
            dets.append([float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(s), int(l)])
        # prepare for tracker (only box+score)
        dets_for_sort = [d[:5] for d in dets]
        tracks = tracker.update(dets_for_sort)
        # associate class label with tracks by nearest IoU match (simple)
        for t_id, bbox in tracks:
            # find matching detection with highest IoU
            best = None; best_iou = 0
            for d in dets:
                # compute IoU
                xx1 = max(bbox[0], d[0]); yy1 = max(bbox[1], d[1])
                xx2 = min(bbox[2], d[2]); yy2 = min(bbox[3], d[3])
                w = max(0., xx2-xx1); h = max(0., yy2-yy1)
                inter = w*h
                area1 = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
                area2 = (d[2]-d[0])*(d[3]-d[1])
                iou = inter/(area1+area2-inter+1e-6)
                if iou > best_iou:
                    best_iou = iou
                    best = d
            label = None
            if best is not None:
                label = int(best[5])
            # update consecutive counter
            if t_id not in consecutive:
                consecutive[t_id] = 0
            if best is not None and best[4] >= args.score_thresh:
                consecutive[t_id] += 1
            else:
                consecutive[t_id] = 0
            # check alert
            if consecutive[t_id] >= 5 and t_id not in alerted:
                # send SMS
                try:
                    sid = send_sms(to_number, cfg.get('sms_message','Weapon detected!'), cfg_path=args.config)
                    print('SMS sent, sid=', sid)
                except Exception as e:
                    print('Failed to send SMS:', e)
                alerted.add(t_id)
            # draw bbox
            x1,y1,x2,y2 = map(int, bbox)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            text = f"ID:{t_id} {class_names[label] if label is not None and label<len(class_names) else ''} cnt={consecutive.get(t_id,0)}"
            cv2.putText(frame, text, (x1, max(0,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imshow('Weapon detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
