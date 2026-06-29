#!/usr/bin/env python3
"""
Python端直接加载.pt模型进行推理
与C++端ONNX推理结果保持完全一致
"""
import argparse
import json
import sys
import cv2
from pathlib import Path
from ultralytics import YOLO

# 强制UTF-8输出
import io

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='输入图片路径')
    parser.add_argument('--model', required=True, help='.pt模型路径')
    parser.add_argument('--conf', type=float, default=0.3, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU阈值')
    args = parser.parse_args()

    try:
        # 加载图像
        img0 = cv2.imread(args.image)
        if img0 is None:
            print(json.dumps({"error": "无法加载图像"}))
            sys.stdout.flush()
            sys.exit(1)

        # 加载YOLOv11 .pt模型
        model = YOLO(args.model, task='detect')

        # 执行推理（使用与C++端完全一致的参数）
        results = model(
            img0,
            conf=args.conf,
            iou=args.iou,
            imgsz=640,
            device='cpu',
            verbose=False,
            agnostic_nms=True
        )

        # 解析结果
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())

                detections.append({
                    "x": float(x1),
                    "y": float(y1),
                    "w": float(x2 - x1),
                    "h": float(y2 - y1),
                    "class": cls,
                    "confidence": float(conf)
                })

        # 输出JSON结果
        print(json.dumps({"detections": detections}))
        sys.stdout.flush()

    except Exception as e:
        print(json.dumps({"error": f"推理失败: {str(e)}"}))
        sys.stdout.flush()
        sys.exit(1)


if __name__ == '__main__':
    main()