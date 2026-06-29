#!/usr/bin/env python3
import argparse
import sys
import os
import io
import logging
import csv
import glob

# ====================== 强制tqdm配置 ======================
os.environ['TQDM_DISABLE'] = '0'
os.environ['TQDM_NCOLS'] = '80'
os.environ['TQDM_ASCII'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['YOLO_VERBOSE'] = '1'
os.environ['ULTRALYTICS_NO_COLORS'] = '1'

import tqdm
import tqdm.auto
import tqdm.std


class ForceDisplayTQDM(tqdm.tqdm):
    def __init__(self, *args, **kwargs):
        kwargs['file'] = sys.stdout
        kwargs['disable'] = False
        kwargs['dynamic_ncols'] = False
        kwargs['leave'] = False
        kwargs['smoothing'] = 0.0
        kwargs['ascii'] = ' #'
        kwargs['bar_format'] = '{l_bar}{bar:40}| {n_fmt}/{total_fmt} {postfix}'
        kwargs['ncols'] = 80
        kwargs['mininterval'] = 1.0
        super().__init__(*args, **kwargs)


tqdm.tqdm = ForceDisplayTQDM
tqdm.auto.tqdm = ForceDisplayTQDM
tqdm.std.tqdm = ForceDisplayTQDM
sys.modules['tqdm'].tqdm = ForceDisplayTQDM
sys.modules['tqdm.auto'].tqdm = ForceDisplayTQDM
sys.modules['tqdm.std'].tqdm = ForceDisplayTQDM

# ====================== 日志配置 ======================
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)


class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


stdout_handler = FlushStreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(logging.Formatter('%(message)s'))
root_logger.addHandler(stdout_handler)

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

# ==================================================================
from ultralytics import YOLO


# ====================== 回调函数 ======================
def on_train_start(trainer):
    print("=" * 80)
    print(f"✅ 训练正式开始，共 {trainer.epochs} 个epoch")
    train_size = len(trainer.train_loader.dataset) if hasattr(trainer, 'train_loader') else "未知"
    val_size = len(trainer.test_loader.dataset) if hasattr(trainer, 'test_loader') else "未知"
    print(f"📊 训练集: {train_size} 张图片")
    print(f"📊 验证集: {val_size} 张图片")
    print("=" * 80)
    print("\n")
    print(
        f"{'Epoch':<8} {'GPU Mem':<8} {'box_loss':<10} {'cls_loss':<10} {'dfl_loss':<10} {'Instances':<10} {'Img Size':<8}")
    print("-" * 80)
    sys.stdout.flush()


def on_epoch_start(trainer):
    print("\n")
    print(
        f"{'Epoch':<8} {'GPU Mem':<8} {'box_loss':<10} {'cls_loss':<10} {'dfl_loss':<10} {'Instances':<10} {'Img Size':<8}")
    print("-" * 80)
    sys.stdout.flush()


def on_train_batch_end(trainer):
    sys.stdout.flush()


def on_epoch_end(trainer):
    metrics = trainer.metrics
    print("\n" + "-" * 80)
    print(f"✅ Epoch {trainer.epoch + 1}/{trainer.epochs} 完成")
    print(f"   mAP50: {metrics.get('metrics/mAP_0.5', 0):.4f}")
    print(f"   mAP50-95: {metrics.get('metrics/mAP_0.5:0.95', 0):.4f}")
    print(f"   训练损失: box={trainer.loss_box.item():.4f}, cls={trainer.loss_cls.item():.4f}")
    print("=" * 80)
    sys.stdout.flush()


def on_train_end(trainer):
    print("\n" + "=" * 80)
    print("🎉 训练全部完成！")
    print(f"🏆 最佳mAP50: {trainer.best_fitness:.4f}")
    print("=" * 80)
    sys.stdout.flush()


# ====================== 主函数 ======================
def find_weight_file(weights_arg):
    # ✅ 直接写死你的模型绝对路径
    return "D:/QQT/DogFaceRecognition/models/yolov11.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--weights', default='models/yolov11.pt')
    args = parser.parse_args()

    data_yaml = os.path.join(args.dataset, 'data.yaml')
    if not os.path.exists(data_yaml):
        sys.stderr.write(f"❌ 错误：找不到 {data_yaml}\n")
        sys.stderr.flush()
        sys.exit(1)

    weight_path = find_weight_file(args.weights)
    print(f"📦 使用预训练模型: {weight_path}")
    sys.stdout.flush()

    model = YOLO(weight_path)

    # 注册回调
    model.add_callback('on_train_start', on_train_start)
    model.add_callback('on_epoch_start', on_epoch_start)
    model.add_callback('on_train_batch_end', on_train_batch_end)
    model.add_callback('on_epoch_end', on_epoch_end)
    model.add_callback('on_train_end', on_train_end)

    # 开始训练
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        lr0=args.lr,
        batch=args.batch,
        optimizer='SGD',
        verbose=True,
        project='runs/train',
        name='exp',
        exist_ok=False,
        plots=False,
        save_period=-1,
        workers=0,
        cache=False
    )

    # ✅ 修复：使用Python内置csv模块读取结果，无依赖
    try:
        # 按修改时间排序，找到最新的实验文件夹
        exp_dirs = glob.glob(os.path.join('runs', 'train', 'exp*'))
        if not exp_dirs:
            raise Exception("未找到任何实验文件夹")

        latest_exp = max(exp_dirs, key=os.path.getmtime)
        csv_path = os.path.join(latest_exp, 'results.csv')

        if not os.path.exists(csv_path):
            raise Exception(f"结果文件不存在: {csv_path}")

        # 读取最后一行数据
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                raise Exception("结果文件为空")

            last_row = rows[-1]

            # 自动处理列名前后的空格
            def get_val(key):
                for k in last_row.keys():
                    if k.strip() == key:
                        return float(last_row[k].strip())
                return 0.0

            map50 = get_val('metrics/mAP_0.5')
            map50_95 = get_val('metrics/mAP_0.5:0.95')
            precision = get_val('metrics/precision(B)')
            recall = get_val('metrics/recall(B)')
            model_path = os.path.join(latest_exp, 'weights', 'best.pt')

            # 统一路径分隔符为正斜杠，避免Windows反斜杠问题
            model_path = model_path.replace('\\', '/')

            print(
                f"\nTRAIN_FINISHED map50={map50:.3f} map50-95={map50_95:.3f} precision={precision:.3f} recall={recall:.3f} model={model_path}")
            sys.stdout.flush()
            return

    except Exception as e:
        sys.stderr.write(f"⚠️ 读取训练结果失败: {str(e)}\n")
        sys.stderr.flush()

    # 保底输出（使用训练过程中记录的最佳mAP50）
    print(
        f"\nTRAIN_FINISHED map50={results.best_fitness:.3f} map50-95=0.000 precision=0.000 recall=0.000 model=runs/train/exp/weights/best.pt")
    sys.stdout.flush()


if __name__ == "__main__":
    main()