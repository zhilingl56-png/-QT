# -QT基于 Qt C++ + Python + YOLOv11 + SQLite 的单机离线目标检测工具，覆盖「数据标注 → 模型训练 → 量化压缩 → 推理验证」完整闭环。
🖥️ 界面交互层
主窗口框架：多标签页布局（标注 / 训练 / 量化 / 推理 / 数据管理），配套工具栏、状态栏与日志输出窗口，整体交互符合桌面软件使用习惯。
图像查看器：基于 QGraphicsView 自定义实现，支持图像缩放、平移、矩形标注绘制 / 拖拽 / 编辑、检测结果可视化叠加渲染。
🛠️ 核心业务模块
标注管理模块（AnnotationManager）：管理图像标注数据的增删改查，支持检测类别自定义、标注坐标归一化计算，可一键导出 YOLO 标准格式标签文件。
训练管理模块（TrainingManager）：可视化配置训练超参数（Epochs、学习率、Batch Size），通过子进程调用 YOLOv11 执行训练，实时回显训练日志，自动解析 mAP、Precision、Recall 等核心指标并入库。
量化管理模块（QuantizationManager）：采用 FP16 量化策略将 .pt 权重导出为 ONNX 模型，强制兼容 Opset 11 / IR v9 标准，自动关联父模型并记录量化元数据。
推理管理模块（InferenceManager）：支持双模式推理对比 ——PyTorch 原生 .pt 推理与 ONNX Runtime 量化推理，使用统一阈值参数，解析检测框结果并可视化绘制。
💾 数据持久层
数据库管理（DatabaseManager）：单例模式封装 SQLite 全量操作，维护 categories /images/annotations /models/inference_results 五张核心表，支持外键级联删除、事务保障与数据全链路追溯。
🔌 跨语言桥接层
PythonBridge：基于 QProcess 封装 Python 子进程的完整生命周期管理，内置进程清理、输出缓冲禁用、日志去重、超时杀死、异常退出捕获等多重健壮性机制，保障跨语言通信稳定。
🐍 算法脚本层
独立 Python 脚本集，与 C++ 端通过标准输入输出 + JSON 结构化数据通信：
train.py：封装 Ultralytics YOLOv11 训练逻辑，自定义回调输出结构化日志
quantize.py：FP16 量化 + ONNX 导出，兼容性功能降级处理
pt_infer.py：PyTorch 后端推理，输出标准化检测结果
infer.py：ONNX 后端推理，对齐推理参数与输出格式
