# 井盖检测系统

一个基于 YOLOv8 和 FastAPI 的井盖检测系统，支持实时视频检测、图片检测和视频文件检测。

## 功能特点

- 实时视频检测：通过摄像头实时检测井盖状态
- 图片检测：上传图片进行井盖检测
- 视频检测：上传视频文件进行井盖检测
- 视觉大模型检测：使用大模型进行图像分析
- 用户认证：支持用户注册和登录
- 实时统计：显示检测目标的统计信息

## 技术栈

- 后端：FastAPI, Python
- 前端：HTML, JavaScript, Bootstrap
- 深度学习：YOLOv8
- 数据库：SQLite
- 认证：JWT

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/your-username/manhole-cover.git
cd manhole-cover
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 运行应用：
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 使用说明

1. 访问 `http://localhost:8000` 进入系统
2. 注册新账号或使用已有账号登录
3. 选择检测模式：
   - 实时检测：使用摄像头进行实时检测
   - 图片检测：上传图片进行检测
   - 视频检测：上传视频文件进行检测
   - 视觉大模型检测：使用大模型进行图像分析

## 项目结构

```
manhole-cover/
├── app/
│   ├── api/          # API 路由
│   ├── core/         # 核心配置
│   ├── models/       # 数据模型
│   ├── schemas/      # Pydantic 模型
│   ├── static/       # 静态文件
│   ├── utils/        # 工具函数
│   └── main.py       # 应用入口
├── uploads/          # 上传文件目录
├── requirements.txt  # 项目依赖
└── README.md         # 项目说明
```

## 许可证

MIT License 