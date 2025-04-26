from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Any, Generator, Tuple
import os
import base64
from ..core.config import settings
import datetime

class YOLODetector:
    def __init__(self):
        self.model = YOLO(settings.MODEL_PATH)
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD
        self.iou_threshold = settings.IOU_THRESHOLD
        self.device = settings.DEVICE
        self.half = settings.HALF

    def detect_image(self, image_path: str) -> Dict[str, Any]:
        """检测单张图片"""
        results = self.model.predict(
            source=image_path,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            half=self.half
        )[0]

        # 处理检测结果
        detections = []
        for box, cls, conf in zip(results.boxes.xyxy.tolist(),
                                results.boxes.cls.tolist(),
                                results.boxes.conf.tolist()):
            detections.append({
                "class_name": results.names[int(cls)],
                "confidence": float(conf),
                "bbox": [float(x) for x in box]
            })

        # 保存带标注的图片
        output_path = os.path.join("results", os.path.basename(image_path))
        os.makedirs(os.path.dirname(os.path.join(settings.UPLOAD_DIR, output_path)), exist_ok=True)
        cv2.imwrite(os.path.join(settings.UPLOAD_DIR, output_path), results.plot())

        return {
            "detections": detections,
            "result_image": output_path
        }

    def detect_video(self, video_path: str) -> str:
        """检测视频并保存结果到uploads目录"""
        if not os.path.exists(video_path):
            raise ValueError(f"视频文件不存在: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("无法打开视频文件")

        try:
            # 获取视频属性
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 创建输出目录
            output_path = os.path.join("results", os.path.basename(video_path))
            os.makedirs(os.path.dirname(os.path.join(settings.UPLOAD_DIR, output_path)), exist_ok=True)
            output_path = os.path.join(settings.UPLOAD_DIR, output_path)

            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            if not out.isOpened():
                raise ValueError("无法创建输出视频文件")

            frame_count = 0
            last_progress = -1

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 检测当前帧
                results = self.model.predict(
                    source=frame,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    device=self.device,
                    half=self.half
                )[0]

                # 绘制结果并写入输出视频
                annotated_frame = results.plot()
                out.write(annotated_frame)

                # 更新进度（只在进度变化时打印）
                frame_count += 1
                progress = int((frame_count / total_frames) * 100)
                if progress != last_progress:
                    print(f"处理进度: {progress}%")
                    last_progress = progress

            # 确保写入最后一帧
            out.release()
            cap.release()

            # 验证输出文件
            if not os.path.exists(output_path):
                raise Exception("输出视频文件创建失败")
            
            file_size = os.path.getsize(output_path)
            if file_size == 0:
                raise Exception("输出视频文件为空")

            print(f"视频处理完成，输出文件大小: {file_size / 1024 / 1024:.2f}MB")
            return output_path

        except Exception as e:
            # 如果发生错误，清理输出文件
            if 'output_path' in locals() and os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
            raise Exception(f"视频处理过程中出错: {str(e)}")

        finally:
            # 确保资源被正确释放
            if 'cap' in locals():
                cap.release()
            if 'out' in locals():
                out.release()

    def detect_camera(self, camera_id: int = 0) -> str:
        """实时摄像头检测"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError("无法打开摄像头")

        # 获取摄像头属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30.0

        # 创建输出视频
        output_path = os.path.join(settings.UPLOAD_DIR, "results", "camera_output.mp4")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 检测当前帧
                results = self.model.predict(
                    source=frame,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    device=self.device,
                    half=self.half
                )[0]

                # 绘制结果并写入输出视频
                out.write(results.plot())

        finally:
            cap.release()
            out.release()

        return output_path
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """处理单帧图像并返回检测结果和标注后的图像"""
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            half=self.half
        )[0]
        
        # 处理检测结果
        detections = []
        for box, cls, conf in zip(results.boxes.xyxy.tolist(),
                                results.boxes.cls.tolist(),
                                results.boxes.conf.tolist()):
            detections.append({
                "class_name": results.names[int(cls)],
                "confidence": float(conf),
                "bbox": [float(x) for x in box]
            })
            
        # 获取标注后的图像
        annotated_frame = results.plot()
        
        return annotated_frame, detections
        
    def process_frame_base64(self, frame_base64: str) -> Dict[str, Any]:
        """处理Base64编码的图像帧并返回检测结果和标注后的图像"""
        # 解码Base64图像
        try:
            # 移除可能的Data URL前缀
            if ',' in frame_base64:
                frame_base64 = frame_base64.split(',')[1]
                
            img_data = base64.b64decode(frame_base64)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # 处理帧
            annotated_frame, detections = self.process_frame(frame)
            
            # 将标注后的图像编码为Base64
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            annotated_frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                "frame": f"data:image/jpeg;base64,{annotated_frame_base64}",
                "detections": detections
            }
        except Exception as e:
            return {
                "error": str(e)
            }

    def update_settings(self, confidence: float = None, iou: float = None,
                       device: str = None, half: bool = None):
        """更新检测参数"""
        if confidence is not None:
            self.confidence_threshold = confidence
        if iou is not None:
            self.iou_threshold = iou
        if device is not None:
            self.device = device
        if half is not None:
            self.half = half
