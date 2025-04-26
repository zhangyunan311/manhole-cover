from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from typing import Optional
import os
import shutil
import json
import asyncio
import cv2
import base64
from ..core.config import settings
from ..core.yolo_detector import YOLODetector
from ..schemas.detection import DetectionSettings, ImageDetectionResponse

router = APIRouter()
detector = YOLODetector()

@router.post("/detect/image", response_model=ImageDetectionResponse)
async def detect_image(file: UploadFile = File(...)):
    """上传并检测图片"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="请上传图片文件")

    # 保存上传的文件
    file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = detector.detect_image(file_path)
        return result
    except Exception as e:
        # 如果发生错误，删除上传的文件
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 确保文件被关闭
        file.file.close()

@router.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    """上传并检测视频"""
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="请上传视频文件")

    # 保存上传的文件
    file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 使用异步任务处理视频
        result_path = await asyncio.to_thread(detector.detect_video, file_path)
        
        # 验证结果文件
        if not os.path.exists(result_path):
            raise HTTPException(status_code=500, detail="视频处理失败：输出文件不存在")
            
        return FileResponse(
            result_path,
            media_type="video/mp4",
            headers={
                "Content-Disposition": f'attachment; filename="{os.path.basename(result_path)}"'
            }
        )
    except Exception as e:
        # 清理上传的文件
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 确保文件被关闭
        file.file.close()

@router.post("/detect/camera")
async def detect_camera():
    """启动摄像头检测"""
    try:
        result_path = detector.detect_camera()
        return FileResponse(result_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/settings")
async def update_settings(settings: DetectionSettings):
    """更新检测参数"""
    try:
        detector.update_settings(
            confidence=settings.confidence,
            iou=settings.iou,
            device=settings.device,
            half=settings.half
        )
        return {"message": "设置已更新"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/video/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket端点，用于实时视频检测"""
    await websocket.accept()
    
    try:
        while True:
            # 接收客户端发送的帧
            data = await websocket.receive_text()
            
            try:
                # 解析JSON数据
                json_data = json.loads(data)
                
                if "frame" in json_data:
                    # 处理帧并返回结果
                    result = detector.process_frame_base64(json_data["frame"])
                    await websocket.send_json(result)
                else:
                    await websocket.send_json({"error": "无效的帧数据"})
            except json.JSONDecodeError:
                await websocket.send_json({"error": "无效的JSON数据"})
            except Exception as e:
                await websocket.send_json({"error": str(e)})
                
    except WebSocketDisconnect:
        # 客户端断开连接
        pass
    except Exception as e:
        # 其他错误
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass

@router.websocket("/video/ws/upload")
async def video_websocket_endpoint(websocket: WebSocket):
    """WebSocket端点，用于上传视频并实时检测"""
    await websocket.accept()
    
    try:
        # 接收视频数据
        data = await websocket.receive_bytes()
        
        # 保存视频文件
        video_path = os.path.join(settings.UPLOAD_DIR, "temp_video.mp4")
        with open(video_path, "wb") as f:
            f.write(data)
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            await websocket.send_json({"error": "无法打开视频文件"})
            return
        
        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 发送视频信息
        await websocket.send_json({
            "info": {
                "fps": fps,
                "frame_count": frame_count
            }
        })
        
        # 处理视频帧
        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 每隔几帧处理一次，减少处理负担
            if frame_index % 3 == 0:
                # 处理帧
                annotated_frame, detections = detector.process_frame(frame)
                
                # 将标注后的图像编码为Base64
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                annotated_frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # 发送结果
                await websocket.send_json({
                    "frame": f"data:image/jpeg;base64,{annotated_frame_base64}",
                    "detections": detections,
                    "frame_index": frame_index,
                    "progress": frame_index / frame_count
                })
                
                # 等待一小段时间，避免发送过快
                await asyncio.sleep(0.05)
            
            frame_index += 1
        
        # 关闭视频
        cap.release()
        
        # 发送完成消息
        await websocket.send_json({
            "status": "completed",
            "message": "视频处理完成"
        })
        
    except WebSocketDisconnect:
        # 客户端断开连接
        pass
    except Exception as e:
        # 其他错误
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
