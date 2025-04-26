from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
import base64
from openai import OpenAI
from typing import Generator
import io
from PIL import Image

router = APIRouter()

async def image_to_base64(upload_file: UploadFile) -> str:
    """将上传的图片转换为base64格式"""
    try:
        # 读取文件内容
        content = await upload_file.read()
        
        # 验证是否为有效的图片
        try:
            Image.open(io.BytesIO(content))
        except Exception:
            raise HTTPException(status_code=400, detail="无效的图片文件")
        
        # 转换为base64
        base64_image = base64.b64encode(content).decode('utf-8')
        return f"data:image/{upload_file.content_type.split('/')[-1]};base64,{base64_image}"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图片处理失败: {str(e)}")

async def generate_stream_response(base64_image: str, prompt: str) -> Generator[str, None, None]:
    """生成流式响应"""
    client = OpenAI(
        api_key="sk-265bd3c271ee4ccebb59dd20740f13af",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model="qwen-vl-plus",
        messages=[
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": base64_image}
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        stream=True
    )

    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

@router.post("/detect/qwenvvl")
async def detect_image(file: UploadFile = File(...), prompt: str = Form(...)):
    """处理上传的图片并返回流式检测结果"""
    try:
        # 将图片转换为base64
        base64_image = await image_to_base64(file)
        
        # 返回流式响应
        return StreamingResponse(
            generate_stream_response(base64_image, prompt),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))