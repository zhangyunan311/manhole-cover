from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from .core.config import settings
from .api.endpoints import router
from .api.qwenvl import router as qwenvl_router
from .api.user import router as user_router
from .models.user import User
from .utils.auth import get_current_user
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import os

# 创建上传目录
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.join(settings.UPLOAD_DIR, "results"), exist_ok=True)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

# 注册路由
app.include_router(router, prefix=settings.API_V1_STR)
app.include_router(qwenvl_router, prefix=settings.API_V1_STR)
app.include_router(user_router, prefix=settings.API_V1_STR)
@app.get("/")
async def root():
    return RedirectResponse(url="/static/login.html")

# 保护需要认证的路由
@app.get("/static/index.html")
async def protected_index(current_user: User = Depends(get_current_user)):
    return FileResponse("app/static/index.html")

@app.get("/static/image.html")
async def protected_image(current_user: User = Depends(get_current_user)):
    return FileResponse("app/static/image.html")

@app.get("/static/video.html")
async def protected_video(current_user: User = Depends(get_current_user)):
    return FileResponse("app/static/video.html")

@app.get("/static/vision.html")
async def protected_vision(current_user: User = Depends(get_current_user)):
    return FileResponse("app/static/vision.html")
