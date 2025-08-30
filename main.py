from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import threading
import time
import cv2
from safety_detect.safety_detector import SafetyDetector
from debug.score_pipeline import ScorerPipeline

latest_frame = None
latest_safety_result = None
latest_score_result = None
latest_safety_image = None
latest_score_image = None

# 初始化检测器
safety_detector = SafetyDetector()
scorer_pipeline = ScorerPipeline()

def camera_loop():
    global latest_frame
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            latest_frame = frame
        time.sleep(0.03)  # ~30fps

def safety_detect_loop():
    global latest_safety_result, latest_safety_image
    while True:
        if latest_frame is not None:
            latest_safety_image, latest_safety_result = safety_detector.detect_frame(latest_frame)
        time.sleep(3)

def score_detect_loop():
    global latest_score_result, latest_score_image
    while True:
        if latest_frame is not None:
            latest_score_image, latest_score_result = scorer_pipeline.detect_frame(latest_frame)
        time.sleep(1)

app = FastAPI()

# 更新CORS设置以允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import base64

threading.Thread(target=camera_loop, daemon=True).start()
threading.Thread(target=safety_detect_loop, daemon=True).start()
threading.Thread(target=score_detect_loop, daemon=True).start()

@app.get("/api/safety_result/")
async def get_safety_result():
    if latest_safety_result is None:
        raise HTTPException(status_code=404, detail="暂无安全检测结果")
    return latest_safety_result

@app.get("/api/score_result/")
async def get_score_result():
    if latest_score_result is None:
        raise HTTPException(status_code=404, detail="暂无打分结果")
    return latest_score_result

@app.get("/api/safety_image/")
async def get_safety_image():
    if latest_safety_image is None:
        raise HTTPException(status_code=404, detail="暂无安全检测图片")
    
    # 将图片编码为base64
    _, buffer = cv2.imencode('.jpg', latest_safety_image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {"image": f"data:image/jpeg;base64,{image_base64}"}

@app.get("/api/score_image/")
async def get_score_image():
    if latest_score_image is None:
        raise HTTPException(status_code=404, detail="暂无打分检测图片")
    
    # 将图片编码为base64
    _, buffer = cv2.imencode('.jpg', latest_score_image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {"image": f"data:image/jpeg;base64,{image_base64}"}

# 添加主程序入口
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

