from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import io
import logging
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="License Plate Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = None
ocr_service = None
executor = ThreadPoolExecutor(max_workers=3)

@app.on_event("startup")
async def startup_event():
    global detector, ocr_service
    try:
        logger.info("🚀 Initializing AI services...")
        from app.services.detection_service import LicensePlateDetector
        from app.services.ocr_service import OCRService

        detector = LicensePlateDetector()
        ocr_service = OCRService(debug=True)

        logger.info("✅ All AI services initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize AI services: {e}")
        logger.error(traceback.format_exc())
        detector = None
        ocr_service = None

@app.get("/")
async def root():
    return {"message": "License Plate Detection API is running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "yolo_loaded": detector is not None,
        "ocr_loaded": ocr_service is not None
    }

@app.post("/detect-license-plate")
async def detect_license_plate(file: UploadFile = File(...)):
    start_time = time.time()
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        logger.info(f"📷 Image loaded: {image.size}")

        if detector is None or ocr_service is None:
            return {"success": False, "message": "AI services not loaded", "combined_text": None, "processing_time": 0}

        # ตรวจจับป้าย (YOLO) ก่อน แต่ถ้า skip YOLO จะใช้ทั้งภาพ
        detected_plates = await asyncio.get_event_loop().run_in_executor(
            executor, detector.detect_license_plates, image
        )
        logger.info(f"✅ Detection: {len(detected_plates)} regions")

        # OCR
        combined_text = ""
        if detected_plates:
            # ใช้ผลแรก (หรือเปลี่ยน logic เลือกที่มั่นใจที่สุด)
            plate_image = detected_plates[0]['image']
            combined_text = await asyncio.get_event_loop().run_in_executor(
                executor, lambda: ocr_service.extract_text(plate_image)
            )
        else:
            # ถ้า YOLO skip ก็ส่งทั้งภาพให้ OCR
            combined_text = await asyncio.get_event_loop().run_in_executor(
                executor, lambda: ocr_service.extract_text(image)
            )

        total_time = time.time() - start_time

        if combined_text:
            return {
                "success": True,
                "message": "ตรวจจับป้ายทะเบียนสำเร็จ",
                "combined_text": combined_text.strip(),
                "processing_time": total_time
            }
        else:
            return {
                "success": False,
                "message": "ไม่สามารถอ่านข้อความจากป้ายทะเบียนได้",
                "combined_text": None,
                "processing_time": total_time
            }

    except Exception as e:
        logger.error(f"💥 Error: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "message": f"เกิดข้อผิดพลาด: {str(e)}",
            "combined_text": None,
            "processing_time": 0
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
