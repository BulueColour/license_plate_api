import os
import logging
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch

logger = logging.getLogger(__name__)

class LicensePlateDetector:
    def __init__(self, confidence_threshold=0.03):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.model_type = "unknown"
        self._load_best_available_model()
    
    def _load_best_available_model(self):
        model_paths = [
            ("new_trained_model.pt", "custom_latest_new"),
            ("models/new_trained_model.pt", "custom_latest_new2"),
            ("models/thai_license_plate.pt", "custom_production"),
            ("models/yolov8n.pt", "pretrained"),
            ("yolov8n.pt", "pretrained_fallback")
        ]
        
        for model_path, model_type in model_paths:
            try:
                if os.path.exists(model_path):
                    self.model = YOLO(model_path)
                    self.model_type = model_type
                    logger.info(f"✅ Loaded {model_type} model: {model_path}")
                    
                    if model_type.startswith('custom'):
                        self.confidence_threshold = 0.05
                    else:
                        self.confidence_threshold = 0.25
                    
                    logger.info(f"Set confidence threshold: {self.confidence_threshold}")
                    return
            except Exception as e:
                logger.warning(f"Failed to load {model_type} model: {e}")
                continue
        
        logger.error("❌ Failed to load any model!")
        self.model = None
        self.model_type = "none"
    
    def detect_license_plates(self, image):  # YOLO version
        try:
            if self.model is None:
                logger.error("No model available for detection")
                return self._fallback_detection(image)

            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            results = self.model(cv_image, conf=self.confidence_threshold, verbose=False)

            detected_boxes = []

            # เก็บ box ทั้งหมดพร้อม area
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        area = (x2 - x1) * (y2 - y1)

                        detected_boxes.append({
                            'box': (x1, y1, x2, y2),
                            'confidence': confidence,
                            'class_id': class_id,
                            'area': area
                        })

            if not detected_boxes:
                return self._fallback_detection(image)

            # เลือก box ที่มีพื้นที่ใหญ่ที่สุด
            selected_box = max(detected_boxes, key=lambda b: b['area'])
            x1, y1, x2, y2 = selected_box['box']
            confidence = selected_box['confidence']
            class_id = selected_box['class_id']

            # วาด label
            labeled_img = cv_image.copy()
            cv2.rectangle(labeled_img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(labeled_img, f"ID:{class_id} Conf:{confidence:.2f}", 
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.imshow("Labeled Detection", labeled_img)
            cv2.waitKey(0)

            # crop detection
            crop = cv_image[y1:y2, x1:x2]
            cv2.imshow("Cropped Plate", crop)
            cv2.waitKey(0)

            detected_plates = [{'image': crop, 'class_id': class_id, 'confidence': confidence}]
            cv2.destroyAllWindows()
            return detected_plates

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return self._fallback_detection(image)
    
    # def detect_license_plates(self, image): # unYOLO version
    #     """
    #     ตรวจจับป้ายทะเบียน (แบบไม่ใช้ YOLO)
    #     ส่ง full image ให้ OCR พร้อม return_province
    #     คืนค่า list ของ dict ที่มี plate_text และ province
    #     """
    #     try:
    #         # แปลงเป็น OpenCV format
    #         cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    #         # ส่งภาพเต็มคืน โดยไม่ใช้ YOLO
    #         logger.warning("🚫 Skipping YOLO and sending full image for OCR directly")
    #         region = {
    #             'bbox': [0, 0, cv_image.shape[1], cv_image.shape[0]],
    #             'confidence': 1.0,
    #             'image': cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB),
    #             'source': 'full_image'
    #         }

    #         # เรียก OCR พร้อม return_province=True
    #         if hasattr(self, 'ocr_service') and self.ocr_service is not None:
    #             plate_text, province = self.ocr_service.extract_text(region['image'], return_province=True)
    #             region['plate_text'] = plate_text
    #             region['province'] = province
    #         else:
    #             region['plate_text'] = ""
    #             region['province'] = None

    #         return [region]

    #     except Exception as e:
    #         logger.error(f"Detection failed: {e}")
    #         return []
    
    def _extract_detection(self, box, cv_image, confidence, model_type):
        """Extract detection data from bounding box"""
        try:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Ensure coordinates are within image bounds
            h, w = cv_image.shape[:2]
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            # เพิ่มขอบเขตรอบๆ detection เล็กน้อย
            padding_x = int((x2 - x1) * 0.05)
            padding_y = int((y2 - y1) * 0.05)
            
            x1 = max(0, x1 - padding_x)
            y1 = max(0, y1 - padding_y)
            x2 = min(w, x2 + padding_x)
            y2 = min(h, y2 + padding_y)
                
            cropped = cv_image[y1:y2, x1:x2]
            if cropped.size == 0:
                return None
                
            return {
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'image': cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB),
                'source': model_type
            }
        except Exception as e:
            logger.error(f"Failed to extract detection: {e}")
            return None
    
    def _extract_vehicle_regions(self, box, cv_image, confidence):
        """Extract potential license plate regions from vehicle detections"""
        try:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            regions = []
            vehicle_w = x2 - x1
            vehicle_h = y2 - y1
            
            # หลายตำแหน่งที่เป็นไปได้สำหรับป้ายทะเบียน
            positions = [
                # ด้านล่างของรถ (70-100%)
                (x1, y1 + int(vehicle_h * 0.7), x2, y2),
                # กลางล่าง (60-90%)
                (x1, y1 + int(vehicle_h * 0.6), x2, y1 + int(vehicle_h * 0.9)),
                # หน้ารถ (75-95%)
                (x1 + int(vehicle_w * 0.2), y1 + int(vehicle_h * 0.75), 
                 x2 - int(vehicle_w * 0.2), y1 + int(vehicle_h * 0.95)),
                # หลังรถ (80-100%)
                (x1 + int(vehicle_w * 0.15), y1 + int(vehicle_h * 0.8), 
                 x2 - int(vehicle_w * 0.15), y2),
            ]
            
            h, w = cv_image.shape[:2]
            
            for i, (rx1, ry1, rx2, ry2) in enumerate(positions):
                # ตรวจสอบขอบเขต
                rx1 = max(0, min(rx1, w))
                ry1 = max(0, min(ry1, h))
                rx2 = max(0, min(rx2, w))
                ry2 = max(0, min(ry2, h))
                
                if rx2 <= rx1 or ry2 <= ry1:
                    continue
                    
                # ตรวจสอบขนาดขั้นต่ำ
                if (rx2 - rx1) < 50 or (ry2 - ry1) < 20:
                    continue
                    
                cropped = cv_image[ry1:ry2, rx1:rx2]
                if cropped.size == 0:
                    continue
                    
                regions.append({
                    'bbox': [rx1, ry1, rx2, ry2],
                    'confidence': confidence * 0.7,  # ลดความมั่นใจเล็กน้อย
                    'image': cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB),
                    'source': f'vehicle_region_{i}'
                })
                
            return regions
        except Exception as e:
            logger.error(f"Failed to extract vehicle regions: {e}")
            return []
    
    def _create_enhanced_fallback_regions(self, cv_image):
        """สร้าง fallback regions ที่ดีขึ้น"""
        try:
            h, w = cv_image.shape[:2]
            regions = []
            
            # ตำแหน่งที่เป็นไปได้มากขึ้น
            positions = [
                # ด้านล่างกลาง - ตำแหน่งทั่วไปของป้ายทะเบียน
                (int(w * 0.15), int(h * 0.7), int(w * 0.85), int(h * 0.95)),
                # ด้านล่างซ้าย
                (int(w * 0.05), int(h * 0.75), int(w * 0.45), int(h * 0.95)),
                # ด้านล่างขวา
                (int(w * 0.55), int(h * 0.75), int(w * 0.95), int(h * 0.95)),
                # กลางภาพ
                (int(w * 0.25), int(h * 0.4), int(w * 0.75), int(h * 0.7)),
                # เต็มด้านล่าง
                (0, int(h * 0.8), w, h),
                # แบ่งครึ่งซ้าย
                (0, int(h * 0.5), int(w * 0.5), h),
                # แบ่งครึ่งขวา
                (int(w * 0.5), int(h * 0.5), w, h),
            ]
            
            for i, (x1, y1, x2, y2) in enumerate(positions):
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # ตรวจสอบขนาดขั้นต่ำ
                if (x2 - x1) < 100 or (y2 - y1) < 50:
                    continue
                    
                cropped = cv_image[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue
                    
                regions.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': 0.15,  # เพิ่มความมั่นใจสำหรับ fallback
                    'image': cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB),
                    'source': f'enhanced_fallback_{i}'
                })
                
            return regions
        except Exception as e:
            logger.error(f"Failed to create enhanced fallback regions: {e}")
            return []
    
    def _create_fallback_regions(self, cv_image):
        """Create basic fallback regions when no detections are found"""
        try:
            h, w = cv_image.shape[:2]
            regions = []
            
            positions = [
                # ตำแหน่งพื้นฐาน
                (int(w * 0.2), int(h * 0.7), int(w * 0.8), int(h * 0.95)),
                (int(w * 0.25), int(h * 0.75), int(w * 0.75), int(h * 0.9)),
                (int(w * 0.3), int(h * 0.4), int(w * 0.7), int(h * 0.6)),
            ]
            
            for i, (x1, y1, x2, y2) in enumerate(positions):
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                cropped = cv_image[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue
                    
                regions.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': 0.1,
                    'image': cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB),
                    'source': f'fallback_{i}'
                })
                
            return regions
        except Exception as e:
            logger.error(f"Failed to create fallback regions: {e}")
            return []
    
    def _remove_duplicates(self, detections):
        """Remove duplicate detections based on overlap"""
        if not detections:
            return []
            
        # เรียงตาม confidence (สูงสุดก่อน)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        unique_detections = []
        for detection in detections:
            is_duplicate = False
            for existing in unique_detections:
                overlap = self._calculate_overlap(detection['bbox'], existing['bbox'])
                if overlap > 0.3:  # ลด threshold จาก 0.5 เป็น 0.3
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_detections.append(detection)
                
        return unique_detections
    
    def _calculate_overlap(self, bbox1, bbox2):
        """Calculate overlap ratio between two bounding boxes"""
        try:
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2
            
            # คำนวณ intersection
            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)
            
            if x2_i <= x1_i or y2_i <= y1_i:
                return 0.0
                
            intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
            
            # คำนวณ union
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union_area = area1 + area2 - intersection_area
            
            return intersection_area / union_area if union_area > 0 else 0.0
        except Exception:
            return 0.0
    
    def _fallback_detection(self, image):
        """Fallback detection when main detection fails"""
        try:
            logger.warning("Using fallback detection")
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            return self._create_enhanced_fallback_regions(cv_image)
        except Exception as e:
            logger.error(f"Fallback detection failed: {e}")
            return []
    
    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            'model_type': self.model_type,
            'confidence_threshold': self.confidence_threshold,
            'model_available': self.model is not None
        }
    
    def set_confidence_threshold(self, threshold):
        """Set confidence threshold for detections"""
        self.confidence_threshold = max(0.01, min(1.0, threshold))
        logger.info(f"Confidence threshold set to: {self.confidence_threshold}")
        
    def debug_detection(self, image):
        """Debug function to show what the detector is seeing"""
        try:
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            h, w = cv_image.shape[:2]
            logger.info(f"Image dimensions: {w}x{h}")
            
            if self.model is not None:
                results = self.model(cv_image, conf=0.01, verbose=True)  # Very low confidence for debug
                logger.info(f"Raw detection results: {len(results)}")
                
                for i, result in enumerate(results):
                    logger.info(f"Result {i}: {result}")
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        logger.info(f"Boxes: {len(result.boxes)}")
                        for j, box in enumerate(result.boxes):
                            logger.info(f"  Box {j}: conf={float(box.conf[0]):.3f}, cls={int(box.cls[0])}, bbox={box.xyxy[0].tolist()}")
                            
        except Exception as e:
            logger.error(f"Debug detection failed: {e}")
            
    def enhance_image_for_detection(self, image):
        """ปรับปรุงภาพเพื่อช่วยในการตรวจจับ"""
        try:
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # ปรับขนาดถ้าเล็กเกินไป
            h, w = cv_image.shape[:2]
            if w < 640 or h < 480:
                scale = max(640/w, 480/h)
                new_w, new_h = int(w * scale), int(h * scale)
                cv_image = cv2.resize(cv_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                logger.info(f"Resized image from {w}x{h} to {new_w}x{new_h}")
                
            # ปรับ brightness/contrast เล็กน้อย
            alpha = 1.1  # contrast
            beta = 10    # brightness
            enhanced = cv2.convertScaleAbs(cv_image, alpha=alpha, beta=beta)
            
            return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
            
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            return image