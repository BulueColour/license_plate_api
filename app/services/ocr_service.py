from PIL import Image
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

import easyocr
import re
import cv2
import numpy as np
from PIL import Image
import logging
import difflib

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class OCRService:
    COMMON_CORRECTIONS = {
        "ขนบ": "ขนษ",
        "รของ": "ระนอง",
        "รยอง": "ระยอง",
        "บก": "บข",
        "6อบ": "660",
        "6บ": "660",
        "66บ": "660",
    }

    def __init__(self, debug=False):
        self.debug = debug
        try:
            self.reader = easyocr.Reader(['th', 'en'], gpu=False, verbose=False)
            logger.info("✅ EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize EasyOCR: {e}")
            self.reader = None

        self.provinces = {
            'กรุงเทพมหานคร', 'กรุงเทพฯ', 'กระบี่', 'กาญจนบุรี', 'กาฬสินธุ์', 'กำแพงเพชร',
            'ขอนแก่น', 'จันทบุรี', 'ฉะเชิงเทรา', 'ชลบุรี', 'ชัยนาท', 'ชัยภูมิ', 'ชุมพร',
            'เชียงราย', 'เชียงใหม่', 'ตรัง', 'ตราด', 'ตาก', 'นครนายก', 'นครปฐม', 'นครพนม',
            'นครราชสีมา', 'นครศรีธรรมราช', 'นครสวรรค์', 'นนทบุรี', 'นราธิวาส', 'น่าน',
            'บึงกาฬ', 'บุรีรัมย์', 'ปทุมธานี', 'ประจวบคีรีขันธ์', 'ปราจีนบุรี', 'ปัตตานี',
            'พระนครศรีอยุธยา', 'พะเยา', 'พังงา', 'พัทลุง', 'พิจิตร', 'พิษณุโลก', 'เพชรบุรี',
            'เพชรบูรณ์', 'แพร่', 'ภูเก็ต', 'มหาสารคาม', 'มุกดาหาร', 'แม่ฮ่องสอน', 'ยโสธร',
            'ยะลา', 'ร้อยเอ็ด', 'ระนอง', 'ระยอง', 'ราชบุรี', 'ลพบุรี', 'ลำปาง', 'ลำพูน',
            'เลย', 'ศรีสะเกษ', 'สกลนคร', 'สงขลา', 'สตูล', 'สมุทรปราการ', 'สมุทรสงคราม',
            'สมุทรสาคร', 'สระแก้ว', 'สระบุรี', 'สิงห์บุรี', 'สุโขทัย', 'สุพรรณบุรี', 'สุราษฎร์ธานี',
            'สุรินทร์', 'หนองคาย', 'หนองบัวลำภู', 'อ่างทอง', 'อำนาจเจริญ', 'อุดรธานี', 'อุตรดิตถ์',
            'อุทัยธานี', 'อุบลราชธานี'
        }

    def fuzzy_match_province(self, text):
        match = difflib.get_close_matches(text, self.provinces, n=1, cutoff=0.6)
        return match[0] if match else None

    def preprocess_image(self, img_array):
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            height, width = gray.shape
            if width < 400 or height < 150:
                scale = max(400 / width, 150 / height)
                gray = cv2.resize(gray, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_CUBIC)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            blurred = cv2.GaussianBlur(enhanced, (5,5), 1.0)
            sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

            return [
                sharpened,
                cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
            ]
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return [img_array]

    def clean_text(self, text):
        if not text:
            return ""
        text = text.strip()
        if text in self.COMMON_CORRECTIONS:
            text = self.COMMON_CORRECTIONS[text]
        text = re.sub(r'[^\u0E00-\u0E7F0-9]', '', text)
        return text[:10]

    def is_valid_license_plate(self, text):
        patterns = [
            r'^[ก-ฮ]{1,3}\d{1,4}$',
            r'^\d{1,2}[ก-ฮ]{1,3}\d{1,4}$',
            r'^[ก-ฮ]{1,2}\d{1,4}[ก-ฮ]{0,2}$',
        ]
        return any(re.match(p, text) for p in patterns)

    def extract_text(self, image):
        if self.reader is None:
            logger.warning("EasyOCR not available")
            return ""

        img_array = np.array(image) if isinstance(image, Image.Image) else image
        processed_images = self.preprocess_image(img_array)

        letters_list = []  # [('กกก', confidence), ...]
        digits_list = []   # [('999', confidence), ...]
        province_candidate = ("", 0)  # (province_name, confidence)

        for idx, img in enumerate(processed_images):
            results = self.reader.readtext(
                img,
                width_ths=0.05, height_ths=0.05, paragraph=False, detail=1,
                allowlist='0123456789กขฃคงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ'
            )
            logger.info(f"🔹 Processed image {idx+1}: found {len(results)} OCR lines")

            for bbox, text, conf in results:
                cleaned = self.clean_text(text)
                fuzzy_prov = self.fuzzy_match_province(cleaned)
                logger.info(f"📝 OCR line: '{cleaned}' | confidence: {conf:.3f} | fuzzy_province: {fuzzy_prov}")

                if fuzzy_prov:
                    if conf > province_candidate[1]:
                        province_candidate = (fuzzy_prov, conf)
                elif re.search(r'[ก-ฮ]', cleaned) and not re.search(r'\d', cleaned):
                    letters_list.append((cleaned, conf))
                elif re.search(r'\d', cleaned) and not re.search(r'[ก-ฮ]', cleaned):
                    digits_list.append((cleaned, conf))
                else:
                    letters_part = ''.join(re.findall(r'[ก-ฮ]', cleaned))
                    digits_part = ''.join(re.findall(r'\d', cleaned))
                    if letters_part:
                        letters_list.append((letters_part, conf))
                    if digits_part:
                        digits_list.append((digits_part, conf))

        # กรองตัวเลขให้ตรงรูปแบบทะเบียนและไม่เกิน 4 ตัว
        valid_digits_list = [
            (d, conf) for d, conf in digits_list
            if (d.isdigit() and 1 <= len(d) <= 4) or self.is_valid_license_plate(d)
        ]

        # เลือกตัวเลขที่มั่นใจที่สุดจาก valid_digits_list
        best_digits = max(valid_digits_list, key=lambda x: x[1])[0] if valid_digits_list else ""

        # เลือกตัวอักษรและจังหวัด
        best_letters = max(letters_list, key=lambda x: x[1])[0] if letters_list else ""
        best_province = province_candidate[0]

        combined_text = f"{best_letters}{best_digits}{best_province}"
        logger.info(f"✅ Selected combined text: '{combined_text}'")
        return combined_text
