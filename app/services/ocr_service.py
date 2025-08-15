from PIL import Image
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

import easyocr
import re
import cv2
import numpy as np
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
        "77อ": "772",
        # เพิ่มการแก้ไขสำหรับกรุงเทพมหานคร
        "งเทพมหาน": "กรุงเทพมหานคร",
        "งทพมหาน": "กรุงเทพมหานคร",
        "งทพมทวน": "กรุงเทพมหานคร",
        "รุงเทพฯ": "กรุงเทพฯ",
        "รุงเทพมหานคร": "กรุงเทพมหานคร",
        "กรงททพยพทนคร": "กรุงเทพมหานคร",
        "กรงทพมห1นคร": "กรุงเทพมหานคร",
        "กรงทพมนวนคร": "กรุงเทพมหานคร",
        "กรงทพมหวนคร": "กรุงเทพมหานคร",
        # เพิ่มการแก้ไขฉะเชิงเทรา
        "ฉรชงททรก": "ฉะเชิงเทรา",
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

    def partial_match_province(self, text, min_length=3):
        """
        จับคู่จังหวัดโดยใช้ partial string matching
        เหมาะสำหรับกรณีที่ชื่อจังหวัดถูกตัดไปบางส่วน
        """
        if len(text) < min_length:
            return None
            
        best_match = None
        best_score = 0
        
        for province in self.provinces:
            max_common_len = 0
            
            # ตรวจสอบว่าเป็น substring ของ province
            if text in province:
                score = len(text) / len(province)
                if province.startswith(text):
                    score += 0.4
                elif province.endswith(text):
                    score += 0.3
                else:
                    score += 0.2
                    
                if score > best_score:
                    best_score = score
                    best_match = province
                    max_common_len = len(text)
            
            # ตรวจสอบ province เป็น substring ของ text
            elif province in text and len(province) >= min_length:
                score = len(province) / len(text) + 0.1
                if score > best_score:
                    best_score = score
                    best_match = province
                    max_common_len = len(province)
            
            else:
                for i in range(len(text) - min_length + 1):
                    for j in range(i + min_length, len(text) + 1):
                        substring = text[i:j]
                        if substring in province and len(substring) > max_common_len:
                            max_common_len = len(substring)
                            # คำนวณคะแนนตามความยาวของส่วนที่ตรงกัน
                            score = (len(substring) / len(province)) * 0.8
                            # ให้คะแนนพิเศษถ้าเริ่มต้นหรือท้ายของจังหวัด
                            if province.startswith(substring):
                                score += 0.4
                            elif province.endswith(substring):
                                score += 0.3
                            else:
                                score += 0.2
                            
                            if score > best_score:
                                best_score = score
                                best_match = province
        
        # ลดเกณฑ์การตัดสินใจลงเหลือ 0.25 เพื่อให้จับได้ง่ายขึ้น
        logger.info(f"🔍 Partial matching '{text}': best_match='{best_match}', score={best_score:.3f}")
        return best_match if best_match and best_score >= 0.25 else None

    def fuzzy_match_province(self, text):
        """
        จับคู่จังหวัดโดยใช้ fuzzy matching (สำหรับตัวอักษรที่ผิด)
        """
        match = difflib.get_close_matches(text, self.provinces, n=1, cutoff=0.6)
        return match[0] if match else None

    def match_province(self, text):
        """
        รวมการจับคู่จังหวัดทั้ง partial และ fuzzy matching
        """
        if not text or len(text) < 2:
            return None
            
        # ลองใช้ exact match ก่อน
        if text in self.provinces:
            return text
            
        # ลองใช้ partial match
        partial_result = self.partial_match_province(text)
        if partial_result:
            logger.info(f"🎯 Partial match found: '{text}' -> '{partial_result}'")
            return partial_result
            
        # ถ้าไม่ได้ก็ใช้ fuzzy match
        fuzzy_result = self.fuzzy_match_province(text)
        if fuzzy_result:
            logger.info(f"🔍 Fuzzy match found: '{text}' -> '{fuzzy_result}'")
            return fuzzy_result
            
        return None

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
        # ลบเฉพาะอักขระที่ไม่ใช่ ไทย/เลข แต่เก็บสระและวรรณยุกต์ครบ
        text = re.sub(r'[^\u0E00-\u0E7F0-9]', '', text)
        # เพิ่มความยาวสำหรับชื่อจังหวัดที่อาจยาว เช่น กรุงเทพมหานคร
        return text[:20]  # เพิ่มจาก 15 เป็น 20

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
        letters_list = []
        digits_list = []
        province_candidates = []
        for idx, img in enumerate(processed_images):
            results = self.reader.readtext(
                img,
                width_ths=0.05, height_ths=0.05, paragraph=False, detail=1,
                allowlist='0123456789กขฃคงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ'
            )
            logger.info(f"🔹 Processed image {idx+1}: found {len(results)} OCR lines")
            for bbox, text, conf in results:
                cleaned = self.clean_text(text)
                matched_province = self.match_province(cleaned)
                logger.info(f"📝 OCR line: '{cleaned}' | confidence: {conf:.3f} | matched_province: {matched_province}")
                if matched_province:
                    province_candidates.append((matched_province, conf, cleaned))
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

        valid_digits_list = [
            (d, conf) for d, conf in digits_list
            if (d.isdigit() and 1 <= len(d) <= 4) or self.is_valid_license_plate(d)
        ]
        best_digits = max(valid_digits_list, key=lambda x: x[1])[0] if valid_digits_list else ""

        # เลือกตัวอักษรไทย 2–3 ตัวก่อน ถ้าไม่มีค่อย fallback
        valid_letters_list = [
            (l, conf) for l, conf in letters_list
            if re.fullmatch(r'[ก-ฮ]{2,3}', l)
        ]
        if valid_letters_list:
            best_letters = max(valid_letters_list, key=lambda x: x[1])[0]
        else:
            best_letters = max(letters_list, key=lambda x: x[1])[0] if letters_list else ""

        best_province = ""
        if province_candidates:
            province_candidates.sort(key=lambda x: x[1], reverse=True)
            best_province = province_candidates[0][0]
            logger.info(f"🏆 Selected province: '{best_province}' (from '{province_candidates[0][2]}', conf: {province_candidates[0][1]:.3f})")

        combined_text = f"{best_letters}{best_digits}"
        if best_province:
            combined_text += f" {best_province}"
        logger.info(f"✅ Selected combined text: '{combined_text}'")
        return combined_text