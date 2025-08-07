import easyocr
import re
import cv2
import numpy as np
from PIL import Image
import logging
import os
import difflib

if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class OCRService:
    COMMON_CORRECTIONS = {
        "ขนบ": "ขนษ",
        "รของ": "ระนอง",
        "รยอง": "ระยอง",
        "บก": "บข",
        # เพิ่มเติมได้ตามปัญหาที่เจอ
    }

    def __init__(self, debug=False):
        self.debug = debug
        try:
            self.reader = easyocr.Reader(['th', 'en'], gpu=False, verbose=False)
            logger.info("✅ EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize EasyOCR: {e}")
            self.reader = None

        self.unwanted_words = {
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
            'อุทัยธานี', 'อุบลราชธานี',
            'thailand', 'ประเทศไทย', 'จังหวัด', 'province', 'ภาค', 'เมือง', 'อำเภอ', 'ตำบล',
            'หมู่บ้าน', 'ชุมชน', 'ไทย', 'thai', 'kingdom', 'ราชอาณาจักร'
        }

    def is_unwanted_text(self, text):
        text_clean = text.lower().strip()
        if len(text_clean) > 15 or len(text_clean) < 2:
            return True

        for unwanted in self.unwanted_words:
            if unwanted.lower() in text_clean:
                return True

        province_patterns = [
            r'กรุง.*เทพ.*',
            r'.*มหานคร.*',
            r'.*จังหวัด.*',
            r'^[ก-ฮ]{4,}$',  # คำยาวเกินไป อาจไม่ใช่ป้ายทะเบียน
        ]
        for pattern in province_patterns:
            if re.search(pattern, text_clean):
                return True

        return False

    def fuzzy_match_province(self, text):
        provinces = list(self.unwanted_words)
        match = difflib.get_close_matches(text, provinces, n=1, cutoff=0.7)
        return match[0] if match else None

    def preprocess_license_plate(self, img_array):
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            height, width = gray.shape
            if width < 400 or height < 150:
                scale = max(400 / width, 150 / height)
                gray = cv2.resize(gray, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_CUBIC)

            processed = []

            # CLAHE เพิ่มความเข้มคมชัด
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # Unsharp mask เพิ่มความคมชัด
            def unsharp_mask(image, kernel_size=(5,5), sigma=1.0, amount=1.5, threshold=0):
                blurred = cv2.GaussianBlur(image, kernel_size, sigma)
                sharpened = float(amount + 1) * image - float(amount) * blurred
                sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
                sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
                sharpened = sharpened.round().astype(np.uint8)
                if threshold > 0:
                    low_contrast_mask = np.absolute(image - blurred) < threshold
                    np.copyto(sharpened, image, where=low_contrast_mask)
                return sharpened

            sharpened = unsharp_mask(enhanced)
            processed.append(sharpened)

            # Threshold แบบ global และ adaptive
            _, thresh1 = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed.append(thresh1)

            adaptive = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)
            processed.append(adaptive)

            # Morphology ปิดรูในตัวอักษร
            morph = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            processed.append(morph)

            # Denoise ลด noise
            denoised = cv2.fastNlMeansDenoising(sharpened, h=10)
            processed.append(denoised)

            return processed

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return [img_array]

    def clean_license_plate_text(self, raw_text):
        if not raw_text:
            return ""

        # แก้คำผิดจาก dictionary
        if raw_text in self.COMMON_CORRECTIONS:
            raw_text = self.COMMON_CORRECTIONS[raw_text]

        # ตัดเฉพาะตัวอักษรไทยและเลข
        text = re.sub(r'[^\u0E00-\u0E7F0-9]', '', raw_text)

        patterns = [
            r'^[ก-ฮ]{1,3}\d{1,4}$',          # กข1234 หรือ กกก999
            r'^\d{1,2}[ก-ฮ]{1,3}\d{1,4}$',  # 1กข1234
            r'^[ก-ฮ]{1,2}\d{1,4}[ก-ฮ]{0,2}$' # ก1234ข
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)[:10]

        # fallback: แยกเลขกับตัวอักษร
        th = re.findall(r'[ก-ฮ]+', text)
        dg = re.findall(r'\d+', text)

        if th and dg:
            return f"{th[0][:3]}{''.join(dg)[:4]}"

        return text[:10]

    def is_valid_thai_license_plate(self, text: str) -> bool:
        if not text or len(text) < 3:
            return False

        has_thai = any('\u0E00' <= c <= '\u0E7F' for c in text)
        has_digit = any(c.isdigit() for c in text)

        if not has_thai or not has_digit:
            return False

        patterns = [
            r'^[ก-ฮ]{1,3}\d{1,4}$',
            r'^\d{1,2}[ก-ฮ]{1,3}\d{1,4}$',
            r'^[ก-ฮ]{1,2}\d{1,4}[ก-ฮ]{0,2}$',
        ]

        for pattern in patterns:
            if re.match(pattern, text):
                return True

        return False

    def debug_save_cropped_image(self, img_array, region_name):
        try:
            os.makedirs("debug_crops", exist_ok=True)
            path = f"debug_crops/{region_name}.jpg"
            cv2.imwrite(path, img_array)
            logger.info(f"🖼️ Saved cropped image for debug: {path}")
        except Exception as e:
            logger.error(f"❌ Failed to save debug image: {e}")

    def extract_text(self, image, region_name="unknown"):
        try:
            if self.reader is None:
                logger.warning("EasyOCR not available")
                return ""

            img_array = np.array(image) if isinstance(image, Image.Image) else image

            if self.debug:
                self.debug_save_cropped_image(img_array, region_name)

            processed_images = self.preprocess_license_plate(img_array)
            all_results = []

            for i, img in enumerate(processed_images):
                try:
                    results = self.reader.readtext(
                        img,
                        width_ths=0.05,
                        height_ths=0.05,
                        paragraph=False,
                        detail=1,
                        allowlist='0123456789กขฃคงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ'
                    )

                    # แยกบรรทัดที่อ่านได้ confidence > 0.35
                    lines = [text.strip() for (_, text, conf) in results if conf > 0.35]

                    filtered_lines = []
                    province_line = None
                    number_line = None

                    for line in lines:
                        corrected = self.COMMON_CORRECTIONS.get(line, line)

                        if self.is_unwanted_text(corrected):
                            # ลอง fuzzy check province
                            fuzzy_prov = self.fuzzy_match_province(corrected)
                            if fuzzy_prov:
                                province_line = fuzzy_prov
                                logger.info(f"✔️ Fuzzy matched province: '{corrected}' -> '{fuzzy_prov}'")
                            else:
                                logger.info(f"❌ Filtered out line (unwanted): '{line}'")
                            continue

                        # ถ้าเป็นเลขล้วนและความยาวไม่เกิน 4 ตัว ให้ถือว่าเป็นเลขป้าย
                        if re.fullmatch(r'\d{1,4}', corrected) or re.fullmatch(r'\d{2}[ก-ฮ]', corrected):
                            corrected_number = self.smart_correct_number(corrected)
                            number_line = corrected_number
                            logger.info(f"✔️ Corrected number line: '{corrected}' -> '{corrected_number}'")
                            continue

                        # ถ้าเป็นข้อความที่ประกอบด้วยตัวอักษรไทย (ไม่น่าจะเป็นจังหวัด)
                        if re.match(r'^[ก-ฮ]{1,3}$', corrected):
                            filtered_lines.append(corrected)
                        else:
                            logger.info(f"❌ Filtered out line (non-matching): '{line}'")

                    prefix = filtered_lines[0] if filtered_lines else ""
                    suffix = number_line if number_line else ""

                    # ถ้า province_line มี ก็เอามาแทรกได้ (แต่ถ้าใช้แบบมอเตอร์ไซค์ไม่ค่อยจำเป็น)
                    combined = prefix + suffix

                    logger.info(f"🔧 Combined text before clean: '{combined}'")

                    cleaned = self.clean_license_plate_text(combined)
                    if not self.is_valid_thai_license_plate(cleaned):
                        logger.info(f"❌ Filtered out final cleaned text as invalid plate: '{cleaned}'")
                        continue

                    all_results.append((cleaned, 1.0))  # confidence placeholder

                except Exception as e:
                    logger.error(f"OCR error in method {i+1}: {e}")

            for line in lines: # กรองตัวเลขที่มากกว่า 5 ตัวออก
                corrected = self.COMMON_CORRECTIONS.get(line, line)

                if self.is_unwanted_text(corrected):
                    # fuzzy check province
                    fuzzy_prov = self.fuzzy_match_province(corrected)
                    if fuzzy_prov:
                        province_line = fuzzy_prov
                        logger.info(f"✔️ Fuzzy matched province: '{corrected}' -> '{fuzzy_prov}'")
                    else:
                        logger.info(f"❌ Filtered out line (unwanted): '{line}'")
                    continue

                # ตัดเลขที่อาจเป็นเบอร์โทรหรือเลขยาวผิดปกติ
                if re.fullmatch(r'\d{5,}', corrected):
                    logger.info(f"❌ Filtered out long numeric line (possible phone): '{corrected}'")
                    continue

                if re.fullmatch(r'\d{1,4}', corrected):
                    number_line = corrected
                    logger.info(f"✔️ Detected number line: '{corrected}'")
                    continue

                if re.match(r'^[ก-ฮ]{1,3}$', corrected):
                    filtered_lines.append(corrected)
                else:
                    logger.info(f"❌ Filtered out line (non-matching): '{line}'")

            if not all_results:
                return ""

            all_results.sort(key=lambda x: x[1], reverse=True)
            best_text = all_results[0][0]

            logger.info(f"✅ Final cleaned text: {best_text}")
            return best_text

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""
    def smart_correct_number(self, text: str) -> str:
        """พยายามแก้ไขเลขทะเบียนที่ OCR อ่านผิด เช่น 66บ -> 660, 66 -> 660"""
        if not text:
            return text

        # กรณีที่ตัวสุดท้ายไม่ใช่เลข แต่คล้ายเลข 0
        if re.fullmatch(r'\d{2}[ก-ฮ]', text):
            likely_zero = text[-1]
            if likely_zero in {'บ', 'อ', 'ต', 'ฃ', 'ญ'}:
                return text[:2] + '0'

        # แค่ 2 ตัวเลข → น่าจะขาดศูนย์ท้าย
        if re.fullmatch(r'\d{2}', text):
            return text + '0'

        # ตัวเลขที่มีตัวอักษรไทยที่คล้ายเลขอยู่ด้านหลัง เช่น 60อ → 600
        if re.fullmatch(r'\d{2}[ก-ฮ]', text):
            if text[-1] in {'อ', 'บ'}:
                return text[:2] + '0'

        return text
