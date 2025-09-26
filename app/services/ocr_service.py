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
        "รุงเทพฯ": "กรุงเทพมหานคร",
        "งทพมหวนคร": "กรุงเทพมหานคร",
        "กรงทศมทวบคร": "กรุงเทพมหานคร",
        "กกรงทพมนวนคร": "กรุงเทพมหานคร",
        "รุงเทพมหานคร": "กรุงเทพมหานคร",
        "กรงททพยพทนคร": "กรุงเทพมหานคร",
        "กรงทพมห1นคร": "กรุงเทพมหานคร",
        "กรงทพมนวนคร": "กรุงเทพมหานคร",
        "กรงทพมหวนคร": "กรุงเทพมหานคร",
        "กรงททมนวนคร": "กรุงเทพมหานคร",
        # เพิ่มการแก้ไขสำหรับกรุงเทพ - แก้จาก กกญจนบร -> กาญจนบุรี (ไม่ใช่กรุงเทพ)
        "กกญจนบร": "กาญจนบุรี",
        "กวญจนบร": "กาญจนบุรี", 
        "กญจนบร": "กาญจนบุรี",
        # เพิ่มการแก้ไขฉะเชิงเทรา
        "ฉรชงททรก": "ฉะเชิงเทรา",
        # เพิ่มการแก้ไข ร - ธ
        "ระยอง": "ระนอง",
        "รของ": "ระยอง",
        "รระบุงร": "ธระบุรี",
        "ธะนอง": "ระนอง",  
        "ธะยอง": "ระยอง",
        "ราชบุธี": "ราชบุรี",
        "ธาชบุรี": "ราชบุรี",
        "ธ้อยเอ็ด": "ร้อยเอ็ด",
        "รังควย": "ราชบุรี",
        "นครธรรมธาช": "นครศรีธรรมราช",
        "สรรบร": "สระบุรี",
        "สร8บร": "สระบุรี",
        "สุธิธธานี": "สุราษฎร์ธานี",
        "สุราดรธานี": "สุราษฎร์ธานี",
        "สุราสธรธานี": "สุราษฎร์ธานี",
        "ฉชงททรว": "ฉะเชิงเทรา",
        # เพิ่มการแก้ไขอุดรธานี
        "อุดรธภนี": "อุดรธานี",
        "อุดธธานี": "อุดรธานี",
        "อุดรราณี": "อุดรธานี",
        # เพิ่มการแก้ไขอักษรป้ายทะเบียนที่อ่านผิด - แก้ กข เป็น กช
        "กข": "กช",
        "1กข": "1กช",
        "2กข": "2กช",
        "3กข": "3กช",
        "4กข": "4กช",
        "5กข": "5กช",
        "6กข": "6กข",  # 6กข เป็นป้ายจริง ไม่ต้องแก้
        "7กข": "7กช",
        "8กข": "8กช",
        "9กข": "9กช",
        "0กข": "0กช",
        # เพิ่มการแก้ไขอักษรอื่นที่มักอ่านผิด
        "บ": "ษ",
        "กค": "กข",
        "กซ": "กช",
        "กฑ": "กด",
        "กฒ": "กท",
        "กฟ": "กผ",
        "1กค": "1กข",
        "1กซ": "1กช",
        "1กฑ": "1กด",
        "1กฒ": "1กท",
        "1กฟ": "1กผ",
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

    def smart_correct_license_chars(self, text):
        """
        แก้ไขอักษรป้ายทะเบียนที่อ่านผิดโดยดูจากบริบท
        """
        if not text:
            return text
            
        # อักษรที่มักอ่านผิด - ใช้ความน่าจะเป็นในการแก้ไข
        char_corrections = {
            'ข': 'ช',  # ข มักอ่านผิดเป็น ช ในป้ายทะเบียน
            'ค': 'ข',  # ค มักอ่านผิดเป็น ข
            'ซ': 'ช',  # ซ มักอ่านผิดเป็น ช
            'ฑ': 'ด',  # ฑ มักอ่านผิดเป็น ด
            'ฒ': 'ท',  # ฒ มักอ่านผิดเป็น ท
            'ฟ': 'ผ',  # ฟ มักอ่านผิดเป็ ผ
        }
        
        # ตรวจสอบว่าเป็นบริบทของป้ายทะเบียนหรือไม่ (มีตัวเลข)
        has_digit = any(c.isdigit() for c in text)
        
        # อักษรป้ายทะเบียนที่มีอยู่จริง - ไม่ควรแก้ไข
        real_license_chars = ['กก', 'กข', 'กค', 'กง', 'กจ', 'กฉ', 'กช', 'กซ', 'กฌ', 'กญ', 
                             'กฎ', 'กฏ', 'กฐ', 'กฑ', 'กฒ', 'กณ', 'กด', 'กต', 'กถ', 'กท', 
                             'กธ', 'กน', 'กบ', 'กป', 'กผ', 'กฝ', 'กพ', 'กฟ', 'กภ', 'กม', 
                             'กย', 'กร', 'กล', 'กว', 'กศ', 'กษ', 'กส', 'กห', 'กฬ', 'กอ', 'กฮ']
        
        corrected = ""
        original_changed = False
        
        for i, char in enumerate(text):
            # ตรวจสอบบริบท - แก้ไขเฉพาะในป้ายทะเบียน และไม่ใช่อักษรป้ายจริง
            if (has_digit and char in char_corrections):
                # ตรวจสอบว่าอักษร 2 ตัวนี้เป็นอักษรป้ายจริงหรือไม่
                if i < len(text) - 1:
                    two_chars = text[i:i+2]
                    if two_chars not in real_license_chars and char == 'ข':
                        corrected += char_corrections[char]
                        original_changed = True
                        logger.info(f"🔧 Smart character correction: '{char}' -> '{char_corrections[char]}' in context '{text}'")
                    else:
                        corrected += char
                else:
                    if char == 'ข':  # แก้ไข ข -> ช เฉพาะในบริบทป้ายทะเบียน
                        corrected += char_corrections[char]
                        original_changed = True
                        logger.info(f"🔧 Smart character correction: '{char}' -> '{char_corrections[char]}' in context '{text}'")
                    else:
                        corrected += char
            else:
                corrected += char
        
        if original_changed:
            logger.info(f"🔄 Smart correction result: '{text}' -> '{corrected}'")
                
        return corrected

    def preprocess_image(self, img_array):
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            # ✅ Resize ให้ใหญ่พอ
            height, width = gray.shape
            if width < 400 or height < 150:
                scale = max(400 / width, 150 / height)
                gray = cv2.resize(gray, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_CUBIC)

            # ✅ CLAHE (ปรับ contrast แบบ local)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # ✅ Blur เล็กน้อยแล้ว sharpen
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
            sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

            # ✅ Threshold หลายแบบ
            otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            adaptive = cv2.adaptiveThreshold(
                sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            # ✅ Morphological cleaning
            kernel = np.ones((2, 2), np.uint8)
            morph_open = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)
            morph_close = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)

            return [sharpened, otsu, adaptive, morph_open, morph_close]

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return [img_array]

    def clean_text(self, text):
        if not text:
            return ""
        text = text.strip()
        
        # ✅ วิธีที่ 1: ใช้ smart correction ก่อน (ดูบริบท)
        text = self.smart_correct_license_chars(text)
        
        # ✅ วิธีที่ 2: ใช้ COMMON_CORRECTIONS (dictionary lookup)
        if text in self.COMMON_CORRECTIONS:
            original_text = text
            text = self.COMMON_CORRECTIONS[text]
            logger.info(f"🔧 Dictionary correction: '{original_text}' -> '{text}'")
        
        # ลบเฉพาะอักขระที่ไม่ใช่ ไทย/เลข แต่เก็บสระและวรรณยุกต์ครบ
        text = re.sub(r'[^\u0E00-\u0E7F0-9]', '', text)
        # เพิ่มความยาวสำหรับชื่อจังหวัดที่อาจยาว เช่น กรุงเทพมหานคร
        return text[:20]  # เพิ่มจาก 15 เป็น 20

    def is_license_plate_fragment(self, text):
        """
        ตรวจสอบว่า text เป็นส่วนหนึ่งของป้ายทะเบียนหรือไม่
        """
        if len(text) < 1:
            return False
            
        # Pattern สำหรับ fragment ของป้ายทะเบียน
        fragment_patterns = [
            r'^\d+[ก-ฮ]+$',        # เช่น "1กช" 
            r'^[ก-ฮ]+\d*$',        # เช่น "กช" หรือ "กช123"
            r'^\d+$',              # เช่น "4559"
            r'^[ก-ฮ]+$',           # เช่น "กช"
        ]
        
        return any(re.match(p, text) for p in fragment_patterns)

    def is_valid_license_plate(self, text):
        """
        ตรวจสอบว่า text เป็นป้ายทะเบียนที่สมบูรณ์หรือไม่
        """
        if len(text) < 3:
            return False
            
        patterns = [
            r'^[ก-ฮ]{1,3}\d{1,4}$',        # เช่น กช4559
            r'^\d{1,2}[ก-ฮ]{1,3}\d{1,4}$', # เช่น 1กช4559
            r'^[ก-ฮ]{1,2}\d{1,4}[ก-ฮ]{0,2}$', # เช่น กช4559ก
        ]
        return any(re.match(p, text) for p in patterns)

    def combine_license_plate_fragments(self, fragments):
        """
        รวม fragments ของป้ายทะเบียนเป็นป้ายทะเบียนเต็ม
        """
        if not fragments:
            return []
            
        # เรียง fragments ตาม confidence
        fragments.sort(key=lambda x: x[1], reverse=True)
        
        combined_plates = []
        used_fragments = set()
        
        logger.info(f"🔧 Combining fragments: {[(f[0], f[1]) for f in fragments]}")
        
        for i, (text1, conf1, bbox1) in enumerate(fragments):
            if i in used_fragments:
                continue
                
            # ลองรวมกับ fragments อื่น
            for j, (text2, conf2, bbox2) in enumerate(fragments):
                if i == j or j in used_fragments:
                    continue
                    
                # ลองรวมแบบต่างๆ
                combinations = [
                    text1 + text2,  # รวมตรงๆ
                    text2 + text1,  # รวมกลับกัน
                ]
                
                for combined in combinations:
                    if self.is_valid_license_plate(combined):
                        avg_conf = (conf1 + conf2) / 2
                        combined_plates.append((combined, avg_conf))
                        used_fragments.add(i)
                        used_fragments.add(j)
                        logger.info(f"✅ Combined '{text1}' + '{text2}' = '{combined}' (conf={avg_conf:.3f})")
                        break
                        
                if i in used_fragments:
                    break
        
        # เพิ่ม fragments ที่เป็นป้ายทะเบียนสมบูรณ์อยู่แล้ว
        for i, (text, conf, bbox) in enumerate(fragments):
            if i not in used_fragments and self.is_valid_license_plate(text):
                combined_plates.append((text, conf))
                logger.info(f"✅ Complete plate found: '{text}' (conf={conf:.3f})")
        
        return combined_plates

    def extract_text(self, image):
        if self.reader is None:
            logger.warning("EasyOCR not available")
            return ""

        img_array = np.array(image) if isinstance(image, Image.Image) else image
        processed_images = self.preprocess_image(img_array)

        plate_fragments = []  # เก็บ fragments ของป้ายทะเบียน
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
                if not cleaned:
                    continue

                logger.info(f"📝 Cleaned text: '{text}' -> '{cleaned}' (conf={conf:.3f})")

                # ตรวจสอบจังหวัด
                matched_province = self.match_province(cleaned)
                if matched_province:
                    province_candidates.append((matched_province, conf, cleaned))
                    continue

                # เก็บ fragments ของป้ายทะเบียน (ทั้งที่สมบูรณ์และไม่สมบูรณ์)
                if self.is_license_plate_fragment(cleaned):
                    plate_fragments.append((cleaned, conf, bbox))
                    logger.info(f"🧩 Plate fragment: '{cleaned}' (conf={conf:.3f})")

        # ✅ รวม fragments เป็นป้ายทะเบียนเต็ม
        combined_plates = self.combine_license_plate_fragments(plate_fragments)
        
        # ✅ เลือกป้ายทะเบียนที่ดีที่สุด
        best_plate = ""
        if combined_plates:
            # เรียงตาม confidence และความยาว (ป้ายยาวกว่าจะดีกว่า)
            combined_plates.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
            best_plate = combined_plates[0][0]
            logger.info(f"🏆 Selected plate: '{best_plate}' (conf={combined_plates[0][1]:.3f})")
        elif plate_fragments:
            # ถ้ารวมไม่ได้ ให้เอา fragment ที่ดีที่สุด
            plate_fragments.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
            best_plate = plate_fragments[0][0]
            logger.info(f"🏆 Selected plate fragment: '{best_plate}' (conf={plate_fragments[0][1]:.3f})")

        # ✅ เลือกจังหวัดที่ดีที่สุด
        best_province = ""
        if province_candidates:
            province_candidates.sort(key=lambda x: x[1], reverse=True)
            best_province = province_candidates[0][0]
            logger.info(f"🏆 Selected province: '{best_province}' (from '{province_candidates[0][2]}', conf: {province_candidates[0][1]:.3f})")

        # ✅ รวมผลลัพธ์
        combined_text = best_plate
        if best_province:
            combined_text += f" {best_province}"

        logger.info(f"✅ Final combined text: '{combined_text}'")
        return combined_text