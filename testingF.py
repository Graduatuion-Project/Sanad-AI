import json
import cv2
import os
import re
from farasa.segmenter import FarasaSegmenter

class SignLanguageTranslator:
    def __init__(self):
        self.farasa = FarasaSegmenter(interactive=True)
        self.signs_db = self.load_dictionary()
        self.stop_words = self.get_stop_words()
        self.setup_synonym_map()
        
    def load_dictionary(self):
        with open("enhanced_metadata.json", "r", encoding="utf-8") as f:
            return json.load(f)
    
    def get_stop_words(self):
        return {
            "في", "من", "إلى", "على", "عن", "أن", "إن", "كان", "كانت",
            "أو", "و", "ثم", "قد", "كل", "كما"
        }
    
    def normalize_word(self, word):
        """توحيد شكل الكلمة للمقارنة"""
        if not word:
            return word
        word = re.sub(r'[\u064b-\u065f]', '', word)  # إزالة التشكيل
        word = re.sub(r'[أإآ]', 'ا', word)          # توحيد الألف
        word = word.replace('ة', 'ه')               # توحيد التاء المربوطة
        word = word.replace('ى', 'ي')              # توحيد الألف المقصورة
        word = word.strip('ًٌٍَُِ')               # إزالة التشكيلات الإضافية
        return word.strip()
    
    def setup_synonym_map(self):
        """إنشاء خريطة عكسية للمرادفات مع تطبيع المرادفات"""
        self.synonym_to_main = {}
        for main_word, data in self.signs_db.items():
            norm_main = self.normalize_word(main_word)
            self.synonym_to_main[norm_main] = main_word
            for synonym in data.get("synonyms", []):
                norm_syn = self.normalize_word(synonym)
                if norm_syn:
                    self.synonym_to_main[norm_syn] = main_word
                    print(f"Mapped synonym '{norm_syn}' to '{main_word}'")  # Logging
    
    def find_sign_match(self, word):
        """البحث عن المطابقة مع دعم كامل للمرادفات"""
        norm_word = self.normalize_word(word)
        print(f"Searching for normalized word: '{norm_word}'")  # Logging
        
        # 1. البحث المباشر في الكلمات الرئيسية
        for main_word in self.signs_db:
            if self.normalize_word(main_word) == norm_word:
                print(f"Direct match found: '{main_word}'")  # Logging
                return self.signs_db[main_word]
        
        # 2. البحث في خريطة المرادفات
        if norm_word in self.synonym_to_main:
            main_word = self.synonym_to_main[norm_word]
            print(f"Synonym match found: '{norm_word}' -> '{main_word}'")  # Logging
            return self.signs_db[main_word]
        
        print(f"No match found for: '{norm_word}'")  # Logging
        return None
    
    def process_sentence(self, sentence):
        """معالجة الجملة الكاملة"""
        if not sentence.strip():
            return False
            
        # تقسيم الجملة باستخدام Farasa
        tokens = self.farasa.segment(sentence).split()
        print(f"Original tokens from Farasa: {tokens}")  # Logging
        
        # محاولة إعادة تجميع الكلمات اللي اتقسمت
        fixed_tokens = []
        i = 0
        while i < len(tokens):
            # جرب الكلمة لوحدها
            current_token = tokens[i]
            fixed_tokens.append(current_token)
            
            # لو الكلمة فيها '+' (يعني اتقسمت)، جرب نلمها مع اللي بعدها
            if '+' in current_token and i + 1 < len(tokens):
                combined = current_token.replace('+', '') + tokens[i + 1]
                norm_combined = self.normalize_word(combined)
                if norm_combined in self.synonym_to_main or norm_combined in [self.normalize_word(w) for w in self.signs_db]:
                    fixed_tokens[-1] = combined
                    i += 2  # اقفز الكلمة اللي بعدها
                else:
                    i += 1
            else:
                i += 1
        print(f"Fixed tokens: {fixed_tokens}")  # Logging
        
        # تصفية الكلمات
        filtered = [w for w in fixed_tokens if self.normalize_word(w) not in self.stop_words]
        print(f"Filtered tokens: {filtered}")  # Logging
        
        # البحث عن المطابقات
        matches = {}
        for word in filtered:
            sign_data = self.find_sign_match(word)
            if sign_data:
                matches[word] = sign_data
        
        print("\nالكلمات المهمة:", filtered)
        print("الكلمات المتطابقة:")
        for word, data in matches.items():
            print(f"- {word}: {data.get('type', '')}")
            if data.get("synonyms"):
                print(f"  المرادفات: {', '.join(data['synonyms'])}")
        
        # تشغيل الفيديوهات
        if matches:
            self.play_videos(matches)
        else:
            print("⚠ لا توجد كلمات متطابقة لتشغيل الفيديوهات")
        return True
    
    def play_videos(self, matches):
        """تشغيل الفيديوهات بضمانات متعددة"""
        for word, data in matches.items():
            video_path = data.get("video_path")
            
            if not video_path:
                print(f"\n⚠ لا يوجد مسار فيديو لـ: {word}")
                continue
                
            # التحقق من وجود الملف
            resolved_path = self.check_video_exists(video_path)
            if not resolved_path:
                print(f"\n⚠ ملف الفيديو غير موجود: {video_path}")
                continue
                
            print(f"\n▶ تشغيل فيديو: {word} (Path: {resolved_path})")
            
            try:
                cap = cv2.VideoCapture(resolved_path)
                if not cap.isOpened():
                    print(f"⚠ فشل فتح ملف الفيديو: {resolved_path}")
                    continue
                    
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    cv2.imshow(f"إشارة: {word}", frame)
                    if cv2.waitKey(30) & 0xFF == ord('q'):
                        break
                        
            except Exception as e:
                print(f"⚠ خطأ أثناء التشغيل: {str(e)}")
            finally:
                if 'cap' in locals():
                    cap.release()
                cv2.destroyAllWindows()
    
    def check_video_exists(self, path):
        """التحقق من وجود الفيديو وإرجاع المسار الصحيح"""
        if os.path.exists(path):
            return path
            
        try:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                return abs_path
        except:
            pass
            
        base_name = os.path.basename(path)
        if os.path.exists(base_name):
            return base_name
            
        try:
            resource_path = os.path.join("Sanad-Resources", "Resized videos", base_name)
            if os.path.exists(resource_path):
                return resource_path
        except:
            pass
            
        return None

# الواجهة الرئيسية
if __name__ == "__main__":
    translator = SignLanguageTranslator()
    
    print("""\n
    ================================
    نظام تحويل النص إلى لغة الإشارة
    ================================
    """)

    while True:
        user_input = input("\nأدخل الجملة (أو 'خروج' للإنهاء): ").strip()
        
        if user_input.lower() == 'خروج':
            print("شكراً لاستخدامك النظام!")
            break
            
        if not translator.process_sentence(user_input):
            print("⚠ الرجاء إدخال جملة عربية صالحة!")