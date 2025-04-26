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
        self.displayed_videos = set()  # لتتبع الفيديوهات المعروضة

    def load_dictionary(self):
        with open("enhanced_metadata.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def get_stop_words(self):
        return {
            "في", "من", "إلى", "على", "عن", "أن", "إن", "كان", "كانت",
            "أو", "و", "ثم", "قد", "كل", "كما"
        }

    def normalize_word(self, word):
        if not word:
            return word
        word = re.sub(r'[\u064b-\u065f]', '', word)
        word = re.sub(r'[أإآ]', 'ا', word)
        word = word.replace('ة', 'ه')
        word = word.replace('ى', 'ي')
        word = word.strip('ًٌٍَُِ')
        return word.strip()

    def setup_synonym_map(self):
        self.synonym_to_main = {}
        for main_word, data in self.signs_db.items():
            norm_main = self.normalize_word(main_word)
            self.synonym_to_main[norm_main] = main_word
            for synonym in data.get("synonyms", []):
                norm_syn = self.normalize_word(synonym)
                if norm_syn:
                    self.synonym_to_main[norm_syn] = main_word

    def find_sign_match(self, word):
        norm_word = self.normalize_word(word)
        
        # البحث في الكلمات الرئيسية أولاً
        for main_word in self.signs_db:
            if self.normalize_word(main_word) == norm_word:
                return main_word, self.signs_db[main_word]
        
        # ثم البحث في المرادفات
        if norm_word in self.synonym_to_main:
            main_word = self.synonym_to_main[norm_word]
            return main_word, self.signs_db[main_word]
        
        return None, None

    def process_sentence(self, sentence):
        if not sentence.strip():
            return False

        self.displayed_videos.clear()  # مسح الفيديوهات المعروضة سابقاً

        tokens = self.farasa.segment(sentence).split()

        fixed_tokens = []
        i = 0
        while i < len(tokens):
            current_token = tokens[i]
            fixed_tokens.append(current_token)

            if '+' in current_token and i + 1 < len(tokens):
                combined = current_token.replace('+', '') + tokens[i + 1]
                norm_combined = self.normalize_word(combined)
                if norm_combined in self.synonym_to_main or norm_combined in [self.normalize_word(w) for w in self.signs_db]:
                    fixed_tokens[-1] = combined
                    i += 2
                else:
                    i += 1
            else:
                i += 1

        filtered = [w for w in fixed_tokens if self.normalize_word(w) not in self.stop_words]

        # تخزين الكلمات الفريدة فقط
        unique_words = set()
        for word in filtered:
            norm_word = self.normalize_word(word)
            if norm_word not in unique_words:
                unique_words.add(norm_word)

        # البحث عن مطابقات للكلمات الفريدة فقط
        matches = {}
        for word in unique_words:
            main_word, sign_data = self.find_sign_match(word)
            if sign_data and main_word not in matches:
                matches[main_word] = {
                    "input_word": word,
                    "data": sign_data
                }

        if matches:
            self.play_videos(matches)
        else:
            print("\n⚠ لا توجد كلمات متطابقة لتشغيل الفيديوهات")
        return True

    def play_videos(self, matches):
        for main_word, info in matches.items():
            input_word = info["input_word"]
            main_video_path = info["data"].get("video_path")
            
            # تجنب عرض نفس الفيديو أكثر من مرة
            if main_video_path in self.displayed_videos:
                continue
                
            self.displayed_videos.add(main_video_path)
            
            resolved_path = self.check_video_exists(main_video_path)
            
            if not resolved_path:
                print(f"\n⚠ ملف الفيديو غير موجود: {main_video_path}")
                continue

            print(f"\n▶ تشغيل فيديو للكلمة: {input_word} (Path: {resolved_path})")

            try:
                cap = cv2.VideoCapture(resolved_path)
                if not cap.isOpened():
                    print(f"⚠ فشل فتح ملف الفيديو: {resolved_path}")
                    continue

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    cv2.imshow(f"إشارة: {input_word}", frame)
                    if cv2.waitKey(30) & 0xFF == ord('q'):
                        break

            except Exception as e:
                print(f"⚠ خطأ أثناء التشغيل: {str(e)}")
            finally:
                if 'cap' in locals():
                    cap.release()
                cv2.destroyAllWindows()

    def check_video_exists(self, path):
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

if __name__ == "__main__":
    translator = SignLanguageTranslator()

    print("""
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