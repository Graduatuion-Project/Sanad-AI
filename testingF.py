import json
import cv2
import os
import re
import tkinter as tk
from tkinter import messagebox
import unittest
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from farasa.segmenter import FarasaSegmenter

app = FastAPI(title="Sanad Sign Language Translator API")

class TranslateRequest(BaseModel):
    sentence: str

class SignLanguageTranslator:
    
    def __init__(self):
        try:
            self.farasa = FarasaSegmenter(interactive=True)
        except Exception as e:
            print(f"⚠ فشل تهيئة Farasa: {str(e)}")
            self.farasa = None
        self.signs_db = self.load_dictionary()
        self.stop_words = self.get_stop_words()
        self.setup_synonym_map()

    def load_dictionary(self):
        try:
            with open("enhanced_metadata.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                if not data:
                    print("⚠ ملف البيانات فارغ")
                    return {}
                return data
        except FileNotFoundError:
            print("⚠ ملف enhanced_metadata.json غير موجود")
            return {}
        except json.JSONDecodeError:
            print("⚠ ملف enhanced_metadata.json تالف")
            return {}

    def get_stop_words(self):
        return {
            "في", "من", "إلى", "على", "عن", "أن", "إن", "كان", "كانت",
            "أو", "و", "ثم", "قد", "كل", "كما"
        }

    def clean_input(self, sentence):
        sentence = re.sub(r'[^\u0600-\u06FF\s]', '', sentence)
        return sentence.strip()

    def normalize_word(self, word):
        if not word:
            return ""
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
            self.signs_db[main_word]["norm_main"] = norm_main
            for synonym in data.get("synonyms", []):
                norm_syn = self.normalize_word(synonym)
                if norm_syn:
                    self.synonym_to_main[norm_syn] = main_word

    def find_sign_match(self, word):
        norm_word = self.normalize_word(word)
        if not norm_word:
            return None, None
        
        if norm_word in self.synonym_to_main:
            main_word = self.synonym_to_main[norm_word]
            return main_word, self.signs_db[main_word]

        if '+' in word:
            combined = word.replace('+', '')
            norm_combined = self.normalize_word(combined)
            if norm_combined in self.synonym_to_main:
                main_word = self.synonym_to_main[norm_combined]
                return main_word, self.signs_db[main_word]

        return None, None

    def process_sentence(self, sentence):
        sentence = self.clean_input(sentence)
        if not sentence.strip() or not self.farasa:
            return {"status": "error", "message": "الجملة غير صالحة أو Farasa غير متاح"}

        tokens = self.farasa.segment(sentence).split()
        fixed_tokens = []
        i = 0
        while i < len(tokens):
            current_token = tokens[i]
            if '+' in current_token and i + 1 < len(tokens):
                combined = current_token.replace('+', '') + tokens[i + 1]
                norm_combined = self.normalize_word(combined)
                if norm_combined in self.synonym_to_main or norm_combined in [d["norm_main"] for d in self.signs_db.values()]:
                    fixed_tokens.append(combined)
                    i += 2
                else:
                    fixed_tokens.append(current_token)
                    i += 1
            else:
                fixed_tokens.append(current_token)
                i += 1

        final_tokens = []
        i = 0
        while i < len(fixed_tokens):
            current_token = fixed_tokens[i]
            if i + 1 < len(fixed_tokens):
                combined = current_token + fixed_tokens[i + 1]
                norm_combined = self.normalize_word(combined)
                if norm_combined in self.synonym_to_main or norm_combined in [d["norm_main"] for d in self.signs_db.values()]:
                    final_tokens.append(combined)
                    i += 2
                else:
                    final_tokens.append(current_token)
                    i += 1
            else:
                final_tokens.append(current_token)
                i += 1

        filtered = [w for w in final_tokens if self.normalize_word(w) not in self.stop_words]
        matches = {}
        for word in filtered:
            main_word, sign_data = self.find_sign_match(word)
            if sign_data and main_word:
                if main_word not in matches:
                    matches[main_word] = {"input_word": word, "data": sign_data}

        if matches:
            return {
                "status": "success",
                "message": f"الكلمات المعثور عليها: {', '.join(matches.keys())}",
                "matches": matches
            }
        return {
            "status": "error",
            "message": f"لا توجد كلمات مدعومة في الجملة: {sentence}"
        }

    def play_videos(self, matches):
        cv2.namedWindow("Sign Language", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Sign Language", 800, 600)

        for main_word, info in matches.items():
            input_word = info["input_word"]
            main_video_path = info["data"].get("video_path")
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
                    cv2.imshow("Sign Language", frame)
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    elif key == ord('p'):
                        while cv2.waitKey(0) != ord('p'):
                            pass
                cap.release()
            except Exception as e:
                print(f"⚠ خطأ أثناء التشغيل: {str(e)}")
        cv2.destroyAllWindows()
        cv2.waitKey(500)

    def check_video_exists(self, path):
        if not path:
            return None
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

@app.post("/translate")
async def translate_sentence(request: TranslateRequest):

    translator = SignLanguageTranslator()
    result = translator.process_sentence(request.sentence)
    return result

class SignLanguageGUI:
    def __init__(self, api_url="http://localhost:8000"):
        """Initialize the GUI with API URL."""
        self.api_url = api_url
        self.translator = SignLanguageTranslator()
        self.root = tk.Tk()
        self.root.title("نظام سند - لغة الإشارة")
        self.root.geometry("600x400")
        
        self.label = tk.Label(self.root, text="أدخل جملة باللغة العربية:", font=("Arial", 14))
        self.label.pack(pady=10)
        
        self.entry = tk.Entry(self.root, width=50, font=("Arial", 12))
        self.entry.pack(pady=10)
        
        self.translate_button = tk.Button(self.root, text="ترجم", command=self.translate, font=("Arial", 12))
        self.translate_button.pack(pady=10)
        
        self.exit_button = tk.Button(self.root, text="خروج", command=self.root.quit, font=("Arial", 12))
        self.exit_button.pack(pady=10)

    def translate(self):
        sentence = self.entry.get()
        try:
            response = requests.post(f"{self.api_url}/translate", json={"sentence": sentence})
            response.raise_for_status()
            result = response.json()
            
            if result["status"] == "success":
                messagebox.showinfo("نجاح", result["message"])
                self.translator.play_videos(result["matches"])
            else:
                messagebox.showerror("خطأ", result["message"])
        except requests.RequestException as e:
            messagebox.showerror("خطأ", f"فشل الاتصال بالـ API: {str(e)}")

    def run(self):
        self.root.mainloop()

class TestSignLanguageTranslator(unittest.TestCase):
    def setUp(self):
        self.translator = SignLanguageTranslator()

    def test_normalize_word(self):
        self.assertEqual(self.translator.normalize_word("كَتَبَ"), "كتب")
        self.assertEqual(self.translator.normalize_word("أكتب"), "اكتب")
        self.assertEqual(self.translator.normalize_word(""), "")
    
    def test_clean_input(self):
        self.assertEqual(self.translator.clean_input("كتب 123 book"), "كتب")
        self.assertEqual(self.translator.clean_input("مرحبا!"), "مرحبا")
    
    def test_find_sign_match(self):
        self.assertIsNotNone(self.translator.find_sign_match("شكر+ا")[0])
        self.assertIsNotNone(self.translator.find_sign_match("شكرا")[0])

if __name__ == "__main__":
    import uvicorn
    import threading
    def start_api():
        uvicorn.run(app, host="0.0.0.0", port=8000)
    
    api_thread = threading.Thread(target=start_api, daemon=True)
    api_thread.start()
    
    gui = SignLanguageGUI()
    gui.run()