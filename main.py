
import json
import os
import re
from fastapi import FastAPI
from pydantic import BaseModel
from farasa.segmenter import FarasaSegmenter

app = FastAPI(title="Sanad Sign Language Translator API")

class TranslateRequest(BaseModel):
    sentence: str

class SignLanguageTranslator:
    signs_db = None
    farasa = None

    def __init__(self):
        if SignLanguageTranslator.signs_db is None:
            SignLanguageTranslator.signs_db = self.load_dictionary()
        self.signs_db = SignLanguageTranslator.signs_db

        if SignLanguageTranslator.farasa is None:
            try:
                SignLanguageTranslator.farasa = FarasaSegmenter(interactive=True)
            except Exception as e:
                print(f"⚠ فشل تهيئة Farasa: {str(e)}")
                SignLanguageTranslator.farasa = None
        self.farasa = SignLanguageTranslator.farasa

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
                    matches[main_word] = {
                        "input_word": word,
                        "video_url": sign_data.get("video_path")
                    }

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

@app.post("/translate")
async def translate_sentence(request: TranslateRequest):
    translator = SignLanguageTranslator()
    result = translator.process_sentence(request.sentence)
    return result

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)