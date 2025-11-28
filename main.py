import json
import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from farasa.segmenter import FarasaSegmenter

app = FastAPI(title="Sanad Sign Language Translator API")

# CORS for frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranslateRequest(BaseModel):
    sentence: str

class SegmentRequest(BaseModel):
    text: str

class SignLanguageTranslator:

    def __init__(self):
        self.signs_db = json.load(open("version2.json", "r", encoding="utf-8"))
        self.stop_words = set(open("stop words.txt", "r", encoding="utf-8").read().splitlines())
        try:
            self.farasa = FarasaSegmenter(interactive=True)
        except Exception as e:
            print("Farasa Error:", e)
            self.farasa = None

        self.setup_synonym_map()

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

        if not sentence.strip():
            return {"status": "error", "message": "الجملة غير صالحة"}

        try:
            result = self.farasa.segment(sentence)
            tokens = []
            for segment in result.split():
                tokens.extend(segment.split('+'))
        except Exception as e:
            return {"status": "error", "message": f"فشل تقطيع Farasa: {str(e)}"}
        
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
        matches_list = []
        seen = set()
        for word in filtered:
            main_word, sign_data = self.find_sign_match(word)
            if sign_data and main_word and main_word not in seen:
                matches_list.append({
                    "word": main_word,
                    "input_word": word,
                    "video_url": sign_data.get("video_path", "VIDEO_PLACEHOLDER")
                })
                seen.add(main_word)
        if matches_list:
            return {
                "status": "success",
                "message": f"الكلمات المعثور عليها: {', '.join([m['word'] for m in matches_list])}",
                "matches": matches_list
            }
        return {
            "status": "error",
            "message": f"لا توجد كلمات مدعومة في الجملة: {sentence}"
        }

@app.post("/translate")
async def translate_sentence(request: TranslateRequest):
    global translator
    result = translator.process_sentence(request.sentence)
    return JSONResponse(content=result)

@app.post("/segment")
async def segment_text(request: SegmentRequest):
    global translator
    if not translator.farasa:
        raise HTTPException(status_code=500, detail="Farasa غير متاح على هذا السيرفر")
    try:
        result = translator.farasa.segment(request.text)
        segments = []
        for segment in result.split():
            segments.extend(segment.split('+'))
        return {"status": "success", "segments": segments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_index():
    return FileResponse("index.html")

@app.on_event("startup")
async def startup_event():
    global translator
    translator = SignLanguageTranslator()

if __name__ == "__main__":
    import uvicorn
    # port = int(os.getenv("PORT", 8000))
    # uvicorn.run(app, host="0.0.0.0", port=port)
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)