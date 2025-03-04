import json
import os
import cv2
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.morphology.analyzer import Analyzer
from fuzzywuzzy import process
import fasttext

# تحميل محلل الجذر من CAMeL Tools
analyzer = Analyzer.builtin("calima-msa")

# تحميل نموذج FastText المدرب على العربية (تأكد من تحميله مسبقًا)
fasttext_model = fasttext.load_model("cc.ar.300.bin")

# تحميل بيانات الفيديوهات من JSON
def load_dataset(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)

# دالة لتحليل الجملة وتقسيمها إلى كلمات
def tokenize_sentence(sentence):
    return simple_word_tokenize(sentence)  # تقسيم الجملة بطريقة صحيحة

# دالة لاستخراج الجذر الصحيح من الكلمة باستخدام CAMeL Tools
def get_root(word):
    analyses = analyzer.analyze(word)
    if analyses:
        return analyses[0].get("root", word)  # استخراج الجذر إذا كان موجودًا
    return word  # إرجاع الكلمة كما هي إذا لم يوجد تحليل

# تحليل الجملة بالكامل واستخراج الجذور
def preprocess_sentence(sentence):
    words = tokenize_sentence(sentence)  # تقسيم الجملة إلى كلمات
    roots = [get_root(word) for word in words]  # استخراج الجذور الصحيحة
    return roots

# البحث عن أقرب كلمة في قاعدة البيانات باستخدام Fuzzy Matching
def find_best_match(word, dataset):
    best_match, score = process.extractOne(word, dataset.keys())
    return best_match if score > 70 else None

# البحث عن كلمة ذات معنى مشابه باستخدام FastText
def find_similar_word(word, dataset):
    similar_words = fasttext_model.get_nearest_neighbors(word)
    for similarity, similar_word in similar_words:
        if similar_word in dataset:
            return similar_word
    return None

# البحث عن الفيديو باستخدام الجملة المدخلة
def find_video(sentence, dataset):
    words = preprocess_sentence(sentence)  # تحليل الجملة واستخراج الجذور
    for word in words:
        if word in dataset:
            return dataset[word]
        best_match = find_best_match(word, dataset)
        if best_match:
            return dataset[best_match]
        similar_word = find_similar_word(word, dataset)
        if similar_word:
            return dataset[similar_word]
    return None

# تشغيل الفيديو باستخدام OpenCV
def play_video(video_path):
    if not os.path.exists(video_path):
        print("❌ الفيديو غير موجود!")
        return
    
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Video", frame)
        
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# تحميل البيانات
dataset = load_dataset("metadata.json")

# تجربة البحث عن فيديو لجملة مدخلة
input_sentence = input("🔍 أدخل الجملة أو الكلمة: ").strip()
video_path = find_video(input_sentence, dataset)

if video_path:
    print(f"✅ تم العثور على الفيديو: {video_path}")
    play_video(video_path)
else:
    print("❌ لم يتم العثور على فيديو مطابق!")
