import json
import nltk
import os
from urllib.parse import quote
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import cv2  # Import OpenCV for video playback

# # تحميل الموارد اللازمة من NLTK
# nltk.download('punkt')

# قراءة ملف الجيسون الذي يحتوي على البيانات
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# قراءة ملف الـ Stop Words
def load_stop_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return set(file.read().splitlines())

# معالجة النص: تقطيع، إزالة الـ Stop Words، وتطبيق Stemming
def process_text(input_text, stop_words):
    tokens = word_tokenize(input_text.lower())  # تحويل الأحرف إلى صغيرة وتقطيع النص
    filtered_words = [word for word in tokens if word.isalnum() and word not in stop_words]
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    
    print("النص بعد التقطيع:", tokens)
    print("النص بعد إزالة الكلمات غير المهمة:", filtered_words)
    print("النص بعد Stemming:", stemmed_words)
    
    return stemmed_words  # إرجاع الكلمات بعد Stemming فقط

# فتح الفيديو باستخدام OpenCV
def play_video_with_opencv(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Read and display video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        cv2.imshow("Video", frame)
        
        # Press 'q' to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# البحث عن الكلمات في ملف الجيسون وفتح الفيديوهات
def find_videos(words, data):
    videos = []
    for word in words:
        if word in data:  # البحث عن الكلمة كمفتاح في ملف JSON
            video_path = data[word]
            videos.append(video_path)
            play_video_with_opencv(video_path)  # فتح الفيديو باستخدام OpenCV
    return videos

if __name__ == "__main__":
    # تحميل البيانات
    json_data = load_json("D:/Sanad/Sanad-AI/enhanced_metadata.json")  # تحديث مسار ملف JSON
    stop_words = load_stop_words("stop words.txt")  # استخدم ملفك الفعلي
    
    # إدخال النص من المستخدم
    user_input = input("أدخل النص: ")
    stemmed_words = process_text(user_input, stop_words)  # الحصول على الكلمات بعد Stemming
    
    # البحث عن الفيديوهات
    videos = find_videos(stemmed_words, json_data)
    
    # عرض النتائج
    if videos:
        print("تم فتح الفيديوهات التالية:")
        for video in videos:
            print(video)
    else:
        print("لا توجد فيديوهات مرتبطة بالكلمات المدخلة.")