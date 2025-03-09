import json
import re
import cv2
import os
from farasa.segmenter import FarasaSegmenter

# تحميل ملف JSON
json_path = "D:\Sanad\Sanad-AI\metadata.json"   

if not os.path.exists(json_path):
    print(f"❌ الملف غير موجود في: {json_path}")
    exit()

with open(json_path, "r", encoding="utf-8") as file:
    word_to_video = json.load(file)

#  تقسيم الكلمات
farasa_segmenter = FarasaSegmenter(interactive=True)

def preprocess_text(text):
    """تنظيف النص وإزالة التشكيل والأحرف الخاصة."""
    text = text.strip()  # إزالة المسافات 
    text = re.sub(r'[^\w\s]', '', text)  # إزالة التشكيل  
    return text

def search_videos(sentence):
    """البحث عن الفيديوهات المتعلقة بالجملة."""
    sentence = preprocess_text(sentence)  # تنظيف الجملة
    if sentence in word_to_video:
        return [word_to_video[sentence]]  # البحث عن الجملة كاملة
    
    words = farasa_segmenter.segment(sentence).split()  # تقسيم الجملة
    videos = [word_to_video[word] for word in words if word in word_to_video]  # البحث عن الكلمات
    return videos

def play_video(video_path):
    """تشغيل الفيديو باستخدام OpenCV."""
    if not os.path.exists(video_path):
        print(f"❌ الفيديو غير موجود: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Video", frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

sentence = "اخوان"
videos = search_videos(sentence)

if videos:
    print("✅ الفيديوهات المرتبطة:", videos)
    for video in videos:
        play_video(video)  
else:
    print("❌ لا يوجد فيديو مرتبط بهذه الكلمات.")


