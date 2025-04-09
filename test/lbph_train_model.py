import cv2
print(cv2.__version__)
import numpy as np
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Đường dẫn đến thư mục chứa ảnh
dataset_path = "images/"

# Tạo bộ nhận diện LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load bộ phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_images_and_labels(dataset_path):
    face_samples = []
    ids = []

    # Duyệt qua từng thư mục (từng sinh viên)
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        
        if not os.path.isdir(person_folder):
            continue  # Bỏ qua nếu không phải thư mục
        
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)

            # Đọc ảnh
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"⚠️ Lỗi đọc ảnh: {image_name}, bỏ qua.")
                continue  
            
            # Tách ID từ tên file (mssv_x.jpg -> lấy mssv)
            try:
                student_id = int(image_name.split('_')[0])
            except ValueError:
                print(f"⚠️ Lỗi: Không thể tách ID từ {image_name}, bỏ qua.")
                continue

            # Nhận diện khuôn mặt trên ảnh
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                print(f"⚠️ Không phát hiện khuôn mặt trong {image_name}, bỏ qua.")
                continue

            for (x, y, w, h) in faces:
                face_samples.append(img[y:y+h, x:x+w])
                ids.append(student_id)

    return face_samples, np.array(ids)

print("📸 Đang xử lý dữ liệu...")
faces, ids = get_images_and_labels(dataset_path)

if len(faces) == 0:
    print("❌ Không có dữ liệu khuôn mặt hợp lệ để huấn luyện. Kiểm tra lại tập dữ liệu.")
    exit()

# Huấn luyện mô hình
print("🤖 Đang huấn luyện mô hình LBPH...")
recognizer.train(faces, ids)

# Lưu mô hình vào file
recognizer.save("lbph_trainer.yml")
print("✅ Huấn luyện hoàn tất! Mô hình đã được lưu vào trainer.yml")
