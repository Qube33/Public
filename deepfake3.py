import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 저장된 모델 로드
model1 = load_model('deepfake_detector5.h5')

# 얼굴 추출 함수
def extract_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        return cv2.resize(face, (224, 224))
    return None

# 영상 파일 딥페이크 판별 함수 (재생 없이 처리)
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    fake_count = 0
    real_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face = extract_face(frame)
        if face is not None:
            face_input = np.expand_dims(face / 255.0, axis=0)
            pred = model1.predict(face_input, verbose=0)[0][0]
            if pred < 0.5:
                fake_count += 1
            else:
                real_count += 1

        frame_count += 1
        # 진행 상태 출력
        if total_frames > 0:
            pct = frame_count / total_frames * 100
            print(f"\rProgress: {pct:5.1f}% ({frame_count}/{total_frames})", end="")
        else:
            print(f"\rProcessed frames: {frame_count}", end="")

    cap.release()
    print("\n\n--- 분석 완료 ---")
    print(f"Total frames processed: {frame_count}")
    print(f"Fake frames: {fake_count}")
    print(f"Real frames: {real_count}")

    if frame_count > 0:
        fake_pct = fake_count / frame_count * 100
        real_pct = real_count / frame_count * 100
        print(f"딥페이크 비율: {fake_pct:.2f}%")
        print(f"실제영상 비율: {real_pct:.2f}%")
        if fake_pct > real_pct:
            print("최종 결과: 딥페이크 영상입니다.")
        else:
            print("최종 결과: 실제 영상입니다.")
    else:
        print("프레임을 하나도 처리하지 못했습니다. 경로를 확인하세요.")

if __name__ == "__main__":
    video_path = input("검사할 영상 경로를 입력하세요: ")
    predict_video(video_path)