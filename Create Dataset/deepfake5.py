import cv2
import os

# 얼굴 추출 함수 (기존에 얼굴이 있으면 224x224로 리사이즈)
def extract_face(image):
    # 이미지를 회색조로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 검출기 초기화
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        # MobileNetV2 모델에 맞게 크기 조정 (224x224)
        face_resized = cv2.resize(face, (224, 224))
        return face_resized
    return None

# 비디오 파일에서 프레임을 추출하고 저장하는 함수
def process_video(video_path, output_folder, label):
    # 비디오 열기
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    # 비디오 파일 이름을 고유한 파일명에 추가
    video_filename = os.path.basename(video_path).split('.')[0]  # 확장자 제외한 비디오 파일명
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 얼굴 추출
        face = extract_face(frame)
        if face is not None:
            # 고유한 파일명 생성: 비디오 파일명_프레임번호.jpg
            filename = os.path.join(output_folder, f"{video_filename}_{label}_{frame_count}.jpg")
            cv2.imwrite(filename, face)
            frame_count += 1

    cap.release()
    print(f"{frame_count}개의 이미지가 저장되었습니다.")

# 폴더 내 모든 비디오 파일을 처리하는 함수
def process_videos_in_folder(folder_path, output_folder, label):
    # 폴더 내 모든 파일을 확인
    for video_file in os.listdir(folder_path):
        # 비디오 파일인지 확인 (확장자가 .mp4, .avi 등)
        if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(folder_path, video_file)
            print(f"{video_path} 프레임 추출 시작...")
            process_video(video_path, output_folder, label)

# 훈련 데이터 및 검증 데이터 경로
train_fake_folder = 'data/processed/train/fake'
train_real_folder = 'data/processed/train/real'
val_fake_folder = 'data/processed/val/fake'
val_real_folder = 'data/processed/val/real'

# 폴더가 없으면 생성
os.makedirs(train_fake_folder, exist_ok=True)
os.makedirs(train_real_folder, exist_ok=True)
os.makedirs(val_fake_folder, exist_ok=True)
os.makedirs(val_real_folder, exist_ok=True)

# 영상이 저장된 폴더 경로 (딥페이크와 실제 영상 폴더 경로)
train_fake_videos_folder = r'D:\vscode\allvideo\fake_videos'
train_real_videos_folder = r'D:\vscode\allvideo\real_videos'
val_fake_videos_folder = r'D:\vscode\allvideo\fake_videos'
val_real_videos_folder = r'D:\vscode\allvideo\real_videos'

# 훈련 데이터 및 검증 데이터에서 얼굴 추출하여 저장
process_videos_in_folder(train_fake_videos_folder, train_fake_folder, 'fake')
process_videos_in_folder(train_real_videos_folder, train_real_folder, 'real')
process_videos_in_folder(val_fake_videos_folder, val_fake_folder, 'fake')
process_videos_in_folder(val_real_videos_folder, val_real_folder, 'real')
