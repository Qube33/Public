import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 모델 정의 & 동결
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)
model = Model(base_model.input, outputs)

for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 데이터 증강
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.2),
])

# 전처리 함수
def preprocess(image, label, training=False):
    image = tf.cast(image, tf.float32) / 255.0
    if training:
        image = augment(image, training=True)
    return image, label

AUTOTUNE = tf.data.AUTOTUNE

# 데이터셋 준비 및 최적화
train_ds = (
    tf.keras.preprocessing.image_dataset_from_directory(
        'data/processed/train',
        batch_size=32,
        image_size=(224, 224),
        label_mode='binary'
    )
    .map(lambda x, y: preprocess(x, y, True), num_parallel_calls=AUTOTUNE)
    .shuffle(100)
    .prefetch(2)
)

val_ds = (
    tf.keras.preprocessing.image_dataset_from_directory(
        'data/processed/val',
        batch_size=32,
        image_size=(224, 224),
        label_mode='binary'
    )
    .map(lambda x, y: preprocess(x, y, False), num_parallel_calls=AUTOTUNE)
    .prefetch(2)
)

# 조기 종료 및 학습률 스케줄러 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# 모델 학습 (에포크 수를 10으로 설정)
history = model.fit(
    train_ds,
    epochs=10,  # 에포크 수를 10으로 늘리기
    validation_data=val_ds,
    verbose=1,
    callbacks=[early_stopping, lr_scheduler]
)

# 모델 저장
model.save('deepfake_detector5.h5')
