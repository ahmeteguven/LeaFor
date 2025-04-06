import numpy as np
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Veri seti yolu
DATASET_PATH = "data"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 1. Veri Hazırlığı (Data Augmentation ve Validation Split)
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # %20 doğrulama verisi
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH + "/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_PATH + "/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# 2. ResNet50 Modelini Kurma (pretrained ağırlıklarla)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # İlk aşamada sadece son katmanlar eğitilecek

# 3. Yeni Modelin Eklenmesi
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='swish'),  # 256 yerine 128 nöron kullandık
    layers.Dropout(0.3),  # Overfitting'i önlemek için dropout ekledik
    layers.Dense(train_generator.num_classes, activation='softmax')
])

# Modeli derleme (düşük öğrenme oranı ile Adam kullanılıyor)
model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# EarlyStopping kullanımı
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 4. Modeli Eğitme
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

# 5. Fine-Tuning (Son katmanları açma)
for layer in base_model.layers[-50:]:
    layer.trainable = True

# Daha düşük bir öğrenme oranı ile yeniden derleme
model.compile(optimizer=optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Son aşama eğitimi
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

# Modeli kaydetme
model.save('resnet_model.h5')  # ResNet modelini kaydediyoruz
