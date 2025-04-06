import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Parametreler
IMG_SIZE = (224, 224)  # Veri boyutu
BATCH_SIZE = 32
DATASET_PATH = "data"  # Veri setinizin yolu

# Veri artırma (Data Augmentation) ve Validation Split
train_datagen = ImageDataGenerator(
    rescale=1.0/255,  # Veriyi normalize etme
    rotation_range=30,  # Dönme
    width_shift_range=0.2,  # Yatay kaydırma
    height_shift_range=0.2,  # Dikey kaydırma
    shear_range=0.2,  # Kayma
    zoom_range=0.2,  # Zoom
    horizontal_flip=True,  # Yatay çevirme
    fill_mode='nearest',  # Boş alanları en yakın değerle doldur
    validation_split=0.2  # Eğitim verisinin %20'sini doğrulama için kullan
)

# Eğitim verisini yükleme
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH + "/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',  # Eğitim verisi
    shuffle=True
)

# Doğrulama verisini yükleme
validation_generator = train_datagen.flow_from_directory(
    DATASET_PATH + "/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',  # Doğrulama verisi
    shuffle=False
)

# VGG16 Modelini Yükleyip Son Katmanları Ekleyelim
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# İlk başta temel katmanları donduralım (feature extraction)
base_model.trainable = False

# Modelin son katmanlarını ekleyelim
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),  # 128 nöronlu katman
    layers.Dropout(0.3),  # Dropout ekleyerek overfitting'i engellemek
    layers.Dense(train_generator.num_classes, activation='softmax')  # Çıkış katmanı
])

# Modeli derleme (daha düşük bir öğrenme oranı ile)
model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# EarlyStopping ile eğitimi erken sonlandırma
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Modeli eğitelim (Feature Extraction aşaması)
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[early_stopping]
)

# Fine-Tuning işlemi (Son katmanları açıyoruz)
for layer in base_model.layers[-50:]:  # Son 50 katmanı eğitiyoruz
    layer.trainable = True

# Daha düşük bir öğrenme oranı ile yeniden derleme
model.compile(optimizer=optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fine-tuning ile eğitme
history_finetune = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[early_stopping]
)

# Eğitim sonrası başarı grafiği çizme
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()

# Eğitim ve doğrulama kaybı
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kaybı')
plt.legend()
plt.show()

# Modeli Kaydetme
model.save('vgg16_finetuned.h5')
