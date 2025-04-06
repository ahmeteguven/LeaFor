import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet, EfficientNetB0
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping  # EarlyStopping import

# 1. Veri Hazırlığı (Data Augmentation ve Validation Split)
DATASET_PATH = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Veri hazırlık işlemi: Resimlerin normalize edilmesi ve validation split
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2  # Eğitim verisinin %20'sini doğrulama için kullanacağız
)

# Eğitim verisi
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH + "/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'  # Eğitim verisi
)

# Doğrulama verisi
validation_generator = train_datagen.flow_from_directory(
    DATASET_PATH + "/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'  # Doğrulama verisi
)

# 2. MobileNet Modelini Oluşturma ve Eğitme
mobilenet_base = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
mobilenet_base.trainable = False  # Önceden eğitilmiş ağırlıkları değiştirmiyoruz

mobilenet_model = models.Sequential([
    mobilenet_base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

mobilenet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# EarlyStopping callback ekleniyor
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# MobileNet eğitimi (10 epoch)
mobilenet_model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[early_stopping]  # EarlyStopping'i buraya ekliyoruz
)

# MobileNet modelini kaydediyoruz
mobilenet_model.save('mobilenet_model.h5')

# 3. EfficientNet Modelini Oluşturma ve Eğitme
efficientnet_base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
efficientnet_base.trainable = False  # Başlangıçta dondurulmuş

# Burada EfficientNet katmanlarını eğitilebilir hale getiriyoruz
for layer in efficientnet_base.layers:
    layer.trainable = True  # Tüm katmanları eğitilebilir hale getiriyoruz

efficientnet_model = models.Sequential([
    efficientnet_base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

efficientnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# EfficientNet eğitimi (20 epoch) - EarlyStopping ekleniyor
efficientnet_model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=[early_stopping]  # EarlyStopping'i buraya da ekliyoruz
)

# EfficientNet modelini kaydediyoruz
efficientnet_model.save('efficientnet_model.h5')
