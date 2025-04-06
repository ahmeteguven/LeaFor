import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Parametreler
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_PATH = "data"

# Test verisini yükleme
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    DATASET_PATH + "/test",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Model isimleri ve dosya yolları
model_paths = {
    "VGG16": "vgg16_finetuned.h5",
    "MobileNetV2": "mobilenet_model.h5",
    "NASNetMobile": "nasnetmobile_finetuned.h5"
}

# Gerçek etiketleri al (tüm test verisi için)
y_true = test_generator.classes

# Sonuçları saklamak için liste
results = []

# Her model için test etme
for model_name, model_path in model_paths.items():
    print(f"✅ {model_name} modeli yükleniyor ve test ediliyor...")

    # Modeli yükle
    model = tf.keras.models.load_model(model_path)

    # Test verisi üzerinde tahmin yap
    y_pred_probs = model.predict(test_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)  # En yüksek olasılığa sahip sınıfı al

    # Accuracy, Precision, Recall ve F1-score hesaplama
    acc = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average='macro') * 100
    recall = recall_score(y_true, y_pred, average='macro') * 100
    f1 = f1_score(y_true, y_pred, average='macro') * 100

    # Sonuçları listeye ekle
    results.append([model_name, acc, precision, recall, f1])

# Sonuçları Pandas tablosu olarak gösterme
df_results = pd.DataFrame(results, columns=["Model", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1-score (%)"])

print("\n📊 Model Performans Sonuçları:\n")
print(df_results)
