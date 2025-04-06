import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Parametreler
IMG_SIZE = (224, 224)  # Model giriş boyutu
BATCH_SIZE = 32
DATASET_PATH = "data"  # Veri setinin yolu

# Test verisini yükleme
test_datagen = ImageDataGenerator(rescale=1.0 / 255)  # Veriyi normalize etme
test_generator = test_datagen.flow_from_directory(
    DATASET_PATH + "/test",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # Karışık olmaması için
)

# Kullanılacak modeller ve isimleri
model_paths = {
    "VGG16": "vgg16_finetuned.h5",
    "MobileNetV2": "mobilenet_model.h5",
    "NASNetMobile": "nasnetmobile_finetuned.h5"
}

# Sonuçları saklamak için liste
results = []

# Her model için doğruluk hesaplama
for model_name, model_path in model_paths.items():
    print(f"✅ {model_name} modeli yükleniyor ve test ediliyor...")

    # Modeli yükle
    model = tf.keras.models.load_model(model_path)

    # Test verisi üzerinde tahmin yap
    test_loss, test_acc = model.evaluate(test_generator, verbose=1)

    # Sonuçları kaydet
    results.append([model_name, test_acc * 100])  # Yüzdelik olarak kaydet

# Sonuçları Pandas tablosu olarak gösterme
df_results = pd.DataFrame(results, columns=["Model Adı", "Test Doğruluk (%)"])
print("\n📊 Test Veri Seti Üzerindeki Model Performansları:\n")
print(df_results)

# Eğer tablonun grafik halinde gösterilmesini istiyorsan:
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.bar(df_results["Model Adı"], df_results["Test Doğruluk (%)"], color=['blue', 'green', 'red'])
plt.xlabel("Model Adı")
plt.ylabel("Test Doğruluk (%)")
plt.title("Modellerin Test Veri Setindeki Doğruluk Oranları")
plt.ylim(0, 100)
plt.show()
