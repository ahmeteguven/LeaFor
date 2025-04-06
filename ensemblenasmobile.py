import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from tensorflow.keras import layers, models

# 1. Veri Hazırlığı (Data Augmentation ve Validation Split)
DATASET_PATH = "data"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Veri hazırlık işlemi: Resimlerin normalize edilmesi ve validation split
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Test verisi
test_generator = test_datagen.flow_from_directory(
    DATASET_PATH + "/test",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Veri artırma (Data Augmentation) işlemi
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH + "/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

# 2. Eğitimli Modelleri Yükle
mobilenet_model = load_model('mobilenet_model.h5')
efficientnet_model = load_model('nasnetmobile_finetuned.h5')

# 3. Modellerin Eğitim Sonuçlarını Al (Doğruluk ve Kayıp Değerleri)
mobilenet_history = mobilenet_model.evaluate(test_generator, verbose=1)
efficientnet_history = efficientnet_model.evaluate(test_generator, verbose=1)

# 4. Tahmin Yapmak İçin Ensemble Modeli Oluşturma (Soft Voting)
mobilenet_preds = mobilenet_model.predict(test_generator, verbose=1)
efficientnet_preds = efficientnet_model.predict(test_generator, verbose=1)

# Weighted Soft Voting: Modellerin tahminlerinin ağırlıklı ortalamasını alıyoruz
weights = [0.5, 0.5]  # MobilNet ve EfficientNet'e eşit ağırlık veriyoruz
ensemble_preds = (weights[0] * mobilenet_preds + weights[1] * efficientnet_preds)

# 5. En Yüksek Olasılığa Sahip Sınıfı Seç
final_predictions = np.argmax(ensemble_preds, axis=1)

# 6. Gerçek Etiketleri Al
true_labels = test_generator.classes

# Label Encoding işlemi (İsimleri almak için)
label_map = {v: k for k, v in test_generator.class_indices.items()}
final_predictions_labels = [label_map[i] for i in final_predictions]
true_labels_names = [label_map[i] for i in true_labels]

# 7. Sonuçları Yazdır
print("Classification Report:\n", classification_report(true_labels_names, final_predictions_labels))

# 8. Ensemble Modelinin Doğruluğunu Hesapla
from sklearn.metrics import accuracy_score
ensemble_accuracy = accuracy_score(true_labels, final_predictions)
ensemble_loss = np.mean(ensemble_preds)  # Ensemble için kayıp (tahminlerin ortalaması)

print("Ensemble Model Accuracy: {:.4f}".format(ensemble_accuracy))

# 9. Eğitim Sonuçlarını (Accuracy ve Loss) Sütun Grafiği Olarak Çizme
model_names = ['MobileNetV2', 'NASNetMobile', 'Ensemble']
accuracies = [mobilenet_history[1], efficientnet_history[1], ensemble_accuracy]
losses = [mobilenet_history[0], efficientnet_history[0], ensemble_loss]

# Sütun Grafiği
x = np.arange(len(model_names))  # Model isimlerinin konumları
width = 0.35  # Sütunların genişliği

fig, ax = plt.subplots(figsize=(10, 6))

# Accuracy ve Loss için iki sütun grafiği
rects1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='g')
rects2 = ax.bar(x + width/2, losses, width, label='Loss', color='r')

# Başlık ve etiketler
ax.set_ylabel('Scores')
ax.set_title('Model Performance Comparison (Accuracy and Loss)')
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.legend()

# Grafiği göster
plt.tight_layout()
plt.show()
