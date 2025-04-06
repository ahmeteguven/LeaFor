import numpy as np
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

# 2. Eğitimli Modelleri Yükle
mobilenet_model = load_model('mobilenet_model.h5')
efficientnet_model = load_model('nasnetmobile_finetuned.h5')

# 3. Tahmin Yapmak İçin Ensemble Modeli Oluşturma (Soft Voting)
mobilenet_preds = mobilenet_model.predict(test_generator, verbose=1)
efficientnet_preds = efficientnet_model.predict(test_generator, verbose=1)

# Soft Voting: Modellerin tahminlerinin ortalamasını alıyoruz
ensemble_preds = (mobilenet_preds + efficientnet_preds) / 2

# 4. En Yüksek Olasılığa Sahip Sınıfı Seç
final_predictions = np.argmax(ensemble_preds, axis=1)

# 5. Gerçek Etiketleri Al
true_labels = test_generator.classes

# Label Encoding işlemi (İsimleri almak için)
label_map = {v: k for k, v in test_generator.class_indices.items()}
final_predictions_labels = [label_map[i] for i in final_predictions]
true_labels_names = [label_map[i] for i in true_labels]

# 6. Sonuçları Yazdır
print("Classification Report:\n", classification_report(true_labels_names, final_predictions_labels))

# 7. Ensemble Modelinin Doğruluğunu Hesapla
from sklearn.metrics import accuracy_score
ensemble_accuracy = accuracy_score(true_labels, final_predictions)
print("Ensemble Model Accuracy: {:.4f}".format(ensemble_accuracy))

# 8. Ensemble Modelini Kaydetme
# Ensemble modelinin giriş katmanını ve çıkış katmanını oluşturuyoruz.
inputs = layers.Input(shape=(224, 224, 3))
mobilenet_out = mobilenet_model(inputs)
efficientnet_out = efficientnet_model(inputs)

# Katmanları birleştiriyoruz (Soft Voting)
ensemble_out = layers.Average()([mobilenet_out, efficientnet_out])

# Yeni Ensemble Modeli
ensemble_model = models.Model(inputs=inputs, outputs=ensemble_out)

# Ensemble Modelini Kaydet
ensemble_model.save('ensemble_model.h5')
