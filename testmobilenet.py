import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Test veri seti yolu
TEST_DATASET_PATH = "data/test"
IMG_SIZE = (224, 224)  # Aynı boyutları kullanman önemli
BATCH_SIZE = 32

# Test veri hazırlığı
test_datagen = ImageDataGenerator(rescale=1.0/255)

test_generator = test_datagen.flow_from_directory(
    TEST_DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # Test verisi için shuffle=False yapıyoruz
)

# MobileNetV2 modelini yükleme (eğitilmiş model)
model = tf.keras.models.load_model('mobilenet_model.h5')

# Modeli test etme
test_loss, test_accuracy = model.evaluate(test_generator)

# Sonuçları yazdırma
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Validation sonuçlarını da aynı şekilde gösterebilirsin (gerçekten validasyon verisi ise)
print(f"Validation Loss: {test_loss}")
print(f"Validation Accuracy: {test_accuracy}")
