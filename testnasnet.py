import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Test veri seti yolu
TEST_DATASET_PATH = "data/test"
IMG_SIZE = (224, 224)
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

# Modeli yükleme (eğitilmiş NASNetMobile modelini yükle)
model = tf.keras.models.load_model('nasnetmobile_finetuned.h5')

# Modeli test etme
test_loss, test_accuracy = model.evaluate(test_generator)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Test sonucunun detaylı görüntülenmesi
print(f"Validation Loss: {test_loss}")
print(f"Validation Accuracy: {test_accuracy}")
