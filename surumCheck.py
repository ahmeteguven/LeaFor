import subprocess

# Uyumlu olduğu bilinen TensorFlow ve Keras versiyonları
tensorflow_version = "2.15.0"
keras_version = "2.15.0"

print(f"TensorFlow {tensorflow_version} ve Keras {keras_version} kuruluyor...")

# Önce eski versiyonları kaldır
subprocess.run(["pip", "uninstall", "-y", "tensorflow", "keras"])

# Yeni versiyonları yükle
subprocess.run(["pip", "install", f"tensorflow=={tensorflow_version}"])
subprocess.run(["pip", "install", f"keras=={keras_version}"])

print("Kurulum tamamlandı!")