import Augmentor

# Veri kumesi yolu
data_directory = '/content/images'
output_directory='/content/augmented_images'

# Augmentor nesnesi olusturun
p = Augmentor.Pipeline(data_directory,output_directory)

# Augmentation islemlerini tanimlayin
p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.flip_left_right(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.random_contrast(probability=0.5, min_factor=0.8, max_factor=1.2)

# Augmentation islemlerini uygulayin ve yeni veri kumesini kaydedin
p.sample(1000)  # Ornek olarak 1000 yeni goruntu olusturun