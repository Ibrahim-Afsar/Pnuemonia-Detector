import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

train_dir = 'D:\Machine learning Datasets\Lungs Pneumonia dataset/archive/xray_dataset_covid19/train'
test_dir = 'D:\Machine learning Datasets\Lungs Pneumonia dataset/archive/xray_dataset_covid19/test'

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='binary')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='binary')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_generator, steps_per_epoch=len(train_generator), epochs=25, validation_data=test_generator, validation_steps=len(test_generator))

test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
print('\nTest accuracy:', test_acc)

sample_image_path = 'D:\Machine learning Datasets\Lungs Pneumonia dataset/archive/xray_dataset_covid19/test\PNEUMONIA/ryct.2020200034.fig5-day7.jpeg'
sample_image = tf.keras.preprocessing.image.load_img(sample_image_path, target_size=(224, 224))
sample_image = tf.keras.preprocessing.image.img_to_array(sample_image)
sample_image = sample_image / 255.0
sample_image = tf.expand_dims(sample_image, 0)
prediction = model.predict(sample_image)
print('Prediction:', prediction)

plt.imshow(sample_image[0])
plt.title('PNEUMONIA' if prediction[0][0] > 0.5 else 'NORMAL')
plt.show()

sample_image_path = 'D:\Machine learning Datasets\Lungs Pneumonia dataset/archive/xray_dataset_covid19/test/NORMAL/NORMAL2-IM-0059-0001.jpeg'
sample_image = tf.keras.preprocessing.image.load_img(sample_image_path, target_size=(224, 224))
sample_image = tf.keras.preprocessing.image.img_to_array(sample_image)
sample_image = sample_image / 255.0
sample_image = tf.expand_dims(sample_image, 0)
prediction = model.predict(sample_image)
print('Prediction:', prediction)

plt.imshow(sample_image[0])
plt.title('PNEUMONIA' if prediction[0][0] > 0.5 else 'NORMAL')
plt.show()

model.save('pneumonia_model.h5')
