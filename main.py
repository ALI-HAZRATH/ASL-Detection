
# # ================== 1. Imports ==================
# import os
# import pickle
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# # ========== Paths & Parameters ==========
# dataset_path = r"C:\Users\HP\Desktop\sign detection\asl_dataset"
# model_path = r"C:\Users\HP\Desktop\sign detection\mobilenetv2_asl_model.h5"
# label_encoder_path = r"C:\Users\HP\Desktop\sign detection\label_encoder.pkl"
# img_size = 224  # You can try 256 if underfitting continues
# batch_size = 32
# num_epochs = 10
# val_split = 0.2

# # ========== Step 1: Validate Images ==========
# def validate_images(path):
#     removed = 0
#     for cls in os.listdir(path):
#         cls_path = os.path.join(path, cls)
#         if not os.path.isdir(cls_path): continue
#         for f in os.listdir(cls_path):
#             img_path = os.path.join(cls_path, f)
#             try:
#                 img = cv2.imread(img_path)
#                 if img is None or img.shape[0] == 0 or img.shape[1] == 0:
#                     os.remove(img_path)
#                     removed += 1
#             except:
#                 os.remove(img_path)
#                 removed += 1
#     print(f"✅ Removed {removed} corrupt images.")
# validate_images(dataset_path)

# # ========== Step 2: Data Generators ==========
# datagen = ImageDataGenerator(
#     preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
#     rotation_range=25,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     zoom_range=0.3,
#     shear_range=0.2,
#     brightness_range=[0.7, 1.3],
#     horizontal_flip=True,
#     validation_split=val_split
# )

# train_gen = datagen.flow_from_directory(
#     dataset_path, target_size=(img_size, img_size),
#     batch_size=batch_size, class_mode="categorical",
#     subset="training", shuffle=True
# )

# val_gen = datagen.flow_from_directory(
#     dataset_path, target_size=(img_size, img_size),
#     batch_size=batch_size, class_mode="categorical",
#     subset="validation", shuffle=False
# )

# # ========== Step 3: Class Weights ==========
# class_weights = dict(enumerate(compute_class_weight(
#     class_weight="balanced",
#     classes=np.unique(train_gen.classes),
#     y=train_gen.classes
# )))

# # ========== Step 4: Model ==========
# base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3))
# x = GlobalAveragePooling2D()(base_model.output)
# x = Dropout(0.4)(x)
# x = Dense(256, activation='relu')(x)
# x = Dropout(0.3)(x)
# output = Dense(train_gen.num_classes, activation='softmax')(x)
# model = Model(inputs=base_model.input, outputs=output)

# # Freeze base and compile
# base_model.trainable = False
# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# # ========== Step 5: Callbacks ==========
# callbacks = [
#     EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
#     ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
#     ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True, verbose=1)
# ]

# # ========== Step 6: Train Top Layers ==========
# print("🚀 Training top layers...")
# history1 = model.fit(
#     train_gen, validation_data=val_gen,
#     epochs=num_epochs, class_weight=class_weights,
#     callbacks=callbacks
# )

# # ========== Step 7: Fine-tune ==========
# print("🔧 Fine-tuning full model...")
# base_model.trainable = True
# model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

# history2 = model.fit(
#     train_gen, validation_data=val_gen,
#     epochs=num_epochs, class_weight=class_weights,
#     callbacks=callbacks
# )

# # ========== Step 8: Evaluate ==========
# val_gen.reset()
# y_true = val_gen.classes
# y_pred_probs = model.predict(val_gen, verbose=1)
# y_pred = np.argmax(y_pred_probs, axis=1)
# class_names = list(train_gen.class_indices.keys())

# print("\n📊 Classification Report:")
# print(classification_report(y_true, y_pred, target_names=class_names))

# ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=class_names, cmap="Blues", xticks_rotation=45)
# plt.title("Confusion Matrix")
# plt.tight_layout()
# plt.show()

# # ========== Step 9: Save ==========
# print("💾 Saving model and label encoder...")
# model.save(model_path)
# with open(label_encoder_path, "wb") as f:
#     pickle.dump({v: k for k, v in train_gen.class_indices.items()}, f)

# # ========== Step 10: Plot Curves ==========
# def plot_curves(history, title):
#     acc = history.history["accuracy"]
#     val_acc = history.history["val_accuracy"]
#     loss = history.history["loss"]
#     val_loss = history.history["val_loss"]
#     plt.figure(figsize=(10, 4))
#     plt.subplot(1, 2, 1)
#     plt.plot(acc + history2.history["accuracy"], label="Train Acc")
#     plt.plot(val_acc + history2.history["val_accuracy"], label="Val Acc")
#     plt.title(f'{title} Accuracy')
#     plt.legend()

#     plt.subplot(1, 2, 2)
#     plt.plot(loss + history2.history["loss"], label="Train Loss")
#     plt.plot(val_loss + history2.history["val_loss"], label="Val Loss")
#     plt.title(f'{title} Loss')
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

# plot_curves(history1, "MobileNetV2")

# print("✅ Training complete and saved.")

# train_model.py
import os, pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Paths
dataset_path = r"C:\Users\HP\Desktop\sign detection\asl_dataset"
model_path = r"C:\Users\HP\Desktop\sign detection\mobilenetv2_asl_model.h5"
label_encoder_path = r"C:\Users\HP\Desktop\sign detection\label_encoder.pkl"
img_size, batch_size, num_epochs = 224, 32, 10

# Remove corrupt images
def validate_images(path):
    for cls in os.listdir(path):
        cls_path = os.path.join(path, cls)
        if not os.path.isdir(cls_path): continue
        for img_name in os.listdir(cls_path):
            try:
                img = cv2.imread(os.path.join(cls_path, img_name))
                if img is None or img.shape[0] == 0:
                    os.remove(os.path.join(cls_path, img_name))
            except:
                os.remove(os.path.join(cls_path, img_name))

validate_images(dataset_path)

# Data generators
datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2,
    rotation_range=20, zoom_range=0.2, width_shift_range=0.2,
    height_shift_range=0.2, shear_range=0.2, horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    dataset_path, target_size=(img_size, img_size), batch_size=batch_size,
    class_mode='categorical', subset='training'
)
val_gen = datagen.flow_from_directory(
    dataset_path, target_size=(img_size, img_size), batch_size=batch_size,
    class_mode='categorical', subset='validation', shuffle=False
)

# Class weights
class_weights = dict(enumerate(compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)))

# Model definition
base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(train_gen.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)
base_model.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(patience=4, restore_best_weights=True),
    ReduceLROnPlateau(patience=2, factor=0.5, verbose=1),
    ModelCheckpoint(model_path, save_best_only=True, monitor="val_accuracy", verbose=1)
]

# Train top layers
model.fit(train_gen, validation_data=val_gen, epochs=num_epochs, class_weight=class_weights, callbacks=callbacks)

# Fine-tune
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=num_epochs, class_weight=class_weights, callbacks=callbacks)

# Evaluate
val_gen.reset()
y_true = val_gen.classes
y_pred = np.argmax(model.predict(val_gen), axis=1)
labels = list(train_gen.class_indices.keys())

print(classification_report(y_true, y_pred, target_names=labels))
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, cmap="Blues", xticks_rotation=45)
plt.show()

# Save model and label encoder
model.save(model_path)
with open(label_encoder_path, "wb") as f:
    pickle.dump({v: k for k, v in train_gen.class_indices.items()}, f)

