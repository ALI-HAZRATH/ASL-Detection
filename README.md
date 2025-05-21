
# 🧠 ASL Sign Language Recognition using MobileNetV2

## 📌 Introduction

This project focuses on recognizing American Sign Language (ASL) hand gestures (A–Z) using a deep learning model. It is designed to aid communication for people with hearing or speech impairments and can be used in education, accessibility tools, or even human-computer interaction systems. The system supports both real-time webcam-based recognition and static image upload via a web interface.

---

## 📦 Requirements

- A dataset of ASL hand gestures (A–Z folders with images)
- Python 3.8 or higher
- A working webcam (for real-time prediction)
- Basic knowledge of Python and machine learning

Python libraries needed (install using `pip`):

- TensorFlow 2.15.0
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- MediaPipe

---

## 🛠️ How to Use

1. Install dependencies using:
   ```
   pip install -r requirements.txt
   pip install streamlit mediapipe
   ```

2. Train the model using your dataset by running `main.py`.

3. For real-time prediction using your webcam, run:
   ```
   python prediction.py
   ```

4. For image upload and webcam interface via web app, run:
   ```
   streamlit run app.py
   ```

---

## 🔄 Pipeline Explanation

1. **Data Preparation**
   - Validate and clean dataset (remove corrupt images).
   - Organize dataset in folders from A to Z.

2. **Data Augmentation**
   - Apply transformations such as zoom, rotation, brightness, and flip for robust training.

3. **Model Training**
   - Use MobileNetV2 (pretrained) as a base.
   - Add custom layers for classification.
   - Fine-tune the model after training top layers.

4. **Evaluation & Saving**
   - Evaluate model with classification report and confusion matrix.
   - Save model and label encoder.

5. **Real-time Prediction**
   - Capture webcam frames.
   - Use MediaPipe to detect and crop hand region.
   - Predict and display output with confidence and FPS.

6. **Streamlit App**
   - Upload images or stream from webcam for prediction.
   - Visual feedback with labels and confidence.

---

## 🤔 Why MobileNetV2?

- **Efficient and Lightweight**: Great for real-time applications on low-resource machines.
- **Pretrained on ImageNet**: Provides good feature extraction without needing huge data.
- **Fast Inference**: Important for smooth webcam experience.
- **Customizable**: You can easily fine-tune or extend it.

---

## 💡 Suggestions for Improvement

- Add a larger, more diverse dataset with multiple hand orientations.
- Include dynamic gestures (like “Hello”, “Thank you”) using video-based models.
- Integrate voice output (Text-to-Speech) for real-time speech generation.
- Deploy on mobile using TensorFlow Lite.
- Add user authentication and profile tracking.
- Create a REST API for backend integration with other apps.

---

## ✅ Conclusion

This ASL Sign Language Recognition project is a robust and scalable system for understanding static hand gestures. It blends deep learning, computer vision, and user-friendly UI via Streamlit. Ideal for educational and accessibility solutions, it demonstrates how AI can empower inclusive communication.

