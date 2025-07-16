import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from PIL import Image, UnidentifiedImageError
import numpy as np
import pandas as pd



# --- Page Configuration ---
st.set_page_config(
    page_title="Solar Panel Classifier",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar Navigation ---
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🌍 EDA", "🖼️ Image Classifier", "📦 Object Detector", "🧾Detailed Report"]
)

 
if page == "🏠 Home":
    st.title("🔍 SolarGuard: Intelligent Defect Detection on Solar Panels using  DeepLearning")
    st.markdown("This application helps analyse and classify solar panel conditions from images")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Explore Dataset")
        st.markdown("View class distribution, sample images, and image properties.")

    with col2:
        st.subheader("🧠 Predict Conditions")
        st.markdown("🧠 Classify panel conditions using **ResNet18**")
        st.markdown("🎯 Detect defects using **YOLOv8s**")
    
    st.markdown("---")
    st.subheader("📈 Project Summary Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("🖼️ Total Images", "756")
    col2.metric("📁 Validation Images", "146")
    col3.metric("🔍 Classes Detected", "6")

    col4, col5 = st.columns(2)
    col4.metric("🧠 Classifier Accuracy", "94.5%")
    col5.metric("🎯 YOLOv8 mAP@0.5", "0.203")

    st.markdown("---")

    
elif page == "🌍 EDA":
    st.title("📊 Dataset Explorer")
    split = st.sidebar.radio("Choose dataset split", ["Training", "Validation"])
# Paths
    train_dir = r"C:\Users\Sathish\Desktop\project_5\dataset\train"
    val_dir = r"C:\Users\Sathish\Desktop\project_5\dataset\val"
    test_dir = r"C:\Users\Sathish\Desktop\project_5\test"  
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')


    def plot_class_distribution(image_dir, title):
        st.subheader(f"📊Class Distribution - {title}")
        classes = os.listdir(image_dir)
        class_counts = {
            cls: len([
               f for f in os.listdir(os.path.join(image_dir, cls))
               if f.lower().endswith(valid_extensions)
            ])
            for cls in classes
        }

        fig, ax = plt.subplots(figsize=(4, 1.5))
        sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()), ax=ax)
        ax.set_title("Class Distribution", fontsize=8)
        ax.set_xlabel("Class", fontsize=6)
        ax.set_ylabel("Number of Images", fontsize=6)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=4)
        ax.tick_params(axis='y', labelsize=9)
        st.pyplot(fig)
        return classes


    def show_images_per_class(image_dir, classes, n=3):
        st.subheader("📷Sample Images Per Class")
        for cls in classes:
           st.markdown(f"**{cls}**")
           class_path = os.path.join(image_dir, cls)
           images = [
                img for img in os.listdir(class_path)
                if img.lower().endswith(valid_extensions)
            ][:n]
           cols = st.columns(n)
           for i, img_name in enumerate(images):
                img_path = os.path.join(class_path, img_name)
                try:
                    img = Image.open(img_path)
                    cols[i].image(img, use_container_width=True)  
                except UnidentifiedImageError:
                    st.warning(f"Unreadable image skipped: {img_name}")


    def image_stats(image_dir, classes):
        st.subheader("📐 Image Size and Channel Info")
        widths, heights, channels = [], [], []
        for cls in classes:
            for img_name in os.listdir(os.path.join(image_dir, cls))[:20]:
                if not img_name.lower().endswith(valid_extensions):
                    continue
                img_path = os.path.join(image_dir, cls, img_name)
                try:
                   img = Image.open(img_path)
                   w, h = img.size
                   widths.append(w)
                   heights.append(h)
                   channels.append(len(img.getbands()))
                except:
                   continue

        st.write(f"**Average Width**: {np.mean(widths):.2f}")
        st.write(f"**Average Height**: {np.mean(heights):.2f}")
        st.write(f"**Image Modes (Number of Channels)**: {set(channels)}")


    def rgb_mean_std(image_dir, classes):
        st.subheader("🎨 RGB Mean and Standard Deviation")
        r_vals, g_vals, b_vals = [], [], []
        for cls in classes:
            folder_path = os.path.join(image_dir, cls)
            for img_name in os.listdir(folder_path)[:20]:
                if not img_name.lower().endswith(valid_extensions):
                    continue
                img_path = os.path.join(folder_path, img_name)
                try:
                    img = Image.open(img_path).resize((100, 100)).convert('RGB')
                    arr = np.array(img) / 255.0
                    r_vals.append(np.mean(arr[:, :, 0]))
                    g_vals.append(np.mean(arr[:, :, 1]))
                    b_vals.append(np.mean(arr[:, :, 2]))
                except UnidentifiedImageError:
                    continue

        st.write(f"**Mean R**: {np.mean(r_vals):.4f}, **G**: {np.mean(g_vals):.4f}, **B**: {np.mean(b_vals):.4f}")
        st.write(f"**Std R**: {np.std(r_vals):.4f}, **G**: {np.std(g_vals):.4f}, **B**: {np.std(b_vals):.4f}")


# --- Streamlit App Flow ---
    if split == "Training":
       st.header("Training Set")
       train_classes = plot_class_distribution(train_dir, "Training")
       show_images_per_class(train_dir, train_classes)
       image_stats(train_dir, train_classes)
       rgb_mean_std(train_dir, train_classes)
   
    elif split == "Validation":
       st.header("Validation Set")
       val_classes = plot_class_distribution(val_dir, "Validation")
       show_images_per_class(val_dir, val_classes)
       image_stats(val_dir, val_classes)
       rgb_mean_std(val_dir, val_classes)


    

elif page == "🖼️ Image Classifier":
    st.title("🧪 Image Prediction")
    st.markdown("Upload a solar panel image to classify its condition.")
    st.write("Upload an image of a solar panel to detect its condition:")
# --- Class names (must match training) ---
    class_names = ['Bird_Drop', 'Clean', 'Dusty', 'Electrical_Damage', 'Physical_Damage', 'Snow_Covered']

# --- Load trained model ---
    @st.cache_resource
    def load_model():
       weights = ResNet18_Weights.DEFAULT
       model = resnet18(weights=weights)
       model.fc = nn.Linear(model.fc.in_features, len(class_names))  # 6 classes
       model.load_state_dict(torch.load("solar_resnet18_model.pth", map_location=torch.device("cpu")))
       model.eval()
       return model
    model = load_model()

# --- Define transforms (must match training/validation) ---
    weights = ResNet18_Weights.DEFAULT
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        weights.transforms()
    ])

# --- Upload image ---
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
    # Open and display image
        image = Image.open(uploaded_file).convert("RGB")

    # Preprocess
        input_tensor = transform(image).unsqueeze(0)

    # Inference
        with torch.no_grad():
           output = model(input_tensor)
           _, pred_class = output.max(1)
           prob = torch.nn.functional.softmax(output[0], dim=0)[pred_class.item()]
           predicted_class_name = class_names[pred_class.item()]

    # Display prediction info
        st.markdown(f"### 🧠 Predicted Class: `{predicted_class_name}`")
        st.markdown(f"### 🔢 Confidence: `{prob:.2%}`")
    # Display annotated image with label
        img_np = np.array(image)
        fig, ax = plt.subplots()
        ax.imshow(img_np)
        ax.axis('off')
        ax.text(10, 25, f'{predicted_class_name}', fontsize=14, color='white', backgroundcolor='red')
        st.pyplot(fig)


elif page == "📦 Object Detector":
    import os
    import tempfile
    import numpy as np
    from PIL import Image
    from ultralytics import YOLO

    st.title("📦 Object Detection - YOLOv8s")
    st.markdown("Upload an image to detect solar panel conditions using YOLOv8s model.")

    # Confidence threshold slider
    conf_threshold = st.slider("Confidence Threshold", min_value=0.01, max_value=0.5, value=0.05, step=0.01)

    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="object_detect")

    if uploaded_file is not None:
        # Load and display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Save to temporary path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            image.save(tmp_file.name)
            tmp_path = tmp_file.name

        # Load the YOLOv8 model from local path
        model_path = r"C:\Users\Sathish\Desktop\project_5\best.pt"
        model = YOLO(model_path)

        # Run object detection
        results = model.predict(source=tmp_path, conf=conf_threshold)

        # Plot and display the detection results
        result_img = results[0].plot()
        st.image(result_img, caption="YOLOv8s Detection", use_container_width=True)

        # Display detected classes
        st.subheader("📋 Detected Classes")
        if results[0].boxes:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls_id]
                st.markdown(f"✅ **{class_name}** with confidence **{conf:.2%}**")
        else:
            st.warning("❌ No detections made at this confidence threshold.")



elif page == "🧾Detailed Report":

    st.header("📊 DS_SolarPanel_Defect_Detection_using_DeepLearning")

    # Project Overview
    st.markdown("""
    ### 📝 Project Overview
    This web-based application aims to classify and detect various solar panel conditions using deep learning. The pipeline is built using:
    - **ResNet18** for image classification  
    - **YOLOv8s** for object detection  
    - **Streamlit** for user interaction  
    - **Matplotlib/Seaborn** for data visualization and exploratory analysis
    """)

    # Section 1 - EDA (Updated without expanders)
    st.markdown("## 1. 📊 Exploratory Data Analysis (EDA)")

    st.markdown("### 1.1 Class Distribution Overview")
    st.markdown("""
    - **Visualization:** Bar charts for both training and validation splits.  
    - **Observation:** Significant class imbalance, with underrepresented categories like `Physical_Damage`, `Objects`, and `Snow`.  
    - **Suggestion:** Employ data augmentation or class rebalancing techniques like oversampling or synthetic data generation (e.g., SMOTE for images or copy-paste augmentation).
    """)

    st.markdown("### 1.2 Sample Image Previews")
    st.markdown("""
    - **Description:** Representative images for each class provide visual validation of the dataset.  
    - **Enhancement:** Add hover tooltips or overlay labels showing:
        - Image dimensions  
        - Channel info (RGB/RGBA)  
        - Source metadata
    """)

    st.markdown("### 1.3 Image Properties")
    st.markdown("""
    - **Computed Statistics:**
        - Average Width & Height  
        - RGB Mean & Std Dev  
        - Number of Channels (3 vs 4)  
    - **Action Required:** Standardize image channels (e.g., convert 4-channel images to RGB).
    """)

    # Section 2 - Classifier
    st.markdown("## 2. 🧠 Image Classification (ResNet18)")
    st.markdown("""
    **Model Configuration:**
    - Architecture: ResNet18 (pretrained)  
    - Training Set Size: 610  
    - Validation Set Size: 146  
    - Epochs: 20  
    - Optimizer: Adam  
    - Loss: CrossEntropy
    """)

    st.markdown("### 📈 Performance Summary")
    classifier_metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Loss"],
        "Train (Final)": ["92.8%", "0.1966"],
        "Validation (Best @ Epoch 16)": ["94.5%", "0.2126"]
    })
    st.table(classifier_metrics)

    st.markdown("""
    - Excellent generalization on a small dataset  
    - Best validation accuracy at **epoch 16**  
    - Slight overfitting observed after epoch 16
    """)

    # Section 3 - Detector
    st.markdown("## 3. 🎯 Object Detection (YOLOv8s)")
    st.markdown("""
    **Dataset Details:**
    - Validation Set: 23 images  
    - Classes: 7  
    - YOLO Version: 8.3.166  
    """)

    st.markdown("### 📊 Detection Metrics (Validation Set)")
    detector_data = {
        "Class": ["Bird_Drop", "Clean", "Dust", "Electrical_Damage", "Physical_Damage", "Snow", "objects"],
        "Instances": [4, 3, 3, 6, 2, 4, 1],
        "Precision": [0.000, 1.000, 0.000, 0.825, 0.818, 1.000, 0.000],
        "Recall": [0.000, 0.000, 0.000, 0.500, 0.500, 0.000, 0.000],
        "mAP@0.5": [0.0398, 0.287, 0.048, 0.546, 0.499, 0.000, 0.000],
        "mAP@0.5:0.95": [0.0103, 0.0995, 0.0076, 0.153, 0.249, 0.000, 0.000],
    }

    detector_df = pd.DataFrame(detector_data)
    st.dataframe(detector_df)

    st.markdown("""
    - **Overall mAP@0.5:** `0.203`  
    - **Overall mAP@0.5:0.95:** `0.0742`  
    - **Precision:** `0.521`  
    - **Recall:** `0.143`
    """)

    # Insights
    st.markdown("## 📌 Key Insights")
    st.markdown("""
    - Only `Electrical_Damage` and `Physical_Damage` are detected with moderate confidence.  
    - All other classes have **zero recall** → No detections match ground truth.  
    - High precision with low recall indicates **overconfident false positives**.
    """)
