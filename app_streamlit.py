import os 
import streamlit as st
import torch
from PIL import Image


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from carClassifier import(
    simpleCarClassifier,
    predict_image,
    transform

)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = [
    "convertible",
    "coupe",
    "hatchback",
    "pickup",
    "sedan",
    "suv",
    "van"
]

model = simpleCarClassifier(
    numclasses=len(class_names)
)

model_path = os.path.join(os.path.dirname(__file__), "car_classifier.pth")
if os.path.exists(model_path):
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )
else:
    st.warning("No saved model weights were found. The app will run with a fresh model.")

model = model.to(device)
model.eval()

st.title("Car Classification App")

fileUpload = st.file_uploader("Upload an image of any " \
"convertible, coupe, hatchback, van, pickup, suv, or sedan",
 type=["jpg", "png", "jpeg"])

if fileUpload is not None:
    image = Image.open(fileUpload).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    st.write("procesing results...")

    result = predict_image(image=image, model=model, transform=transform, device=device, class_names=class_names
                           )
    st.subheader("Prediction Results")
    st.write("Predicted Class:", result)

