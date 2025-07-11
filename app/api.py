from fastapi import FastAPI, UploadFile, File
from app.predict import load_model, predict_image
import io

app = FastAPI()

class_names = ["animals", "fruits", "vegetables", "vehicles"]
model = load_model("model.pth", num_classes=len(class_names))

@app.get("/")
def root():
    return {"message": "API is running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image_bytes = io.BytesIO(contents)
    prediction = predict_image(image_bytes, model, class_names)
    return {"prediction": prediction}
