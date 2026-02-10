import os
import gdown

MODEL_URL = "https://drive.google.com/uc?id=1PW5PqmamLpFYWOinVyXAP-bBO0OzejTF"
MODEL_PATH = "crop_yield_model.pkl"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading ML model...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    else:
        print("Model already exists")

if __name__ == "__main__":
    download_model()
