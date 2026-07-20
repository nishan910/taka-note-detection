import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    raise ValueError("ROBOFLOW_API_KEY not found. Check your .env file.")

rf = Roboflow(api_key=api_key)

project = rf.workspace("tanvirtain").project("bangladeshi-currency-detection")
version = project.version(3)

dataset = version.download("yolov11", location="../data")

print("Download complete. Dataset saved at:", dataset.location)