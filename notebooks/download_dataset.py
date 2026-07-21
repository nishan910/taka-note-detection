import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace("tanvirtain").project("bangladeshi-currency-detection")
version = project.version(3)
dataset = version.download("yolov11", location="data")