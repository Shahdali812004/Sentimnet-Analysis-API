import os
import dotenv
from dotenv import load_dotenv
#load environment variables from .env file
load_dotenv(override=True) 

#get environment variables
APP_NAME = os.getenv("APP_NAME")
VERSION = os.getenv("VERSION") 
API_KEY = os.getenv("API_KEY")


SRC_FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))

STORAGE_FOLDER_PATH = os.path.join(SRC_FOLDER_PATH, "assets","storage")

os.makedirs(STORAGE_FOLDER_PATH, exist_ok=True)