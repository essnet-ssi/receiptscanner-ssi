import time
import os

def log(str):
    print(str)
    budget_pvc = os.getenv("IMAGE_FOLDER", "/mnt/images")
    with open(f"{budget_pvc}/log.txt", "a") as file:
        file.write(f'{time.ctime()}: {str}\n')