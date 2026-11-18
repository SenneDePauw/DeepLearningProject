"""Downloadt enkel de Free_ST_English subset (~1.6 GB) van Hugging Face.
Gebruik: python download_data.py
Zet de data BUITEN je OneDrive-map, anders probeert OneDrive 1.6 GB te synchroniseren."""
from huggingface_hub import snapshot_download

TARGET = r"C:\kws_data"

path = snapshot_download(
    repo_id="lugan/Syntts-Commands-Media-Dataset",
    repo_type="dataset",
    allow_patterns=["*Free_ST_English*"],
    local_dir=TARGET,
)
print("Klaar. Data staat in:", path)
