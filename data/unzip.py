import zipfile

with zipfile.ZipFile("raw/raw.zip", mode="r") as archive:
    archive.extractall()

with zipfile.ZipFile("clean/clean.zip", mode="r") as archive:
    archive.extractall()