import os
import shutil
import urllib
import csv
import json

DEFLAMEL_IMAGES = "deflamel_images/"
DEFLAMEL_BASE_URL = "https://yadi.sk/d/-44S_GEJg8nulw/"


class DeflamelImageStock:
    def __init__(self):
        if not (os.path.exists(DEFLAMEL_IMAGES)):
            os.makedirs(DEFLAMEL_IMAGES)

    def image_ids(self):
        annotation_path = os.path.join(DEFLAMEL_IMAGES, "annotation.csv")

        if not (os.path.exists(annotation_path)):
            annotation_url = DEFLAMEL_BASE_URL + "with_bounds.csv"
            with urllib.request.urlopen(self.make_download_url(annotation_url)) as resp, open(annotation_path, 'wb') as out:
                shutil.copyfileobj(resp, out)

        image_ids = []
        with open(annotation_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)
            for row in reader:
                image_id = row[1]
                image_ids.append(image_id)

        return image_ids

    def load_image(self, image_id):
        image_url = DEFLAMEL_BASE_URL + image_id + "/image.png"
        image_dir = os.path.join(DEFLAMEL_IMAGES, image_id)
        image_path = os.path.join(image_dir, "image.png")

        if not(os.path.exists(image_dir)):
            os.makedirs(image_dir)

        if not(os.path.exists(image_path)):
            print("Download image " + image_url + " ...")
            with urllib.request.urlopen(self.make_download_url(image_url)) as resp, open(image_path, 'wb') as out:
                shutil.copyfileobj(resp, out)

        return image_path

    def load_annotation(self, image_id):
        annotation_url = DEFLAMEL_BASE_URL + image_id + "/annotation.json"
        image_dir = os.path.join(DEFLAMEL_IMAGES, image_id)
        annotation_path = os.path.join(image_dir, "annotation.json")

        if not (os.path.exists(image_dir)):
            os.makedirs(image_dir)

        if not(os.path.exists(annotation_path)):
            print("Download annotation " + annotation_url + " ...")
            with urllib.request.urlopen(self.make_download_url(annotation_url)) as resp, open(annotation_path, 'wb') as out:
                shutil.copyfileobj(resp, out)

        with open(annotation_path, "r") as read_file:
            annotation_json = read_file.read()\
                .replace("\'", "\"")\
                .replace("False", "\"False\"")\
                .replace("True", "\"True\"")

            return json.loads(annotation_json)

    @staticmethod
    def make_download_url(resource_path):
        return "https://getfile.dokpub.com/yandex/get/" + resource_path
