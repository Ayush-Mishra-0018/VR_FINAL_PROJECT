from transformers import AutoImageProcessor, AutoModelForObjectDetection
import torch

processor = AutoImageProcessor.from_pretrained("valentinafevu/yolos-fashionpedia")
model = AutoModelForObjectDetection.from_pretrained("valentinafevu/yolos-fashionpedia")

print(model.config.id2label)
