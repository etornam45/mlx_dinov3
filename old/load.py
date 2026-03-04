from transformers import AutoModel, AutoProcessor
from PIL import Image

model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
model = AutoModel.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

image = Image.open("image.jpg")

inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)

# print(outputs)

print(model.rope_embeddings)