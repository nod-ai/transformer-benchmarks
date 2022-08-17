import time
import os
from transformers import AutoFeatureExtractor, ResNetForImageClassification
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")


#PyTorch
import torch

model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
warmup = 5
total_iter = 100
num_iter = total_iter - warmup
for i in range(num_iter):
    if(i == warmup-1):
        start = time.time()
    inputs = feature_extractor(image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_label = logits.argmax(-1).item()
end = time.time()
total_time = end - start
print("PyTorch: time/iter in ms : "+str(total_time*1000/num_iter))
#print(model.config.id2label[predicted_label])


# OnnxRuntime
from onnxruntime import InferenceSession
import urllib.request

if not os.path.isfile("model.onnx"):
    urllib.request.urlretrieve('https://huggingface.co/OWG/resnet-50/resolve/main/onnx/model.onnx',"model.onnx")

session = InferenceSession("model.onnx")

warmup = 5
total_iter = 100
num_iter = total_iter - warmup
for i in range(num_iter):
    if(i == warmup-1):
        start = time.time()
    #print(BertCompiled.learn(predict_sample_input,np.random.randint(5, size=(BATCH_SIZE))))
    # ONNX Runtime expects NumPy arrays as input
    inputs = feature_extractor(image, return_tensors="np")
    outputs = session.run(output_names=["last_hidden_state"], input_feed=dict(inputs))
end = time.time()
total_time = end - start
print("Onnx: time/iter in ms : "+str(total_time*1000/num_iter))

