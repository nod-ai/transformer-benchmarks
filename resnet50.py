from transformers import AutoFeatureExtractor
from onnxruntime import InferenceSession
from datasets import load_dataset
import time

# load image
dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

# load model
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
session = InferenceSession("model.onnx")

# ONNX Runtime expects NumPy arrays as input
inputs = feature_extractor(image, return_tensors="np")
outputs = session.run(output_names=["last_hidden_state"], input_feed=dict(inputs))

warmup = 5
total_iter = 100
num_iter = total_iter - warmup
for i in range(num_iter):
    if(i == warmup-1):
        start = time.time()
    #print(BertCompiled.learn(predict_sample_input,np.random.randint(5, size=(BATCH_SIZE))))
    outputs = session.run(output_names=["last_hidden_state"], input_feed=dict(inputs))
end = time.time()
total_time = end - start
print("time/iter in ms : "+str(total_time*1000/num_iter))
