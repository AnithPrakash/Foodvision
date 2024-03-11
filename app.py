import gradio as gr 
import torch 
import torchvision 
import os 

from model import model_efficientb3
from timeit import default_timer as Timer
from typing import Tuple,Dict

with open("class_names.txt", "r") as f:
    class_name= [food.strip() for food in  f.readlines()]

effenetb3,effenetb3_tranform=model_efficientb3(out_feature=101)

effenetb3.load_state_dict(
    torch.load(
        f="09_pretrained_effnetb3_feature_extractor_food101_percent.pth",
        map_location=torch.device("cpu")
    )
)

def predict(img) -> Tuple[Dict,float]:

    start_time=Timer()

    img=effnetb3_tranform(img).unsqueeze(0)

    effnetb3.eval()
    with torch.inference_mode():
        pred_probs=torch.softmax(effnetb3(img),dim=1)

    pred_labels_and_probs={class_name[i]: float(pred_probs[0][i]) for i in range(len(class_name))}

    pred_time=round(Timer()-start_time,5)

    return pred_labels_and_probs,pred_time


title="Food EuroPlate 🍕🥩🍣"
description= "An EfficientNetB2 feature extractor computer vision model to classify images of food those are seen in europeans ."

example_list=[["example_list/"+example] for example in os.listdir("examples")]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=5, label="Predictions"),
        gr.Number(label="Prediction time (s)"),
    ],
    examples=example_list,
    title=title,
    description=description,
    #article=article,
)

# Launch the app!
demo.launch()


