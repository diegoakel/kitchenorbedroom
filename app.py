import gradio as gr
from fastai.vision.all import *

learn = load_learner('bedroom_or_kitchen.pkl', 'rb')

categories = ("Bedroom", "Kitchen")

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label()

examples = ['bedroom.jpg', 'quarto.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)

intf.launch()