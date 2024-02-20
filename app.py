from fastai.vision.all import *
import gradio as gr

description = 'this is a model that predicts whether an album is metal or pop based on its cover art'

def is_metal_album(x): return x[0].isupper()

learn = load_learner("model.pkl")

categories = ('Pop Album', 'Metal Album')

labels = learn.dls.vocab

def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

examples = ['examples/metalkinggizz.png', 'examples/popkinggizz.jpeg', 'examples/TI.png','examples/TI2.jpeg']

gr.Interface(fn=predict, inputs=gr.Image(height=512,width=512), outputs=gr.Label(num_top_classes=2), examples=examples, description=description).launch()