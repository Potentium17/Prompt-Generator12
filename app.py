from transformers import pipeline, set_seed
import gradio as grad
import random


gpt2_pipe = pipeline('text-generation', model='succinctly/text2image-prompt-generator')
set_seed(42)

def generate(starting_text):
    seed = random.randint(1, 10000000)
    response= gpt2_pipe(starting_text, max_length=20, num_return_sequences=5)
    return response
    
txt=grad.Textbox(lines=1, label="English", placeholder="English Text here")
out=grad.Textbox(lines=1, label="Generated Text")

grad.Interface(fn=generate, inputs=txt, outputs=out, 
             allow_flagging='never',
             cache_examples=False,
             theme="default").launch(enable_queue=True, debug=True)