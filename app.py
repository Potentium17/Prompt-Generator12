from transformers import pipeline, set_seed
import gradio as grad
import random
import re

gpt2_pipe = pipeline('text-generation', model='succinctly/text2image-prompt-generator')

with open("name.txt", "r") as f:
    line = f.readlines()


def generate(starting_text):
    seed = random.randint(1, 100000)
    set_seed(seed)
    
    # If the text field is empty
    if starting_text == "":
        starting_text: str = line[random.randrange(0, len(line))]
        starting_text: str = re.sub(r"[,:\-–.!;?_]", '', starting_text)
        print(starting_text)
        
    response = gpt2_pipe(starting_text, max_length=random.randint(20, 45), num_return_sequences=random.randint(5, 15))
    response_list = []
    for x in response:
        if x['generated_text'].strip() != starting_text:
            response_list.append(x['generated_text'])

    response_end = "\n".join(response_list)
    return response_end


txt = grad.Textbox(lines=1, label="English", placeholder="English Text here")
out = grad.Textbox(lines=5, label="Generated Text")

grad.Interface(fn=generate,
               inputs=txt,
               outputs=out,
               allow_flagging='never',
               cache_examples=False,
               theme="default").launch(enable_queue=True, debug=True)
