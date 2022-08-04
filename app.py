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
        starting_text: str = line[random.randrange(0, len(line))].replace("\n", "")
        starting_text: str = re.sub(r"[,:\-–.!;?_]", '', starting_text)
        print(starting_text)

    response = gpt2_pipe(starting_text, max_length=random.randint(20, 45), num_return_sequences=random.randint(5, 15))
    response_list = []
    for x in response:
        if x['generated_text'].strip() != starting_text and len(x['generated_text'].strip()) > (len(starting_text) + 4):
            response_list.append(x['generated_text'])

    response_end = "\n".join(response_list)
    return response_end


txt = grad.Textbox(lines=1, label="English", placeholder="English Text here")
out = grad.Textbox(lines=5, label="Generated Text")
title = "Prompt Generator"
article = "<div><center><img src='https://visitor-badge.glitch.me/badge?page_id=max_skobeev_prompt_generator_public' alt='visitor badge'></center></div>"

grad.Interface(fn=generate,
               inputs=txt,
               outputs=out,
               title=title,
               article=article,
               allow_flagging='never',
               cache_examples=False,
               theme="default").launch(enable_queue=True, debug=True)
