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
        starting_text: str = line[random.randrange(0, len(line))].replace("\n", "").lower().capitalize()
        starting_text: str = re.sub(r"[,:\-–.!;?_]", '', starting_text)
        print(starting_text)

    response = gpt2_pipe(starting_text, max_length=random.randint(20, 45), num_return_sequences=random.randint(5, 15))
    response_list = []
    for x in response:
        resp = x['generated_text'].strip()
        if resp != starting_text and len(resp) > (len(starting_text) + 4) and resp.endswith((":", "-", "—")) is False:
            response_list.append(resp)

    response_end = "\n".join(response_list)
    return response_end


txt = grad.Textbox(lines=1, label="English", placeholder="English Text here")
out = grad.Textbox(lines=6, label="Generated Text")
examples = [["mythology of the Slavs"], ["All-seeing eye monitors these world"], ["astronaut dog"],
            ["A monochrome forest of ebony trees"], ["sad view of worker in office,"],
            ["Headshot photo portrait of John Lennon"], ["wide field with thousands of blue nemophila,"]]
title = "Midjourney Prompt Generator"
description = "This is an unofficial demo for Midjourney Prompt Generator. To use it, simply send your text, or click one of the examples to load them. Read more at the links below.<br>Model: https://huggingface.co/succinctly/text2image-prompt-generator<br>Telegram bot: https://t.me/prompt_generator_bot<br>[![](https://img.shields.io/twitter/follow/DoEvent?label=@DoEvent&style=social)](https://twitter.com/DoEvent)"
article = "<div><center><img src='https://visitor-badge.glitch.me/badge?page_id=max_skobeev_prompt_generator_public' alt='visitor badge'></center></div>"

grad.Interface(fn=generate,
               inputs=txt,
               outputs=out,
               examples=examples,
               title=title,
               description=description,
               article=article,
               allow_flagging='never',
               cache_examples=False,
               theme="default").launch(enable_queue=True, debug=True)
