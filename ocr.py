import click
import openai
import asyncio
import textwrap
import tiktoken

from pycomfort.files import *
from typing import List, Tuple
from click import Context
from pathlib import Path
from time import sleep

openai.api_key = open('openaiapi.key', 'r', encoding='utf-8').read()
default_model = "gpt-3.5-turbo"
payload_wrap = '```<<PAYLOAD>>```'
payload_keyword = '```<<PAYLOAD>>```'
prompt_cap = 4096
completion = 1000
max_retry = 10

CHATCOMPLETION_OPTIONS = {
    "temperature": 0.6,
    "max_tokens": 1000,
    "top_p": 1,
    "frequency_penalty": 0.25,
    "presence_penalty": 0,
    "stop": ['<<END>>']
}


def generate_prompt_messages(message: str, prompt: str, dialog_messages: List[str]):
    messages = [{"role": "system", "content": prompt}]
    for dialog_message in dialog_messages:
        messages.append({"role": "user", "content": dialog_message["user"]})
        messages.append({"role": "assistant", "content": dialog_message["bot"]})
    messages.append({"role": "user", "content": message})

    return messages


async def prompt_ai(id: str, prompt: str, payload: str, model: str, wrap: str) -> Tuple[str, int, int]:
    take = 0
    input_tokens = 0
    output_tokens = 0
    message = wrap.replace(payload_keyword, payload)
    message = message.encode(encoding='ASCII', errors='ignore').decode()
    messages = generate_prompt_messages(message, prompt, [])
    answer = None
    while answer is None:
        try:
            if take>2:
                print(f"id={id}/try number {take+1} of {max_retry}!")

            r = await openai.ChatCompletion.acreate(
                model=model,
                messages=messages,
                **CHATCOMPLETION_OPTIONS
            )
            answer = r.choices[0].message["content"]
            input_tokens, output_tokens = r.usage.prompt_tokens, r.usage.completion_tokens
        except openai.error.RateLimitError as e:
            sleep(5)
        except openai.error.InvalidRequestError as e:  # too many tokens
            take += 1
            if take >= max_retry:
                raise e
    print(f"id={id}/finished on try {take+1} of {max_retry}!")
    return answer, input_tokens, output_tokens


def num_tokens(string: str, model: str) -> int:
    """Returns the number of tokens for a model"""
    encoding = tiktoken.encoding_for_model(model)
    n_tokens = len(encoding.encode(string))
    return n_tokens


async def prompt_in_chunks(text: str, prompt: str, model: str, wrap: str) -> Tuple[str, int]:
    c_limit = prompt_cap - completion - num_tokens(prompt, model)  # both input and completion must fit
    act_size = c_limit
    size = c_limit
    while act_size >= c_limit:
        size = size - 1 - (act_size - size) // 4
        chunks = textwrap.wrap(text, size * 4)
        act_size = 0
        for chunk in chunks:
            tokens = num_tokens(chunk, model)
            if act_size < tokens:
                act_size = tokens
        #print(f"{size}/{act_size}/{c_limit} - expectation/real size/limit")

    print(f"expectation={size}/tokens={act_size}/limit={c_limit}/total_chunks={len(chunks)}")
    tasks = [prompt_ai(f"ID_{i+1}", prompt, chunk, model, wrap) for i, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks)
    joined_results = ''.join(result[0] for result in results)
    return joined_results, len(chunks)


def traverse_folder(folder: Path) -> List:
    if folder is None:
        folder = Path.cwd()
    texts = []
    for p in folder.glob("**/*.txt"):
        if '_proofread' in p.stem:
            continue
        output_path = p.with_stem(p.stem + '_proofread')
        if output_path.exists():
            print(f"{output_path} - proofread exists! Skipping ocr input")
            continue
        texts.append(p)
    return texts

async def write_async(model: str, keyword: str, prompt_file: str, base: str):
    texts = traverse_folder(Path(base))
    with open(prompt_file, 'r', encoding='utf-8') as pf:
        prompt = pf.read()
    for t in texts:
        # doi = f"http://doi.org/{t.parent.name}/{t.stem}"
        with open(t, 'r', encoding='utf-8') as inf:
            text = inf.read()
            print(f"{t} - processing ocr input")
            if len(text) < 10:
                print(f"{t.parent.name}/{t.stem} - TOO SHORT TEXT")
                continue
            processed_text, chunks = await prompt_in_chunks(text, prompt, model, keyword)
            output_path = t.with_stem(t.stem + '_proofread')
            print(f"{output_path} - writing output")
            with open(output_path, 'w', encoding='utf-8') as outf:
                outf.write(processed_text)


@click.group(invoke_without_command=False)
@click.pass_context
def app(ctx: Context):
    # if ctx.invoked_subcommand is None:
    #    click.echo('Running the default command...')
    #    test()
    pass


@app.command("ocr")
@click.option('--model', default=default_model, help='model to use, gpt-3.5-turbo by default')
@click.option('--wrap', default=payload_wrap, help='prompt isolation wrap')
@click.option('--prompt_file', default='prompt.txt', help='input prompt')
@click.option('--base', default=' ./data/papers/', help='base folder')
def write(model: str, wrap: str, prompt_file: str, base: str):
    asyncio.run(write_async(model, wrap, prompt_file, base))


if __name__ == '__main__':
    app()
