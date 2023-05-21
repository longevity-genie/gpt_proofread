# GPT OCR

GPT OCR is a Python application that leverages the OpenAI GPT-3.5-turbo model to process and proofread text documents produced from PDF OCR, Scan OCR etc. The application is specifically designed for handling `.txt` files. It provides various functionalities such as navigating through a directory of text files, chunking the text for efficient processing, and communicating with the AI model interactively.

## Installation
To install the required libraries, you can use pip:

``` bash
pip install openai
pip install asyncio
pip install textwrap
pip install tiktoken
pip install click
```

## Usage
Run the application via the command line:

``` bash
python main.py ocr --model MODEL --wrap WRAP --prompt_file PROMPT_FILE --base BASE
```

- `MODEL`: The model to use, default is `gpt-3.5-turbo`.
- `WRAP`: The prompt isolation wrap, default is '\`\`\`\<\<PAYLOAD\>\>\`\`\`'.
- `PROMPT_FILE`: The input prompt file, default is `prompt.txt`.
- `BASE`: The base directory to search for text files, default is `./data/papers/`.

Make sure you have a valid OpenAI API key stored in a file named `openaiapi.key` in the same directory as the main script.

## Features

- **Traverse Folder**: The application can traverse through a given directory, find all `.txt` files, and ignore any files that already include `_proofread` in the name or have a corresponding `_proofread` file.

- **Prompt in Chunks**: If a text file is too large, the application can process it in chunks. Each chunk is guaranteed to be within the token limit of the OpenAI model.

- **Write Async**: The application writes the processed and proofread content to a new file with `_proofread` appended to the original file's name.

- **Error Handling**: The application includes handling for rate limit errors and invalid request errors.

## Contributing
Pull requests are welcome. Please ensure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
