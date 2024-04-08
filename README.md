# AITherapist

## Setup

`pip install -r requirements.txt`

To use local models, you need to [download Ollama from here](https://ollama.com/download). After this, run

`ollama run llama2`

All of your local models are automatically served on `localhost:11434`. Run `ollama run <name-of-model>` to start interacting via the command line directly.


## Run

`streamlit run main.py`