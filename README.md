# AITherapist

<!-- ## Setup

`pip install -r requirements.txt`

To use local models, you need to [download Ollama from here](https://ollama.com/download). After this, run

`ollama run llama2`

All of your local models are automatically served on `localhost:11434`. Run `ollama run <name-of-model>` to start interacting via the command line directly.


## Run

`streamlit run main.py` -->

## Run
run this app with CUDA devices:
```
docker compose -f docker-compose-gpu.yml up -d --build
```

run this app without CUDA devices:
```
docker compose -f docker-compose-cpu.yml up -d --build
```
stop this app without delete docker volumes:
```
docker compose -f docker-compose-gpu.yml down
```
or
```
docker compose -f docker-compose-cpu.yml down
```
when necessary to delete the volume containing chat history, use flag *--volumes*. Please note this will also remove the volume containing ollama's local models.
