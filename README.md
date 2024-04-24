# AITherapist

<!-- ## Setup

`pip install -r requirements.txt`

To use local models, you need to [download Ollama from here](https://ollama.com/download). After this, run

`ollama run llama2`

All of your local models are automatically served on `localhost:11434`. Run `ollama run <name-of-model>` to start interacting via the command line directly.


## Run

`streamlit run main.py` -->

## Run
1. run this app with CUDA devices:
    ```
    docker compose -f docker-compose-gpu.yml up --build
    ```

2. or run this app without CUDA devices:
    ```
    docker compose -f docker-compose-cpu.yml up --build
    ```
## Stop
1. stop this app without delete docker volumes, if you started this app with file *docker-compose-gpu.yml*:
    ```
    docker compose -f docker-compose-gpu.yml down
    ```
2. otherwise
    ```
    docker compose -f docker-compose-cpu.yml down
    ```
3. (option) Use flag *--volumes* when necessary to delete the volume containing chat history. Please note this will also remove the volume containing Ollama's downloaded models.


## Unit Test

1. SQL database
    ```
    python test_database_operation.py
    ```
2. Ollama
    ```
    python test_ollama_connection.py
    ```