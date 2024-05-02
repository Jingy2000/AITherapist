# AITherapist
This is the repository for **AI Therapist** Chatbot providing a safe, non-judgmental space to express thoughts, concerns, and emotions.

## Setup Guide
This guide will help you set up the application on your machine. Ensure you have the latest version of Docker installed on your system. The application is compatible with macOS, Linux, and Windows Subsystem for Linux (WSL).

### Prerequisites:
- **Docker**: Make sure Docker is up-to-date on your machine.

### Download from Github
    git clone https://github.com/Jingy2000/AITherapist.git

### Installation Notes:
1. **MacOS Compatibility**:  
Users with MacOS devices that use M1, M2, or M3 chips should note that this application cannot utilize the mps backend for computation acceleration. This limitation is due to Docker operating as a virtual machine on MacOS, which does not support GPU-like hardware passthrough.
2. **CUDA Device Support**:  
The application supports CUDA devices. For optimal performance, ensure that your device has at least 6GB of available memory. Note that memory usage might increase if extended conversations are processed.
3. **Model Management**:
    - Initial Download:  
    Upon first selection, the application will automatically download the chosen models.
    - Storage:  
    All downloaded models are stored in a dedicated Docker volume. This ensures that models persist across application restarts and updates.
    - Size Considerations:  
    Each model requires approximately 4.7GB of storage space.
4. To use the features in this application that rely on OpenAI's models, you will need an API key from OpenAI.

## Preparing to Run the Application
Before running the application, you need to navigate to the app directory, which contains all the necessary files for running the application. Use the following command to change to the app directory:

    cd app

## Run the Application
### Run with CUDA Devices
If you have a CUDA-enabled device and wish to leverage GPU acceleration, use the following command to start the application. This configuration optimizes performance using GPU resources:

    docker compose -f docker-compose-gpu.yml up --build

### Run without CUDA Devices
For systems without CUDA support, you can run the application using the CPU-only version.

    docker compose -f docker-compose-cpu.yml up --build

## Stop the Application
### Regular Shutdown
To stop the application without removing Docker volumes, use the command corresponding to how you started the application:
- If started with GPU support:
    ```
    docker compose -f docker-compose-gpu.yml down
    ```
- If started without GPU support:
    ```
    docker compose -f docker-compose-cpu.yml down
    ```

### Remove All Volumes
If you need to delete Docker volumes, which includes all stored data such as chat history and downloaded models, add the --volumes flag. Use this option cautiously as it will permanently delete all data in the volumes:
-  For GPU configuration:
    ```
    docker compose -f docker-compose-gpu.yml down --volumes
    ```
- For CPU configuration:
    ```
    docker compose -f docker-compose-cpu.yml down --volumes
    ```

### Remove Specific Volumes
If you only want to remove the volume containing chat history, follow these steps to locate and delete the specific volume:
1. List All Docker Volumes:  
Identify the volume associated with chat history, likely named *app_sql-data-db*:
    ```
    docker volume ls
    ```
2. Remove the Specific Volume:  
Once you have confirmed the correct volume name, remove it with:
    ```
    docker volume remove app_sql-data-db
    ```

## Unit Tests
### Prerequisites
- Ensure that you have the necessary Python environment and dependencies installed to run the tests.
- Same as mentioned previously, you need to navigate to the app directory:
    ```
    cd app
    ```

### Set Up Python Environment
Below are the instructions for setting up a virtual environment using **Python 3.11** and installing the required packages.
- If you prefer using Conda, follow these steps:  
    - creat a env with python 3.11:
        ```
        conda create -n yourenvname python=3.11
        ```
    - activate the env:
        ```
        conda activate yourenvname
        ```
    - install the required packages:
        ```
        pip install -r requirements.txt
        ```

- You also can use Python's venv (the instruction will not be included in this guide)

### SQL Database Operations
This test verifies that all database operations such as creation, modification, and retrieval are performing correctly.

    python test_database_operation.py

### Ollama Connectivity
This test checks the connectivity by mocking interaction with the Ollama API. Execute the following command:

    python test_ollama_connection.py
