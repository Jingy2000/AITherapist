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
Users using MacOS devices with Apple silicon should note that this application cannot utilize the mps backend for computation acceleration. This limitation is due to Docker operating as a virtual machine on MacOS, which does not support GPU-like hardware passthrough.
2. **CUDA Device Support**:  
The application supports CUDA devices. For optimal performance, ensure that your device has at least 6GB of available memory. Note that memory usage might increase if extended conversations are processed.
Note if you are running the app under WSL2, you need to [Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
4. **Model Management**:
    - Initial Download:  
    Upon first selection, the application will automatically download the chosen models.
    - Storage:  
    All downloaded models are stored in a dedicated Docker volume. This ensures that models persist across application restarts and updates.
    - Size Considerations:  
    Each model requires approximately 4.7GB of storage space.
5. To use the features in this application that rely on OpenAI's models, you will need an API key from OpenAI. You can obtain an API key by logging into your OpenAI account and [creating a new API key](https://platform.openai.com/account/api-keys).

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

## Get Started with Your AI Therapist
### Address
- Go to <http://localhost:8501>
    - note: SQL container may take a few extra seconds for its initiation.
### Configuration
Before starting a new conversation, configure your session:
- Select the Model: Click on the dropdown menu under "Model" to select the appropriate AI model (e.g., gpt-3.5-turbo, llama2).
- Enter Your OpenAI API Key: If you are using models that require an OpenAI API key (such as gpt-3.5-turbo), enter your key in the provided field.
- Adjust the Temperature: Use the slider to set the 'Temperature' which influences the variety in responses. A lower temperature results in more predictable responses, while a higher temperature generates more diverse outputs.
- Submit Configuration: Click 'Submit' to save your settings.

### Start a New Conversation
To initiate a new chat:
- Open a New Conversation Session: Click on the 'Start' button under the "New Conversation" area to begin interacting with the AI therapist.
- Engage with Your AI Therapist: Use the chat input box at the bottom of the "Chat" tab to type your messages.

### Review and Resume Conversations
If you wish to review and/or resume past conversations:
- Access Chat History: Navigate to the "Chat History" section located on the configuration sidebar. Use the dropdown menu to select a previous conversation you wish to review or continue.
- Load and Resume Conversation: Click 'Confirm' to load the selected conversation into the chat interface. Once the conversation is displayed, you can continue interacting with the AI therapist from where you left off.

### Generate a Summary
Once you feel the conversation has reached a point where a summary could be useful, or at the end of your session, navigate to the "Summary" tab next to the "Chat" tab.
- View Your Session Summary  
Click the "Generate Summary" button. This action prompts the AI to analyze the conversation and extract key points, issues, and themes. The summary will appear within the tab, highlighting significant aspects of your dialogue.

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
