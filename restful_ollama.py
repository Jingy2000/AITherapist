import requests


def pull(local_model_name: str):
    url = "http://ollama:11434/api/pull"
    data = {"name": local_model_name,
            "stream": False,
            }
    response = requests.post(url, json=data)
    return response

def generate(local_model_name: str, prompt: str):
    url = "http://localhost:11434/api/generate"
    data = {"model": local_model_name,
            "prompt": prompt,
            "stream": False
            }
    response = requests.post(url, json=data)
    return response


if __name__ == '__main__':
    pass
