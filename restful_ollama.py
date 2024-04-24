import requests


def send_post_request(local_model_name: str):
    url = "http://ollama:11434/api/pull"
    data = {"name": local_model_name,
            "stream": False,
            }
    response = requests.post(url, json=data)
    return response


if __name__ == '__main__':
    pass
