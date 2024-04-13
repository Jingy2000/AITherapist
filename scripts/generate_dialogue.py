import getpass
import os
import json
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from transformers import AutoTokenizer

os.environ["OPENAI_API_KEY"] = getpass.getpass()
convert_prompt = ("You're a therapist specializing in the field of psychological counseling. "
                  "Your task is to simulate and generate a multi-turn empathetic conversation using the information provided and output a JSON file. "
                  "The requirements are as follows: \n"
                  "1) the conversation contains several turns, \"client\" and \"counselor\" alternate turns in speaking, the same role does not appear in two consecutive messages; \n"
                  "2) The counselor's replies should incorporate elements of empathy based on the user's descriptions, "
                  "such as listening, leading , comforting, understanding, trust, acknowledgment, sincerity, and emotional support; \n"
                  "3) The number of turns between the client and the counselor should be determined by the content of the conversation, ranging from 8 to 20 turns; \n"
                  "4) Within a single turn, the dialogue length between the user and counselor should be appropriate and not overly lengthy, considering the history of their conversation. Important! Please note the conversation is a natural chat between 2 people.\n"
                  "5) In the first message from the client, the client gives a short description about the feeling or the issue. With the development of the conversation, the client provides more information. \n"
                  "6) The conversation starts with a greeting from the counselor like this: Hello, how are you doing. \n"
                  "Please carefully analyze the needs of the user and the empathetic techniques of the counselor in the above single-turn dialogue. Then, rewrite it into a multi-turn empathetic conversation.\n"
                  "Please output a JSON for the multi-turn empathetic conversation. The JSON structure consists of a list named \"messages\". Each item in this list represents a message in the conversation, containing two key-value pairs: \"role\" and \"content\". The \"role\" can be either \"client\" or \"counselor\", indicating who is speaking. The \"content\" is a string that holds the actual text of the message. Important! Please note that it's structured so that \"client\" and \"counselor\" alternate turns in speaking, ensuring that the same role does not appear in two consecutive messages. \n"
                  "This is information is: \n "
                  "{qa} ")

gpt_4_name = "gpt-4-0125-preview"
gpt_3_name = "gpt-3.5-turbo"


def get_top_chat(data_path: str, key: str = 'upvotes') -> pd.DataFrame:
    """
    Reads a CSV file containing counseling chat data, selects relevant columns,
    removes NaN values, sorts the data based on questions and a given key (default: 'upvotes'),
    and returns the top voted answers for each unique question.

    :param data_path: The file path to the CSV containing the counseling chat data.
    :param key: The column to determine the ranking of answers (default: 'upvotes').
    :return: DataFrame containing the top voted answers for each unique question.
    """
    counsel_chat = pd.read_csv(data_path)
    counsel_qa = counsel_chat[["questionText", "answerText", "upvotes", "views"]]

    print(f"There are {len(counsel_qa)} samples in the raw dataset.")
    counsel_qa = counsel_qa.dropna()
    print(f"There are {len(counsel_qa)} samples after remove NaN.")

    qa_sorted = counsel_qa.sort_values(by=['questionText', key], ascending=[True, False])
    qa_top = qa_sorted.groupby('questionText').first().reset_index()
    print(f"There are {len(qa_top)} unique questions in total.")
    return qa_top


def generate_dialogue_from_qa(chain, question: str, answer: str, attempt_times: int = 2) -> dict:
    """
    Generates dialogue based on the question-answer pair.

    :param chain: a langchain runnable
    :param question: the question from client
    :param answer: the answer from counselor
    :param attempt_times: maximum time model attempts to generate
    :return: return the generated dialogue if successfully generated, else return {}
    """
    success = False
    i = 0  # Counter for attempts
    dialogue = {}

    # Loop until successful generation or reaching the maximum attempt times
    while not success and i < attempt_times:
        msg = chain.invoke(
            f"The summary of client's question is :\n {question} \n The summary of counselor's suggestion is: \n {answer}")
        if isinstance(msg, dict) and "messages" in msg:
            dialogue = {"questionText": question, "answerText": answer, "messages": msg["messages"]}
            success = True
        else:
            i += 1
    return dialogue


def generate_dialogues(data: pd.DataFrame, model_name: str, output_path: str, batch_size: int = 10, start_row: int = 0):
    """
    Generate dialogues from given DataFrame using specified model and save them as JSON files.

    :param data: DataFrame containing questions and answers.
    :param model_name: Name of the model to use for generating responses.
    :param output_path: Path to save the generated dialogues.
    :param batch_size: Number of dialogues to generate in each batch.
    :param start_row: Starting row index in the DataFrame.
    :return: None
    """
    llm_model = ChatOpenAI(temperature=0.5, model=model_name).bind(response_format={"type": "json_object"})
    prompt = ChatPromptTemplate.from_template(convert_prompt)
    chain = prompt | llm_model | JsonOutputParser()

    all_dialogue = []
    failed_idx = []

    num_batches = len(data) // batch_size

    for batch_index in range(num_batches + 1):
        batch_dialogues = []
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, len(data))
        if start_index >= end_index:
            break
        batch_data = data.iloc[start_index:end_index]

        for idx, row in enumerate(batch_data.itertuples()):
            question = row.questionText
            answer = row.answerText

            dialogue = generate_dialogue_from_qa(chain, question, answer)
            if dialogue != {}:
                batch_dialogues.append(dialogue)
                print(f" {start_index + idx + 1} dialogue successfully generated.")
            else:
                failed_idx.append(start_row + start_index + idx)
                print(f"Failed on number {start_row + start_index + idx}, the question is: {question} \n ")

        all_dialogue.extend(batch_dialogues)

        json_string = json.dumps(batch_dialogues, indent=4)
        with open(f'{output_path}/dialogue_{start_row + start_index}_{start_row + end_index - 1}.json', 'w') as file:
            file.write(json_string)
        print(f"{end_index} / {len(data)} of dialogue saved")

    # save all dialogues and failed indices as JSON files
    json_string = json.dumps(all_dialogue, indent=4)
    with open(f'{output_path}/all_dialogue_{start_row}_{start_row + end_index - 1}.json', 'w') as file:
        file.write(json_string)

    failed_idx_string = json.dumps(failed_idx)
    with open(f'{output_path}/failed_idx_{start_row}_{start_row + end_index}.json', 'w') as file:
        file.write(failed_idx_string)

    print(f"All dialogues saved!")


def combine_json(data_path: str, file_name: str, batch_indices: list) -> None:
    all_dialogues = []
    for batch_start, batch_end in zip(batch_indices[:-1], batch_indices[1:]):
        with open(f"{data_path}/{file_name}_{batch_start}_{batch_end - 1}.json", 'r') as file:
            all_dialogues.extend(json.load(file))

    with open(f"{data_path}/{file_name}.json", 'w') as file:
        json.dump(all_dialogues, file, indent=4)
        
def validate_dialogue_format(dialogue: dict) -> bool:
    
    """
    Validate the format of a dialogue dictionary.

    Args:
        dialogue (dict): A dictionary representing a dialogue.

    Returns:
        bool: True if the dialogue format is valid, False otherwise.
    """

    role_set = {"client", "counselor"}

    if "messages" not in dialogue:
        print("missing key: message.")
        return False
    
    # meesage should start with counselor
    if len(dialogue["messages"]) == 0 or dialogue["messages"][0]["role"] != "counselor":
        print("Empty message list or message doesn't start with counselor.")
        return False
    else:
        previous_role = "client"
    for msg in dialogue["messages"]:
        # check if role and content are in the mseeage
        if msg["role"] not in role_set:
            print("Invalid role name.")
            return False
        if "content" not in msg or msg["content"] == "":
            print("Include empty content.")
            return False
        # check if same role appears consecutively
        if msg["role"] == previous_role:
            print("The same role appears in two consecutive messages.")
            return False
        previous_role = msg["role"]
    return True
        

def validate_dialouge_dataset(dialouges: list[dict]) -> list[int]:
    invalid_index = []
    
    for index, dialouge in enumerate(dialouges):
        is_dialouge = validate_dialogue_format(dialouge)
        if not is_dialouge:
            print(f"{index} dialouge has format issue.\n")
            invalid_index.append(index)

    print(f"Format validation completed, {len(invalid_index)} / {len(dialouges)} of dialougus have invalid format.")
    return invalid_index


def merge_consecutive_msg(dialouges: dict, invalid_index: list[int]) -> dict:
    """
    Merge consecutive messages with the same role in dialogues.

    Args:
        dialouges (dict): A dictionary containing dialogues.
        invalid_index (list[int]): A list of invalid indices.

    Returns:
        dict: A dictionary with merged consecutive messages.
    """
    for index in invalid_index:
        msgs = dialouges[index]["messages"]
        merged_msgs = []
        i = 0
        while i < len(msgs):
            # print(f"{msg['role']}: {msg['content']}")
            if i < len(msgs) - 1 and msgs[i]["role"] == msgs[i + 1]["role"]:
                content = msgs[i]["content"] + msgs[i + 1]["content"]
                msg = {"role": msgs[i]["role"], "content": content}
                merged_msgs.append(msg)
                i += 2
            else:
                merged_msgs.append(msgs[i])
                i += 1
        dialouges[index]["messages"] = merged_msgs
    print("Merged all consecutive message.")
    return dialouges
    
        
# # Do not use this now, directly use tokenizer.apply_chat_template   
# def format_multiturn_dialogue_llama_(messages: list[dict]) -> str:
#     """
#     Format a list of message into a llama-chat trainable forma. 
#     Noete, the only special character in llama tokenizer is <s>, </s> and <unk>. 
#     Check the link here: https://huggingface.co/NousResearch/Llama-2-7b-chat-hf/blob/main/tokenizer_config.json
#     So, here is some example of the tokenizer:
    
#     "<s></s>" -> ['<s>', '</', 's', '>']
#     "<s> </s>" -> ['<s>', '</s>']
#     "</s><s>" -> ['</s>', '<', 's', '>']
#     "<s>[INST]" ->  ['<s>', '[', 'INST', ']']
#     "<s> hello, world! </s><s> Hi </s>" -> ['<s>', 'hello', ',', 'world', '!', '</s>', '<', 's', '>', 'Hi', '</s>']
    
#     This is the format, as described in this link: https://huggingface.co/blog/llama2#how-to-prompt-llama-2
#     <s>[INST] <<SYS>>
#     {{ system_prompt }}
#     <</SYS>>

#     {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]

  

#     Args:
#         messages (list[dict]): _description_

#     Returns:
#         str: _description_
#     """
    
#     result = ""
#     system_msg = "You are a therapist having a counseling with a visitor."
#     result += f"<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n"
    
#     result += "Hi! [/INST] "
#     result += messages[0]["content"] + " </s>"
    
#     for msg in messages[1:]:
#         result += f"<s>[INST] {msg['content']} [/INST] " if msg['role'] == "client" else f"{msg['content']} </s>"
    
#     return result.strip()

# # Do not use this now, directly use tokenizer.apply_chat_template   
# def generate_dialouge_csv_llama(dialouges: dict, output_path: str, file_name: str = "all_dialogues_formatted_llama") -> None:
#     formatted_dialouge_strings = [format_multiturn_dialogue_llama(dialouge['messages']) for dialouge in dialouges]
#     df = pd.DataFrame(formatted_dialouge_strings, columns=['text'])
#     df.to_csv(f"{output_path}/{file_name}.csv", index=False)



def get_llama_trainable_multiturn_messages(messages: list[dict]) -> dict:
    initial_msg = [{
        "role": "system",
        "content": "You are a therapist having a counseling with a visitor. "
                   "The counselor's replies should incorporate elements of empathy based on the user's descriptions, "
                  "such as listening, leading, comforting, understanding, trust, acknowledgment, sincerity, and emotional support."
    },
    {
        "role": "user",
        "content": "Hi"
    }]
    for msg in messages:
        if msg["role"] == "client":
            msg["role"] = 'user'
        elif msg["role"] == "counselor":
            msg["role"] = 'assistant'
    return initial_msg + messages


def generate_llama_template_dialouge_messages(dialouges: dict, output_path: str, file_name: str = "all_dialogues_formatted_llama"):
    model_name = "NousResearch/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    all_messages = [dialouge['messages'] for dialouge in dialouges]
    
    # save messages list
    all_messages_string = json.dumps(all_messages, indent=4)
    with open(f"{output_path}/{file_name}.json", 'w') as f:
         f.write(all_messages_string)
    
    # save csv dataset
    formatted_dialouge_strings = [tokenizer.apply_chat_template(get_llama_trainable_multiturn_messages(messages), tokenize=False) for messages in all_messages]
    df = pd.DataFrame(formatted_dialouge_strings, columns=['text'])
    df.to_csv(f"{output_path}/{file_name}.csv", index=False)
    
        


if __name__ == "__main__":
    counsel_chat_data = get_top_chat("../data/counsel_chat_dialogue/20220401_counsel_chat.csv", key="upvotes")
    # model = "gpt-4-0125-preview"
    model = "gpt-3.5-turbo"
    start_row = 0
    generate_dialogues(counsel_chat_data.iloc[start_row:],
                       model_name=model,
                       batch_size=10,
                       output_path="../data/test_counsel_chat_dialogue",
                       start_row=start_row)

    # # combine my batch data into a single json file
    # combine_json(data_path="../data/counsel_chat_dialogue", file_name="all_dialogue",
    #              batch_indices=[0, 100, 200, 300, 500, 863])
    
    # dataset validation
    with open("../data/counsel_chat_dialogue/all_dialogue.json", "r") as f:
        dialouges = json.load(f)
    invalid_index = validate_dialouge_dataset(dialouges)
    
    # clean the invalid dialouge by combining consecutive message into one
    dialouges = merge_consecutive_msg(dialouges, invalid_index)  
    validate_dialouge_dataset(dialouges)
    
    # save cleaned dialouge dataset
    json_string = json.dumps(dialouges, indent=4)
    with open("../data/counsel_chat_dialogue/all_dialogue_cleaned.json", "w") as f:
        f.write(json_string)
        
    # generate llama trainable dataset
    with open("../data/counsel_chat_dialogue/all_dialogue_cleaned.json", "r") as f:
        dialouges = json.load(f)
    generate_llama_template_dialouge_messages(dialouges, output_path="../data/counsel_chat_dialogue", file_name="all_dialogue_formatted_llama")
    
    