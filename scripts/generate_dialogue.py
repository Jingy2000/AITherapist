import getpass
import os
import json
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

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


if __name__ == "__main__":
    counsel_chat_data = get_top_chat("../data/20220401_counsel_chat.csv", key="upvotes")
    model = "gpt-4-0125-preview"
    # model = "gpt-3.5-turbo"
    start_row = 300
    generate_dialogues(counsel_chat_data.iloc[start_row:start_row + 200],
                       model_name=model,
                       batch_size=10,
                       output_path="../data/counsel_chat_dialogue",
                       start_row=start_row)
