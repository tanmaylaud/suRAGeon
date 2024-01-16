# -*- coding: utf-8 -*-
"""suRAGeon fine-tuning

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Nt2SpQxvLT9HjwPb9CmivZ0rMOSdLFK7

<center><a href="https://www.together.ai" ><img src="https://raw.githubusercontent.com/clam004/together-examples/main/files/togetherlogo.jpg" align="center" width="400"/></a> and <a href="https://wandb.ai" ><img src="https://github.com/wandb/wandb/raw/main/docs/README_images/logo-light.svg#gh-light-mode-only" align="center" width="400"/></a></center>

**[together.ai]("https://www.together.ai")** allows you to train, finetune and deploy large language models on fast compute using 5 lines of code. After you adapt these open source models for your tasks/use cases, you can download the entire model for yourself and/or serve the model on our fast inference engines. see the [docs](https://docs.together.ai/docs/fine-tuning-python) or follow along below.

This python notebook is an example before and after demonstration/tutorial of using Together's finetune API to adapt a base model to a domain specific dataset in law. The dataset consists of legal related questions and answers.

**[Weights & Biases]("https://wandb.ai")** is a system of record for all your ML needs, and has a simple integration into Together finetuning API, which this collab will walk you through
"""

import pandas as pd
import json
import together
import wandb
import pandas as pd

"""Click the 🗝️ in the colab sidebar and add these keys:

TOGETHER_API_KEY

WANDB_API_KEY

HF_TOKEN

 You can get your Together API Key by signing up at [together.ai]("https://www.together.ai"), then going to the profile icon > Settings > API Keys.


<center><img src="https://raw.githubusercontent.com/clam004/together-examples/main/files/API_KEY.png" height=250 width=350></center>

You can get your WANDB_API_KEY by going to [wandb.ai/authorize]("wandb.ai/authorize") and copying the key

Huggingface HF_TOKEN will be used to download datasets and can be retrieved [here](https://huggingface.co/settings/tokens)

First lets import our python packages and check to make sure our API Key is a 64 character alphanumeric string. Also make sure you are using the latest version of together API python library.

"""

WANDB_API_KEY = '<wandb api key>'
TOGETHER_API_KEY = '<together api key>'
HF_TOKEN = '<hugging face api key>'

# check to make sure you have the right key, ie: xxxx.....c04c
print(f"using TOGETHER_API_KEY ending in {TOGETHER_API_KEY[-4:]}")
print("together.VERSION", together.VERSION)

together.api_key = TOGETHER_API_KEY

"""Go to https://docs.together.ai/docs/fine-tuning-models to see a full list of our current base models available for finetuning"""

# lets use our base model to see how it works before we finetune it
base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"

"""See [fine-tuning-task-specific-sequences](https://docs.together.ai/docs/fine-tuning-task-specific-sequences) for a full description of how to template/format/structure your finetuning data for your use case.

The TL;DR is that you can teach your model how to follow a new set of special symbols and grammar, or you can build on top of the patte you can give your model the background knowledge, context or identity in the system prompt `<s>[INST] <<SYS>> {system_prompt} <</SYS>>` and then train it to be consistent with the system prompt instructions as it does multi-turn chat by concatenating repeated ` {user_msg} [/INST] {model_answer} </s>` utterance pairs to the right end of a growing sequence of dialog history. The interface you build will have to use these special sequences like `</s>` and `[/INST]` to extract out the model's responses and input back into the model the growing history of conversational turns. Lets first see how the the model behaves with one method of formatting text, the Llama-2 style.

## Fine-Tuning with Together.ai
"""
dataset_qa = []
with open('/content/results.jsonl', 'r') as file:
    for line in file:

        json_line = json.loads(line)
        if 'Question' not in json_line or 'Answer 1' not in json_line or 'Answer 2' not in json_line or 'Answer 3' not in json_line or 'Answer 4' not in json_line or 'Explanation' not in json_line or 'Correct Answer' not in json_line:
          continue
        correct_answer = "Answer " + json_line['Correct Answer']
        dataset_qa.append((json_line['Question'], json_line[correct_answer]))


print(dataset_qa)

def format_to_llama2_chat(system_prompt, user_model_chat_list):

    """ this function follows from
    https://docs.together.ai/docs/fine-tuning-task-specific-sequences

    It converts this legal dataset into the Llama-2 prompting structure

    Args:
      system_prompt (str): instructions from you the developer to the AI
      user_model_chat_list (List[Tuple[str,str]]): a list of tuples,
        where each tuple is a pair or exchange of string utterances, the first by the user,
        the second by the AI. The earlier exchanges are on the left, meaning time
        runs left to right.
    Returns:
      growing_prompt (str): the concatenated sequence starting with system_prompt and
        alternating utterances between the user and AI with the last AI utternance on the right.
    """

    growing_prompt = f"""<s>[INST] <<SYS>> {system_prompt} <</SYS>>"""

    for user_msg, model_answer in user_model_chat_list:
        growing_prompt += f""" {user_msg} [/INST] {model_answer} </s>"""

    return growing_prompt

# format_to_llama2_chat(
#     "You are a good robot",
#     [("hi robot", "hello human"),("are you good?", "yes im good"),("are you bad?", "no, im good")]
# )

data_list = []

for sample in dataset_qa:
    training_sequence = format_to_llama2_chat(
        "You are answering questions regarding Otolaryngology, Head & Neck Operative with brief one sentence answers",
        [(sample[0],sample[1])]
    )

    data_list.append({
        "text":training_sequence
    })

print(len(data_list))
print(data_list[0])

# save the reformatted dataset locally
together.Files.save_jsonl(data_list, "surgery.jsonl")

# check your data with your base model prompting type before uploading
resp = together.Files.check(file="surgery.jsonl")
print(resp)

resp = together.Files.list()
print(resp)

# upload your dataset file to together and save the file-id, youll need it to start your finetuning run
file_resp = together.Files.upload(file="surgery.jsonl")
file_id = file_resp["id"]
print(file_id)
print(file_resp)

"""Expected output: ```{'filename': 'legal_dataset.jsonl', 'id': 'file-69649a68-6a36-41ad-8420-1750e99c26a7', 'object': 'file', 'report_dict': {'is_check_passed': True, 'model_special_tokens': 'we are not yet checking end of sentence tokens for this model', 'file_present': 'File found', 'file_size': 'File siz```

## Finetuning the base model on the new dataset to create the new finetuned model

depending on the size of the model and dataset this could take a few minutes to hours
"""

# Submit your finetune job
ft_resp = together.Finetune.create(
  training_file = file_id ,
  model = base_model_name,
  n_epochs = 4,
  batch_size = 4,
  n_checkpoints = 2,
  learning_rate = 1e-5,
  wandb_api_key = WANDB_API_KEY,
  #estimate_price = True,
  suffix = 'law',
)

fine_tune_id = ft_resp['id']
model_output_name = ft_resp['model_output_name']
print(fine_tune_id, model_output_name)

# run this from time to time to check on the status of your job
print(together.Finetune.retrieve(fine_tune_id=fine_tune_id)) # retrieves information on finetune event
print("-"*50)
print(together.Finetune.get_job_status(fine_tune_id=fine_tune_id)) # pending, running, completed
print(together.Finetune.is_final_model_available(fine_tune_id=fine_tune_id)) # True, False
print(together.Finetune.get_checkpoints(fine_tune_id=fine_tune_id)) # list of checkpoints

"""Monitor your training/finetuning progress using weights and biases. Below we verify that our model is learning and our training loss is decreasing.

<center><img src="https://raw.githubusercontent.com/clam004/together-examples/main/files/wandb.jpg" height=300 width=600> </center>

"""

print(together.Finetune.get_job_status(fine_tune_id=fine_tune_id)) # pending, running, completed

"""get_job_status will transition from `pending`, `queued`, `running`, to `complete`.

when the job is finished, you should see:
```
{'training_file': 'file-69649a68-6a36-41ad-8420-1750e99c26a7', 'model_output_name': 'carson/llama-2-7b-chat-law-2023-09-22-20-37-12', 'model_output_path': 's3://together-dev/finetune/64c4302a5cb247a0c80a3ddb/carson/llama-2-7b-chat-law-2023-09-22-20-37-12/ft-2aaecf7b-ff6f-4341-813a-45d84ae2b1bf-2023-09-22-13-47-25', 'Suffix': 'law', 'model': 'togethercomputer/llama-2-7b-chat', 'n_epochs': 2, 'n_checkpoints': 1, 'batch_size': 4, 'learning_rate': 0.0001, 'user_id': '64c4302a5cb247a0c80a3ddb', 'staring_epoch': 0, 'training_offset': 0, 'checkspoint_path': '', 'random_seed': '', 'created_at': '2023-09-22T20:37:12.007Z', 'updated_at': '2023-09-22T20:52:52.675Z', 'status': 'completed', 'owner_address': '0xef5286fc0a1ac5bc4d4221cf3d51f1d97c45eaf7', 'id': 'ft-2aaecf7b-ff6f-4341-813a-45d84ae2b1bf', 'job_id': '1509', 'token_count': 1676524, 'param_count': 6738415616, 'total_price': 5000000000, 'epochs_completed': 2, 'events': [{'object': 'fine-tune-event', 'created_at': '2023-09-22T20:37:12.007Z .....

completed

True
[]
```

The name of your finetuned model will show up in your list of models, but before you can start using it, you need to start it and it needs to finish deploying.

You can also find the name of your new model, start it and stop it, at https://api.together.xyz/playground

under `Models` > `My Model Instances`

<center><img src="https://raw.githubusercontent.com/clam004/together-examples/main/files/mymodels.jpg" height=300 width=600></center>
"""

# replace this name with the name of your newly finetuned model
new_model_name = 'paderno@stanford.edu/Mistral-7B-Instruct-v0.2-law-2024-01-14-04-21-04'

model_list = together.Models.list()

print(f"{len(model_list)} models available")

available_model_names = [model_dict['name'] for model_dict in model_list]

new_model_name in available_model_names

# deploy your newly finetuned model
together.Models.start(new_model_name)

# check if your model is finished deploying, if this returns {"ready": true}, you model is ready for inference
together.Models.ready(new_model_name)

test_chat_prompt = "<s>[INST] <<SYS>> You are answering questions regarding Otolaryngology, Head & Neck Operative with brief several sentence answers <</SYS>>\
<insert question here>?[/INST]"

# use the inference API to generate text / create completion / chat
new_model_name = 'paderno@stanford.edu/Mistral-7B-Instruct-v0.2-law-2024-01-14-04-21-04'
prompt = test_chat_prompt
print(new_model_name)

output = together.Complete.create(
  prompt = test_chat_prompt,
  model = new_model_name,
  max_tokens = 256,
  temperature = 0.6,
  top_k = 90,
  top_p = 0.8,
  repetition_penalty = 1.1,
  stop = ['</s>']
)

# print generated text
print(output['prompt'][0]+" -> "+output['output']['choices'][0]['text'])