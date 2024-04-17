# training notes

## model training

- use tokenizer.apply_chat_template instead of manually writing a format string function
- You MUST NOT assign padding token to be same as the end of sentence (eos) token, because it will cause problem for multi-turn conversation data (eos appears in multi places, confusing the model). We should assign pad as '<pad>' and add it into the tokenizer if it doesn't exist. As mentioned in this artical https://medium.com/@xuebinbin12/fine-tuning-chat-based-llm-with-multi-turn-conversational-data-part-i-d8c64d01a20d
- We should load data collator whnen training multi turn conversations. And we should be careful about the response_template and instruction_template. Those two parameters decide how to calculate the loss, which parts are target labels. This is really import to make the model label the correct token in multi-turn conversation training. See here, https://huggingface.co/docs/trl/sft_trainer#using-tokenids-directly-for-responsetemplate

## merge lora

```python
# Reload model in FP16 and merge it with LoRA weights
new_model = new_model = "../results/Llama-2-7b-chat-therapist-0"
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    # device_map=device_map,
)

model = PeftModel.from_pretrained(base_model, model_id=new_model)
merged_model = model.merge_and_unload()
merged_model.save_pretrained(new_model)

# Reload tokenizer to save it
# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                          trust_remote_code=True,
                                          use_fast=True, 
                                          padding='right')
# tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.save_pretrained(new_model)

```



## Convert the pytorch model into gguf model

ollama run model based on the llama.cpp, we need a gguf model file to run.


### Get the llama.cpp library

```
git clone https://github.com/ggerganov/llama.cpp/tree/master
cd llama.cpp
pip install -r reqiurement.txt
cd ..
```

### Convert the model into gguf format.

```
python llama.cpp/convert.py <your-model-path>
```

In this step, I encountered error:


```
Exception: Vocab size mismatch (model has 32000, but llama2-prueba1-spanish\tokenizer.model has 32001).
```

Then I notice that my "cofnig.json" has a "vocab_size":32000, so I edited it to 32001 and then i can run the converte.py. It successfully generated a gguf model file.

When it finishes, I try to run the model on ollama, I recieve the next error:

```
llama_model_load: error loading model: create_tensor: tensor 'token_embd.weight' has wrong shape; expected 4096, 32001, got 4096, 32000, 1, 1
llama_load_model_from_file: failed to load model
llama_init_from_gpt_params: error: failed to load model 'models/llama2-prueba2.gguf'
{"timestamp":1707825486,"level":"ERROR","function":"load_model","line":599,"message":"unable to load model","model":"models/llama2-prueba2.gguf"}
terminate called without an active exception
```

I finally realized that is because when I load the llama2 model from huggingface, there is a special token `<pad>` into the tokenizer.json. It expands the size of the vocab from 32000 to 32001. 

I think the llama.cpp/convert.py will use the `vocab_size` in config.json to initialize the weight matrix and generate the gguf model file. So it will check if the `vocab_size` is equal to the actual number of tokens in tokenizer.json. Those 2 need to be matched. This is the reason for the first error.

And when running it on ollama, it will load the weights matrix in your gguf model. So the model's token_embd.weight expects the vocab_size in your config.json, but the dimension of embeddings seem to be 32000.

To fix this issue, I just delete this, and convert it again, and run it again, everything is fine. 

```json
{
       "id": 32000,
       "content": "<pad>",
       "single_word": false,
       "lstrip": false,
       "rstrip": false,
       "normalized": true,
       "special": false
}
```

And during my training, due to the limit of the memory, my batch size is 1, I didn't use padding, so it will not affect the model's performance.

I don't know expand the tokenizer vocab and run. If anyone knows, please help!


### Quantize model

The detailed description is in this [link](https://mlabonne.github.io/blog/posts/Quantize_Llama_2_models_using_ggml.html).

 They follow a particular naming convention: “q” + the number of bits used to store the weights (precision) + a particular variant. Here is a list of all the possible quant methods and their corresponding use cases, based on model cards made by TheBloke:

q2_k: Uses Q4_K for the attention.vw and feed_forward.w2 tensors, Q2_K for the other tensors.
q3_k_l: Uses Q5_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K
q3_k_m: Uses Q4_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K
q3_k_s: Uses Q3_K for all tensors
q4_0: Original quant method, 4-bit.
q4_1: Higher accuracy than q4_0 but not as high as q5_0. However has quicker inference than q5 models.
q4_k_m: Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K
q4_k_s: Uses Q4_K for all tensors
q5_0: Higher accuracy, higher resource usage and slower inference.
q5_1: Even higher accuracy, resource usage and slower inference.
q5_k_m: Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K
q5_k_s: Uses Q5_K for all tensors
q6_k: Uses Q8_K for all tensors
q8_0: Almost indistinguishable from float16. High resource use and slow. Not recommended for most users.

As a rule of thumb, I recommend using Q5_K_M as it preserves most of the model’s performance. Alternatively, you can use Q4_K_M if you want to save some memory. In general, K_M versions are better than K_S versions. I cannot recommend Q2_K or Q3_* versions, as they drastically decrease model performance.

In my project, I will use Q4_K_M to save memory

```bash
cd llama.cpp
make
./quantize [your path to]/bonito/ggml-model-f32.gguf [your path to]/bonito/ggml-model-f32.gguf-Q4_K_M.gguf Q4_K_M
```

## Run model in ollama

First, we need to create a Modelfile containing the model information, the template information.

create model in ollama

```bash
ollama create <your-model-name> -f <your-Modelfile-path>
```






