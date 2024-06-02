from transformers import AutoTokenizer, AutoModelForCausalLM
from time import sleep
from model_util import Util
from datetime import datetime
import sys

if len(sys.argv) <= 1:
    print("Missing name")
    exit(1)

name_query = sys.argv[1]
if name_query == '--':
    name_query = None
    target_name = 'everyone'
else:
    target_name = name_query

cap_name = target_name[0].upper() + target_name[1:]
print(f"Using model {cap_name}.")

# Load the trained model and tokenizer
model_name = Util.get_model_user(target_name)
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Model loaded!")
sleep(1)

# Function to generate a response
def generate_response(prompt, history, max_length=1000, num_return_sequences=1):
    prompt = prompt.strip()
    if not (prompt.endswith('.') or prompt.endswith('!') or prompt.endswith('?')):
        prompt += '.'
    prompt_with_question = f"{prompt} "

    while True:
        inputs = tokenizer(prompt_with_question, return_tensors='pt', truncation=True, padding=True)
        outputs = model.generate(
            inputs['input_ids'], 
            max_length=max_length, 
            num_return_sequences=num_return_sequences, 
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,  # You can set this to False for deterministic results
            top_k=50,        # Adjust this for sampling (if do_sample=True)
            top_p=0.95       # Adjust this for nucleus sampling (if do_sample=True)
        )
        response = [tokenizer.decode(output, skip_special_tokens=True).replace(prompt_with_question, '').strip() for output in outputs][0].strip()
        if len(response) != 0:
            break
    
    history.append(f"{response}")

    return prompt, response, history

history = []
with open('transcript.txt', 'w') as f:
    def write(s):
        print(s)
        f.write(f"{s}\n")

    current_year = datetime.now().year
    year_notice = f"2024 - {current_year}" if current_year != 2024 else "2024"
    f.write(f"Using model {cap_name} trailed by https://github.com/Kirdow\nCopyright (c) {year_notice} Kirdow\n")

    while True:
        write("------------------------------------")
        prompt = input("Enter Prompt (:q to exit): ")
        if prompt == ":q":
            print("Quitting...")
            exit(0)

        input_prompt, response, history = generate_response(prompt, history)
        write(f"Your Input: {input_prompt}")
        write(f"Model {cap_name} Response: {response}")