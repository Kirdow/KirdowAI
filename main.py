from transformers import AutoTokenizer, AutoModelForCausalLM
from time import sleep
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
print(f"Using model: {target_name}")

# Load the trained model and tokenizer
model_name = f'./{target_name}_trained_model'
print("Loading model")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Model loaded!")
sleep(1)

# Function to generate a response
def generate_response(prompt, history, max_length=1000, num_return_sequences=1):
    if not prompt.strip().endswith('?'):
        prompt = prompt.strip() + '?'

    prompt_with_question = f"Q: {prompt}\nA:"

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
    responses = [tokenizer.decode(output, skip_special_tokens=True).replace(prompt_with_question, '').strip() for output in outputs]
    
    for entry in responses:
        history.append(f"{entry}")

    return prompt, responses, history

history = []
while True:
    print("------------------------------------")
    prompt = input("Enter Prompt (:q to exit): ")
    if prompt == ":q":
        print("Quitting...")
        exit(0)

    input_prompt, responses, history = generate_response(prompt, history)
    print(f"Your Input: {input_prompt}")
    for i, response in enumerate(responses):
        print(f"Response {i + 1}: {response}")