from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
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
print(f"Query name: {target_name}")

# Load the datasets
train_dataset = load_dataset('text', data_files={'train': f'{target_name}_train.txt'})
eval_dataset = load_dataset('text', data_files={'eval': f'{target_name}_val.txt'})

# Initialize the tokenizer and model
model_name = 'gpt2'  # You can choose a different model if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

# Tokenize the datasets
def tokenize_function(examples):
    tokenized = tokenizer(examples['text'], return_tensors='pt', truncation=True, padding='max_length', max_length=512)
    tokenized['labels'] = tokenized['input_ids']
    return tokenized

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset['train'],
    eval_dataset=tokenized_eval_dataset['eval'],
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained(f'./{target_name}_trained_model')
tokenizer.save_pretrained(f'./{target_name}_trained_model')