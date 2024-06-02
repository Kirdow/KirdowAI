from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from model_util import Util
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
train_dataset = load_dataset('text', data_files={'train': Util.get_train(target_name, False)})
eval_dataset = load_dataset('text', data_files={'eval': Util.get_val(target_name, False)})

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
    output_dir=Util.get_result(),
    num_train_epochs=3,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=Util.get_logs(),
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
model.save_pretrained(Util.get_model_user(target_name))
tokenizer.save_pretrained(Util.get_model_user(target_name))