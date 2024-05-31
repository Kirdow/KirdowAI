import re
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
import tensorflow as tf
import sys

if len(sys.argv) <= 1:
    print("No username specified")
    exit(1)

user_query = sys.argv[1]

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

def load_dataset(file_path):
    print(f"Loading dataset: {file_path}")
    print("Loading CSV")
    df = pd.read_csv(file_path)
    print("Tokenizing...")
    tokenized_texts = df['text'].apply(lambda x: tokenizer(x, truncation=True, padding='max_length', max_length=128, return_tensors='tf'))
    input_ids = []
    attention_masks = []
    print("Gathering lists...")
    for item in tokenized_texts:
        input_ids.append(item['input_ids'])
        attention_masks.append(item['attention_mask'])
    
    print("Returning dataset from slices...")
    return tf.data.Dataset.from_tensor_slices(({ 'input_ids': tf.concat(input_ids, axis=0), 'attention_mask': tf.concat(attention_masks, axis=0)}, tf.concat(input_ids, axis=0)))

print("Loading training dataset")
train_dataset = load_dataset(f"{user_query}_train.csv")
print("Loading eval dataset")
val_dataset = load_dataset(f"{user_query}_val.csv")

def to_tf_dataset(encoded_dataset):
    def gen():
        for ex in encoded_dataset:
            yield ({ 'input_ids': ex['input_ids'], 'attention_mask': ex['attention_mask']}, ex['input_ids'])
    return tf.data.Dataset.from_generator(gen, ({ 'input_ids': tf.int32, 'attention_mask': tf.int32}, tf.int32))

print("Converting to TF dataset")
print("Converting training TF dataset...")
train_tf_dataset = to_tf_dataset(train_dataset).batch(4)
print("Converting eval TF dataset...")
val_tf_dataset = to_tf_dataset(val_dataset).batch(4)

print("Fetching GPT-2 model...")
model = TFGPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

print("Creating Adam optimizer")
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

print("Compiling model with optimizer...")
model.compile(optimizer=optimizer, loss=model.compute_loss)

@tf.function
def train_step(inputs):
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    with tf.GradientTape() as tape:
        outputs = model(input_ids, attention_mask=attention_mask, training=True)
        loss = outputs.loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

print("Fitting model from datasets...")
for epoch in range(3):
    print(f"Epoch {epoch + 1}/{3}")
    for batch in train_dataset.batch(4):
        loss = train_step(batch[0])
        print(f"Loss: {loss.numpy()}")

print(f"Saving model to ./model_{user_query}")
model.save_pretrained(f"./model_{user_query}")
print(f"Saving tokenizer to ./model_{user_query}")
tokenizer.save_pretrained(f"./model_{user_query}")

print(f"Saved as ./model_{user_query}")