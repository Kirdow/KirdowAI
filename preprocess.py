import re
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os

regex_pattern = r"^(\[\d{4}\-\d{2}\-\d{2} \d{2}\:\d{2}\:\d{2}\] )?\d+ \=\> \[CHAT\]{\¤IceCord.\:\:\#([a-z_-]+)\} @([a-zA-Z0-9._]{2,32}): (.+)$"

def preprocess_logs(file_path, user_search=None):
    with open(file_path, 'r', encoding='utf-8') as file:
        logs = file.readlines()
    
    cleaned_logs = []
    for line in logs:
        if match := re.search(regex_pattern, line):
            channel = match.group(2).lower()
            if channel in ['admin_chat', 'admin_data', 'testing']:
                continue
            username = match.group(3).lower()
            username = re.sub(r"[\.\_]", '', username)
            if not user_search is None and username != user_search:
                continue
            message = match.group(4).replace('\b', '\n').strip()
            if len(message) == 0:
                continue

            cleaned_logs.append(message)

    return cleaned_logs

if len(sys.argv) <= 1:
    print("Missing name")
    exit(1)

limit = None
if len(sys.argv) >= 3:
    limit = int(sys.argv[2])

name_query = sys.argv[1]

def load_logs(file_path, cleaned_logs):
    print(f"Loading logs from: {file_path}")
    new_cleaned_logs = preprocess_logs(file_path, name_query)
    print(f"Found {len(new_cleaned_logs)} logs")
    cleaned_logs.extend(new_cleaned_logs)

cleaned_logs = []
for filename in os.listdir("./icelog"):
    file_path = os.path.join("./icelog", filename)
    if os.path.isfile(file_path) and filename.endswith('.log'):
        load_logs(file_path, cleaned_logs)

print(f"Processed {len(cleaned_logs)} lines of messaging from IceCord")

if not limit is None:
    cleaned_logs = cleaned_logs[-limit:]
    print(f"Limit argument! New line count is {len(cleaned_logs)}.")

df = pd.DataFrame(cleaned_logs, columns=['text'])
train_df, val_df = train_test_split(df, test_size=0.1)

train_df.to_csv(f"{name_query}_train.csv", index=False)
val_df.to_csv(f"{name_query}_val.csv", index=False)

def fix_file(path_from, path_to):
    with open(path_from, 'r') as infile, open(path_to, 'w') as outfile:
        next(infile)
        for line in infile:
            outfile.write(line)

fix_file(f"{name_query}_train.csv", f"{name_query}_train.txt")
fix_file(f"{name_query}_val.csv", f"{name_query}_val.txt")