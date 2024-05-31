import re
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

regex_pattern = r"^\[\d{4}\-\d{2}\-\d{2} \d{2}\:\d{2}\:\d{2}\] \d+ \=\> \[CHAT\]{\¤IceCord.\:\:\#general_discussion\} @([a-zA-Z0-9._]{2,32}): (.+)$"

def preprocess_logs(file_path, user_search=None):
    with open(file_path, 'r', encoding='utf-8') as file:
        logs = file.readlines()
    
    cleaned_logs = []
    for line in logs:
        if match := re.search(regex_pattern, line):
            username = match.group(1)
            username = re.sub(r"[\.\_]", '', username)
            if not user_search is None and username != user_search:
                continue
            message = match.group(2).replace('\b', '\n').strip()
            if len(message) == 0:
                continue

            cleaned_logs.append(message)

    return cleaned_logs

file_path = 'log_1706270868.log'
name_query = sys.argv[1] if len(sys.argv) > 1 else None
cleaned_logs = preprocess_logs(file_path, name_query)

print(f"Processed {len(cleaned_logs)} lines of messaging from #general_discussion")

df = pd.DataFrame(cleaned_logs, columns=['text'])
train_df, val_df = train_test_split(df, test_size=0.1)

train_df.to_csv(f"{name_query}_train.csv", index=False)
val_df.to_csv(f"{name_query}_val.csv", index=False)
