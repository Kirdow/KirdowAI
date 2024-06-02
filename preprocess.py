import re
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os
from model_util import Util

# Predicates used for filtering
# Modify as needed
predicate_min_len = 0
predicate_max_len = 500
predicate_min_words = 8
predicate_max_words = 0
predicate_allow_new_line = False

# This regex pattern is for the specifc format I use in the logs
# luckily the logs never included the discriminator, so I won't
# need to keep track of two types of usernames here.
regex_pattern = r"^(\[\d{4}\-\d{2}\-\d{2} \d{2}\:\d{2}\:\d{2}\] )?\d+ \=\> \[CHAT\]{\¤IceCord.\:\:\#([a-z_-]+)\} @([a-zA-Z0-9._]{2,32}): (.+)$"

def preprocess_logs(file_path, user_search=None, blacklist=None):
    with open(file_path, 'r', encoding='utf-8') as file:
        logs = file.readlines()
    
    cleaned_logs = []
    for line in logs:
        if match := re.search(regex_pattern, line):
            # Get channel name, we don't want anything from the admin channels
            # We also filter out the NSFW channel because it's often used for
            # messages that are written less carefully or contain heavily
            # out of context chatting
            channel = match.group(2).lower()
            if channel in ['admin_chat', 'admin_data', 'testing', 'nsfw']:
                continue

            # we need to filter out symbols in order for the filter to work.
            # both before the username update and after. without this, the
            # search would potentially miss whenever the username wasn't
            # available after the update
            username = match.group(3).lower()
            username = re.sub(r"[^a-z]", '', username)
            
            # we include a user search for the cases when the model is to be
            # trained from a single user only.
            if not user_search is None and username != user_search:
                continue

            # The blacklist will be used for users who have opted-out of the big/global model
            if not blacklist is None and username in blacklist:
                continue

            # fetch the message first
            message = match.group(4).replace('\b', '\n').strip()

            # do the filter second and before predicates
            # as predicates rely on the messages that are ready to be
            # used in training

            # remove tags
            message = re.sub(r"\<\@\!?\d+\>", '', message)
            # remove channel refs
            message = re.sub(r"\<\#\d+\>", '', message)
            # remove emotes
            message = re.sub(r"\<\:[^\:]\:\d+\>", '', message)
            # remove symbols
            message = re.sub(r"[\"\:\`\*\~\,]", '', message)
            # remove understrikes (hopefully)
            message = re.sub(r"__", '', message)

            # finalize removal with a trim in order to get rid of spaces at the start and end
            # these would appear due to the space after (or before) a mention tag
            message = message.strip()

            # split into words for use with a few predicates
            words = [x for x in message.split(' ') if len(x.strip()) > 0]
            
            # run predicates
            # first condition makes sure the predicate is enabled
            if predicate_min_len > 0 and len(message) < predicate_min_len:
                continue
            if predicate_max_len > 0 and len(message) > predicate_max_len:
                continue
            if predicate_min_words > 0 and len(words) < predicate_min_words:
                continue
            if predicate_max_words > 0 and len(words) > predicate_max_words:
                continue
            if not predicate_allow_new_line and '\n' in message:
                continue

            # final check for empty message, needed in case no predicates were used
            if len(message) == 0:
                continue

            # add the message
            cleaned_logs.append(message)

    return cleaned_logs

if len(sys.argv) <= 1:
    print("Missing name")
    exit(1)

blacklist = []

name_query = sys.argv[1]
if name_query == '--':
    if len(sys.argv) < 3:
        print("Missing limit")
        exit(1)
    
    limit = int(sys.argv[2])
    name_query = None
    blacklist = [x for x in sys.argv[3:]]
    target_name = 'everyone'
else:
    limit = None
    if len(sys.argv) >= 3:
        limit = int(sys.argv[2])
    target_name = name_query

def load_logs(file_path, cleaned_logs):
    print(f"Loading logs from: {file_path}")
    new_cleaned_logs = preprocess_logs(file_path, name_query, blacklist)
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
def save(df, name):
    with open(name, 'w') as f:
        f.write('\n'.join([item for item in df['text']]))
save(train_df, Util.get_train(target_name, False))
save(val_df, Util.get_val(target_name, False))
