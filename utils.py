import os

def read_regex(file_path, content=r"^(\[\d+\] )?([a-z]+): (.+)$"):
    found = True
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write(content)
        found = False

    with open(file_path, 'r') as f:
        return (found, f.read())

def validate_logs():
    file_path = "./inputlogs"
    exists = os.path.exists(file_path) and os.path.isdir(file_path)
    if not exists:
        os.mkdir(file_path)
    return (exists, file_path)
    