import os

def validate_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)

def validate_dirs(*dirs):
    for dir in dirs:
        validate_dir(dir)

class ModelUtil:
    def __init__(self, data_name):
        self.data = f"./{data_name}"
        self.training_data = f"{self.data}/training"
        self.model_data = f"{self.data}/models"
        self.result_data = f"{self.data}/progress"
        self.logs_data = f"{self.data}/logs"

        validate_dirs(self.data, self.training_data, self.model_data, self.result_data, self.logs_data)

    def get_training_user(self, name):
        user_dir = f"{self.training_data}/{name}"
        validate_dir(user_dir)
        return user_dir

    def get_train(self, name, is_csv=True):
        user_dir = self.get_training_user(name)
        return f"{user_dir}/train.{'csv' if is_csv else 'txt'}"

    def get_val(self, name, is_csv=True):
        user_dir = self.get_training_user(name)
        return f"{user_dir}/val.{'csv' if is_csv else 'txt'}"

    def get_model_user(self, name):
        user_dir = f"{self.model_data}/{name}_trained_model"
        validate_dir(user_dir)
        return user_dir

    def get_result(self):
        result_dir = f"{self.result_data}"
        validate_dir(result_dir)
        return result_dir 

    def get_logs(self):
        logs_dir = f"{self.logs_data}"
        validate_dir(logs_dir)
        return logs_dir
        

Util = ModelUtil('appdata')
