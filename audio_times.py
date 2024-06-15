import json
from aeneas.task import Task
from aeneas.tools.execute_task import ExecuteTask
from nltk.tokenize import sent_tokenize

transcript_file_path = "/Users/ada/Desktop/AI_open_mic_dataset/used/DG_W_text_11.txt"
with open(transcript_file_path, "r") as file:
    transcript = file.read()

sentences = sent_tokenize(transcript)

sentence_file_path = "/Users/ada/Desktop/AI_open_mic_dataset/used/dg_sentences.txt"
with open(sentence_file_path, "w") as file:
    for sentence in sentences:
        file.write(sentence + '\n') 

audio_file_path = "/Users/ada/Desktop/AI_open_mic_dataset/used/audio/DG_W_audio_11.mp3"
output_file_path = "/Users/ada/Desktop/AI_open_mic_dataset/used/dg_output.json"
config_string = "task_language=eng|is_text_type=plain|os_task_file_format=json"

task = Task(config_string=config_string)
task.audio_file_path_absolute = audio_file_path
task.text_file_path_absolute = sentence_file_path
task.sync_map_file_path_absolute = output_file_path

ExecuteTask(task).execute()
task.output_sync_map_file()