import json
import pandas as pd
import numpy as np
from pydantic import BaseModel

from py_irt.config import IrtConfig
from py_irt.training import IrtModelTrainer
from py_irt.io import write_jsonlines
from py_irt.dataset import Dataset

class Item(BaseModel):
    valid: bool
    difficulty: float
    item_id: str
    category: str = 'all'

class Subject(BaseModel):
    subject_id: str
    skill: float

class Item(BaseModel):
    valid: bool
    difficulty: float
    item_id: str
    category: str = 'all'

class Subject(BaseModel):
    subject_id: str
    skill: float

min_diff = -4
max_diff = 4
validity_rate = .9

max_skill = 4
min_skill = -4


# Load your exam results from JSON
with open('auto-rag-eval/MultiHopData/gov_report/exam_results/gemma2-27b_open_exam_new_gemma2_9b_processed_v2.json.json', 'r') as f:
    data_single = json.load(f)
with open('auto-rag-eval/MultiHopData/gov_report/exam_results/gemma2-27b_open_gemma2_9b_single_hop_exam_processed.json.json', 'r') as f:
    data_multi = json.load(f)

data = data_multi.append(data_single, ignore_index=True)
# Convert data to a pandas DataFrame for easier manipulation
df = pd.DataFrame(data)

# Extract subject and item data (assuming the structure you provided)
subjects = df.groupby('is_correct')['question'].count().reset_index().rename(columns={'question': 'skill'})
items = df[['question', 'number_of_hops']].drop_duplicates().rename(columns={'question': 'item_id', 'number_of_hops': 'difficulty'})

# Convert 'is_correct' to response (0/1)
df['response'] = df['is_correct'].astype(int)

# Function to create the JSON Lines data
def create_irt_data(df, subjects, items):
    rows = []
    for subject_id, skill in subjects.to_records(index=True).values:
        responses = df[df['is_correct'].isin([subject_id])] # Filter responses by subject ID
        responses = responses[['question', 'response']].set_index('question')['response'].to_dict()  # Create response dictionary
        rows.append({"subject_id": subject_id, "skill": skill, "responses": responses})
    return rows

# Generate data and write to JSON Lines
irt_data = create_irt_data(df.copy(), subjects.copy(), items.copy())
write_jsonlines('irt_dataset.jsonlines', irt_data)

print("IRT dataset created and saved to irt_dataset.jsonlines")

dataset = Dataset.from_jsonlines("irt_dataset.jsonlines")

config = IrtConfig(model_type='1pl', log_every=500, dropout=.2)
trainer = IrtModelTrainer(config=config, data_path=None, dataset=dataset)
trainer.train(epochs=5000, device='mps')

for subject, skill, acc in sorted(list(zip(subjects, trainer.last_params['ability'], score_by_subject.values())), key=lambda v: v[0].skill):
    print(subject.subject_id, "Real Skill", subject.skill, "Inferred Skill", skill, "Acc", acc)

for item, diff in sorted(list(zip(items, trainer.last_params["diff"]))[:20], key=lambda v: v[0].difficulty):
    print(item.difficulty, diff)
