import json
import torch
from collections import defaultdict
from py_irt.dataset import Dataset
from py_irt.config import IrtConfig
from py_irt.training import IrtModelTrainer


# Load your exam results from JSON
with open('auto-rag-eval/MultiHopData/gov_report/exam_results/llama_3_1_8b_open_gemma2_9b_single_hop_exam_processed.json.json', 'r') as f:
    data_single = json.load(f)
with open('auto-rag-eval/MultiHopData/gov_report/exam_results/llama_3_1_8b_open_exam_new_gemma2_9b_processed_v2.json.json', 'r') as f:
    data_multi = json.load(f)

# exam_data = data_multi + data_single
exam_data = data_multi

# Step 2: Transform data into py-irt format
responses = {}
item_hops = {}  # Map item_id to number_of_hops
for idx, entry in enumerate(exam_data):
    item_id = f"item_{idx}"
    is_correct = int(entry["is_correct"])  # Convert boolean to int
    responses[item_id] = is_correct
    try:
        item_hops[item_id] = entry["number_of_hops"]
    except:
        item_hops[item_id] = 1

# Wrap in a subject response dictionary
dataset = [{"subject_id": "LLM_1", "responses": responses}]

# Save as JSONL file for py-irt
jsonl_path = "irt_dataset.jsonlines"
with open(jsonl_path, "w") as f:
    for row in dataset:
        f.write(json.dumps(row) + "\n")

# Step 3: Load dataset into py-irt
irt_dataset = Dataset.from_jsonlines(jsonl_path)

# Step 4: Configure and train the IRT model
config = IrtConfig(model_type="1pl", log_every=500, dropout=0.2)
trainer = IrtModelTrainer(config=config, data_path=None, dataset=irt_dataset)

# Train the model
trainer.train(epochs=5000, device="cuda" if torch.cuda.is_available() else "cpu")

# Step 5: Analyze results
# Map difficulty scores back to items
item_difficulties = trainer.last_params["diff"]

# Step 6: Group difficulties by number_of_hops
hops_difficulty = defaultdict(list)
for idx, diff in enumerate(item_difficulties):
    item_id = f"item_{idx}"
    num_hops = item_hops[item_id]
    hops_difficulty[num_hops].append(diff)

# Step 7: Calculate average difficulty for each num_hops group
print("\nDifficulty Analysis by Number of Hops:")
for num_hops, difficulties in sorted(hops_difficulty.items()):
    avg_difficulty = sum(difficulties) / len(difficulties)
    min_difficulty = min(difficulties)
    max_difficulty = max(difficulties)
    print(f"""
        Number of Hops: {num_hops},
        Average Difficulty: {avg_difficulty:.3f},
        Min Difficulty: {min_difficulty:.3f},
        Max Difficulty: {max_difficulty:.3f}, 
        Samples: {len(difficulties)}"""
        )
