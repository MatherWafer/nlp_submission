import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
from tqdm import tqdm
from safetensors.torch import load_file
# ==========================================
# 1. RE-DEFINE YOUR CUSTOM ARCHITECTURE
# ==========================================
class CustomRobertaNetwork(nn.Module):
    def __init__(self, model_name="roberta-base", num_labels=2):
        super(CustomRobertaNetwork, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.roberta = AutoModel.from_pretrained(model_name, config=self.config)
        
        self.custom_network = nn.Sequential(
            nn.Linear(self.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        second_to_last_layer = outputs.hidden_states[-2]
        cls_embedding = second_to_last_layer[:, 0, :]
        logits = self.custom_network(cls_embedding)
        return SequenceClassifierOutput(logits=logits)

# ==========================================
# 2. LOAD THE COMMITTEE
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# The tokenizer is the same for all 3 models
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

models = []
for i in range(1, 4):
    print(f"Loading Ensemble Model {i}...")
    model_path = f"ensemble_model_{i}"
    
    # Initialize the custom architecture
    model = CustomRobertaNetwork("roberta-base", num_labels=2)

    # Load Safetensors weights
    state_dict = load_file(f"{model_path}/model.safetensors")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval() 
    models.append(model)
import torch.nn.functional as F

# ==========================================
# 3. DEFINE ROBUST INFERENCE FUNCTIONS
# ==========================================
def predict_dev_ensemble(ids_csv, main_tsv, output_txt_file, threshold=0.50):
    print(f"Loading IDs from {ids_csv} and matching text from {main_tsv}...")
    
    # 1. Load the IDs and Main Text
    ids_df = pd.read_csv(ids_csv)
    df_full = pd.read_csv(main_tsv, sep='\t', header=None, 
                          names=['par_id', 'art_id', 'keyword', 'country', 'text', 'label'], 
                          on_bad_lines='skip')
    
    df_merged = ids_df.merge(df_full[['par_id', 'text']], on='par_id', how='left')
    texts = df_merged['text'].fillna("").tolist()
    
    predictions = []
    print("Generating Ensemble Predictions for Dev Set...")
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Soft Voting from all 3 models
            model_probs = []
            for model in models:
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                pcl_prob = probs[0][1].item() 
                model_probs.append(pcl_prob)
                
            average_prob = sum(model_probs) / len(model_probs)
            
            if average_prob >= threshold:
                predictions.append("1")
            else:
                predictions.append("0")
                
    with open(output_txt_file, 'w') as f:
        f.write('\n'.join(predictions) + '\n')
    print(f" Saved exactly {len(predictions)} predictions to {output_txt_file}")


def predict_test_ensemble(test_tsv, output_txt_file, threshold=0.50):
    print(f"Loading official test set from {test_tsv}...")
    # The official test set doesn't have an ID file to merge against, so we just read it directly
    df_test = pd.read_csv(test_tsv, sep='\t', header=None, 
                          names=['par_id', 'art_id', 'keyword', 'country', 'text'], 
                          on_bad_lines='skip')
    texts = df_test['text'].fillna("").tolist()
    
    predictions = []
    print("Generating Ensemble Predictions for Test Set...")
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            model_probs = []
            for model in models:
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                pcl_prob = probs[0][1].item() 
                model_probs.append(pcl_prob)
                
            average_prob = sum(model_probs) / len(model_probs)
            
            if average_prob >= threshold:
                predictions.append("1")
            else:
                predictions.append("0")
                
    with open(output_txt_file, 'w') as f:
        f.write('\n'.join(predictions) + '\n')
    print(f"✅ Saved exactly {len(predictions)} predictions to {output_txt_file}")

# ==========================================
# 4. EXECUTE AND GENERATE FILES
# ==========================================


predict_dev_ensemble(
    ids_csv="dev_semeval_parids-labels.csv", 
    main_tsv="dontpatronizeme_pcl.tsv", 
    output_txt_file="dev.txt",
    threshold=0.50 
)


predict_test_ensemble(
    test_tsv="task4_test.tsv", 
    output_txt_file="test.txt",
    threshold=0.50
)
