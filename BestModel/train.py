import pandas as pd
import numpy as np
import torch
import optuna
from torch import nn
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoConfig, 
    AutoModel, 
    Trainer, 
    TrainingArguments, 
    EarlyStoppingCallback
)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import precision_recall_fscore_support

# ==========================================
# 1. MODEL ARCHITECTURE & UTILS
# ==========================================

class CustomRobertaNetwork(nn.Module):
    """
    Custom RoBERTa architecture that extracts the penultimate hidden layer 
    for classification to improve feature representation.
    """
    def __init__(self, model_name, num_labels=2):
        super(CustomRobertaNetwork, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.roberta = AutoModel.from_pretrained(model_name, config=self.config)
        
        self.custom_network = nn.Sequential(
            nn.Linear(self.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Access penultimate layer from hidden_states
        second_to_last_layer = outputs.hidden_states[-2]
        cls_embedding = second_to_last_layer[:, 0, :]
        logits = self.custom_network(cls_embedding)
        
        return SequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def get_llrd_optimizer(model, base_lr=2e-5, weight_decay=0.01, decay_factor=0.95):
    """
    Implements Layer-wise Learning Rate Decay (LLRD) to mitigate 
    catastrophic forgetting in transformer layers.
    """
    opt_parameters = []
    opt_parameters.append({
        "params": model.custom_network.parameters(), 
        "lr": base_lr, 
        "weight_decay": weight_decay
    })
    
    layers = [model.roberta.embeddings] + list(model.roberta.encoder.layer)
    layers.reverse()
    lr = base_lr
    for layer in layers:
        lr *= decay_factor
        opt_parameters.append({
            "params": layer.parameters(), 
            "lr": lr, 
            "weight_decay": weight_decay
        })
    return torch.optim.AdamW(opt_parameters)

def compute_metrics(pred):
    labels = pred.label_ids
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    preds = logits.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    return {'f1': f1, 'precision': precision, 'recall': recall}

# ==========================================
# 2. DATA PROCESSING
# ==========================================

df = pd.read_csv("dontpatronizeme_pcl.tsv", sep='\t', header=None, 
                 names=['par_id', 'art_id', 'keyword', 'country', 'text', 'label'], 
                 on_bad_lines='skip').dropna(subset=['text', 'label'])

# Map labels: [0,1] -> 0 (Negative), [2,3,4] -> 1 (Positive)
df['label'] = df['label'].astype(int).apply(lambda x: 1 if x >= 2 else 0)

train_ids_df = pd.read_csv('train_semeval_parids-labels.csv')
full_train_df = df[df['par_id'].isin(train_ids_df['par_id'])].copy()

# Stratified split for internal validation set
train_df, val_df = train_test_split(
    full_train_df, 
    test_size=0.15, 
    random_state=42, 
    stratify=full_train_df['label']
)

train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
val_dataset = Dataset.from_pandas(val_df[['text', 'label']])

model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_fn(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_train = train_dataset.map(tokenize_fn, batched=True)
tokenized_val = val_dataset.map(tokenize_fn, batched=True)

# ==========================================
# 3. HPO & TRAINER CUSTOMIZATION
# ==========================================

def model_init():
    model = CustomRobertaNetwork(model_name, num_labels=2)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model

class HPOTrainer(Trainer):
    """
    Extends Trainer to support dynamic class weighting and LLRD optimizer 
    integration during hyperparameter optimization.
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        
        dynamic_pcl_weight = getattr(self.args, "pcl_weight", 3.0)
        weight_tensor = torch.tensor([1.0, dynamic_pcl_weight], dtype=torch.float).to(labels.device)
        
        loss_fct = nn.CrossEntropyLoss(weight=weight_tensor)
        loss = loss_fct(outputs.logits.view(-1, 2), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        self.optimizer = get_llrd_optimizer(
            self.model, 
            base_lr=self.args.learning_rate, 
            weight_decay=self.args.weight_decay
        )
        return self.optimizer

# ==========================================
# 4. HYPERPARAMETER SEARCH
# ==========================================

training_args = TrainingArguments(
    output_dir="/vol/bitbucket/aam223/ensemble_temp",
    eval_strategy="epoch",        
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=1,
    disable_tqdm=False
)

def objective(trial):
    lr = trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True)
    batch_size = trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32])
    wd = trial.suggest_float("weight_decay", 0.01, 0.1)
    pcl_weight = trial.suggest_float("pcl_weight", 1.5, 5.0)
    
    training_args.learning_rate = lr
    training_args.num_train_epochs = 8
    training_args.per_device_train_batch_size = batch_size
    training_args.weight_decay = wd
    training_args.pcl_weight = pcl_weight 
    
    trainer = HPOTrainer(
        model=model_init(),
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    trainer.train()
    metrics = trainer.evaluate()
    return metrics["eval_f1"]

print("Initializing Optuna study: 10 trials")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

# ==========================================
# 5. FINAL ENSEMBLE TRAINING
# ==========================================

print("Hyperparameter search complete. Training top 3 ensemble members.")

valid_trials = [t for t in study.trials if t.value is not None]
top_3_trials = sorted(valid_trials, key=lambda t: t.value, reverse=True)[:3]

for i, trial in enumerate(top_3_trials):
    print(f"Training ensemble member {i+1} (Target F1: {trial.value:.4f})")
    
    training_args.learning_rate = trial.params["learning_rate"]
    training_args.per_device_train_batch_size = trial.params["per_device_train_batch_size"]
    training_args.weight_decay = trial.params["weight_decay"]
    training_args.pcl_weight = trial.params["pcl_weight"]
    
    final_trainer = HPOTrainer(
        model=model_init(),
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics
    )
    final_trainer.train()
    
    # 
    save_path = f"ensemble_model_{i+1}"
    final_trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model {i+1} saved to {save_path}")

print("Training pipeline complete.")