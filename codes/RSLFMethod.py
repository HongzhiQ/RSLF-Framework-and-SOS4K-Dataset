import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from tqdm import tqdm
import random
from test import test_model
import os
# ================================
#  Dividing line
# ================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)


def compute_metrics(preds, labels):
    """
    Compute accuracy and classification report.
    """
    accuracy = accuracy_score(labels, preds)
    report = classification_report(
        labels, preds, output_dict=True, zero_division=0,digits=4)
    # Print reports with pandas, retaining 4 decimal places
    report_df = pd.DataFrame(report).transpose().round(4)
    print("\nClassification Report:\n", report_df)
    return accuracy, report


# ================================
# Dataset
# ================================
#max_len默认300
class SuicideRiskDataset(Dataset):
    def __init__(self, dataframe, tokenizer, dictionary=None, embedder=None, max_len=400, use_dictionary=True,
                 use_explain=True):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.dictionary = dictionary
        self.embedder = embedder
        self.max_len = max_len
        self.use_dictionary = use_dictionary
        self.use_explain = use_explain
        self.categories = sorted(set(dictionary.values())) if dictionary else []

    def extract_dictionary_features(self, combined_text):
        """
        Extract dictionary features, based on dictionary category statistics; if the dictionary is not used, a zero vector is returned
        """
        if not self.use_dictionary or self.dictionary is None:
            return torch.zeros(len(set(self.dictionary.values())), dtype=torch.float)

        words = combined_text.split()
        category_counts = {category: 0 for category in set(
            self.dictionary.values())}
        for word in words:
            if word in self.dictionary:
                category_counts[self.dictionary[word]] += 1
        return torch.tensor(list(category_counts.values()), dtype=torch.float)

    def __getitem__(self, index):
        comment = str(self.data.iloc[index]['comment'])
        label = self.data.iloc[index]['myLabel']
        explain = str(self.data.iloc[index]['explain']
                      ) if 'explain' in self.data.columns else ""

        # Tokenize comment
        comment_encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        if self.use_explain:
            # Tokenize explain
            explain_encoding = self.tokenizer.encode_plus(
                explain,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            combined_text = f"{comment} {explain}"
        else:
            explain_encoding = None
            combined_text = comment

        # Extract dictionary features
        dict_features = self.extract_dictionary_features(combined_text)

        return {
            'comment_ids': comment_encoding['input_ids'].flatten(),
            'comment_mask': comment_encoding['attention_mask'].flatten(),
            'explain_ids': explain_encoding['input_ids'].flatten() if explain_encoding else None,
            'explain_mask': explain_encoding['attention_mask'].flatten() if explain_encoding else None,
            'dict_features': dict_features,
            'dict_embed_vector': self.embedder.encode_text(combined_text) if self.embedder else torch.zeros(
                len(self.categories) * 768),
            'targets': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.data)


# ================================
# Model with Freezing and Dropout
# ================================

class LexiconDistillationModule(nn.Module):
    def __init__(self, freq_dim=11, embed_dim=768, num_classes=11, hidden_dim=256):
        super().__init__()
        self.input_dim = freq_dim + (freq_dim * embed_dim)  # e.g., 11 + 11*768
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, freq_vec, embed_vec):
        # freq_vec: (batch_size, 11)
        # embed_vec: (batch_size, 11*768)
        x = torch.cat([freq_vec, embed_vec], dim=-1)
        return torch.softmax(self.mlp(x), dim=-1)



class CustomBERTModel(nn.Module):
    def __init__(self, model_path, num_labels, dictionary_size, dropout_prob=0.2,
                 freeze_layers=8, use_dictionary=True, use_explain=True):
        super(CustomBERTModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            model_path, num_labels=num_labels
        )
        self.use_dictionary = use_dictionary
        self.use_explain = use_explain
        self.num_labels = num_labels

        for name, param in self.bert.bert.named_parameters():
            if "encoder.layer" in name:
                layer_num = int(name.split(".")[2])
                if layer_num < freeze_layers:
                    param.requires_grad = False

        self.dropout = nn.Dropout(dropout_prob)
        self.dynamic_weight_layer = nn.Linear(768 * 2, 1)

        input_dim = 768 + dictionary_size if use_dictionary else 768
        self.dict_feature_layer = nn.Linear(input_dim, 768)
        self.classifier = nn.Linear(768, num_labels)


        self.auxiliary_branch = LexiconDistillationModule(freq_dim=dictionary_size,
                                                          embed_dim=768,
                                                          num_classes=num_labels)

    def forward(self, comment_ids, comment_mask, explain_ids=None, explain_mask=None,
                dict_features=None, dict_embed_vector=None, labels=None):

        comment_outputs = self.bert.bert(
            input_ids=comment_ids, attention_mask=comment_mask).pooler_output

        if self.use_explain and explain_ids is not None and explain_mask is not None:
            explain_outputs = self.bert.bert(
                input_ids=explain_ids, attention_mask=explain_mask).pooler_output
            combined_inputs = torch.cat([comment_outputs, explain_outputs], dim=-1)
            explain_weight = torch.sigmoid(self.dynamic_weight_layer(combined_inputs))
            final_output = explain_weight * comment_outputs + (1 - explain_weight) * explain_outputs
        else:
            final_output = comment_outputs

        if self.use_dictionary and dict_features is not None:
            combined_features = torch.cat([final_output, dict_features], dim=-1)
        else:
            combined_features = final_output

        # 主干输出
        combined_features = self.dict_feature_layer(combined_features)
        combined_features = self.dropout(combined_features)
        logits_main = self.classifier(combined_features)

        # ====== 辅助模块输出 soft 分布 ======
        if dict_embed_vector is not None:
            soft_risk_pred = self.auxiliary_branch(dict_features, dict_embed_vector)  # P_dict
        else:
            soft_risk_pred = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_ce = loss_fct(logits_main, labels)

            if soft_risk_pred is not None:
                prob_main = torch.softmax(logits_main, dim=-1)
                kl_loss = nn.functional.kl_div(prob_main.log(), soft_risk_pred.detach(), reduction='batchmean')
                loss = loss_ce + 0.2 * kl_loss
            else:
                loss = loss_ce

            return loss, logits_main, soft_risk_pred

        return logits_main, soft_risk_pred



from transformers import BertTokenizer, BertModel

class LexiconEmbedder:
    def __init__(self, model_path, dictionary):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.bert = BertModel.from_pretrained(model_path).eval()
        self.dictionary = dictionary
        self.categories = sorted(set(dictionary.values()))

    def encode_text(self, text):
        tokens = text.split()
        category_tokens = {cat: [] for cat in self.categories}
        for token in tokens:
            if token in self.dictionary:
                cat = self.dictionary[token]
                category_tokens[cat].append(token)

        # Take the average embedding for each class
        category_vectors = []
        for cat in self.categories:
            word_vecs = []
            for word in category_tokens[cat]:
                inputs = self.tokenizer(word, return_tensors='pt', truncation=True)
                with torch.no_grad():
                    output = self.bert(**inputs).last_hidden_state.mean(dim=1)
                word_vecs.append(output)
            if word_vecs:
                avg_vec = torch.mean(torch.stack(word_vecs), dim=0)  # [1, hidden]
            else:
                avg_vec = torch.zeros((1, self.bert.config.hidden_size))
            category_vectors.append(avg_vec)

        #  [1, 11*768]
        embed_vector = torch.cat(category_vectors, dim=-1).squeeze(0)
        return embed_vector  # shape: [11*768]





# ================================
# Training and Evaluation
# ================================

# ================================
def train_model(model, train_loader, val_loader, optimizer, scheduler, device, epochs=10):
    best_val_f1 = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_preds, total_targets = [], []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            comment_ids = batch['comment_ids'].to(device)
            comment_mask = batch['comment_mask'].to(device)
            explain_ids = batch['explain_ids'].to(device)
            explain_mask = batch['explain_mask'].to(device)
            dict_features = batch['dict_features'].to(device)
            dict_embed_vector = batch['dict_embed_vector'].to(device)
            targets = batch['targets'].to(device)

            optimizer.zero_grad()
            loss, logits, _ = model(
                comment_ids, comment_mask, explain_ids, explain_mask,
                dict_features, dict_embed_vector, labels=targets
            )
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            total_targets.extend(targets.cpu().numpy())

        # train metrics
        train_accuracy, _ = compute_metrics(total_preds, total_targets)
        print(f"Epoch {epoch+1} - Train Loss: {total_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}")

        # Validation
        val_loss, val_accuracy, val_f1 = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "best_model.pt")
            print(f"Best model saved at epoch {epoch+1}")

    print("Training complete!")

# ================================
# Validation function (supports KL module)
# ================================
def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    total_preds, total_targets = [], []

    with torch.no_grad():
        for batch in data_loader:
            comment_ids = batch['comment_ids'].to(device)
            comment_mask = batch['comment_mask'].to(device)
            explain_ids = batch['explain_ids'].to(device)
            explain_mask = batch['explain_mask'].to(device)
            dict_features = batch['dict_features'].to(device)
            dict_embed_vector = batch['dict_embed_vector'].to(device)
            targets = batch['targets'].to(device)

            loss, logits, _ = model(
                comment_ids, comment_mask, explain_ids, explain_mask,
                dict_features, dict_embed_vector, labels=targets
            )

            total_loss += loss.item()
            total_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            total_targets.extend(targets.cpu().numpy())

    accuracy, report = compute_metrics(total_preds, total_targets)
    val_f1 = report['weighted avg']['f1-score']

    print(f"Validation Loss: {total_loss / len(data_loader):.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation F1 Score: {val_f1:.4f}")

    avg_loss = total_loss / len(data_loader)
    return avg_loss, accuracy, val_f1

# ================================
# Main Script
# ================================
if __name__ == "__main__":
    set_seed(66)

    use_dictionary = True
    use_explain = True

    current_path = os.path.dirname(os.path.abspath(__file__))
    parent_path = os.path.dirname(current_path)


    data_path = os.path.join(parent_path, "data_final/second")
    train_data_path = os.path.join(data_path, "train_fold_2_withExplainFewShot.tsv")
    val_data_path = os.path.join(data_path, "val_fold_2_withExplainFewShot.tsv")
    test_data_path = os.path.join(data_path, "test_data_withExplainFewShot.tsv")
    dictionary_path = os.path.join(parent_path, "data_final/dict_final/Chinese suicide lexicon_unique_words.csv")

    train_data = pd.read_csv(train_data_path, sep='\t')
    val_data = pd.read_csv(val_data_path, sep='\t')
    test_data = pd.read_csv(test_data_path, sep='\t')

    def convert_to_binary_labels(data):
        data['myLabel'] = data['myLabel'].apply(lambda x: 0 if x <= 5 else 1)
        return data

    # Convert if needed
    # train_data = convert_to_binary_labels(train_data)
    # val_data = convert_to_binary_labels(val_data)
    # test_data = convert_to_binary_labels(test_data)

    dictionary_df = pd.read_csv(dictionary_path, header=None, names=['word', 'label'])
    dictionary = {row['word']: row['label'] for _, row in dictionary_df.iterrows()}
    dictionary_categories = list(set(dictionary.values()))

    model_path = os.path.join(parent_path, "codes/model")
    print(f"Model path: {model_path}")

    tokenizer = BertTokenizer.from_pretrained(model_path)


    embedder = LexiconEmbedder(model_path, dictionary)


    train_dataset = SuicideRiskDataset(train_data, tokenizer, dictionary,
                                       embedder=embedder, use_dictionary=use_dictionary, use_explain=use_explain)
    val_dataset = SuicideRiskDataset(val_data, tokenizer, dictionary,
                                     embedder=embedder, use_dictionary=use_dictionary, use_explain=use_explain)
    test_dataset = SuicideRiskDataset(test_data, tokenizer, dictionary,
                                      embedder=embedder, use_dictionary=use_dictionary, use_explain=use_explain)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomBERTModel(
        model_path=model_path,
        num_labels=11,
        dictionary_size=len(dictionary_categories),
        freeze_layers=8,
        dropout_prob=0.4,
        use_dictionary=use_dictionary
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.03)
    total_steps = len(train_loader) * 6
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=200, num_training_steps=total_steps
    )

    train_model(model, train_loader, val_loader,
                optimizer, scheduler, device, epochs=9)

    test_model(model, test_loader, device)
