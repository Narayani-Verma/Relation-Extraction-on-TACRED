from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score
import random
import pickle
import torch.nn as nn
from transformers import BertModel


def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def select_stratified_subset(data, subset_size):
    # Determine the distribution of relations
    relation_counts = {}
    for item in data:
        relation = item['relation']
        if relation not in relation_counts:
            relation_counts[relation] = []
        relation_counts[relation].append(item)

    # Calculate the number of items to sample per relation
    total_items = sum(len(items) for items in relation_counts.values())
    subset_counts = {relation: int(len(items) / total_items * subset_size) for relation, items in relation_counts.items()}

    # Ensure at least one sample per relation if subset size allows
    for relation in subset_counts:
        if subset_counts[relation] == 0 and subset_size > 0:
            subset_counts[relation] = 1
            subset_size -= 1  # Adjust subset_size for added samples

    # Sample items from each relation
    subset = []
    for relation, items in relation_counts.items():
        if subset_counts[relation] > 0:
            sampled_items = random.sample(items, subset_counts[relation])
            subset.extend(sampled_items)

    return subset


relation_types = ['org:city_of_headquarters', 'per:cities_of_residence', 'org:website', 'org:country_of_headquarters', 'per:origin', 'per:charges', 'org:parents', 'org:alternate_names', 'per:religion', 'per:stateorprovince_of_birth', 'per:other_family', 'org:members', 'org:shareholders', 'per:alternate_names', 'per:children', 'org:member_of', 'per:spouse', 'per:stateorprovinces_of_residence','per:title', 'per:city_of_death', 'per:age', 'per:date_of_death', 'per:country_of_birth', 'no_relation', 'org:number_of_employees/members','per:country_of_death', 'org:political/religious_affiliation', 'per:cause_of_death', 'per:city_of_birth', 'per:employee_of', 'org:dissolved', 'per:siblings', 'org:subsidiaries', 'per:schools_attended','per:date_of_birth', 'per:parents', 'org:top_members/employees', 'org:founded', 'per:stateorprovince_of_death', 'org:stateorprovince_of_headquarters', 'per:countries_of_residence', 'org:founded_by']  # Your list of relation types
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
label_encoder = LabelEncoder()
label_encoder.fit(relation_types)
def preprocess_data(data_item):
    # Tokenize the sentence
    tokens = data_item['token']
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
  
    
    # Create attention masks (assuming all tokens are relevant)
    attention_mask = [1] * len(input_ids)
    
    # Identify entity positions and create entity masks
    subj_positions = [0] * len(input_ids)
    obj_positions = [0] * len(input_ids)
    
    # Adjust for [CLS] token
    subj_start = data_item['subj_start'] 
    subj_end = data_item['subj_end'] 
    obj_start = data_item['obj_start'] 
    obj_end = data_item['obj_end'] 
    
    for i in range(subj_start, subj_end + 1):
        subj_positions[i] = 1
    for i in range(obj_start, obj_end + 1):
        obj_positions[i] = 1
    
    data_item['encoded_relation'] = label_encoder.transform([data_item['relation']])[0]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'subj_positions': subj_positions,
        'obj_positions': obj_positions,
        'labels': data_item['encoded_relation']  # You might need to convert this to a numerical ID
    }


class TACREDDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'subj_positions': torch.tensor(item['subj_positions'], dtype=torch.long),
            'obj_positions': torch.tensor(item['obj_positions'], dtype=torch.long),
            'labels': torch.tensor(item['labels'], dtype=torch.long)  # Adjust as necessary
        }
    

def collate_fn(batch):
    # Extracting input_ids, attention_mask, subj_positions, obj_positions, and labels
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    subj_positions = [item['subj_positions'] for item in batch]
    obj_positions = [item['obj_positions'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Padding sequences so they are all the same length
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    subj_positions_padded = pad_sequence(subj_positions, batch_first=True, padding_value=-100)  # Use an appropriate padding value for positions
    obj_positions_padded = pad_sequence(obj_positions, batch_first=True, padding_value=-100)
    labels = torch.stack(labels)  # Assuming labels can be directly stacked

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'subj_positions': subj_positions_padded,
        'obj_positions': obj_positions_padded,
        'labels': labels
    }




class SubjectObjectAwareModel(nn.Module):
    def __init__(self, num_labels):
        super(SubjectObjectAwareModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Assuming the hidden size of BERT base model is 768
        self.hidden_size = 768
        self.num_labels = num_labels
        
        # Linear layers for subject and object
        self.subject_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.object_linear = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Final classifier
        self.classifier = nn.Linear(self.hidden_size * 3, num_labels)  # *3 for [CLS], subject, and object
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, token_type_ids, subject_positions, object_positions):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        # Apply position masks to get subject and object representations
        subject_output = (sequence_output * subject_positions.unsqueeze(-1)).sum(1) / subject_positions.sum(1, keepdim=True)
        object_output = (sequence_output * object_positions.unsqueeze(-1)).sum(1) / object_positions.sum(1, keepdim=True)

        # Pass through the respective linear layers
        subject_output = self.subject_linear(subject_output)
        object_output = self.object_linear(object_output)

        # Concatenate pooled output (CLS token) with subject and object representations
        concat_output = torch.cat((pooled_output, subject_output, object_output), dim=1)
        concat_output = self.dropout(concat_output)

        logits = self.classifier(concat_output)

        return logits



def evaluate_model(model, dataloader, device):
    loss_fn = nn.CrossEntropyLoss()
    model.eval()  # Set the model to evaluation mode
    predictions = []
    true_labels = []
    total_eval_loss = 0

    with torch.no_grad():  # Deactivate autograd for evaluation
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            subject_positions = batch['subj_positions'].to(device)
            object_positions = batch['obj_positions'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask, None, subject_positions, object_positions)
            
            # Calculate loss
            loss = loss_fn(outputs, labels)
            total_eval_loss += loss.item()
            
            # Convert logits to predictions
            _, preds = torch.max(outputs, dim=1)
            
            # Move preds and labels to CPU
            predictions.extend(preds.detach().cpu().numpy())
            true_labels.extend(labels.detach().cpu().numpy())

    # Calculate the average loss and accuracy
    avg_loss = total_eval_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)

    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")

    return true_labels,predictions
    