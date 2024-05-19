import os
import numpy as np
import torch
import yaml
import pickle
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender.sasrec import SASRec
from recbole.trainer import Trainer
from recbole.utils import get_trainer

# Step 1: Data preparation
domains = ["All_Beauty", "Video_Games", "Baby_Products"]
datasets = {}
for domain in domains:
    datasets[domain] = {}
    datasets[domain]["reviews"] = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"0core_timestamp_w_his_{domain}", trust_remote_code=True)
    datasets[domain]['metadata'] = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{domain}", split="full", trust_remote_code=True)

def preprocess_metadata(examples):
    examples['features'] = [' '.join(features) for features in examples['features']]
    return examples

for domain in domains:
    datasets[domain]['metadata'] = datasets[domain]['metadata'].map(preprocess_metadata, batched=True, num_proc=4)

for domain in domains:
    items_with_images_ids = set(datasets[domain]['metadata'].filter(lambda example: len(example['images']) > 0)['parent_asin'])
    datasets[domain]['reviews'] = datasets[domain]['reviews'].filter(lambda example: example['parent_asin'] in items_with_images_ids)

# Step 2: Baseline Models Implementation
tokenizer = AutoTokenizer.from_pretrained("hyp1231/blair-roberta-base")
blair_model = AutoModel.from_pretrained("hyp1231/blair-roberta-base")
# blair_model = blair_model.to('mps') if torch.backends.mps.is_available() else blair_model.to('cpu')
# info_model = "mps" if torch.backends.mps.is_available() else "cpu"
blair_model = blair_model.to('cuda') if torch.cuda.is_available() else blair_model.to('cpu')
info_model = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {info_model}")

for domain in tqdm(domains, desc="Processing domains"):
    data_dir = f"dataset/{domain}"
    os.makedirs(data_dir, exist_ok=True)

    inter_path = os.path.join(data_dir, f"{domain}.inter")

    if os.path.exists(inter_path):
        print(f"Interaction dataset already exists for {domain}. Skipping creation.")
    else:
        df = datasets[domain]['reviews']['train'].to_pandas()
        # Ensure the .inter file has the correct headers
        header = "user_id:token\titem_id:token\trating:float\ttimestamp:float\thistory:token_seq"
        df.to_csv(inter_path, index=False, sep='\t', header=False)
        with open(inter_path, 'r') as original:
            data = original.read()
        with open(inter_path, 'w') as modified:
            modified.write(header + '\n' + data)

def extract_item_embeddings(item_ids, batch_size=1024):
    item_metadata = datasets[domain]['metadata'].filter(lambda example: example['parent_asin'] in item_ids)
    item_texts = [example['title'] + ' ' + ' '.join(example['features']) for example in item_metadata]
    item_embeddings = []
    for i in tqdm(range(0, len(item_texts), batch_size), desc="Extracting embeddings"):
        batch_texts = item_texts[i:i+batch_size]
        encoded_inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=64, return_tensors='pt')
        encoded_inputs = {k: v.to(blair_model.device) for k, v in encoded_inputs.items()}
        with torch.no_grad():
            batch_embeddings = blair_model(**encoded_inputs).last_hidden_state[:, 0]
            item_embeddings.append(batch_embeddings.cpu().numpy())
    item_embeddings = np.concatenate(item_embeddings)
    return dict(zip(item_ids, item_embeddings))

item_embeddings = {}
for domain in domains:
    embeddings_file = f"item_embeddings_{domain}.pkl"
    if os.path.exists(embeddings_file):
        print(f"Loading saved item embeddings for {domain}")
        with open(embeddings_file, "rb") as f:
            item_embeddings[domain] = pickle.load(f)
    else:
        print(f"Extracting item embeddings for {domain}")
        item_ids = set(datasets[domain]['metadata']['parent_asin'])
        item_embeddings[domain] = extract_item_embeddings(item_ids)
        with open(embeddings_file, "wb") as f:
            pickle.dump(item_embeddings[domain], f)

# Load the existing config.yaml
with open('config.yaml', 'r') as f:
    config_dict = yaml.safe_load(f)

# if torch.backends.mps.is_available():
#     config_dict['gpu_id'] = 'mps'
# else:
#     config_dict['gpu_id'] = '0'

if torch.cuda.is_available():
    gpu_id = '0'  # Set the desired GPU ID
    config_dict['gpu_id'] = gpu_id
    torch.cuda.set_device(int(gpu_id))
else:
    config_dict['gpu_id'] = '0'

config_dict['train_neg_sample_args'] = None

for domain in tqdm(domains, desc="Processing domains"):
    dataset_path = f"dataset/{domain}"
    config_dict["dataset"] = domain

    # Create Config object
    config = Config(model="SASRec", config_dict=config_dict)
    print(f"Config for {domain}: {config}")

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    class BLAIRSASRec(SASRec):
        def __init__(self, config, dataset):
            super().__init__(config, dataset)
            self.item_embeddings = torch.tensor(np.array(list(item_embeddings[domain].values())), dtype=torch.float32).to(config['device'])
            self.item_embedding_size = self.item_embeddings.shape[1]
            self.position_embedding = torch.nn.Embedding(config['MAX_ITEM_LIST_LENGTH'], self.item_embedding_size)
            self.trm_encoder = torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(d_model=self.item_embedding_size, nhead=config['n_heads']),
                num_layers=config['num_layers']
            )

        def forward(self, item_seq, item_seq_len):
            item_emb = self.item_embeddings[item_seq].to(self.device)
            position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device).unsqueeze(0).expand(item_seq.size(0), -1)
            position_embedding = self.position_embedding(position_ids)
            seq_emb = item_emb + position_embedding
            seq_emb = self.dropout(seq_emb)

            mask = (torch.arange(seq_emb.size(1), device=item_seq.device).unsqueeze(0).expand(seq_emb.size(0), -1) < item_seq_len.unsqueeze(-1))
            seq_emb *= mask.unsqueeze(-1)

            batch_size = seq_emb.size(0)
            x = seq_emb.transpose(0, 1)  # Transpose to shape [seq_len, batch_size, embed_size]
            mask = ~mask  # Invert the mask
            seq_output = self.trm_encoder(x, src_key_padding_mask=mask)
            seq_output = seq_output.transpose(0, 1)  # Transpose back to shape [batch_size, seq_len, embed_size]

            return seq_output

    model = BLAIRSASRec(config, train_data.dataset).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
    test_result = trainer.evaluate(test_data)

    print(f"Domain: {domain}")
    print(f"Best valid result: {best_valid_result}")
    print(f"Test result: {test_result}")
