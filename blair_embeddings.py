import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm
import requests
from io import BytesIO
import numpy as np
import os
from datasets import load_dataset



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


# Set the device to MPS if available, else use CPU
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.to(device)
model.eval()

# Create the embeddings directory if it doesn't exist
os.makedirs("embeddings", exist_ok=True)

for domain in domains:
    print(f"Processing domain: {domain}")
    item_image_urls = {item['parent_asin']: item['images']['large'] for item in datasets[domain]['metadata'] if len(item['images']['large']) > 0}
    
    domain_dir = os.path.join("embeddings", domain)
    os.makedirs(domain_dir, exist_ok=True)

    # Generate CLIP embeddings for the images and save them to disk
    for item_id, image_urls in tqdm(item_image_urls.items(), desc=f"Generating CLIP embeddings for {domain}"):
        embedding_path = os.path.join(domain_dir, f"{item_id}.npy")
        
        # Check if the embedding already exists
        if os.path.exists(embedding_path):
            print(f"Embedding for item {item_id} already exists. Skipping.")
            continue

        try:
            all_image_features = []
            for image_url in image_urls:
                # Download the image from the URL
                response = requests.get(image_url)
                image = Image.open(BytesIO(response.content)).convert("RGB")

                # Preprocess the image
                inputs = processor(images=image, return_tensors="pt").to(device)

                # Generate CLIP embedding
                with torch.no_grad():
                    image_features = model.get_image_features(**inputs)
                    all_image_features.append(image_features.squeeze().cpu().numpy())

            # Aggregate the image embeddings (e.g., by averaging)
            if all_image_features:
                aggregated_image_features = np.mean(all_image_features, axis=0)

                # Save the aggregated embedding to disk
                np.save(embedding_path, aggregated_image_features)
        except Exception as e:
            print(f"Error processing item {item_id}: {str(e)}")
            continue
