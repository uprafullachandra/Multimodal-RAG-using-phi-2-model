import torch
import faiss
from transformers import CLIPProcessor, CLIPModel, AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
from PIL import Image

# Load CLIP model for image-text embedding
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load small language model
llm_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

# Load text embedding model (smaller size)
text_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# FAISS index for text and image retrieval
text_index = faiss.IndexFlatIP(384)  # Using cosine similarity for better relevance
image_index = faiss.IndexFlatIP(512)  # Using cosine similarity for better relevance

# Store text and image data
text_data = []  # To map FAISS indices to text
image_data = []  # To map FAISS indices to image paths

SIMILARITY_THRESHOLD = 0.3  # Adjusted threshold for better filtering

def normalize_embedding(embedding):
    return embedding / np.linalg.norm(embedding)

def add_text_to_index(texts):
    embeddings = text_embedder.encode(texts, convert_to_numpy=True)
    normalized_embeddings = np.array([normalize_embedding(e) for e in embeddings])
    text_index.add(normalized_embeddings)
    text_data.extend(texts)
    return normalized_embeddings

def add_image_to_index(image_paths):
    for path in image_paths:
        try:
            image = Image.open(path).convert("RGB")
            inputs = clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                embedding = clip_model.get_image_features(**inputs).squeeze().numpy()
                normalized_embedding = normalize_embedding(embedding)
            image_index.add(np.expand_dims(normalized_embedding, axis=0))
            image_data.append(path)
        except FileNotFoundError:
            print(f"Warning: Image file '{path}' not found. Skipping.")

def retrieve_text(query, top_k=3):
    query_embedding = text_embedder.encode([query], convert_to_numpy=True)
    query_embedding = normalize_embedding(query_embedding[0]).reshape(1, -1)
    distances, indices = text_index.search(query_embedding, top_k)
    return [text_data[i] for i, d in zip(indices[0], distances[0]) if i < len(text_data) and d > SIMILARITY_THRESHOLD]

def retrieve_image(query, top_k=3):
    inputs = clip_processor(text=[query], return_tensors="pt")
    with torch.no_grad():
        query_embedding = clip_model.get_text_features(**inputs).squeeze().numpy()
        query_embedding = normalize_embedding(query_embedding)
    distances, indices = image_index.search(np.expand_dims(query_embedding, axis=0), top_k)
    return [image_data[i] for i, d in zip(indices[0], distances[0]) if i < len(image_data) and d > SIMILARITY_THRESHOLD]

def generate_response(context, query):
    input_text = f"Context: {context}\nQuery: {query}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt").to(llm_model.device)
    outputs = llm_model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example Usage:
texts = ["The Eiffel Tower is in Paris.", "Photosynthesis occurs in plants.", "AI is transforming industries."]
add_text_to_index(texts)

query = "Where is the Eiffel Tower?"
retrieved_texts = retrieve_text(query)
response = generate_response(" ".join(retrieved_texts), query)
print("Text Response:", response)

# Adding images (Example: Replace with actual image paths)
image_paths = ["z.jpg", "x.jpg", "y.jpg"]
add_image_to_index(image_paths)

image_query = "Find an image of the Eiffel Tower"
retrieved_images = retrieve_image(image_query)
print("Retrieved Images:", retrieved_images)
