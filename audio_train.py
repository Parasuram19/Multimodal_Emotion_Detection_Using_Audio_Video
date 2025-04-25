import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, Audio
import librosa
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the emotions mapping
emotions = {
    'a': 'anger',
    'd': 'disgust',
    'f': 'fear',
    'h': 'happiness',
    'n': 'neutral',
    'sa': 'sadness',
    'su': 'surprise'
}

# Function to load and preprocess the SAVEE dataset
def load_savee_dataset(root_dir="archive\AudioData\AudioData"):
    data = []
    
    # Walk through the directory structure
    for actor_dir in os.listdir(root_dir):
        actor_path = os.path.join(root_dir, actor_dir)
        if os.path.isdir(actor_path):
            for audio_file in os.listdir(actor_path):
                if audio_file.endswith('.wav'):
                    # Extract emotion label from the filename
                    emotion_code = ""
                    for code in emotions.keys():
                        if code in audio_file:
                            emotion_code = code
                            break
                    
                    if emotion_code:
                        data.append({
                            'path': os.path.join(actor_path, audio_file),
                            'emotion': emotions[emotion_code],
                            'actor': actor_dir,
                            'emotion_code': emotion_code
                        })
    
    return pd.DataFrame(data)

# Load the dataset
df = load_savee_dataset()
print(f"Dataset loaded with {len(df)} samples")
print(df.emotion.value_counts())

# Convert emotions to numeric labels
emotion_to_id = {emotion: idx for idx, emotion in enumerate(df.emotion.unique())}
id_to_emotion = {idx: emotion for emotion, idx in emotion_to_id.items()}
df['label'] = df.emotion.map(emotion_to_id)

# Split the dataset into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df.emotion)
print(f"Training set: {len(train_df)} samples")
print(f"Test set: {len(test_df)} samples")

# Function to load and preprocess audio files
def preprocess_audio(batch):
    # Load audio file
    audio_array, sampling_rate = librosa.load(batch["path"], sr=16000)
    
    # Ensure it's mono
    if len(audio_array.shape) > 1:
        audio_array = librosa.to_mono(audio_array)
    
    # Resample to 16kHz if not already
    if sampling_rate != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
    
    batch["audio"] = {"array": audio_array, "sampling_rate": 16000}
    return batch

# Convert pandas DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Apply preprocessing to the datasets
train_dataset = train_dataset.map(preprocess_audio, num_proc=4)
test_dataset = test_dataset.map(preprocess_audio, num_proc=4)

# Load the feature extractor
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")

# Function to prepare features for Wav2Vec2
def prepare_features(batch):
    audio = batch["audio"]["array"]
    
    # Extract features
    features = feature_extractor(
        audio, 
        sampling_rate=16000,
        padding=True,
        return_tensors="pt"
    )
    
    batch["input_values"] = features.input_values[0]
    batch["attention_mask"] = features.attention_mask[0]
    
    return batch

# Apply feature extraction
train_dataset = train_dataset.map(prepare_features, remove_columns=["audio"])
test_dataset = test_dataset.map(prepare_features, remove_columns=["audio"])

# Set the format of the datasets
train_dataset.set_format(type="torch", columns=["input_values", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_values", "attention_mask", "label"])

# Load the pre-trained Wav2Vec2 model
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=len(emotion_to_id),
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
)

# Freeze the feature extractor part
for param in model.wav2vec2.feature_extractor.parameters():
    param.requires_grad = False

# Move model to device
model = model.to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./wav2vec2-savee-emotion",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=10,
    fp16=torch.cuda.is_available(),
    save_steps=50,
    eval_steps=50,
    logging_steps=50,
    learning_rate=1e-4,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    warmup_steps=500,
    weight_decay=0.01,
    report_to="none",
)

# Define compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions, target_names=list(emotion_to_id.keys()), output_dict=True)
    
    # Focus on sadness metrics
    sadness_metrics = report.get('sadness', {})
    
    return {
        'accuracy': accuracy,
        'sadness_precision': sadness_metrics.get('precision', 0),
        'sadness_recall': sadness_metrics.get('recall', 0),
        'sadness_f1': sadness_metrics.get('f1-score', 0),
    }

# Define data collator
def data_collator(features):
    input_values = [feature["input_values"] for feature in features]
    attention_masks = [feature["attention_mask"] for feature in features]
    labels = [feature["label"] for feature in features]
    
    # Pad input values and attention masks
    input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
    
    return {
        "input_values": input_values,
        "attention_mask": attention_masks,
        "labels": torch.tensor(labels, dtype=torch.long),
    }

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
print("Starting training...")
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./wav2vec2-savee-emotion-final")
feature_extractor.save_pretrained("./wav2vec2-savee-emotion-final")

# Evaluate the model
print("Evaluating the model...")
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Get predictions on the test set
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids

# Display confusion matrix
cm = confusion_matrix(labels, preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=list(emotion_to_id.keys()), yticklabels=list(emotion_to_id.keys()))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

# Display classification report
print("Classification Report:")
print(classification_report(labels, preds, target_names=list(emotion_to_id.keys())))

# Function to predict emotion from a new audio file
def predict_emotion(audio_path):
    # Load and preprocess the audio
    audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
    if len(audio_array.shape) > 1:
        audio_array = librosa.to_mono(audio_array)
    if sampling_rate != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
    
    # Extract features
    inputs = feature_extractor(
        audio_array, 
        sampling_rate=16000,
        padding=True,
        return_tensors="pt"
    ).to(device)
    
    # Make prediction
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Get the predicted class
    predicted_class_idx = torch.argmax(logits, dim=-1).item()
    predicted_emotion = id_to_emotion[predicted_class_idx]
    
    # Get probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    emotion_probs = {id_to_emotion[i]: probs[i].item() for i in range(len(probs))}
    
    return predicted_emotion, emotion_probs

# Example usage of the prediction function
print("\nTesting prediction function on a sample from the test set:")
sample_audio_path = test_df.iloc[0]['path']
sample_true_emotion = test_df.iloc[0]['emotion']
predicted_emotion, emotion_probs = predict_emotion(sample_audio_path)

print(f"True emotion: {sample_true_emotion}")
print(f"Predicted emotion: {predicted_emotion}")
print("Emotion probabilities:")
for emotion, prob in emotion_probs.items():
    print(f"{emotion}: {prob:.4f}")