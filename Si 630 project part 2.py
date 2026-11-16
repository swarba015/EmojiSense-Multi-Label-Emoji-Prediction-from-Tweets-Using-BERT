import pandas as pd
import numpy as np
import emoji
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from transformers import TFBertModel, BertTokenizer



df = pd.read_pickle("/content/drive/MyDrive/nlp/train_balanced.pkl")
dev_df = pd.read_pickle("/content/drive/MyDrive/nlp/dev_balanced.pkl")
test_df = pd.read_pickle("/content/drive/MyDrive/nlp/test_balanced.pkl")


# Extract emojis
def extract_emojis(text):
    return [char for char in str(text) if char in emoji.EMOJI_DATA]

for df_item in [df, dev_df, test_df]:
    df_item['emojis'] = df_item['Text'].apply(extract_emojis)

# Filter empty emoji rows
train_df = df[df['emojis'].str.len() > 0].reset_index(drop=True)
dev_df = dev_df[dev_df['emojis'].str.len() > 0].reset_index(drop=True)
test_df = test_df[test_df['emojis'].str.len() > 0].reset_index(drop=True)




# Keep top 100 emojis
all_emojis = sum(train_df['emojis'].tolist(), [])
emoji_counts = pd.Series(all_emojis).value_counts().head(100)
common_emojis = set(emoji_counts.index)

for df_item in [train_df, dev_df, test_df]:
    df_item['filtered_emojis'] = df_item['emojis'].apply(lambda x: [e for e in x if e in common_emojis])
    df_item.drop(df_item[df_item['filtered_emojis'].str.len() == 0].index, inplace=True)



# MultiLabelBinarizer for labels
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_df['filtered_emojis'])
y_dev = mlb.transform(dev_df['filtered_emojis'])
y_test = mlb.transform(test_df['filtered_emojis'])

# Define num_labels based on the MLBinarizer output
num_labels = len(mlb.classes_)


# Text preprocessing ( doing again just formmy sanity check )
def preprocess_text(text):
    text = ''.join(char for char in text if char not in emoji.EMOJI_DATA)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()

for df_item in [train_df, dev_df, test_df]:
    df_item['cleaned_text'] = df_item['Text'].apply(preprocess_text)


# Load BERT
max_length = 128
batch_size = 24
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(texts):
    return tokenizer(texts.tolist(), padding='max_length', truncation=True, max_length=max_length, return_tensors='tf')

# Tokenize datasets
train_encodings = tokenize(train_df['cleaned_text'])
dev_encodings = tokenize(dev_df['cleaned_text'])
test_encodings = tokenize(test_df['cleaned_text'])

# Create TF datasets
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
dev_dataset = tf.data.Dataset.from_tensor_slices((dict(dev_encodings), y_dev)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)



threshold = int(0.05 * train_df.shape[0])  

def augment_text(text):
    augmentations = [
        text,  
        text.lower(),
        ' '.join(text.split(' ')[::-1]),
        ' '.join(random.sample(text.split(' '), len(text.split(' ')))),  # shuffle words
    ]
    return random.choice(augmentations)

# Augment rare emoji classes
rare_indices = np.where(np.sum(y_train, axis=0) < threshold)[0]
augmented_texts = []
augmented_labels = []

for idx, row in train_df.iterrows():
    if any(y_train[idx, rare_idx] == 1 for rare_idx in rare_indices):
        for _ in range(3):  
            augmented_texts.append(augment_text(row['cleaned_text']))
            augmented_labels.append(y_train[idx])

# Add augmented data
train_df_augmented = pd.DataFrame({'cleaned_text': augmented_texts})
y_train_augmented = np.array(augmented_labels)

# Tokenize augmented data
augmented_encodings = tokenize(train_df_augmented['cleaned_text'])

# Create dataset from augmented samples
augmented_dataset = tf.data.Dataset.from_tensor_slices((dict(augmented_encodings), y_train_augmented))

# Need to unbatch original dataset before concatenating
train_dataset_unbatched = train_dataset.unbatch()
combined_dataset = train_dataset_unbatched.concatenate(tf.data.AUTOTUNE)

# Then rebatch
train_dataset = combined_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# Define the F1 metric function
def f1_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.3, tf.float32)
    precision = tf.reduce_sum(y_true * y_pred) / (tf.reduce_sum(y_pred) + tf.keras.backend.epsilon())
    recall = tf.reduce_sum(y_true * y_pred) / (tf.reduce_sum(y_true) + tf.keras.backend.epsilon())
    f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
    return f1

# Define the model architecture with fine-tuning parameters
class BertEmojiModel(tf.keras.Model):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = TFBertModel.from_pretrained('bert-base-uncased')

        # Fine-tuning: freeze the first 8 layers of BERT
        for layer in self.bert.bert.encoder.layer[:8]:
            layer.trainable = False

        for layer in self.bert.bert.encoder.layer[8:]:
            layer.trainable = True

        # Make embeddings trainable
        self.bert.bert.embeddings.trainable = True

        self.pool_dropout = Dropout(0.1)
        self.avg_pool = GlobalAveragePooling1D()
        self.seq_dropout = Dropout(0.1)
        self.dense1 = Dense(512, activation='gelu')
        self.dropout1 = Dropout(0.2)
        self.dense2 = Dense(256, activation='gelu')
        self.dropout2 = Dropout(0.2)
        self.classifier = Dense(num_labels, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.bert(**inputs, return_dict=True)
        pooled = self.pool_dropout(x.pooler_output, training=training)
        avg = self.avg_pool(x.last_hidden_state)
        avg = self.seq_dropout(avg, training=training)
        combined = Concatenate()([pooled, avg])
        x = self.dense1(combined)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return self.classifier(x)

# Calculate class weights for imbalanced dataset
class_weights = {}
for i in range(num_labels):
    class_freq = np.sum(y_train[:, i]) / y_train.shape[0]
    class_weights[i] = max(1.0, 1.0 / (class_freq + 1e-5))

# Create an ensemble of models
models = []
thresholds_list = []  # To store thresholds for each model

# Define the F1 metric function
def f1_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.3, tf.float32)
    y_true = tf.cast(y_true, tf.float32) # Cast y_true to float32
    precision = tf.reduce_sum(y_true * y_pred) / (tf.reduce_sum(y_pred) + tf.keras.backend.epsilon())
    recall = tf.reduce_sum(y_true * y_pred) / (tf.reduce_sum(y_true) + tf.keras.backend.epsilon())
    f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
    return f1

for i in range(3):  # Train 3 models
    print(f"Training model {i+1}/3...")

    # Initialize model
    model = BertEmojiModel(num_labels)

    # Learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=3e-5,
        first_decay_steps=1000,
        t_mul=2.0,
        m_mul=0.9,
        alpha=1e-6
    )

    if i == 0:
        model.compile(
            optimizer=Adam(learning_rate=lr_schedule),
            loss='binary_crossentropy',
            metrics=[
                f1_metric,
                tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
    elif i == 1:
        model.compile(
            optimizer=Adam(learning_rate=2e-5),
            loss='binary_crossentropy',
            metrics=[
                f1_metric,
                tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
    else:
        model.compile(
            optimizer=Adam(learning_rate=3e-5),  
            loss='binary_crossentropy',
            metrics=[
                f1_metric,
                tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True),
        ModelCheckpoint(f'bert_emoji_model_{i}.keras', monitor='val_auc', mode='max', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    ]

    # Train with different data subsets for diversity
    if i < 2:
        model.fit(train_dataset, validation_data=dev_dataset, epochs=5, callbacks=callbacks)
    else:
        model.fit(train_dataset, validation_data=dev_dataset, epochs=6, callbacks=callbacks)  # Extra epoch

    models.append(model)

    # Calculate thresholds for this model
    thresholds = []
    for j in range(num_labels):
        freq = np.sum(y_train[:, j]) / y_train.shape[0]
        thresholds.append(max(0.1, 0.5 - 0.4 * (1 - freq)))

    thresholds_list.append(thresholds)

# Average the predictions from all models
print("Generating ensemble predictions...")
ensemble_preds = np.mean([model.predict(test_dataset) for model in models], axis=0)

# Average the thresholds too
avg_thresholds = np.mean(thresholds_list, axis=0)

# Apply thresholds to get binary predictions
y_preds_bin = np.zeros_like(ensemble_preds)
for i in range(num_labels):
    y_preds_bin[:, i] = (ensemble_preds[:, i] > avg_thresholds[i]).astype(int)


# Evaluate
y_true_bin = []
for _, y in test_dataset:
    y_true_bin.extend(y.numpy())

y_true_bin = np.array(y_true_bin)
macro_f1 = f1_score(y_true_bin[:len(y_preds_bin)], y_preds_bin, average='macro')
print(f"Ensemble Macro F1 Score: {macro_f1:.4f}")

# Function to predict emojis for a new text
def predict_emojis(text, top_k=5):
    text = preprocess_text(text)
    inputs = tokenizer(text, return_tensors='tf', padding='max_length', truncation=True, max_length=max_length)

    # Get predictions from all models and average them
    all_preds = []
    for model in models:
        all_preds.append(model(inputs=inputs, training=False).numpy()[0])

    # Average predictions
    logits = np.mean(all_preds, axis=0)

    # Apply thresholds and get predicted classes
    predicted_classes = []
    for i, score in enumerate(logits):
        if score > avg_thresholds[i]:  # Use class-specific threshold
            predicted_classes.append((mlb.classes_[i], float(score)))

    # Sort by confidence score and take top-k
    predicted_classes.sort(key=lambda x: x[1], reverse=True)
    return predicted_classes[:top_k]
