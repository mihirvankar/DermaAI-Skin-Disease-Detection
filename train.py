# train.py
# ══════════════════════════════════════════════════════════════
# AI-Based Skin Disease Detection — Transfer Learning (MobileNet)
# Dataset  : HAM10000
# Run once : python train.py
# Output   : skin_model.h5
# ══════════════════════════════════════════════════════════════

import os, numpy as np, pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── CONFIG ─────────────────────────────────────────────────────
IMG_SIZE          = 128
BATCH_SIZE        = 16
EPOCHS            = 20
DATASET_DIR       = 'dataset'
SAMPLES_PER_CLASS = 1000

# Column names in your CSV
CLASS_COLS  = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
CLASS_NAMES = ['mel', 'nv',  'bcc', 'akiec', 'bkl', 'df', 'vasc']

print("=" * 60)
print("  AI Skin Disease Detection — MobileNet Transfer Learning")
print("=" * 60)

# ── STEP 1: Check GPU ──────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\n✅ GPU detected: {gpus[0].name}")
    tf.config.experimental.set_memory_growth(gpus[0], True)
else:
    print("\n⚠️  No GPU found — using CPU (will be slower)")

# ── STEP 2: Load CSV ───────────────────────────────────────────
csv_path = os.path.join(DATASET_DIR, 'HAM10000_metadata.csv')
if not os.path.exists(csv_path):
    print(f"\n❌ ERROR: {csv_path} not found!")
    exit()

df = pd.read_csv(csv_path)
print(f"\n✅ Metadata loaded — {len(df)} records")
print(f"   Columns: {df.columns.tolist()}")

# ── STEP 3: Convert one-hot to single label ────────────────────
# Your CSV has one-hot encoding: MEL, NV, BCC etc
# We convert it to a single label column called 'dx'
df['dx'] = df[CLASS_COLS].idxmax(axis=1)  # get column name with highest value
df['dx'] = df['dx'].str.lower()           # convert MEL -> mel, NV -> nv etc
print(f"   Class distribution:\n{df['dx'].value_counts().to_string()}")

# ── STEP 4: Find images ────────────────────────────────────────
image_dir    = os.path.join(DATASET_DIR, 'images')
df['path']   = df['image'].apply(lambda x: os.path.join(image_dir, x + '.jpg'))
df           = df[df['path'].apply(os.path.exists)].reset_index(drop=True)
print(f"\n✅ Images found on disk: {len(df)}")

df['label']  = df['dx'].map({c: i for i, c in enumerate(CLASS_NAMES)})
df           = df.dropna(subset=['label'])
df['label']  = df['label'].astype(int)

# ── STEP 5: Balance classes ────────────────────────────────────
print(f"\n⚖️  Balancing to {SAMPLES_PER_CLASS} samples per class...")
balanced = []
for cls in CLASS_NAMES:
    subset = df[df['dx'] == cls]
    if len(subset) == 0:
        print(f"   ⚠️  No samples found for class: {cls} — skipping")
        continue
    subset = resample(subset, replace=len(subset) < SAMPLES_PER_CLASS,
                      n_samples=SAMPLES_PER_CLASS, random_state=42)
    balanced.append(subset)
df_bal = pd.concat(balanced).reset_index(drop=True)
print(f"✅ Balanced dataset: {len(df_bal)} total samples")

# ── STEP 6: Load images into memory ───────────────────────────
print("\n📦 Loading images (this may take a few minutes)...")
X, y = [], []
for i, (_, row) in enumerate(df_bal.iterrows()):
    try:
        img = Image.open(row['path']).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        X.append(np.array(img, dtype='float32'))
        y.append(row['label'])
    except:
        pass
    if (i + 1) % 2000 == 0:
        print(f"   {i+1}/{len(df_bal)} images loaded...")

X = np.array(X) / 255.0
y = np.array(y)
print(f"✅ Loaded {len(X)} images — shape: {X.shape}")

# ── STEP 7: Train/val split ────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print(f"✅ Train: {len(X_train)} | Val: {len(X_val)}")

# ── STEP 8: Augmentation ───────────────────────────────────────
datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.15,
    shear_range=0.1
)
datagen.fit(X_train)

# ── STEP 9: Build MobileNetV2 model ───────────────────────────
print("\n🧠 Building MobileNetV2 Transfer Learning model...")

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

x      = base_model.output
x      = GlobalAveragePooling2D()(x)
x      = BatchNormalization()(x)
x      = Dense(256, activation='relu')(x)
x      = Dropout(0.4)(x)
x      = Dense(128, activation='relu')(x)
x      = Dropout(0.3)(x)
output = Dense(len(CLASS_NAMES), activation='softmax')(x)

model  = Model(inputs=base_model.input, outputs=output)
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print(f"✅ Model built — Total params: {model.count_params():,}")

# ── STEP 10: Phase 1 — Train top layers ───────────────────────
print("\n🚀 Phase 1: Training top layers (5 epochs)...")
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint('skin_model.h5', save_best_only=True, verbose=1),
    ReduceLROnPlateau(factor=0.3, patience=3, min_lr=1e-7, verbose=1)
]

model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val),
    epochs=5, callbacks=callbacks, verbose=1
)

# ── STEP 11: Phase 2 — Fine tune ──────────────────────────────
print("\n🔧 Phase 2: Fine-tuning (unfreezing top 30 layers)...")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val),
    epochs=EPOCHS, callbacks=callbacks, verbose=1
)

# ── STEP 12: Final results ─────────────────────────────────────
best_acc = max(history.history['val_accuracy']) * 100
print(f"\n{'='*60}")
print(f"  ✅ Training Complete!")
print(f"  Best Validation Accuracy : {best_acc:.2f}%")
print(f"  Model saved as           : skin_model.h5")
print(f"{'='*60}")
print("\n▶  Now run: python app.py")

# ── STEP 13: Save accuracy to file ────────────────────────────
import json
with open('model_info.json', 'w') as f:
    json.dump({
        'accuracy': round(best_acc, 2),
        'model'   : 'MobileNetV2',
        'dataset' : 'HAM10000',
        'classes' : 7,
        'img_size': IMG_SIZE
    }, f)
print(f"✅ Accuracy saved to model_info.json")
