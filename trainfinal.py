import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Model
from tqdm import tqdm

# ─────────────────────────────────────────────
# 2. MODEL ARCHITECTURE
# ─────────────────────────────────────────────
class PianoTranscriptionModel(nn.Module):
    def __init__(self, pretrained_name=PRETRAINED_MODEL, num_classes=NUM_KEYS):
        super().__init__()
        # Load the pre-trained wav2vec2 model
        self.backbone = Wav2Vec2Model.from_pretrained(pretrained_name)
        
        # Freeze the backbone to speed up training and prevent forgetting
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Add a custom classification head mapping to 88 piano keys
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Extract features from audio
        outputs = self.backbone(x).last_hidden_state
        
        # Pool the features over the time dimension (average pooling)
        pooled_output = outputs.mean(dim=1)
        
        # Pass through our classifier to get predictions for the 88 keys
        logits = self.classifier(pooled_output)
        return logits

# ─────────────────────────────────────────────
# 3. TRAINING SETUP & LOOP
# ─────────────────────────────────────────────
def train():
    # Initialize Dataset and DataLoader (Assuming CSV and Audio are in './maestro')
    print("Loading Dataset...")
    train_dataset = MaestroDataset(csv_file='./maestro/maestro-v3.0.0.csv', root_dir='./maestro')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize Model, Loss function, and Optimizer
    print("Initializing Model...")
    model = PianoTranscriptionModel().to(DEVICE)
    
    # We use BCEWithLogitsLoss because multiple piano keys can be pressed at once (Multi-label classification)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    print("Starting Training Loop...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        # Progress bar for the epoch
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for waveforms, labels in progress_bar:
            # Move data to GPU/CPU
            waveforms = waveforms.to(DEVICE)
            labels = labels.to(DEVICE)

            # 1. Forward pass
            outputs = model(waveforms)
            loss = criterion(outputs, labels)

            # 2. Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        print(f"End of Epoch {epoch+1} | Average Loss: {avg_loss:.4f}\n")

if __name__ == "__main__":
    # train() # Uncomment to run the training loop
    print("Ready to train!")