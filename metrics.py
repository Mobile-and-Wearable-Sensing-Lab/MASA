import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.data import DataLoader, random_split

# MASA specific imports
from moco.builder_dist import MASA
from feeder.single_dataset.ISLGoaNew import ISL_GOA

# ==========================================
# CONFIGURATION
# ==========================================
NUM_CLASSES = 500  
DATA_ROOT = "/home/nithin/Desktop/ISL_Goa_Data/MASA/Data/ISL_GOA"
CHECKPOINT_DIR = "/home/nithin/Desktop/ISL_Goa_Data/MASA/save_path"
EPOCHS_TO_EVALUATE = [0, 5, 9]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    return new_state_dict

def collate_fn_eval(batch):
    out = {"right": {}, "left": {}, "body": {}}
    labels = []
    target_T = 64  # Temporal normalization

    for item in batch:
        labels.append(item['label'])
        T_raw = item['right']['kp2d'].shape[0]
        # Masking required by MASA feeder logic
        item['right']['masked'] = torch.zeros((T_raw, 1), dtype=torch.float32)
        item['left']['masked'] = torch.zeros((T_raw, 1), dtype=torch.float32)
        item['body']['masked'] = torch.zeros((T_raw, 1), dtype=torch.float32)

        for part in ['right', 'left', 'body']:
            if 'vid_len' not in out[part]: out[part]['vid_len'] = []
            out[part]['vid_len'].append(torch.tensor(target_T, dtype=torch.long))

            for k, v in item[part].items():
                if k not in out[part]: out[part][k] = []
                # Temporal interpolation to 64 frames
                indices = np.linspace(0, max(0, v.shape[0] - 1), target_T).astype(int)
                out[part][k].append(v[indices] if v.shape[0] > 0 else torch.zeros((target_T,) + v.shape[1:]))

    for part in ['right', 'left', 'body']:
        for k in out[part].keys():
            out[part][k] = torch.stack(out[part][k])
    return out, torch.stack(labels)

class LinearClassifier(nn.Module):
    def __init__(self, pretrained_model, num_classes, feature_dim=128):
        super().__init__()
        self.encoder = pretrained_model.encoder_q
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, inputs):
        with torch.no_grad():
            feat, enc_mask = self.encoder(inputs)
            q, _, _ = self.encoder.predict_head(feat, inputs, enc_mask=enc_mask)
            q = F.normalize(q, dim=1)
        return self.fc(q)

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print(f"Initializing Dataset: {DATA_ROOT}...")
    dataset = ISL_GOA(data_root=DATA_ROOT, split="train") #
    ISL_GOA.__getitem__ = ISL_GOA.get_sample

    # 80/20 Train-Test split for Linear Probing
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn_eval)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn_eval)

    # Base model initialization with correct Queue Size
    base_model = MASA(skeleton_representation='graph-based', num_class=NUM_CLASSES, K=6912, mlp=True, pretrain=True)

    # --- STEP 1: t-SNE PROGRESSION ---
    fig, axes = plt.subplots(1, len(EPOCHS_TO_EVALUATE), figsize=(18, 6))
    fig.suptitle('Evolution of Feature Embeddings (Filtered: 15 Sample Signs)', fontsize=16)

    for idx, epoch in enumerate(EPOCHS_TO_EVALUATE):
        path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{epoch:04d}.pth.tar")
        print(f"Processing Epoch {epoch}...")
        
        checkpoint = torch.load(path, map_location='cpu')
        base_model.load_state_dict(remove_module_prefix(checkpoint['state_dict']))
        base_model.to(DEVICE).eval()

        all_features, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                # Filter classes < 15 to visualize clear clusters
                mask = labels < 15
                if not mask.any(): continue

                for k, v in inputs.items():
                    for k_1, v_1 in v.items():
                        inputs[k][k_1] = v_1.float().to(DEVICE)

                feat, enc_mask = base_model.encoder_q(inputs)
                q, _, _ = base_model.encoder_q.predict_head(feat, inputs, enc_mask=enc_mask)
                q = F.normalize(q, dim=1)

                all_features.append(q[mask].cpu().numpy())
                all_labels.append(labels[mask].numpy())
                if len(np.concatenate(all_labels)) > 400: break

        features_2d = TSNE(n_components=2, perplexity=50, init='pca', random_state=42).fit_transform(np.concatenate(all_features))
        labels_plot = np.concatenate(all_labels)

        ax = axes[idx]
        ax.scatter(features_2d[:, 0], features_2d[:, 1], c=labels_plot, cmap='tab20', s=60, edgecolors = 'white', linewidth = 0.5, alpha=1.0)
        ax.set_title(f'Epoch {epoch}')
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig('tsne_evolution_refined.png', dpi=300)
    print("Saved: tsne_evolution_refined.png")

    # --- STEP 2: LINEAR PROBING (EPOCH 9) ---
    print("\n--- Training Linear Probe (Epoch 9) ---")
    linear_model = LinearClassifier(base_model, NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(linear_model.fc.parameters(), lr=0.005)
    
    for e in range(10):
        total_loss = 0
        linear_model.train()
        for inputs, labels in train_loader:
            labels = labels.to(DEVICE)
            for k, v in inputs.items():
                for k_1, v_1 in v.items(): inputs[k][k_1] = v_1.float().to(DEVICE)

            outputs = linear_model(inputs)
            loss = F.cross_entropy(outputs, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        print(f"Linear Probe Epoch {e+1}/10 | Loss: {total_loss/len(train_loader):.4f}")

    # --- STEP 3: FINAL CONFUSION MATRIX ---
    linear_model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            for k, v in inputs.items():
                for k_1, v_1 in v.items(): inputs[k][k_1] = v_1.float().to(DEVICE)
            out = linear_model(inputs)
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            targets.extend(labels.numpy())

    plt.figure(figsize=(12,10))

    sns.heatmap(confusion_matrix(targets, preds)[:25,:25], annot=True, cmap='Blues', fmt ='g', cbar = True)
    plt.title(f'Confusion Matrix (Acc: {accuracy_score(targets, preds)*100:.1f}%)')
    plt.savefig('final_confusion_matrix.png', dpi=300)
    print("Saved: final_confusion_matrix.png")

if __name__ == '__main__':
    main()
