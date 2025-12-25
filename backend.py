from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import base64
from io import BytesIO
from PIL import Image
import json
import os

app = Flask(__name__)
CORS(app)

# ----------------------------------------
# CONFIGURATION
# ----------------------------------------
# Updated to local path
MODEL_PATH = "D:\BML_website\checkpoint_fold3_epoch20_epoch.pth"
IMG_SIZE = 224
NUM_CLASSES = 4
MC_SAMPLES = 20
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['No Dementia', 'Very Mild Dementia', 'Mild Dementia', 'Moderate Dementia']

# ----------------------------------------
# BAYESIAN LAYERS
# ----------------------------------------
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_sigma=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.w_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.b_mu = nn.Parameter(torch.empty(out_features))
        self.b_rho = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_normal_(self.w_mu)
        nn.init.constant_(self.w_rho, -5.0)
        nn.init.constant_(self.b_mu, 0.0)
        nn.init.constant_(self.b_rho, -5.0)
        self.prior_sigma = prior_sigma
        
    def forward(self, x):
        w_sigma = torch.log1p(torch.exp(self.w_rho))
        b_sigma = torch.log1p(torch.exp(self.b_rho))
        if self.training:
            w = self.w_mu + w_sigma * torch.randn_like(w_sigma)
            b = self.b_mu + b_sigma * torch.randn_like(b_sigma)
        else:
            w = self.w_mu
            b = self.b_mu
        return F.linear(x, w, b)
    
    def kl_loss(self):
        w_sigma = torch.log1p(torch.exp(self.w_rho))
        b_sigma = torch.log1p(torch.exp(self.b_rho))
        kl = torch.sum(math.log(self.prior_sigma) - torch.log(w_sigma) + 
                       (w_sigma**2 + self.w_mu**2) / (2 * self.prior_sigma**2) - 0.5)
        kl += torch.sum(math.log(self.prior_sigma) - torch.log(b_sigma) + 
                        (b_sigma**2 + self.b_mu**2) / (2 * self.prior_sigma**2) - 0.5)
        return kl

class BayesianAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = BayesianLinear(dim, dim * 3)
        self.proj = BayesianLinear(dim, dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x, attn

class BayesianMlp(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU):
        super().__init__()
        self.fc1 = BayesianLinear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = BayesianLinear(hidden_features, in_features)
        
    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.act(x1)
        x3 = self.fc2(x2)
        return x3, x1, x2

class BayesianBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = BayesianAttention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = BayesianMlp(dim, int(dim * mlp_ratio))
        
    def forward(self, x):
        normed1 = self.norm1(x)
        attn_out, attn_weights = self.attn(normed1)
        x = x + attn_out
        normed2 = self.norm2(x)
        mlp_out, mlp_hidden, mlp_act = self.mlp(normed2)
        x = x + mlp_out
        return x, attn_weights, mlp_hidden

class BayesianViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=NUM_CLASSES, 
                 embed_dim=256, depth=6, num_heads=4):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)
        self.blocks = nn.ModuleList([BayesianBlock(dim=embed_dim, num_heads=num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = BayesianLinear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            
    def forward_with_intermediates(self, x):
        B = x.shape[0]
        
        # Patch embedding
        patches = self.patch_embed(x)
        patches_flat = patches.flatten(2).transpose(1, 2)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, patches_flat), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Store intermediates
        block_outputs = []
        attention_maps = []
        mlp_hiddens = []
        
        for block in self.blocks:
            x, attn_weights, mlp_hidden = block(x)
            block_outputs.append(x.clone())
            attention_maps.append(attn_weights.detach())
            mlp_hiddens.append(mlp_hidden.detach())
        
        # Final classification
        cls_output = self.norm(x)[:, 0]
        logits = self.head(cls_output)
        
        return {
            'patches': patches.detach(),
            'patches_flat': patches_flat.detach(),
            'cls_token': cls_tokens.detach(),
            'with_pos': (x - self.pos_drop(x)).detach(),
            'block_outputs': block_outputs,
            'attention_maps': attention_maps,
            'mlp_hiddens': mlp_hiddens,
            'cls_output': cls_output.detach(),
            'logits': logits.detach()
        }
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x, _, _ = block(x)
        return self.head(self.norm(x)[:, 0])

# ----------------------------------------
# LOAD MODEL
# ----------------------------------------
print("Loading model...")
model = BayesianViT(num_classes=NUM_CLASSES)
if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(DEVICE)
    model.eval()
    print(f"✓ Model loaded on {DEVICE}")
else:
    print(f"✗ Model file not found at {MODEL_PATH}")
    print("Please check the path and try again.")

# ----------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------
def preprocess_image(image_data):
    """Preprocess base64 image"""
    img = Image.open(BytesIO(base64.b64decode(image_data.split(',')[1])))
    img = img.convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_tensor = torch.from_numpy(img_array).float().permute(2, 0, 1).unsqueeze(0)
    return img_tensor

def extract_patches_visual(img_tensor, patch_size=16):
    """Extract visual patches from image"""
    B, C, H, W = img_tensor.shape
    patches = []
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            patch = img_tensor[:, :, i:i+patch_size, j:j+patch_size]
            patches.append(patch.squeeze(0).permute(1, 2, 0).cpu().numpy())
    return patches

# ----------------------------------------
# API ENDPOINTS
# ----------------------------------------
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        image_data = data['image']
        
        print("Processing image...")
        
        # Preprocess
        img_tensor = preprocess_image(image_data).to(DEVICE)
        
        # Get detailed intermediates and stats
        with torch.no_grad():
            # 1. Standard Forward Pass (Mean weights) for main visualization
            model.eval()
            intermediates = model.forward_with_intermediates(img_tensor)
            
            # 2. Extract Bayesian Stats (from first block QKV)
            # We pick one specific weight to visualize its distribution
            qkv = model.blocks[0].attn.qkv
            # Pick a random index (e.g., middle of the matrix)
            row, col = 128, 128 
            w_mu_val = qkv.w_mu[row, col].item()
            w_rho_val = qkv.w_rho[row, col].item()
            w_sigma_val = math.log(1 + math.exp(w_rho_val))
            
            # 3. MC Sampling (Switch to train mode to activate Bayesian noise)
            model.train()
            all_predictions = []
            sampled_weight_val = 0
            
            for i in range(MC_SAMPLES):
                # Capture one sampled value for visualization
                if i == 0:
                    param = qkv.w_mu[row, col] + math.log(1 + math.exp(qkv.w_rho[row, col])) * torch.randn(1).to(DEVICE)
                    sampled_weight_val = param.item()
                    
                logits = model(img_tensor)
                probs = F.softmax(logits, dim=1)
                all_predictions.append(probs)
            
            model.eval() # Switch back
            all_predictions = torch.stack(all_predictions, dim=0)
            mean_probs = all_predictions.mean(dim=0).squeeze(0)
            
        # Extract patches for visualization
        patches = extract_patches_visual(img_tensor)
        
        # Extract Projection Filters (First 64)
        # Shape: [768, 3, 16, 16] -> Take first 64, Normalize to 0-1 for frontend
        filters = model.patch_embed.weight.data[:64].clone().cpu()
        # Min-max normalize per filter
        for i in range(len(filters)):
            f_min, f_max = filters[i].min(), filters[i].max()
            filters[i] = (filters[i] - f_min) / (f_max - f_min + 1e-8)
        
        # Convert to list [64, 3, 16, 16]
        filter_list = filters.numpy().tolist()

        print(f"✓ Prediction complete: {CLASS_NAMES[mean_probs.argmax().item()]}")
        
        # Prepare response
        # Convert patches to base64 for frontend visualization
        patch_images = []
        for p in patches:
            # p is (16, 16, 3) float [0,1]
            p_uint8 = (p * 255).astype(np.uint8)
            im = Image.fromarray(p_uint8)
            buf = BytesIO()
            im.save(buf, format='PNG')
            b64_str = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')
            patch_images.append(b64_str)

        response = {
            'patches': [[float(val) for val in patch.flatten()[:100]] for patch in patches[:49]],
            'patch_images': patch_images, # New field
            'projection_filters': filter_list,
            'patch_embeddings': intermediates['patches_flat'][0, :50].cpu().numpy().tolist(),
            'attention_maps': [attn[0, 0, :50, :50].cpu().numpy().tolist() for attn in intermediates['attention_maps'][:3]],
            'bayesian_stats': {
                'mu': w_mu_val,
                'sigma': w_sigma_val,
                'sampled': sampled_weight_val
            },
            'block_outputs': [block[0, 0, :50].cpu().numpy().tolist() for block in intermediates['block_outputs'][:3]],
            'cls_output': intermediates['cls_output'][0].cpu().numpy().tolist(),
            'logits': intermediates['logits'][0].cpu().numpy().tolist(),
            'probabilities': mean_probs.cpu().numpy().tolist(),
            'predicted_class': mean_probs.argmax().item(),
            'predicted_label': CLASS_NAMES[mean_probs.argmax().item()],
            'confidence': float(mean_probs.max().item() * 100),
            'class_names': CLASS_NAMES,
            'patch_size': model.patch_size,
            'num_patches': len(patches)
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy', 
        'device': str(DEVICE),
        'model_loaded': True
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Starting Bayesian ViT Backend Server")
    print("="*50)
    print(f"Device: {DEVICE}")
    print("="*50 + "\n")
    
    # Run on 0.0.0.0 to match the user's manual change in App.jsx (likely running in a container or binding to all interfaces)
    # However, for local dev normally localhost is fine. User changed App.jsx to use specific IP.
    # We will bind to 0.0.0.0 to be safe.
    app.run(host='0.0.0.0', port=5500, debug=False)
