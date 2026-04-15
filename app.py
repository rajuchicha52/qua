import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
import faiss
from tqdm import tqdm
from PIL import Image
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_aer.primitives import EstimatorV2 as AerEstimator

# ──────────────────────────────────────────────────────────────
# 1. CONFIGURATION & DEVICE
# ──────────────────────────────────────────────────────────────
device = torch.device("cpu")

# Smart path detection
COLAB_PATH = "/content/drive/MyDrive/SOCOFing/Real"
LOCAL_PATH = os.path.join(os.getcwd(), "dataset", "Real")
DEFAULT_DATASET_PATH = COLAB_PATH if os.path.exists(COLAB_PATH) else LOCAL_PATH

# EER thresholds from latest training runs
Q_THRESHOLD = 0.50
C_THRESHOLD = 0.538

# GLOBAL STATE for FAISS
gallery_index = None
gallery_labels = []
gallery_filenames = []
active_gallery_path = DEFAULT_DATASET_PATH

# ──────────────────────────────────────────────────────────────
# 2. QUANTUM CIRCUIT SETUP
# ──────────────────────────────────────────────────────────────
num_qubits   = 4
qc           = QuantumCircuit(num_qubits)
input_params = [Parameter(f"x{i}") for i in range(12)]
weights_p    = [Parameter(f"w{i}") for i in range(24)]

# Encoding (3 params per qubit: RX, RY, RZ)
for i in range(4):
    qc.rx(input_params[3*i], i)
    qc.ry(input_params[3*i+1], i)
    qc.rz(input_params[3*i+2], i)

qc.cx(0,1); qc.cx(1,2); qc.cx(2,3); qc.cx(3,0)

# Variational Layers (2 layers of RY, RZ, RX = 24 params)
for layer in range(2):
    offset = layer * 12
    for i in range(4):
        qc.ry(weights_p[offset + 3*i], i)
        qc.rz(weights_p[offset + 3*i + 1], i)
        qc.rx(weights_p[offset + 3*i + 2], i)
    qc.cx(0,1); qc.cx(1,2); qc.cx(2,3); qc.cx(3,0)

observable = SparsePauliOp.from_list([
    ("ZIII", 1), ("IZII", 1), ("IIZI", 1), ("IIIZ", 1),
    ("ZZII", 0.5), ("IIZZ", 0.5)
])

qnn = EstimatorQNN(
    circuit=qc,
    estimator=AerEstimator(),
    observables=observable,
    input_params=input_params,
    weight_params=weights_p,
    input_gradients=False
)
quantum_layer = TorchConnector(qnn)

# ──────────────────────────────────────────────────────────────
# 3. MODEL DEFINITIONS
# ──────────────────────────────────────────────────────────────
def make_resnet(num_outputs=4):
    resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    for name, param in resnet.named_parameters():
        param.requires_grad = ('layer4' in name or 'fc' in name)
    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    resnet.conv1.weight.data = (
        models.resnet18(weights=ResNet18_Weights.DEFAULT)
        .conv1.weight.data.mean(dim=1, keepdim=True)
    )
    resnet.fc = nn.Linear(512, num_outputs)
    return resnet

class HybridQNNModel(nn.Module):
    def __init__(self, base_model, q_layer):
        super().__init__()
        self.base  = base_model
        self.qnn   = q_layer
        self.scale = nn.Tanh()

    def forward(self, x1, x2):
        f1, f2     = self.base(x1), self.base(x2)
        diff       = torch.abs(f1 - f2)
        diff_sc    = self.scale(diff) * torch.pi
        raw        = self.qnn(diff_sc)
        return ((raw + 5) / 10.0).squeeze()

class ClassicalSiamese(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.classical_head = nn.Sequential(
            nn.Linear(8, 128), nn.ReLU(), nn.Dropout(p=0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, x1, x2):
        f1, f2 = self.base(x1), self.base(x2)
        diff   = torch.abs(f1 - f2)
        multi  = f1 * f2
        return self.classical_head(torch.cat((diff, multi), dim=1)).squeeze()

# ──────────────────────────────────────────────────────────────
# 4. LOAD MODELS
# ──────────────────────────────────────────────────────────────
q_model = HybridQNNModel(make_resnet(num_outputs=12), quantum_layer).to(device)
try:
    q_model.load_state_dict(torch.load("hybrid_qnn_best.pth", map_location=device))
    Q_LOADED = True
    print("Quantum weights loaded successfully.")
except Exception as e:
    Q_LOADED = False
    print(f"Quantum weights missing or mismatch: {e}")
q_model.eval()

c_model = ClassicalSiamese(make_resnet(num_outputs=4)).to(device)
for param in c_model.base.layer2.parameters(): param.requires_grad = True
for param in c_model.base.layer3.parameters(): param.requires_grad = True
try:
    c_model.load_state_dict(torch.load("classic_siamese_best.pth", map_location=device))
    C_LOADED = True
    print("Classical weights loaded successfully.")
except Exception as e:
    C_LOADED = False
    print(f"Classical weights missing or mismatch: {e}")
c_model.eval()

# ──────────────────────────────────────────────────────────────
# 5. PREPROCESSING & UTILS
# ──────────────────────────────────────────────────────────────
def apply_clahe(img):
    img_np = np.array(img).astype(np.uint8)
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return Image.fromarray(clahe.apply(img_np))

val_transform = transforms.Compose([
    transforms.Lambda(apply_clahe),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def get_label(filename):
    return filename.split("__")[0]

def preprocess_single(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = img.convert("L")
    return val_transform(img).unsqueeze(0).to(device), apply_clahe(img)

# ──────────────────────────────────────────────────────────────
# 6. FAISS INDEXING CORE
# ──────────────────────────────────────────────────────────────
def build_gallery_index(path):
    global gallery_index, gallery_labels, gallery_filenames, active_gallery_path
    
    try:
        # Handle URL mistake
        if path.startswith("http"):
            return "❌ Error: Google Drive links are not supported locally. Please provide a local folder path (e.g., G:\\My Drive\\SOCOFing\\Real) or run this in Google Colab."

        if not os.path.exists(path):
            return f"❌ Error: Dataset path not found: {path}"
        
        active_gallery_path = path
        
        all_files = [f for f in os.listdir(path) if f.lower().endswith((".bmp", ".jpg", ".png"))]
        if not all_files:
            return "❌ Error: No images found in directory."
            
        # Build gallery (one image per subject)
        temp_gallery = {}
        for f in all_files:
            lbl = get_label(f)
            if lbl not in temp_gallery:
                temp_gallery[lbl] = f
        
        gallery_labels = list(temp_gallery.keys())
        gallery_filenames = [temp_gallery[l] for l in gallery_labels]
        
        # Extract embeddings
        q_model.eval()
        embeddings = []
        for fname in tqdm(gallery_filenames, desc="Building Gallery Index"):
            img_path = os.path.join(path, fname)
            img = Image.open(img_path).convert("L")
            img_t, _ = preprocess_single(img) # Ensure same preproc
            with torch.no_grad():
                emb = q_model.base(img_t).cpu().numpy()
            embeddings.append(emb[0])
            
        embeddings_np = np.array(embeddings, dtype=np.float32)
        gallery_index = faiss.IndexFlatL2(embeddings_np.shape[1])
        gallery_index.add(embeddings_np)
        
        return f"✅ SUCCESS: Indexed {len(gallery_labels)} subjects from {path}"
    except Exception as e:
        return f"❌ Indexing failed: {str(e)}"

# ──────────────────────────────────────────────────────────────
# 7. INFERENCE FUNCTIONS
# ──────────────────────────────────────────────────────────────
def predict_pair(img1, img2):
    if img1 is None or img2 is None:
        err = {"Error": "⚠️ Upload BOTH images."}
        return err, err, None, None

    t1, clahe1 = preprocess_single(img1)
    t2, clahe2 = preprocess_single(img2)

    with torch.no_grad():
        q_sim = q_model(t1, t2).item()
        c_raw = c_model(t1, t2).item()
        c_sim = torch.sigmoid(torch.tensor(c_raw)).item()

    q_match = q_sim >= Q_THRESHOLD
    c_match = c_sim >= C_THRESHOLD

    quantum_result = {
        "Model": "Hybrid 4-Qubit QNN",
        "Verdict": "MATCH" if q_match else "MISMATCH",
        "Score": round(q_sim, 4),
        "Threshold": Q_THRESHOLD,
        "Status": "Weights Loaded" if Q_LOADED else "Error Loading"
    }
    classical_result = {
        "Model": "Classical Siamese",
        "Verdict": "MATCH" if c_match else "MISMATCH",
        "Score": round(c_sim, 4),
        "Threshold": C_THRESHOLD,
        "Status": "Weights Loaded" if C_LOADED else "Error Loading"
    }
    return quantum_result, classical_result, clahe1, clahe2

def identify_query(query_img, top_k):
    global gallery_index, gallery_labels, gallery_filenames, active_gallery_path
    
    try:
        if query_img is None:
            return {"Error": "⚠️ Upload a query image first."}, ""
            
        # Check if index exists
        if gallery_index is None:
            return {"Error": "Gallery not indexed. Click 'Build Search Gallery' first."}, ""

        t_q, _ = preprocess_single(query_img)
        with torch.no_grad():
            q_emb = q_model.base(t_q).cpu().numpy().astype(np.float32)

        # Step 1: FAISS Search
        D, I = gallery_index.search(q_emb, int(top_k))
        topk_indices = I[0]
        faiss_dists = D[0]

        # Step 2: Quantum Re-ranking
        results = []
        quantum_scores = [] 
        for i, idx in enumerate(topk_indices):
            cand_fname = gallery_filenames[idx]
            cand_label = gallery_labels[idx]
            
            # Load candidate image
            cand_path = os.path.join(active_gallery_path, cand_fname)
            try:
                cand_img = Image.open(cand_path).convert("L")
                t_c, _ = preprocess_single(cand_img) 
                with torch.no_grad():
                    q_sim = q_model(t_q, t_c).item()
                quantum_scores.append(q_sim)
            except Exception as e:
                print(f"Error loading candidate {cand_fname}: {e}")
                quantum_scores.append(0.0)

            results.append({
                "rank": i + 1,
                "label": cand_label,
                "faiss_l2": round(float(faiss_dists[i]), 4),
                "q_score": round(float(quantum_scores[-1]), 4)
            })

        # Sort by quantum score for final decision
        results.sort(key=lambda x: x['q_score'], reverse=True)
        best = results[0]

        summary = {
            "Identified": best['label'],
            "Quantum_Score": best['q_score'],
            "Engine": f"FAISS+QNN"
        }
        
        # Build Table HTML (omitting full code for brevity in replace chunk, but will preserve logic)
        table_html = build_result_table(results) # I'll refactor the table builder to a helper
        return summary, table_html
        
    except Exception as e:
        return {"Error": f"ID Failed: {str(e)}"}, f"<p style='color:red;'>System Error: {str(e)}</p>"

def build_result_table(results):
    table_html = """
    <div style='font-family:Orbitron,monospace; color:#00e0ff; margin-top:16px;'>
    <table style='width:100%; border-collapse:collapse; font-size:0.75rem;'>
      <thead>
        <tr style='background:rgba(0,114,255,0.2);'>
          <th style='padding:8px; border:1px solid rgba(0,200,255,0.2);'>Rank</th>
          <th style='padding:8px; border:1px solid rgba(0,200,255,0.2);'>Identity</th>
          <th style='padding:8px; border:1px solid rgba(0,200,255,0.2);'>FAISS L2</th>
          <th style='padding:8px; border:1px solid rgba(0,200,255,0.2);'>Quantum Score</th>
        </tr>
      </thead><tbody>
    """
    for i, r in enumerate(results):
        bg = "rgba(0,200,255,0.08)" if i == 0 else "transparent"
        icon = "🏆" if i == 0 else f"#{i+1}"
        table_html += f"""
        <tr style='background:{bg}; color:white;'>
          <td style='padding:4px; border:1px solid rgba(0,200,255,0.15); text-align:center;'>{icon}</td>
          <td style='padding:4px; border:1px solid rgba(0,200,255,0.15); text-align:center;'>{r['label']}</td>
          <td style='padding:4px; border:1px solid rgba(0,200,255,0.15); text-align:center;'>{r['faiss_l2']:.4f}</td>
          <td style='padding:4px; border:1px solid rgba(0,200,255,0.15); text-align:center;'>{r['q_score']:.4f}</td>
        </tr>"""
    table_html += "</tbody></table></div>"
    return table_html

# ──────────────────────────────────────────────────────────────
# 8. UI STYLING (PREMIUM THEME)
# ──────────────────────────────────────────────────────────────
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500&display=swap');
body, .gradio-container {
    background: radial-gradient(ellipse at 15% 20%, #0d1b4b 0%, #060912 50%, #000000 100%) !important;
    font-family: 'Inter', sans-serif; color: #ecf0f1 !important;
}
.tab-nav button {
    font-family:'Orbitron',monospace !important; letter-spacing:2px !important; color:#7dd3fc !important;
    background:transparent !important; border-bottom:2px solid transparent !important;
}
.tab-nav button.selected { color:#00e0ff !important; border-bottom:2px solid #00e0ff !important; }
.upload-panel {
    background:rgba(10,15,42,0.85) !important; border:1px solid rgba(0,224,255,0.4) !important;
    border-radius:14px !important; backdrop-filter:blur(20px) !important; padding:15px !important;
}
label, .label-wrap span { font-family:'Orbitron',monospace !important; color:#00e0ff !important; font-size:0.75rem !important; font-weight:700 !important; }
#verify-btn, #identify-btn, #build-btn {
    background:linear-gradient(135deg,#00c6ff 0%,#0072ff 50%,#a855f7 100%) !important;
    border:none !important; border-radius:10px !important; font-family:'Orbitron',monospace !important; font-weight:900 !important;
    color:#fff !important; transition:all 0.3s !important;
}
#verify-btn:hover { transform:scale(1.02); filter:brightness(1.1); }
#q-result, #c-result, #id-result {
    background:rgba(0,10,40,0.8) !important; border:1px solid rgba(0,200,255,0.2) !important;
    border-radius:14px !important; font-family:'Orbitron',monospace !important; color:#00e0ff !important;
}
.info-card { background:rgba(5,15,50,0.8); border:1px solid rgba(0,200,255,0.2); border-radius:14px; padding:15px; }
.info-card h3 { font-family:'Orbitron',monospace; color:#00e0ff; font-size:0.65rem; letter-spacing:2px; }
.section-label { font-family:'Orbitron',monospace; font-size:0.65rem; color:#00e0ff; text-align:center; margin:15px 0; }
"""

HEADER_HTML = """
<div style="text-align:center;padding:20px;">
  <h1 style="font-family:'Orbitron',monospace;font-weight:900;font-size:1.8rem;letter-spacing:4px;
      background:linear-gradient(90deg,#00e0ff,#0072ff,#a855f7);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
    ⚛️ QUANTUM FINGERPRINT AUTH MATRIX
  </h1>
  <p style="color:#7dd3fc;font-size:0.9rem;">Hybrid 4-Qubit QNN • ResNet18 • FAISS Retrieval</p>
</div>"""

# ──────────────────────────────────────────────────────────────
# 9. GRADIO UI
# ──────────────────────────────────────────────────────────────
with gr.Blocks(css=custom_css) as demo:
    gr.HTML(HEADER_HTML)
    
    with gr.Tabs():
        # TAB 1: 1:1 VERIFICATION
        with gr.Tab("⚛️ 1:1 VERIFICATION"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["upload-panel"]):
                    img1_in = gr.Image(type="pil", label="Base Fingerprint", height=220)
                with gr.Column(scale=1, elem_classes=["upload-panel"]):
                    img2_in = gr.Image(type="pil", label="Target Fingerprint", height=220)
            
            with gr.Row():
                clahe1_out = gr.Image(label="CLAHE A", height=180, interactive=False)
                clahe2_out = gr.Image(label="CLAHE B", height=180, interactive=False)

            submit_btn = gr.Button("⚛ INITIATE QUANTUM VERIFICATION", elem_id="verify-btn")
            
            with gr.Row():
                q_res = gr.JSON(label="Quantum Verdict", elem_id="q-result")
                c_res = gr.JSON(label="Classical Verdict", elem_id="c-result")

        # TAB 2: 1:N IDENTIFICATION
        with gr.Tab("🔍 1:N IDENTIFICATION"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("""<div class="info-card"><h3>⚙️ Identification Setup</h3>
                               <p style='font-size:0.7rem;'>First, mount your Drive and index the gallery.</p></div>""")
                    ds_path = gr.Textbox(value=DEFAULT_DATASET_PATH, label="Gallery Path (Google Drive)")
                    build_btn = gr.Button("🚀 BUILD SEARCH GALLERY", elem_id="build-btn")
                    build_status = gr.Markdown("Status: Gallery not indexed")
                
                with gr.Column(scale=1, elem_classes=["upload-panel"]):
                    query_in = gr.Image(type="pil", label="Query Fingerprint", height=220)
                    topk_sld = gr.Slider(1, 10, value=5, step=1, label="Top-K Candidates")
                    id_btn = gr.Button("🔍 IDENTIFY SUBJECT", elem_id="identify-btn")

            id_res = gr.JSON(label="🏆 Best Match", elem_id="id-result")
            rank_html = gr.HTML()

        # TAB 3: SYSTEM INFO
        with gr.Tab("📊 SYSTEM STATS"):
            with gr.Row():
                gr.HTML("""<div class="info-card"><h3>⚛️ Circuit Architecture</h3>
                    <table><tr><td>Qubits</td><td>4</td></tr><tr><td>Backend</td><td>AerV2</td></tr></table></div>""")
                gr.HTML("""<div class="info-card"><h3>📈 Performance</h3>
                    <table><tr><td>Q-Acc</td><td>71%</td></tr><tr><td>C-Acc</td><td>70%</td></tr></table></div>""")

    # WIRE UP
    submit_btn.click(fn=predict_pair, inputs=[img1_in, img2_in], outputs=[q_res, c_res, clahe1_out, clahe2_out])
    build_btn.click(fn=build_gallery_index, inputs=[ds_path], outputs=[build_status])
    id_btn.click(fn=identify_query, inputs=[query_in, topk_sld], outputs=[id_res, rank_html])

if __name__ == "__main__":
    # Explicitly binding to 127.0.0.1 for local Windows access to avoid 'not reachable' issues
    demo.launch(
        share=True, 
        server_name="127.0.0.1", 
        server_port=7860
    )
