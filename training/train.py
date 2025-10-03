import torch, json
from pathlib import Path
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTokenizer, CLIPTextModel
from .captions import load_captions

ENC_OUT = load_captions()
files = list(ENC_OUT.rglob("*.json"))
print("Training on", len(files), "items (mesh)")

class Mesh3DDataset(Dataset):
    def __init__(self, files, tokenizer, text_encoder, device):
        self.files = files
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device

    def __getitem__(self, i):
        meta = json.loads(self.files[i].read_text())
        text = meta.get("caption", "a 3d object")
        tokens = self.tokenizer(text, return_tensors="pt", padding="max_length",
                                truncation=True, max_length=64)
        with torch.no_grad():
            out = self.text_encoder(input_ids=tokens["input_ids"].to(self.device))[0][:,0,:].squeeze(0)
        latent = torch.randn(128) * 0.5
        return out, latent

    def __len__(self):
        return len(self.files)

class TinyCondNet(nn.Module):
    def __init__(self, in_dim=768, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, out_dim)
        )
    def forward(self, x): return self.net(x)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(device)

dataset = Mesh3DDataset(files[:40], tokenizer, text_encoder, device)
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

hidden_size = text_encoder.config.hidden_size
model = TinyCondNet(in_dim=hidden_size, out_dim=128).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = nn.MSELoss()

for epoch in range(10):
    model.train()
    tot = 0
    for txt_emb, latent in loader:
        txt_emb = txt_emb.to(device)
        latent = latent.to(device)
        pred = model(txt_emb)
        loss = loss_fn(pred, latent)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
    print(f"Epoch {epoch+1} loss {tot/len(loader):.4f}")

torch.save(model.state_dict(), "./text_cond.pt")
print("Saved text model.")
