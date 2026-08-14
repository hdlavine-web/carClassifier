Installation notes

1) Upgrade pip

```bash
python -m pip install --upgrade pip
```

2) CPU-only PyTorch (Windows example)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

3) Common GPU examples (select the exact command for your CUDA version at https://pytorch.org/get-started/locally/)

CUDA 11.8 (example):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

CUDA 12.1 (example):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

4) Install the rest of the Python packages

```bash
pip install -r requirements_clean.txt
```

5) Run the Streamlit app

```bash
streamlit run app_streamlit.py
```

Notes:
- `timm` will download pretrained weights when `pretrained=True` on first model creation; allow network access or set `pretrained=False`.
- If you want me to detect your CUDA version and give the exact command, tell me whether you want me to run a GPU probe on this machine.
