# src/export_onnx.py
import os, argparse, numpy as np, torch, onnx, onnxruntime as ort
from onnx import checker, shape_inference
from .model import SimpleMLP

def export_onnx(pt_path="models/staging/best.pt", onnx_path="models/model_fp32.onnx"):
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    model = SimpleMLP(in_features=4, hidden=16, out_features=2)
    model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 4, dtype=torch.float32)

    # export più “stabile” per l’inferenza shape/quantizzazione
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["features"], output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
        opset_version=18  # 18 o 19 se la toolchain è aggiornata
    )
    print("[OK] Esportato:", onnx_path)

    # valida e riallinea le shape (salva di nuovo)
    m = onnx.load(onnx_path)
    checker.check_model(m)
    try:
        m = shape_inference.infer_shapes(m)   # strict=False di default
        onnx.save(m, onnx_path)
    except Exception as e:
        print("[WARN] infer_shapes post-export ha fallito (continua pure):", e)

    # verifica ORT
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 4).astype(np.float32)
    out = sess.run(["logits"], {"features": x})[0]
    print("Check ORT output shape:", out.shape)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="pt_path",   default="models/staging/best.pt")
    ap.add_argument("--out", dest="onnx_path", default="models/model_fp32.onnx")
    a = ap.parse_args()
    export_onnx(a.pt_path, a.onnx_path)
