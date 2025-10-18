# src/quantize_onnx.py
import argparse, os, onnx
from onnx import shape_inference
from onnxruntime.quantization import quantize_dynamic, QuantType

def _clean_shapes(in_path: str, out_path: str):
    m = onnx.load(in_path)

    # ✅ Svuota le value_info intermedie in modo corretto
    # Opzione A:
    m.graph.ClearField("value_info")
    # (In alternativa: del m.graph.value_info[:] )

    # Prova a reinferire le shape (modo permissivo)
    try:
        m = shape_inference.infer_shapes(m)  # strict_mode=False di default
    except Exception as e:
        print("[WARN] infer_shapes nella fase di pulizia ha fallito:", e)

    onnx.save(m, out_path)
    print("[OK] Modello pulito:", out_path)

def quantize(fp32_path="models/model_fp32.onnx", int8_path="models/model_int8.onnx"):
    try:
        quantize_dynamic(
            model_input=fp32_path,
            model_output=int8_path,
            weight_type=QuantType.QInt8
        )
        print("[OK] Quantizzato:", int8_path)
        return
    except Exception as e:
        print("[WARN] Quantizzazione fallita, provo pulizia shape:", e)

    clean_path = os.path.splitext(fp32_path)[0] + "_clean.onnx"
    _clean_shapes(fp32_path, clean_path)

    # Riprova sulla versione pulita
    quantize_dynamic(
        model_input=clean_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8
    )
    print("[OK] Quantizzato (dopo pulizia):", int8_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="fp32_path", default="models/model_fp32.onnx")
    ap.add_argument("--out", dest="int8_path", default="models/model_int8.onnx")
    a = ap.parse_args()
    quantize(a.fp32_path, a.int8_path)
