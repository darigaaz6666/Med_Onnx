import numpy as np
import onnxruntime as ort
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from .data import make_loaders

def evaluate_onnx(model_path, X_test, y_test):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    out = sess.get_outputs()[0].name

    preds_all = []
    bs = 64
    for i in range(0, len(X_test), bs):
        xb = X_test[i:i+bs]
        logits = sess.run([out], {inp: xb})[0]
        preds_all.append(np.argmax(logits, axis=1))
    preds = np.concatenate(preds_all, axis=0)

    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds)
    cm  = confusion_matrix(y_test, preds)
    return acc, f1, cm

if __name__ == "__main__":
    # test set coerente con training
    _, test_loader = make_loaders(batch_size=64)

    X_test, y_test = [], []
    for xb, yb in test_loader:
        X_test.append(xb.numpy())
        y_test.append(yb.numpy())
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    acc32, f132, cm32 = evaluate_onnx("models/model_fp32.onnx", X_test, y_test)
    print("\n=== FP32 ===")
    print(f"Accuracy: {acc32:.3f} | F1: {f132:.3f}")
    print(cm32)

    acc8, f18, cm8 = evaluate_onnx("models/model_int8.onnx", X_test, y_test)
    print("\n=== INT8 ===")
    print(f"Accuracy: {acc8:.3f} | F1: {f18:.3f}")
    print(cm8)

    print(f"\nΔ accuracy (FP32-INT8): {acc32 - acc8:+.3f}")
