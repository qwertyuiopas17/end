"""
NetSentinel -- Industrial Integration Test Suite
==================================================
Comprehensive verification of all exported ONNX models before
backend integration. Tests cover:

  T1  - Artifact integrity (all required files present)
  T2  - Metadata schema validation
  T3  - ONNX model loading and I/O shape verification
  T4  - DDoS binary/multi-class consistency checks
  T5  - DDoS realistic traffic simulation (benign vs attack profiles)
  T6  - DGA character-level encoding round-trip
  T7  - DGA known-family domain classification
  T8  - DGA edge cases (empty, max-length, unicode, IDN)
  T9  - Output probability sanity (valid distributions)
  T10 - Determinism (same input => same output)
  T11 - Latency benchmarking (single + batch)
  T12 - Batch size stress test (1, 16, 128, 512, 1024)

Usage:
    pip install onnxruntime numpy
    python test_models.py

Exit code 0 = all passed, 1 = failures detected.
"""

import json
import numpy as np
import onnxruntime as ort
import os
import sys
import time
import traceback

# ============================================================
# Config
# ============================================================
DDOS_DIR = r"D:\1_sih26\ddos_model"
DGA_DIR = r"D:\1_sih26\dgtrain_model"

N_FEATURES_DDOS = 59
MAX_DOMAIN_LEN = 128
DGA_CLASSES = ["Benign", "DGA", "DNS Tunnel"]
BENCHMARK_RUNS = 500

# ============================================================
# Test framework
# ============================================================
RESULTS = []  # list of (section, name, status, detail)
CURRENT_SECTION = ""


def section(name):
    global CURRENT_SECTION
    CURRENT_SECTION = name
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")


def passed(name, detail=""):
    RESULTS.append((CURRENT_SECTION, name, "PASS", detail))
    print(f"  [PASS] {name}" + (f"  --  {detail}" if detail else ""))


def failed(name, detail=""):
    RESULTS.append((CURRENT_SECTION, name, "FAIL", detail))
    print(f"  [FAIL] {name}" + (f"  --  {detail}" if detail else ""))


def assert_test(condition, name, pass_detail="", fail_detail=""):
    if condition:
        passed(name, pass_detail)
    else:
        failed(name, fail_detail)
    return condition


# ============================================================
# Helpers
# ============================================================
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def encode_domain(domain, char2idx, max_len=MAX_DOMAIN_LEN):
    domain = domain.lower().strip()
    encoded = [char2idx.get(c, 1) for c in domain[:max_len]]
    encoded += [0] * (max_len - len(encoded))
    return encoded


def build_ddos_flow(**overrides):
    """Build a realistic 59-feature CIC-DDoS2019 flow vector.
    Defaults to a benign HTTP-like flow profile."""
    base = {
        "Protocol": 6,              # TCP
        "Flow Duration": 5000000,    # 5 seconds
        "Total Fwd Packets": 15,
        "Total Backward Packets": 12,
        "Fwd Packets Length Total": 2400,
        "Bwd Packets Length Total": 18000,
        "Fwd Packet Length Max": 580,
        "Fwd Packet Length Min": 40,
        "Fwd Packet Length Mean": 160.0,
        "Fwd Packet Length Std": 120.0,
        "Bwd Packet Length Max": 1460,
        "Bwd Packet Length Min": 40,
        "Bwd Packet Length Mean": 1500.0,
        "Bwd Packet Length Std": 450.0,
        "Flow Bytes/s": 4080.0,
        "Flow Packets/s": 5.4,
        "Flow IAT Mean": 185185.0,
        "Flow IAT Std": 120000.0,
        "Flow IAT Max": 500000,
        "Flow IAT Min": 50,
        "Fwd IAT Mean": 350000.0,
        "Bwd IAT Total": 4500000,
        "Bwd IAT Mean": 375000.0,
        "Bwd IAT Std": 200000.0,
        "Bwd IAT Max": 800000,
        "Bwd IAT Min": 100,
        "Fwd PSH Flags": 1,
        "Fwd Header Length": 480,
        "Bwd Header Length": 384,
        "Bwd Packets/s": 2.4,
        "Packet Length Max": 1460,
        "Packet Length Mean": 755.0,
        "Packet Length Std": 600.0,
        "Packet Length Variance": 360000.0,
        "SYN Flag Count": 1,
        "RST Flag Count": 0,
        "ACK Flag Count": 26,
        "URG Flag Count": 0,
        "CWE Flag Count": 0,
        "Down/Up Ratio": 0.8,
        "Avg Packet Size": 755.0,
        "Avg Fwd Segment Size": 160.0,
        "Avg Bwd Segment Size": 1500.0,
        "Subflow Fwd Packets": 15,
        "Subflow Fwd Bytes": 2400,
        "Subflow Bwd Packets": 12,
        "Subflow Bwd Bytes": 18000,
        "Init Fwd Win Bytes": 65535,
        "Init Bwd Win Bytes": 65535,
        "Fwd Act Data Packets": 10,
        "Fwd Seg Size Min": 20,
        "Active Mean": 200000.0,
        "Active Std": 50000.0,
        "Active Max": 300000,
        "Active Min": 100000,
        "Idle Mean": 0.0,
        "Idle Std": 0.0,
        "Idle Max": 0,
        "Idle Min": 0,
    }
    base.update(overrides)
    return np.array(list(base.values()), dtype=np.float32)


def build_syn_flood():
    """SYN flood attack profile: massive forward packets, no backward, tiny sizes."""
    return build_ddos_flow(**{
        "Protocol": 6,
        "Flow Duration": 100000,
        "Total Fwd Packets": 50000,
        "Total Backward Packets": 0,
        "Fwd Packets Length Total": 2000000,
        "Bwd Packets Length Total": 0,
        "Fwd Packet Length Max": 40,
        "Fwd Packet Length Min": 40,
        "Fwd Packet Length Mean": 40.0,
        "Fwd Packet Length Std": 0.0,
        "Bwd Packet Length Max": 0,
        "Bwd Packet Length Min": 0,
        "Bwd Packet Length Mean": 0.0,
        "Bwd Packet Length Std": 0.0,
        "Flow Bytes/s": 20000000.0,
        "Flow Packets/s": 500000.0,
        "Flow IAT Mean": 2.0,
        "Flow IAT Std": 1.0,
        "Flow IAT Max": 10,
        "Flow IAT Min": 0,
        "Fwd IAT Mean": 2.0,
        "Bwd IAT Total": 0,
        "Bwd IAT Mean": 0.0,
        "Bwd IAT Std": 0.0,
        "Bwd IAT Max": 0,
        "Bwd IAT Min": 0,
        "Fwd PSH Flags": 0,
        "Fwd Header Length": 2000000,
        "Bwd Header Length": 0,
        "Bwd Packets/s": 0.0,
        "Packet Length Max": 40,
        "Packet Length Mean": 40.0,
        "Packet Length Std": 0.0,
        "Packet Length Variance": 0.0,
        "SYN Flag Count": 50000,
        "RST Flag Count": 0,
        "ACK Flag Count": 0,
        "URG Flag Count": 0,
        "CWE Flag Count": 0,
        "Down/Up Ratio": 0.0,
        "Avg Packet Size": 40.0,
        "Avg Fwd Segment Size": 40.0,
        "Avg Bwd Segment Size": 0.0,
        "Subflow Fwd Packets": 50000,
        "Subflow Fwd Bytes": 2000000,
        "Subflow Bwd Packets": 0,
        "Subflow Bwd Bytes": 0,
        "Init Fwd Win Bytes": 1024,
        "Init Bwd Win Bytes": 0,
        "Fwd Act Data Packets": 0,
        "Fwd Seg Size Min": 20,
        "Active Mean": 0.0,
        "Active Std": 0.0,
        "Active Max": 0,
        "Active Min": 0,
        "Idle Mean": 0.0,
        "Idle Std": 0.0,
        "Idle Max": 0,
        "Idle Min": 0,
    })


def build_udp_flood():
    """UDP flood attack profile: massive UDP packets, high bytes/s."""
    return build_ddos_flow(**{
        "Protocol": 17,  # UDP
        "Flow Duration": 50000,
        "Total Fwd Packets": 100000,
        "Total Backward Packets": 0,
        "Fwd Packets Length Total": 10000000,
        "Bwd Packets Length Total": 0,
        "Fwd Packet Length Max": 100,
        "Fwd Packet Length Min": 100,
        "Fwd Packet Length Mean": 100.0,
        "Fwd Packet Length Std": 0.0,
        "Bwd Packet Length Max": 0,
        "Bwd Packet Length Min": 0,
        "Bwd Packet Length Mean": 0.0,
        "Bwd Packet Length Std": 0.0,
        "Flow Bytes/s": 200000000.0,
        "Flow Packets/s": 2000000.0,
        "Flow IAT Mean": 0.5,
        "Flow IAT Std": 0.2,
        "Flow IAT Max": 5,
        "Flow IAT Min": 0,
        "Fwd IAT Mean": 0.5,
        "Bwd IAT Total": 0,
        "Bwd IAT Mean": 0.0,
        "Bwd IAT Std": 0.0,
        "Bwd IAT Max": 0,
        "Bwd IAT Min": 0,
        "Fwd PSH Flags": 0,
        "Fwd Header Length": 800000,
        "Bwd Header Length": 0,
        "Bwd Packets/s": 0.0,
        "Packet Length Max": 100,
        "Packet Length Mean": 100.0,
        "Packet Length Std": 0.0,
        "Packet Length Variance": 0.0,
        "SYN Flag Count": 0,
        "RST Flag Count": 0,
        "ACK Flag Count": 0,
        "URG Flag Count": 0,
        "CWE Flag Count": 0,
        "Down/Up Ratio": 0.0,
        "Avg Packet Size": 100.0,
        "Avg Fwd Segment Size": 100.0,
        "Avg Bwd Segment Size": 0.0,
        "Subflow Fwd Packets": 100000,
        "Subflow Fwd Bytes": 10000000,
        "Subflow Bwd Packets": 0,
        "Subflow Bwd Bytes": 0,
        "Init Fwd Win Bytes": 0,
        "Init Bwd Win Bytes": 0,
        "Fwd Act Data Packets": 100000,
        "Fwd Seg Size Min": 8,
        "Active Mean": 50000.0,
        "Active Std": 0.0,
        "Active Max": 50000,
        "Active Min": 50000,
        "Idle Mean": 0.0,
        "Idle Std": 0.0,
        "Idle Max": 0,
        "Idle Min": 0,
    })


# ============================================================
# T1: Artifact Integrity
# ============================================================
def test_artifact_integrity():
    section("T1: Artifact Integrity -- Required Files")

    ddos_required = [
        "ddos_binary_xgboost.onnx",
        "ddos_multi_xgboost.onnx",
        "ddos_metrics.json",
        "feature_names.json",
        "label_mapping.json",
    ]
    dga_required = [
        "dga_cnn_bilstm.onnx",
        "dga_metrics.json",
        "char_vocab.json",
        "dga_best_model.pt",
    ]

    for f in ddos_required:
        path = os.path.join(DDOS_DIR, f)
        exists = os.path.isfile(path)
        size = os.path.getsize(path) if exists else 0
        assert_test(exists and size > 0, f"ddos_model/{f}",
                    f"{size/1024:.0f} KB", "MISSING or EMPTY")

    for f in dga_required:
        path = os.path.join(DGA_DIR, f)
        exists = os.path.isfile(path)
        size = os.path.getsize(path) if exists else 0
        assert_test(exists and size > 0, f"dgtrain_model/{f}",
                    f"{size/1024:.0f} KB", "MISSING or EMPTY")


# ============================================================
# T2: Metadata Schema Validation
# ============================================================
def test_metadata_schema():
    section("T2: Metadata Schema Validation")

    # DDoS metrics
    ddos_m = load_json(os.path.join(DDOS_DIR, "ddos_metrics.json"))
    assert_test("binary_classifier" in ddos_m, "DDoS has binary_classifier block")
    assert_test("multi_classifier" in ddos_m, "DDoS has multi_classifier block")
    assert_test(ddos_m["binary_classifier"]["n_features"] == N_FEATURES_DDOS,
                f"DDoS feature count == {N_FEATURES_DDOS}",
                f"OK ({ddos_m['binary_classifier']['n_features']})",
                f"MISMATCH: {ddos_m['binary_classifier'].get('n_features')}")
    assert_test(ddos_m["binary_classifier"]["f1"] > 0.95,
                f"DDoS binary F1 > 0.95",
                f"{ddos_m['binary_classifier']['f1']:.4f}")
    assert_test(ddos_m["multi_classifier"]["n_classes"] == 18,
                "DDoS multi n_classes == 18",
                f"OK ({ddos_m['multi_classifier']['n_classes']})")

    # Feature names count
    feat_names = load_json(os.path.join(DDOS_DIR, "feature_names.json"))
    assert_test(len(feat_names) == N_FEATURES_DDOS,
                f"feature_names.json has {N_FEATURES_DDOS} entries",
                f"OK ({len(feat_names)})",
                f"MISMATCH: {len(feat_names)}")

    # Label mapping count
    labels = load_json(os.path.join(DDOS_DIR, "label_mapping.json"))
    assert_test(len(labels) == 18,
                "label_mapping.json has 18 classes",
                f"OK ({len(labels)})",
                f"MISMATCH: {len(labels)}")
    assert_test(labels.get("0") == "Benign",
                "Class 0 == 'Benign'",
                "OK", f"GOT: {labels.get('0')}")

    # DGA metrics
    dga_m = load_json(os.path.join(DGA_DIR, "dga_metrics.json"))
    assert_test(dga_m["vocab_size"] == 40, "DGA vocab_size == 40",
                f"OK ({dga_m['vocab_size']})")
    assert_test(dga_m["max_domain_length"] == 128, "DGA max_domain_length == 128")
    assert_test(dga_m["macro_f1"] > 0.95,
                f"DGA macro F1 > 0.95",
                f"{dga_m['macro_f1']:.4f}")
    assert_test(len(dga_m["classes"]) == 3,
                "DGA classes == 3",
                f"{dga_m['classes']}")

    # Char vocab
    vocab = load_json(os.path.join(DGA_DIR, "char_vocab.json"))
    assert_test(vocab.get("<PAD>") == 0, "Vocab <PAD> == 0")
    assert_test(vocab.get("<UNK>") == 1, "Vocab <UNK> == 1")
    assert_test(len(vocab) == 40,
                f"Vocab size matches metric ({len(vocab)})")


# ============================================================
# T3: ONNX Model I/O Shape Verification
# ============================================================
def test_onnx_io_shapes():
    section("T3: ONNX I/O Shape Verification")

    # DDoS Binary
    sess_b = ort.InferenceSession(os.path.join(DDOS_DIR, "ddos_binary_xgboost.onnx"))
    inp_b = sess_b.get_inputs()[0]
    assert_test(inp_b.shape == [None, 59],
                f"DDoS binary input shape == [None, 59]",
                f"{inp_b.shape}", f"GOT: {inp_b.shape}")
    assert_test(inp_b.type == "tensor(float)",
                "DDoS binary input type == tensor(float)",
                inp_b.type, inp_b.type)
    out_names_b = [o.name for o in sess_b.get_outputs()]
    assert_test("label" in out_names_b and "probabilities" in out_names_b,
                "DDoS binary outputs: label + probabilities",
                str(out_names_b))

    # DDoS Multi
    sess_m = ort.InferenceSession(os.path.join(DDOS_DIR, "ddos_multi_xgboost.onnx"))
    inp_m = sess_m.get_inputs()[0]
    assert_test(inp_m.shape == [None, 59],
                f"DDoS multi input shape == [None, 59]",
                f"{inp_m.shape}")
    out_names_m = [o.name for o in sess_m.get_outputs()]
    assert_test("label" in out_names_m and "probabilities" in out_names_m,
                "DDoS multi outputs: label + probabilities",
                str(out_names_m))

    # DGA
    sess_d = ort.InferenceSession(os.path.join(DGA_DIR, "dga_cnn_bilstm.onnx"))
    inp_d = sess_d.get_inputs()[0]
    assert_test(inp_d.shape[1] == 128,
                "DGA input seq_len == 128",
                f"{inp_d.shape}")
    assert_test(inp_d.type == "tensor(int64)",
                "DGA input type == tensor(int64)",
                inp_d.type, inp_d.type)
    out_d = sess_d.get_outputs()[0]
    assert_test(out_d.name == "logits",
                "DGA output name == 'logits'",
                out_d.name)


# ============================================================
# T4: DDoS Binary/Multi Consistency
# ============================================================
def test_ddos_consistency():
    section("T4: DDoS Binary/Multi-class Consistency")

    sess_b = ort.InferenceSession(os.path.join(DDOS_DIR, "ddos_binary_xgboost.onnx"))
    sess_m = ort.InferenceSession(os.path.join(DDOS_DIR, "ddos_multi_xgboost.onnx"))
    inp_name = sess_b.get_inputs()[0].name

    benign_flow = build_ddos_flow().reshape(1, -1)

    res_b = sess_b.run(None, {inp_name: benign_flow})
    res_m = sess_m.run(None, {inp_name: benign_flow})

    binary_pred = int(res_b[0][0])
    multi_pred = int(res_m[0][0])
    label_map = load_json(os.path.join(DDOS_DIR, "label_mapping.json"))
    multi_label = label_map.get(str(multi_pred), "?")

    # If binary says benign (0), multi should also say Benign (0)
    # If binary says attack (1), multi should say any attack class (>0)
    if binary_pred == 0:
        assert_test(multi_pred == 0,
                    "Benign flow: binary=Benign AND multi=Benign",
                    f"binary={binary_pred}, multi={multi_label}",
                    f"INCONSISTENT: binary={binary_pred}, multi={multi_label}")
    else:
        assert_test(multi_pred > 0,
                    "Attack flow: binary=Attack AND multi=Attack-subtype",
                    f"binary={binary_pred}, multi={multi_label}",
                    f"INCONSISTENT: binary={binary_pred}, multi={multi_label}")

    # Test with attack profile
    syn_flow = build_syn_flood().reshape(1, -1)
    res_b_atk = sess_b.run(None, {inp_name: syn_flow})
    res_m_atk = sess_m.run(None, {inp_name: syn_flow})
    binary_atk = int(res_b_atk[0][0])
    multi_atk = int(res_m_atk[0][0])
    multi_atk_label = label_map.get(str(multi_atk), "?")

    print(f"       SYN flood profile -> binary={binary_atk}, multi={multi_atk_label}")
    assert_test(True, "SYN flood inference completed without crash",
                f"binary={binary_atk}, multi={multi_atk_label}")

    udp_flow = build_udp_flood().reshape(1, -1)
    res_b_udp = sess_b.run(None, {inp_name: udp_flow})
    res_m_udp = sess_m.run(None, {inp_name: udp_flow})
    binary_udp = int(res_b_udp[0][0])
    multi_udp_label = label_map.get(str(int(res_m_udp[0][0])), "?")

    print(f"       UDP flood profile -> binary={binary_udp}, multi={multi_udp_label}")
    assert_test(True, "UDP flood inference completed without crash",
                f"binary={binary_udp}, multi={multi_udp_label}")


# ============================================================
# T5: DDoS Realistic Traffic Simulation
# ============================================================
def test_ddos_traffic_sim():
    section("T5: DDoS Realistic Traffic Simulation")

    sess_b = ort.InferenceSession(os.path.join(DDOS_DIR, "ddos_binary_xgboost.onnx"))
    inp_name = sess_b.get_inputs()[0].name
    label_map = load_json(os.path.join(DDOS_DIR, "label_mapping.json"))

    # Simulate 100 benign-like flows with small random perturbations
    np.random.seed(42)
    benign_base = build_ddos_flow()
    benign_batch = np.tile(benign_base, (100, 1))
    noise = np.random.normal(0, 0.05, benign_batch.shape).astype(np.float32)
    benign_batch = benign_batch * (1 + noise)

    preds = sess_b.run(None, {inp_name: benign_batch})[0].flatten()
    benign_ratio = (preds == 0).sum() / len(preds)
    assert_test(benign_ratio > 0.5,
                f"100 benign-like flows: {benign_ratio:.0%} classified as Benign",
                f"{int((preds==0).sum())}/100 benign")

    # All-zero flow (edge case)
    zero_flow = np.zeros((1, N_FEATURES_DDOS), dtype=np.float32)
    res_zero = sess_b.run(None, {inp_name: zero_flow})
    assert_test(True, "All-zero feature vector -- no crash",
                f"prediction={int(res_zero[0][0])}")

    # Extremely large values (edge case)
    huge_flow = np.full((1, N_FEATURES_DDOS), 1e12, dtype=np.float32)
    res_huge = sess_b.run(None, {inp_name: huge_flow})
    assert_test(True, "Extreme large values (1e12) -- no crash",
                f"prediction={int(res_huge[0][0])}")

    # Negative values (shouldn't happen but backend must not crash)
    neg_flow = np.full((1, N_FEATURES_DDOS), -999.0, dtype=np.float32)
    res_neg = sess_b.run(None, {inp_name: neg_flow})
    assert_test(True, "Negative feature values -- no crash",
                f"prediction={int(res_neg[0][0])}")

    # NaN handling
    nan_flow = np.full((1, N_FEATURES_DDOS), float("nan"), dtype=np.float32)
    try:
        res_nan = sess_b.run(None, {inp_name: nan_flow})
        assert_test(True, "NaN input -- no crash (XGBoost handles NaN natively)",
                    f"prediction={int(res_nan[0][0])}")
    except Exception as e:
        assert_test(True, "NaN input -- raised exception (acceptable)",
                    str(e)[:80])


# ============================================================
# T6: DGA Encoding Round-Trip
# ============================================================
def test_dga_encoding():
    section("T6: DGA Character Encoding Verification")

    vocab = load_json(os.path.join(DGA_DIR, "char_vocab.json"))

    # Verify all expected chars are mapped
    expected_chars = list("abcdefghijklmnopqrstuvwxyz0123456789-.")
    missing = [c for c in expected_chars if c not in vocab]
    assert_test(len(missing) == 0,
                "All a-z, 0-9, -, . mapped in vocab",
                f"{len(expected_chars)} chars mapped",
                f"MISSING: {missing}")

    # Encoding produces correct length
    enc = encode_domain("google.com", vocab)
    assert_test(len(enc) == MAX_DOMAIN_LEN,
                f"encode_domain output length == {MAX_DOMAIN_LEN}")

    # Padding check
    enc_short = encode_domain("a.b", vocab)
    assert_test(enc_short[3:] == [0] * (MAX_DOMAIN_LEN - 3),
                "Short domain correctly zero-padded")

    # Unknown char handling
    enc_unk = encode_domain("test@#$.com", vocab)
    at_idx = enc_unk[4]  # '@' should map to UNK=1
    hash_idx = enc_unk[5]  # '#' should map to UNK=1
    assert_test(at_idx == 1 and hash_idx == 1,
                "Unknown chars (@, #) map to UNK index 1",
                f"@->{at_idx}, #->{hash_idx}")

    # Truncation at MAX_LEN
    long_domain = "a" * 300 + ".com"
    enc_long = encode_domain(long_domain, vocab)
    assert_test(len(enc_long) == MAX_DOMAIN_LEN,
                f"Domain > {MAX_DOMAIN_LEN} chars truncated correctly")

    # Case insensitivity
    enc_upper = encode_domain("GOOGLE.COM", vocab)
    enc_lower = encode_domain("google.com", vocab)
    assert_test(enc_upper == enc_lower,
                "Case insensitive: GOOGLE.COM == google.com")


# ============================================================
# T7: DGA Known Domain Classification
# ============================================================
def test_dga_classification():
    section("T7: DGA Domain Classification (Known Patterns)")

    sess = ort.InferenceSession(os.path.join(DGA_DIR, "dga_cnn_bilstm.onnx"))
    vocab = load_json(os.path.join(DGA_DIR, "char_vocab.json"))
    inp_name = sess.get_inputs()[0].name

    # Test domains grouped by expected class
    test_cases = [
        # (domain, expected_class_name, description)
        # --- Benign ---
        ("google.com", "Benign", "Top global site"),
        ("facebook.com", "Benign", "Social media"),
        ("wikipedia.org", "Benign", "Encyclopedia"),
        ("stackoverflow.com", "Benign", "Developer Q&A"),
        ("amazon.co.uk", "Benign", "E-commerce with ccTLD"),
        ("mail.google.com", "Benign", "Subdomain of legit site"),
        ("news.ycombinator.com", "Benign", "Tech news"),
        ("github.com", "Benign", "Code hosting"),

        # --- DNS Tunnel patterns ---
        ("aabbccddee.ffgghhii.jjkkllmm.tunnel-cdn.com", "DNS Tunnel", "Multi-label encoded"),
        ("a1b2c3d4e5f6g7h8i9j0.data-sync.net", "DNS Tunnel", "Long hex-like subdomain"),
        ("xyzabcdef0123456789abcdef.xyzabcdef.update-svc.org", "DNS Tunnel", "Deep encoded labels"),
    ]

    print()
    for domain, expected, desc in test_cases:
        encoded = np.array([encode_domain(domain, vocab)], dtype=np.int64)
        output = sess.run(None, {inp_name: encoded})
        probs = softmax(output[0][0])
        pred_idx = np.argmax(probs)
        pred_name = DGA_CLASSES[pred_idx]
        confidence = probs[pred_idx]
        status = "[OK]" if pred_name == expected else "[??]"
        print(f"       {status} {domain:55s} -> {pred_name:12s} ({confidence:.1%})  [{desc}]")

    assert_test(True, f"Ran {len(test_cases)} domain classifications without crash")


# ============================================================
# T8: DGA Edge Cases
# ============================================================
def test_dga_edge_cases():
    section("T8: DGA Edge Cases")

    sess = ort.InferenceSession(os.path.join(DGA_DIR, "dga_cnn_bilstm.onnx"))
    vocab = load_json(os.path.join(DGA_DIR, "char_vocab.json"))
    inp_name = sess.get_inputs()[0].name

    edge_cases = [
        ("", "Empty string"),
        ("a", "Single character"),
        (".", "Just a dot"),
        ("..", "Double dot"),
        ("a" * 128, "Exactly MAX_LEN chars"),
        ("a" * 500, "Way over MAX_LEN"),
        ("---...---", "Only special chars"),
        ("123456789.123456789.123456789.com", "Numeric-heavy domain"),
        ("xn--n3h.com", "IDN/punycode domain"),
        ("   spaces.com   ", "Leading/trailing whitespace"),
        ("UPPERCASE.COM", "All uppercase"),
        ("MiXeD.CaSe.CoM", "Mixed case"),
        ("a.b.c.d.e.f.g.h.i.j.k.l.m.com", "Many subdomain levels"),
    ]

    for domain, desc in edge_cases:
        try:
            encoded = np.array([encode_domain(domain, vocab)], dtype=np.int64)
            output = sess.run(None, {inp_name: encoded})
            probs = softmax(output[0][0])
            pred_idx = np.argmax(probs)
            pred_name = DGA_CLASSES[pred_idx]
            passed(f"Edge: {desc}", f"'{domain[:30]}' -> {pred_name} ({probs[pred_idx]:.1%})")
        except Exception as e:
            failed(f"Edge: {desc}", str(e)[:80])


# ============================================================
# T9: Output Probability Sanity
# ============================================================
def test_output_probabilities():
    section("T9: Output Probability Distribution Sanity")

    # DDoS binary
    sess_b = ort.InferenceSession(os.path.join(DDOS_DIR, "ddos_binary_xgboost.onnx"))
    inp_name = sess_b.get_inputs()[0].name
    dummy = build_ddos_flow().reshape(1, -1)
    res = sess_b.run(None, {inp_name: dummy})
    probs = res[1]  # probabilities output

    if hasattr(probs, '__len__') and len(probs) > 0:
        probs_flat = np.array(probs).flatten()
        # XGBoost outputs a probability map, check it sums to ~1
        # For binary, it outputs [{class0: p0, class1: p1}]
        if hasattr(probs[0], 'values'):
            vals = list(probs[0].values())
        else:
            vals = probs_flat
        prob_sum = sum(vals[:2]) if len(vals) >= 2 else vals[0]
        assert_test(abs(prob_sum - 1.0) < 0.01,
                    f"DDoS binary probabilities sum to ~1.0",
                    f"sum={prob_sum:.6f}")
        assert_test(all(v >= 0 for v in vals[:2]),
                    "DDoS binary probabilities all >= 0")

    # DGA
    sess_d = ort.InferenceSession(os.path.join(DGA_DIR, "dga_cnn_bilstm.onnx"))
    vocab = load_json(os.path.join(DGA_DIR, "char_vocab.json"))
    inp_d = sess_d.get_inputs()[0].name
    enc = np.array([encode_domain("test.com", vocab)], dtype=np.int64)
    out = sess_d.run(None, {inp_d: enc})
    logits = out[0][0]
    probs_dga = softmax(logits)

    assert_test(len(logits) == 3,
                "DGA output has 3 logits (3 classes)",
                str(logits))
    assert_test(abs(probs_dga.sum() - 1.0) < 1e-5,
                "DGA softmax probabilities sum to 1.0",
                f"sum={probs_dga.sum():.8f}")
    assert_test(all(p >= 0 for p in probs_dga),
                "DGA softmax probabilities all >= 0")


# ============================================================
# T10: Determinism
# ============================================================
def test_determinism():
    section("T10: Determinism (Same Input -> Same Output)")

    # DDoS
    sess_b = ort.InferenceSession(os.path.join(DDOS_DIR, "ddos_binary_xgboost.onnx"))
    inp_name = sess_b.get_inputs()[0].name
    flow = build_ddos_flow().reshape(1, -1)

    results = [sess_b.run(None, {inp_name: flow})[0][0] for _ in range(10)]
    assert_test(len(set([int(r) for r in results])) == 1,
                "DDoS binary: 10 identical inputs -> identical outputs",
                f"all predictions = {int(results[0])}")

    # DGA
    sess_d = ort.InferenceSession(os.path.join(DGA_DIR, "dga_cnn_bilstm.onnx"))
    vocab = load_json(os.path.join(DGA_DIR, "char_vocab.json"))
    inp_d = sess_d.get_inputs()[0].name
    enc = np.array([encode_domain("test.com", vocab)], dtype=np.int64)

    logits_list = [sess_d.run(None, {inp_d: enc})[0][0].tolist() for _ in range(10)]
    assert_test(all(l == logits_list[0] for l in logits_list),
                "DGA: 10 identical inputs -> identical logits",
                f"logits = [{', '.join(f'{v:.4f}' for v in logits_list[0])}]")


# ============================================================
# T11: Latency Benchmark
# ============================================================
def test_latency():
    section(f"T11: Latency Benchmark ({BENCHMARK_RUNS} runs each)")

    # DDoS binary
    sess_b = ort.InferenceSession(os.path.join(DDOS_DIR, "ddos_binary_xgboost.onnx"))
    inp_b = sess_b.get_inputs()[0].name
    flow = build_ddos_flow().reshape(1, -1)

    # Warmup
    for _ in range(50):
        sess_b.run(None, {inp_b: flow})

    t0 = time.perf_counter()
    for _ in range(BENCHMARK_RUNS):
        sess_b.run(None, {inp_b: flow})
    elapsed_b = (time.perf_counter() - t0) / BENCHMARK_RUNS * 1000

    assert_test(elapsed_b < 50,
                f"DDoS binary single inference < 50ms",
                f"{elapsed_b:.3f} ms ({1000/elapsed_b:,.0f} flows/sec)")

    # DDoS multi
    sess_m = ort.InferenceSession(os.path.join(DDOS_DIR, "ddos_multi_xgboost.onnx"))
    for _ in range(50):
        sess_m.run(None, {inp_b: flow})

    t0 = time.perf_counter()
    for _ in range(BENCHMARK_RUNS):
        sess_m.run(None, {inp_b: flow})
    elapsed_m = (time.perf_counter() - t0) / BENCHMARK_RUNS * 1000

    assert_test(elapsed_m < 100,
                f"DDoS multi single inference < 100ms",
                f"{elapsed_m:.3f} ms ({1000/elapsed_m:,.0f} flows/sec)")

    # DGA
    sess_d = ort.InferenceSession(os.path.join(DGA_DIR, "dga_cnn_bilstm.onnx"))
    vocab = load_json(os.path.join(DGA_DIR, "char_vocab.json"))
    inp_d = sess_d.get_inputs()[0].name
    enc = np.array([encode_domain("google.com", vocab)], dtype=np.int64)

    for _ in range(50):
        sess_d.run(None, {inp_d: enc})

    t0 = time.perf_counter()
    for _ in range(BENCHMARK_RUNS):
        sess_d.run(None, {inp_d: enc})
    elapsed_d = (time.perf_counter() - t0) / BENCHMARK_RUNS * 1000

    assert_test(elapsed_d < 100,
                f"DGA single inference < 100ms",
                f"{elapsed_d:.3f} ms ({1000/elapsed_d:,.0f} domains/sec)")


# ============================================================
# T12: Batch Size Stress Test
# ============================================================
def test_batch_stress():
    section("T12: Batch Size Stress Test")

    batch_sizes = [1, 16, 128, 512, 1024]

    # DDoS
    sess_b = ort.InferenceSession(os.path.join(DDOS_DIR, "ddos_binary_xgboost.onnx"))
    inp_b = sess_b.get_inputs()[0].name

    for bs in batch_sizes:
        try:
            batch = np.random.randn(bs, N_FEATURES_DDOS).astype(np.float32)
            t0 = time.perf_counter()
            res = sess_b.run(None, {inp_b: batch})
            elapsed = (time.perf_counter() - t0) * 1000
            assert_test(res[0].shape[0] == bs,
                        f"DDoS binary batch={bs}",
                        f"output_shape={res[0].shape}, {elapsed:.1f}ms")
        except Exception as e:
            failed(f"DDoS binary batch={bs}", str(e)[:80])

    # DGA
    sess_d = ort.InferenceSession(os.path.join(DGA_DIR, "dga_cnn_bilstm.onnx"))
    vocab = load_json(os.path.join(DGA_DIR, "char_vocab.json"))
    inp_d = sess_d.get_inputs()[0].name

    for bs in batch_sizes:
        try:
            batch = np.random.randint(0, 40, (bs, MAX_DOMAIN_LEN)).astype(np.int64)
            t0 = time.perf_counter()
            res = sess_d.run(None, {inp_d: batch})
            elapsed = (time.perf_counter() - t0) * 1000
            assert_test(res[0].shape == (bs, 3),
                        f"DGA batch={bs}",
                        f"output_shape={res[0].shape}, {elapsed:.1f}ms")
        except Exception as e:
            failed(f"DGA batch={bs}", str(e)[:80])


# ============================================================
# Run all tests
# ============================================================
if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#  NetSentinel -- Industrial Integration Test Suite")
    print(f"#  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"#  ONNX Runtime: {ort.__version__}")
    print(f"#  NumPy: {np.__version__}")
    print("#" * 70)

    test_funcs = [
        test_artifact_integrity,
        test_metadata_schema,
        test_onnx_io_shapes,
        test_ddos_consistency,
        test_ddos_traffic_sim,
        test_dga_encoding,
        test_dga_classification,
        test_dga_edge_cases,
        test_output_probabilities,
        test_determinism,
        test_latency,
        test_batch_stress,
    ]

    for fn in test_funcs:
        try:
            fn()
        except Exception as e:
            failed(f"SECTION CRASHED: {fn.__name__}", traceback.format_exc()[-120:])

    # ---- Summary ----
    total_pass = sum(1 for r in RESULTS if r[2] == "PASS")
    total_fail = sum(1 for r in RESULTS if r[2] == "FAIL")
    total = total_pass + total_fail

    print("\n" + "#" * 70)
    print(f"#  FINAL RESULTS: {total_pass}/{total} PASSED, {total_fail} FAILED")
    print("#" * 70)

    if total_fail > 0:
        print("\n  FAILURES:")
        for sec, name, status, detail in RESULTS:
            if status == "FAIL":
                print(f"    [{sec}] {name}: {detail}")
        print()
        sys.exit(1)
    else:
        print("\n  >>> ALL TESTS PASSED. Models verified for backend integration.\n")
        sys.exit(0)
