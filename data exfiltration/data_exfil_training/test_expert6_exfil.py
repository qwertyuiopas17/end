# -*- coding: utf-8 -*-
"""
NetSentinel -- Expert 6: Data Exfiltration Test Suite
======================================================
Comprehensive verification of the DNS-exfiltration VAE anomaly detector.

Test Categories:
  T1  - Artifact Integrity (all files present & loadable)
  T2  - Metadata Schema Validation
  T3  - ONNX Model I/O Shape Verification
  T4  - Scaler & Isolation Forest Pipeline Check
  T5  - Easy Tests: Obvious Exfiltration Patterns
  T6  - Easy Tests: Obviously Benign Domains
  T7  - Industrial: CIC-Bell-DNS-EXF-2021 Replay Simulation
  T8  - Industrial: MITRE ATT&CK T1071.004 DNS Tunneling Payloads
  T9  - Industrial: DGA-style Exfiltration (Iodine/DNScat2)
  T10 - Industrial: Slow-Drip / Low-and-Slow Exfiltration
  T11 - Edge Cases (empty, max-length, unicode, numeric-only)
  T12 - Adversarial Evasion Attempts
  T13 - Determinism (same input => same output)
  T14 - Latency Benchmarking (single + batch)
  T15 - Batch Stress Test
  T16 - Score Distribution Analysis

Usage:
    pip install onnxruntime numpy joblib scikit-learn
    python test_expert6_exfil.py

Exit code 0 = all passed, 1 = failures detected.
"""

import json
import math
import numpy as np
import onnxruntime as ort
import os
import sys
import time
import traceback
import string
import random
from collections import Counter
from itertools import groupby

# ============================================================
# Config
# ============================================================
EXFIL_DIR = r"D:\1_sih26\Data Exfiltration"
REQUIRED_FILES = [
    "expert6_meta.json",
    "expert6_vae.onnx",
    "expert6_vae.pt",
    "expert6_scaler.joblib",
    "expert6_iso.joblib",
    "expert6_graphs.png",
]
SEED = 42
BENCHMARK_RUNS = 500

# ============================================================
# Test Framework
# ============================================================
RESULTS = []
CURRENT_SECTION = ""
STATS = {}  # accumulate statistics for final report


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


def skipped(name, detail=""):
    RESULTS.append((CURRENT_SECTION, name, "SKIP", detail))
    print(f"  [SKIP] {name}" + (f"  --  {detail}" if detail else ""))


def assert_test(condition, name, pass_detail="", fail_detail=""):
    if condition:
        passed(name, pass_detail)
    else:
        failed(name, fail_detail)
    return condition


# ============================================================
# Feature Engineering (mirrors notebook exactly)
# ============================================================
VOWELS = set('aeiou')
CONSONANTS = set('bcdfghjklmnpqrstvwxyz')


def shannon_entropy(s):
    s = str(s)
    if len(s) == 0:
        return 0.0
    n = len(s)
    return float(-sum((c / n) * math.log2(c / n) for c in Counter(s).values()))


def bigram_entropy(s):
    s = str(s)
    if len(s) < 2:
        return 0.0
    bg = [s[i:i + 2] for i in range(len(s) - 1)]
    n = len(bg)
    return float(-sum((c / n) * math.log2(c / n) for c in Counter(bg).values()))


def engineer_single_domain(domain):
    """Engineer features for a single domain string.
    Returns a dict of feature_name -> value matching the notebook's feature set."""
    s = str(domain)
    L = float(len(s))
    Lp = L + 1.0

    feats = {}
    feats['dns_len'] = L
    feats['dns_log_len'] = math.log1p(L)
    feats['dns_label_count'] = float(s.count('.') + 1)
    parts = s.replace('-', '.').replace('_', '.').split('.')
    feats['dns_longest_token'] = float(max((len(p) for p in parts), default=0))
    feats['dns_entropy'] = shannon_entropy(s)
    feats['dns_bigram_entropy'] = bigram_entropy(s)
    feats['dns_norm_entropy'] = feats['dns_entropy'] / math.log2(Lp + 1) if Lp > 0 else 0.0
    feats['dns_digit_ratio'] = sum(1 for c in s if c.isdigit()) / Lp
    feats['dns_upper_ratio'] = sum(1 for c in s if c.isupper()) / Lp
    feats['dns_lower_ratio'] = sum(1 for c in s if c.islower()) / Lp
    feats['dns_special_ratio'] = sum(1 for c in s if c not in string.ascii_letters + string.digits + '.') / Lp
    feats['dns_hex_ratio'] = sum(1 for c in s if c in '0123456789abcdefABCDEF') / Lp
    feats['dns_vowel_ratio'] = sum(1 for c in s.lower() if c in VOWELS) / max(len(s), 1)
    feats['dns_unique_chars'] = float(len(set(s)))
    feats['dns_unique_ratio'] = feats['dns_unique_chars'] / Lp
    feats['dns_max_repeat'] = float(max((sum(1 for _ in g) for _, g in groupby(s)), default=0))
    return feats


def build_feature_vector(domain, meta):
    """Build a full feature vector matching the model's expected input order."""
    feats = engineer_single_domain(domain)

    # For features in meta['features'] that aren't DNS-engineered, use 0.0
    vector = []
    for f in meta['features']:
        if f in feats:
            vector.append(feats[f])
        else:
            # Native numeric features from dataset — use 0.0 for synthetic tests
            vector.append(0.0)
    return np.array(vector, dtype=np.float32)


def build_batch(domains, meta):
    """Build a batch of feature vectors."""
    return np.stack([build_feature_vector(d, meta) for d in domains])


# ============================================================
# T1 - Artifact Integrity
# ============================================================
def test_artifact_integrity():
    section("T1 - Artifact Integrity")
    for fname in REQUIRED_FILES:
        path = os.path.join(EXFIL_DIR, fname)
        exists = os.path.isfile(path)
        size = os.path.getsize(path) if exists else 0
        assert_test(exists and size > 0, f"File exists: {fname}",
                    f"{size / 1024:.1f} KB", f"Missing or empty: {path}")


# ============================================================
# T2 - Metadata Schema Validation
# ============================================================
def test_metadata_schema():
    section("T2 - Metadata Schema Validation")
    meta_path = os.path.join(EXFIL_DIR, "expert6_meta.json")
    meta = json.load(open(meta_path))

    required_keys = ["model", "type", "roc_auc", "pr_auc", "best_f1",
                     "accuracy", "flip", "n_features", "features",
                     "scorer_aucs", "mitre", "dataset"]
    for key in required_keys:
        assert_test(key in meta, f"Meta has key: {key}",
                    f"value={str(meta.get(key, ''))[:60]}", f"Missing key: {key}")

    assert_test(isinstance(meta['features'], list) and len(meta['features']) > 0,
                "Features list non-empty", f"{len(meta['features'])} features")

    assert_test(meta['n_features'] == len(meta['features']),
                "n_features matches features list length",
                f"{meta['n_features']} == {len(meta['features'])}")

    assert_test(0.0 <= meta['roc_auc'] <= 1.0, "ROC-AUC in [0,1]",
                f"{meta['roc_auc']:.4f}")
    assert_test(0.0 <= meta['pr_auc'] <= 1.0, "PR-AUC in [0,1]",
                f"{meta['pr_auc']:.4f}")
    assert_test(0.0 <= meta['best_f1'] <= 1.0, "Best F1 in [0,1]",
                f"{meta['best_f1']:.4f}")

    # MITRE ATT&CK mapping
    mitre_expected = {"T1041", "T1048", "T1071.004"}
    assert_test(set(meta['mitre'].keys()) == mitre_expected,
                "MITRE ATT&CK mappings present",
                str(meta['mitre']))

    STATS['meta'] = meta


# ============================================================
# T3 - ONNX Model I/O Shapes
# ============================================================
def test_onnx_io_shapes():
    section("T3 - ONNX Model I/O Shape Verification")
    meta = STATS.get('meta', json.load(open(os.path.join(EXFIL_DIR, "expert6_meta.json"))))
    onnx_path = os.path.join(EXFIL_DIR, "expert6_vae.onnx")

    sess = ort.InferenceSession(onnx_path)
    STATS['onnx_sess'] = sess

    inputs = sess.get_inputs()
    outputs = sess.get_outputs()

    assert_test(len(inputs) == 1, "Single input tensor", f"name={inputs[0].name}")
    assert_test(len(outputs) == 1, "Single output tensor", f"name={outputs[0].name}")

    in_shape = inputs[0].shape
    out_shape = outputs[0].shape
    n_feat = meta['n_features']

    assert_test(in_shape[-1] == n_feat, "Input dim matches n_features",
                f"shape={in_shape}, expected last dim={n_feat}",
                f"shape={in_shape}, expected last dim={n_feat}")

    assert_test(out_shape[-1] == n_feat, "Output dim matches n_features (reconstruction)",
                f"shape={out_shape}", f"shape={out_shape}")

    # Smoke test
    dummy = np.random.randn(1, n_feat).astype(np.float32)
    result = sess.run(None, {inputs[0].name: dummy})
    assert_test(result[0].shape == (1, n_feat), "Smoke test output shape",
                f"{result[0].shape}")


# ============================================================
# T4 - Scaler & Isolation Forest Pipeline
# ============================================================
def test_pipeline():
    section("T4 - Scaler & Isolation Forest Pipeline")
    try:
        import joblib
    except ImportError:
        skipped("joblib import", "pip install joblib")
        return

    scaler_path = os.path.join(EXFIL_DIR, "expert6_scaler.joblib")
    iso_path = os.path.join(EXFIL_DIR, "expert6_iso.joblib")

    scaler = joblib.load(scaler_path)
    iso = joblib.load(iso_path)
    STATS['scaler'] = scaler
    STATS['iso'] = iso

    meta = STATS.get('meta', json.load(open(os.path.join(EXFIL_DIR, "expert6_meta.json"))))
    n_feat = meta['n_features']

    # Test scaler transform
    dummy = np.random.randn(5, n_feat).astype(np.float32)
    scaled = scaler.transform(dummy)
    assert_test(scaled.shape == (5, n_feat), "Scaler output shape correct",
                f"{scaled.shape}")
    assert_test(np.all(np.isfinite(scaled)), "Scaler output finite")

    # Test Isolation Forest
    scores = iso.score_samples(scaled)
    assert_test(scores.shape == (5,), "IsoForest output shape correct",
                f"{scores.shape}")
    assert_test(np.all(np.isfinite(scores)), "IsoForest output finite")


# ============================================================
# Scoring helper (used by many tests)
# ============================================================
def get_recon_score(domains):
    """Score domains using the full pipeline. Returns anomaly scores.
    Higher = more anomalous (flip applied)."""
    meta = STATS.get('meta', json.load(open(os.path.join(EXFIL_DIR, "expert6_meta.json"))))
    sess = STATS.get('onnx_sess')
    if sess is None:
        sess = ort.InferenceSession(os.path.join(EXFIL_DIR, "expert6_vae.onnx"))
        STATS['onnx_sess'] = sess

    batch = build_batch(domains, meta)

    # Apply scaler if available
    scaler = STATS.get('scaler')
    if scaler is not None:
        batch = np.clip(scaler.transform(batch), -10, 10).astype(np.float32)

    inp_name = sess.get_inputs()[0].name
    recon = sess.run(None, {inp_name: batch})[0]
    mse = np.mean((recon - batch) ** 2, axis=1)

    # The VAE was trained on benign data only.
    # Higher reconstruction error = further from benign distribution = more anomalous.
    # meta['flip']=True was used during Kaggle evaluation because the scorer
    # was inverted there. For raw MSE, higher = more anomalous (no flip needed).
    return mse


# ============================================================
# T5 - Easy Tests: Obvious Exfiltration Patterns
# ============================================================
def test_obvious_exfil():
    section("T5 - Easy Tests: Obvious Exfiltration Patterns")

    exfil_domains = [
        # Base64-encoded data in subdomain (classic DNS exfil)
        "aGVsbG8gd29ybGQgdGhpcyBpcyBhIHRlc3Q.evil.com",
        # Hex-encoded payload
        "4f70656e5468654761746573.c2server.net",
        # Long random subdomain (data tunneling)
        "x8k2mfqa9b3n7y4p1c6d0e5h.tunnel.attacker.io",
        # Multiple subdomains with encoded data
        "dGhpcw.aXM.YQ.dGVzdA.exfil.bad.org",
        # Very high entropy random chars
        "j3k9x2v8m1q4z7w0p5b6n.leak.malware.xyz",
        # Simulated iodine tunnel format
        "t0x00.zAABCDEF0123456789abcdef.dns.tunnel.com",
        # DNScat2 style
        "dnscat.556e69636f7265.c2.evil.net",
        # Long continuous hex string
        "48656c6c6f576f726c6448656c6c6f576f726c64.data.exfil.com",
    ]

    benign_domains = [
        "www.google.com",
        "mail.yahoo.com",
        "api.github.com",
        "cdn.cloudflare.com",
        "docs.microsoft.com",
    ]

    exfil_scores = get_recon_score(exfil_domains)
    benign_scores = get_recon_score(benign_domains)

    exfil_mean = np.mean(exfil_scores)
    benign_mean = np.mean(benign_scores)
    separation = exfil_mean - benign_mean

    STATS['easy_exfil_scores'] = exfil_scores
    STATS['easy_benign_scores'] = benign_scores

    print(f"  Exfil mean score:  {exfil_mean:.6f}")
    print(f"  Benign mean score: {benign_mean:.6f}")
    print(f"  Separation:        {separation:.6f}")

    for d, s in zip(exfil_domains, exfil_scores):
        label = "[Y]" if s > benign_mean else "[N]"
        print(f"    {label} [{s:+.4f}] {d[:55]}")

    assert_test(separation > 0, "Exfil scores higher than benign (mean separation)",
                f"sep={separation:.6f}")

    # Individual detection: at least 60% of exfil detected above benign mean
    detected = sum(1 for s in exfil_scores if s > benign_mean)
    rate = detected / len(exfil_domains)
    assert_test(rate >= 0.5, f"Exfil detection rate >= 50%",
                f"{detected}/{len(exfil_domains)} = {rate:.0%}",
                f"{detected}/{len(exfil_domains)} = {rate:.0%}")


# ============================================================
# T6 - Easy Tests: Obviously Benign Domains
# ============================================================
def test_obvious_benign():
    section("T6 - Easy Tests: Obviously Benign Domains")

    benign_domains = [
        "www.google.com",
        "www.facebook.com",
        "www.amazon.com",
        "www.wikipedia.org",
        "news.bbc.co.uk",
        "mail.yahoo.com",
        "stackoverflow.com",
        "www.reddit.com",
        "play.google.com",
        "apps.apple.com",
        "www.linkedin.com",
        "www.netflix.com",
        "www.youtube.com",
        "github.com",
        "twitter.com",
        "www.microsoft.com",
        "docs.python.org",
        "en.wikipedia.org",
        "www.nytimes.com",
        "www.cnn.com",
    ]

    scores = get_recon_score(benign_domains)
    mean_s = np.mean(scores)
    std_s = np.std(scores)

    STATS['benign_baseline_mean'] = mean_s
    STATS['benign_baseline_std'] = std_s

    print(f"  Benign score mean: {mean_s:.6f}")
    print(f"  Benign score std:  {std_s:.6f}")
    for d, s in zip(benign_domains, scores):
        print(f"    [{s:+.6f}] {d}")

    # All benign scores should be relatively tightly clustered
    assert_test(std_s < abs(mean_s) * 5 + 1.0, "Benign scores have bounded variance",
                f"std={std_s:.4f}")


# ============================================================
# T7 - Industrial: Realistic Exfil Traffic Simulation
# ============================================================
def test_industrial_realistic():
    section("T7 - Industrial: CIC-Bell-DNS-EXF-2021 Replay Simulation")

    # Simulate various exfiltration techniques from the CIC-Bell dataset
    rng = random.Random(SEED)

    # Generate 50 benign-looking domains
    benign_tlds = ['.com', '.org', '.net', '.co.uk', '.io', '.dev', '.edu']
    benign_words = ['cloud', 'api', 'auth', 'mail', 'web', 'portal', 'login',
                    'shop', 'store', 'app', 'docs', 'help', 'support', 'blog',
                    'news', 'data', 'cdn', 'static', 'assets', 'images']
    benign_prefixes = ['www', 'mail', 'api', 'cdn', 'app', 'm', 'dev', 'staging']

    synthetic_benign = []
    for _ in range(50):
        prefix = rng.choice(benign_prefixes)
        word = rng.choice(benign_words)
        tld = rng.choice(benign_tlds)
        synthetic_benign.append(f"{prefix}.{word}{tld}")

    # Generate 50 exfil-style domains (mimicking real attack patterns)
    synthetic_exfil = []

    # Pattern 1: Base64 chunks in subdomains
    for _ in range(12):
        payload = ''.join(rng.choices(string.ascii_letters + string.digits + '+/', k=rng.randint(20, 45)))
        synthetic_exfil.append(f"{payload}.evil.com")

    # Pattern 2: Hex-encoded data
    for _ in range(12):
        payload = ''.join(rng.choices('0123456789abcdef', k=rng.randint(24, 48)))
        synthetic_exfil.append(f"{payload}.data.exfil.net")

    # Pattern 3: Multiple short encoded labels
    for _ in range(8):
        labels = [
            ''.join(rng.choices(string.ascii_lowercase + string.digits, k=rng.randint(6, 12)))
            for _ in range(rng.randint(3, 6))
        ]
        synthetic_exfil.append('.'.join(labels) + '.c2.org')

    # Pattern 4: Very long subdomains (max DNS label = 63 chars)
    for _ in range(8):
        payload = ''.join(rng.choices(string.ascii_lowercase + string.digits, k=55))
        synthetic_exfil.append(f"{payload}.tunnel.io")

    # Pattern 5: High-entropy mixed case
    for _ in range(10):
        payload = ''.join(rng.choices(string.ascii_letters + string.digits, k=rng.randint(30, 50)))
        synthetic_exfil.append(f"{payload}.dns.bad.com")

    benign_scores = get_recon_score(synthetic_benign)
    exfil_scores = get_recon_score(synthetic_exfil)

    b_mean = np.mean(benign_scores)
    e_mean = np.mean(exfil_scores)

    # Use median of benign as threshold
    threshold = np.median(benign_scores) + 1.5 * np.std(benign_scores)

    tp = np.sum(exfil_scores > threshold)
    fp = np.sum(benign_scores > threshold)
    tn = np.sum(benign_scores <= threshold)
    fn = np.sum(exfil_scores <= threshold)

    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tpr
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    STATS['industrial_sim'] = {
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'tpr': float(tpr), 'fpr': float(fpr),
        'precision': float(precision), 'f1': float(f1)
    }

    print(f"  Benign mean:    {b_mean:.6f}")
    print(f"  Exfil mean:     {e_mean:.6f}")
    print(f"  Threshold:      {threshold:.6f}")
    print(f"  TP={tp:3d}  FP={fp:3d}  TN={tn:3d}  FN={fn:3d}")
    print(f"  TPR={tpr:.2%}  FPR={fpr:.2%}  Precision={precision:.2%}  F1={f1:.4f}")

    assert_test(e_mean > b_mean, "Exfil scores > Benign scores",
                f"sep={e_mean - b_mean:.6f}")
    assert_test(tpr >= 0.4, f"True positive rate >= 40%",
                f"TPR={tpr:.2%}", f"TPR={tpr:.2%}")


# ============================================================
# T8 - Industrial: MITRE ATT&CK T1071.004 DNS Tunneling
# ============================================================
def test_mitre_dns_tunnel():
    section("T8 - Industrial: MITRE ATT&CK T1071.004 DNS Tunneling")

    # Real-world DNS tunneling tool patterns
    tunnel_payloads = {
        "iodine": [
            "t0x00.zAABCDEF0123456789abcdef01234567.dns.tunnel.com",
            "t0x01.z89ABCDEF0123456789ABCDEF01234567.dns.tunnel.com",
            "t1x00.z0123456789ABCDEF0123456789ABCDEF.dns.tunnel.com",
        ],
        "dnscat2": [
            "dnscat.556e69636f7265446174614578.c2.evil.net",
            "dnscat.48656c6c6f576f726c6448656c6c6f.c2.evil.net",
            "dnscat.4d79536563726574446174614578.c2.evil.net",
        ],
        "dns2tcp": [
            "AAAAAB4AAAAAAAAAAAAAAAAAAAAAAAAA.dns2tcp.example.com",
            "BBBBBCD2FFFFFFFFFFFFFFFFFF00AAAA.dns2tcp.example.com",
        ],
        "cobalt_strike": [
            "aabbccdd.www.stage.payload.evil-cdn.com",
            "eeff0011.www.stage.payload.evil-cdn.com",
            "deadbeef.beacon.evil-cdn.com",
        ],
        "sliver_c2": [
            "base64encodedshellcodehere123456.implant.c2server.io",
            "YW5vdGhlcnBheWxvYWRkYXRh.implant.c2server.io",
        ],
    }

    all_tunnel_domains = []
    all_tunnel_labels = []
    for tool, domains in tunnel_payloads.items():
        all_tunnel_domains.extend(domains)
        all_tunnel_labels.extend([tool] * len(domains))

    scores = get_recon_score(all_tunnel_domains)
    benign_mean = STATS.get('benign_baseline_mean', 0.0)

    print(f"  Benign baseline mean: {benign_mean:.6f}")
    print()

    detected_by_tool = {}
    for tool, doms in tunnel_payloads.items():
        tool_scores = get_recon_score(doms)
        det = sum(1 for s in tool_scores if s > benign_mean)
        detected_by_tool[tool] = (det, len(doms))
        rate_str = f"{det}/{len(doms)}"
        print(f"  {tool:15s}: {rate_str:>5s} detected")
        for d, s in zip(doms, tool_scores):
            flag = "[Y]" if s > benign_mean else "[N]"
            print(f"    {flag} [{s:+.4f}] {d[:55]}")

    total_det = sum(d for d, _ in detected_by_tool.values())
    total_all = sum(t for _, t in detected_by_tool.values())
    overall_rate = total_det / total_all if total_all > 0 else 0

    STATS['mitre_detection_by_tool'] = detected_by_tool
    STATS['mitre_overall_rate'] = overall_rate

    assert_test(overall_rate >= 0.4, f"DNS tunnel detection >= 40%",
                f"{total_det}/{total_all} = {overall_rate:.0%}",
                f"{total_det}/{total_all} = {overall_rate:.0%}")


# ============================================================
# T9 - Industrial: DGA-style Exfiltration
# ============================================================
def test_dga_exfil():
    section("T9 - Industrial: DGA-style Exfiltration")

    rng = random.Random(SEED + 1)

    # DGA patterns that also serve as exfil channels
    dga_exfil = []
    # Wordlist-based DGA
    words = ['apple', 'orange', 'banana', 'grape', 'melon', 'kiwi', 'plum', 'cherry']
    for _ in range(10):
        w1, w2 = rng.sample(words, 2)
        dga_exfil.append(f"{w1}{w2}{rng.randint(100,999)}.dga.net")

    # Character-based DGA (Necurs-like)
    for _ in range(10):
        length = rng.randint(12, 25)
        domain = ''.join(rng.choices(string.ascii_lowercase, k=length))
        dga_exfil.append(f"{domain}.com")

    # Arithmetic DGA (Conficker-like)
    for _ in range(10):
        chars = ''.join(chr(rng.randint(97, 122)) for _ in range(rng.randint(8, 15)))
        dga_exfil.append(f"{chars}.info")

    scores = get_recon_score(dga_exfil)
    benign_mean = STATS.get('benign_baseline_mean', 0.0)
    detected = sum(1 for s in scores if s > benign_mean)
    rate = detected / len(dga_exfil)

    print(f"  DGA-exfil domains: {len(dga_exfil)}")
    print(f"  Detected: {detected}/{len(dga_exfil)} = {rate:.0%}")
    for d, s in zip(dga_exfil[:10], scores[:10]):
        flag = "[Y]" if s > benign_mean else "[N]"
        print(f"    {flag} [{s:+.4f}] {d}")

    STATS['dga_exfil_rate'] = rate


# ============================================================
# T10 - Industrial: Slow-Drip / Low-and-Slow Exfiltration
# ============================================================
def test_slow_drip():
    section("T10 - Industrial: Low-and-Slow Exfiltration")

    # These look more like benign traffic but have subtle encoded payloads
    slow_drip = [
        # Short base32-like labels (small data chunks)
        "mfrgg.cts.example.com",
        "nbswy3dp.cdn.example.com",
        "jbswy3dpehpk3pxp.static.example.com",
        # Encoded filenames
        "document-report-q3.updates.company.com",
        "file-2024-annual-budget.sync.company.com",
        # Slightly elevated entropy but not extreme
        "x7b2k.api.legit-service.com",
        "m3n9p.auth.legit-service.com",
        # Mimicking CDN patterns
        "ab12cd34.assets.cdn-provider.net",
        "ef56gh78.images.cdn-provider.net",
        # Typo-squatting with data
        "goog1e.com.data.evil.io",
    ]

    scores = get_recon_score(slow_drip)
    benign_mean = STATS.get('benign_baseline_mean', 0.0)

    print(f"  Low-and-slow domains: {len(slow_drip)}")
    for d, s in zip(slow_drip, scores):
        flag = "[Y]" if s > benign_mean else "[N]"
        print(f"    {flag} [{s:+.4f}] {d}")

    # These are hard — even 30% is acceptable
    detected = sum(1 for s in scores if s > benign_mean)
    rate = detected / len(slow_drip)
    STATS['slow_drip_rate'] = rate
    print(f"  Detection rate: {detected}/{len(slow_drip)} = {rate:.0%}")
    print(f"  (These are intentionally hard — low-and-slow mimics benign traffic)")


# ============================================================
# T11 - Edge Cases
# ============================================================
def test_edge_cases():
    section("T11 - Edge Cases")
    meta = STATS.get('meta', json.load(open(os.path.join(EXFIL_DIR, "expert6_meta.json"))))

    cases = [
        ("empty string", ""),
        ("single char", "a"),
        ("single dot", "."),
        ("dots only", "..."),
        ("max length (253)", "a" * 63 + "." + "b" * 63 + "." + "c" * 63 + "." + "d" * 60 + ".com"),
        ("all digits", "1234567890.123.456"),
        ("all uppercase", "ABCDEFGHIJKLMNOP.COM"),
        ("unicode/IDN", "xn--nxasmq6b.xn--jxalpdlp"),
        ("special chars", "test!@#$%.weird.com"),
        ("ip-like", "192.168.1.1"),
        ("localhost", "localhost"),
        ("very short domain", "t.co"),
    ]

    for label, domain in cases:
        try:
            score = get_recon_score([domain])[0]
            passed(f"Edge case: {label}", f"score={score:+.4f}")
        except Exception as e:
            failed(f"Edge case: {label}", str(e)[:80])


# ============================================================
# T12 - Adversarial Evasion Attempts
# ============================================================
def test_adversarial():
    section("T12 - Adversarial Evasion Attempts")

    # Attackers trying to evade detection
    evasion_domains = [
        # Padding with real words to lower entropy
        "hello.world.the.quick.brown.fox.4f70656e.evil.com",
        # Using common TLD to look benign
        "aGVsbG8gd29ybGQ.google.com",
        # Mixing real subdomains with payload
        "www.login.48656c6c6f.accounts.secure-bank.com",
        # Using punycode to hide data
        "xn--aGVsbG8gd29ybGQ.evil.com",
        # Splitting payload across many short labels
        "a1.b2.c3.d4.e5.f6.g7.h8.evil.com",
        # Using legitimate-looking words + subtle hex suffix
        "update-check-ff0011.services.microsoft-cdn.com",
    ]

    scores = get_recon_score(evasion_domains)
    benign_mean = STATS.get('benign_baseline_mean', 0.0)

    print(f"  Adversarial evasion domains: {len(evasion_domains)}")
    for d, s in zip(evasion_domains, scores):
        flag = "[Y] DETECTED" if s > benign_mean else "[N] EVADED"
        print(f"    {flag} [{s:+.4f}] {d}")

    detected = sum(1 for s in scores if s > benign_mean)
    rate = detected / len(evasion_domains)
    STATS['adversarial_evasion_rate'] = rate
    print(f"  Evasion resistance: {detected}/{len(evasion_domains)} = {rate:.0%}")


# ============================================================
# T13 - Determinism
# ============================================================
def test_determinism():
    section("T13 - Determinism (same input => same output)")
    test_domains = [
        "aGVsbG8gd29ybGQ.evil.com",
        "www.google.com",
        "48656c6c6f576f726c64.data.exfil.com",
    ]

    scores1 = get_recon_score(test_domains)
    scores2 = get_recon_score(test_domains)

    for i, d in enumerate(test_domains):
        diff = abs(scores1[i] - scores2[i])
        assert_test(diff < 1e-5, f"Deterministic: {d[:40]}",
                    f"diff={diff:.2e}", f"diff={diff:.2e}")


# ============================================================
# T14 - Latency Benchmarking
# ============================================================
def test_latency():
    section("T14 - Latency Benchmarking")
    meta = STATS.get('meta', json.load(open(os.path.join(EXFIL_DIR, "expert6_meta.json"))))
    sess = STATS.get('onnx_sess')
    if sess is None:
        sess = ort.InferenceSession(os.path.join(EXFIL_DIR, "expert6_vae.onnx"))

    n_feat = meta['n_features']
    inp_name = sess.get_inputs()[0].name

    # Single inference latency
    single_input = np.random.randn(1, n_feat).astype(np.float32)
    # Warmup
    for _ in range(10):
        sess.run(None, {inp_name: single_input})

    times = []
    for _ in range(BENCHMARK_RUNS):
        t0 = time.perf_counter()
        sess.run(None, {inp_name: single_input})
        times.append((time.perf_counter() - t0) * 1000)  # ms

    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)
    p99 = np.percentile(times, 99)

    STATS['latency_single'] = {'p50': p50, 'p95': p95, 'p99': p99}
    print(f"  Single inference ({BENCHMARK_RUNS} runs):")
    print(f"    P50: {p50:.3f} ms")
    print(f"    P95: {p95:.3f} ms")
    print(f"    P99: {p99:.3f} ms")

    assert_test(p50 < 50, "P50 latency < 50ms",
                f"{p50:.3f}ms", f"{p50:.3f}ms")

    # Batch inference latency
    for batch_size in [16, 64, 256]:
        batch_input = np.random.randn(batch_size, n_feat).astype(np.float32)
        times_b = []
        for _ in range(50):
            t0 = time.perf_counter()
            sess.run(None, {inp_name: batch_input})
            times_b.append((time.perf_counter() - t0) * 1000)
        p50_b = np.percentile(times_b, 50)
        per_item = p50_b / batch_size
        print(f"  Batch={batch_size:>4d}: P50={p50_b:.3f}ms ({per_item:.4f}ms/item)")

    STATS['latency_batch_256'] = p50_b


# ============================================================
# T15 - Batch Stress Test
# ============================================================
def test_batch_stress():
    section("T15 - Batch Stress Test")
    meta = STATS.get('meta', json.load(open(os.path.join(EXFIL_DIR, "expert6_meta.json"))))
    sess = STATS.get('onnx_sess')
    if sess is None:
        sess = ort.InferenceSession(os.path.join(EXFIL_DIR, "expert6_vae.onnx"))

    n_feat = meta['n_features']
    inp_name = sess.get_inputs()[0].name

    for batch_size in [1, 16, 128, 512, 1024]:
        try:
            batch = np.random.randn(batch_size, n_feat).astype(np.float32)
            result = sess.run(None, {inp_name: batch})
            ok = result[0].shape == (batch_size, n_feat)
            assert_test(ok, f"Batch size {batch_size}",
                        f"output={result[0].shape}",
                        f"output={result[0].shape}")
        except Exception as e:
            failed(f"Batch size {batch_size}", str(e)[:80])


# ============================================================
# T16 - Score Distribution Analysis
# ============================================================
def test_score_distribution():
    section("T16 - Score Distribution Analysis")
    rng = random.Random(SEED + 99)

    # Generate a large mixed set
    n_benign = 100
    n_exfil = 100

    benign_words = ['www', 'mail', 'api', 'cdn', 'auth', 'login', 'app', 'dev',
                    'cloud', 'portal', 'shop', 'store', 'blog', 'docs', 'help']
    benign_tlds = ['.com', '.org', '.net', '.io', '.co', '.dev']

    benign_set = []
    for _ in range(n_benign):
        w = rng.choice(benign_words)
        t = rng.choice(benign_tlds)
        benign_set.append(f"{w}.example{rng.randint(1,999)}{t}")

    exfil_set = []
    for _ in range(n_exfil):
        payload_len = rng.randint(15, 45)
        payload = ''.join(rng.choices(string.ascii_letters + string.digits, k=payload_len))
        exfil_set.append(f"{payload}.exfil{rng.randint(1,99)}.com")

    b_scores = get_recon_score(benign_set)
    e_scores = get_recon_score(exfil_set)

    # Compute ROC-AUC manually
    all_scores = np.concatenate([b_scores, e_scores])
    all_labels = np.array([0] * n_benign + [1] * n_exfil)

    # Sort by score descending
    order = np.argsort(-all_scores)
    sorted_labels = all_labels[order]

    # AUC via trapezoidal
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    tpr_arr = tp / n_exfil
    fpr_arr = fp / n_benign
    auc = np.trapz(tpr_arr, fpr_arr)
    if auc < 0:
        auc = -auc

    STATS['synthetic_auc'] = float(auc)

    # Best F1
    thresholds = np.linspace(np.min(all_scores), np.max(all_scores), 200)
    best_f1 = 0.0
    best_thr = 0.0
    for thr in thresholds:
        pred = (all_scores > thr).astype(int)
        tp_c = np.sum((pred == 1) & (all_labels == 1))
        fp_c = np.sum((pred == 1) & (all_labels == 0))
        fn_c = np.sum((pred == 0) & (all_labels == 1))
        prec = tp_c / max(tp_c + fp_c, 1)
        rec = tp_c / max(tp_c + fn_c, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    STATS['synthetic_best_f1'] = float(best_f1)

    print(f"  Synthetic test: {n_benign} benign + {n_exfil} exfil")
    print(f"  Benign scores:  mean={np.mean(b_scores):.4f}, std={np.std(b_scores):.4f}")
    print(f"  Exfil scores:   mean={np.mean(e_scores):.4f}, std={np.std(e_scores):.4f}")
    print(f"  ROC-AUC:        {auc:.4f}")
    print(f"  Best F1:        {best_f1:.4f} (at threshold={best_thr:.4f})")

    assert_test(auc >= 0.5, "Synthetic ROC-AUC >= 0.50",
                f"AUC={auc:.4f}", f"AUC={auc:.4f}")


# ============================================================
# Final Report
# ============================================================
def print_final_report():
    meta = STATS.get('meta', {})

    print("\n" + "#" * 70)
    print("#  EXPERT 6: DATA EXFILTRATION -- STATISTICS REPORT")
    print("#" * 70)

    print("\n  MODEL METADATA:")
    print(f"    Model:        {meta.get('model', 'N/A')}")
    print(f"    Type:         {meta.get('type', 'N/A')}")
    print(f"    Dataset:      {meta.get('dataset', 'N/A')}")
    print(f"    Features:     {meta.get('n_features', 'N/A')}")
    print(f"    Flip:         {meta.get('flip', 'N/A')}")

    print("\n  KAGGLE TRAINING METRICS:")
    print(f"    ROC-AUC:      {meta.get('roc_auc', 0):.4f}")
    print(f"    PR-AUC:       {meta.get('pr_auc', 0):.4f}")
    print(f"    Best F1:      {meta.get('best_f1', 0):.4f}")
    print(f"    Accuracy:     {meta.get('accuracy', 0):.4f}")

    print("\n  SCORER COMPARISON (on training selection set):")
    for name, auc in sorted(meta.get('scorer_aucs', {}).items(), key=lambda x: -x[1]):
        print(f"    {name:15s} {auc:.4f}")

    sim = STATS.get('industrial_sim', {})
    if sim:
        print("\n  INDUSTRIAL SIMULATION:")
        print(f"    TP={sim['tp']:3d}  FP={sim['fp']:3d}  TN={sim['tn']:3d}  FN={sim['fn']:3d}")
        print(f"    TPR:          {sim['tpr']:.2%}")
        print(f"    FPR:          {sim['fpr']:.2%}")
        print(f"    Precision:    {sim['precision']:.2%}")
        print(f"    F1:           {sim['f1']:.4f}")

    mitre = STATS.get('mitre_detection_by_tool', {})
    if mitre:
        print("\n  MITRE ATT&CK DNS TUNNEL DETECTION:")
        for tool, (det, total) in mitre.items():
            print(f"    {tool:15s} {det}/{total} ({det/total:.0%})")
        print(f"    Overall:      {STATS.get('mitre_overall_rate', 0):.0%}")

    if 'synthetic_auc' in STATS:
        print("\n  SYNTHETIC TEST METRICS:")
        print(f"    ROC-AUC:      {STATS['synthetic_auc']:.4f}")
        print(f"    Best F1:      {STATS['synthetic_best_f1']:.4f}")

    if 'dga_exfil_rate' in STATS:
        print(f"\n  DGA-EXFIL RATE:   {STATS['dga_exfil_rate']:.0%}")
    if 'slow_drip_rate' in STATS:
        print(f"  SLOW-DRIP RATE:   {STATS['slow_drip_rate']:.0%}")
    if 'adversarial_evasion_rate' in STATS:
        print(f"  EVASION RESIST:   {STATS['adversarial_evasion_rate']:.0%}")

    lat = STATS.get('latency_single', {})
    if lat:
        print("\n  LATENCY:")
        print(f"    Single P50:   {lat['p50']:.3f} ms")
        print(f"    Single P95:   {lat['p95']:.3f} ms")
        print(f"    Single P99:   {lat['p99']:.3f} ms")
        if 'latency_batch_256' in STATS:
            print(f"    Batch-256:    {STATS['latency_batch_256']:.3f} ms")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#  NetSentinel -- Expert 6: Data Exfiltration Test Suite")
    print(f"#  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"#  ONNX Runtime: {ort.__version__}")
    print(f"#  NumPy: {np.__version__}")
    print(f"#  Model dir: {EXFIL_DIR}")
    print("#" * 70)

    test_funcs = [
        test_artifact_integrity,
        test_metadata_schema,
        test_onnx_io_shapes,
        test_pipeline,
        test_obvious_exfil,
        test_obvious_benign,
        test_industrial_realistic,
        test_mitre_dns_tunnel,
        test_dga_exfil,
        test_slow_drip,
        test_edge_cases,
        test_adversarial,
        test_determinism,
        test_latency,
        test_batch_stress,
        test_score_distribution,
    ]

    for fn in test_funcs:
        try:
            fn()
        except Exception as e:
            failed(f"SECTION CRASHED: {fn.__name__}", traceback.format_exc()[-120:])

    # ---- Summary ----
    print_final_report()

    total_pass = sum(1 for r in RESULTS if r[2] == "PASS")
    total_fail = sum(1 for r in RESULTS if r[2] == "FAIL")
    total_skip = sum(1 for r in RESULTS if r[2] == "SKIP")
    total = total_pass + total_fail + total_skip

    print("\n" + "#" * 70)
    print(f"#  FINAL: {total_pass}/{total} PASSED, {total_fail} FAILED, {total_skip} SKIPPED")
    print("#" * 70)

    if total_fail > 0:
        print("\n  FAILURES:")
        for sec, name, status, detail in RESULTS:
            if status == "FAIL":
                print(f"    [{sec}] {name}: {detail}")
        print()
        sys.exit(1)
    else:
        print("\n  >>> ALL TESTS PASSED. Expert 6 verified for integration.\n")
        sys.exit(0)
