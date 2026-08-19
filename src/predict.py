"""
predict.py
----------
Live prediction pipeline for Smart Airbag Helmet.

Features per spec:
    - 3-class labels: Normal / Near-Crash / Crash
    - Consecutive-window multi-check before airbag deploy (prevents false triggers)
    - Persistent one-time deploy lock (file flag -- survives script restarts)
    - Crash probability threshold gate
    - Near-Crash warning output (LED/buzzer stub)

Usage:
    python src/predict.py                      # simulate with synthetic stream
    python src/predict.py --session 42         # replay session from dataset
    python src/predict.py --hardware           # enables GPIO output (Raspberry Pi)
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import joblib

# -------------------------------------------------
#  LABEL CONFIG  (3-class)
# -------------------------------------------------
LABEL_MAP = {
    0: ("Normal",     "o", "#2ecc71"),
    1: ("Near-Crash", "!", "#f1c40f"),
    2: ("Crash",      "X", "#e74c3c"),
}

SENSOR_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]
HG_COLS     = ["hg_ax", "hg_ay", "hg_az"]

# -------------------------------------------------
#  SAFETY CONFIG
# -------------------------------------------------
CRASH_PROB_THRESHOLD    = 0.70   # Crash probability must exceed this per window
CONSECUTIVE_WINDOWS_REQ = 3      # Number of consecutive Crash predictions required
NEAR_CRASH_THRESHOLD    = 0.40   # Near-Crash probability threshold for warning

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Persistent lock file — prevents re-deployment even after script restart
DEPLOY_LOCK_FILE = os.path.join(PROJECT_ROOT, "AIRBAG_DEPLOYED.lock")


# -------------------------------------------------
#  LOAD MODEL
# -------------------------------------------------

def load_model(model_dir: str = "models"):
    """Load best_model.pkl + metadata."""
    if not os.path.isabs(model_dir) and not os.path.exists(os.path.join(model_dir, "best_model.pkl")):
        alt_dir = os.path.join(PROJECT_ROOT, model_dir)
        if os.path.exists(os.path.join(alt_dir, "best_model.pkl")):
            model_dir = alt_dir

    model_path = os.path.join(model_dir, "best_model.pkl")
    meta_path  = os.path.join(model_dir, "model_meta.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No model found at {model_path}.\n"
            "Run: python src/train_model.py  (or run the full pipeline first)"
        )

    model = joblib.load(model_path)
    meta  = joblib.load(meta_path) if os.path.exists(meta_path) else {}

    print(f"[MODEL] Loaded : {meta.get('model_name', 'Unknown')}")
    print(f"        Accuracy : {meta.get('accuracy', '?'):.4f}")
    print(f"        F1 macro : {meta.get('f1_macro', '?'):.4f}")
    print(f"        Infer ms : {meta.get('infer_ms', '?'):.4f} ms/sample")
    print(f"        Features : {len(meta.get('feature_names', []))}")
    return model, meta



# -------------------------------------------------
#  DEPLOY LOCK HELPERS
# -------------------------------------------------

def is_deployed() -> bool:
    """Check if airbag has already been deployed (persistent file lock)."""
    return os.path.exists(DEPLOY_LOCK_FILE)


def set_deployed_lock():
    """Write the persistent lock file after airbag deployment."""
    with open(DEPLOY_LOCK_FILE, "w") as f:
        f.write(f"DEPLOYED at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"[LOCK] Deploy lock written: {DEPLOY_LOCK_FILE}")


def clear_deploy_lock():
    """Reset the lock (for testing only)."""
    if os.path.exists(DEPLOY_LOCK_FILE):
        os.remove(DEPLOY_LOCK_FILE)
        print("[LOCK] Deploy lock cleared (testing mode).")


# -------------------------------------------------
#  HARDWARE STUBS
# -------------------------------------------------

def trigger_airbag_gpio():
    """
    Pull the airbag GPIO pin HIGH.
    On Raspberry Pi: uses RPi.GPIO to set the pin high.
    In simulation: prints a message.
    """
    try:
        import RPi.GPIO as GPIO
        AIRBAG_PIN = 17   # BCM pin 17 — connect to MOSFET gate
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(AIRBAG_PIN, GPIO.OUT)
        GPIO.output(AIRBAG_PIN, GPIO.HIGH)
        print(f"[GPIO] Pin {AIRBAG_PIN} pulled HIGH — airbag signal sent!")
    except ImportError:
        print("[GPIO STUB] RPi.GPIO not available — would pull GPIO pin HIGH now.")
    except Exception as e:
        print(f"[GPIO ERROR] {e}")


def trigger_near_crash_warning():
    """
    Blink LED / sound buzzer for Near-Crash state.
    Stub: replace with GPIO pulse for actual hardware.
    """
    try:
        import RPi.GPIO as GPIO
        WARNING_PIN = 27   # BCM pin 27 — connect to LED/buzzer
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(WARNING_PIN, GPIO.OUT)
        for _ in range(3):
            GPIO.output(WARNING_PIN, GPIO.HIGH)
            time.sleep(0.05)
            GPIO.output(WARNING_PIN, GPIO.LOW)
            time.sleep(0.05)
    except ImportError:
        print("[WARNING STUB] Near-Crash: would pulse LED/buzzer 3x.")
    except Exception as e:
        print(f"[WARNING ERROR] {e}")


def send_sms_alert(message: str = "CRASH DETECTED — AIRBAG DEPLOYED"):
    """
    Send emergency SMS via SIM800L over UART.
    Stub: replace serial port path as needed.
    """
    try:
        import serial
        with serial.Serial("/dev/ttyS0", 9600, timeout=1) as ser:
            ser.write(b"AT+CMGF=1\r\n")
            time.sleep(0.5)
            ser.write(b'AT+CMGS="+91XXXXXXXXXX"\r\n')  # Replace with real number
            time.sleep(0.5)
            ser.write((message + "\x1A").encode())
            time.sleep(1)
        print("[SMS] Emergency alert sent via SIM800L.")
    except ImportError:
        print(f"[SMS STUB] pyserial not installed — would send: '{message}'")
    except Exception as e:
        print(f"[SMS ERROR] {e}")


def log_to_sd(entry: dict, log_path: str = "logs/flight_log.csv"):
    """Log sensor readings and predictions to SD card (file)."""
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    df = pd.DataFrame([entry])
    df.to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)


# -------------------------------------------------
#  SINGLE WINDOW PREDICTION
# -------------------------------------------------

def predict_window(model, window_df: pd.DataFrame, feature_names: list) -> dict:
    """
    Given a 200-row window DataFrame, extract features and predict.
    Returns dict with label, probabilities, deploy flag, warning flag.
    """
    try:
        from src.feature_engineering import extract_features_window
    except ImportError:
        from feature_engineering import extract_features_window

    feats    = extract_features_window(window_df)
    # Use a named DataFrame to avoid sklearn feature-name warning
    feat_df  = pd.DataFrame([[feats[f] for f in feature_names]], columns=feature_names)

    t0         = time.perf_counter()
    pred_label = int(model.predict(feat_df)[0])
    infer_ms   = (time.perf_counter() - t0) * 1000

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feat_df)[0]
    elif hasattr(model, "named_steps") and hasattr(
            model.named_steps.get("clf", None), "predict_proba"):
        proba = model.predict_proba(feat_df)[0]
    else:
        proba = np.zeros(3)
        proba[pred_label] = 1.0

    # Pad if model outputs fewer classes than expected
    while len(proba) < 3:
        proba = np.append(proba, 0.0)

    crash_prob     = float(proba[2]) if len(proba) > 2 else 0.0
    near_crash_prob = float(proba[1]) if len(proba) > 1 else 0.0

    label_name, symbol, color = LABEL_MAP[pred_label]

    return {
        "label"          : pred_label,
        "label_name"     : label_name,
        "symbol"         : symbol,
        "proba"          : proba,
        "crash_prob"     : crash_prob,
        "near_crash_prob": near_crash_prob,
        "infer_ms"       : infer_ms,
        "deploy"         : crash_prob > CRASH_PROB_THRESHOLD,
        "warn"           : near_crash_prob > NEAR_CRASH_THRESHOLD or pred_label == 1,
    }


# -------------------------------------------------
#  LIVE PREDICTION LOOP (with multi-check safety)
# -------------------------------------------------

def run_live_simulation(
    model,
    meta: dict,
    source: str      = "synthetic",
    dataset_path: str = None,
    session_id: int  = None,
    delay: float     = 0.001,    # 1 ms simulated inter-sample delay
    verbose: bool    = True,
    hardware: bool   = False,
    reset_lock: bool = False,
):
    """
    Simulate streaming IMU data at 1000 Hz through the prediction pipeline.

    Safety: Airbag fires only after CONSECUTIVE_WINDOWS_REQ consecutive
    windows all predict Crash above CRASH_PROB_THRESHOLD.

    source: 'synthetic' -> generate scenario data
            'replay'    -> replay a session from CSV
    """
    # ---- Import timing constants ----
    try:
        from src.data_generator import WINDOW_SIZE, STRIDE, SAMPLE_RATE_HZ
    except ImportError:
        from data_generator import WINDOW_SIZE, STRIDE, SAMPLE_RATE_HZ

    feature_names = meta.get("feature_names", [])

    if reset_lock:
        clear_deploy_lock()

    print("\n" + "=" * 60)
    print("  SMART AIRBAG HELMET — LIVE PREDICTION LOOP")
    print("  Pre-Impact Rider State Classifier")
    print("=" * 60)
    print(f"  Sample rate     : {SAMPLE_RATE_HZ} Hz")
    print(f"  Window size     : {WINDOW_SIZE} samples (200 ms)")
    print(f"  Crash threshold : crash_prob > {CRASH_PROB_THRESHOLD}")
    print(f"  Multi-check     : {CONSECUTIVE_WINDOWS_REQ} consecutive Crash windows")
    print(f"  Hardware mode   : {'ON' if hardware else 'OFF (simulation)'}")
    print("=" * 60 + "\n")

    # Check persistent lock
    if is_deployed():
        print("[LOCK] Airbag already deployed! System locked.")
        print("[LOCK] Clear lock with: python src/predict.py --reset-lock")
        return

    # ---- Build data stream ----
    if source == "replay" and dataset_path and session_id is not None:
        df      = pd.read_csv(dataset_path)
        sess_df = df[df["session_id"] == session_id].reset_index(drop=True)
        rows    = [sess_df.iloc[i] for i in range(len(sess_df))]
        actual_labels = sess_df["label"].tolist()
        print(f"  Replaying session {session_id} ({len(rows)} timesteps @ {SAMPLE_RATE_HZ} Hz)\n")
    else:
        try:
            from src.data_generator import _normal, _crash, _near_crash
        except ImportError:
            from data_generator import _normal, _crash, _near_crash

        rng  = np.random.default_rng(777)
        rows = []
        actual_labels = []
        # Scenario: Normal -> Near-Crash -> Crash -> Normal
        # Crash segment = 400 samples = 4 windows @stride=100 -> gate of 3 fires
        for gen, lbl, n in [
            (_normal,     0, 400),
            (_near_crash, 1, 300),
            (_crash,      2, 400),
            (_normal,     0, 200),
        ]:
            sig = gen(n, rng)
            all_cols = SENSOR_COLS + ["hg_ax", "hg_ay", "hg_az"]
            for i in range(n):
                rows.append({col: sig[col][i] for col in all_cols})
                actual_labels.append(lbl)
        print(f"  Scenario: Normal(400) -> Near-Crash(200) -> Crash(200) -> Normal(200)\n")

    buffer             = []
    consecutive_crash  = 0
    deploy_fired       = False
    step               = 0
    log_path           = os.path.join(os.path.dirname(__file__), "..", "logs", "prediction_log.csv")

    for i, row_data in enumerate(rows):
        # Fill buffer
        if isinstance(row_data, dict):
            buffer.append(row_data)
        else:
            buffer.append({col: row_data[col] for col in (SENSOR_COLS + ["hg_ax", "hg_ay", "hg_az"])
                           if col in row_data.index})

        # Only evaluate when buffer has a full window and we're at a stride boundary
        if len(buffer) >= WINDOW_SIZE and (len(buffer) - WINDOW_SIZE) % STRIDE == 0:
            window_df = pd.DataFrame(buffer[-WINDOW_SIZE:])
            result    = predict_window(model, window_df, feature_names)

            true_label = actual_labels[i] if i < len(actual_labels) else -1
            true_name  = LABEL_MAP.get(true_label, ("?", "?", "?"))[0]
            correct    = "OK" if result["label"] == true_label else "XX"

            crash_bar = "#" * int(result["crash_prob"] * 20) + "." * (20 - int(result["crash_prob"] * 20))

            if verbose:
                print(
                    f"  t={i:>5}ms  [{result['symbol']}] {result['label_name']:<12} "
                    f"| Crash: [{crash_bar}] {result['crash_prob']:.1%} "
                    f"| Infer: {result['infer_ms']:.2f}ms "
                    f"| True: {true_name:<12} {correct}"
                )

            # ---- Near-Crash warning ----
            if result["warn"] and not result["deploy"] and not deploy_fired:
                print(f"\n  [!] NEAR-CRASH WARNING at t={i} ms — {result['label_name']} ({result['near_crash_prob']:.1%})")
                if hardware:
                    trigger_near_crash_warning()

            # ---- Multi-check crash gate ----
            if result["deploy"] and not deploy_fired:
                consecutive_crash += 1
                print(f"\n  [GATE] Crash window #{consecutive_crash}/{CONSECUTIVE_WINDOWS_REQ} "
                      f"(crash_prob={result['crash_prob']:.1%})")

                if consecutive_crash >= CONSECUTIVE_WINDOWS_REQ:
                    # All checks passed — DEPLOY
                    print(f"\n  {'!'*60}")
                    print(f"  AIRBAG DEPLOY TRIGGERED at t={i} ms")
                    print(f"  Crash prob: {result['crash_prob']:.1%}")
                    print(f"  Multi-check: {consecutive_crash}/{CONSECUTIVE_WINDOWS_REQ} passed")
                    print(f"  {'!'*60}\n")

                    set_deployed_lock()
                    deploy_fired = True

                    if hardware:
                        trigger_airbag_gpio()
                        send_sms_alert(f"CRASH at t={i}ms | p={result['crash_prob']:.1%}")
            else:
                if not result["deploy"]:
                    consecutive_crash = 0   # Reset counter on non-crash window

            # ---- Log to SD ----
            log_entry = {
                "timestamp_ms"  : i,
                "label_pred"    : result["label"],
                "label_name"    : result["label_name"],
                "crash_prob"    : round(result["crash_prob"], 4),
                "nc_prob"       : round(result["near_crash_prob"], 4),
                "infer_ms"      : round(result["infer_ms"], 4),
                "deployed"      : deploy_fired,
            }
            log_to_sd(log_entry, log_path)
            step += 1
            time.sleep(delay)

    print("\n" + "=" * 60)
    print(f"  Simulation complete. Windows predicted: {step}")
    print(f"  Airbag deployed: {'YES' if deploy_fired else 'NO'}")
    print("=" * 60)


# -------------------------------------------------
#  CLI
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Airbag Helmet -- Live Predictor")
    parser.add_argument("--model-dir",   default="models",                              help="Path to models directory")
    parser.add_argument("--dataset",     default="data/synthetic/helmet_imu_raw.csv",   help="Path to raw dataset CSV")
    parser.add_argument("--session",     type=int,   default=None,                      help="Session ID to replay")
    parser.add_argument("--delay",       type=float, default=0.0,                       help="Inter-sample delay in seconds")
    parser.add_argument("--no-verbose",  action="store_true",                           help="Suppress per-window output")
    parser.add_argument("--hardware",    action="store_true",                           help="Enable GPIO / SMS hardware output")
    parser.add_argument("--reset-lock",  action="store_true",                           help="Clear the deploy lock (testing only)")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    model, meta = load_model(args.model_dir)

    dataset_path = args.dataset
    if not os.path.isabs(dataset_path) and not os.path.exists(dataset_path):
        alt_dataset = os.path.join(PROJECT_ROOT, dataset_path)
        if os.path.exists(alt_dataset):
            dataset_path = alt_dataset

    if args.session is not None and os.path.exists(dataset_path):
        run_live_simulation(model, meta,
                            source="replay",
                            dataset_path=dataset_path,
                            session_id=args.session,
                            delay=args.delay,
                            verbose=not args.no_verbose,
                            hardware=args.hardware,
                            reset_lock=args.reset_lock)
    else:
        run_live_simulation(model, meta,
                            source="synthetic",
                            delay=args.delay,
                            verbose=not args.no_verbose,
                            hardware=args.hardware,
                            reset_lock=args.reset_lock)
