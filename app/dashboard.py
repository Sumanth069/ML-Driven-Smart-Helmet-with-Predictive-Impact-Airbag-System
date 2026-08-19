"""
dashboard.py
------------
Live monitoring dashboard for Smart Airbag Helmet.

Streams IMU data through the ML model and displays real-time state in the terminal.
Includes ASCII visualization of crash/near-crash probabilities and an event history log.

Run:
    python app/dashboard.py                     # synthetic demo
    python app/dashboard.py --session 5         # replay session 5
    python app/dashboard.py --reset-lock        # clear lock & run
    python app/dashboard.py --delay 0.05        # smooth terminal streaming
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import joblib
from collections import deque

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.feature_engineering import extract_features_window
from src.data_generator      import WINDOW_SIZE, STRIDE, SAMPLE_RATE_HZ
from src.predict             import (CRASH_PROB_THRESHOLD, CONSECUTIVE_WINDOWS_REQ,
                                     NEAR_CRASH_THRESHOLD, is_deployed,
                                     set_deployed_lock, trigger_airbag_gpio,
                                     trigger_near_crash_warning, send_sms_alert,
                                     log_to_sd)

SENSOR_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]
HG_COLS     = ["hg_ax", "hg_ay", "hg_az"]
ALL_COLS    = SENSOR_COLS + HG_COLS

RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
ORANGE = "\033[38;5;208m"
BG_RED = "\033[41m\033[97m\033[1m"

LABEL_STYLES = {
    0: ("Normal",     "[ OK ]",  GREEN),
    1: ("Near-Crash", "[ !! ]",  YELLOW),
    2: ("Crash",      "[CRASH]", RED),
}


def prob_bar(p: float, width: int = 30, is_crash: bool = False) -> str:
    filled = int(p * width)
    bar    = "█" * filled + "░" * (width - filled)
    if is_crash:
        color = RED if p > 0.4 else (YELLOW if p > 0.2 else GREEN)
    else:
        color = ORANGE if p > 0.4 else (YELLOW if p > 0.2 else GREEN)
    return f"{color}[{bar}]{RESET} {p:.1%}"


def draw_header(meta: dict):
    os.system("cls" if os.name == "nt" else "clear")
    print(f"{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  SMART AIRBAG HELMET — LIVE MONITORING DASHBOARD{RESET}")
    print(f"{'='*70}")
    print(f"  Model     : {meta.get('model_name','?')}  |  "
          f"Acc={meta.get('accuracy',0):.1%}  |  "
          f"F1={meta.get('f1_macro',0):.1%}")
    print(f"  Spec      : {SAMPLE_RATE_HZ} Hz | {WINDOW_SIZE} smp/window (200ms) | "
          f"Crash gate: {CONSECUTIVE_WINDOWS_REQ} windows")
    print(f"{'='*70}\n")


def draw_state(result: dict, t_ms: int, consecutive: int, n_windows: int, event_log: list, deploy_fired: bool):
    label      = result["label"]
    label_name = result["label_name"]
    symbol, lbl_str, color = LABEL_STYLES.get(label, ("?", "[ ?? ]", ""))

    crash_p = result["crash_prob"]
    nc_p    = result["near_crash_prob"]
    inf_ms  = result["infer_ms"]

    print(f"  t={t_ms:>6}ms  {color}{BOLD}{lbl_str} {label_name:<12}{RESET}  "
          f"Infer: {inf_ms:.2f}ms  Win: {n_windows:>5}\n")

    print(f"  Crash Probability : {prob_bar(crash_p, 30, is_crash=True)}")
    print(f"  Near-Crash Prob   : {prob_bar(nc_p,    30, is_crash=False)}\n")

    # ---- STATUS ALERT BANNERS ----
    if deploy_fired:
        print(f"  {BG_RED} 💥💥💥 AIRBAG DEPLOYED! SOLENOID TRIGGERED & LOCK WRITTEN 💥💥💥 {RESET}\n")
    elif label == 2:
        print(f"  {RED}{BOLD}┌────────────────────────────────────────────────────────────┐{RESET}")
        print(f"  {RED}{BOLD}│ 🚨 CRASH DETECTED! Gate Check: {consecutive}/{CONSECUTIVE_WINDOWS_REQ} consecutive windows  │{RESET}")
        print(f"  {RED}{BOLD}└────────────────────────────────────────────────────────────┘{RESET}\n")
    elif label == 1 or nc_p > NEAR_CRASH_THRESHOLD:
        print(f"  {YELLOW}{BOLD}┌────────────────────────────────────────────────────────────┐{RESET}")
        print(f"  {YELLOW}{BOLD}│ ⚠️ NEAR-CRASH WARNING! (Pothole / Sudden Brake / Near-Miss) │{RESET}")
        print(f"  {YELLOW}{BOLD}│    LED / Buzzer Pulse Triggered (GPIO 27)                  │{RESET}")
        print(f"  {YELLOW}{BOLD}└────────────────────────────────────────────────────────────┘{RESET}\n")
    else:
        print(f"  {GREEN}[ SYSTEM NORMAL ] Riding Telemetry Nominal.{RESET}\n")

    # ---- EVENT LOG HISTORY ----
    print(f"  {BOLD}{CYAN}── RECENT ALERTS & EVENTS HISTORY ───────────────────────────{RESET}")
    if not event_log:
        print(f"  {CYAN}(Monitoring stream for events...){RESET}")
    else:
        for ev in event_log[-6:]:
            print(f"  {ev}")
    print(f"  {'─'*60}")


def run_dashboard(model, meta, stream, hardware=False, delay=0.04):
    feature_names = meta.get("feature_names", [])
    buffer        = deque(maxlen=WINDOW_SIZE * 2)
    consecutive   = 0
    deploy_fired  = is_deployed()
    n_windows     = 0
    samples_seen  = 0
    event_log     = []
    last_event_state = None

    draw_header(meta)

    if deploy_fired:
        print("\033[91m[LOCKED] Airbag already deployed! Pass --reset-lock to continue.\033[0m")
        return

    log_path = os.path.join(project_root, "logs", "dashboard_log.csv")
    stride_counter = 0

    for sample in iter(stream.read_sample, None):
        buffer.append(sample)
        samples_seen  += 1
        stride_counter += 1

        if len(buffer) >= WINDOW_SIZE and stride_counter >= STRIDE:
            stride_counter = 0
            window_df = pd.DataFrame(list(buffer)[-WINDOW_SIZE:])

            t0         = time.perf_counter()
            feats      = extract_features_window(window_df)
            feat_vec   = np.array([[feats[f] for f in feature_names]])
            pred_label = int(model.predict(feat_vec)[0])
            infer_ms   = (time.perf_counter() - t0) * 1000

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(feat_vec)[0]
            else:
                proba = np.zeros(3)
                proba[pred_label] = 1.0
            while len(proba) < 3:
                proba = np.append(proba, 0.0)

            crash_prob = float(proba[2])
            nc_prob    = float(proba[1])

            result = {
                "label"          : pred_label,
                "label_name"     : LABEL_STYLES.get(pred_label, ("?","?","?"))[0],
                "crash_prob"     : crash_prob,
                "near_crash_prob": nc_prob,
                "infer_ms"       : infer_ms,
            }

            n_windows += 1
            t_ms = int(samples_seen / SAMPLE_RATE_HZ * 1000)

            # Near-crash warning logic
            if pred_label == 1 or nc_prob > NEAR_CRASH_THRESHOLD:
                if last_event_state != 1:
                    event_log.append(f"{YELLOW}[{t_ms:>5}ms] ⚠️  NEAR-CRASH WARNING (NC Prob: {nc_prob:.1%}){RESET}")
                    last_event_state = 1
                if hardware:
                    trigger_near_crash_warning()

            # Crash gate logic
            if crash_prob > CRASH_PROB_THRESHOLD:
                consecutive += 1
                if last_event_state != 2:
                    event_log.append(f"{RED}[{t_ms:>5}ms] 🚨 CRASH DETECTED — Window {consecutive}/{CONSECUTIVE_WINDOWS_REQ} (Crash Prob: {crash_prob:.1%}){RESET}")
                    last_event_state = 2

                if consecutive >= CONSECUTIVE_WINDOWS_REQ and not deploy_fired:
                    event_log.append(f"{BG_RED}[{t_ms:>5}ms] 💥 AIRBAG DEPLOYED! Lock file written.{RESET}")
                    set_deployed_lock()
                    deploy_fired = True
                    if hardware:
                        trigger_airbag_gpio()
                        send_sms_alert(f"CRASH t={t_ms}ms p={crash_prob:.1%}")
            else:
                consecutive = 0
                if pred_label == 0 and last_event_state != 0:
                    event_log.append(f"{GREEN}[{t_ms:>5}ms] ✅ System state nominal (Normal riding){RESET}")
                    last_event_state = 0

            # Redraw state
            draw_header(meta)
            draw_state(result, t_ms, consecutive, n_windows, event_log, deploy_fired)

            log_to_sd({
                "t_ms": t_ms, "label": pred_label,
                "crash_prob": round(crash_prob, 4),
                "nc_prob"   : round(nc_prob, 4),
                "infer_ms"  : round(infer_ms, 4),
                "deployed"  : deploy_fired,
            }, log_path)

            if delay > 0:
                time.sleep(delay)

    print(f"\n  Dashboard complete. Windows: {n_windows} | Deployed: {deploy_fired}")


# -------------------------------------------------
#  CLI
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Airbag Helmet -- Live Dashboard")
    parser.add_argument("--model-dir",  default=os.path.join(project_root, "models"), help="Model directory")
    parser.add_argument("--dataset",    default=os.path.join(project_root, "data", "synthetic", "helmet_imu_raw.csv"))
    parser.add_argument("--session",    type=int, default=None)
    parser.add_argument("--hardware",   action="store_true")
    parser.add_argument("--port",       default="/dev/ttyUSB0",   help="Serial port (hardware mode)")
    parser.add_argument("--reset-lock", action="store_true",     help="Reset deployment lock before running")
    parser.add_argument("--delay",      type=float, default=0.04, help="Pacing delay in seconds per window (default 0.04s)")
    args = parser.parse_args()

    from src.predict import clear_deploy_lock
    if args.reset_lock:
        clear_deploy_lock()

    model_dir = args.model_dir
    if not os.path.isabs(model_dir) and not os.path.exists(os.path.join(model_dir, "best_model.pkl")):
        alt_dir = os.path.join(project_root, model_dir)
        if os.path.exists(os.path.join(alt_dir, "best_model.pkl")):
            model_dir = alt_dir

    model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    meta  = joblib.load(os.path.join(model_dir, "model_meta.pkl"))

    # Build stream
    if args.hardware:
        from src.raspberry_pi_interface import SerialIMUReader, gpio_setup
        gpio_setup()
        stream = SerialIMUReader(args.port)
    elif args.session is not None:
        from src.raspberry_pi_interface import SyntheticIMUStream
        dataset_path = args.dataset
        if not os.path.isabs(dataset_path) and not os.path.exists(dataset_path):
            alt_dataset = os.path.join(project_root, dataset_path)
            if os.path.exists(alt_dataset):
                dataset_path = alt_dataset
        df      = pd.read_csv(dataset_path)
        sess_df = df[df["session_id"] == args.session].reset_index(drop=True)

        class ReplayStream:
            def __init__(self, df):
                self._rows = [df.iloc[i].to_dict() for i in range(len(df))]
                self._idx  = 0
            def read_sample(self):
                if self._idx >= len(self._rows): return None
                r = self._rows[self._idx]; self._idx += 1
                return {c: r.get(c, 0.0) for c in ALL_COLS}
            def close(self): pass

        stream = ReplayStream(sess_df)
    else:
        from src.raspberry_pi_interface import SyntheticIMUStream
        stream = SyntheticIMUStream()

    try:
        run_dashboard(model, meta, stream, hardware=args.hardware, delay=args.delay)
    finally:
        if hasattr(stream, "close"):
            stream.close()
