# VSign

**Real-time Vietnamese Sign Language recognition with IoT.**

VSign films a person signing with a webcam, recognizes the sign with MediaPipe Holistic and an LSTM model, and sends the result over MQTT to an ESP32 device. The device shows the word on an OLED screen and signals it with an LED and a buzzer. The goal is to help deaf and hearing people communicate more easily.

> Course project for CE340 (University of Information Technology, VNU-HCM).

## How it works

```
Webcam ──► Camera node ──► MQTT broker ──► Inference service ──► MQTT broker ──► ESP32 (OLED, LED, buzzer)
           (MediaPipe       (Mosquitto)    (LSTM model)          sign/prediction └► Node-RED dashboard
            keypoints)
           sign/keypoints
```

1. **Camera node**: extracts 30 frames × 1662 keypoints per sign with MediaPipe Holistic and publishes them to `sign/keypoints`.
2. **Inference service**: classifies the sequence and publishes the label, confidence and timestamp to `sign/prediction`.
3. **ESP32 device**: shows the Vietnamese word (with accents) on a 1.3" 128×64 OLED.
4. **Node-RED dashboard**: shows the current label, a confidence chart and the recognition history.

## Status

| Part | Status |
|---|---|
| Keypoint extraction (`keypoints.py`) | ✅ Done |
| Webcam data collection tool (`collect_data.py`) | ✅ Done |
| ViSignLanguage-Video preprocessing (`prepare_visl.py`) | ✅ Done: 970 samples for 30 signs |
| Training pipeline (`train.py`, `features.py`) | ✅ Done: LSTM / GRU / Transformer, TFLite export |
| Remote collection web app (`webapp/`, `import_web_takes.py`) | ✅ Built, not deployed yet |
| Self-recorded data (2 signers × 2 sessions) | 🚧 To record |
| Trained model (≥ 85% on session-2 test) | 🚧 Waiting for data |
| MQTT nodes, ESP32 firmware, Node-RED dashboard | 🚧 In progress |

## Vocabulary

There are 30 everyday signs, listed in [`labels.json`](labels.json):

Chào, Cảm ơn, Đồng ý, Mời vào, Giúp, Xin phép, Tôi, Họ, Đâu, Ăn, Uống, Đi, Chạy, Nghe, Nhìn, Nói, Biết, Nhớ, Mua, Nghỉ ngơi, Thức dậy, Chậm lại, Nhà, Trường học, Bệnh viện, Hôm nay, Trưa, Chiều, Cứu, Khóc

## Features

Each frame becomes a vector of **1662** numbers, always in this order:

| Part | Points | Values |
|---|---|---|
| Pose | 33 × (x, y, z, visibility) | 132 |
| Face | 468 × (x, y, z) | 1404 |
| Left hand | 21 × (x, y, z) | 63 |
| Right hand | 21 × (x, y, z) | 63 |

A part that isn't detected is filled with zeros. One sample is a `.npy` file of shape **(30, 1662)**: one whole sign, from the rest pose before it to the rest pose after it, resampled in time to 30 frames.

- **ViSL** (`prepare_visl.py`): each clip is a whole rest-to-rest sign; all its clean frames are resampled to 30 (nearest frame).
- **Web app takes with a video clip** (`import_web_takes.py`): the sign is found as the span where a wrist, measured relative to the shoulders in the same frame, leaves its rest position by more than 0.3 shoulder widths for at least 0.1 s (movement that ends before the frame turns red, such as adjusting the hands during the countdown, is ignored), and a window is placed around it so the sign fills about 22–23 of the 30 frames, with about 3 rest frames before and 4 after (the median in ViSL), then resampled to 30. Signers who sign slowly or start late still give a whole sign.
- **Older takes without video** (`collect_data.py`, web app before version 3): 30 frames at 12 FPS from a fixed 2.5 s window.

> At inference time, the camera node must process frames the same way: center-square crop, un-mirrored frame, MediaPipe Holistic, then **segment each sign** (wrists leave the rest pose, then return to it), place the window around it the same way, and resample that span to 30 frames. A fixed 2.5 s sliding window would not match the training data. Use the constants in `import_web_takes.py` (`REST_DEPTH`, `ACTIVE_SW`, `LEAD_FRAMES`, `TAIL_FRAMES`).

## Setup

Requires Python 3.10+ and a webcam.

```bash
pip install -r requirements.txt
```

The MediaPipe model (`models/holistic_landmarker.task`) is downloaded automatically the first time it's needed.

## Usage

### Record your own data

```bash
python collect_data.py --signer tran --session 1   # session 1: 3 takes per sign (train/validation)
python collect_data.py --signer tran --session 2   # session 2, on another day: 2 takes per sign (test)
python collect_data.py --signer tran --session 1 --start 7   # start from sign #7
python collect_data.py --summary                    # check what has been recorded
```

Keys in the recording window:

| Key | Action |
|---|---|
| `SPACE` | Start / pause recording the current sign |
| `N` / `P` | Next / previous sign |
| `D` | Delete the last saved take |
| `C` | Switch camera |
| `Q` / `ESC` | Quit |

Output goes to `data/raw/<id>_<slug>/` and each take is logged in `data/manifest.csv`.

### Record remotely with the web app

Members who are far apart can record in the browser: each take is uploaded straight to a shared Google Drive folder, then imported on the training machine with the same MediaPipe pipeline. Setup and member instructions: [`webapp/README.md`](webapp/README.md).

```bash
python import_web_takes.py --src "<synced Drive folder>"   # zip per take → data/raw + data/manifest.csv
```

### Preprocess ViSignLanguage-Video

Download `dataset.zip` from [ViSignLanguage-Video on Hugging Face](https://huggingface.co/datasets/star092304/ViSignLanguage-Video) (CC-BY-4.0) into `data/external/visl_download/`, then run:

```bash
python prepare_visl.py --zip data/external/visl_download/dataset.zip   # safe to re-run; skips finished videos
python prepare_visl.py --report                                         # per-class statistics
```

The script drops corrupted single-color frames, merges duplicate videos, extracts keypoints, resamples each video to 30 frames, and rejects samples where hands or body are rarely visible. Output goes to `data/external/visl_keypoints/`.

### Train the models

```bash
pip install -r requirements-train.txt
python train.py --model lstm --features pose_hands   # one experiment
python train.py --all                                # 3 models × 2 feature sets = 6 experiments
```

`train.py` reads ViSL keypoints from `data/external/visl_keypoints/` and your recordings from `data/manifest.csv`.

- **Test set:** session 2 recordings, which are never seen during training. Until those exist, 15% of ViSL is held out instead, and a warning is printed.
- **Validation set:** 15% of the remaining data (ViSL + session 1), split per sign.
- **Feature sets:** `full` (1662 numbers) or `pose_hands` (258, no face).
- **Normalization:** keypoints are centered on the shoulders and scaled by shoulder width.
- **Augmentation:** time stretch, scale, shift, small rotation, noise, and mirroring (`pose_hands` only).

Each run is saved to `runs/<time>_<model>_<features>/`: `model.keras`, `model.tflite`, `config.json`, `metrics.json`, `confusion_matrix.png`, and `classification_report.txt`. All runs are compared in `runs/results.csv`.

> The inference service must call `features.preprocess(seq, config["features"])` on each (30, 1662) sequence before the model, exactly as in training.

## Project structure

```
VSign/
├── keypoints.py       # MediaPipe Holistic feature extraction (shared by all tools)
├── collect_data.py    # Webcam recording tool
├── prepare_visl.py    # ViSignLanguage-Video → keypoints
├── import_web_takes.py # Web app takes (zip on Drive) → keypoints
├── vercel.json        # Vercel deploy config for webapp/
├── webapp/            # Browser collection tool + Apps Script upload endpoint (deployed on Vercel)
├── features.py        # Normalization, feature sets, augmentation (training + inference)
├── train.py           # Train / compare LSTM, GRU, Transformer; export TFLite
├── labels.json        # The 30 signs
├── requirements.txt
├── requirements-train.txt
├── data/              # Not in git (datasets and keypoints)
├── models/            # Not in git (downloaded automatically)
└── runs/              # Not in git (training results)
```

## Tech stack

MediaPipe Holistic · TensorFlow / TFLite · Eclipse Mosquitto (MQTT) · paho-mqtt · ESP32 (Arduino, PubSubClient, U8g2) · Node-RED
