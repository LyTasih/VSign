"""Nhập các lượt quay từ web app (webapp/) vào dữ liệu huấn luyện.

Web app tải mỗi lượt lên Google Drive thành một file zip: 30 ảnh JPEG vuông (00.jpg … 29.jpg, chụp ở
12 khung/giây, chưa lật gương) và meta.json; từ app bản 3 có thêm clip video gốc video.webm / video.mp4.
Script này trích keypoints bằng đúng MediaPipe Holistic Python dùng cho tập ViSL (keypoints.py), áp cùng
ngưỡng với collect_data.py.

Có clip video (mặc định dùng clip, --frames-only để dùng 30 ảnh như cũ): chạy Holistic trên mọi khung của
clip, tìm đoạn ký hiệu (cổ tay, đo so với vai trong cùng khung, rời vị trí nghỉ hơn ACTIVE_SW độ rộng vai ít nhất
MIN_ACTIVE_S), rồi đặt cửa sổ quanh đoạn đó sao cho, giống tập ViSL, ký hiệu chiếm khoảng 22–23/30 khung với
~3 khung nghỉ trước và ~4 khung nghỉ sau,
và lấy 30 khung cách đều theo thời gian (khung gần nhất, không nội suy, như prepare_visl.py). Nhờ vậy
người làm chậm hay bắt đầu trễ vẫn có trọn ký hiệu. Cử động đã kết thúc trước lúc khung chuyển đỏ (nhích tay
trong lúc đếm ngược) không tính là ký hiệu. Ký hiệu chạm mép đầu/cuối clip thì loại (bị cắt).
--review ghi ảnh 30 khung đã chọn của mỗi lượt vào data/web_review/ để duyệt bằng mắt.

Đầu ra:
    data/raw/<id>_<slug>/<tên lượt>.npy     (30, 1662), giống collect_data.py
    data/manifest.csv                       thêm một dòng mỗi lượt đạt
    data/web_import_log.csv                 mọi lượt đã xử lý (đạt / loại và lý do), để không xử lý lại
Thư mục nguồn có dạng <src>/<người>/buoi1|buoi2/*.zip (Apps Script tự chia); zip nằm thẳng ở <src> cũng được đọc.
Lượt bị loại được chuyển vào <src>/_rejected/. Nếu <src> là thư mục Drive đồng bộ, web app sẽ thấy
lượt đó còn thiếu và người quay chỉ cần quay lại.

Cách dùng (thư mục nguồn: thư mục Drive đã đồng bộ về máy, hoặc nơi chứa các zip đã tải về):
    python import_web_takes.py --src "~/Google Drive/My Drive/VSign_uploads"
    python import_web_takes.py --src data/web_inbox --dry-run      # chỉ liệt kê, không ghi
    python import_web_takes.py --src data/web_inbox --dry-run --review   # thêm ảnh duyệt vào data/web_review/
    python collect_data.py --summary                               # xem tiến độ sau khi nhập
"""
from __future__ import annotations

import os

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import csv
import json
import re
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

import collect_data as cd
import keypoints as kp

ROOT = Path(__file__).resolve().parent
NAME_RE = re.compile(r"^([a-z0-9]+)_s([12])_(\d{2})_([a-z0-9_]+)_(\d{8}-\d{6}-\d{3})$")
LOG_FIELDS = ["zip", "status", "reason", "file", "hand_ratio", "pose_ratio", "shoulder_width", "imported_at",
              "method", "sign_start_s", "sign_end_s", "window_start_s", "window_end_s", "clip_s"]
REJECTED_DIR = "_rejected"
FRAME_MS = round(1000 / kp.CAPTURE_FPS)   # timestamp cho chế độ VIDEO, giống nhịp chụp của web app

# Cắt từ video. Đo trên 931 mẫu ViSL của 30 ký hiệu (cùng cách đo hoạt động, theo vai): ký hiệu chiếm trung vị
# 22/30 khung, trước đó ~3 khung nghỉ, sau đó ~4 khung nghỉ.
REST_DEPTH = 0.8          # cổ tay thấp hơn đường vai ≥ 0,8 độ rộng vai = tư thế nghỉ (như webapp/config.js)
ACTIVE_SW = 0.3           # cổ tay rời vị trí nghỉ hơn 0,3 độ rộng vai = đang làm ký hiệu
LEAD_FRAMES, TAIL_FRAMES = 3, 4
EDGE_S = 0.15             # ký hiệu còn diễn ra trong 0,15 s đầu/cuối clip = bị cắt
MIN_SIGN_S = 0.3          # ngắn hơn: nhiễu, không phải ký hiệu
MIN_ACTIVE_S = 0.1        # đợt "rời nghỉ" ngắn hơn (một hai khung nhiễu) không tính là ký hiệu


def read_log(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {r["zip"] for r in csv.DictReader(f)}


def append_log(path: Path, row: dict) -> None:
    if path.exists():   # log cũ thiếu cột mới → viết lại với header đầy đủ, giữ nguyên các dòng cũ
        with path.open(newline="", encoding="utf-8-sig") as f:
            r = csv.DictReader(f)
            if r.fieldnames != LOG_FIELDS:
                old = list(r)
                with path.open("w", newline="", encoding="utf-8-sig") as g:
                    w = csv.DictWriter(g, fieldnames=LOG_FIELDS, extrasaction="ignore")
                    w.writeheader()
                    w.writerows(old)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig" if new else "utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def load_take(zip_path: Path) -> tuple[dict, list[np.ndarray], tuple[str, bytes] | None]:
    """→ (meta, 30 ảnh BGR, (tên, bytes) của clip video hoặc None)."""
    with zipfile.ZipFile(zip_path) as z:
        meta = json.loads(z.read("meta.json"))
        names = sorted(n for n in z.namelist() if n.lower().endswith(".jpg"))
        frames = [cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR) for n in names]
        vids = [n for n in z.namelist() if n.startswith("video.")]
        video = (vids[0], z.read(vids[0])) if vids else None
    return meta, frames, video


def read_video(name: str, data: bytes) -> tuple[list[np.ndarray], np.ndarray]:
    """Giải mã clip (WebM/MP4 từ MediaRecorder, nhịp khung không đều) → (khung BGR, thời điểm từng khung, giây)."""
    fd, path = tempfile.mkstemp(suffix=Path(name).suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        cap = cv2.VideoCapture(path)
        frames, ts = [], []
        while True:
            ok, im = cap.read()
            if not ok:
                break
            frames.append(im)
            ts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
        cap.release()
    finally:
        os.remove(path)
    ts = np.array(ts, dtype=float)
    if len(ts) > 1 and not np.all(np.diff(ts) > 0):   # thiếu timestamp: giả định nhịp đều từ số khung đọc được
        ts = np.arange(len(frames)) / 30.0
    return frames, ts


def activity(seq: np.ndarray) -> np.ndarray | None:
    """Mức rời tư thế nghỉ của mỗi khung: khoảng cách lớn nhất của hai cổ tay so với vị trí nghỉ, theo độ rộng vai.
    Vị trí cổ tay tính so với giữa hai vai trong CÙNG khung (chia cho độ rộng vai của khung đó), nên camera tự zoom/lia
    (ví dụ tự động căn khung của Windows) hay người hơi lắc không bị tính là làm ký hiệu. Vị trí nghỉ = trung vị các
    khung có cả hai cổ tay thấp hơn vai ≥ REST_DEPTH. None nếu không thấy khung nghỉ nào."""
    P = seq[:, :kp.N_POSE * 4].reshape(len(seq), kp.N_POSE, 4)
    seen = P[:, 11, 3] > 0
    if seen.sum() < 2:
        return None
    sw = np.linalg.norm(P[:, 11, :2] - P[:, 12, :2], axis=1)
    sw[~seen | (sw <= 0)] = np.nan
    rel = (P[:, [15, 16], :2] - ((P[:, 11, :2] + P[:, 12, :2]) / 2)[:, None, :]) / sw[:, None, None]
    rest = seen & (np.nan_to_num(rel[:, :, 1], nan=-1).min(1) >= REST_DEPTH)
    if rest.sum() < 2:
        return None
    base = np.nanmedian(rel[rest], axis=0)
    a = np.linalg.norm(rel - base, axis=2).max(1)
    a[~seen] = np.nan
    return a


def active_frames(a: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Chỉ số các khung đang làm ký hiệu, bỏ các đợt ngắn hơn MIN_ACTIVE_S (nhiễu nhận diện một hai khung)."""
    on = np.nan_to_num(a) > ACTIVE_SW
    keep = np.zeros_like(on)
    i = 0
    while i < len(on):
        if on[i]:
            j = i
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            if ts[j] - ts[i] >= MIN_ACTIVE_S:
                keep[i:j + 1] = True
            i = j + 1
        else:
            i += 1
    return np.flatnonzero(keep)


def drop_before_red(on: np.ndarray, ts: np.ndarray, t_red: float | None) -> np.ndarray:
    """Bỏ các đợt rời nghỉ đã kết thúc trước lúc khung chuyển đỏ (t_red, giây từ đầu clip): đó là người quay nhích
    hay chỉnh tay trong lúc đếm ngược, không phải ký hiệu. Đợt bắt đầu sớm nhưng kéo dài qua khung đỏ vẫn giữ."""
    if t_red is None or not len(on):
        return on
    runs = np.split(on, np.flatnonzero(np.diff(on) > 1) + 1)
    kept = [r for r in runs if ts[r[-1]] >= t_red]
    return np.concatenate(kept) if kept else on[:0]


def extract_from_video(video: tuple[str, bytes], model: Path, t_red: float | None = None):
    """Clip → (keypoints (30, 1662), tỉ lệ tay, người, mặt, thông tin đoạn cắt, ảnh 30 khung đã chọn) hoặc (lý do loại, info).
    t_red: lúc khung chuyển đỏ (giây từ đầu clip), để bỏ cử động trong lúc đếm ngược; None nếu không biết."""
    frames, ts = read_video(*video)
    info = {"method": "video", "clip_s": f"{ts[-1]:.2f}" if len(ts) else "0"}
    if len(frames) < kp.SEQUENCE_LENGTH:
        return f"video_frames:{len(frames)}", info
    feats, flags = [], []
    with kp.create_holistic(model) as holistic:
        last = -1
        for im, t in zip(frames, ts):
            stamp = max(int(round(t * 1000)), last + 1)
            last = stamp
            r = kp.detect(holistic, kp.center_square(im), stamp)
            feats.append(kp.extract_keypoints(r))
            flags.append((kp.has_hand(r), kp.has_pose(r), kp.has_face(r)))
    feats = np.stack(feats).astype(np.float32)
    a = activity(feats)
    if a is None:
        return "no_rest_pose", info
    on = drop_before_red(active_frames(a, ts), ts, t_red)
    if not len(on):
        return "no_sign", info
    t_on, t_off = ts[on[0]], ts[on[-1]]
    info.update(sign_start_s=f"{t_on:.2f}", sign_end_s=f"{t_off:.2f}")
    if t_off - t_on < MIN_SIGN_S:
        return "sign_too_short", info
    if t_on - ts[0] < EDGE_S:
        return "sign_cut_start", info
    if ts[-1] - t_off < EDGE_S:
        return "sign_cut_end", info
    # cửa sổ quanh ký hiệu: ký hiệu chiếm (30 - LEAD - TAIL)/30 như ViSL; thiếu khung nghỉ ở mép clip thì lấy tới mép
    core = kp.SEQUENCE_LENGTH - LEAD_FRAMES - TAIL_FRAMES
    span = t_off - t_on
    w0 = max(ts[0], t_on - span * LEAD_FRAMES / core)
    w1 = min(ts[-1], t_off + span * TAIL_FRAMES / core)
    info.update(window_start_s=f"{w0:.2f}", window_end_s=f"{w1:.2f}")
    pick = [int(np.abs(ts - t).argmin()) for t in np.linspace(w0, w1, kp.SEQUENCE_LENGTH)]
    ratios = [float(np.mean([flags[p][i] for p in pick])) for i in range(3)]
    return (feats[pick], *ratios, info, [frames[p] for p in pick]), info


def review_sheet(images: list[np.ndarray], title: str, path: Path) -> None:
    """Ảnh duyệt: 30 khung đã chọn (vuông, lật gương như người quay thấy), 3 hàng × 10."""
    tiles = [cv2.resize(cv2.flip(kp.center_square(im), 1), (120, 120)) for im in images]
    for i, t in enumerate(tiles):
        cv2.putText(t, str(i), (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    grid = np.vstack([np.hstack(tiles[r * 10:(r + 1) * 10]) for r in range(3)])
    head = np.full((26, grid.shape[1], 3), 255, np.uint8)
    cv2.putText(head, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack([head, grid]))


def extract(frames: list[np.ndarray], model: Path):
    """30 ảnh BGR → (keypoints (30, 1662), tỉ lệ thấy tay, người, mặt)."""
    seq, flags = [], []
    with kp.create_holistic(model) as holistic:
        for i, frame in enumerate(frames):
            result = kp.detect(holistic, kp.center_square(frame), i * FRAME_MS)
            seq.append(kp.extract_keypoints(result))
            flags.append((kp.has_hand(result), kp.has_pose(result), kp.has_face(result)))
    ratios = [float(np.mean([f[i] for f in flags])) for i in range(3)]
    return np.stack(seq).astype(np.float32), *ratios


def shoulder_width(seq: np.ndarray) -> float | None:
    """Trung vị độ rộng vai (theo cạnh khung vuông) trên các khung thấy rõ hai vai. ViSL: trung vị 0,33, 5–95% là 0,29–0,42."""
    pose = seq[:, :kp.N_POSE * 4].reshape(len(seq), kp.N_POSE, 4)
    l, r = pose[:, 11], pose[:, 12]
    ok = (l[:, 3] > 0.5) & (r[:, 3] > 0.5)
    if not ok.any():
        return None
    return float(np.median(np.linalg.norm(l[ok, :2] - r[ok, :2], axis=1)))


def process(zip_path: Path, labels: dict[int, dict], data_dir: Path, args) -> dict:
    stem = zip_path.stem
    row = {"zip": zip_path.name, "imported_at": datetime.now().isoformat(timespec="seconds")}
    m = NAME_RE.match(stem)
    if not m:
        return {**row, "status": "rejected", "reason": "bad_name"}
    signer, session, label_id = m.group(1), int(m.group(2)), int(m.group(3))
    label = labels.get(label_id)
    if label is None or label["slug"] != m.group(4):
        return {**row, "status": "rejected", "reason": "unknown_label"}
    try:
        meta, frames, video = load_take(zip_path)
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError) as e:
        return {**row, "status": "rejected", "reason": f"bad_zip:{type(e).__name__}"}

    picked = None
    if video and not args.frames_only:
        times = (meta.get("video") or {}).get("frame_times_ms") or []
        t_red = times[0] / 1000 if times else None   # ảnh đầu tiên được chụp đúng lúc khung chuyển đỏ
        out, info = extract_from_video(video, Path(args.model), t_red)
        row.update(info)
        if isinstance(out, str):
            return {**row, "status": "rejected", "reason": out}
        seq, hand, pose, face, _, picked = out
        fps = (kp.SEQUENCE_LENGTH - 1) / max(float(info["window_end_s"]) - float(info["window_start_s"]), 1e-6)
    else:
        if len(frames) != kp.SEQUENCE_LENGTH or any(f is None for f in frames):
            return {**row, "status": "rejected", "reason": f"frames:{len(frames)}"}
        seq, hand, pose, face = extract(frames, Path(args.model))
        row["method"] = "frames"
        picked, fps = frames, meta.get("fps", "")
    if args.review:
        ws = f"{row['window_start_s']}–{row['window_end_s']}s, ky hieu {row['sign_start_s']}–{row['sign_end_s']}s" \
            if row.get("method") == "video" else "30 anh goc"
        review_sheet(picked, f"{stem}  [{ws}]", data_dir / "web_review" / f"{stem}.jpg")
    row.update(hand_ratio=f"{hand:.2f}", pose_ratio=f"{pose:.2f}")
    if hand < args.min_hand_ratio:
        return {**row, "status": "rejected", "reason": "hands_missing"}
    if pose < args.min_pose_ratio:
        return {**row, "status": "rejected", "reason": "pose_missing"}
    sw = shoulder_width(seq)
    row["shoulder_width"] = "" if sw is None else f"{sw:.2f}"
    if sw is not None and sw > args.max_shoulder:
        return {**row, "status": "rejected", "reason": "too_close"}
    if sw is not None and sw < args.min_shoulder:
        return {**row, "status": "rejected", "reason": "too_far"}

    out = cd.label_dir(data_dir, label) / f"{stem}.npy"
    rel = out.relative_to(data_dir).as_posix()
    if not args.dry_run:
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, seq)
        cd.append_manifest(data_dir, {
            "file": rel, "label_id": label_id, "slug": label["slug"], "text": label["text"],
            "signer": signer, "session": session, "recorded_at": meta.get("recorded_at", ""),
            "frames": kp.SEQUENCE_LENGTH, "features": kp.FEATURE_DIM,
            "fps": f"{fps:.1f}" if isinstance(fps, float) else fps,   # cắt từ video: nhịp hiệu dụng sau khi co giãn
            "hand_ratio": f"{hand:.2f}", "pose_ratio": f"{pose:.2f}", "face_ratio": f"{face:.2f}",
            "mediapipe_version": mp.__version__,
        })
    return {**row, "status": "ok", "reason": "", "file": rel}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Nhập lượt quay từ web app (zip trên Drive) thành keypoints.")
    p.add_argument("--src", required=True, help="thư mục chứa các file zip (thư mục Drive đã đồng bộ hoặc đã tải về)")
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--labels", default=str(ROOT / "labels.json"))
    p.add_argument("--model", default=str(kp.DEFAULT_MODEL_PATH))
    p.add_argument("--min-hand-ratio", type=float, default=0.5, help="cùng ngưỡng với collect_data.py")
    p.add_argument("--min-pose-ratio", type=float, default=0.8)
    p.add_argument("--min-shoulder", type=float, default=0.22, help="độ rộng vai / cạnh khung tối thiểu (quá xa), như webapp/config.js")
    p.add_argument("--max-shoulder", type=float, default=0.48, help="độ rộng vai / cạnh khung tối đa (quá gần), như webapp/config.js")
    p.add_argument("--retry-rejected", action="store_true", help="xử lý lại cả các lượt đã bị loại trước đó")
    p.add_argument("--keep-rejected", action="store_true", help="không chuyển lượt bị loại vào thư mục _rejected")
    p.add_argument("--dry-run", action="store_true", help="chỉ kiểm tra, không ghi .npy / manifest / log")
    p.add_argument("--frames-only", action="store_true", help="dùng 30 ảnh như trước, bỏ qua clip video")
    p.add_argument("--review", action="store_true", help="ghi ảnh 30 khung đã chọn của mỗi lượt vào <data-dir>/web_review/")
    args = p.parse_args(argv)

    src, data_dir = Path(args.src).expanduser(), Path(args.data_dir)
    if not src.is_dir():
        sys.exit(f"Không thấy thư mục {src}")
    labels = {l["id"]: l for l in cd.load_labels(Path(args.labels))}
    log_path = data_dir / "web_import_log.csv"
    done = read_log(log_path)
    if args.retry_rejected and log_path.exists():
        with log_path.open(newline="", encoding="utf-8-sig") as f:
            done = {r["zip"] for r in csv.DictReader(f) if r["status"] == "ok"}
    existing = {Path(r).stem for r in _manifest_files(data_dir)}

    zips = [z for z in find_zips(src, include_rejected=args.retry_rejected) if z.name not in done and z.stem not in existing]
    print(f"{len(zips)} lượt mới trong {src} (bỏ qua {len(done)} lượt đã xử lý).")
    counts = {"ok": 0, "rejected": 0}
    for i, z in enumerate(zips, 1):
        row = process(z, labels, data_dir, args)
        counts[row["status"]] += 1
        note = f"tay {row.get('hand_ratio', '-')}, người {row.get('pose_ratio', '-')}, vai {row.get('shoulder_width', '-')}"
        if row.get("method") == "video":
            note += (f"; video {row.get('clip_s')}s, ký hiệu {row.get('sign_start_s', '-')}–{row.get('sign_end_s', '-')}s"
                     f", cửa sổ {row.get('window_start_s', '-')}–{row.get('window_end_s', '-')}s")
        elif row.get("method") == "frames":
            note += "; 30 ảnh"
        print(f"  [{i}/{len(zips)}] {z.name}: {'đạt' if row['status'] == 'ok' else 'LOẠI ' + row['reason']} ({note})")
        if not args.dry_run:
            data_dir.mkdir(parents=True, exist_ok=True)
            append_log(log_path, row)
            if row["status"] == "rejected" and not args.keep_rejected:
                # chuyển ra khỏi thư mục gốc: web app chỉ đếm file ở thư mục gốc, nên người quay sẽ thấy thiếu và quay lại
                if REJECTED_DIR not in z.relative_to(src).parts:
                    (src / REJECTED_DIR).mkdir(exist_ok=True)
                    z.replace(src / REJECTED_DIR / z.name)
    print(f"Xong: {counts['ok']} đạt, {counts['rejected']} loại.")
    if counts["rejected"] and not args.dry_run and not args.keep_rejected:
        print(f"Lượt bị loại đã chuyển vào {src / REJECTED_DIR}. Khi thư mục Drive đồng bộ xong, web app "
              "sẽ tính các lượt đó là còn thiếu để người quay quay lại.")
    return 0


def find_zips(src: Path, include_rejected: bool = False) -> list[Path]:
    """Mọi file zip trong src và thư mục con (<người>/buoi<n>/), bỏ qua _rejected/ trừ khi xử lý lại.
    Trùng tên (ví dụ cùng lượt ở thư mục gốc và thư mục con) thì chỉ lấy một."""
    seen, out = set(), []
    for z in sorted(src.rglob("*.zip")):
        rel = z.relative_to(src).parts
        if (REJECTED_DIR in rel[:-1] and not include_rejected) or z.name in seen:
            continue
        seen.add(z.name)
        out.append(z)
    return out


def _manifest_files(data_dir: Path) -> list[str]:
    path = data_dir / "manifest.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return [r["file"] for r in csv.DictReader(f)]


if __name__ == "__main__":
    sys.exit(main())
