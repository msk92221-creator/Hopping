#!/usr/bin/env python3
"""
generate_ultimate_dataset.py
============================
실측 IQ 데이터를 Seed로 사용하여 YOLOv8 학습용
'강건한(Robust) 호핑 신호 데이터셋'을 자동 생성하는 스크립트.

주요 기능:
  1) 실측 IQ(.npy) 로드 → 신호 구간 추출
  2) 주파수/시간 호핑(Frequency/Time Hopping) 시뮬레이션
  3) 채널 손상(CFO, Rayleigh Fading, AWGN) 적용
  4) 가변 NFFT STFT → 640×640 스펙트로그램 생성
  5) YOLO 포맷 바운딩 박스 자동 라벨링

Author : DSP/AI System Architect
Date   : 2026-02-14
"""

import os
import glob
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from scipy.signal import stft as scipy_stft
import cv2
import matplotlib
matplotlib.use("Agg")  # GUI 없는 환경에서도 동작하도록 백엔드 설정
import matplotlib.pyplot as plt


# ============================================================================
#  [0] 전역 설정 (Global Configuration)
# ============================================================================

# YOLO 출력 이미지 크기
YOLO_IMG_SIZE = 640

# 가변 NFFT 후보 리스트
NFFT_CANDIDATES = [256, 512, 1024, 2048, 4096]

# Overlap 비율 범위 (NFFT 대비)
OVERLAP_RATIO_MIN = 0.50
OVERLAP_RATIO_MAX = 0.75

# 한 이미지당 배치할 홉 신호 개수 범위
# ★ 실제 호핑 시나리오: 한 관측 윈도우에 수십 개의 홉이 존재
MIN_HOPS_PER_IMAGE = 5
MAX_HOPS_PER_IMAGE = 50

# 개별 홉 신호의 길이 범위 (샘플 수)
# ★ STFT 윈도우(최대 4096) 대비 충분한 길이 필요 → 최소 1024
#    짧은 체류(1024 = 0.1ms @10MHz) ~ 긴 체류(16384 = 1.6ms @10MHz)
HOP_LEN_MIN = 1024
HOP_LEN_MAX = 16384

# SNR 범위 (dB)
SNR_MIN_DB = -10.0
SNR_MAX_DB = 20.0

# CFO 최대 편차 (정규화 주파수 기준, Fs 대비 비율)
CFO_MAX_RATIO = 0.01

# Rayleigh Fading 블록 크기 (샘플 단위)
FADING_BLOCK_SIZE = 64

# YOLO 클래스 ID (호핑 신호 = 0)
HOPPING_CLASS_ID = 0


# ============================================================================
#  [1] 데이터 로드 함수 (Seed Loading)
# ============================================================================

def load_seed_signals(
    seed_dir: str,
    min_power_ratio: float = 0.1,
) -> List[np.ndarray]:
    """
    실측 IQ 데이터 파일(.npy)에서 신호 구간만 추출하여 리스트로 반환.
    다양한 길이의 세그먼트를 추출하여 호핑 시뮬레이션의 다양성을 확보.

    Parameters
    ----------
    seed_dir : str
        .npy 파일들이 위치한 디렉토리 경로.
    min_power_ratio : float
        전체 평균 파워 대비 이 비율 이상인 구간만 '신호'로 간주.

    Returns
    -------
    List[np.ndarray]
        추출된 IQ 신호 세그먼트 리스트. 각 원소는 complex64/128 배열.
        다양한 길이(HOP_LEN_MIN ~ HOP_LEN_MAX)의 세그먼트 포함.
    """
    seed_files = sorted(glob.glob(os.path.join(seed_dir, "*.npy")))
    if not seed_files:
        print(f"[경고] '{seed_dir}' 에서 .npy 파일을 찾지 못했습니다.")
        print("[안내] 내장 합성 신호(Chirp/Burst)를 Seed로 대체합니다.")
        return _generate_synthetic_seeds(count=50)

    segments: List[np.ndarray] = []

    # 다양한 세그먼트 길이로 추출 (STFT 윈도우 이상의 길이)
    extract_lengths = [1024, 2048, 4096, 8192, 16384]

    for fpath in seed_files:
        # .npy 로드 (complex IQ 데이터 가정)
        iq_data = np.load(fpath).astype(np.complex128)

        # 파워 계산 (슬라이딩 윈도우)
        power = np.abs(iq_data) ** 2
        avg_power = np.mean(power) if np.mean(power) > 0 else 1e-12
        threshold = avg_power * min_power_ratio

        # 신호 구간 탐색: 다양한 길이로 추출
        above = power > threshold
        for segment_len in extract_lengths:
            idx = 0
            while idx < len(iq_data) - segment_len:
                if above[idx]:
                    seg = iq_data[idx : idx + segment_len]
                    # 세그먼트 파워 검증
                    if np.mean(np.abs(seg) ** 2) > threshold:
                        # 정규화: 최대 진폭 기준 0~1 스케일
                        seg = seg / (np.max(np.abs(seg)) + 1e-12)
                        segments.append(seg)
                    idx += segment_len  # 다음 구간으로 점프
                else:
                    idx += 1

    if not segments:
        print("[경고] 유효한 신호 구간을 찾지 못했습니다. 합성 신호로 대체합니다.")
        return _generate_synthetic_seeds(count=50)

    print(f"[로드 완료] 총 {len(segments)}개 신호 세그먼트 추출됨 "
          f"(파일 {len(seed_files)}개, "
          f"길이 범위: {min(len(s) for s in segments)}~{max(len(s) for s in segments)} 샘플)")
    return segments


def _generate_synthetic_seeds(
    count: int = 50,
) -> List[np.ndarray]:
    """
    실측 데이터가 없을 때 사용할 합성 Seed 신호 생성.
    Chirp, 단일 톤(CW), BPSK Burst 등 다양한 파형을
    다양한 길이(HOP_LEN_MIN ~ HOP_LEN_MAX)로 생성.
    """
    seeds: List[np.ndarray] = []

    for i in range(count):
        # ★ 매 Seed마다 랜덤한 길이 (짧은 홉 ~ 긴 홉)
        segment_len = random.randint(HOP_LEN_MIN, HOP_LEN_MAX)
        t = np.arange(segment_len, dtype=np.float64)

        waveform_type = random.choice(["chirp", "cw", "bpsk"])

        if waveform_type == "chirp":
            # 선형 Chirp: 주파수가 f0에서 f1로 선형 변화
            f0 = random.uniform(0.01, 0.1)
            f1 = random.uniform(0.1, 0.4)
            phase = 2.0 * np.pi * (f0 * t + (f1 - f0) / (2.0 * segment_len) * t ** 2)
            sig = np.exp(1j * phase)

        elif waveform_type == "cw":
            # 단일 톤 (Continuous Wave)
            freq = random.uniform(0.05, 0.35)
            sig = np.exp(1j * 2.0 * np.pi * freq * t)

        else:
            # BPSK Burst
            symbol_len = random.choice([4, 8, 16, 32])
            n_symbols = max(segment_len // symbol_len, 1)
            bits = np.random.choice([-1, 1], size=n_symbols)
            baseband = np.repeat(bits, symbol_len)[:segment_len].astype(np.complex128)
            carrier_freq = random.uniform(0.05, 0.3)
            sig = baseband * np.exp(1j * 2.0 * np.pi * carrier_freq * t)

        # 정규화
        sig = sig / (np.max(np.abs(sig)) + 1e-12)
        seeds.append(sig)

    lens = [len(s) for s in seeds]
    print(f"[합성 Seed] {count}개 합성 신호 생성 완료 "
          f"(chirp/cw/bpsk, 길이: {min(lens)}~{max(lens)} 샘플)")
    return seeds


# ============================================================================
#  [2] 채널 손상 모델링 (Channel Impairment Augmentation)
# ============================================================================

def apply_cfo(signal: np.ndarray, fs: float) -> Tuple[np.ndarray, float]:
    """
    CFO (Carrier Frequency Offset) 적용.
    랜덤한 주파수 오차를 신호에 추가.

    Parameters
    ----------
    signal : np.ndarray
        입력 IQ 신호 (complex).
    fs : float
        샘플링 레이트 (Hz).

    Returns
    -------
    Tuple[np.ndarray, float]
        CFO 적용된 신호, 실제 적용된 CFO 값 (Hz).
    """
    # 정규화 주파수 오차: -CFO_MAX_RATIO ~ +CFO_MAX_RATIO (Fs 대비)
    cfo_normalized = random.uniform(-CFO_MAX_RATIO, CFO_MAX_RATIO)
    cfo_hz = cfo_normalized * fs

    n = np.arange(len(signal), dtype=np.float64)
    cfo_signal = signal * np.exp(1j * 2.0 * np.pi * cfo_normalized * n)

    return cfo_signal, cfo_hz


def apply_rayleigh_fading(signal: np.ndarray) -> np.ndarray:
    """
    Rayleigh Fading 적용.
    블록 단위로 Rayleigh 분포의 감쇠 계수를 곱하여
    신호 크기 변동(Fluctuation)을 시뮬레이션.

    Parameters
    ----------
    signal : np.ndarray
        입력 IQ 신호 (complex).

    Returns
    -------
    np.ndarray
        Fading 적용된 신호.
    """
    sig_len = len(signal)
    faded = signal.copy()

    # 블록 단위 Rayleigh 계수 생성
    n_blocks = int(np.ceil(sig_len / FADING_BLOCK_SIZE))
    fading_coeffs = (np.random.randn(n_blocks) + 1j * np.random.randn(n_blocks)) / np.sqrt(2.0)

    for b in range(n_blocks):
        start = b * FADING_BLOCK_SIZE
        end = min(start + FADING_BLOCK_SIZE, sig_len)
        faded[start:end] *= fading_coeffs[b]

    return faded


def apply_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    AWGN (Additive White Gaussian Noise) 적용.

    Parameters
    ----------
    signal : np.ndarray
        입력 IQ 신호 (complex).
    snr_db : float
        목표 SNR (dB).

    Returns
    -------
    np.ndarray
        노이즈가 추가된 신호.
    """
    sig_power = np.mean(np.abs(signal) ** 2)
    if sig_power < 1e-20:
        sig_power = 1e-12
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))

    noise = np.sqrt(noise_power / 2.0) * (
        np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    )
    return signal + noise


def augment_signal(signal: np.ndarray, fs: float) -> np.ndarray:
    """
    CFO + Rayleigh Fading 을 순차적으로 적용하는 통합 함수.
    (AWGN은 전체 프레임에 별도 적용하므로 여기서는 제외)

    Parameters
    ----------
    signal : np.ndarray
        입력 IQ 신호.
    fs : float
        샘플링 레이트.

    Returns
    -------
    np.ndarray
        손상 모델링이 적용된 신호.
    """
    # 1단계: CFO 적용
    sig_cfo, _ = apply_cfo(signal, fs)

    # 2단계: Rayleigh Fading 적용
    sig_faded = apply_rayleigh_fading(sig_cfo)

    return sig_faded


# ============================================================================
#  [3] 호핑 시뮬레이션 (Frequency/Time Hopping)
# ============================================================================

def frequency_shift(signal: np.ndarray, shift_freq: float, fs: float) -> np.ndarray:
    """
    신호를 지정된 주파수만큼 이동 (Frequency Shift).
    exp(j * 2 * pi * shift_freq * t)를 곱하는 방식.

    Parameters
    ----------
    signal : np.ndarray
        입력 IQ 신호.
    shift_freq : float
        이동할 주파수 (Hz). 양수=상향, 음수=하향.
    fs : float
        샘플링 레이트 (Hz).

    Returns
    -------
    np.ndarray
        주파수 이동된 신호.
    """
    n = np.arange(len(signal), dtype=np.float64)
    t = n / fs  # 시간 벡터 (초)
    shifted = signal * np.exp(1j * 2.0 * np.pi * shift_freq * t)
    return shifted


def place_hops_in_frame(
    seeds: List[np.ndarray],
    frame_len: int,
    fs: float,
    n_hops: int,
    nfft: int = 1024,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    배경 노이즈 프레임에 여러 개의 호핑 신호를 랜덤 배치.

    Parameters
    ----------
    seeds : List[np.ndarray]
        Seed 신호 세그먼트 리스트.
    frame_len : int
        전체 프레임 길이 (샘플 수).
    fs : float
        샘플링 레이트 (Hz).
    n_hops : int
        배치할 홉 개수.
    nfft : int
        현재 STFT 윈도우 크기. 홉 길이의 하한을 결정하는 데 사용.

    Returns
    -------
    Tuple[np.ndarray, List[Dict]]
        - 합성된 프레임 (complex 배열)
        - 각 홉의 물리 정보 리스트:
          [{"t_start": float, "t_end": float,
            "f_center": float, "f_bw": float}, ...]
    """
    # 빈 프레임 (노이즈는 나중에 별도 추가)
    frame = np.zeros(frame_len, dtype=np.complex128)
    hop_info_list: List[Dict] = []

    # ★ 최소 홉 길이: NFFT의 2배 이상이어야 STFT에서 최소 2~3 타임빈에 걸침
    #    이래야 스펙트로그램에서 눈에 보이고 bbox도 유효한 크기가 됨
    effective_hop_min = max(HOP_LEN_MIN, nfft * 2)
    effective_hop_max = min(HOP_LEN_MAX, frame_len // 2)
    if effective_hop_min > effective_hop_max:
        effective_hop_min = effective_hop_max

    for _ in range(n_hops):
        # 랜덤 Seed 선택
        seed = random.choice(seeds).copy()

        # ★ 홉 길이 결정: NFFT 연동 최소 길이 보장
        desired_len = random.randint(effective_hop_min, effective_hop_max)

        # Seed가 desired_len보다 짧으면 반복(tile)으로 늘림
        if len(seed) < desired_len:
            reps = int(np.ceil(desired_len / len(seed)))
            seed = np.tile(seed, reps)
        seed = seed[:desired_len]
        seg_len = len(seed)

        # ── 채널 손상 적용 ──
        seed = augment_signal(seed, fs)

        # ── 랜덤 주파수 이동 ──
        # Nyquist 범위 내에서 랜덤 중심 주파수 선택
        max_shift = fs * 0.4  # ±40% Fs 범위
        shift_freq = random.uniform(-max_shift, max_shift)
        seed = frequency_shift(seed, shift_freq, fs)

        # ── 랜덤 시간 위치에 삽입 ──
        max_start = frame_len - seg_len
        if max_start <= 0:
            t_start_sample = 0
        else:
            t_start_sample = random.randint(0, max_start)

        t_end_sample = t_start_sample + seg_len

        # 랜덤 진폭 스케일링 (0.3 ~ 1.0)
        amplitude = random.uniform(0.3, 1.0)
        frame[t_start_sample:t_end_sample] += amplitude * seed

        # ── 물리 정보 기록 ──
        # 신호의 대역폭(BW) 추정: Seed 길이와 파형 유형에 따라 달라지지만,
        # 단순화를 위해 Seed의 99% 파워 대역폭을 계산
        bw = _estimate_bandwidth(seed, fs)

        hop_info = {
            "t_start_sample": int(t_start_sample),
            "t_end_sample": int(t_end_sample),
            "t_start_sec": t_start_sample / fs,
            "t_end_sec": t_end_sample / fs,
            "f_center_hz": shift_freq,       # 베이스밴드 기준 중심 주파수
            "f_bw_hz": bw,                   # 추정 대역폭
            "f_min_hz": shift_freq - bw / 2, # 하한 주파수
            "f_max_hz": shift_freq + bw / 2, # 상한 주파수
        }
        hop_info_list.append(hop_info)

    return frame, hop_info_list


def _estimate_bandwidth(signal: np.ndarray, fs: float) -> float:
    """
    신호의 99% 파워 대역폭을 FFT 기반으로 추정.
    """
    N = len(signal)
    if N == 0:
        return fs * 0.1  # fallback

    spectrum = np.fft.fftshift(np.abs(np.fft.fft(signal, n=max(N, 256))) ** 2)
    total_power = np.sum(spectrum)
    if total_power < 1e-20:
        return fs * 0.1

    # 누적 파워로 99% 경계 탐색
    cumsum = np.cumsum(spectrum)
    cumsum /= total_power

    lower_idx = np.searchsorted(cumsum, 0.005)
    upper_idx = np.searchsorted(cumsum, 0.995)

    fft_len = len(spectrum)
    freq_resolution = fs / fft_len
    bw = (upper_idx - lower_idx) * freq_resolution

    # 최소 대역폭 보장
    return max(bw, fs * 0.02)


# ============================================================================
#  [4] STFT 및 스펙트로그램 생성 (Variable NFFT)
# ============================================================================

def compute_stft(
    signal: np.ndarray,
    fs: float,
    nfft: int,
    overlap: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    STFT를 계산하여 스펙트로그램(파워 dB)을 반환.

    Parameters
    ----------
    signal : np.ndarray
        입력 IQ 신호 (complex).
    fs : float
        샘플링 레이트 (Hz).
    nfft : int
        FFT 윈도우 크기.
    overlap : int
        오버랩 샘플 수.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - freqs : 주파수 축 배열 (Hz)
        - times : 시간 축 배열 (초)
        - Sxx_db : 파워 스펙트로그램 (dB 스케일), shape=(n_freq, n_time)
    """
    hop_length = nfft - overlap

    freqs, times, Zxx = scipy_stft(
        signal,
        fs=fs,
        window="hann",
        nperseg=nfft,
        noverlap=overlap,
        nfft=nfft,
        return_onesided=False,  # IQ 신호이므로 양면 스펙트럼
    )

    # fftshift: 음의 주파수를 왼쪽으로 배치
    freqs = np.fft.fftshift(freqs)
    Zxx = np.fft.fftshift(Zxx, axes=0)

    # 파워 스펙트로그램 (dB)
    Sxx = np.abs(Zxx) ** 2
    Sxx_db = 10.0 * np.log10(Sxx + 1e-12)

    return freqs, times, Sxx_db


def spectrogram_to_image(Sxx_db: np.ndarray) -> np.ndarray:
    """
    dB 스펙트로그램을 0~255 범위의 그레이스케일 이미지로 변환.
    주파수 축 반전: 높은 주파수가 이미지 상단에 오도록 처리.

    Parameters
    ----------
    Sxx_db : np.ndarray
        파워 스펙트로그램 (dB), shape=(n_freq, n_time).

    Returns
    -------
    np.ndarray
        그레이스케일 이미지, shape=(YOLO_IMG_SIZE, YOLO_IMG_SIZE), dtype=uint8.
    """
    # ── 주파수 축 반전 (Y축 반전) ──
    # 스펙트로그램: 아래→위 = 낮은→높은 주파수
    # 이미지: 위→아래 = 행 인덱스 증가
    # 따라서 주파수 축(행)을 뒤집어야 높은 주파수가 이미지 상단에 위치
    Sxx_flipped = Sxx_db[::-1, :]

    # 동적 범위 정규화 (min-max → 0~255)
    vmin = np.percentile(Sxx_flipped, 2)   # 하위 2% 클리핑
    vmax = np.percentile(Sxx_flipped, 99.5) # 상위 0.5% 클리핑
    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0

    img = (Sxx_flipped - vmin) / (vmax - vmin)
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)

    # 640×640 리사이징
    img_resized = cv2.resize(img, (YOLO_IMG_SIZE, YOLO_IMG_SIZE),
                             interpolation=cv2.INTER_LINEAR)

    return img_resized


# ============================================================================
#  [5] 좌표 매핑 및 YOLO 라벨링 (Auto-Labeling)
# ============================================================================

def compute_bbox_yolo(
    hop_info: Dict,
    freqs: np.ndarray,
    times: np.ndarray,
    spec_shape: Tuple[int, int],
) -> Optional[Tuple[int, float, float, float, float]]:
    """
    물리적 신호 정보 → 스펙트로그램 원본 픽셀 좌표 → 640×640 정규화 좌표 변환.

    ★ 핵심 로직 ★
    1) 물리 좌표 (t_start, t_end, f_min, f_max) 확보
    2) freqs/times 배열에서 원본 스펙트로그램 픽셀 좌표 계산
    3) Y축 반전 고려 (spectrogram_to_image에서 [::-1] 처리됨)
    4) 640×640 리사이징 비율 적용
    5) YOLO 정규화 포맷 출력

    Parameters
    ----------
    hop_info : Dict
        홉 물리 정보 딕셔너리.
    freqs : np.ndarray
        STFT 주파수 축 (Hz), fftshifted, 오름차순.
    times : np.ndarray
        STFT 시간 축 (초).
    spec_shape : Tuple[int, int]
        원본 스펙트로그램 shape (n_freq, n_time).

    Returns
    -------
    Optional[Tuple[int, float, float, float, float]]
        (class_id, x_center_norm, y_center_norm, w_norm, h_norm)
        또는 유효하지 않으면 None.
    """
    n_freq, n_time = spec_shape

    # ── (1) 물리 좌표 ──
    t_start = hop_info["t_start_sec"]
    t_end = hop_info["t_end_sec"]
    f_min = hop_info["f_min_hz"]
    f_max = hop_info["f_max_hz"]

    # ── (2) 스펙트로그램 시간 축 → 픽셀 x 좌표 ──
    # times 배열에서 가장 가까운 인덱스 찾기
    if len(times) < 2:
        return None

    t_resolution = times[1] - times[0]
    t_min_spec = times[0]
    t_max_spec = times[-1]

    # 시간 좌표를 픽셀 인덱스로 변환
    x1_frac = (t_start - t_min_spec) / (t_max_spec - t_min_spec + 1e-12) * (n_time - 1)
    x2_frac = (t_end - t_min_spec) / (t_max_spec - t_min_spec + 1e-12) * (n_time - 1)

    # ── (3) 스펙트로그램 주파수 축 → 픽셀 y 좌표 ──
    # freqs는 fftshift 후 오름차순 (낮은→높은 주파수)
    f_min_spec = freqs[0]
    f_max_spec = freqs[-1]

    # 원본 스펙트로그램에서의 주파수 인덱스 (아래=낮은 주파수, 위=높은 주파수)
    y1_frac_orig = (f_min - f_min_spec) / (f_max_spec - f_min_spec + 1e-12) * (n_freq - 1)
    y2_frac_orig = (f_max - f_min_spec) / (f_max_spec - f_min_spec + 1e-12) * (n_freq - 1)

    # ── (4) Y축 반전 처리 ──
    # spectrogram_to_image()에서 Sxx[::-1, :] 처리했으므로
    # 이미지 y좌표 = (n_freq - 1) - 원본 y좌표
    y1_img = (n_freq - 1) - y2_frac_orig  # f_max → 이미지 상단 (작은 y)
    y2_img = (n_freq - 1) - y1_frac_orig  # f_min → 이미지 하단 (큰 y)

    # ── (5) 640×640 리사이징 비율 적용 ──
    scale_x = YOLO_IMG_SIZE / n_time
    scale_y = YOLO_IMG_SIZE / n_freq

    x1_final = x1_frac * scale_x
    x2_final = x2_frac * scale_x
    y1_final = y1_img * scale_y
    y2_final = y2_img * scale_y

    # ── (6) YOLO 정규화 포맷 ──
    # 바운딩 박스를 이미지 크기로 정규화 (0~1)
    x_center = (x1_final + x2_final) / 2.0 / YOLO_IMG_SIZE
    y_center = (y1_final + y2_final) / 2.0 / YOLO_IMG_SIZE
    w = abs(x2_final - x1_final) / YOLO_IMG_SIZE
    h = abs(y2_final - y1_final) / YOLO_IMG_SIZE

    # 클리핑: 0~1 범위 강제
    x_center = np.clip(x_center, 0.0, 1.0)
    y_center = np.clip(y_center, 0.0, 1.0)
    w = np.clip(w, 0.001, 1.0)
    h = np.clip(h, 0.001, 1.0)

    # 바운딩 박스가 이미지 밖으로 벗어나지 않도록 보정
    x_min_norm = max(x_center - w / 2, 0.0)
    x_max_norm = min(x_center + w / 2, 1.0)
    y_min_norm = max(y_center - h / 2, 0.0)
    y_max_norm = min(y_center + h / 2, 1.0)

    x_center = (x_min_norm + x_max_norm) / 2.0
    y_center = (y_min_norm + y_max_norm) / 2.0
    w = x_max_norm - x_min_norm
    h = y_max_norm - y_min_norm

    # 너무 작은 박스 제거 (최소 크기 임계값)
    if w < 0.005 or h < 0.005:
        return None

    return (HOPPING_CLASS_ID, float(x_center), float(y_center), float(w), float(h))


# ============================================================================
#  [6] 데이터셋 생성 메인 파이프라인
# ============================================================================

def generate_one_sample(
    seeds: List[np.ndarray],
    fs: float,
    frame_duration_sec: float,
) -> Tuple[np.ndarray, List[Tuple], Dict]:
    """
    단일 학습 샘플(이미지 + 라벨) 생성 파이프라인.

    Parameters
    ----------
    seeds : List[np.ndarray]
        Seed 신호 리스트.
    fs : float
        샘플링 레이트 (Hz).
    frame_duration_sec : float
        프레임 지속 시간 (초).

    Returns
    -------
    Tuple[np.ndarray, List[Tuple], Dict]
        - 640×640 그레이스케일 이미지 (uint8)
        - YOLO 바운딩 박스 리스트 [(class_id, xc, yc, w, h), ...]
        - 메타데이터 딕셔너리
    """
    frame_len = int(fs * frame_duration_sec)

    # ── (a) 가변 NFFT를 먼저 선택 (홉 길이 결정에 필요) ──
    nfft = random.choice(NFFT_CANDIDATES)

    # Overlap: NFFT의 50%~75% 사이 랜덤
    overlap_ratio = random.uniform(OVERLAP_RATIO_MIN, OVERLAP_RATIO_MAX)
    overlap = int(nfft * overlap_ratio)

    # NFFT가 프레임 길이보다 크면 조정
    if nfft > frame_len:
        nfft = min(NFFT_CANDIDATES[0], frame_len)
        overlap = int(nfft * overlap_ratio)

    # ── (b) 홉 개수 랜덤 결정 ──
    n_hops = random.randint(MIN_HOPS_PER_IMAGE, MAX_HOPS_PER_IMAGE)

    # ── (c) 호핑 신호 배치 ──
    # ★ NFFT를 전달하여 홉 길이가 STFT 해상도에 맞도록 보장
    frame, hop_infos = place_hops_in_frame(seeds, frame_len, fs, n_hops, nfft)

    # ── (d) 전체 프레임에 AWGN 적용 ──
    snr_db = random.uniform(SNR_MIN_DB, SNR_MAX_DB)
    frame = apply_awgn(frame, snr_db)

    # ── (e) STFT 계산 ──
    freqs, times, Sxx_db = compute_stft(frame, fs, nfft, overlap)
    spec_shape = Sxx_db.shape  # (n_freq, n_time)

    # ── (f) 스펙트로그램 → 이미지 변환 ──
    img = spectrogram_to_image(Sxx_db)

    # ── (g) 바운딩 박스 라벨링 ──
    labels: List[Tuple] = []
    for hop_info in hop_infos:
        bbox = compute_bbox_yolo(hop_info, freqs, times, spec_shape)
        if bbox is not None:
            labels.append(bbox)

    # ── (h) 메타데이터 ──
    metadata = {
        "n_hops": n_hops,
        "snr_db": snr_db,
        "nfft": nfft,
        "overlap": overlap,
        "spec_shape": spec_shape,
        "n_valid_labels": len(labels),
    }

    return img, labels, metadata


def generate_dataset(
    seed_dir: str,
    output_dir: str,
    n_samples: int,
    fs: float,
    frame_duration_sec: float,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> None:
    """
    전체 데이터셋 생성 함수.
    YOLO 디렉토리 구조에 맞춰 이미지와 라벨 저장.

    Parameters
    ----------
    seed_dir : str
        Seed IQ 데이터 디렉토리.
    output_dir : str
        출력 데이터셋 루트 디렉토리.
    n_samples : int
        생성할 총 샘플 수.
    fs : float
        샘플링 레이트 (Hz).
    frame_duration_sec : float
        프레임 지속 시간 (초).
    train_ratio : float
        훈련 세트 비율.
    val_ratio : float
        검증 세트 비율.
    test_ratio : float
        테스트 세트 비율.
    """
    # ── 디렉토리 구조 생성 ──
    # YOLOv8 표준 디렉토리 구조:
    #   dataset/
    #     images/
    #       train/  val/  test/
    #     labels/
    #       train/  val/  test/
    splits = ["train", "val", "test"]
    for split in splits:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    # ── Seed 신호 로드 ──
    seeds = load_seed_signals(seed_dir)
    if not seeds:
        print("[오류] Seed 신호를 로드할 수 없습니다. 종료합니다.")
        return

    # ── 분할 인덱스 계산 ──
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    # 나머지는 test로
    n_test = n_samples - n_train - n_val

    split_assignments = (
        ["train"] * n_train +
        ["val"] * n_val +
        ["test"] * n_test
    )
    random.shuffle(split_assignments)

    # ── 통계 추적 ──
    stats = {
        "total": 0,
        "train": 0, "val": 0, "test": 0,
        "total_hops": 0,
        "nfft_dist": {k: 0 for k in NFFT_CANDIDATES},
    }

    print(f"\n{'='*60}")
    print(f"  호핑 신호 데이터셋 생성 시작")
    print(f"  총 샘플: {n_samples}  |  Fs: {fs/1e6:.1f} MHz")
    print(f"  프레임: {frame_duration_sec*1e3:.1f} ms")
    print(f"  분할: train={n_train}, val={n_val}, test={n_test}")
    print(f"{'='*60}\n")

    for i in range(n_samples):
        split = split_assignments[i]

        # ── 샘플 생성 ──
        img, labels, meta = generate_one_sample(seeds, fs, frame_duration_sec)

        # ── 파일명 ──
        sample_id = f"hop_{i:06d}"
        img_path = os.path.join(output_dir, "images", split, f"{sample_id}.png")
        lbl_path = os.path.join(output_dir, "labels", split, f"{sample_id}.txt")

        # ── 이미지 저장 ──
        cv2.imwrite(img_path, img)

        # ── 라벨 저장 (YOLO 포맷) ──
        with open(lbl_path, "w") as f:
            for bbox in labels:
                class_id, xc, yc, w, h = bbox
                f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        # ── 통계 업데이트 ──
        stats["total"] += 1
        stats[split] += 1
        stats["total_hops"] += meta["n_valid_labels"]
        nfft_used = meta["nfft"]
        if nfft_used in stats["nfft_dist"]:
            stats["nfft_dist"][nfft_used] += 1

        # ── 진행 상황 출력 ──
        if (i + 1) % max(1, n_samples // 20) == 0 or i == n_samples - 1:
            pct = (i + 1) / n_samples * 100
            print(f"  [{pct:5.1f}%] {i+1}/{n_samples}  "
                  f"split={split:5s}  nfft={meta['nfft']:4d}  "
                  f"hops={meta['n_valid_labels']}  "
                  f"snr={meta['snr_db']:+.1f}dB  "
                  f"spec={meta['spec_shape']}")

    # ── YOLO data.yaml 생성 ──
    yaml_path = os.path.join(output_dir, "data.yaml")
    abs_output = os.path.abspath(output_dir)
    with open(yaml_path, "w") as f:
        f.write(f"# YOLOv8 호핑 신호 데이터셋 설정\n")
        f.write(f"path: {abs_output}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"test: images/test\n\n")
        f.write(f"nc: 1\n")
        f.write(f"names: ['hopping_signal']\n")

    # ── 최종 통계 출력 ──
    print(f"\n{'='*60}")
    print(f"  데이터셋 생성 완료!")
    print(f"{'='*60}")
    print(f"  총 이미지   : {stats['total']}")
    print(f"  Train       : {stats['train']}")
    print(f"  Val         : {stats['val']}")
    print(f"  Test        : {stats['test']}")
    print(f"  총 홉 라벨  : {stats['total_hops']}")
    print(f"  평균 홉/이미지: {stats['total_hops']/max(stats['total'],1):.1f}")
    print(f"\n  NFFT 분포:")
    for nfft_val, cnt in sorted(stats["nfft_dist"].items()):
        bar = "█" * int(cnt / max(stats["total"], 1) * 40)
        print(f"    {nfft_val:4d}: {cnt:4d} ({cnt/max(stats['total'],1)*100:.1f}%) {bar}")
    print(f"\n  출력 경로   : {abs_output}")
    print(f"  YOLO config : {yaml_path}")
    print(f"{'='*60}\n")


# ============================================================================
#  [7] 시각화 유틸리티 (디버깅/검증용)
# ============================================================================

def visualize_sample(
    img: np.ndarray,
    labels: List[Tuple],
    save_path: Optional[str] = None,
) -> None:
    """
    생성된 샘플을 바운딩 박스와 함께 시각화.
    디버깅 및 라벨 검증 용도.

    Parameters
    ----------
    img : np.ndarray
        640×640 그레이스케일 이미지.
    labels : List[Tuple]
        YOLO 바운딩 박스 리스트.
    save_path : Optional[str]
        저장 경로. None이면 화면에 표시.
    """
    # 컬러맵 적용 (Viridis)
    img_color = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)

    for bbox in labels:
        class_id, xc, yc, w, h = bbox

        # YOLO 정규화 좌표 → 픽셀 좌표 변환
        x1 = int((xc - w / 2) * YOLO_IMG_SIZE)
        y1 = int((yc - h / 2) * YOLO_IMG_SIZE)
        x2 = int((xc + w / 2) * YOLO_IMG_SIZE)
        y2 = int((yc + h / 2) * YOLO_IMG_SIZE)

        # 바운딩 박스 그리기 (녹색)
        cv2.rectangle(img_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_color, f"hop",
                    (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if save_path:
        cv2.imwrite(save_path, img_color)
        print(f"  [시각화 저장] {save_path}")
    else:
        cv2.imshow("Sample", img_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def visualize_random_samples(
    output_dir: str,
    n_vis: int = 16,
) -> None:
    """
    생성된 데이터셋에서 랜덤 샘플을 선택하여 그리드 시각화.

    Parameters
    ----------
    output_dir : str
        데이터셋 루트 경로.
    n_vis : int
        시각화할 샘플 수.
    """
    vis_dir = os.path.join(output_dir, "visualization")
    os.makedirs(vis_dir, exist_ok=True)

    # 모든 이미지 파일 수집
    all_images = []
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(output_dir, "images", split)
        if os.path.exists(img_dir):
            all_images.extend(glob.glob(os.path.join(img_dir, "*.png")))

    if not all_images:
        print("[경고] 시각화할 이미지가 없습니다.")
        return

    # 랜덤 선택
    selected = random.sample(all_images, min(n_vis, len(all_images)))

    for img_path in selected:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # 대응하는 라벨 파일 찾기
        base = os.path.splitext(os.path.basename(img_path))[0]
        split = img_path.split(os.sep)[-2]
        lbl_path = os.path.join(output_dir, "labels", split, f"{base}.txt")

        labels = []
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        labels.append((
                            int(parts[0]),
                            float(parts[1]),
                            float(parts[2]),
                            float(parts[3]),
                            float(parts[4]),
                        ))

        vis_path = os.path.join(vis_dir, f"vis_{base}.png")
        visualize_sample(img, labels, vis_path)

    print(f"[시각화 완료] {len(selected)}개 샘플 → {vis_dir}")


# ============================================================================
#  [8] 메인 엔트리 포인트
# ============================================================================

def parse_args() -> argparse.Namespace:
    """커맨드라인 인자 파싱."""
    parser = argparse.ArgumentParser(
        description="YOLOv8용 호핑 신호 스펙트로그램 데이터셋 생성기",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--seed_dir", type=str, default="./seed_data",
        help="Seed IQ 데이터(.npy) 디렉토리 경로 (기본: ./seed_data)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./dataset_hopping",
        help="출력 데이터셋 경로 (기본: ./dataset_hopping)",
    )
    parser.add_argument(
        "--n_samples", type=int, default=5000,
        help="생성할 총 이미지 수 (기본: 5000)",
    )
    parser.add_argument(
        "--fs", type=float, default=10e6,
        help="샘플링 레이트 Hz (기본: 10 MHz)",
    )
    parser.add_argument(
        "--frame_duration", type=float, default=0.05,
        help="프레임 지속 시간 초 (기본: 0.05 = 50ms, 많은 홉 수용)",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8,
        help="훈련 세트 비율 (기본: 0.8)",
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1,
        help="검증 세트 비율 (기본: 0.1)",
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.1,
        help="테스트 세트 비율 (기본: 0.1)",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="생성 후 랜덤 샘플 시각화 수행",
    )
    parser.add_argument(
        "--n_vis", type=int, default=16,
        help="시각화할 샘플 수 (기본: 16)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="난수 시드 (재현성용, 기본: None)",
    )

    return parser.parse_args()


def main():
    """메인 실행 함수."""
    args = parse_args()

    # 난수 시드 설정 (재현성)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"[설정] 난수 시드: {args.seed}")

    # 비율 검증
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 0.01:
        print(f"[경고] 분할 비율 합이 1.0이 아닙니다: {ratio_sum:.2f}")
        print(f"       자동 정규화합니다.")
        args.train_ratio /= ratio_sum
        args.val_ratio /= ratio_sum
        args.test_ratio /= ratio_sum

    # 데이터셋 생성
    generate_dataset(
        seed_dir=args.seed_dir,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        fs=args.fs,
        frame_duration_sec=args.frame_duration,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    # 시각화 (선택)
    if args.visualize:
        visualize_random_samples(args.output_dir, args.n_vis)

    print("[완료] 모든 작업이 종료되었습니다.")


if __name__ == "__main__":
    main()
