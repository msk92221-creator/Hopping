#!/usr/bin/env python3
"""
create_sample_seeds.py
======================
테스트용 샘플 IQ Seed 데이터(.npy)를 생성하는 유틸리티.
실측 IQ 데이터가 없을 때 generate_ultimate_dataset.py 를 검증하기 위해 사용.

사용법:
    python create_sample_seeds.py --output_dir ./seed_data --n_files 10
"""

import os
import argparse
import numpy as np


def generate_chirp(n_samples: int, fs: float) -> np.ndarray:
    """선형 Chirp 신호 생성."""
    t = np.arange(n_samples) / fs
    f0 = np.random.uniform(0.5e6, 1.5e6)
    f1 = np.random.uniform(2.0e6, 4.0e6)
    phase = 2.0 * np.pi * (f0 * t + (f1 - f0) / (2.0 * t[-1]) * t ** 2)
    return np.exp(1j * phase).astype(np.complex64)


def generate_fsk_burst(n_samples: int, fs: float) -> np.ndarray:
    """FSK Burst 신호 생성."""
    symbol_rate = np.random.uniform(50e3, 200e3)
    samples_per_symbol = max(int(fs / symbol_rate), 4)
    n_symbols = n_samples // samples_per_symbol

    freqs = np.random.uniform(-1e6, 1e6, size=n_symbols)
    sig = np.zeros(n_samples, dtype=np.complex128)

    for i in range(n_symbols):
        start = i * samples_per_symbol
        end = min(start + samples_per_symbol, n_samples)
        t = np.arange(end - start) / fs
        sig[start:end] = np.exp(1j * 2.0 * np.pi * freqs[i] * t)

    return sig.astype(np.complex64)


def generate_ofdm_like(n_samples: int, fs: float) -> np.ndarray:
    """OFDM 유사 신호 생성."""
    n_carriers = 64
    fft_size = 128
    n_symbols = n_samples // fft_size

    sig = np.zeros(n_samples, dtype=np.complex128)
    for s in range(n_symbols):
        freq_domain = np.zeros(fft_size, dtype=np.complex128)
        active = np.random.choice(fft_size, n_carriers, replace=False)
        freq_domain[active] = np.exp(1j * np.random.uniform(0, 2 * np.pi, n_carriers))
        time_domain = np.fft.ifft(freq_domain)
        start = s * fft_size
        end = min(start + fft_size, n_samples)
        sig[start:end] = time_domain[: end - start]

    return sig.astype(np.complex64)


def main():
    parser = argparse.ArgumentParser(description="테스트용 Seed IQ 데이터 생성")
    parser.add_argument("--output_dir", type=str, default="./seed_data",
                        help="출력 디렉토리 (기본: ./seed_data)")
    parser.add_argument("--n_files", type=int, default=10,
                        help="생성할 .npy 파일 수 (기본: 10)")
    parser.add_argument("--n_samples", type=int, default=100000,
                        help="파일당 IQ 샘플 수 (기본: 100000)")
    parser.add_argument("--fs", type=float, default=10e6,
                        help="샘플링 레이트 Hz (기본: 10 MHz)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    generators = [generate_chirp, generate_fsk_burst, generate_ofdm_like]

    for i in range(args.n_files):
        gen_func = generators[i % len(generators)]
        # 노이즈 배경에 신호 삽입
        noise_floor = 0.01 * (np.random.randn(args.n_samples)
                              + 1j * np.random.randn(args.n_samples)).astype(np.complex64)

        # 랜덤 위치에 신호 버스트 삽입
        sig_len = np.random.randint(args.n_samples // 4, args.n_samples // 2)
        sig_start = np.random.randint(0, args.n_samples - sig_len)
        burst = gen_func(sig_len, args.fs)
        noise_floor[sig_start: sig_start + sig_len] += burst

        fpath = os.path.join(args.output_dir, f"seed_iq_{i:03d}.npy")
        np.save(fpath, noise_floor)
        print(f"  [{i+1}/{args.n_files}] 저장: {fpath}  "
              f"(타입: {gen_func.__name__}, 신호길이: {sig_len})")

    print(f"\n[완료] {args.n_files}개 Seed 파일 → {args.output_dir}")


if __name__ == "__main__":
    main()
