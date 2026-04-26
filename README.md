# Audio Fingerprinting on FMA-small

Course project — applied time-series methods.

A from-scratch implementation of [Avery Wang's 2003 landmark-hashing
algorithm](https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf)
(the algorithm behind Shazam), evaluated on the
[FMA-small](https://github.com/mdeff/fma) dataset.

The goal of the project is to manipulate raw audio as a time series, and to go
through a full pipeline that turns a time-domain signal into a compact,
noise-robust representation usable for retrieval.

## Pipeline

```
audio (1-D time series, 22 050 Hz)
    │
    ▼  Short-Time Fourier Transform
spectrogram (time × frequency)
    │
    ▼  2-D local maxima
constellation map (sparse points)
    │
    ▼  combinatorial pairing of peaks
landmark hashes (32-bit integers)
    │
    ▼  inverted index
catalog
    │
    ▼  cross-correlation via offset voting
identified track + alignment
```

Each step is implemented in [`fingerprint.py`](fingerprint.py) and run on the
dataset in [`shazam.ipynb`](shazam.ipynb).

## Results

Top-1 accuracy on a 300-track catalog (30 queries per condition):

| Query                     | Accuracy |
|---------------------------|---------:|
| 10 s, clean               |     100% |
| 10 s, SNR 10 dB           |     100% |
| 10 s, SNR 0 dB            |     100% |
|  5 s, clean               |     100% |
|  3 s, clean               |     100% |
|  3 s, SNR 5 dB            |     100% |

Out-of-catalog rejection: median top-1 vote count is ~400 for true matches and
~3 for unknown queries, so a simple threshold separates the two.

## Repository layout

```
.
├── fingerprint.py        # core: STFT, peak picking, hashing, indexing, matching
├── shazam.ipynb          # pipeline run + visualisations on FMA-small
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Get the dataset

The FMA-small dataset is **not** included in this repository (it is ~7.2 GB).
Download it from the [official FMA repo](https://github.com/mdeff/fma):

```bash
curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip
unzip fma_small.zip
```

The notebook expects the extracted folder at `./fma_small/`.

## Run

```bash
jupyter notebook shazam.ipynb
```

The catalog-build cell takes ~25 s for 300 tracks; the rest is near-instant.

## Dataset citation

If you use the FMA dataset, please cite:

> Defferrard, M., Benzi, K., Vandergheynst, P., Bresson, X. (2017).
> *FMA: A Dataset for Music Analysis.* ISMIR.
> [arXiv:1612.01840](https://arxiv.org/abs/1612.01840) ·
> [github.com/mdeff/fma](https://github.com/mdeff/fma)

```bibtex
@inproceedings{fma_dataset,
  title     = {{FMA}: A Dataset for Music Analysis},
  author    = {Defferrard, Micha\"el and Benzi, Kirell
               and Vandergheynst, Pierre and Bresson, Xavier},
  booktitle = {18th International Society for Music Information
               Retrieval Conference (ISMIR)},
  year      = {2017},
  archiveprefix = {arXiv},
  eprint    = {1612.01840},
}
```

## Algorithm reference

> Wang, A. (2003). *An Industrial-Strength Audio Search Algorithm.*
> ISMIR. [PDF](https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf)

## License

MIT — see [`LICENSE`](LICENSE).
