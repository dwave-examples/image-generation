# Quantum FFHQ Image Generation

An **inference-only** Dash demo that generates four 256 x 256 faces using fresh
D-Wave QPU samples and the current pretrained FFHQ decoder from
[`dwavesystems/ml-etc`](https://github.com/dwavesystems/ml-etc).
There are no training, tuning, dataset download, or model-upload paths.

## Installation and usage

Use Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:8050/. The existing development container also installs
`requirements.txt`. Your environment must already have normal Ocean/Leap access
configured. The demo uses that configuration through the standard D-Wave clients;
it does not read or display credentials.

Choose any QPU offered in the dropdown and click **Generate four faces**.
Only accessible QPUs with at least one valid precomputed embedding of the trained
logical graph appear. **Refresh QPUs** rechecks availability and working graphs.
Unavailable systems and banks with no surviving embeddings are reported explicitly.
Embeddings are checked again immediately before every submission; an invalidated
embedding is skipped in favor of another valid saved embedding. No embedding
search runs during startup or generation.

The bundled banks target `Advantage2_system1` and `Advantage2_system5`. Access and
working graphs can change; the demo does not assume either is available. The
valid embedding with the shortest maximum chain, then fewest physical qubits,
then lowest ID is used, following the `ml-etc` bank-loading policy.

The decoder and Ising parameters are independent of the physical QPU. Each
submission uses the saved logical Hamiltonian and 0.1-microsecond fast annealing,
with the same Ocean defaults as the source inference workflow. Sampling errors
are surfaced; cached bitstrings, classical samplers, and random replacements are
never used.

## Generation and selection

Each request:

1. Obtains **32 fresh QPU samples** in one submission.
2. Decodes candidates in bounded-size batches and scores each image exactly once
   with OpenCV's **eDifFIQA(T)** face-quality model.
3. Retains only the **ten highest scores**, with stable ordering for ties.
4. Extracts normalized **AlexNet LPIPS features once per retained image** and
   computes their calibrated pairwise perceptual distances.
5. Examines all 210 four-image subsets and maximizes the minimum pairwise
   distance. Ties are broken by total pairwise distance, then quality rank.
   The four chosen images are shown in quality order in a 2 x 2 grid.

This exact max-min selection improves on the research script's greedy selection.
The images are not re-decoded for selection or display. Quality scores and the
selected QPU/embedding are visible in the interface.

## Model and performance

`models/ffhq/` contains the **EMA weights at step 540,000** from
`ffhq256_z5t3_multi20m_fast100ns_normadd_norecongan_fresh`. It includes only the
decoder, its fitted global-mode state, the logical Ising parameters, and portable
embedding banks. Tensor-only shards are each below GitHub's 100 MB file limit;
no separate `ml-etc` checkout, full training checkpoint, or Git LFS is needed.
The manifest records checkpoint provenance, the exact logical node/edge order,
inference settings, and SHA-256 checksums verified when loading.

The port preserves the source's floor-based 660-to-4096 latent and Hamiltonian
resizing, spin-valued inputs, `sH` local transform, normalized-add global-mode
fusion, and learned binary-code embedding. It does not import the trainer,
encoder, GAN discriminator, optimizers, or training-only dependencies.

CPU inference is the default. The 32-candidate default keeps CPU latency bounded;
increase `CANDIDATE_COUNT` to 128 for the research script's larger candidate pool.
`demo_configs.py` controls the candidate count,
decoder batch size, device, and a CPU thread limit (four by default). Eager
PyTorch is intentional: previous measurements of this decoder did not justify
`torch.compile` startup cost for this demo. Decoder batches default to one to
bound CPU memory, and convolutions use PyTorch's channels-last memory layout.
Models are loaded once per server process, and inference is
serialized to avoid oversubscribing CPU threads and sharing mutable OpenCV
inference state across requests. Images are returned directly to the requesting
browser, not written to shared filenames.

The first initialization downloads and caches the public eDifFIQA ONNX model
from `opencv/face_image_quality_assessment_ediffiqa` on Hugging Face and the
standard pretrained AlexNet weights through torchvision. Subsequent requests
reuse both models. Internet access is needed for these first downloads.

To re-export a newer **compatible, trusted** checkpoint:

```bash
python -m src.export_model \
  --checkpoint /path/to/ml-etc/artifacts/checkpoints/RUN/weights_step_00540000.pt \
  --embedding-banks /path/to/system1/bank /path/to/system5/bank \
  --output /path/to/new-model-directory
```

The exporter refuses incompatible architectures or an existing output directory
and never copies the training configuration, sample banks, or optimizer state.
Set `MODEL_DIRECTORY` and `EMBEDDING_DIRECTORY` in `demo_configs.py` to use the
new export.

## Tests

```bash
python -m unittest discover
```

Unit tests do not connect to Leap or submit QPU jobs. They cover embedding
validation and invalid-bank fallback, logical sample ordering, streaming quality
ranking, exact diversity selection, feature reuse, decoder compatibility, and
the inference-only interface.

## License

Released under the Apache License 2.0. See [LICENSE](LICENSE).
The decoder and sampling conventions are adapted from `ml-etc/qvae/model.py`,
`qvae/qpu_prior.py`, and its FFHQ evaluation scripts. Public quality/perceptual
models retain their respective upstream licenses.
