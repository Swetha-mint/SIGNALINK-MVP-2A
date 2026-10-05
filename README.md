# SIGNALINK MVP-2A — Landmark Tokenizer

Camera → MediaPipe 21 landmarks → normalization → 7 representative points → 7-bit quantization → 3×7 matrix → reversible Toffoli-style bit transform.

The baseline is 21 × 3 × 32 = 2016 bits (float32 estimate). The compact representation is 21 × 7 = 147 bits, a 92.7% fixed-width reduction.

This is **not lossless compression**. The website reports quantization RMSE so we can measure information loss. The reversible transform changes the bit pattern but does not reduce its length.

Run through GitHub Pages or localhost/HTTPS because camera access is required.
