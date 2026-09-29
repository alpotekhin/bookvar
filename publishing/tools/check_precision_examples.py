"""Reproduce the scalar arithmetic examples in the precision chapter.

Uses IEEE binary16/binary32 packing and explicit BF16 round-to-nearest-even.
This checks representational arithmetic, not GPU kernels or a training run.
"""
import struct


def fp16(x):
    return struct.unpack(">e", struct.pack(">e", x))[0]


def fp32(x):
    return struct.unpack(">f", struct.pack(">f", x))[0]


def bf16(x):
    """Round finite FP32 input to BF16, ties to even."""
    bits = struct.unpack(">I", struct.pack(">f", x))[0]
    assert bits & 0x7F800000 != 0x7F800000, "finite inputs only"
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack(">f", struct.pack(">I", rounded))[0]


def sequential_sum(cast, count):
    total = cast(0.0)
    for _ in range(count):
        total = cast(total + 1.0)
    return total


def main():
    exact = 1 + 2**-10
    assert fp16(exact) == exact
    assert bf16(exact) == 1
    assert struct.unpack(">H", struct.pack(">e", 6.25))[0] == int(
        "0" + "10001" + "1001000000", 2
    )

    gradient, scale = 2**-26, 2**16
    assert fp16(gradient) == 0
    assert fp32(fp16(gradient * scale) / scale) == gradient

    sums = {name: sequential_sum(cast, 4096)
            for name, cast in [("BF16", bf16), ("FP16", fp16), ("FP32", fp32)]}
    assert sums == {"BF16": 256.0, "FP16": 2048.0, "FP32": 4096.0}

    weight16, master = fp16(1), fp32(1)
    for _ in range(10):
        weight16 = fp16(weight16 - 1e-4)
        master = fp32(master - fp32(1e-4))
    assert weight16 == 1
    assert master == 0.998999834060669
    assert fp16(master) == 0.9990234375

    # E4M3 normal numbers in [1,2) have spacing 1/8.
    e4m3_near_one = [1 + k / 8 for k in range(8)]
    assert min(e4m3_near_one, key=lambda x: abs(x - 1.2)) == 1.25
    assert 896 / 2 == 448
    assert 1 / 32 * 100 == 3.125

    print("PASS: FP16 encoding of 6.25; FP16/BF16 rounding near 1")
    print("PASS: underflow =", fp16(gradient), "; scaled recovery =", gradient)
    print("PASS: sequential sums =", sums)
    print("PASS: FP16 weight =", weight16, "; FP32 master =", master,
          "; working copy =", fp16(master))
    print("PASS: E4M3 rounding, FP8 scale, MXFP8 metadata overhead")


if __name__ == "__main__":
    main()
