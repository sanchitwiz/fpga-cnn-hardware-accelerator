
import math
import torch


def quantize_multiplier(real_multiplier):
    """
    Converts a positive real multiplier into
    Q31 integer multiplier and right shift.

    real_multiplier = multiplier / (2^shift)
    """

    if real_multiplier < 0:
        raise ValueError("Multiplier must be non-negative.")

    if real_multiplier == 0:
        return 0, 0

    significand, exponent = math.frexp(real_multiplier)

    multiplier = int(round(significand * (1 << 31)))

    if multiplier == (1 << 31):
        multiplier //= 2
        exponent += 1

    shift = 31 - exponent

    return multiplier, shift


def fixed_point_requantize(accumulator, real_multiplier):
    """
    Performs fixed-point requantization using
    Q31 multiplication and power-of-two shifting.

    INT64 is used for safe intermediate multiplication.

    Rounding: nearest, ties away from zero.
    """

    multiplier, shift = quantize_multiplier(real_multiplier)

    accumulator = accumulator.to(torch.int64)

    if multiplier == 0:
        return torch.zeros_like(accumulator)

    product = accumulator * multiplier

    if shift > 0:

        offset = 1 << (shift - 1)

        result = torch.where(
            product >= 0,
            (product + offset) >> shift,
            -((-product + offset) >> shift)
        )

    elif shift < 0:

        result = product << (-shift)

    else:

        result = product

    return result


def print_multiplier_details(real_multiplier):

    multiplier, shift = quantize_multiplier(real_multiplier)

    print(f"Real multiplier    : {real_multiplier:.12e}")
    print(f"Integer multiplier : {multiplier}")
    print(f"Right shift        : {shift}")


if __name__ == "__main__":

    test_multiplier = 0.125

    print("Testing fixed-point multiplier...")

    print_multiplier_details(test_multiplier)

    test_values = torch.tensor(
        [-1000, -100, -10, 0, 10, 100, 1000],
        dtype=torch.int64
    )

    output = fixed_point_requantize(
        test_values,
        test_multiplier
    )

    print("\nInput :", test_values.tolist())
    print("Output:", output.tolist())
