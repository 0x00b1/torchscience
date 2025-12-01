import hypothesis.strategies


@hypothesis.strategies.composite
def reflection_test_values(
    draw: hypothesis.strategies.DrawFn,
    avoid_integers: bool = True,
) -> float:
    """Generate values for testing reflection formulas."""
    x = draw(
        hypothesis.strategies.floats(
            min_value=0.01,
            max_value=0.99,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    if avoid_integers:
        # Ensure we're not too close to integers
        return x
    return x
