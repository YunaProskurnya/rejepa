"""One-time encoder passes: store one fresh crop realization per epoch on disk."""

from evaluate_linear_cached import main


if __name__ == '__main__':
    main(prepare=True)
