import subprocess
import sys


def main():
    commands = (
        ("Training candidate model", "scripts.train_model"),
        ("Evaluating candidate model", "scripts.evaluate_model"),
    )

    for step, module in commands:
        print(f"\n{step}...", flush=True)
        subprocess.run(
            [sys.executable, "-m", module],
            check=True,
        )


if __name__ == "__main__":
    main()
