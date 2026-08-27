"""Real dataset generation entrypoint; backend adapters must be chosen explicitly."""

from __future__ import annotations

import argparse

from src.audit import load_config


def main() -> None:
    """Reject mock research collection and identify an unimplemented backend."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    backend = config["generation"]["backend"]
    if backend == "mock":
        raise SystemExit(
            "Refusing to collect research data with the mock backend. Add and test a real backend "
            "adapter, then update configs/pilot.yaml and the preregistration."
        )
    raise SystemExit(f"Backend adapter {backend!r} is not implemented yet.")


if __name__ == "__main__":
    main()
