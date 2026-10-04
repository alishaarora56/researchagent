"""Allow the CLI to run with ``python -m research_agent``."""

from research_agent.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
