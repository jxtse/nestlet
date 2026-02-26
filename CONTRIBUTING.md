# Contributing to Nestlet

Thank you for your interest in contributing to Nestlet! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/jxtse/nestlet.git
   cd nestlet
   ```
3. Install [uv](https://github.com/astral-sh/uv) (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
4. Install development dependencies:
   ```bash
   uv sync --extra dev
   ```

## Development Workflow

### Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **Ruff** for linting
- **MyPy** for type checking

Before submitting a PR, please run:

```bash
# Format code
uv run black .

# Lint
uv run ruff check .

# Type check
uv run mypy inception/
```

### Running Tests

```bash
uv run pytest tests/
```

### Commit Messages

Please use clear, descriptive commit messages. We recommend following the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `refactor:` for code refactoring
- `test:` for adding or updating tests

## Pull Request Process

1. Create a new branch for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them

3. Push to your fork and submit a pull request

4. Ensure all CI checks pass

5. Wait for review and address any feedback

## Reporting Issues

When reporting issues, please include:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Your environment (Python version, OS, etc.)

## Feature Requests

We welcome feature requests! Please open an issue and describe:

- The problem you're trying to solve
- Your proposed solution
- Any alternatives you've considered

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to maintain a welcoming and inclusive community.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
