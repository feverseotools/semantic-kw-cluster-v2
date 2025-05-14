# Semantic Keyword Clustering - Development Guide

## Table of Contents
- [Project Setup](#project-setup)
- [Development Environment](#development-environment)
- [Project Structure](#project-structure)
- [Contributing Guidelines](#contributing-guidelines)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Documentation](#documentation)
- [Deployment](#deployment)

## Project Setup

### Prerequisites
- Python 3.8+
- pip or conda
- Git

### Installation for Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/semantic-keyword-clustering.git
cd semantic-keyword-clustering
```

2. Create a virtual environment:
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Or using conda
conda create -n semantic-clustering python=3.8
conda activate semantic-clustering
```

3. Install development dependencies:
```bash
# Install in editable mode with development extras
pip install -e ".[dev]"

# Or install development requirements
pip install -r requirements-dev.txt
```

4. Install additional development tools:
```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

## Development Environment

### Recommended Tools
- IDE: PyCharm, VSCode
- Python Version: 3.8 - 3.10
- Virtual Environment: venv or conda
- Version Control: Git

### Environment Configuration

Create a `.env` file for local development settings:
```ini
# Example environment variables
OPENAI_API_KEY=your_api_key_here
LOGGING_LEVEL=DEBUG
```

## Project Structure

### Key Directories
- `semantic_clustering/`: Main package source code
- `tests/`: Test suite
- `docs/`: Project documentation
- `data/`: Sample data and models
- `scripts/`: Utility scripts

### Package Components
- `clustering/`: Clustering algorithms and embeddings
- `nlp/`: Natural Language Processing utilities
- `export/`: Export functionality for reports

## Contributing Guidelines

### Contribution Process
1. Fork the repository
2. Create a feature branch
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes
4. Write or update tests
5. Run tests and linters
6. Commit with a clear, descriptive message
7. Push to your fork
8. Create a Pull Request

### Commit Message Convention
```
<type>(<scope>): <description>

[optional body]
[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

## Development Workflow

### Code Development
- Follow PEP 8 style guidelines
- Use type hints
- Write docstrings for all functions and classes
- Keep functions small and focused

### Running the Application
```bash
# Run Streamlit application
streamlit run semantic_clustering/app.py

# Run CLI
python -m semantic_clustering --input keywords.csv
```

## Testing

### Running Tests
```bash
# Run all tests
pytest

# Run specific test module
pytest tests/test_clustering.py

# Run with coverage
pytest --cov=semantic_clustering
```

### Test Coverage
- Aim for > 80% code coverage
- Write unit tests for all major components
- Include edge cases and error scenarios

## Code Quality

### Linting and Formatting
```bash
# Run Black for formatting
black .

# Run Flake8 for linting
flake8 semantic_clustering

# Run MyPy for type checking
mypy semantic_clustering
```

### Pre-commit Hooks
Create `.pre-commit-config.yaml`:
```yaml
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files

-   repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
    -   id: black

-   repo: https://github.com/PyCQA/flake8
    rev: 6.0.0
    hooks:
    -   id: flake8
```

## Documentation

### Updating Docs
- Keep `README.md` up to date
- Update `docs/` with latest changes
- Add docstrings to new functions
- Generate API documentation using Sphinx

### Building Documentation
```bash
# Install documentation tools
pip install sphinx sphinx_rtd_theme

# Generate documentation
cd docs
make html
```

## Deployment

### Release Process
1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create a git tag
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

### Publishing to PyPI
```bash
# Build distribution
python -m build

# Upload to PyPI
twine upload dist/*
```

## Troubleshooting

### Common Issues
- Ensure all NLTK and spaCy resources are downloaded
- Check compatibility of dependencies
- Verify Python and library versions

### Getting Help
- Open an issue on GitHub
- Check documentation
- Join our community discussion channels

## Advanced Development

### Adding New Features
1. Open an issue to discuss the feature
2. Create a detailed implementation plan
3. Write tests before implementation
4. Follow code review process

### Performance Optimization
- Profile code using `cProfile`
- Use vectorized operations
- Consider Numba or Cython for critical paths

## Contributing

We welcome contributions! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
