# Installation

This guide will help you install GatorAI for development or production use.

## Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- Git (for cloning the repository)

## Quick Install (PyPI)

```bash
pip install gatorai
```

## Development Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/gatorai.git
cd gatorai
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

For basic usage:
```bash
pip install -r requirements.txt
```

For development (includes testing, linting, docs):
```bash
pip install -e ".[dev,docs]"
```

Or using the new pyproject.toml:
```bash
pip install -e .
pip install -e ".[dev]"
pip install -e ".[docs]"
```

## Docker Installation

### Using Docker Compose (Recommended)

```bash
# Clone repository
git clone https://github.com/your-org/gatorai.git
cd gatorai

# Start all services
docker-compose up -d

# Access dashboard at http://localhost:8501
# Access Jupyter at http://localhost:8888 (token: gatorai2024)
```

### Using Docker Directly

```bash
# Build image
docker build -t gatorai .

# Run container
docker run -p 8501:8501 gatorai
```

## Platform-Specific Instructions

### Windows

```powershell
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### macOS

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Linux

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Optional Dependencies

### Jupyter Notebook Support

```bash
pip install jupyter
```

### API Server (FastAPI)

```bash
pip install -e ".[api]"
```

## Verification

After installation, verify everything works:

```bash
# Test import
python -c "import gatorai; print('GatorAI imported successfully')"

# Run basic test
python -m pytest tests/test_data.py -v

# Launch dashboard
gatorai-dashboard
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're in the correct virtual environment and have installed dependencies.

2. **Permission Errors**: On Linux/macOS, you might need to adjust permissions for data directories.

3. **Memory Issues**: For large datasets, ensure you have sufficient RAM (8GB+ recommended).

### Getting Help

- Check the [FAQ](faq.md)
- Open an issue on [GitHub](https://github.com/your-org/gatorai/issues)
- Join our [Discord community](https://discord.gg/gatorai)

## Next Steps

- [Quick Start Guide](quick-start.md)
- [Configuration](configuration.md)
- [User Guide](../user-guide/data-pipeline.md)
