# Installation Guide

## Prerequisites

- **Conda**: Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).
- **Python 3.8+**: Kamaji requires Python 3.8 or newer.

## Step 1: Clone the Repository

```bash
git clone https://github.com/JCorbin406/Kamaji.git
cd Kamaji
```

## Step 2: Create the Conda Environment

Kamaji includes an `environment.yml` file for easy setup.

### 2.1 Create the Environment

```bash
conda env create -f environment.yml
```

This creates a new Conda environment (typically named `kamaji`) with all dependencies.

### 2.2 Activate the Environment

```bash
conda activate kamaji
```

> 🪟 On Windows, use **Anaconda Prompt** or install Git Bash for bash-style commands.

---

## Step 3: Install the Kamaji Package

If you're doing development or want editable access:

```bash
pip install -e .
```

This ensures that local changes are immediately reflected when using the package.

To install additional packages:

```bash
conda install <package>
# or
pip install <package>
```

---

## Step 4: Verify Installation

You can check that Kamaji is installed by running:

```bash
python -c "import kamaji; print(kamaji.__version__)"
```

If successful, you'll see the version number printed.

---

## Step 5: Running Kamaji

To launch the GUI:

```bash
python kamaji/gui/gui_main.py
```

Or run a headless script:

```bash
python examples/basic_simulation.py
```

(Coming soon: More examples and CLI tools.)

---

## Step 6: Deactivate the Environment

```bash
conda deactivate
```

---

## Step 7: Updating the Environment

To sync the environment with any changes in `environment.yml`:

```bash
conda env update -f environment.yml --prune
```

> The `--prune` option removes packages not listed in the file.

---

# Development

Want to contribute? Fork the repo and open a pull request with:

- Clear, documented changes
- Unit tests (if applicable)
- A brief description in the PR

Kamaji adheres to standard Python code style (PEP8). Contributions are welcome!

---

# License

Kamaji is released under the [MIT License](LICENSE).