# Kamaji

Kamaji is a multi-agent simulation package designed for modeling and running complex agent-based simulations. This guide will walk you through the steps to install the package and set up the environment using Conda.

## Installation Guide

### Prerequisites

- **Conda**: You will need Conda (or Miniconda) installed on your machine. You can download Conda from [Anaconda's website](https://www.anaconda.com/products/distribution).
  
- **Python 3.8+**: This package is compatible with Python 3.8 and above.

### Step 1: Clone the Repository

Start by cloning the Kamaji repository to your local machine.

```bash
git clone https://github.com/JCorbin406/Kamaji.git
cd Kamaji
```

### Step 2: Create a Conda Environment

The easiest way to create a Conda environment with all the required dependencies is to use the environment.yml file provided in this repository.

#### Step 2.1: Create the Conda Environment

Run the following command in the repository directory to create the Conda environment:

```bash
conda env create -f environment.yml
```

This will create a new Conda environment named kamaji with all the dependencies listed in the environment.yml file. 

#### Step 2.2: Activate the Environment

Activate the environmnet using the following command:

```bash
conda activate kamaji
```

### Step 3: Install the Kamaji Package

Once the Conda environment is activated, you can install the Kamaji package in editable mode by running the following command:

```bash
pip install -e .
```

THis will install the Kamaji package and any additional dependencies needed for the package to work. 

### Step 4: Verify Installation

To verify that Kamaji has been installed correctly, you can run the following command to check the installed version:

```bash
python -c "import kamaji; print(kamaji.__version__)"
```

If Kamaji is installed correctly, you should see the version number printed in the terminal. 

### Step 5: Running Kamaji

TBD 

### Step 6: Deactivate the Conda Environment

When you're done, you can deactivate the Conda environment:

```bash
conda deactivate
```

### Step 7: Updating Dependencies

To update the Conda environment and its dependencies, you can run:

```bash
conda env update -f environment.yml
```

This will update the environment with any new dependencies that have been added to the environment.yml file. 

## Development 

If you'd like to contribute to the Kamaji project, please fork the repository and create a pull request with your changes. Ensure that you follow the code style and include appropriate tests for new features or bug fixes.

## License 

Kamaji is licensed under the MIT License. See the LICENSE file for more details.
