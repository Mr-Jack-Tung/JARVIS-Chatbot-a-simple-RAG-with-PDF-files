# Cách cài đặt Python 3.12 trên macOS:

## 1. Dùng Homebrew (Brew)
### Cài Homebrew nếu chưa có
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"  
### Cài Python 3.12
brew install python@3.12  
### Link vào python3 mặc định (tuỳ chọn)
brew link --overwrite --force python@3.12  
### Kiểm tra phiên bản
python3.12 --version

### Cài đặt pip và venv
python3.12 -m pip install --upgrade pip
python3.12 -m venv .venv
source .venv/bin/activate

pip install poetry
pip install --upgrade pip

poetry lock
poetry install
poetry update
poetry lock
poetry check
poetry list

poetry add httpx==0.27.0
poetry add gpt4all@^2.8.2

poetry env use 3.11
poetry lock --no-cache --regenerate

brew install mactex # Cài LaTeX

Note: By downgrading to NumPy 1.26.4, we provided a version that PyTorch 2.2.2 can properly interact with Python 3.10~3.12

## UV help
uv supports pylock.toml as an export target and in the uv pip CLI. For example:
- To export a uv.lock to the pylock.toml format, run: uv export -o pylock.toml
- To generate a pylock.toml file from a set of requirements, run: uv pip compile -o pylock.toml requirements.txt
- To install from a pylock.toml file, run: uv pip sync pylock.toml or uv pip install -r pylock.toml

## 2. Dùng Anaconda
### Cài Anaconda
https://www.anaconda.com/products/distribution
### Tạo môi trường conda
conda create -n jarvis python=3.12
### Activate môi trường
conda activate jarvis
### Kiểm tra phiên bản
python --version


## 3. Dùng Miniconda
### Cài Miniconda
https://docs.conda.io/en/latest/miniconda.html
### Tạo môi trường conda
conda create -n jarvis python=3.12
### Activate môi trường
conda activate jarvis
### Kiểm tra phiên bản
python --version


## 4. Dùng pyenv
### Cài pyenv
https://github.com/pyenv/pyenv
### Tạo môi trường pyenv
pyenv install 3.12.4
pyenv virtualenv 3.12.4 jarvis
pyenv activate jarvis
### Kiểm tra phiên bản
python --version


## 5. Dùng pyenv-virtualenv
### Cài pyenv-virtualenv
https://github.com/pyenv/pyenv-virtualenv
### Tạo môi trường pyenv-virtualenv
pyenv virtualenv 3.12.4 jarvis
pyenv activate jarvis
### Kiểm tra phiên bản
python --version

## 6. Dùng Official installer (python.org)
### Cài Python 3.12
Vào https://www.python.org/downloads/release/python-3124/
Tải file macOS 64-bit installer (.pkg)
Mở và làm theo wizard để cài đặt
Sau đó python3.12 --version

# Cách cài đặt Python 3.12 trên Windows:

## 1. Dùng Official installer (python.org)
### Cài Python 3.12
Vào https://www.python.org/downloads/release/python-3124/
Tải file Windows installer (.exe)
Mở và làm theo wizard để cài đặt
Sau đó python3.12 --version

## 2. Dùng Anaconda
### Cài Anaconda
https://www.anaconda.com/products/distribution
### Tạo môi trường conda
conda create -n jarvis python=3.12
### Activate môi trường
conda activate jarvis
### Kiểm tra phiên bản
python --version

## 3. Dùng Miniconda
### Cài Miniconda
https://docs.conda.io/en/latest/miniconda.html
### Tạo môi trường conda
conda create -n jarvis python=3.12
### Activate môi trường
conda activate jarvis
### Kiểm tra phiên bản
python --version

## 4. Dùng pyenv
### Cài pyenv
https://github.com/pyenv/pyenv
### Tạo môi trường pyenv
pyenv install 3.12.4
pyenv virtualenv 3.12.4 jarvis
pyenv activate jarvis
### Kiểm tra phiên bản
python --version

## 5. Dùng pyenv-virtualenv
### Cài pyenv-virtualenv
https://github.com/pyenv/pyenv-virtualenv
### Tạo môi trường pyenv-virtualenv
pyenv virtualenv 3.12.4 jarvis
pyenv activate jarvis
### Kiểm tra phiên bản
python --version


# Cách khởi tạo và cài đặt Poetry trên môi trường và dự án đang sẵn có:
https://python-poetry.org/docs/basic-usage/

## 1. Khởi tạo project mới với poetry
cd /path/to/project/folder
poetry init # Follow prompts or provide answers in a config file.
poetry add <package_name> # Add packages one by one.
poetry remove <package_name>
poetry show # Show installed packages.

or:
First, generate a `requirements.txt` file using `pip freeze > requirements.txt`. Then, you can use a loop to add each package from the `requirements.txt` file to your `pyproject.toml` using a loop:
You can use a loop to iterate through each package in the `requirements.txt` file and add it individually using `poetry add`.

```
while IFS= read -r package; do
  poetry add "$package"
done < requirements.txt
```

Other commands:
poetry add <package_name> --group group_name # Add dependencies under different groups defined in pyproject.toml.
poetry add <package_name> --dev # Add dev dependencies.
poetry run <command> # Run commands within the virtual environment.
poetry build # Build distribution archives.
poetry publish # Publish built distributions.
poetry export --format requirements.txt > requirements.txt # Export dependencies as a requirements file.
poetry lock # Lock dependencies and write them into the lockfile.

## 2. Cài đặt Poetry trong một dự án đã tồn tại
cd /path/to/existing/project/folder
poetry install # Install all dependencies listed in pyproject.toml.
poetry update # Update dependencies based on latest versions specified in pyproject.toml.
poetry add <package_name> # Add new dependencies to both pyproject.toml and lock file.
poetry remove <package_name> # Remove dependencies from both pyproject.toml and lock file.
poetry show # List currently installed dependencies along with their versions.
poetry env info # Display information about the current virtual environment.
poetry shell # Open an interactive shell session within the virtual environment.
poetry run <command> # Execute any command inside the virtual environment without activating it manually.
poetry version # Print the current project version number.
poetry version <new_version_number> # Set a new version for the project.
poetry config # View configuration settings related to Poetry itself.
poetry config --list # List all available configuration options.
poetry config --unset key # Unset specific configuration keys.
poetry config --local key value # Modify local configuration values directly in the pyproject.toml file.
poetry config --global key value # Change global configuration settings across multiple projects.
