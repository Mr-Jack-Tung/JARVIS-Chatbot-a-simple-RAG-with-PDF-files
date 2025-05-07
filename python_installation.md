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
poetry install
poetry update
poetry lock
poetry check
poetry list


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
