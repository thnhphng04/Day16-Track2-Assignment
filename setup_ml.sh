sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-venv
python3 -m pip install --upgrade pip
pip3 install lightgbm scikit-learn pandas numpy kaggle
mkdir -p ~/ml-benchmark
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json << 'EOF'
{"username": "thanhphongvjpro", "key": "KGAT_1d163fd15adc009cac71d7bcc1975deb"}
EOF
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p ~/ml-benchmark/
