#!/bin/bash
set -e

cd /share/home/u2515283028/caries_project

module load http-proxy 2>/dev/null || true

mkdir -p external_repos
cd external_repos

echo "===== 清理旧目录 ====="
rm -rf AHF-Fusion-U-Net AHF-Fusion-U-Net-main AHF-Fusion-U-Net.zip

echo "===== 尝试 git clone ====="
if git clone https://github.com/AfsanaAhmedMunia/AHF-Fusion-U-Net.git; then
  echo "git clone 成功"
else
  echo "git clone 失败，改用 wget 下载 zip"
  wget -O AHF-Fusion-U-Net.zip https://github.com/AfsanaAhmedMunia/AHF-Fusion-U-Net/archive/refs/heads/main.zip
  unzip -o AHF-Fusion-U-Net.zip
fi

cd /share/home/u2515283028/caries_project

echo "===== 检查是否真的下载到 notebook ====="
find external_repos -maxdepth 3 -type f | grep -E "AHF.*ipynb|UA_AHF.*ipynb|README|LICENSE" || true

if ! find external_repos -maxdepth 3 -type f | grep -q "AHF_U_Net"; then
  echo "错误：没有找到 AHF_U_Net notebook，说明下载失败。"
  exit 1
fi

echo "===== 打包官方代码 ====="
rm -f ahf_official_code_for_chatgpt.tar.gz
tar -czf ahf_official_code_for_chatgpt.tar.gz external_repos

echo "===== 打包完成 ====="
ls -lh ahf_official_code_for_chatgpt.tar.gz
tar -tzf ahf_official_code_for_chatgpt.tar.gz | head -50
