import requests
import json
import os
with open("C:/Users/tunahan/Downloads/Telegram Desktop/lipyum_st_agency_calls (3).json", 'r', encoding='utf-8') as f:
    data = json.load(f)
download_folder = 'indirilen_ses_dosyalarii'
os.makedirs(download_folder, exist_ok=True)

for index, item in enumerate(data):
    url = item["record"]  # JSON dosyanızda URL'nin bulunduğu anahtar
    response = requests.get(url)
    
    # Dosya ismini oluşturun
    filename = os.path.join(download_folder, f"ses_{index+51}.wav")
    
    # Dosyayı kaydedin
    with open(filename, 'wb') as f:
        f.write(response.content)
    
    print(f"{filename} indirildi.")

