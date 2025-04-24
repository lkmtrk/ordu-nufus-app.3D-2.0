import pandas as pd

# 1) Excel dosyasını oku
df = pd.read_excel("koordinatlı_nufus_verisi.xlsx")

# 2) Parquet formatına dönüştür ve kaydet
df.to_parquet("koordinatlı_nufus_verisi.parquet", index=False)

print("Parquet dosyası oluşturuldu: koordinatlı_nufus_verisi.parquet")
