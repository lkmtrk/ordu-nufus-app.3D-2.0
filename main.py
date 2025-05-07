import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
import base64
import json
import re
from io import BytesIO
import altair as alt
import numpy as np
from streamlit.components.v1 import html

# -----------------------------
# 0) SAYFA AYARLARI & HEADER
# -----------------------------
st.set_page_config(page_title="Ordu İli Nüfus Analizi",layout="wide", initial_sidebar_state="expanded")
st.markdown("<meta name='language' content='tr'>", unsafe_allow_html=True)

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = get_base64_image("logo.png")

st.markdown(f"""
<div style="display: flex; align-items: center; justify-content: center; gap: 25px; padding: 15px 0;">
    <img src="data:image/png;base64,{logo_base64}" width="140">
    <div style="text-align: left;">
        <h2 style="margin: 0; color: white;">NÜFUS ANALİZ PORTALİ</h2>
    </div>
</div>
<hr>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; font-size: 16px; color: #ccc; margin-top: -10px; margin-bottom: 10px;'>
Bu uygulama Ordu iline ait nüfus verilerini yıl, ilçe ve mahalle bazında analiz etmenizi sağlar. 
Aşağıdaki grafikler üzerinden verileri karşılaştırabilir ve Excel formatında indirebilirsiniz.
</div>
""", unsafe_allow_html=True)

# -----------------------------
# 0) ÖNBELLEKLENMİŞ FONKSİYONLAR
# -----------------------------

@st.cache_data
def load_parquet_data(path: str) -> pd.DataFrame:
    """Loads the full parquet data."""
    try:
        df = pd.read_parquet(path)  # all columns
        df.rename(columns={"Latitude":"lat","Longitude":"lon"}, inplace=True, errors="ignore")
        return df
    except FileNotFoundError:
        st.error(f"Veri dosyası bulunamadı: {path}")
        st.stop()
    except Exception as e:
        st.error(f"Veri yüklenirken hata oluştu {path}: {e}")
        st.stop()


@st.cache_data
def load_parquet_ilce(path: str, year: str) -> pd.DataFrame:
    """Loads and aggregates district-level population data for a specific year."""
    cols = ["İLÇE", f"{year} YILI NÜFUSU", "Latitude", "Longitude"]
    try:
        df = pd.read_parquet(path, columns=cols)
    except FileNotFoundError:
        st.error(f"İlçe veri dosyası bulunamadı: {path}")
        st.stop()
    except Exception as e:
        st.error(f"İlçe verisi yüklenirken hata oluştu {path} ({year} yılı): {e}")
        st.stop()
    df = df.rename(columns={
        f"{year} YILI NÜFUSU": "NÜFUS",
        "Latitude": "lat",
        "Longitude": "lon"
    })
    df_ilce = (
        df
        .groupby("İLÇE", as_index=False)
        .agg({
            "NÜFUS": "sum",
            "lat":  "mean",
            "lon":  "mean"
        })
    )
    return df_ilce

@st.cache_data
def load_parquet_mahalle(path: str, year: str) -> pd.DataFrame:
    """Loads mahalle-level population data for a specific year."""
    cols = [
        "İLÇE",
        "MAHALLE",
        "MAHALLE KODU (AKS)",
        f"{year} YILI NÜFUSU",
        "KONUMA GİT"
    ]
    try:
        df = pd.read_parquet(path, columns=cols)
    except FileNotFoundError:
        st.error(f"Mahalle veri dosyası bulunamadı: {path}")
        st.stop()
    except Exception as e:
        st.error(f"Mahalle verisi yüklenirken hata oluştu {path} ({year} yılı): {e})")
        st.stop()
    df = df.rename(
        columns={
            f"{year} YILI NÜFUSU": "NÜFUS",
            "Latitude": "lat",
            "Longitude": "lon",
        },
        errors="ignore"
    )
    return df

# load_parquet_demo fonksiyonu, demografi haritası için tek bir yüzde sütunu yüklüyor
@st.cache_data
def load_parquet_demo(path: str, pct_col: str) -> pd.DataFrame:
    """Loads demographic data for a single percentage column."""
    cols = [
        "İLÇE",
        "MAHALLE",
        "MAHALLE KODU (AKS)",
        pct_col, # Load only the selected percentage column
        "Latitude",
        "Longitude"
    ]
    try:
        df = pd.read_parquet(path, columns=cols)
    except FileNotFoundError:
        st.error(f"Demografi (tek sütun) veri dosyası bulunamadı: {path}")
        st.stop()
    except KeyError as e:
        st.error(f"Demografi (tek sütun) verisi yüklenirken beklenen sütun bulunamadı: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Demografi (tek sütun) verisi yüklenirken hata oluştu {path}: {e}")
        st.stop()

    df = df.rename(
        columns={
            pct_col: "PCT", 
            "Latitude": "lat",
            "Longitude": "lon",
        },
        errors="ignore"
    )
    return df


@st.cache_data(show_spinner=False)
def get_demo_data():
    import os
    parquet_path = "bar_grafik_verisi.parquet"
    # Eğer Parquet cache varsa doğrudan yükle
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
    else:
        # Excel'den oku, temizle ve Parquet'e yaz
        df = pd.read_excel("bar_grafik_verisi.xlsx")
        # Sütun isimlerini normalize et
        df.columns = [col.strip() for col in df.columns]
        # Yüzde sütunlarını float'a çevir
        for col in df.columns:
            if col.upper().endswith("YAŞ YÜZDE"):
                df[col] = (
                    df[col].astype(str)
                         .str.replace('%','', regex=False)
                         .str.replace(',','.', regex=False)
                         .astype(float)
                )
        # Parquet olarak kaydet
        df.to_parquet(parquet_path)
    return df

# load_all_age_demographics fonksiyonu, dropdown grafik için tüm yaş yüzde sütunlarını yükler
@st.cache_data
def load_all_age_demographics() -> pd.DataFrame:
    """Loads demographic data including all age percentage columns for the dropdown graph."""
    # pct_columns değişkeninin global kapsamda tanımlı olduğundan emin olun.
    global pct_columns # Kullanıcının sağladığı snippet'te pct_columns kullanılıyor
    # global all_pct_columns # Önceki versiyonlarla tutarlılık için tutalım

    cols_to_load = ["İLÇE", "MAHALLE", "MAHALLE KODU (AKS)"] + pct_columns + ["Latitude", "Longitude"]

    try:
         df_ages = pd.read_parquet("koordinatlı_nufus_verisi.parquet", columns=cols_to_load)
         df_ages.rename(columns={"Latitude": "lat", "Longitude": "lon"}, inplace=True, errors="ignore")

    except KeyError as e:
        st.error(f"Demografi veri dosyası (dropdown için) yüklenirken beklenen sütunlardan biri bulunamadı: {e}")
        return pd.DataFrame()
    except FileNotFoundError:
        st.error(f"Demografi veri dosyası (dropdown için) bulunamadı: koordinatlı_nufus_verisi.parquet")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Demografi verisi (dropdown için) yüklenirken hata oluştu: {e}")
        return pd.DataFrame()

    return df_ages


@st.cache_data
def load_geojson(path: str) -> dict:
    """Loads GeoJSON data from a file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"GeoJSON dosyası bulunamadı: {path}")
        st.stop()
    except json.JSONDecodeError:
        st.error(f"GeoJSON dosyası çözümlenirken hata oluştu: {path}. Lütfen dosya formatını kontrol edin.")
        st.stop()
    except Exception as e:
        st.error(f"GeoJSON dosyası yüklenirken hata oluştu {path}: {e}")
        st.stop()


@st.cache_data
def build_geo_lookup(geojson: dict, key_prop: str) -> dict:
    """Builds a lookup dictionary from GeoJSON features."""
    if not geojson or not isinstance(geojson, dict) or 'features' not in geojson:
        st.warning("Geçersiz veya boş GeoJSON sağlandı.")
        return {}
    return {
        feat["properties"].get(key_prop): feat
        for feat in geojson.get("features", [])
        if feat.get("properties") and key_prop in feat["properties"]
    }

def rgba_list_to_full_css_string(color_list, nan_color=[128, 128, 128, 100]):
    """Converts [r, g, b, a] list (0-255) to full CSS 'rgba(r,g,b,a)' string (alpha 0-1)."""
    if color_list is None or len(color_list) != 4:
        r, g, b, a = nan_color
        return f"rgba({r},{g},{b},{a/255.0:.2f})"
    r, g, b, a = color_list
    r, g, b, a = max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)), max(0, min(255, a))
    return f"rgba({r},{g},{b},{a/255.0:.2f})"

# Function to create safe column names (kullanılmasa bile tutalım)
def create_safe_col_name(col_name):
    """Converts column names to a safe format."""
    safe_name = col_name.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct").replace(".", "_").replace("-", "_")
    return safe_name


# -----------------------------
# 1) SAYFA KONFİGÜRASYONU & META YÜKLEME
# -----------------------------


# Tüm sütun isimleri ve lat/lon’u almak için
df_full = load_parquet_data("koordinatlı_nufus_verisi.parquet")

# --- Create Safe Column Name Mapping (Kullanılmasa bile tutalım) ---
col_name_mapping = {}
if df_full is not None and not df_full.empty:
    for col in df_full.columns:
        safe_col = col.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct").replace(".", "_").replace("-", "_")
        col_name_mapping[col] = safe_col
else:
     st.error("Tam veri yüklenemediği veya boş olduğu için sütun eşleşmesi oluşturulamadı.")
     col_name_mapping = {}

# Create reverse mapping (useful for displaying original names)
if col_name_mapping:
     safe_to_original_col_mapping = {v: k for k, v in col_name_mapping.items()}
else:
     safe_to_original_col_mapping = {}


year_columns  = [c for c in df_full.columns if "YILI NÜFUSU" in c]
dropdown_years = [c.split()[0] for c in year_columns]

pct_columns = [c for c in df_full.columns if c.endswith(" YAŞ YÜZDE")]
pct_labels  = [c.replace(" YAŞ YÜZDE","") for c in pct_columns]
label_to_col = dict(zip(pct_labels, pct_columns)) 

all_pct_columns = pct_columns #


if not df_full.empty and 'lat' in df_full.columns and 'lon' in df_full.columns:
    center_lat = df_full["lat"].mean()
    center_lon = df_full["lon"].mean()
else:
    center_lat = 41.0
    center_lon = 38.0

ilce_geojson_path = "ILCELER.geojson"
mahalle_geojson_path = "MAHALLELER.geojson"

ilce_geojson = load_geojson(ilce_geojson_path)
ilce_lookup  = build_geo_lookup(ilce_geojson, "AD")

mahalle_geojson = load_geojson(mahalle_geojson_path)
mahalle_lookup  = build_geo_lookup(mahalle_geojson, "KOD")

# -------------------------------
# 1. ORDU İLİ NÜFUS ANALİZİ 
# -------------------------------

# ──────────── 4) PIVOT TO LONG ────────────
year_cols = [c for c in df_full.columns if c.strip().startswith("20") and "YILI NÜFUSU" in c]
df_long = pd.melt(
    df_full,
    id_vars=["İLÇE", "MAHALLE"],
    value_vars=year_cols,
    var_name="YIL",
    value_name="NÜFUS (KİŞİ SAYISI)"
)
df_long["YIL"] = df_long["YIL"].str.extract(r"(20\d{2})")
df_long["NÜFUS (KİŞİ SAYISI)"] = pd.to_numeric(df_long["NÜFUS (KİŞİ SAYISI)"], errors="coerce")
years = sorted(df_long["YIL"].dropna().unique().tolist())

# ──────────── 5) SELECTIONS & FIRST CHART ────────────

st.markdown("### 📈Ordu İli Nüfus Nüfus Analizi")

# Orta bloğu 3 kolonlu dış düzenle sarıyoruz (1-2-1)
outer1, outer2, outer3 = st.columns([1, 2, 1])
with outer2:
    # içte 2 kolon: biri Başlangıç, diğeri Bitiş
    col1, col2 = st.columns(2)
    start_year = col1.selectbox("Başlangıç Yılı", years, index=0)
    end_year   = col2.selectbox("Bitiş Yılı",     years, index=len(years)-1)

    if start_year > end_year:
        st.warning("Başlangıç yılı, bitiş yılından büyük olamaz!")
    else:
        df_filtered = df_long[(df_long["YIL"] >= start_year) & (df_long["YIL"] <= end_year)]

        st.markdown(
        f"<h4 style='font-size:22px; margin-bottom: 8px;'>📈 Genel Nüfus Değişimi ({start_year} - {end_year})</h4>",
        unsafe_allow_html=True
        )
        ordu_geneli = (
            df_filtered
            .groupby("YIL")["NÜFUS (KİŞİ SAYISI)"]
            .sum()
            .reset_index()
        )
        st.plotly_chart(
            px.line(ordu_geneli, x="YIL", y="NÜFUS (KİŞİ SAYISI)", markers=True),
            key="chart_ordu"
        )



# -------------------------------
# 2. İLÇE BAZLI NÜFUS Harita 
# -------------------------------

st.markdown("### 🗺️ İlçe Bazlı Nüfus Haritası (Yıl & Nüfus Aralığı)")

# use_container_width=True kaldırıldı
secili_yil_ilce = st.selectbox(
    "İlçe Haritası için Yıl Seçiniz",
    dropdown_years,
    index=dropdown_years.index("2024"),  # 2024 varsayılan
    key="ilce_yil"
)

if secili_yil_ilce:
    # 1) Veri hazırlama
    df_ilce = load_parquet_ilce("koordinatlı_nufus_verisi.parquet", secili_yil_ilce)

    # 2) Filtre UI
    st.session_state.setdefault("ilce_filter", False)
    st.session_state.setdefault("ilce_range", "")

    def fmt_ilce():
        txt = st.session_state.ilce_range
        parts = re.split(r"[-–—]", txt)
        if len(parts) == 2:
            try:
                lo = int(parts[0].replace(".", ""))
                hi = int(parts[1].replace(".", ""))
                st.session_state.ilce_range = f"{lo}-{hi}"
            except:
                pass

    def clear_ilce_filter():
        st.session_state.ilce_range = ""
        st.session_state.ilce_filter = False

    st.text_input("Nüfus Aralığı Seç (örn: 500-1000)",
        key="ilce_range", placeholder="500-1000", on_change=fmt_ilce)

    c1, c2, _ = st.columns([1, 1, 8])
    with c1:
        ilce_gir = st.button("Gir", type="primary", key="ilce_gir", use_container_width=True)
    with c2:
        st.button("Temizle", type="secondary", key="ilce_temizle", use_container_width=True, on_click=clear_ilce_filter)

    if ilce_gir:
        st.session_state.ilce_filter = True

    # 3) Filtreleme
    if st.session_state.ilce_filter and st.session_state.ilce_range:
        try:
            lo, hi = map(int, st.session_state.ilce_range.split("-"))
            df_ilce = df_ilce[df_ilce["NÜFUS"].between(lo, hi)]
            st.markdown(f"**Seçilen İlçe Aralığı:** {lo} – {hi}")
            st.info(f"Kriterlere uygun {df_ilce.shape[0]} ilçe bulundu")
        except ValueError:
            st.error("Geçersiz nüfus aralığı formatı. Lütfen 'örn: 500-1000' gibi girin.")


    # 4) Formatlama & renk — vektörize
    df_ilce["NÜFUS_FMT"] = (df_ilce["NÜFUS"].astype(int).map("{:,.0f}".format).astype(str).str.replace(",", "."))

    # İlçe için kategorilere ayırma ve renk atama
    bins_i = [-float("inf"), 10000, 13000, 20000, 25000, 100000, 200000, float("inf")]
    colors_i = [
        [166,86,40,180],
        [152,78,163,180],
        [77,175,74,180],
        [55,126,184,180],
        [255,127,0,180],
        [255,255,51,180],
        [228,26,28,180],
    ]
    df_ilce["cat"]  = pd.cut(df_ilce["NÜFUS"], bins=bins_i, labels=False, right=True)
    df_ilce["color"] = df_ilce["cat"].map(dict(enumerate(colors_i)))
    df_ilce.drop(columns=["cat"], inplace=True)


    # 5) ColumnLayer
    layer_ilce = pdk.Layer("ColumnLayer", data=df_ilce,
                            get_position="[lon, lat]", get_elevation="NÜFUS",
                            elevation_scale=0.3, radius=3000, get_fill_color="color",
                            pickable=True, auto_highlight=True, extruded=True,)

    # 6) Sınırları gösterme checkbox’ı
    goster_ilce_sinirlar = st.checkbox("İlçe Sınırlarını Göster",
                                       value=True, key="show_ilce_borders")

    # 7) Filtrelenmiş GeoJSON oluştur
    allowed_ilce = set(df_ilce["İLÇE"])

    filtered_ilce_features = [
        ilce_lookup[name]
        for name in allowed_ilce
        if name in ilce_lookup
    ]
    filtered_ilce_geojson = {
        "type": "FeatureCollection",
        "features": filtered_ilce_features
    }

    # 8) Katman listesini derle
    layers = [layer_ilce]
    if goster_ilce_sinirlar:
        geojson_to_use = filtered_ilce_geojson if st.session_state.ilce_filter else ilce_geojson
        # GeoJSON geçerli ve feature içeriyor mu kontrolü
        if geojson_to_use and geojson_to_use.get('features'):
             border_layer = pdk.Layer(
                 "GeoJsonLayer",
                 geojson_to_use,
                 stroked=True,
                 filled=False,
                 get_line_color=[255, 0, 255, 200],
                 line_width_min_pixels=1
             )
             layers.append(border_layer)
        elif st.session_state.ilce_filter and not filtered_ilce_features:
            border_layer = None

    # 9) Haritayı çiz
    st.pydeck_chart(pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
        initial_view_state=pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=8,
            pitch=40
        ),
        layers=layers,
        tooltip={"text": "{İLÇE}: {NÜFUS_FMT}"}
    ))

# 10) Excel indirme butonları
ea, eb, _ = st.columns([1, 1, 8])
with ea:
    out_ilce = BytesIO()
    df_export = df_ilce.copy()
    df_export["YIL"] = secili_yil_ilce
    cols_to_export = ["İLÇE", "YIL", "NÜFUS"]
    cols_to_export = [c for c in cols_to_export if c in df_export.columns]
    df_export[cols_to_export].to_excel(out_ilce, index=False, sheet_name="Ham İlçe Verisi")
    st.download_button(
        "Ham Veriyi İndir",
        data=out_ilce.getvalue(),
        file_name=f"ilce_ham_{secili_yil_ilce}.xlsx",
        mime="application/vnd.openxmlformats-officedocument-spreadsheetml.sheet",
        use_container_width=True,
        type="secondary",
        key="ham_ilce_download"
    )

with eb:
    outp_ilce = BytesIO()
    df_piv_source = load_parquet_ilce("koordinatlı_nufus_verisi.parquet", secili_yil_ilce).copy()
    if st.session_state.ilce_filter and st.session_state.ilce_range:
        lo, hi = map(int, st.session_state.ilce_range.split("-"))
        df_piv_source = df_piv_source[df_piv_source["NÜFUS"].between(lo, hi)]

    if not df_piv_source.empty:
        piv = (
            df_piv_source[["İLÇE", "NÜFUS"]]
            .groupby("İLÇE", as_index=False)
            .sum()
            .assign(YIL=secili_yil_ilce)
        )
    else:
        # Boş DataFrame, ama sütunları tanımlı olsun
        piv = pd.DataFrame(columns=["İLÇE", "NÜFUS", "YIL"])

    # Genel Toplam satırını ekle
    totals = piv.select_dtypes(include="number").sum().to_dict()
    totals["İLÇE"] = "Genel Toplam"
    totals["YIL"] = ""   # yıl hücresi boş
    piv = pd.concat([piv, pd.DataFrame([totals])], ignore_index=True)

    # --- SÜTUN SIRASINI DÜZELT ---
    cols_to_export = ["İLÇE", "YIL", "NÜFUS"]
    # Var olanlar arasında sırayı uygula
    cols_to_export = [c for c in cols_to_export if c in piv.columns]
    piv = piv[cols_to_export]

    # Excel’e yaz
    with pd.ExcelWriter(outp_ilce, engine="xlsxwriter") as writer:
        piv.to_excel(writer, sheet_name="Pivot İlçe", index=False)

    # İndirme butonu (her koşulda gösterilir)
    st.download_button(
        "Pivot Tabloyu İndir",
        data=outp_ilce.getvalue(),
        file_name=f"ilce_pivot_{secili_yil_ilce}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        type="primary",
        key="pivot_ilce_download"
    )



# -------------------------------
# 2. MAHALLE BAZLI NÜFUS (Pydeck) 
# -------------------------------
st.markdown("### 🏘️ Mahalle Bazlı Nüfus Haritası (Yıl & Nüfus Aralığı)")

# use_container_width=True kaldırıldı
secili_yil_mahalle = st.selectbox(
    "Mahalle Haritası için Yıl Seçiniz",
    dropdown_years,
    index=dropdown_years.index("2024"),
    key="mahalle_yil"
)

if secili_yil_mahalle:
    # 1) Önbellekli mahalle verisini al
    df_mahalle = load_parquet_mahalle("koordinatlı_nufus_verisi.parquet", secili_yil_mahalle)


    # 2) Filtre UI ayarları
    st.session_state.setdefault("filter_active", False)
    st.session_state.setdefault("pop_min", None)
    st.session_state.setdefault("pop_max", None)
    st.session_state.setdefault("range_input", "")

    # 3) Sabit aralıklar (df değil df_mahalle)
    if not df_mahalle.empty and "NÜFUS" in df_mahalle.columns:
        min_pop, max_pop = int(df_mahalle["NÜFUS"].min()), int(df_mahalle["NÜFUS"].max())
    else:
        min_pop, max_pop = 0, 100000 # Varsayılan değerler veri yoksa
        st.warning("Mahalle nüfus verisi yüklenemedi veya boş.")

    sabit_araliklar = {
        f"{min_pop}-{500}": (min_pop, 500),
        "500-1000": (500, 1000),
        "1000-2000": (1000, 2000),
        "2000-5000": (2000, 5000),
        f"5000-{max_pop}": (5000, max_pop)
    }

    # 4) Formatter ve temizleme
    def _format_and_store():
        txt = st.session_state.range_input.strip()
        parts = re.split(r"\s*[-–—]\s*", txt)
        fmt = []
        for p in parts:
            nums = re.sub(r"\D", "", p)
            if nums:
                fmt.append(f"{int(nums):,}".replace(",", "."))
        if len(fmt) == 2:
            st.session_state.range_input = f"{fmt[0]}-{fmt[1]}"

    def clear_mahalle_filter():
        st.session_state.range_input = ""
        st.session_state.pop_min = None
        st.session_state.pop_max = None
        st.session_state.filter_active = False

    # 5) Aralık girişi ve butonlar
    st.text_input(
        "Nüfus Aralığı Seç (örn: 5.000-10.000)",
        key="range_input",
        placeholder="5.000-10.000 formatında",
        on_change=_format_and_store
    )
    c1, c2, _ = st.columns([1,1,8])
    with c1:
        gir = st.button("Gir", type="primary", key="mahalle_gir", use_container_width=True)
    with c2:
        st.button("Temizle", type="secondary", key="mahalle_temizle", use_container_width=True, on_click=clear_mahalle_filter)

    # 6) Filtre işleme
    if gir:
        raw = st.session_state.range_input.replace(".", "").replace(" ", "")
        if raw in sabit_araliklar:
            lo, hi = sabit_araliklar[raw]
        else:
            parts = re.split(r"[-–—]", raw)
            try:
                lo, hi = sorted(int(re.sub(r"\D","",p)) for p in parts if p)
            except:
                st.error("Geçersiz format. Örnek: 5.000-10.000 veya 500-1000")
                st.stop() # Hata durumunda dur
        st.session_state.pop_min = lo
        st.session_state.pop_max = hi
        st.session_state.filter_active = True

    df_mahalle_filtered = df_mahalle.copy() # Filtreleme için kopya üzerinde çalış

    if st.session_state.filter_active and not df_mahalle_filtered.empty:
        lo, hi = st.session_state.pop_min, st.session_state.pop_max
        if lo is not None and hi is not None: # lo ve hi tanımlıysa filtrele
             df_mahalle_filtered = df_mahalle_filtered[df_mahalle_filtered["NÜFUS"].between(lo, hi)].copy()
             st.markdown(f"**Seçilen Aralık:** {lo:,} – {hi:,}".replace(",", "."))
             count_ilce = df_mahalle_filtered["İLÇE"].nunique()
             count_mah  = df_mahalle_filtered.shape[0]
             st.info(f"Kriterlere uygun {count_ilce} ilçede {count_mah} mahalle bulundu")
        else:
             st.warning("Geçersiz nüfus aralığı filtresi.")


    # 7) Formatlama & renk — vektörize
    if not df_mahalle_filtered.empty:
        df_mahalle_filtered["NÜFUS_FMT"] = (df_mahalle_filtered["NÜFUS"].astype(int).map("{:,.0f}".format).str.replace(",", "."))

        # Mahalle için kategorilere ayırma ve renk atama
        bins_m = [-float("inf"), 5000, 10000, 15000, 20000, 25000, 30000, float("inf")]
        colors_m = [
            [166,86,40,180],
            [152,78,163,180],
            [77,175,74,180],
            [55,126,184,180],
            [255,127,0,180],
            [255,255,51,180],
            [228,26,28,180],
        ]
        df_mahalle_filtered["cat"]  = pd.cut(df_mahalle_filtered["NÜFUS"], bins=bins_m, labels=False, right=True)
        df_mahalle_filtered["color"] = df_mahalle_filtered["cat"].map(dict(enumerate(colors_m)))
        df_mahalle_filtered.drop(columns=["cat"], inplace=True)


        # 8) Kümeli ScatterplotLayer (Mahalle)
        clustered_mahalle_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_mahalle_filtered, # Filtrelenmiş veriyi kullan
            get_position="[lon, lat]",
            get_fill_color="color",
            get_radius=150,
            pickable=True,
            cluster=True,
            cluster_radius=50,
        )


        # 9) Sınır checkbox & GeoJSON filtresi
        show_borders = st.checkbox(
            "Mahalle Sınırlarını Göster",
            value=True,
            key="show_mahalle_borders_only"
        )

        # Seçili mahalle kodlarına göre filtrelenmiş GeoJSON
        allowed = set(df_mahalle_filtered["MAHALLE KODU (AKS)"].astype(int))
        filtered_mahalle_features = [
            mahalle_lookup[code]
            for code in allowed
            if code in mahalle_lookup
        ]
        # Hangi GeoJSON’u kullanacağımızı seçiyoruz
        if st.session_state.filter_active:
            geo_to_use = {
                "type": "FeatureCollection",
                "features": filtered_mahalle_features
            }
        else:
            geo_to_use = mahalle_geojson

        # 10) Harita çizimi
        layers_mahalle = [clustered_mahalle_layer]

        # Sınırları ekle (sadece mahalle sınırları, ilçe sınırları eklenmiyor)
        if show_borders and geo_to_use.get("features"):
            border_layer = pdk.Layer(
                "GeoJsonLayer",
                geo_to_use,
                stroked=True,
                filled=False,
                get_line_color=[3, 32, 252, 180],
                line_width_min_pixels=1
            )
            layers_mahalle.append(border_layer)

        st.pydeck_chart(pdk.Deck(
            map_style="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
            initial_view_state=pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=8,
                pitch=40
            ),
            layers=layers_mahalle,
            tooltip={
                "html": (
                    "<b>{MAHALLE}</b><br/>"
                    "İlçe: {İLÇE}<br/>"
                    f"Nüfus ({secili_yil_mahalle}): "+"{NÜFUS_FMT}"
                )
            }
        ))


        # Excel indirme butonları
        col_excel1, col_excel2, _ = st.columns([1, 1, 8])

    # Ham veri indir
    with col_excel1:
        output = BytesIO()

        # 1) İhracat DataFrame’ini hazırla ve sütun sırasını kesinleştir
        df_export_mahalle_ham = (
            df_mahalle_filtered
            .assign(YIL=secili_yil_mahalle)
            # burada ŞU SIRAYLA SEÇİYORUZ:
            [["İLÇE", "MAHALLE", "YIL", "NÜFUS", "KONUMA GİT"]]
        )

        # 2) Excel’e yaz ve 'KONUMA GİT' linklerini "Git" metniyle tıklanabilir yap
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            sheet_name = "Ham Mahalle Verisi"
            df_export_mahalle_ham.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]

            link_col_idx = df_export_mahalle_ham.columns.get_loc("KONUMA GİT")
            for row_idx, url in enumerate(df_export_mahalle_ham["KONUMA GİT"], start=1):
                if isinstance(url, str) and url.startswith("http"):
                    ws.write_url(row_idx, link_col_idx, url, string="Git")

        # 3) İndirme butonu
        st.download_button(
            "Ham Veriyi İndir",
            data=output.getvalue(),
            file_name=f"mahalle_ham_veri_{secili_yil_mahalle}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            type="secondary",
            key="ham_mahalle_download"
        )



    # Pivot tablo indir
    with col_excel2:
        pivot_output = BytesIO()
        # Pivot tablo için filtrelenmiş df_mahalle verisini kullan
        df_piv_source_mahalle = df_mahalle_filtered.copy()

        # --- PIVOT TABLO HAZIRLA ---
        if not df_piv_source_mahalle.empty and pd.api.types.is_numeric_dtype(df_piv_source_mahalle["NÜFUS"]):
            pivot_df_mahalle = (
                df_piv_source_mahalle[["İLÇE", "MAHALLE", "NÜFUS", "KONUMA GİT"]]
                .groupby(["İLÇE", "MAHALLE", "KONUMA GİT"], as_index=False)
                .sum()
            )
        else:
            # Boş bir şablon oluştur
            pivot_df_mahalle = pd.DataFrame(columns=["İLÇE", "MAHALLE", "NÜFUS", "KONUMA GİT"])

        # YIL sütununu ekle (Genel Toplam’da boş bırakacağız)
        pivot_df_mahalle["YIL"] = secili_yil_mahalle

        # Genel Toplam satırı
        totals = pivot_df_mahalle.select_dtypes(include="number").sum().to_dict()
        totals["İLÇE"]      = "Genel Toplam"
        totals["MAHALLE"]   = ""
        totals["YIL"]       = ""      # boş bırak
        totals["KONUMA GİT"] = ""
        pivot_df_mahalle = pd.concat([pivot_df_mahalle, pd.DataFrame([totals])], ignore_index=True)

        # Sütun sırasını kesinleştir
        cols = ["İLÇE", "MAHALLE", "YIL", "NÜFUS", "KONUMA GİT"]
        pivot_df_mahalle = pivot_df_mahalle.reindex(columns=cols)

        # --- EXCEL’E YAZ VE 'KONUMA GİT' LİNKİNİ "Git" METNİYLE EKLE ---
        with pd.ExcelWriter(pivot_output, engine="xlsxwriter") as writer:
            sheet_name = "Pivot Mahalle"
            pivot_df_mahalle.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]

            link_col_idx = pivot_df_mahalle.columns.get_loc("KONUMA GİT")
            for row_idx, url in enumerate(pivot_df_mahalle["KONUMA GİT"], start=1):
                if isinstance(url, str) and url.startswith("http"):
                    ws.write_url(row_idx, link_col_idx, url, string="Git")

        # --- BUTONU HER ZAMAN GÖSTER ---
        st.download_button(
            "Pivot Tabloyu İndir",
            data=pivot_output.getvalue(),
            file_name=f"mahalle_pivot_{secili_yil_mahalle}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            type="primary",
            key="pivot_mahalle_download"
        )

        # Eğer sadece başlık + Genel Toplam kaldıysa bilgi ver
        if pivot_df_mahalle.shape[0] <= 1:
            st.info("Seçiminize uygun mahalle verisi bulunamadığından pivot tablo yalnızca başlıkları içeriyor.")


# -------------------------------
# 4. MAHALLELERİN YILLIK NÜFUS GRAFİĞİ
# -------------------------------


# ► İlçe seçimi ve ilçe dataframe’i oluşturma# ► İlçe ve mahalle seçimi + grafikler

# 1) İlçe seçimi
ilceler = sorted(df_filtered["İLÇE"].unique().tolist())
default_idx = ilceler.index("Altınordu") if "Altınordu" in ilceler else 0
secili_ilce = st.selectbox(
    "🔽 İlçe Seçin",
    ilceler,
    index=default_idx,
    key="secili_ilce"
)
# İlçeye ait tüm mahalle-veri
ilce_df = df_filtered[df_filtered["İLÇE"] == secili_ilce]

# 2) Tüm mahallelerin grafiği (1-2-1 sütun düzeni)
outer1, outer2, outer3 = st.columns([1, 2, 1])
with outer2:
    st.subheader(f"🏘️ {secili_ilce.upper()} İlçesi Mahallelerinin Yıllık Nüfus Grafiği")
    st.plotly_chart(
        px.line(
            ilce_df,
            x="YIL",
            y="NÜFUS (KİŞİ SAYISI)",
            color="MAHALLE",
            markers=True
        ),
        key="chart_all_mahalle"
    )

# 3) Mahalle seçimi arayüzü
# Session state’i hazırla
if "secili_mahalleler" not in st.session_state:
    st.session_state.secili_mahalleler = []

# … önceki bloklar …

# ► İlçe seçiminden hemen sonra …
# secili_ilce tanımlı olduğuna emin olun

# ► Mahalle seçim ve grafik bloğu
outer1, outer2, outer3 = st.columns([1, 2, 1])
with outer2:

    st.markdown("🔽 Aşağıdan bir veya birden fazla mahalle seçin. Grafikler ve indirme dosyaları seçiminize göre güncellenir.")

    # Butonlar
    btn1_col, btn2_col = st.columns([1, 1], gap="small")
    with btn1_col:
        if st.button("Tümünü Seç", type="primary", use_container_width=False, key="btn_select_all"):
            st.session_state.secili_mahalleler = sorted(ilce_df["MAHALLE"].unique().tolist())
    with btn2_col:
        if st.button("❌ Temizle", type="secondary", use_container_width=False, key="btn_clear_selection"):
            st.session_state.secili_mahalleler = []

    # Çoklu seçim kutusu
    secili_mahalleler = st.multiselect(
       "Mahalle Seçin",
       options=sorted(ilce_df["MAHALLE"].unique().tolist()),
       key="secili_mahalleler",
       label_visibility="collapsed",
       placeholder="Bir veya birden fazla mahalle seçin"
    )

    st.info(f"🟢 Seçili mahalle sayısı: {len(secili_mahalleler)}")

    # Seçilen mahallelerin grafiği + indirme
    if secili_mahalleler:
        df_sel = ilce_df[ilce_df["MAHALLE"].isin(secili_mahalleler)]

        st.subheader(f"📊 Seçilen Mahallelerin Yıllık Nüfus Grafiği")
        st.plotly_chart(
            px.line(
                df_sel,
                x="YIL",
                y="NÜFUS (KİŞİ SAYISI)",
                color="MAHALLE",
                markers=True
            ),
            key="chart_selected_mahalle"
        )

        # Ham veri indir
        ham_out = BytesIO()
        df_sel.to_excel(ham_out, index=False)
        st.download_button(
            "Ham Veri İndir",
            type="secondary",
            data=ham_out.getvalue(),
            file_name=f"{secili_ilce}_mahalle_ham.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Pivot tablo indir
        pivot = df_sel.pivot_table(
            index="MAHALLE",
            columns="YIL",
            values="NÜFUS (KİŞİ SAYISI)",
            aggfunc="sum"
        )
        pivot.loc["TOPLAM"] = pivot.sum(numeric_only=True)
        pivot.reset_index(inplace=True)
        piv_out = BytesIO()
        pivot.to_excel(piv_out, index=False)
        st.download_button(
            "Pivot Tablo İndir",
            type="primary",
            data=piv_out.getvalue(),
            file_name=f"{secili_ilce}_mahalle_pivot.xlsx",
            mime="application/vnd.openxmlformats-officedocument-spreadsheetml.sheet"
        )




# -------------------------------
# 5. DEMOGRAFİ HARİTASI (% Yaş Dağılımı) (Pydeck)
# -------------------------------

st.markdown("### 👥 Demografi Haritası (% Yaş Dağılımı)") # Bölüm başlığı

# --- Demografi Haritası Kodları (Pydeck) ---

# use_container_width=True kaldırıldı
selected_label_map = st.selectbox("Yaş Grubu Seçiniz", pct_labels, key="demography_pct_map")
selected_pct_original = label_to_col.get(selected_label_map)
if not selected_pct_original:
    st.error("Lütfen listeden bir yaş grubu seçin.")

# 1) Cache’lenmiş demografi verisi alın (Tek yüzde sütunu yüklüyor)
if selected_pct_original: # Sütun seçildiyse veriyi yükle
     df_demo = load_parquet_demo("koordinatlı_nufus_verisi.parquet", selected_pct_original)
else:
     df_demo = pd.DataFrame() # Sütun seçilmediyse boş DataFrame


# 2) Filtre UI
st.session_state.setdefault("dem_filter", False)
st.session_state.setdefault("dem_range", "")

def fmt_demo():
    txt = st.session_state.dem_range.strip()
    parts = re.split(r"[-–—]", txt)
    if len(parts) == 2:
        try:
            lo, hi = [float(p) for p in parts]
            def fmt_num(x):
                return str(int(x)) if x == int(x) else str(x)
            st.session_state.dem_range = f"{fmt_num(lo)}-{fmt_num(hi)}"
        except:
            pass

def clear_demo():
    st.session_state.dem_range = ""
    st.session_state.dem_filter = False

st.text_input("Yüzde Aralığı Seç (örn: 5-7)", key="dem_range", on_change=fmt_demo)
c1,c2,_ = st.columns([1,1,8])
with c1: btn = st.button("Gir", type="primary", key="demogr_gir", use_container_width=True)
with c2: st.button("Temizle", type="secondary", key="demogr_temizle", on_click=clear_demo, use_container_width=True)

df_demo_filtered = df_demo.copy()

if btn:
    st.session_state.dem_filter = True

# 3) Filtre aktiveyse uygula
if st.session_state.dem_filter and st.session_state.dem_range and not df_demo_filtered.empty:
    try:
        lo, hi = map(float, st.session_state.dem_range.split("-"))
        df_demo_filtered = df_demo_filtered[df_demo_filtered["PCT"].between(lo, hi)].copy()
        st.markdown(f"**Seçilen Yüzde:** {lo:g} – {hi:g}")
        cnt_i = df_demo_filtered["İLÇE"].nunique()
        cnt_m = df_demo_filtered.shape[0]
        st.info(f"Kriterlere uygun {cnt_i} ilçede {cnt_m} mahalle bulundu")
    except ValueError:
        st.error("Geçersiz yüzde aralığı formatı. Lütfen 'örn: 5-7' gibi girin.")
elif st.session_state.dem_filter and st.session_state.dem_range and df_demo_filtered.empty:
     st.warning("Filtre uygulanacak veri bulunamadı.")

# 4) Yüzde formatlama (tutarlı iki ondalık için .map)
if not df_demo_filtered.empty: # <-- Demografi Haritası ve Excel İndirme Kodları bu blok içine taşındı
     df_demo_filtered["pct_numeric"]  = df_demo_filtered["PCT"]
     df_demo_filtered["Yüzde Aralığı"] = df_demo_filtered["pct_numeric"].map("{:.2f} %".format)

     # Demografi için kategorilere ayırma ve renk atama
     # bins_d listesi 8 elemanlı olmalı (7 aralık için 8 sınır)
     bins_d = [-float("inf"), 5, 10, 15, 20, 25, 30, 35, float("inf")] # Bir sınır daha eklendi
     # colors_d listesi 8 renkli olmalı (8 aralık için 8 renk)
     colors_d = [
         [166,86,40,180], # Koyu Kahve
         [152,78,163,180], # Mor
         [77,175,74,180],  # Yeşil
         [55,126,184,180], # Mavi
         [247,129,191,180],# Pembe (Yeni renk)
         [255,127,0,180],  # Turuncu
         [255,255,51,180], # Sarı
         [228,26,28,180],  # Kırmızı (En yüksek aralık)
     ] # Renk sayısı bins sayısı - 1 kadar olmalı
     df_demo_filtered["cat"]  = pd.cut(df_demo_filtered["PCT"], bins=bins_d, labels=False, right=True)
     df_demo_filtered["color"] = df_demo_filtered["cat"].map(dict(enumerate(colors_d)))
     df_demo_filtered.drop(columns=["cat"], inplace=True)

     # 5) Kümeli ScatterplotLayer (Demografi)
     clustered_demo_layer = pdk.Layer(
         "ScatterplotLayer",
         data=df_demo_filtered, # Filtrelenmiş veriyi kullan
         get_position="[lon, lat]",
         get_fill_color="color",
         get_radius=150,
         pickable=True, # Etkileşim için pickable True
         cluster=True,
         cluster_radius=50,
     )

     # 6) Sınır katmanı (lookup ile)
     show_map_borders = st.checkbox("Mahalle Sınırlarını Göster", value=True, key="show_demo_borders_map")

     layers = [clustered_demo_layer] # Başlangıçta sadece nokta katmanı

     if show_map_borders:
         allowed = set(df_demo_filtered["MAHALLE KODU (AKS)"].astype(int))
         features = [mahalle_lookup[k] for k in allowed if k in mahalle_lookup]
         demo_geo = {"type":"FeatureCollection","features":features}

         # Filtre aktifse filtrelenmiş geojson kullan, değilse tam geojson kullan
         geo_to_use = demo_geo if st.session_state.dem_filter else mahalle_geojson
         # GeoJSON geçerli ve feature içeriyor mu kontrolü
         if geo_to_use and geo_to_use.get('features'):
              border_layer = pdk.Layer(
                  "GeoJsonLayer",
                  geo_to_use,
                  stroked=True,
                  filled=False,
                  get_line_color=[3,32,252,180],
                  line_width_min_pixels=1
              )
              layers.append(border_layer) # Sınır katmanını ekle
         elif st.session_state.dem_filter and not features:
              st.warning("Filtre kriterlerinize uyan mahalle bulunamadığı için mahalle sınırları gösterilemiyor.")
         elif not mahalle_geojson:
              st.warning("Mahalle sınırları GeoJSON verisi yüklenemedi.")


     # 9) Harita çizimi
     st.pydeck_chart(pdk.Deck(
          map_style  = "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
          initial_view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=8, pitch=40),
          layers     = layers,
          tooltip={
            "html": (
              "<b>{MAHALLE}</b><br/>"
              "İlçe: {İLÇE}<br/>"
              f"{selected_label_map} Yüzde: "+"{Yüzde Aralığı}" # Araç ipucunda yüzdeyi göster
            )
          }
     ))

     # --- İndirme Butonları (Demografi Haritası - Pydeck - Filtresine Ait) ---
     st.markdown("---") # Ayırıcı çizgi

     col_ham_dem, col_piv_dem, _ = st.columns([1,1,8])

     # 1) Ham veri indir (Demografi haritası filtresine göre)
     with col_ham_dem:
         out_ham = BytesIO()
         # df_demo_filtered zaten yukarıdaki if bloğu içinde tanımlı ve boş değilse burası çalışır.
         df_ham = df_demo_filtered[["İLÇE","MAHALLE","pct_numeric"]].copy()
         df_ham.rename(columns={"pct_numeric": "Yüzde Aralığı"}, inplace=True)
         df_ham["Yüzde Aralığı"] = df_ham["Yüzde Aralığı"].round(2)

         if not df_ham.empty: # df_ham boş değilse yazma işlemini yap
             with pd.ExcelWriter(out_ham, engine="xlsxwriter") as writer:
                 sheet = "Ham Demografi Verisi"
                 wb = writer.book
                 ws = wb.add_worksheet(sheet)
                 writer.sheets[sheet] = ws
                 # 1. satıra kullanıcı girdilerini yaz
                 ws.write(0, 0, "Seçili Yaş Grubu:")
                 ws.write(0, 1, selected_label_map) # selected_label yerine selected_label_map kullanıldı
                 ws.write(0, 2, "Filtre Aralığı (%):")
                 ws.write(0, 3, st.session_state.dem_range or "—")
                 # 3. satırdan itibaren gerçek veri
                 df_ham.to_excel(writer, sheet_name=sheet, index=False, startrow=1)

             st.download_button(
                 "Ham Veriyi İndir",
                 data=out_ham.getvalue(),
                 file_name=f"demografi_ham_{selected_label_map}.xlsx", # Dosya adı selected_label_map kullanılarak güncellendi
                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                 use_container_width=True,
                 type="secondary"
             )
         else:
              st.warning("İndirilecek ham demografi verisi bulunamadı (Filtrelemeye uygun veri yok).")


     # 2) Pivot veri indir (Demografi haritası filtresine göre)
     with col_piv_dem: # Değişken adı güncellendi
         out_piv = BytesIO()

         # Pivot: ilçede mahallelerin ortalama yüzdesini sayısal tut
         if not df_demo_filtered.empty and "pct_numeric" in df_demo_filtered.columns and "İLÇE" in df_demo_filtered.columns:
             piv_dem = (
                 df_demo_filtered
                 .groupby("İLÇE", as_index=False)
                 .agg({"pct_numeric": "mean"})
                 .rename(columns={"pct_numeric": selected_label_map + " Yüzde Ortalama"}) # Sütun adı güncellendi
             )
             if not piv_dem.empty:
                 # Sadece yüzde sütununu yuvarla (artık sadece bir tane var)
                 yuzde_col_name = selected_label_map + " Yüzde Ortalama"
                 if yuzde_col_name in piv_dem.columns:
                      piv_dem[yuzde_col_name] = piv_dem[yuzde_col_name].round(2)
                      # Genel toplamı hesapla (sadece ortalama sütunu için)
                      numeric_cols_in_piv = piv_dem.select_dtypes(include=np.number).columns.tolist()
                      if numeric_cols_in_piv:
                          toplam_row_data = {"İLÇE":"Genel Ortalama"}
                          for num_col in numeric_cols_in_piv:
                              toplam_row_data[num_col] = piv_dem[num_col].mean().round(2)
                          piv_dem = pd.concat([piv_dem, pd.DataFrame([toplam_row_data])], ignore_index=True)
                      else:
                          st.warning("Pivot tablo genel ortalaması için sayısal sütun bulunamadı.")
             else:
                  st.warning("Pivot tablo oluşturmak için uygun demografi verisi bulunamadı.")

         else: # df_demo_filtered boş veya gerekli sütunlar yok
             st.warning("Pivot tablo oluşturmak için uygun demografi verisi bulunamadı.")
             piv_dem = pd.DataFrame() # Eğer veri yoksa boş DataFrame tanımla


         # piv_dem artık bu noktada her zaman tanımlı (veri içerse de içermese de)
         if not piv_dem.empty:
             with pd.ExcelWriter(out_piv, engine="xlsxwriter") as writer:
                 sheet = "Pivot Demografi"
                 wb = writer.book
                 ws = wb.add_worksheet(sheet)
                 writer.sheets[sheet] = ws
                 ws.write(0, 0, "Harita Rengi Yaş Grubu:")
                 ws.write(0, 1, selected_label_map)
                 ws.write(0, 2, "Filtre Aralığı (%):")
                 ws.write(0, 3, st.session_state.dem_range or "—")
                 piv_dem.to_excel(writer, sheet_name=sheet, index=False, startrow=1)

             st.download_button(
                 "Pivot Tabloyu İndir",
                 data=out_piv.getvalue(),
                 file_name=f"demografi_pivot_{selected_label_map}.xlsx", # Dosya adı selected_label_map kullanılarak güncellendi
                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                 use_container_width=True,
                 type="primary"
             )
         else:
              st.warning("İndirilecek pivot demografi verisi bulunamadı.")

else: # df_demo_filtered is empty (from initial load or filter result)
     st.warning("Harita için veri bulunamadı (Demografi).")


# Uyarı: Karşılaştırmak istediğiniz mahalleleri seçin
st.markdown("---")  # Ayırıcı çizgi
st.markdown("### 📊 Seçilen Mahallelerin Yaş Dağılım Grafikleri")
# Bilgilendirme metnini öne çıkarmak için renk ve kalın yazı stili:
st.markdown("*Karşılaştırma Yapmak İstediğiniz Mahalleleri Seçin.*")

# -----------------------------
# 1) Sabit Tanımlar ve Veri Yükleme
# -----------------------------
age_group_order = ["0-5", "6-13", "14-17", "18-34", "35-64", "65+"]

demo_df = get_demo_data()

# Yaş yüzde kolon eşleştirme
pct_columns = [c for c in demo_df.columns if c.endswith(" YAŞ YÜZDE")]
pct_labels  = [c.replace(" YAŞ YÜZDE", "") for c in pct_columns]
label_to_col = dict(zip(pct_labels, pct_columns))
all_ilces_list = sorted(demo_df["İLÇE"].unique())

@st.cache_data
def build_chart(mahalle: str, show_suffix: bool = True) -> alt.Chart:
    df = pd.DataFrame([
        {"Yaş Grubu": lbl,
         "Yüzde": float(row_val := (demo_df.loc[demo_df["MAHALLE"] == mahalle, label_to_col[lbl]].iloc[0] if lbl in label_to_col else 0) or 0),
         "PctFmt": f"%{row_val:.0f}"}
        for lbl in age_group_order
    ])
    ilce = demo_df.loc[demo_df["MAHALLE"] == mahalle, "İLÇE"].iloc[0]
    # Eksende gösterilecek etiket biçimini ayarlıyoruz: suffix gösterimi
    label_expr = "datum.value + ' Yaş'" if show_suffix else "datum.value"
    bar = alt.Chart(df).mark_bar().encode(
        x=alt.X(
            "Yaş Grubu:N",
            sort=age_group_order,
            axis=alt.Axis(labelAngle=0, labelExpr=label_expr),
            title=None
        ),
        y=alt.Y("Yüzde:Q", title="Yüzde (%)"),
        color=alt.Color("Yaş Grubu:N", legend=None),
        tooltip=[alt.Tooltip("PctFmt:N", title="Yüzde")]
    ).properties(title=f"Yaş Dağılımı - {mahalle} ({ilce})", height=300)
    text_layer = alt.Chart(df).mark_text(
        align='center',
        baseline='bottom',
        dy=-10,
        size=18
    ).encode(
        x=alt.X('Yaş Grubu:N', sort=age_group_order),
        y='Yüzde:Q',
        text='PctFmt:N',
        color=alt.value('white')
    )
    return bar + text_layer

# -----------------------------
# 2) Seçim Alanları
# -----------------------------

col1, col2 = st.columns([2, 2])
with col1:
    selected_ilces = st.multiselect(
        "İlçeler:",
        all_ilces_list,
        placeholder="Lütfen bir ya da daha fazla ilçe seçin"
    )
with col2:
    available_mahalles = sorted(
        demo_df[demo_df["İLÇE"].isin(selected_ilces)]["MAHALLE"].unique()
    ) if selected_ilces else []
    selected_mahalles = st.multiselect(
        "Mahalleler:",
        available_mahalles,
        placeholder="Lütfen bir ya da daha fazla mahalle seçin"
    )

# -----------------------------
# Bu blok, `last_ilce_list` ve `last_mahalle_list` session state’lerini karşılaştırarak
# seçimler değiştiğinde `show_charts` bayrağını sıfırlıyor. Böylece kullanıcı seçimleri
# güncellediğinde grafikler gizlenecek ve buton tekrar görünür olacak.
# -----------------------------
if 'last_ilce_list' not in st.session_state:
    st.session_state.last_ilce_list = []
if 'last_mahalle_list' not in st.session_state:
    st.session_state.last_mahalle_list = []
# Seçimler değiştiyse grafik gösterimini kapat
if selected_ilces != st.session_state.last_ilce_list or selected_mahalles != st.session_state.last_mahalle_list:
    st.session_state.show_charts = False
# Son seçimleri sakla
st.session_state.last_ilce_list = selected_ilces.copy() if isinstance(selected_ilces, list) else []

# -----------------------------
# 3) Form: Dinamik Buton ile Grafik Göster/Güncelleme
# -----------------------------
if not st.session_state.get('show_charts', False):
    first_time = not st.session_state.last_mahalle_list
    button_label = "Grafiği Göster" if first_time else "Grafiği Güncelle"
    with st.form("grafik_form"):
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([2, 0.5, 2, 0.5])
        with btn_col4:
            show = st.form_submit_button(
                button_label,
                type="primary",
                use_container_width=True
            )
    if show:
        st.session_state.show_charts = True
        st.session_state.last_mahalle_list = selected_mahalles.copy()
    else:
        st.info("Grafiğini görmek istediğiniz ilçe ve mahalleleri seçtikten sonra 'Grafiği Göster' veya 'Grafiği Güncelle' butonuna basınız.")
        st.stop()

# -----------------------------
# 4) Grafik Gösterim
# -----------------------------
if st.session_state.get('show_charts', False):
    if not selected_ilces:
        st.warning("Lütfen önce bir ilçe seçin.")
    elif not selected_mahalles:
        st.warning("Lütfen en az bir mahalle seçin.")
    else:
        for i in range(0, len(selected_mahalles), 5):
            row_mahalle = selected_mahalles[i : i + 5]
            cols = st.columns(len(row_mahalle))
            hide_suffix = len(row_mahalle) > 4
            for idx, m in enumerate(row_mahalle):
                with cols[idx]:
                    chart = build_chart(m, show_suffix=not hide_suffix)
                    st.altair_chart(chart, use_container_width=True)
else:
    # Grafiklar gizliyken butonun görünmesi için placeholder boş bırak
    pass
