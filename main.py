import pandas as pd
import pydeck as pdk
import streamlit as st
import json
import re
from io import BytesIO
import altair as alt
import numpy as np
# import plotly.express as px # Plotly bu düzenlemede kullanılmıyor
# import plotly.graph_objects as go # Plotly bu düzenlemede kullanılmıyor


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
        "Latitude",
        "Longitude"
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
            pct_col: "PCT", # Rename the percentage column to 'PCT' for consistency
            "Latitude": "lat",
            "Longitude": "lon",
        },
        errors="ignore"
    )
    return df

# load_all_age_demographics fonksiyonu, dropdown grafik için tüm yaş yüzde sütunlarını yükler
@st.cache_data
def load_all_age_demographics() -> pd.DataFrame:
    """Loads demographic data including all age percentage columns for the dropdown graph."""
    # pct_columns değişkeninin global kapsamda tanımlı olduğundan emin olun.
    global pct_columns # Kullanıcının sağladığı snippet'te pct_columns kullanılıyor
    # global all_pct_columns # Önceki versiyonlarla tutarlılık için tutalım

    # Kullanıcının son snippet'i pct_columns kullandığı için onu tercih edelim.
    # all_pct_columns da tanımlı kalsın, belki başka yerde kullanılıyordur.
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
# Define global variables and load initial data here
# -----------------------------

st.set_page_config(page_title="Ordu Nüfus Haritası", layout="wide")
# Pydeck tooltip rengini pembe yapmak için style (kullanıcının isteği üzerine)
st.markdown("""<style>
   .deck-tooltip { background-color: magenta!important; color: white!important;
               border-radius: 4px; padding: 4px; }
</style>""", unsafe_allow_html=True)
st.markdown("## 📊 Ordu İli Nüfus Haritası (2007 - 2024)")

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

# Identify all age percentage columns based on the column names
# Kullanıcının sağladığı kodda pct_columns ve all_pct_columns kullanılıyor.
# Her ikisini de globalde tanımlayalım ve pct_columns'ı hem demografi haritası hem de dropdown grafik için kullanalım.
# all_pct_columns artık load_parquet_demo fonksiyonunda kullanılmıyor, sadece pct_columns kullanılıyor.
# load_all_age_demographics de pct_columns kullanıyor.
# all_pct_columns'ı kaldırabiliriz veya pct_columns ile aynı yapabiliriz. Aynı yapalım şimdilik.
pct_columns = [c for c in df_full.columns if c.endswith(" YAŞ YÜZDE")]
pct_labels  = [c.replace(" YAŞ YÜZDE","") for c in pct_columns]
label_to_col = dict(zip(pct_labels, pct_columns)) # Original label to original column name mapping

all_pct_columns = pct_columns # all_pct_columns artık load_parquet_demo'da kullanılmasa da, önceki versiyonlarla tutarlılık için tanımlı kalsın.


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
# 2. İLÇE BAZLI NÜFUS Harita ve Filtre (Pydeck)
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
    df_ilce["NÜFUS_FMT"] = (df_ilce["NÜFUS"].astype(int).map("{:,.0f}".format).str.replace(",", "."))

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
             st.warning("Filtre kriterlerinize uyan ilçe bulunamadığı için ilçe sınırları gösterilemiyor.")
        elif not ilce_geojson:
             st.warning("İlçe sınırları GeoJSON verisi yüklenemedi.")

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

        # Define the desired order of columns for the Ham export
        desired_ham_order = ["İLÇE", "YIL", "NÜFUS"] # <-- Sütun sırası değiştirildi

        cols_to_export = desired_ham_order # Use the desired order list directly

        # Ensure these columns actually exist in the dataframe before selecting
        cols_to_export_present = [col for col in desired_ham_order if col in df_export.columns]
        # Handle case where not all desired columns are present (unlikely here but good practice)
        if len(cols_to_export_present) != len(desired_ham_order):
            st.warning(f"Expected columns {desired_ham_order} not all present in İlçe ham data. Exporting available columns.")

        # Ham veri exportundan 'lat' ve 'lon' sütunları kaldırıldı (Önceki istek)
        cols_to_export_present = [col for col in cols_to_export_present if col not in ['lat', 'lon']] # <-- 'lat', 'lon' eklendiği satır buraya taşındı ve düzenlendi

        df_export[cols_to_export_present].to_excel(out_ilce, index=False, sheet_name="Ham İlçe Verisi")


        st.download_button(
            "Ham Veriyi İndir",
            data=out_ilce.getvalue(),
            file_name=f"ilce_ham_{secili_yil_ilce}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            type="secondary"
        )

with eb:
        outp_ilce = BytesIO()
        df_piv_source = load_parquet_ilce("koordinatlı_nufus_verisi.parquet", secili_yil_ilce).copy()
        # Filtre uygulanmışsa pivot tablo kaynağını da filtrele
        if st.session_state.ilce_filter and st.session_state.ilce_range:
             try:
                lo, hi = map(int, st.session_state.ilce_range.split("-"))
                df_piv_source = df_piv_source[df_piv_source["NÜFUS"].between(lo, hi)]
             except ValueError:
                pass

        if not df_piv_source.empty:
            piv = (
                df_piv_source[["İLÇE", "NÜFUS"]]
                .groupby("İLÇE")
                .sum()
                .reset_index()
                .assign(YIL=secili_yil_ilce)
            )

            pivot_cols_order = ["İLÇE", "YIL", "NÜFUS"] # <-- Sütun sırası değiştirildi
            # Ensure all desired columns exist in piv before reordering
            pivot_cols_present = [col for col in pivot_cols_order if col in piv.columns]
            # Sadece mevcut sütunlarla yeniden indexleme yaparak sırayı uygula
            piv = piv[pivot_cols_present]

            # Genel Toplam satırı için veriyi hazırla
            totals_numeric = piv.select_dtypes(include=np.number).sum().to_dict()
            toplam_row_data = {"İLÇE": "Genel Toplam"}
            toplam_row_data.update(totals_numeric) # Sayısal toplamları ekle

            piv = pd.concat([piv, pd.DataFrame([toplam_row_data])], ignore_index=True) # Toplam satırını ekle

            # Excel'e yazma kısmı
            with pd.ExcelWriter(outp_ilce, engine="xlsxwriter") as writer:
                 sheet = "Pivot İlçe"
                 wb = writer.book
                 ws = wb.add_worksheet(sheet)
                 writer.sheets[sheet] = ws

                 piv.to_excel(writer, sheet_name=sheet, index=False, startrow=0) # <-- startrow 0 olarak değiştirildi

            st.download_button(
                "Pivot Tabloyu İndir",
                data=outp_ilce.getvalue(),
                file_name=f"ilce_pivot_{secili_yil_ilce}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                type="primary"
            )
        else:
             st.warning("Pivot tablo oluşturmak için uygun nüfus verisi bulunamadı.")


# -------------------------------
# 2. MAHALLE BAZLI NÜFUS (Pydeck) - Kullanıcının sağladığı ve aktif olduğunu belirttiği bölüm
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
        show = st.checkbox("Mahalle Sınırlarını Gö göster", value=True, key="show_mahalle_borders_only") # Yazım hatası düzeltildi

        allowed = set(df_mahalle_filtered["MAHALLE KODU (AKS)"].astype(int))

        filtered_mahalle_features = [
            mahalle_lookup[code]
            for code in allowed
            if code in mahalle_lookup
        ]
        # Filtre aktifse filtrelenmiş geojson kullan, değilse tam geojson kullan
        geo = {"type":"FeatureCollection","features": filtered_mahalle_features} if show and st.session_state.filter_active else mahalle_geojson

        # GeoJSON geçerli ve feature içeriyor mu kontrolü
        if geo and geo.get('features'):
             border = pdk.Layer("GeoJsonLayer", geo, stroked=True, filled=False,
                                 get_line_color=[3,32,252,180], line_width_min_pixels=1)
        elif st.session_state.filter_active and not filtered_mahalle_features:
             st.warning("Filtre kriterlerinize uyan mahalle bulunamadığı için mahalle sınırları gösterilemiyor.")
             border = None
        elif not mahalle_geojson:
             st.warning("Mahalle sınırları GeoJSON verisi yüklenemedi.")
             border = None
        else:
             border = None # Diğer durumlar için border yok

        # 10) Harita çizimi
        layers_mahalle = [clustered_mahalle_layer]
        if border: # Border katmanı None değilse ekle
            layers_mahalle.append(border)

        st.pydeck_chart(pdk.Deck(
            map_style="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
            initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=8, pitch=40),
            layers=layers_mahalle,
            tooltip={"html": "<b>{MAHALLE}</b><br/>İlçe: {İLÇE}<br/>Nüfus ({YIL}): {NÜFUS_FMT}".replace("{YIL}", str(secili_yil_mahalle))}
        ))

        # Excel indirme butonları
        col_excel1, col_excel2, _ = st.columns([1, 1, 8])

        # Ham veri indir
        with col_excel1:
            output = BytesIO()
            # Ham veri için filtrelenmiş veriyi kullan
            df_export_mahalle_ham = df_mahalle_filtered.copy()
            df_export_mahalle_ham["YIL"] = secili_yil_mahalle
            df_export_mahalle_ham["KONUMA GİT"] = df_export_mahalle_ham.apply(
                lambda row: f"https://www.google.com/maps?q={row['lat']},{row['lon']}&z=13&hl=tr",
                axis=1
            )
            cols_to_export = ["İLÇE", "MAHALLE", "YIL", "NÜFUS", "KONUMA GİT"]
            if all(col in df_export_mahalle_ham.columns for col in cols_to_export):
                 df_export_mahalle_ham = df_export_mahalle_ham[cols_to_export]
            else:
                 st.warning("Ham mahalle verisi için gerekli sütunlar bulunamadı.")
                 df_export_mahalle_ham = pd.DataFrame(columns=cols_to_export)


            if not df_export_mahalle_ham.empty:
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    sheet_name = "Ham Mahalle Verisi"
                    df_export_mahalle_ham.to_excel(writer, sheet_name=sheet_name, index=False)
                    ws = writer.sheets[sheet_name]
                    if "KONUMA GİT" in df_export_mahalle_ham.columns:
                        link_col = df_export_mahalle_ham.columns.get_loc("KONUMA GİT")
                        # Link sütununda sadece geçerli URL'leri yazdır
                        for idx, url in enumerate(df_export_mahalle_ham["KONUMA GİT"], start=1):
                            if url and isinstance(url, str) and url.startswith("http"):
                                ws.write_url(idx, link_col, url, string="Git")

                st.download_button(
                    "Ham Veriyi İndir",
                    data=output.getvalue(),
                    file_name=f"mahalle_ham_veri_{secili_yil_mahalle}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    type="secondary"
                )
            else:
                st.warning("İndirilecek ham mahalle verisi bulunamadı.")

        # Pivot tablo indir
        with col_excel2:
            pivot_output = BytesIO()
            # Pivot tablo için filtrelenmiş df_mahalle verisini kullan
            df_piv_source_mahalle = df_mahalle_filtered.copy()

            if not df_piv_source_mahalle.empty:
                # Nüfus sütunu mevcut ve sayısal ise pivot oluştur
                if "NÜFUS" in df_piv_source_mahalle.columns and pd.api.types.is_numeric_dtype(df_piv_source_mahalle["NÜFUS"]):
                    # Düzeltilmiş Pivot Mantığı: Sadece İLÇE ve MAHALLE'ye göre grupla
                    pivot_df_mahalle = (
                        df_piv_source_mahalle[["İLÇE", "MAHALLE", "NÜFUS"]]
                        .groupby(["İLÇE", "MAHALLE"])
                        .sum() # Mahalle düzeyinde zaten toplam olduğu için sum etkisiz kalır ama kod standardı
                        .reset_index()
                    )

                    # YIL sütununu pivot tablo oluştuktan sonra ekle
                    pivot_df_mahalle["YIL"] = secili_yil_mahalle # secili_yil_mahalle zaten ilgili yıl stringi

                    # Genel Toplam satırı
                    totals = pivot_df_mahalle.select_dtypes(include=[int, float]).sum().to_frame().T
                    totals["İLÇE"] = "Genel Toplam"
                    totals["MAHALLE"] = ""
                    # YIL sütunu toplamda NaN olacağından, Genel Toplam satırına selected_yil_mahalle'yi atayalım
                    totals["YIL"] = secili_yil_mahalle
                    pivot_df_mahalle = pd.concat([pivot_df_mahalle, totals], ignore_index=True)

                    # Eğer ham veri export'u başarısız olursa bu kısım çalışmayabilir.
                    if 'df_export_mahalle_ham' in locals() and "KONUMA GİT" in df_export_mahalle_ham.columns:
                         coord_map = df_export_mahalle_ham.set_index(["İLÇE", "MAHALLE"])["KONUMA GİT"]
                         pivot_df_mahalle["KONUMA GİT"] = pivot_df_mahalle.apply(
                             lambda row: coord_map.get((row["İLÇE"], row["MAHALLE"]), ""),
                             axis=1
                         )
                    else:
                         pivot_df_mahalle["KONUMA GİT"] = "" # KONUMA GİT sütunu yoksa boş ekle


                else:
                     st.warning("Pivot tablo oluşturmak için uygun nüfus verisi veya sütunlar bulunamadı.")
                     pivot_df_mahalle = pd.DataFrame(columns=["İLÇE", "MAHALLE", "YIL", "KONUMA GİT"]) # Boş DataFrame oluştur

            else:
                st.warning("Pivot tablo oluşturmak için mahalle verisi bulunamadı.")
                pivot_df_mahalle = pd.DataFrame(columns=["İLÇE", "MAHALLE", "YIL", "KONUMA GİT"]) # Boş DataFrame oluştur


            if not pivot_df_mahalle.empty:
                with pd.ExcelWriter(pivot_output, engine="xlsxwriter") as writer:
                    sheet_name = "Pivot Mahalle"
                    pivot_df_mahalle.to_excel(writer, sheet_name=sheet_name, index=False)
                    ws = writer.sheets[sheet_name]

                    if "KONUMA GİT" in pivot_df_mahalle.columns:
                         git_col = pivot_df_mahalle.columns.get_loc("KONUMA GİT")
                         # Link sütununda sadece geçerli URL'leri yazdır
                         for idx, url in enumerate(pivot_df_mahalle["KONUMA GİT"], start=1):
                              if url and isinstance(url, str) and url.startswith("http"):
                                   ws.write_url(idx, git_col, url, string="Git")


                st.download_button(
                    "Pivot Tabloyu İndir",
                    data=pivot_output.getvalue(),
                    file_name=f"mahalle_pivot_{secili_yil_mahalle}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    type="primary"
                )
            else:
                 st.warning("İndirilecek pivot mahalle verisi bulunamadı.")
    else:
        st.info("Mahalle verisi yüklenemediği için bu bölüm gösterilemiyor.")



# -------------------------------
# 3. DEMOGRAFİ HARİTASI (% Yaş Dağılımı) (Pydeck)
# -------------------------------

st.markdown("### 👥 Demografi Haritası (% Yaş Dağılımı)") # Bölüm başlığı

# --- Demografi Haritası Kodları (Pydeck) ---

# use_container_width=True kaldırıldı
selected_label_map = st.selectbox("Harita Rengi için Yaş Grubu Yüzdesi Seçiniz", pct_labels, key="demography_pct_map")
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


# --- Yaş Dağılım Grafikleri (Çoklu Mahalle Seçimi ve Yan Yana Gösterim) ---
st.markdown("---") # Ayırıcı çizgi
st.markdown("### 📊 Seçilen Mahallelerin Yaş Dağılım Grafikleri")

# Yaş aralıklarını küçükten büyüğe sıralamak için etiketlerin istediğimiz sırasını belirleyelim (Grafikler için)
age_group_order = ["0-5", "6-13", "14-17", "18-34", "35-64", "65+"]

# Bu nedenle sadece bir değer alacak şekilde çağrıyı düzeltelim
demo_df_for_dropdown_raw = load_all_age_demographics() # Çağrı düzeltildi, tek değer bekleniyor

# Koşulda, global olarak tanımlı olan pct_columns değişkenini kullanalım
if not demo_df_for_dropdown_raw.empty and 'pct_columns' in globals() and pct_columns:

    # Get unique list of İlçes for the first selectbox
    all_ilces_list = sorted(demo_df_for_dropdown_raw["İLÇE"].unique().tolist())

    # İlçe ve Mahalle seçim alanları için aynı satırda iki sütun oluştur
    ilce_select_col, mahalle_select_col = st.columns([1, 1]) # Sütun genişliklerini ayarlayabilirsiniz

    with ilce_select_col:
        # Çoklu İlçe Seçimi
        selected_ilces_graph = st.multiselect(
            "Grafik için İlçe Seçin:",
            all_ilces_list,
            key="graph_ilce_multiselect",
            placeholder="Lütfen İlçe Seçiniz",
        )

    # Seçilen İlçe(ler)e göre mahalle listesini filtrele
    filtered_mahalles_list = []
    if selected_ilces_graph:
        # Sadece seçili ilçelerdeki mahalleleri al
        mahalles_in_selected_ilces = demo_df_for_dropdown_raw[
            demo_df_for_dropdown_raw["İLÇE"].isin(selected_ilces_graph)
        ]["MAHALLE"].unique().tolist()
        filtered_mahalles_list = sorted(mahalles_in_selected_ilces)

    with mahalle_select_col:
        # Çoklu Mahalle Seçimi (Seçili ilçelere göre filtrelenmiş)
        selected_mahalles_graph = st.multiselect(
            "Grafik için Mahalle Seçin:",
            filtered_mahalles_list, # Filtrelenmiş listeyi kullan
            key="graph_mahalle_multiselect",
            placeholder="Lütfen Mahalle Seçiniz",
        )

    charts_to_display = [] # Altair grafik nesnelerini tutacak liste

    # Kullanıcı hem ilçe hem de mahalle seçtiyse devam et
    if selected_ilces_graph and selected_mahalles_graph:
        # Ham veriyi seçilen ilçe ve mahallelere göre filtrele
        data_for_selected_mahalles = demo_df_for_dropdown_raw[
             (demo_df_for_dropdown_raw["İLÇE"].isin(selected_ilces_graph)) &
             (demo_df_for_dropdown_raw["MAHALLE"].isin(selected_mahalles_graph))
        ].copy() # Kopya üzerinde çalış

        # Filtrelenmiş veride mahalleler varsa grafik oluşturma döngüsüne gir
        if not data_for_selected_mahalles.empty:
            # Filtrelenmiş verideki her benzersiz mahalle için grafik oluştur
            unique_selected_mahalles = data_for_selected_mahalles["MAHALLE"].unique().tolist()

            for mahalle_name in unique_selected_mahalles:
                # Şu anki mahalle için filtrele (Zaten seçili ilçeler ve mahalleler içinde)
                mahalle_data_row_dropdown = data_for_selected_mahalles[
                     (data_for_selected_mahalles["MAHALLE"] == mahalle_name)
                ].iloc[0] # Bu mahalle adına ait tek satırı al

                graph_data = []
                # Bu mahalle için grafik verilerini hazırla
                for label in age_group_order:
                    original_col = label_to_col.get(label) 
                    if original_col and original_col in mahalle_data_row_dropdown: 
                        value = mahalle_data_row_dropdown[original_col]
                        if pd.isna(value):
                            value = 0
                        graph_data.append({"Yaş Grubu": label, "Yüzde": value})
                    else:
                        graph_data.append({"Yaş Grubu": label, "Yüzde": 0}) # Veri yoksa 0 ekle

                if graph_data:
                    graph_df = pd.DataFrame(graph_data)
                    # Formatlanmış yüzde sütununu ekle
                    graph_df['Yüzde_FMT'] = graph_df['Yüzde'].apply(lambda x: f"%{x:.0f}") # Türkçe format

                    ilce_name_for_title = data_for_selected_mahalles[
                        data_for_selected_mahalles["MAHALLE"] == mahalle_name
                    ]["İLÇE"].iloc[0]

                    bar_chart = alt.Chart(graph_df).mark_bar().encode(
                        x=alt.X('Yaş Grubu:N', sort=age_group_order, title=None, axis=alt.Axis(labelAngle=0)),
                        y=alt.Y('Yüzde:Q', title="Yüzde (%)"),
                        color=alt.Color('Yaş Grubu:N', legend=None),
                        tooltip=['Yaş Grubu', alt.Tooltip('Yüzde:Q', format='.2f')]
                    ).properties(
                        title=f"Yaş Dağılımı - {mahalle_name} ({ilce_name_for_title})", # Başlığa İlçe adını ekle
                        height=300 # Birden çok grafik için yüksekliği ayarla
                    )

                    # Metin Katmanını Oluştur
                    text_layer = alt.Chart(graph_df).mark_text(
                        align='center',
                        baseline='bottom',
                        dy=-15, # Çubuğun üstünde konumlandır
                        size=18 # Yazı boyutu
                    ).encode(
                        x=alt.X('Yaş Grubu:N', sort=age_group_order, title=None),
                        y=alt.Y('Yüzde:Q'),
                        text=alt.Text('Yüzde_FMT:N'), # Formatlanmış string
                        color=alt.value('white')
                    )

                    # Katmanları Birleştir
                    chart = bar_chart + text_layer
                    charts_to_display.append(chart) # Grafiği listeye ekle

            # Grafik listesi boş değilse (yani seçilen mahalleler için veri bulunduysa)
            if charts_to_display:
                charts_per_row = 5 # Bir satırda gösterilecek max grafik sayısı
                # Grafik listesini 5'erli gruplara ayır
                for i in range(0, len(charts_to_display), charts_per_row):
                    row_charts = charts_to_display[i : i + charts_per_row]
                    # Bu satırdaki grafik sayısına göre sütun oluştur
                    cols = st.columns(len(row_charts))
                    # Bu satırdaki grafikler üzerinde döngü kur ve sütunlarda göster
                    for j in range(len(row_charts)):
                         with cols[j]:
                             st.altair_chart(row_charts[j], use_container_width=True)
        else:
             st.info("Seçilen ilçe ve mahalle kombinasyonu için veri bulunamadı.")


    elif selected_ilces_graph: # İlçe(ler) seçili ama mahalle(ler) henüz seçili değil
        st.info(f"Lütfen seçili ilçe ({', '.join(selected_ilces_graph)}) içinden grafik çizmek için bir veya daha fazla mahalle seçin.")
    elif all_ilces_list: # İlçe(ler) henüz seçili değil
        st.info("Lütfen grafik çizmek için bir veya daha fazla ilçe seçin.")
    # else: all_ilces_list boşsa, dış if tarafından zaten uyarı verilir.

else:
    st.warning("Grafikler için mahalle demografi verileri yüklenemedi.")
