import pandas as pd
import pydeck as pdk
import streamlit as st
import json
import re
from io import BytesIO

# -----------------------------
# 0) ÖNBELLEKLENMİŞ FONKSİYONLAR
# -----------------------------

@st.cache_data
def load_parquet_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)  # tüm sütunlar
    df.rename(columns={"Latitude":"lat","Longitude":"lon"}, inplace=True, errors="ignore")
    return df


@st.cache_data
def load_parquet_ilce(path: str, year: str) -> pd.DataFrame:
    # Parquet'ten bu sütunları oku
    cols = ["İLÇE", f"{year} YILI NÜFUSU", "Latitude", "Longitude"]
    df = pd.read_parquet(path, columns=cols)
    # Adları normalize et
    df = df.rename(columns={
        f"{year} YILI NÜFUSU": "NÜFUS",
        "Latitude": "lat",
        "Longitude": "lon"
    })
    # İşte burası önemli: ilçe düzeyinde toplam al
    df_ilce = (
        df
        .groupby("İLÇE", as_index=False)
        .agg({
            "NÜFUS": "sum",
            "lat":   "mean",
            "lon":   "mean"
        })
    )
    return df_ilce

@st.cache_data
def load_parquet_mahalle(path: str, year: str) -> pd.DataFrame:
    cols = [
        "İLÇE", 
        "MAHALLE", 
        "MAHALLE KODU (AKS)", 
        f"{year} YILI NÜFUSU", 
        "Latitude", 
        "Longitude"
    ]
    df = pd.read_parquet(path, columns=cols)
    df = df.rename(
        columns={
            f"{year} YILI NÜFUSU": "NÜFUS",
            "Latitude": "lat",
            "Longitude": "lon",
        },
        errors="ignore"
    )
    return df

@st.cache_data
def load_parquet_demo(path: str, pct_col: str) -> pd.DataFrame:
    cols = [
        "İLÇE", 
        "MAHALLE", 
        "MAHALLE KODU (AKS)", 
        pct_col, 
        "Latitude", 
        "Longitude"
    ]
    df = pd.read_parquet(path, columns=cols)
    df = df.rename(
        columns={
            pct_col: "PCT",
            "Latitude": "lat",
            "Longitude": "lon",
        },
        errors="ignore"
    )
    return df

@st.cache_data
def load_geojson(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def build_geo_lookup(geojson: dict, key_prop: str) -> dict:
    return {
        feat["properties"][key_prop]: feat
        for feat in geojson.get("features", [])
        if key_prop in feat.get("properties", {})
    }


# -----------------------------
# 1) SAYFA KONFİGÜRASYONU & META YÜKLEME
# -----------------------------

st.set_page_config(page_title="Ordu Nüfus Haritası", layout="wide")
st.markdown("""<style>
  .deck-tooltip { background-color: magenta!important; color: white!important;
                  border-radius: 4px; padding: 4px; }
</style>""", unsafe_allow_html=True)
st.markdown("## 📊 Ordu İli Nüfus Haritası (2007 - 2024)")

# Tüm sütun isimleri ve lat/lon’u almak için
df = load_parquet_data("koordinatlı_nufus_verisi.parquet")
year_columns   = [c for c in df.columns if "YILI NÜFUSU" in c]
dropdown_years = [c.split()[0] for c in year_columns]
center_lat     = df["lat"].mean()
center_lon     = df["lon"].mean()


# GeoJSON + lookup (buraya ekledik)
ilce_geojson    = load_geojson("ILCELER.geojson")
ilce_lookup     = build_geo_lookup(ilce_geojson, "AD")
mahalle_geojson = load_geojson("MAHALLELER.geojson")
mahalle_lookup  = build_geo_lookup(mahalle_geojson, "KOD")

# -------------------------------
# 2. İLÇE BAZLI NÜFUS Harita ve Filtre
# -------------------------------

st.markdown("### 🗺️ İlçe Bazlı Nüfus Haritası (Yıl & Nüfus Aralığı)")

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
        lo, hi = map(int, st.session_state.ilce_range.split("-"))
        df_ilce = df_ilce[df_ilce["NÜFUS"].between(lo, hi)]
        st.markdown(f"**Seçilen İlçe Aralığı:** {lo} – {hi}")
        st.info(f"Kriterlere uygun {df_ilce.shape[0]} ilçe bulundu")

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
    df_ilce["cat"]   = pd.cut(df_ilce["NÜFUS"], bins=bins_i, labels=False, right=True)
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
        border_layer = pdk.Layer(
            "GeoJsonLayer",
            geojson_to_use,
            stroked=True,
            filled=False,
            get_line_color=[255, 0, 255, 200],
            line_width_min_pixels=1
        )
        layers.append(border_layer)

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
        df_export.to_excel(out_ilce, index=False, sheet_name="Ham İlçe Verisi")
        st.download_button(
            "Ham Veriyi indir",
            data=out_ilce.getvalue(),
            file_name=f"ilce_ham_{secili_yil_ilce}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            type="secondary"
        )
    with eb:
        outp_ilce = BytesIO()
        piv = (
            df_ilce[["İLÇE", "NÜFUS"]]
            .groupby("İLÇE")
            .sum()
            .reset_index()
            .assign(YIL=secili_yil_ilce)
        )
        toplam = piv["NÜFUS"].sum()
        piv = pd.concat([piv, pd.DataFrame([{"İLÇE": "Genel Toplam", "NÜFUS": toplam, "YIL": secili_yil_ilce}])], ignore_index=True)
        piv.to_excel(outp_ilce, index=False, sheet_name="Pivot İlçe")
        st.download_button(
            "Pivot Tabloyu indir",
            data=outp_ilce.getvalue(),
            file_name=f"ilce_pivot_{secili_yil_ilce}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            type="primary"
        )


# -------------------------------
# 2. MAHALLE BAZLI NÜFUS
# -------------------------------
st.markdown("### 🏘️ Mahalle Bazlı Nüfus Haritası (Yıl & Nüfus Aralığı)")

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
    min_pop, max_pop = int(df_mahalle["NÜFUS"].min()), int(df_mahalle["NÜFUS"].max())
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
                st.stop()
        st.session_state.pop_min = lo
        st.session_state.pop_max = hi
        st.session_state.filter_active = True

    if st.session_state.filter_active:
        lo, hi = st.session_state.pop_min, st.session_state.pop_max
        st.markdown(f"**Seçilen Aralık:** {lo:,} – {hi:,}".replace(",", "."))
        df_mahalle = df_mahalle[df_mahalle["NÜFUS"].between(lo, hi)].copy()
        count_ilce = df_mahalle["İLÇE"].nunique()
        count_mah  = df_mahalle.shape[0]
        st.info(f"Kriterlere uygun {count_ilce} ilçede {count_mah} mahalle bulundu")


    # 7) Formatlama & renk — vektörize
    df_mahalle["NÜFUS_FMT"] = (df_mahalle["NÜFUS"].astype(int).map("{:,.0f}".format).str.replace(",", "."))

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
    df_mahalle["cat"]   = pd.cut(df_mahalle["NÜFUS"], bins=bins_m, labels=False, right=True)
    df_mahalle["color"] = df_mahalle["cat"].map(dict(enumerate(colors_m)))
    df_mahalle.drop(columns=["cat"], inplace=True)


    # 8) Kümeli ScatterplotLayer (Mahalle)
    clustered_mahalle_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_mahalle,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius=150,
        pickable=True,
        cluster=True,        
        cluster_radius=50,
    )


    # 9) Sınır checkbox & GeoJSON filtresi
    show = st.checkbox("Mahalle Sınırlarını Göster", value=True, key="show_mahalle_borders_only")

    allowed = set(df_mahalle["MAHALLE KODU (AKS)"].astype(int))

    filtered_mahalle_features = [
        mahalle_lookup[code]
        for code in allowed
        if code in mahalle_lookup
    ]
    geo = {"type":"FeatureCollection","features": filtered_mahalle_features} if show and st.session_state.filter_active else mahalle_geojson
    border = pdk.Layer("GeoJsonLayer", geo, stroked=True, filled=False,
                       get_line_color=[3,32,252,180], line_width_min_pixels=1)

    # 10) Harita çizimi
    st.pydeck_chart(pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=8, pitch=40),
        layers=[clustered_mahalle_layer] + ([border] if show else []),
        tooltip={"html": "<b>{MAHALLE}</b><br/>İlçe: {İLÇE}<br/>Nüfus ({YIL}): {NÜFUS_FMT}".replace("{YIL}", str(secili_yil_mahalle))}
    ))

    # Excel indirme butonları
    col_excel1, col_excel2, _ = st.columns([1, 1, 8])

    # Ham veri indir
    with col_excel1:
        output = BytesIO()
        df_export = df_mahalle.copy()
        df_export["YIL"] = secili_yil_mahalle
        df_export["KONUMA GİT"] = df_export.apply(
            lambda row: f"https://www.google.com/maps?q={row['lat']},{row['lon']}&z=13&hl=tr",
            axis=1
        )
        df_export = df_export[["İLÇE", "MAHALLE", "YIL", "NÜFUS", "KONUMA GİT"]]
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_export.to_excel(writer, sheet_name="Ham Veri", index=False)
            ws = writer.sheets["Ham Veri"]
            link_col = df_export.columns.get_loc("KONUMA GİT")
            for idx, url in enumerate(df_export["KONUMA GİT"], start=1):
                ws.write_url(idx, link_col, url, string="Git")
        st.download_button(
            "Ham Veriyi indir",
            data=output.getvalue(),
            file_name=f"mahalle_ham_veri_{secili_yil_mahalle}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            type="secondary"
        )

    # Pivot tablo indir
    with col_excel2:
        pivot_output = BytesIO()
        pivot_df = pd.pivot_table(
            df_export,
            index=["İLÇE", "MAHALLE"],
            columns="YIL",
            values="NÜFUS",
            aggfunc="sum"
        ).reset_index()
        totals = pivot_df.select_dtypes(include=[int, float]).sum()
        totals["İLÇE"] = "Genel Toplam"
        totals["MAHALLE"] = ""
        pivot_df = pd.concat([pivot_df, totals.to_frame().T], ignore_index=True)
        coord_map = df_export.set_index(["İLÇE", "MAHALLE"])["KONUMA GİT"]
        pivot_df["KONUMA GİT"] = pivot_df.apply(
            lambda row: coord_map.get((row["İLÇE"], row["MAHALLE"]), ""),
            axis=1
        )
        with pd.ExcelWriter(pivot_output, engine="xlsxwriter") as writer:
            pivot_df.to_excel(writer, sheet_name="Pivot Tablo", index=False)
            ws = writer.sheets["Pivot Tablo"]
            git_col = pivot_df.columns.get_loc("KONUMA GİT")
            for idx, url in enumerate(pivot_df["KONUMA GİT"], start=1):
                if url:
                    ws.write_url(idx, git_col, url, string="Git")
        st.download_button(
            "Pivot Tabloyu indir",
            data=pivot_output.getvalue(),
            file_name=f"mahalle_pivot_{secili_yil_mahalle}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            type="primary"
        )

# -------------------------------
# 3. DEMOGRAFİ HARİTASI (Yaş Grupları Yüzde)
# -------------------------------
st.markdown("### 👥 Demografi Haritası (% Yaş Dağılımı)")

# a) Yüzde sütunlarını topla
pct_columns = [c for c in df.columns if c.endswith(" YAŞ YÜZDE")]
pct_labels  = [c.replace(" YAŞ YÜZDE","") for c in pct_columns]
label_to_col = dict(zip(pct_labels, pct_columns))

selected_label = st.selectbox("Yaş Grubu Yüzdesi Seçiniz", pct_labels, key="demography_pct")
selected_pct   = label_to_col.get(selected_label)
if not selected_pct:
    st.error("Lütfen listeden bir yaş grubu seçin.")
    st.stop()

# 1) Cache’lenmiş demografi verisi alın
df_demo = load_parquet_demo("koordinatlı_nufus_verisi.parquet", selected_pct)


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
                # eğer tam sayıysa .0’ı at, değilse olduğu gibi bırak
                return str(int(x)) if x.is_integer() else str(x)
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

if btn:
    st.session_state.dem_filter = True

# 3) Filtre aktiveyse uygula
if st.session_state.dem_filter and st.session_state.dem_range:
    lo, hi = map(float, st.session_state.dem_range.split("-"))
    df_demo = df_demo[df_demo["PCT"].between(lo, hi)].copy()
    st.markdown(f"**Seçilen Yüzde:** {lo:g} – {hi:g}")
    cnt_i = df_demo["İLÇE"].nunique()
    cnt_m = df_demo.shape[0]
    st.info(f"Kriterlere uygun {cnt_i} ilçede {cnt_m} mahalle bulundu")

# 4) Yüzde formatlama (tutarlı iki ondalık için .map)
df_demo["pct_numeric"]   = df_demo["PCT"]
df_demo["Yüzde Aralığı"] = df_demo["pct_numeric"].map("{:.2f} %".format)

# Demografi için kategorilere ayırma ve renk atama
bins_d = [-float("inf"), 5, 10, 15, 20, 25, 30, float("inf")]
colors_d = [
    [166,86,40,180],
    [152,78,163,180],
    [77,175,74,180],
    [55,126,184,180],
    [255,127,0,180],
    [255,255,51,180],
    [228,26,28,180],
]
# PCT sütunu zaten 0–1 aralığında; yüzdeye çevirmek için *100
df_demo["cat"]   = pd.cut(df_demo["PCT"], bins=bins_d, labels=False, right=True)
df_demo["color"] = df_demo["cat"].map(dict(enumerate(colors_d)))
df_demo.drop(columns=["cat"], inplace=True)


# 5) Kümeli ScatterplotLayer (Demografi)
clustered_demo_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_demo,
    get_position="[lon, lat]",
    get_fill_color="color",
    get_radius=150,
    pickable=True,
    cluster=True,     
    cluster_radius=50,
)


# 6) Sınır katmanı (lookup ile)
show = st.checkbox("Mahalle Sınırlarını Göster", value=True, key="show_demo_borders")
allowed = set(df_demo["MAHALLE KODU (AKS)"].astype(int))
features = [mahalle_lookup[k] for k in allowed if k in mahalle_lookup]
demo_geo = {"type":"FeatureCollection","features":features}

# 7) Katman listesi & render
layers = [clustered_demo_layer]
if show:
    geo_to_use = demo_geo if st.session_state.dem_filter else mahalle_geojson
    border_layer = pdk.Layer(
        "GeoJsonLayer",
        geo_to_use,
        stroked=True,
        filled=False,
        get_line_color=[3,32,252,180],
        line_width_min_pixels=1
    )
    layers.append(border_layer)


st.pydeck_chart(pdk.Deck(
    map_style  = "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
    initial_view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=8, pitch=40),
    layers     = layers,
    tooltip={
      "html": (
         "<b>{MAHALLE}</b><br/>"
         "İlçe: {İLÇE}<br/>"
         f"{selected_label} Yüzde: "+"{Yüzde Aralığı}"
      )
    }
))

# 1) Ham veri indir
col_ham, col_piv, _ = st.columns([1,1,8])

# 1) Ham veri indir
with col_ham:
    out_ham = BytesIO()
    # Ham veri: koordinatları at, pct_numeric’i iki ondalığa yuvarla ve sayısal bırak
    df_ham = df_demo[["İLÇE","MAHALLE","pct_numeric"]].copy()
    df_ham.rename(columns={"pct_numeric": "Yüzde Aralığı"}, inplace=True)
    df_ham["Yüzde Aralığı"] = df_ham["Yüzde Aralığı"].round(2)

    with pd.ExcelWriter(out_ham, engine="xlsxwriter") as writer:
        sheet = "Ham Demografi Verisi"
        wb = writer.book
        ws = wb.add_worksheet(sheet)
        writer.sheets[sheet] = ws
        # 1. satıra kullanıcı girdilerini yaz
        ws.write(0, 0, "Seçili Yaş Grubu:")
        ws.write(0, 1, selected_label)
        ws.write(0, 2, "Filtre Aralığı (%):")
        ws.write(0, 3, st.session_state.dem_range or "—")
        # 3. satırdan itibaren gerçek veri
        df_ham.to_excel(writer, sheet_name=sheet, index=False, startrow=1)

    st.download_button(
        "Ham Veriyi İndir",
        data=out_ham.getvalue(),
        file_name=f"demografi_ham_{selected_label}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        type="secondary"
    )

# 2) Pivot veri indir
with col_piv:
    out_piv = BytesIO()
    # Pivot: ilçede mahallelerin ortalama yüzdesini sayısal tut
    piv = (
        df_demo
        .groupby("İLÇE", as_index=False)
        .agg({"pct_numeric": "mean"})
        .rename(columns={"pct_numeric": "Yüzde Aralığı"})
    )
    piv["Yüzde Aralığı"] = piv["Yüzde Aralığı"].round(2)
    toplam = piv["Yüzde Aralığı"].mean().round(2)
    piv = pd.concat([piv, pd.DataFrame([{"İLÇE":"Ortalama Yüzde", "Yüzde Aralığı":toplam}])],
                    ignore_index=True)

    with pd.ExcelWriter(out_piv, engine="xlsxwriter") as writer:
        sheet = "Pivot Demografi"
        wb = writer.book
        ws = wb.add_worksheet(sheet)
        writer.sheets[sheet] = ws
        ws.write(0, 0, "Seçili Yaş Grubu:")
        ws.write(0, 1, selected_label)
        ws.write(0, 2, "Filtre Aralığı (%):")
        ws.write(0, 3, st.session_state.dem_range or "—")
        piv.to_excel(writer, sheet_name=sheet, index=False, startrow=1)

    st.download_button(
        "Pivot Tabloyu İndir",
        data=out_piv.getvalue(),
        file_name=f"demografi_pivot_{selected_label}.xlsx",
        mime="application/vnd.openxmlformats-officedocument-spreadsheetml.sheet",
        use_container_width=True,
        type="primary"
    )

