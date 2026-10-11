import streamlit as st
import pandas as pd
import numpy as np
import re
import os

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Distance to Neighbours - Miền Nam",
    page_icon="📡",
    layout="wide"
)

# Custom CSS cho uploader
st.markdown("""
    <style>
    div[data-testid="stFileUploaderDropzoneInstructions"] > * {
        display: none !important;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"]::after {
        content: "10MB per file • CSV";
        font-size: 14px;
        color: #808495;
    }
    </style>
""", unsafe_allow_html=True)

# GIỚI HẠN ĐỊA LÝ KHU VỰC MIỀN NAM VIỆT NAM
LAT_MIN, LAT_MAX = 7.0, 14.0
LON_MIN, LON_MAX = 103.0, 114.0

# --- THUẬT TOÁN XỬ LÝ VÀ LÀM SẠCH TỌA ĐỘ ---

def normalize_columns(df):
    """Tự động chuẩn hóa tên cột về ['Site', 'Lat', 'Lon']."""
    cols_map = {}
    id_candidates = ['cellname', 'cell name', 'cell_name', 'site', 'site name', 'site_name', 'cell', 'cell_id']
    lat_candidates = ['lat', 'latitude', 'lat_deg', 'vĩ độ', 'vi do']
    lon_candidates = ['lon', 'long', 'longitude', 'lon_deg', 'kinh độ', 'kinh do']

    for col in df.columns:
        col_clean = str(col).strip().lower()
        if col_clean in id_candidates and 'Site' not in cols_map.values():
            cols_map[col] = 'Site'
        elif col_clean in lat_candidates and 'Lat' not in cols_map.values():
            cols_map[col] = 'Lat'
        elif col_clean in lon_candidates and 'Lon' not in cols_map.values():
            cols_map[col] = 'Lon'

    return df.rename(columns=cols_map)

def extract_float(val):
    """Trích xuất giá trị số từ chuỗi."""
    if pd.isna(val) or val is None:
        return np.nan
    val_str = str(val).strip().replace(',', '.')
    match = re.search(r'[-+]?\d*\.?\d+', val_str)
    if not match:
        return np.nan
    try:
        return float(match.group())
    except ValueError:
        return np.nan

def fix_coord_pair_miennam(lat_val, lon_val):
    """
    Sửa lỗi tọa độ và kiểm tra điều kiện Miền Nam Việt Nam:
    Vĩ độ Lat: [7, 14], Kinh độ Lon: [103, 114]
    """
    lat_num = extract_float(lat_val)
    lon_num = extract_float(lon_val)

    if pd.isna(lat_num) and pd.isna(lon_num):
        return np.nan, np.nan

    # 1. Sửa lỗi thiếu dấu chấm thập phân trước (nếu số quá lớn)
    if not pd.isna(lat_num):
        while abs(lat_num) > 180:
            lat_num /= 10.0
    if not pd.isna(lon_num):
        while abs(lon_num) > 180:
            lon_num /= 10.0

    # 2. Phát hiện đảo ngược Lat và Lon (Nếu Lat > 50 hoặc Lat > Lon)
    if (not pd.isna(lat_num) and lat_num > 50) or \
       (not pd.isna(lat_num) and not pd.isna(lon_num) and lat_num > lon_num):
        lat_num, lon_num = lon_num, lat_num

    # 3. Điều chỉnh tỷ lệ nếu vẫn vượt dải Miền Nam
    if not pd.isna(lat_num):
        while abs(lat_num) > LAT_MAX:
            lat_num /= 10.0
    if not pd.isna(lon_num):
        while abs(lon_num) > LON_MAX:
            lon_num /= 10.0

    # 4. Kiểm tra giới hạn địa lý Miền Nam Việt Nam [7, 14] và [103, 114]
    if not pd.isna(lat_num) and not (LAT_MIN <= lat_num <= LAT_MAX):
        lat_num = np.nan
    if not pd.isna(lon_num) and not (LON_MIN <= lon_num <= LON_MAX):
        lon_num = np.nan

    return lat_num, lon_num

def inspect_and_clean_data(df, file_label):
    """
    Kiểm tra, làm sạch dữ liệu và ghi log các trường lỗi (thiếu Cellname hoặc tọa độ sai dải).
    """
    df_norm = normalize_columns(df.copy())
    
    required_cols = {'Site', 'Lat', 'Lon'}
    if not required_cols.issubset(df_norm.columns):
        return None, pd.DataFrame([{
            "File": file_label, "Dòng CSV": "-", "Cellname": "-", "Lat gốc": "-", "Lon gốc": "-",
            "Lý do lỗi": f"Thiếu các cột bắt buộc: Cellname/Site, Lat, Lon"
        }])

    cleaned_rows = []
    invalid_records = []

    for idx, row in df_norm.iterrows():
        row_num = idx + 2  # Tính từ dòng header trong CSV
        cell_name = row['Site']
        raw_lat = row['Lat']
        raw_lon = row['Lon']

        is_cell_missing = pd.isna(cell_name) or str(cell_name).strip() == ""
        lat_clean, lon_clean = fix_coord_pair_miennam(raw_lat, raw_lon)
        is_coord_invalid = pd.isna(lat_clean) or pd.isna(lon_clean)

        if is_cell_missing or is_coord_invalid:
            reasons = []
            if is_cell_missing:
                reasons.append("Thiếu Cellname / Site")
            if is_coord_invalid:
                reasons.append(f"Tọa độ không hợp lệ hoặc nằm ngoài dải Miền Nam [{LAT_MIN}-{LAT_MAX}, {LON_MIN}-{LON_MAX}]")

            invalid_records.append({
                "File": file_label,
                "Dòng CSV": row_num,
                "Cellname": cell_name if not is_cell_missing else "<THIẾU>",
                "Lat gốc": raw_lat,
                "Lon gốc": raw_lon,
                "Lý do lỗi": " | ".join(reasons)
            })
        else:
            cleaned_rows.append({
                "Site": str(cell_name).strip(),
                "Lat": lat_clean,
                "Lon": lon_clean
            })

    cleaned_df = pd.DataFrame(cleaned_rows)
    invalid_df = pd.DataFrame(invalid_records)

    return cleaned_df, invalid_df

def haversine_np(lat1, lon1, lat2, lon2):
    """Tính khoảng cách Haversine vector hóa bằng NumPy (mét)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2)**2
    return 6371000 * 2 * np.arcsin(np.sqrt(a))

def process_neighbors(target_clean_df, current_clean_df, min_dist, max_dist, include_target):
    """Quét và tính toán khoảng cách giữa các trạm."""
    if include_target:
        all_sites_df = pd.concat([target_clean_df, current_clean_df], ignore_index=True)
    else:
        all_sites_df = current_clean_df.copy()

    all_sites_df = all_sites_df.drop_duplicates(subset=['Site', 'Lat', 'Lon']).reset_index(drop=True)

    all_lats = all_sites_df['Lat'].values
    all_lons = all_sites_df['Lon'].values

    neighbor_list = []

    for idx, target in target_clean_df.iterrows():
        distances = haversine_np(target['Lat'], target['Lon'], all_lats, all_lons)
        valid_mask = (distances >= min_dist) & (distances <= max_dist) & (all_sites_df['Site'] != target['Site'])
        neighbors = all_sites_df[valid_mask].copy()

        if not neighbors.empty:
            neighbors['Distance (m)'] = np.round(distances[valid_mask]).astype(int)
            neighbors['Target Cell'] = target['Site']

            min_distance = neighbors['Distance (m)'].min()
            neighbors['Is Nearest'] = neighbors['Distance (m)'] == min_distance

            neighbors = neighbors.rename(columns={'Site': 'Neighbor Cell'})
            neighbor_list.append(neighbors[['Target Cell', 'Neighbor Cell', 'Distance (m)', 'Is Nearest']])

    if neighbor_list:
        output_df = pd.concat(neighbor_list, ignore_index=True)
        return output_df.drop_duplicates(subset=['Target Cell', 'Neighbor Cell'])
    else:
        return pd.DataFrame(columns=['Target Cell', 'Neighbor Cell', 'Distance (m)', 'Is Nearest'])

# --- GIAO DIỆN STREAMLIT ---

st.title("📡 Distance to Neighbours (Khu vực Miền Nam)")
st.caption("Ứng dụng tự động chuẩn hóa tọa độ trong dải Vĩ độ **[7.0, 14.0]** & Kinh độ **[103.0, 114.0]**.")

# Tải file Sample
st.markdown("### 📥 Tải file dữ liệu mẫu")
col_sample1, col_sample2 = st.columns(2)

with col_sample1:
    sample_target_path = "Site_Cell_List_Sample.csv"
    if os.path.exists(sample_target_path):
        with open(sample_target_path, "rb") as f:
            st.download_button(
                label="📄 Tải mẫu File Site_Cell_list (.csv)",
                data=f.read(),
                file_name="Site_Cell_List_Sample.csv",
                mime="text/csv",
                key="btn_sample_target"
            )

with col_sample2:
    sample_rim_path = "RIM_Sample.csv"
    if os.path.exists(sample_rim_path):
        with open(sample_rim_path, "rb") as f:
            st.download_button(
                label="📄 Tải mẫu File RIMS (.csv)",
                data=f.read(),
                file_name="RIM_Sample.csv",
                mime="text/csv",
                key="btn_sample_rim"
            )

st.divider()

# Cấu hình thông số
st.sidebar.header("⚙️ Cấu hình thông số")
min_dist = st.sidebar.number_input("Khoảng cách tối thiểu (m):", min_value=0, value=0, step=50)
max_dist = st.sidebar.number_input("Khoảng cách tối đa (m):", min_value=1, value=5000, step=100)
include_target = st.sidebar.checkbox("Tìm Neighbor cả trong File Site_Cell_list", value=True)

# Upload File
col1, col2 = st.columns(2)
with col1:
    target_file = st.file_uploader("Chọn File Site_Cell_list (.csv)", type=["csv"])
with col2:
    data_file = st.file_uploader("Chọn File RIMS (.csv)", type=["csv"])

# Thực hiện tính toán
if st.button("🚀 Calculate...", type="primary"):
    if target_file is None or data_file is None:
        st.error("Vui lòng tải lên đầy đủ cả 2 file `.csv`!")
    elif min_dist >= max_dist:
        st.error("Khoảng cách tối thiểu phải nhỏ hơn khoảng cách tối đa!")
    else:
        try:
            # Thanh tiến độ và Log tiến trình
            progress_bar = st.progress(0)
            with st.status("🔄 Đang xử lý và kiểm tra dữ liệu...", expanded=True) as status:
                
                # Bước 1: Đọc dữ liệu
                st.write("📂 **Bước 1/4:** Đang đọc các file CSV...")
                progress_bar.progress(15)
                target_df_raw = pd.read_csv(target_file)
                current_df_raw = pd.read_csv(data_file)

                # Bước 2: Kiểm tra & Làm sạch dữ liệu File 1
                st.write("🔍 **Bước 2/4:** Kiểm tra Cellname & Tọa độ Miền Nam (File Site_Cell_list)...")
                progress_bar.progress(35)
                target_clean, target_invalid = inspect_and_clean_data(target_df_raw, "Site_Cell_list")

                # Bước 3: Kiểm tra & Làm sạch dữ liệu File 2
                st.write("🔍 **Bước 3/4:** Kiểm tra Cellname & Tọa độ Miền Nam (File RIMS)...")
                progress_bar.progress(60)
                current_clean, current_invalid = inspect_and_clean_data(current_df_raw, "RIMS")

                all_invalid_df = pd.concat([target_invalid, current_invalid], ignore_index=True)

                if target_clean is None or current_clean is None:
                    status.update(label="❌ Cấu trúc File không hợp lệ!", state="error", expanded=True)
                    st.error("File tải lên thiếu các cột bắt buộc. Vui lòng kiểm tra lại!")
                elif target_clean.empty:
                    status.update(label="❌ Không có dữ liệu hợp lệ trong File Site_Cell_list!", state="error", expanded=True)
                    st.error("Tất cả các dòng trong File Site_Cell_list đều bị lỗi tọa độ hoặc thiếu Cellname!")
                else:
                    # Bước 4: Tính toán khoảng cách
                    st.write("🧮 **Bước 4/4:** Đang tính toán khoảng cách Neighbor Haversine...")
                    progress_bar.progress(85)
                    
                    result_df = process_neighbors(target_clean, current_clean, min_dist, max_dist, include_target)
                    progress_bar.progress(100)
                    
                    status.update(label="✅ Xử lý hoàn tất thành công!", state="complete", expanded=False)

                    # Lưu session state
                    st.session_state['result_df'] = result_df
                    st.session_state['invalid_df'] = all_invalid_df
                    st.session_state['total_targets_clean'] = len(target_clean)
                    st.session_state['min_dist'] = min_dist
                    st.session_state['max_dist'] = max_dist

        except Exception as e:
            st.error(f"Có lỗi xảy ra trong quá trình xử lý: {str(e)}")

# --- HIỂN THỊ KẾT QUẢ & LOG BÁO CÁO LỖI ---
if 'result_df' in st.session_state:
    result_df = st.session_state['result_df']
    invalid_df = st.session_state['invalid_df']
    total_targets_clean = st.session_state['total_targets_clean']

    # Thống kê tổng quan
    st.subheader("📊 Thống kê kết quả")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tổng Cell Target hợp lệ", total_targets_clean)
    m2.metric("Số cặp Neighbor tìm thấy", len(result_df))
    m3.metric("Cell Target có Neighbor", result_df['Target Cell'].nunique() if not result_df.empty else 0)
    m4.metric("Số dòng dữ liệu lỗi/loại bỏ", len(invalid_df))

    # Báo cáo danh sách lỗi (nếu có)
    if not invalid_df.empty:
        st.warning(f"⚠️ Phát hiện **{len(invalid_df)}** dòng dữ liệu không hợp lệ (thiếu Cellname hoặc tọa độ nằm ngoài dải Miền Nam [{LAT_MIN}-{LAT_MAX}, {LON_MIN}-{LON_MAX}]).")
        with st.expander("🔍 Xem danh sách chi tiết các dòng bị lỗi / loại bỏ"):
            st.dataframe(invalid_df, use_container_width=True)
            
            err_csv = invalid_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Tải xuống báo cáo danh sách lỗi (.csv)",
                data=err_csv,
                file_name="Log_Error_Report.csv",
                mime="text/csv",
                key="btn_download_error"
            )

    st.divider()

    # Bảng kết quả chính
    st.subheader("📋 Bảng kết quả Neighbor")
    st.dataframe(result_df, use_container_width=True)

    # Nút Download kết quả
    csv_data = result_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Tải xuống kết quả Neighbor (.csv)",
        data=csv_data,
        file_name=f"NeighborList_{st.session_state['min_dist']}m_to_{st.session_state['max_dist']}m.csv",
        mime="text/csv",
        key="btn_download_result"
    )
