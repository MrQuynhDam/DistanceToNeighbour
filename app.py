import streamlit as st
import pandas as pd
import numpy as np
import re
import os

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Công Cụ Tìm Trạm Neighbor",
    page_icon="📡",
    layout="wide"
)

# --- CÁC HÀM XỬ LÝ DỮ LIỆU & TÍNH TOÁN ---

def clean_coord(coord):
    """
    Làm sạch tọa độ: Giữ lại số, dấu chấm (.) và dấu trừ (-).
    Xử lý tốt tọa độ bị định dạng Text, khoảng trắng, hậu tố E/N/W/S...
    """
    if pd.isna(coord):
        return np.nan
    if isinstance(coord, (int, float)):
        return float(coord)
    
    coord_str = str(coord).strip()
    match = re.search(r'[-+]?\d*\.?\d+', coord_str)
    if match:
        return float(match.group())
    return np.nan

def haversine_np(lat1, lon1, lat2, lon2):
    """Tính khoảng cách Haversine vector hóa bằng NumPy (tốc độ cao)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2)**2
    return 6371000 * 2 * np.arcsin(np.sqrt(a))

def process_neighbors(target_df, current_df, min_dist, max_dist, include_target):
    """Xử lý quét và tính toán khoảng cách Neighbor."""
    # Làm sạch dữ liệu tọa độ
    for df in (target_df, current_df):
        df['Lon'] = df['Lon'].apply(clean_coord)
        df['Lat'] = df['Lat'].apply(clean_coord)

    # Gộp dữ liệu theo tùy chọn
    if include_target:
        all_sites_df = pd.concat([target_df, current_df], ignore_index=True)
    else:
        all_sites_df = current_df.copy()

    all_lats = all_sites_df['Lat'].values
    all_lons = all_sites_df['Lon'].values

    neighbor_list = []
    
    # Duyệt từng trạm Site_Cell_list để quét khoảng cách
    for idx, target in target_df.iterrows():
        distances = haversine_np(target['Lat'], target['Lon'], all_lats, all_lons)
        
        # Điều kiện lọc: nằm trong khoảng cách quy định và không trùng Site
        valid_mask = (distances >= min_dist) & (distances <= max_dist) & (all_sites_df['Site'] != target['Site'])
        neighbors = all_sites_df[valid_mask].copy()

        if not neighbors.empty:
            neighbors['Distance (m)'] = np.round(distances[valid_mask]).astype(int)
            neighbors['Target Site'] = target['Site']

            min_distance = neighbors['Distance (m)'].min()
            neighbors['Is Nearest'] = neighbors['Distance (m)'] == min_distance

            neighbors = neighbors.rename(columns={'Site': 'Neighbor'})
            neighbor_list.append(neighbors[['Target Site', 'Neighbor', 'Distance (m)', 'Is Nearest']])

    if neighbor_list:
        output_df = pd.concat(neighbor_list, ignore_index=True)
        output_df = output_df.drop_duplicates(subset=['Target Site', 'Neighbor'])
    else:
        output_df = pd.DataFrame(columns=['Target Site', 'Neighbor', 'Distance (m)', 'Is Nearest'])

    return output_df

# --- GIAO DIỆN STREAMLIT ---

st.title("📡 Công Cụ Tìm Trạm Neighbor")
st.write("Tải lên file dữ liệu `.csv` để quét khoảng cách giữa các trạm.")

# Tải file Sample trực tiếp từ repo GitHub
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
                mime="text/csv"
            )

with col_sample2:
    sample_rim_path = "RIM_Sample.csv"
    if os.path.exists(sample_rim_path):
        with open(sample_rim_path, "rb") as f:
            st.download_button(
                label="📄 Tải mẫu File RIMS (.csv)",
                data=f.read(),
                file_name="RIM_Sample.csv",
                mime="text/csv"
            )

st.divider()

# Thanh bên (Sidebar) cấu hình tham số
st.sidebar.header("⚙️ Cấu hình thông số")

min_dist = st.sidebar.number_input("Khoảng cách tối thiểu (m):", min_value=0, value=0, step=50)
max_dist = st.sidebar.number_input("Khoảng cách tối đa (m):", min_value=1, value=5000, step=100)

include_target = st.sidebar.checkbox("Tìm Neighbor cả trong các site của File Site_Cell_list", value=True)

# Tải file CSV lên
col1, col2 = st.columns(2)

with col1:
    target_file = st.file_uploader("Chọn File Site_Cell_list (.csv)", type=["csv"])

with col2:
    data_file = st.file_uploader("Chọn File RIMS (.csv)", type=["csv"])

# Nút xử lý
if st.button("🚀 Tiến hành quét Neighbor", type="primary"):
    if target_file is None or data_file is None:
        st.error("Vui lòng tải lên đầy đủ cả 2 file `.csv` trước khi thực hiện!")
    elif min_dist >= max_dist:
        st.error("Khoảng cách tối thiểu phải nhỏ hơn khoảng cách tối đa!")
    else:
        try:
            with st.spinner("Đang xử lý dữ liệu và quét khoảng cách..."):
                # Đọc file CSV
                target_df = pd.read_csv(target_file)
                current_df = pd.read_csv(data_file)

                # Kiểm tra định dạng cột bắt buộc
                required_cols = {'Site', 'Lat', 'Lon'}
                if not required_cols.issubset(target_df.columns) or not required_cols.issubset(current_df.columns):
                    st.error(f"Cả 2 file phải chứa các cột bắt buộc: {required_cols}")
                else:
                    # Thực hiện tính toán
                    result_df = process_neighbors(target_df, current_df, min_dist, max_dist, include_target)

                    st.success("Xử lý hoàn tất!")

                    # Thống kê tổng quan
                    st.subheader("📊 Thống kê kết quả")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Tổng trạm Target (Site_Cell_list)", len(target_df))
                    m2.metric("Số cặp Neighbor tìm thấy", len(result_df))
                    m3.metric("Số trạm Target có Neighbor", result_df['Target Site'].nunique() if not result_df.empty else 0)

                    # Hiển thị dữ liệu
                    st.subheader("📋 Bảng kết quả")
                    st.dataframe(result_df, use_container_width=True)

                    # Export ra CSV
                    csv_data = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Tải xuống kết quả (.csv)",
                        data=csv_data,
                        file_name=f"NeighborList_{min_dist}m_to_{max_dist}m.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Có lỗi xảy ra trong quá trình xử lý: {str(e)}")
