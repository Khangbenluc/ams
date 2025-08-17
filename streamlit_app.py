import streamlit as st
import easyocr
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os

# --- Khởi tạo và Thiết lập ---
# Khởi tạo EasyOCR reader (chỉ chạy một lần, để tiết kiệm tài nguyên)
@st.cache_resource
def get_reader():
    return easyocr.Reader(['vi', 'en'], gpu=False) # Đặt gpu=True nếu có card đồ họa

reader = get_reader()
lich_su_file = 'lich_su_giao_dich.csv'

# Kiểm tra và tạo file CSV nếu chưa tồn tại
if not os.path.exists(lich_su_file):
    df = pd.DataFrame(columns=['Thời gian', 'Họ và Tên', 'Số CCCD', 'Quê quán', 'Khối lượng', 'Đơn giá', 'Thành tiền'])
    df.to_csv(lich_su_file, index=False)

# --- Các hàm xử lý logic ---
def trich_xuat_cccd(image):
    if image is None: return "", "", ""
    np_image = np.array(image)
    results = reader.readtext(np_image)
    ho_ten, so_cccd, que_quan = "", "", ""
    for (bbox, text, prob) in results:
        cleaned_text = text.strip().upper()
        if "HỌ VÀ TÊN" in cleaned_text:
            for i in range(results.index((bbox, text, prob)) + 1, len(results)):
                if results[i][1].strip().upper() not in ["", "CĂN CƯỚC CÔNG DÂN", "CỘNG HOÀ XÃ HỘI CHỦ NGHĨA VIỆT NAM", "ĐỘC LẬP - TỰ DO - HẠNH PHÚC"]:
                    ho_ten = results[i][1]
                    break
        elif ("SỐ" in cleaned_text or "SỐ:" in cleaned_text) and len(text.replace('SỐ', '').strip()) == 12:
            so_cccd = text.replace('SỐ', '').strip()
        elif "QUÊ QUÁN" in cleaned_text:
            for i in range(results.index((bbox, text, prob)) + 1, len(results)):
                que_quan = results[i][1]
                break
    return ho_ten, so_cccd, que_quan

def trich_xuat_can(image):
    if image is None: return ""
    np_image = np.array(image)
    results = reader.readtext(np_image, allowlist='0123456789.')
    for (bbox, text, prob) in results:
        cleaned_text = ''.join(c for c in text if c.isdigit() or c == '.')
        if cleaned_text: return cleaned_text
    return ""

def xu_ly_giao_dich(ho_va_ten, so_cccd, que_quan, so_luong_str, don_gia_str):
    try:
        so_luong = float(so_luong_str.replace(',', ''))
        don_gia = float(don_gia_str.replace(',', ''))
        thanh_tien = so_luong * don_gia
        ngay_tao = datetime.now().strftime("%d/%m/%Y")
        noi_dung_ban_ke = f"""
  BẢN KÊ MUA HÀNG

  ----------------------------------------
  Thông tin khách hàng:
  - Họ và Tên: {ho_va_ten}
  - Số CCCD: {so_cccd}
  - Quê quán: {que_quan}
  ----------------------------------------
  Chi tiết giao dịch:
  - Khối lượng: {so_luong}
  - Đơn giá: {don_gia:,.0f} VNĐ
  - Thành tiền: {thanh_tien:,.0f} VNĐ
  ----------------------------------------
  TP. HCM, ngày {ngay_tao}
      NGƯỜI BÁN                   KHÁCH HÀNG
      (Ký tên)                     (Ký tên)
    """
        thoi_gian = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df_moi = pd.DataFrame([{'Thời gian': thoi_gian, 'Họ và Tên': ho_va_ten, 'Số CCCD': so_cccd, 'Quê quán': que_quan, 'Khối lượng': so_luong, 'Đơn giá': don_gia, 'Thành tiền': thanh_tien}])
        df_moi.to_csv(lich_su_file, mode='a', header=False, index=False)
        return noi_dung_ban_ke, thanh_tien
    except ValueError:
        return "Lỗi: Khối lượng và Đơn giá phải là số.", 0.0

# --- Giao diện Streamlit ---
st.set_page_config(layout="wide")
st.title("ỨNG DỤNG TẠO BẢN KÊ MUA HÀNG")
st.markdown("---")

st.header("1. Nhập thông tin khách hàng")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Trích xuất từ CCCD")
    anh_cccd = st.camera_input("Chụp hoặc tải ảnh CCCD")
    if anh_cccd:
        ho_ten, so_cccd, que_quan = trich_xuat_cccd(np.array(anh_cccd.read()))
    else:
        ho_ten, so_cccd, que_quan = "", "", ""
    
with col2:
    st.subheader("Nhập liệu thủ công")
    ho_ten_input = st.text_input("Họ và Tên", value=ho_ten)
    so_cccd_input = st.text_input("Số Căn cước công dân", value=so_cccd)
    que_quan_input = st.text_input("Quê quán", value=que_quan)

st.markdown("---")

st.header("2. Nhập thông tin giao dịch")
col3, col4 = st.columns(2)

with col3:
    st.subheader("Trích xuất từ cân")
    anh_can = st.camera_input("Chụp hoặc tải ảnh màn hình cân")
    if anh_can:
        so_luong_str = trich_xuat_can(np.array(anh_can.read()))
    else:
        so_luong_str = ""

with col4:
    st.subheader("Nhập liệu thủ công")
    so_luong_input = st.text_input("Khối lượng (chỉ)", value=so_luong_str)
    don_gia_input = st.text_input("Đơn giá (VNĐ/chỉ)")

st.markdown("---")

st.header("3. Tạo bản kê")
if st.button("Tính tiền và Tạo bản kê"):
    if not ho_ten_input or not so_luong_input or not don_gia_input:
        st.error("Vui lòng nhập đầy đủ thông tin.")
    else:
        noi_dung_ban_ke, thanh_tien = xu_ly_giao_dich(ho_ten_input, so_cccd_input, que_quan_input, so_luong_input, don_gia_input)
        st.text_area("Bản Kê Mua Hàng", value=noi_dung_ban_ke, height=300)
        st.metric(label="Thành Tiền", value=f"{thanh_tien:,.0f} VNĐ")
        
st.markdown("---")

st.header("4. Lịch sử giao dịch")
if os.path.exists(lich_su_file):
    df_lich_su = pd.read_csv(lich_su_file)
    st.dataframe(df_lich_su)
else:
    st.info("Chưa có giao dịch nào.")
