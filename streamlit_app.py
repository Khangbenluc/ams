# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
import pytz
import easyocr
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import mm
import re
import os
import matplotlib.pyplot as plt
import tempfile
import json # Thêm thư viện để lưu mảng vào DB

# ========== CẤU HÌNH =============
st.set_page_config(layout="wide")

# --- Quản lý người dùng (đơn giản, demo) ---
users = {
    "admin": "admin123",
    "user1": "user123"
}

# --- Khởi tạo EasyOCR (cache) ---
@st.cache_resource
def get_reader():
    return easyocr.Reader(['vi', 'en'], gpu=False)

reader = get_reader()

# --- Kết nối SQLite ---
conn = sqlite3.connect("lich_su_giao_dich.db", check_same_thread=False)
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS lich_su (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thoi_gian TEXT,
    ho_va_ten TEXT,
    so_cccd TEXT,
    que_quan TEXT,
    hang_hoa_json TEXT,
    tong_thanh_tien REAL
)
''')
conn.commit()

# ========== HỖ TRỢ NHIỀU HÀM =============
def doc_so_thanh_chu(number):
    if not isinstance(number, (int, float)) or number < 0:
        return "Số không hợp lệ"
    number = int(number)
    chu_so = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
    don_vi = ["", "nghìn", "triệu", "tỷ", "nghìn tỷ", "triệu tỷ", "tỷ tỷ"]
    def doc_ba_so(so):
        if so == 0: return ""
        tram = so // 100
        chuc = (so % 100) // 10
        don_vi_le = so % 10
        chuoi = ""
        if tram > 0:
            chuoi += chu_so[tram] + " trăm "
        if chuc == 0 and don_vi_le > 0 and tram > 0:
            chuoi += "linh "
        elif chuc == 1:
            chuoi += "mười "
        elif chuc > 1:
            chuoi += chu_so[chuc] + " mươi "
        if don_vi_le == 5 and chuc != 0:
            chuoi += "lăm"
        elif don_vi_le == 1 and chuc != 0 and chuc != 1:
            chuoi += "mốt"
        elif don_vi_le > 0:
            chuoi += chu_so[don_vi_le]
        return chuoi.strip()
    s = str(number)
    parts = []
    while len(s) > 0:
        if len(s) >= 3:
            part = s[-3:]
            s = s[:-3]
        else:
            part = s
            s = ""
        parts.insert(0, int(part))
    ket_qua = ""
    for i, p in enumerate(parts):
        if p != 0:
            ket_qua += doc_ba_so(p) + " " + don_vi[len(parts) - 1 - i] + " "
    return ket_qua.strip().capitalize() + " đồng"

# --- Image helpers ---
def _bytes_to_bgr(image_bytes):
    return cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

def preprocess_image_for_ocr(image_bytes):
    img = _bytes_to_bgr(image_bytes)
    if img is None:
        return None
    # cải thiện: grayscale -> bilateral -> adaptive threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 9)
    # trả về màu BGR vì easyocr chấp nhận cả ảnh màu/ngang
    return cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)

# --- EasyOCR extract helper (dùng detail=0 -> list text) ---
def _easyocr_texts_from_bytes(image_bytes):
    img = _bytes_to_bgr(image_bytes)
    if img is None:
        return []
    try:
        texts = reader.readtext(img, detail=0)
        return [str(t).strip() for t in texts if t is not None]
    except Exception:
        # fallback: dùng preprocessed
        proc = preprocess_image_for_ocr(image_bytes)
        if proc is None:
            return []
        try:
            texts = reader.readtext(proc, detail=0)
            return [str(t).strip() for t in texts if t is not None]
        except Exception:
            return []

# --- Hàm OCR CCCD bằng EasyOCR ---
def trich_xuat_cccd_easy(image_bytes):
    ho_ten, so_cccd, que_quan = "", "", ""
    try:
        texts = _easyocr_texts_from_bytes(image_bytes)
        if not texts:
            return "", "", ""
        texts_upper = [t.upper() for t in texts]

        # Tìm "HỌ VÀ TÊN" hoặc "HỌ TÊN" hoặc "HỌ & TÊN"
        for i, t in enumerate(texts_upper):
            if "HỌ VÀ TÊN" in t or "HỌ TÊN" in t or "HỌ VÀ TÊN:" in t or "HỌ & TÊN" in t:
                if i + 1 < len(texts):
                    ho_ten = texts[i + 1]
                break
        # fallback: tìm dòng chứa "Họ" + dấu ví dụ "Họ tên: NGUYEN VAN A"
        if not ho_ten:
            for t in texts:
                m = re.search(r"Họ( và)? tên[:\s\-]*([A-Za-zÀ-ỹ\s]+)", t, re.IGNORECASE)
                if m:
                    ho_ten = m.group(2).strip()
                    break

        # Số CCCD (12 chữ số)
        pat_cccd = re.compile(r"\d{12}")
        for t in texts:
            m = pat_cccd.search(t.replace(" ", ""))
            if m:
                so_cccd = m.group(0)
                break

        # Quê quán
        for i, t in enumerate(texts_upper):
            if "QUÊ QUÁN" in t or "QUE QUAN" in t:
                if i + 1 < len(texts):
                    que_quan = texts[i + 1]
                break
        # fallback: nếu vẫn rỗng, tìm dòng chứa từ "QUÊ" hoặc "QUÊ QUÁN"
        if not que_quan:
            for t in texts:
                if "QUÊ" in t.upper():
                    que_quan = t
                    break

        return ho_ten, so_cccd, que_quan
    except Exception:
        return "", "", ""

# --- Hàm OCR cân bằng EasyOCR ---
def trich_xuat_can_easy(image_bytes):
    try:
        # dùng preprocessed ảnh cân để tăng độ chính xác số
        proc = preprocess_image_for_ocr(image_bytes)
        texts = []
        if proc is not None:
            try:
                texts = reader.readtext(proc, detail=0)
            except Exception:
                pass
        if not texts:
            texts = _easyocr_texts_from_bytes(image_bytes)
        if not texts:
            return ""
        candidates = []
        for t in texts:
            for m in re.findall(r"[0-9]+(?:[.,][0-9]+)?", str(t)):
                val = m.replace(",", ".")
                try:
                    candidates.append((m, float(val)))
                except:
                    pass
        if not candidates:
            return ""
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0].replace(",", ".")
    except Exception:
        return ""

# ========== Hàm tính tiền & PDF (giữ nguyên chức năng) ==========
def xu_ly_giao_dich(ho_va_ten, so_cccd, que_quan, items_list):
    try:
        # Tính tổng thành tiền từ list items
        tong_thanh_tien = 0
        hang_hoa_luu = []
        for item in items_list:
            so_luong = float(str(item['so_luong']).replace(',', ''))
            don_gia = float(str(item['don_gia']).replace(',', ''))
            thanh_tien = so_luong * don_gia
            tong_thanh_tien += thanh_tien
            hang_hoa_luu.append({
                "ten": item['ten_hang'],
                "so_luong": so_luong,
                "don_gia": don_gia,
                "thanh_tien": thanh_tien
            })

        vn_timezone = pytz.timezone('Asia/Ho_Chi_Minh')
        current_time = datetime.now(vn_timezone)
        thoi_gian_luu = current_time.strftime("%Y-%m-%d %H:%M:%S")

        # Chuyển list items thành JSON string để lưu vào DB
        hang_hoa_json = json.dumps(hang_hoa_luu)

        c.execute('''
            INSERT INTO lich_su (thoi_gian, ho_va_ten, so_cccd, que_quan, hang_hoa_json, tong_thanh_tien)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (thoi_gian_luu, ho_va_ten, so_cccd, que_quan, hang_hoa_json, tong_thanh_tien))
        conn.commit()

        return {
            "ho_va_ten": ho_va_ten,
            "so_cccd": so_cccd,
            "que_quan": que_quan,
            "items": hang_hoa_luu,
            "tong_thanh_tien": tong_thanh_tien,
            "ngay_tao": current_time.strftime("%d/%m/%Y")
        }
    except (ValueError, TypeError) as e:
        st.error(f"Lỗi: Dữ liệu nhập không hợp lệ. {e}")
        return None

FONT_FILE = "arial.ttf"
FONT_NAME = "Arial"
try:
    if os.path.exists(FONT_FILE):
        pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_FILE))
    else:
        st.warning(f"Không tìm thấy font '{FONT_FILE}'. Vui lòng đặt font vào cùng thư mục với file app.")
except Exception as e:
    st.error(f"Lỗi khi đăng ký font: {e}")
    FONT_NAME = "Helvetica"

def tao_pdf_mau_01(data, ten_don_vi=""):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # Sử dụng font đã đăng ký
    pdf.setFont(FONT_NAME, 12)

    # Vị trí cố định
    if ten_don_vi:
        pdf.drawString(20*mm, height - 15*mm, ten_don_vi.upper())

    pdf.drawCentredString(width/2, height - 20*mm, "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM")
    pdf.drawCentredString(width/2, height - 25*mm, "Độc lập - Tự do - Hạnh phúc")
    pdf.drawCentredString(width/2, height - 30*mm, "--------------------------")
    pdf.drawRightString(width - 20*mm, height - 35*mm, "Mẫu số: 01/TNDN")

    pdf.setFont(FONT_NAME, 14)
    pdf.drawCentredString(width/2, height - 50*mm, "BẢNG KÊ THU MUA HÀNG HÓA, DỊCH VỤ")
    pdf.drawCentredString(width/2, height - 55*mm, "KHÔNG CÓ HÓA ĐƠN")

    pdf.setFont(FONT_NAME, 12)
    pdf.drawString(20*mm, height - 70*mm, f"Họ và tên người bán: {data['ho_va_ten']}")
    pdf.drawString(20*mm, height - 75*mm, f"Số CCCD: {data['so_cccd']}")
    pdf.drawString(20*mm, height - 80*mm, f"Quê quán: {data['que_quan']}")
    pdf.drawString(20*mm, height - 85*mm, f"Ngày lập: {data['ngay_tao']}")

    # --- Bảng hàng hóa (động) ---
    y_start_table = height - 100*mm
    pdf.rect(20*mm, y_start_table - (len(data['items']) + 1) * 10 * mm, 170*mm, (len(data['items']) + 1) * 10 * mm)

    # Vẽ tiêu đề
    pdf.drawString(22*mm, y_start_table - 5*mm, "STT")
    pdf.drawString(35*mm, y_start_table - 5*mm, "Tên hàng hóa, dịch vụ")
    pdf.drawString(100*mm, y_start_table - 5*mm, "Đơn vị tính")
    pdf.drawString(120*mm, y_start_table - 5*mm, "Số lượng")
    pdf.drawString(140*mm, y_start_table - 5*mm, "Đơn giá")
    pdf.drawString(170*mm, y_start_table - 5*mm, "Thành tiền")

    # Vẽ các dòng hàng hóa
    y_item = y_start_table - 15*mm
    for i, item in enumerate(data['items']):
        pdf.drawString(22*mm, y_item, str(i + 1))
        
        # Xử lý tên hàng hóa nếu quá dài
        ten_hang = item['ten']
        if pdf.stringWidth(ten_hang, FONT_NAME, 12) > 60*mm:
            ten_hang = ten_hang[:int(len(ten_hang)*60/pdf.stringWidth(ten_hang, FONT_NAME, 12))] + "..."
            
        pdf.drawString(35*mm, y_item, ten_hang)
        pdf.drawString(100*mm, y_item, "chỉ")
        pdf.drawString(120*mm, y_item, f"{item['so_luong']:,.2f}")
        pdf.drawString(140*mm, y_item, f"{item['don_gia']:,.0f}")
        pdf.drawString(170*mm, y_item, f"{item['thanh_tien']:,.0f}")
        y_item -= 10*mm
    
    # --- Tổng cộng ---
    y = y_item - 5*mm
    pdf.drawString(20*mm, y, f"Tổng cộng: {data['tong_thanh_tien']:,.0f} VNĐ")

    y -= 5*mm
    pdf.drawString(20*mm, y, f"Bằng chữ: {doc_so_thanh_chu(data['tong_thanh_tien'])}")

    # --- Xuống ngay dưới để thêm ngày tháng và chữ ký ---
    y -= 20*mm
    pdf.setFont(FONT_NAME, 11)
    vn_timezone = pytz.timezone('Asia/Ho_Chi_Minh')
    current_time = datetime.now(vn_timezone)
    current_date_str = f"Bến Lức, ngày {current_time.day} tháng {current_time.month} năm {current_time.year}"
    pdf.drawRightString(width - 30*mm, y, current_date_str)

    # Người mua (bên trái)
    pdf.drawString(30*mm, y - 20, "Người mua")
    pdf.drawString(30*mm, y - 30, "(Ký, ghi rõ họ tên)")

    # Giám đốc (bên phải)
    pdf.drawRightString(width - 30*mm, y - 20, "Giám đốc")
    pdf.drawRightString(width - 30*mm, y - 30, "(Ký, đóng dấu)")

    pdf.save()
    buffer.seek(0)
    return buffer


# ========== GIAO DIỆN ==========
def login_page():
    st.title("Đăng nhập/Đăng ký")
    menu = ["Đăng nhập", "Đăng ký"]
    choice = st.selectbox("Chọn", menu)
    if choice == "Đăng nhập":
        st.subheader("Đăng nhập")
        username = st.text_input("Tên đăng nhập")
        password = st.text_input("Mật khẩu", type="password")
        if st.button("Đăng nhập"):
            if username in users and users[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Chào mừng, {username}!")
                st.rerun()
            else:
                st.error("Tên đăng nhập hoặc mật khẩu không đúng.")
    elif choice == "Đăng ký":
        st.subheader("Đăng ký tài khoản mới")
        new_user = st.text_input("Tên đăng nhập mới")
        new_password = st.text_input("Mật khẩu mới", type="password")
        if st.button("Đăng ký"):
            if new_user in users:
                st.warning("Tên đăng nhập đã tồn tại.")
            else:
                users[new_user] = new_password
                st.success("Đăng ký thành công! Vui lòng đăng nhập.")
                st.balloons()

def main_app():
    st.set_page_config(layout="wide")
    st.title("ỨNG DỤNG TẠO BẢN KÊ MUA HÀNG - 01/TNDN")
    st.markdown("---")
    
    col_reset, col_logout = st.columns([1,1])
    with col_reset:
        if st.button("🔴 Clear Session State"):
            # Explicitly clear the session state to fix corrupted data
            for key in list(st.session_state.keys()): # Use list() to avoid issues with modifying the dictionary during iteration
                del st.session_state[key]
            st.session_state.logged_in = True # Keep the user logged in
            # Thêm dòng này để đảm bảo items được khởi tạo lại đúng cách sau khi xóa
            st.session_state.items = [{"ten_hang": "", "so_luong": "", "don_gia": ""}]
            st.rerun()

    with col_logout:
        if st.button("Đăng xuất"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.rerun()

    tab1, tab2 = st.tabs(["Tạo giao dịch", "Lịch sử & Thống kê"])
    with tab1:
        create_new_transaction_page()
    with tab2:
        history_and_stats_page()

def create_new_transaction_page():
    # SỬA LỖI: Luôn đảm bảo st.session_state.items là một list
    if 'items' not in st.session_state or not isinstance(st.session_state.items, list):
        st.session_state.items = [{"ten_hang": "", "so_luong": "", "don_gia": ""}]

    # Khởi tạo các biến session_state mặc định khác
    defaults = {
        "ho_ten": "",
        "so_cccd": "",
        "que_quan": "",
        "pdf_for_download": None,
        "giao_dich_data": None,
        "ten_don_vi": "",
        "phuong_thuc": "Nhập thủ công"
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    st.subheader("1. Chọn phương thức nhập liệu")
    st.session_state.phuong_thuc = st.radio("Chọn phương thức:", ["Nhập thủ công", "Sử dụng OCR"], index=0 if st.session_state.phuong_thuc == "Nhập thủ công" else 1)
    
    st.markdown("---")
    
    # Logic OCR (chỉ hiển thị khi chọn OCR)
    if st.session_state.phuong_thuc == "Sử dụng OCR":
        st.subheader("Trích xuất thông tin từ ảnh 🖼️")
        col_cccd, col_can = st.columns([1,1])

        with col_cccd:
            st.subheader("Chụp ảnh hoặc tải ảnh CCCD")
            anh_cccd = st.camera_input("Chụp ảnh CCCD")
            uploaded_cccd = st.file_uploader("Hoặc tải ảnh CCCD", type=["jpg", "jpeg", "png"], key="cccd_uploader")
            if anh_cccd:
                with st.spinner("Đang xử lý OCR CCCD..."):
                    ho_ten, so_cccd, que_quan = trich_xuat_cccd_easy(anh_cccd.read())
                if ho_ten: st.session_state.ho_ten = ho_ten
                if so_cccd: st.session_state.so_cccd = so_cccd
                if que_quan: st.session_state.que_quan = que_quan
                st.success("Đã trích xuất thông tin CCCD!")
                st.image(anh_cccd, use_container_width=True)
            elif uploaded_cccd:
                with st.spinner("Đang xử lý OCR CCCD..."):
                    ho_ten, so_cccd, que_quan = trich_xuat_cccd_easy(uploaded_cccd.read())
                if ho_ten: st.session_state.ho_ten = ho_ten
                if so_cccd: st.session_state.so_cccd = so_cccd
                if que_quan: st.session_state.que_quan = que_quan
                st.success("Đã trích xuất thông tin CCCD!")
                st.image(uploaded_cccd, use_container_width=True)
        
        # Hiện tại OCR chỉ hỗ trợ 1 món, nên chỉ hiện OCR cân cho món 1
        with col_can:
            st.subheader("Chụp ảnh hoặc tải ảnh cân")
            anh_can = st.camera_input("Chụp ảnh màn hình cân")
            uploaded_can = st.file_uploader("Hoặc tải ảnh cân", type=["jpg", "jpeg", "png"], key="can_uploader")
            if anh_can:
                with st.spinner("Đang xử lý OCR cân..."):
                    so_luong_item1 = trich_xuat_can_easy(anh_can.read())
                if so_luong_item1:
                    if len(st.session_state.items) > 0:
                        st.session_state.items[0]['so_luong'] = so_luong_item1
                st.success("Đã trích xuất khối lượng!")
                st.image(anh_can, use_container_width=True)
            elif uploaded_can:
                with st.spinner("Đang xử lý OCR cân..."):
                    so_luong_item1 = trich_xuat_can_easy(uploaded_can.read())
                if so_luong_item1:
                    if len(st.session_state.items) > 0:
                        st.session_state.items[0]['so_luong'] = so_luong_item1
                st.success("Đã trích xuất khối lượng!")
                st.image(uploaded_can, use_container_width=True)
        
        st.markdown("---")

    st.subheader("2. Nhập thông tin và lưu giao dịch 📝")
    st.write("**(Nếu OCR đã trích xuất được, ô tương ứng sẽ bị khóa. Nếu chưa có, bạn có thể nhập thủ công.)**")
    
    # Hiển thị tóm tắt thông tin CCCD
    st.info(f"Họ và Tên: **{st.session_state.ho_ten}**")
    st.info(f"Số CCCD: **{st.session_state.so_cccd}**")
    st.info(f"Quê quán: **{st.session_state.que_quan}**")
    
    # Nhập thông tin CCCD
    ho_ten_input = st.text_input("Họ và tên người bán", value=st.session_state.ho_ten, disabled=st.session_state.phuong_thuc == "Sử dụng OCR", key="ho_ten_input")
    so_cccd_input = st.text_input("Số CCCD", value=st.session_state.so_cccd, disabled=st.session_state.phuong_thuc == "Sử dụng OCR", key="so_cccd_input")
    que_quan_input = st.text_area("Quê quán", value=st.session_state.que_quan, disabled=st.session_state.phuong_thuc == "Sử dụng OCR", key="que_quan_input")
    
    st.text_input("Tên đơn vị (không bắt buộc)", key="ten_don_vi_input")
    
    st.markdown("---")
    
    st.subheader("3. Nhập thông tin hàng hóa")

    # Các nút để thêm/xóa món hàng
    col_add_item, col_remove_item = st.columns([1,1])
    with col_add_item:
        if st.button("➕ Thêm món hàng", disabled=(len(st.session_state.items) >= 3)):
            st.session_state.items.append({"ten_hang": "", "so_luong": "", "don_gia": ""})
            st.rerun()
    with col_remove_item:
        if st.button("➖ Xóa món hàng cuối", disabled=(len(st.session_state.items) <= 1)):
            st.session_state.items.pop()
            st.rerun()

    # Tạo các cột nhập liệu cho từng món hàng
    for i in range(len(st.session_state.items)):
        st.markdown(f"**Món hàng {i+1}**")
        cols = st.columns([2, 1, 1])
        with cols[0]:
            st.session_state.items[i]['ten_hang'] = st.text_input(f"Tên hàng hóa", 
                                                                    value=st.session_state.items[i].get('ten_hang', ''),
                                                                    key=f"ten_hang_{i}")
        with cols[1]:
            st.session_state.items[i]['so_luong'] = st.text_input(f"Khối lượng (chỉ)", 
                                                                    value=st.session_state.items[i].get('so_luong', ''),
                                                                    disabled=(i == 0 and st.session_state.phuong_thuc == "Sử dụng OCR"),
                                                                    key=f"so_luong_{i}")
        with cols[2]:
            st.session_state.items[i]['don_gia'] = st.text_input(f"Đơn giá (VNĐ/chỉ)", 
                                                                    value=st.session_state.items[i].get('don_gia', ''),
                                                                    key=f"don_gia_{i}")
    
    st.markdown("---")

    if st.button("Lưu giao dịch"):
        # Lấy giá trị từ input hoặc session_state
        ho_va_ten = ho_ten_input if st.session_state.phuong_thuc == "Nhập thủ công" else st.session_state.ho_ten
        so_cccd_val = so_cccd_input if st.session_state.phuong_thuc == "Nhập thủ công" else st.session_state.so_cccd
        que_quan_val = que_quan_input if st.session_state.phuong_thuc == "Nhập thủ công" else st.session_state.que_quan
        ten_don_vi_val = st.session_state.ten_don_vi_input

        # Lấy danh sách items đã nhập
        items_list = st.session_state.items
        
        # Kiểm tra dữ liệu bắt buộc
        valid_items = [item for item in items_list if item['ten_hang'] and item['so_luong'] and item['don_gia']]
        if not ho_va_ten or not so_cccd_val or not valid_items:
             st.error("Vui lòng đảm bảo đã nhập đầy đủ Họ tên, Số CCCD và ít nhất một món hàng.")
        else:
            giao_dich_data = xu_ly_giao_dich(ho_va_ten, so_cccd_val, que_quan_val, valid_items)
            if giao_dich_data:
                st.success("Giao dịch đã được lưu thành công!")
                st.metric(label="Tổng Thành Tiền", value=f"{giao_dich_data['tong_thanh_tien']:,.0f} VNĐ")
                st.write(f"Bằng chữ: {doc_so_thanh_chu(giao_dich_data['tong_thanh_tien'])}")

                pdf_buffer = tao_pdf_mau_01(giao_dich_data, ten_don_vi_val)
                st.session_state.pdf_for_download = pdf_buffer
                st.session_state.giao_dich_data = giao_dich_data

    # Hiển thị download PDF nếu có
    if st.session_state.pdf_for_download:
        st.download_button(
            "Tải bản kê PDF (Mẫu 01/TNDN)",
            data=st.session_state.pdf_for_download.getvalue(),
            file_name=f"bang_ke_{(st.session_state.giao_dich_data['ho_va_ten']).replace(' ', '_')}.pdf",
            mime="application/pdf"
        )
    
    st.markdown("---")
    if st.button("Làm mới trang", key="refresh_button"):
        # reset keys (giữ login)
        for k in ["ho_ten", "so_cccd", "que_quan", "pdf_for_download", "giao_dich_data", "ten_don_vi", 
                  "phuong_thuc", "items", "ho_ten_input", "so_cccd_input", "que_quan_input", "ten_don_vi_input"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

def history_and_stats_page():
    st.header("Lịch sử và Thống kê")
    df = pd.read_sql_query("SELECT * FROM lich_su ORDER BY thoi_gian DESC", conn)
    
    if df.empty:
        st.info("Chưa có giao dịch nào được ghi lại.")
        return

    # Sửa tên cột
    df = df.rename(columns={
        'id': 'ID',
        'thoi_gian': 'Thời gian',
        'ho_va_ten': 'Họ và tên',
        'so_cccd': 'Số CCCD',
        'que_quan': 'Quê quán',
        'hang_hoa_json': 'Hàng hóa',
        'tong_thanh_tien': 'Thành tiền'
    })

    # Cột 'Hàng hóa' sẽ là JSON string, cần parse để hiển thị
    df['Hàng hóa'] = df['Hàng hóa'].apply(lambda x: json.loads(x) if pd.notnull(x) else [])
    
    st.subheader("Bộ lọc")
    col1, col2 = st.columns(2)
    with col1:
        ho_ten_search = st.text_input("Tìm kiếm theo tên khách hàng")
    with col2:
        cccd_search = st.text_input("Tìm kiếm theo CCCD")

    df_filtered = df.copy()
    if ho_ten_search:
        df_filtered = df_filtered[df_filtered['Họ và tên'].str.contains(ho_ten_search, case=False, na=False)]
    if ccd_search:
        df_filtered = df_filtered[df_filtered['Số CCCD'].str.contains(cccd_search, case=False, na=False)]

    st.markdown("---")
    st.subheader("Thống kê")
    col_stats1, col_stats2 = st.columns(2)
    with col_stats1:
        tong_giao_dich = len(df_filtered)
        st.metric("Tổng giao dịch", value=f"{tong_giao_dich}")
    with col_stats2:
        tong_thanh_tien = df_filtered['Thành tiền'].sum()
        st.metric("Tổng thành tiền", value=f"{tong_thanh_tien:,.0f} VNĐ")

    st.markdown("---")
    st.subheader("Biểu đồ doanh thu")
    df_filtered['Thời gian'] = pd.to_datetime(df_filtered['Thời gian'])
    df_filtered['Ngày'] = df_filtered['Thời gian'].dt.date
    daily_revenue = df_filtered.groupby('Ngày')['Thành tiền'].sum()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    daily_revenue.plot(kind='line', ax=ax, marker='o')
    
    ax.set_title("Doanh thu hàng ngày")
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Thành tiền (VNĐ)")
    
    # Định dạng trục y để hiển thị số lớn dễ đọc hơn
    formatter = plt.FuncFormatter(lambda x, p: f'{x:,.0f}')
    ax.yaxis.set_major_formatter(formatter)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("Lịch sử giao dịch")
    st.dataframe(df_filtered, hide_index=True)

    # Cho phép chọn 1 dòng để edit hoặc xóa
    st.markdown("**Chỉnh sửa / Xóa 1 bản ghi**")
    ids = df_filtered['ID'].astype(str).tolist()
    chosen = st.selectbox("Chọn ID để chỉnh sửa/xóa", [""] + ids)
    
    if chosen:
        row = df_filtered[df_filtered['ID'].astype(str) == chosen].iloc[0]
        st.markdown(f"**Đang chỉnh sửa bản ghi ID: {chosen}**")
        e_name = st.text_input("Họ và tên", value=row['Họ và tên'], key=f"edit_name_{chosen}")
        e_cccd = st.text_input("Số CCCD", value=row['Số CCCD'], key=f"edit_cccd_{chosen}")
        e_qq = st.text_area("Quê quán", value=row['Quê quán'], key=f"edit_qq_{chosen}")
        
        # Hiển thị và cho phép chỉnh sửa các món hàng
        edited_items = st.session_state.get(f"edited_items_{chosen}", row['Hàng hóa'])
        st.session_state[f"edited_items_{chosen}"] = edited_items
        
        st.subheader("Chỉnh sửa các món hàng")
        for i, item in enumerate(edited_items):
            st.markdown(f"**Món hàng {i+1}**")
            cols = st.columns([2, 1, 1])
            with cols[0]:
                item['ten'] = st.text_input(f"Tên hàng hóa", value=item.get('ten', ''), key=f"edit_ten_{chosen}_{i}")
            with cols[1]:
                item['so_luong'] = st.text_input(f"Khối lượng (chỉ)", value=str(item.get('so_luong', 0)), key=f"edit_sl_{chosen}_{i}")
            with cols[2]:
                item['don_gia'] = st.text_input(f"Đơn giá (VNĐ/chỉ)", value=str(item.get('don_gia', 0)), key=f"edit_dg_{chosen}_{i}")

        if st.button("Cập nhật bản ghi"):
            try:
                # Tính lại tổng thành tiền
                new_items = st.session_state[f"edited_items_{chosen}"]
                new_tong_tien = 0
                for item in new_items:
                    new_tong_tien += float(item.get('so_luong', 0)) * float(item.get('don_gia', 0))
                
                # Chuyển về JSON
                new_items_json = json.dumps(new_items)
                
                c.execute('''
                    UPDATE lich_su
                    SET ho_va_ten=?, so_cccd=?, que_quan=?, hang_hoa_json=?, tong_thanh_tien=?
                    WHERE id=?
                ''', (e_name, e_cccd, e_qq, new_items_json, new_tong_tien, int(chosen)))
                conn.commit()
                st.success("Cập nhật thành công.")
                st.experimental_rerun()
            except Exception as ex:
                st.error(f"Lỗi cập nhật: {ex}")

        if st.button("Xóa bản ghi"):
            try:
                c.execute('DELETE FROM lich_su WHERE id=?', (int(chosen),))
                conn.commit()
                st.success("Đã xóa bản ghi.")
                st.experimental_rerun()
            except Exception as ex:
                st.error(f"Lỗi xóa: {ex}")

    csv_file = df_filtered.to_csv(index=False)
    st.download_button(label="Tải xuống CSV", data=csv_file, file_name='lich_su_giao_dich.csv', mime='text/csv')

# --- Chạy ứng dụng ---
if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if st.session_state.logged_in:
        main_app()
    else:
        login_page()
