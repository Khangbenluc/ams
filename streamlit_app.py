# app.py
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
import pytz
from paddleocr import PaddleOCR
import io
import locale

# cố gắng import reportlab, nếu không có thì sẽ xuất HTML thay PDF
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# ---- Cấu hình ----
st.set_page_config(layout="wide", page_title="Bảng kê 01/TNDN", page_icon="📄")
locale.setlocale(locale.LC_ALL, '')  # dùng để format số theo locale nếu khả dụng

# ---- File lưu lịch sử ----
lich_su_file = 'lich_su_giao_dich.csv'
if not os.path.exists(lich_su_file):
    df = pd.DataFrame(columns=['Thời gian', 'Họ và Tên', 'Số CCCD', 'Quê quán', 'Khối lượng', 'Đơn giá', 'Thành tiền'])
    df.to_csv(lich_su_file, index=False)

# ---- Khởi tạo OCR (PaddleOCR) ----
@st.cache_resource
def get_reader():
    # language vi (tiếng Việt)
    return PaddleOCR(lang="vi", use_angle_cls=False)

ocr = get_reader()

# ---- Hàm tiện ích ----
def to_number_str(x):
    """Chuyển chuỗi có dấu phẩy/thừa thành số thô (float)"""
    if x is None:
        return 0.0
    s = str(x).strip()
    s = s.replace(',', '').replace(' ', '')
    try:
        return float(s)
    except:
        return 0.0

# Hàm chuyển số sang chữ (VNĐ) - bản rút gọn đủ dùng
dv_words = ['không','một','hai','ba','bốn','năm','sáu','bảy','tám','chín']
def read3(n):
    s=''; n = int(n)
    tr, ch, d = n//100, (n%100)//10, n%10
    if tr>0:
        s += dv_words[tr]+' trăm'
        if ch==0 and d>0: s+=' linh'
    if ch>1:
        s += (' ' if s else '')+dv_words[ch]+' mươi'
        if d==1: s+=' mốt'
        elif d==5: s+=' lăm'
        elif d>0: s+=' '+dv_words[d]
    elif ch==1:
        s += (' ' if s else '')+'mười'
        if d==5: s+=' lăm'
        elif d>0: s+=' '+dv_words[d]
    elif ch==0 and d>0:
        s += (' ' if s else '')+dv_words[d]
    return s.strip()

def to_words_vnd(num):
    num = int(round(num))
    if num <= 0:
        return "Không đồng"
    unit = ['',' nghìn',' triệu',' tỷ',' nghìn tỷ',' triệu tỷ']
    out=[]; i=0
    while num>0 and i<len(unit):
        chunk = num % 1000
        if chunk>0:
            out.insert(0, (read3(chunk) + unit[i]).strip())
        num //= 1000
        i += 1
    s = ' '.join(out).strip()
    s = s[0].upper() + s[1:] + ' đồng'
    return s

def format_money(v):
    try:
        return f"{int(round(v)):,}".replace(',', '.')
    except:
        return "0"

# ---- OCR trích xuất thông tin từ CCCD và cân ----
def trich_xuat_cccd(image_path):
    """Trích họ tên, số CCCD, quê quán từ ảnh CCCD (có thể không hoàn hảo)."""
    if image_path is None or not os.path.exists(image_path):
        return "", "", ""
    img = cv2.imread(image_path)
    result = ocr.ocr(img, cls=False)
    if not result or not result[0]:
        return "", "", ""
    ho_ten, so_cccd, que_quan = "", "", ""
    # Duyệt từng dòng OCR
    lines = [line for line in result[0]]
    for i, line in enumerate(lines):
        text = line[1][0].strip()
        up = text.upper()
        # Họ và tên
        if "HỌ VÀ TÊN" in up or "HỌ TÊN" in up:
            # thường tên sẽ ở dòng tiếp theo
            if i+1 < len(lines):
                ho_ten = lines[i+1][1][0].strip()
        # Số CCCD (12 chữ số)
        if "SỐ" in up and any(tok.isdigit() and len(tok)==12 for tok in up.split()):
            for tok in up.split():
                if tok.isdigit() and len(tok)==12:
                    so_cccd = tok
                    break
        # Quê quán
        if "QUÊ QUÁN" in up:
            if i+1 < len(lines):
                que_quan = lines[i+1][1][0].strip()
    # fallback: nếu chưa có số cccd, thử tìm ở mọi dòng
    if not so_cccd:
        for line in lines:
            txt = line[1][0]
            digits = ''.join(ch for ch in txt if ch.isdigit())
            if len(digits) >= 12:
                so_cccd = digits[:12]
                break
    return ho_ten, so_cccd, que_quan

def trich_xuat_can(image_path):
    """Trích giá trị số từ ảnh cân (lấy chuỗi số đầu tiên có xuất hiện)."""
    if image_path is None or not os.path.exists(image_path):
        return ""
    img = cv2.imread(image_path)
    result = ocr.ocr(img, cls=False)
    if not result or not result[0]:
        return ""
    for line in result[0]:
        text = line[1][0]
        cleaned = ''.join(c for c in text if c.isdigit() or c in '.,')
        # chọn token có chữ số
        if any(ch.isdigit() for ch in cleaned):
            # loại bỏ dấu thừa
            cleaned = cleaned.replace(',', '.')
            # giữ 1 dấu chấm
            parts = cleaned.split('.')
            if len(parts) > 2:
                cleaned = ''.join(parts[:-1]) + '.' + parts[-1]
            # trả về
            return cleaned
    return ""

# ---- Hàm tạo PDF theo mẫu 01/TNDN (một bản kê cho 1 giao dịch) ----
def create_pdf_bytes(row_dict):
    """
    row_dict cần chứa các trường:
      don_vi, mst, dia_chi, dia_diem, phu_trach, ngay_lap (dd/mm/yyyy),
      ho_va_ten, so_cccd, que_quan, so_luong, don_gia, thanh_tien
    """
    buffer = io.BytesIO()
    if not REPORTLAB_OK:
        return None  # caller sẽ fallback sang HTML
    # đăng ký font hỗ trợ tiếng Việt (DejaVuSans)
    try:
        pdfmetrics.registerFont(TTFont('DejaVu', 'DejaVuSans.ttf'))
        font_name = 'DejaVu'
    except Exception:
        # nếu không tìm thấy DejaVu, dùng font mặc định
        font_name = 'Helvetica'

    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    left_margin = 18 * mm
    right_margin = 18 * mm
    cur_y = height - 18 * mm

    c.setFont(font_name, 10)
    c.drawString(left_margin, cur_y, "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM")
    c.drawRightString(width - right_margin, cur_y, "Mẫu số: 01/TNDN")
    cur_y -= 12
    c.setFont(font_name, 9)
    c.drawString(left_margin, cur_y, "Độc lập - Tự do - Hạnh phúc")
    c.setFont(font_name, 9)
    c.drawRightString(width - right_margin, cur_y, "(Ban hành kèm theo Thông tư 78/2014/TT-BTC)")
    cur_y -= 20

    c.setFont(font_name, 13)
    c.drawCentredString(width / 2, cur_y, "BẢNG KÊ THU MUA HÀNG HÓA, DỊCH VỤ MUA VÀO KHÔNG CÓ HÓA ĐƠN")
    cur_y -= 18

    c.setFont(font_name, 10)
    # Thông tin đơn vị
    c.drawString(left_margin, cur_y, f"Đơn vị: {row_dict.get('don_vi','')}")
    cur_y -= 12
    c.drawString(left_margin, cur_y, f"Mã số thuế: {row_dict.get('mst','')}")
    cur_y -= 12
    c.drawString(left_margin, cur_y, f"Địa chỉ: {row_dict.get('dia_chi','')}")
    cur_y -= 16

    # Thông tin thu mua
    c.drawString(left_margin, cur_y, f"Địa điểm thu mua: {row_dict.get('dia_diem','')}")
    cur_y -= 12
    c.drawString(left_margin, cur_y, f"Người phụ trách: {row_dict.get('phu_trach','')}")
    cur_y -= 12
    c.drawString(left_margin, cur_y, f"Ngày lập bảng kê: {row_dict.get('ngay_lap','')}")
    cur_y -= 18

    # Thông tin người bán
    c.setFont(font_name, 11)
    c.drawString(left_margin, cur_y, "Thông tin người bán:")
    cur_y -= 14
    c.setFont(font_name, 10)
    c.drawString(left_margin + 6*mm, cur_y, f"Họ và tên: {row_dict.get('ho_va_ten','')}")
    cur_y -= 12
    c.drawString(left_margin + 6*mm, cur_y, f"Số CCCD/CMND: {row_dict.get('so_cccd','')}")
    cur_y -= 12
    c.drawString(left_margin + 6*mm, cur_y, f"Quê quán: {row_dict.get('que_quan','')}")
    cur_y -= 18

    # Bảng chi tiết (dạng đơn hàng 1 dòng)
    c.setFont(font_name, 10)
    table_x = left_margin
    table_w = width - left_margin - right_margin
    # vẽ header
    headers = ["STT", "Tên hàng/ dịch vụ", "ĐVT", "Số lượng", "Đơn giá (VNĐ)", "Thành tiền (VNĐ)"]
    col_w = [20*mm, 70*mm, 20*mm, 30*mm, 35*mm, 45*mm]
    # header background box
    y_top = cur_y
    # draw header texts
    x = table_x
    c.setFont(font_name, 9)
    for i, h in enumerate(headers):
        c.rect(x, cur_y - 14, col_w[i], 16, stroke=1, fill=0)
        c.drawCentredString(x + col_w[i]/2, cur_y - 10, h)
        x += col_w[i]
    cur_y -= 18

    # one row content
    x = table_x
    c.rect(x, cur_y - 12, col_w[0], 14, stroke=1, fill=0)
    c.drawCentredString(x + col_w[0]/2, cur_y - 8, "1")
    x += col_w[0]

    c.rect(x, cur_y - 12, col_w[1], 14, stroke=1, fill=0)
    c.drawString(x + 4, cur_y - 10, row_dict.get('mieu_ta','Hàng hóa'))
    x += col_w[1]

    c.rect(x, cur_y - 12, col_w[2], 14, stroke=1, fill=0)
    c.drawCentredString(x + col_w[2]/2, cur_y - 8, row_dict.get('don_vi',''))
    x += col_w[2]

    c.rect(x, cur_y - 12, col_w[3], 14, stroke=1, fill=0)
    c.drawCentredString(x + col_w[3]/2, cur_y - 8, f"{row_dict.get('so_luong',0)}")
    x += col_w[3]

    c.rect(x, cur_y - 12, col_w[4], 14, stroke=1, fill=0)
    c.drawRightString(x + col_w[4] - 4, cur_y - 8, format_money(row_dict.get('don_gia',0)))
    x += col_w[4]

    c.rect(x, cur_y - 12, col_w[5], 14, stroke=1, fill=0)
    c.drawRightString(x + col_w[5] - 4, cur_y - 8, format_money(row_dict.get('thanh_tien',0)))
    cur_y -= 26

    # Tổng cộng & bằng chữ
    c.setFont(font_name, 10)
    c.drawRightString(width - right_margin, cur_y, "Tổng cộng: " + format_money(row_dict.get('thanh_tien',0)) + " VNĐ")
    cur_y -= 14
    c.drawString(left_margin, cur_y, "Số tiền bằng chữ: " + to_words_vnd(row_dict.get('thanh_tien',0)))
    cur_y -= 28

    # Nơi ký
    c.drawString(left_margin, cur_y, f"{row_dict.get('dia_diem','')}, ngày {row_dict.get('ngay_lap','')}")
    c.drawString(left_margin + 6*mm, cur_y - 18, "Người lập bảng kê")
    c.drawString(width/2, cur_y - 18, "Người bán")
    c.drawString(width - right_margin - 80*mm, cur_y - 18, "Thủ trưởng đơn vị")
    # leave space for signatures
    c.line(left_margin, cur_y - 60, left_margin + 60*mm, cur_y - 60)
    c.line(width/2, cur_y - 60, width/2 + 60*mm, cur_y - 60)
    c.line(width - right_margin - 80*mm, cur_y - 60, width - right_margin + 10*mm, cur_y - 60)

    c.save()
    buffer.seek(0)
    return buffer.read()

# ---- Hàm tạo HTML (dự phòng nếu reportlab không có) ----
def create_html(row_dict):
    html = f"""
<html>
<head>
<meta charset="utf-8">
<title>Bảng kê 01/TNDN</title>
<style>
body{{font-family: Arial, Helvetica, sans-serif; padding:20px; color:#111}}
h2{{text-align:center}}
.table{{width:100%;border-collapse:collapse;margin-top:10px}}
.table, .table th, .table td{{border:1px solid #ddd}}
.table th, .table td{{padding:6px;text-align:left}}
.right{{text-align:right}}
.mono{{font-family:monospace}}
</style>
</head>
<body>
<p><strong>CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM</strong> <span style="float:right">Mẫu số: 01/TNDN</span></p>
<p><em>Độc lập - Tự do - Hạnh phúc</em></p>
<h2>BẢNG KÊ THU MUA HÀNG HÓA, DỊCH VỤ MUA VÀO KHÔNG CÓ HÓA ĐƠN</h2>
<p><strong>Đơn vị:</strong> {row_dict.get('don_vi','')} &nbsp;&nbsp; <strong>MST:</strong> {row_dict.get('mst','')}</p>
<p><strong>Địa điểm thu mua:</strong> {row_dict.get('dia_diem','')} &nbsp;&nbsp; <strong>Người phụ trách:</strong> {row_dict.get('phu_trach','')}</p>
<p><strong>Ngày lập:</strong> {row_dict.get('ngay_lap','')}</p>

<table class="table">
<thead>
<tr><th>STT</th><th>Tên hàng/dịch vụ</th><th>ĐVT</th><th>Số lượng</th><th>Đơn giá (VNĐ)</th><th>Thành tiền (VNĐ)</th></tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>{row_dict.get('mieu_ta','Hàng hóa')}</td>
<td>{row_dict.get('don_vi','')}</td>
<td class="right">{row_dict.get('so_luong',0)}</td>
<td class="right">{format_money(row_dict.get('don_gia',0))}</td>
<td class="right">{format_money(row_dict.get('thanh_tien',0))}</td>
</tr>
</tbody>
</table>

<p class="right"><strong>Tổng cộng: {format_money(row_dict.get('thanh_tien',0))} VNĐ</strong></p>
<p><strong>Số tiền bằng chữ:</strong> {to_words_vnd(row_dict.get('thanh_tien',0))}</p>

<p>{row_dict.get('dia_diem','')}, ngày {row_dict.get('ngay_lap','')}</p>
<table style="width:100%; margin-top:40px; border:none">
<tr>
<td style="width:33%; text-align:center">Người lập bảng kê<br>(Ký, ghi rõ họ tên)</td>
<td style="width:33%; text-align:center">Người bán<br/>(Ký, ghi rõ họ tên)</td>
<td style="width:33%; text-align:center">Thủ trưởng đơn vị<br/>(Ký, đóng dấu)</td>
</tr>
</table>

</body>
</html>
"""
    return html.encode('utf-8')

# ---- Hàm xử lý giao dịch (giá trị trả về: dict + thanh_tien) ----
def xu_ly_giao_dich_save(ho_va_ten, so_cccd, que_quan, so_luong_str, don_gia_str,
                         don_vi_unit='chỉ', mieu_ta='Hàng hóa', dia_diem='Bến Lức',
                         don_vi_name='', mst='', dia_chi='', phu_trach=''):
    try:
        so_luong = float(str(so_luong_str).replace(',', '.'))
        don_gia = float(str(don_gia_str).replace(',', '').replace('.', '')) if isinstance(don_gia_str, str) and ',' in don_gia_str else float(str(don_gia_str).replace(',', ''))
        # cố gắng xử lý nếu người dùng nhập với dấu chấm phân cách hàng nghìn
        thanh_tien = so_luong * don_gia

        vn_timezone = pytz.timezone('Asia/Ho_Chi_Minh')
        current_time = datetime.now(vn_timezone)
        ngay_tao_display = current_time.strftime("%d/%m/%Y")
        thoi_gian_luu = current_time.strftime("%Y-%m-%d %H:%M:%S")

        # Lưu lịch sử
        df_moi = pd.DataFrame([{
            'Thời gian': thoi_gian_luu,
            'Họ và Tên': ho_va_ten,
            'Số CCCD': so_cccd,
            'Quê quán': que_quan,
            'Khối lượng': so_luong,
            'Đơn giá': don_gia,
            'Thành tiền': thanh_tien
        }])
        df_moi.to_csv(lich_su_file, mode='a', header=False, index=False)

        # Chuẩn bị dict cho in
        row = {
            'don_vi': don_vi_name or '',
            'mst': mst or '',
            'dia_chi': dia_chi or '',
            'dia_diem': dia_diem,
            'phu_trach': phu_trach,
            'ngay_lap': ngay_tao_display,
            'ho_va_ten': ho_va_ten,
            'so_cccd': so_cccd,
            'que_quan': que_quan,
            'so_luong': so_luong,
            'don_gia': don_gia,
            'thanh_tien': thanh_tien,
            'don_vi_unit': don_vi_unit,
            'mieu_ta': mieu_ta,
            'don_vi_unit_text': don_vi_unit
        }

        return row, thanh_tien
    except Exception as e:
        return None, str(e)

# ---- Giao diện Streamlit ----
st.title("ỨNG DỤNG TẠO BẢN KÊ 01/TNDN (OCR CCCD + Cân)")
st.write("Nhập hoặc chụp ảnh CCCD và ảnh cân → tạo bản kê theo mẫu 01/TNDN và xuất PDF/HTML để in.")

# Thông tin đơn vị (có thể để trống)
with st.expander("Thông tin đơn vị (tùy chọn)"):
    don_vi_name = st.text_input("Tên đơn vị", value="")
    mst = st.text_input("Mã số thuế", value="")
    dia_chi = st.text_input("Địa chỉ trụ sở", value="")
    dia_diem = st.text_input("Địa điểm thu mua", value="Bến Lức")
    phu_trach = st.text_input("Người phụ trách thu mua", value="")

st.markdown("---")
# Session state khởi tạo
if 'ho_ten' not in st.session_state: st.session_state.ho_ten = ""
if 'so_cccd' not in st.session_state: st.session_state.so_cccd = ""
if 'que_quan' not in st.session_state: st.session_state.que_quan = ""
if 'so_luong' not in st.session_state: st.session_state.so_luong = ""

st.header("1. Thông tin người bán (khách hàng)")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Trích xuất từ CCCD")
    anh_cccd = st.file_uploader("Tải ảnh CCCD (JPG/PNG) hoặc chụp", type=['jpg','jpeg','png'])
    if anh_cccd:
        tmp_path = "temp_cccd.jpg"
        with open(tmp_path, "wb") as f:
            f.write(anh_cccd.getbuffer())
        ho, so, que = trich_xuat_cccd(tmp_path)
        # gán session state
        st.session_state.ho_ten = ho or st.session_state.ho_ten
        st.session_state.so_cccd = so or st.session_state.so_cccd
        st.session_state.que_quan = que or st.session_state.que_quan
        os.remove(tmp_path)

with col2:
    st.subheader("Nhập thủ công (hoặc chỉnh sửa)")
    ho_va_ten = st.text_input("Họ và Tên", value=st.session_state.ho_ten)
    so_cccd = st.text_input("Số CCCD/CMND", value=st.session_state.so_cccd)
    que_quan = st.text_input("Quê quán", value=st.session_state.que_quan)

st.markdown("---")
st.header("2. Thông tin giao dịch")
col3, col4 = st.columns(2)
with col3:
    st.subheader("Trích xuất từ cân (ảnh)")
    anh_can = st.file_uploader("Tải ảnh cân (hoặc chụp màn hình)", type=['jpg','jpeg','png'], key="can")
    if anh_can:
        tmp_path2 = "temp_can.jpg"
        with open(tmp_path2, "wb") as f:
            f.write(anh_can.getbuffer())
        so_luong_ex = trich_xuat_can(tmp_path2)
        st.session_state.so_luong = so_luong_ex or st.session_state.so_luong
        os.remove(tmp_path2)

with col4:
    st.subheader("Nhập thủ công")
    so_luong_input = st.text_input("Khối lượng (ví dụ: 1.0, 2.5)", value=str(st.session_state.so_luong))
    don_gia_input = st.text_input("Đơn giá (VNĐ)", value="1000000")
    don_vi_unit = st.text_input("Đơn vị tính (ví dụ: chỉ, kg)", value="chỉ")
    mieu_ta = st.text_input("Mô tả hàng (VD: Vàng miếng...)", value="Hàng hóa")

st.markdown("---")
st.header("3. Tạo bản kê (theo mẫu 01/TNDN)")

if st.button("Tính tiền và Tạo bản kê"):
    # kiểm tra dữ liệu
    if not ho_va_ten.strip():
        st.error("Vui lòng nhập Họ và Tên.")
    elif not so_luong_input.strip() or not don_gia_input.strip():
        st.error("Vui lòng nhập Khối lượng và Đơn giá.")
    else:
        row, thanh_tien_or_err = xu_ly_giao_dich_save(
            ho_va_ten.strip(), so_cccd.strip(), que_quan.strip(),
            so_luong_input.strip(), don_gia_input.strip(),
            don_vi_unit=don_vi_unit.strip(), mieu_ta=mieu_ta.strip(),
            dia_diem=dia_diem.strip(), don_vi_name=don_vi_name.strip(),
            mst=mst.strip(), dia_chi=dia_chi.strip(), phu_trach=phu_trach.strip()
        )
        if row is None:
            st.error(f"Lỗi khi xử lý: {thanh_tien_or_err}")
        else:
            st.success("Đã lưu giao dịch vào lịch sử.")
            # Hiển thị bản kê dạng text
            txt = f"""
CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc
-----------------------------

MẪU SỐ 01/TNDN
BẢNG KÊ THU MUA HÀNG HÓA, DỊCH VỤ MUA VÀO KHÔNG CÓ HÓA ĐƠN

Đơn vị: {row.get('don_vi','')}
Mã số thuế: {row.get('mst','')}
Địa chỉ: {row.get('dia_chi','')}

Địa điểm thu mua: {row.get('dia_diem','')}
Người phụ trách: {row.get('phu_trach','')}
Ngày lập bảng kê: {row.get('ngay_lap','')}

THÔNG TIN NGƯỜI BÁN:
- Họ và tên: {row.get('ho_va_ten','')}
- Số CCCD: {row.get('so_cccd','')}
- Quê quán: {row.get('que_quan','')}

CHI TIẾT GIAO DỊCH:
- Khối lượng: {row.get('so_luong',0)} {don_vi_unit}
- Đơn giá: {format_money(row.get('don_gia',0))} VNĐ/{don_vi_unit}
- Thành tiền: {format_money(row.get('thanh_tien',0))} VNĐ

Tổng cộng: {format_money(row.get('thanh_tien',0))} VNĐ
Số tiền bằng chữ: {to_words_vnd(row.get('thanh_tien',0))}

{row.get('dia_diem','')}, ngày {row.get('ngay_lap','')}
Người lập bảng kê: (Ký, ghi rõ họ tên)      Người bán: (Ký, ghi rõ họ tên)
"""
            st.code(txt, language='text')

            # Tạo PDF (nếu có reportlab), hoặc HTML fallback
            pdf_bytes = None
            if REPORTLAB_OK:
                try:
                    pdf_bytes = create_pdf_bytes(row)
                except Exception as e:
                    st.warning(f"Tạo PDF thất bại (reportlab): {e}")
                    pdf_bytes = None

            if pdf_bytes:
                st.success("Tạo PDF thành công.")
                st.download_button("Tải PDF Bảng Kê (01/TNDN)", data=pdf_bytes, file_name=f"bang_ke_01_TNDN_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")
            else:
                # Tạo HTML và cho tải
                html_bytes = create_html(row)
                st.info("Không thể tạo PDF tự động (reportlab không có hoặc lỗi). Tải HTML để in từ trình duyệt.")
                st.download_button("Tải file HTML (In từ trình duyệt -> Save as PDF)", data=html_bytes, file_name=f"bang_ke_01_TNDN_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html", mime="text/html")

st.markdown("---")
st.header("4. Lịch sử giao dịch")
if os.path.exists(lich_su_file):
    try:
        df_lich_su = pd.read_csv(lich_su_file)
        st.dataframe(df_lich_su.sort_values(by='Thời gian', ascending=False).reset_index(drop=True))
    except Exception as e:
        st.error("Không thể đọc lịch sử giao dịch: " + str(e))
else:
    st.info("Chưa có giao dịch nào.")
