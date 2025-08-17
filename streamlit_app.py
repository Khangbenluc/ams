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
import base64

# cố gắng import reportlab (PDF). Nếu không có, báo và fallback sang HTML.
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# --- Cấu hình ---
st.set_page_config(page_title="Bảng kê 01/TNDN", layout="wide", page_icon="📄")

# File lưu lịch sử
LICH_SU_FILE = "lich_su_giao_dich.csv"
if not os.path.exists(LICH_SU_FILE):
    df_empty = pd.DataFrame(columns=[
        "Thời gian", "Đơn vị bán hàng", "MST", "Địa chỉ đơn vị",
        "Địa điểm thu mua", "Người phụ trách",
        "Họ và Tên", "Số CCCD", "Quê quán",
        "Khối lượng", "Đơn vị tính", "Đơn giá", "Thành tiền"
    ])
    df_empty.to_csv(LICH_SU_FILE, index=False)

# --- OCR init ---
@st.cache_resource
def get_ocr():
    # PaddleOCR tiếng Việt
    return PaddleOCR(lang="vi", use_angle_cls=False)
ocr = get_ocr()

# --- Tiện ích ---
def img_from_upload(uploaded_file):
    if uploaded_file is None:
        return None
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img

# chuyển số thành chữ (VNĐ) - đủ dùng
dv_words = ['không','một','hai','ba','bốn','năm','sáu','bảy','tám','chín']
def read3(n):
    s=''; n=int(n)
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
    units = ['',' nghìn',' triệu',' tỷ',' nghìn tỷ',' triệu tỷ']
    out=[]; i=0
    while num>0 and i < len(units):
        chunk = num % 1000
        if chunk>0:
            out.insert(0, (read3(chunk) + units[i]).strip())
        num//=1000; i+=1
    s = ' '.join(out).strip()
    s = s[0].upper()+s[1:]+' đồng'
    return s

def fmt_money(v):
    try:
        return f"{int(round(v)):,}".replace(',', '.')
    except:
        return "0"

# --- OCR trích xuất ---
def trich_xuat_cccd_from_image(img):
    """Trả về ho_ten, so_cccd, que_quan (có thể rỗng nếu không tìm được)."""
    try:
        res = ocr.ocr(img, cls=False)
    except Exception:
        return "", "", ""
    if not res or not res[0]:
        return "", "", ""
    lines = res[0]
    ho_ten = so_cccd = que_quan = ""
    for i, ln in enumerate(lines):
        text = ln[1][0].strip()
        up = text.upper()
        if "HỌ VÀ TÊN" in up or "HỌ TÊN" in up:
            if i+1 < len(lines):
                ho_ten = lines[i+1][1][0].strip()
        if "SỐ" in up and any(tok.isdigit() and len(tok)==12 for tok in up.split()):
            for tok in up.split():
                if tok.isdigit() and len(tok)==12:
                    so_cccd = tok
                    break
        if "QUÊ QUÁN" in up:
            if i+1 < len(lines):
                que_quan = lines[i+1][1][0].strip()
    # fallback: tìm token 12 chữ số ở bất kỳ dòng nào
    if not so_cccd:
        for ln in lines:
            txt = ln[1][0]
            digits = ''.join(ch for ch in txt if ch.isdigit())
            if len(digits) >= 12:
                so_cccd = digits[:12]; break
    return ho_ten, so_cccd, que_quan

def trich_xuat_can_from_image(img):
    """Trích số từ ảnh cân (lấy chuỗi số/decimal đầu tiên)."""
    try:
        res = ocr.ocr(img, cls=False)
    except Exception:
        return ""
    if not res or not res[0]:
        return ""
    for ln in res[0]:
        txt = ln[1][0]
        cleaned = ''.join(ch for ch in txt if ch.isdigit() or ch in '.,')
        if any(ch.isdigit() for ch in cleaned):
            cleaned = cleaned.replace(',', '.')
            parts = cleaned.split('.')
            if len(parts) > 2:
                cleaned = ''.join(parts[:-1]) + '.' + parts[-1]
            return cleaned
    return ""

# --- Tạo PDF (reportlab) ---
def create_pdf_bytes(row):
    """
    row: dict gồm các trường cần thiết.
    Trả về bytes PDF hoặc None nếu lỗi / không có reportlab.
    """
    if not REPORTLAB_OK:
        return None
    # đăng ký font DejaVu nếu có (hỗ trợ tiếng Việt)
    try:
        pdfmetrics.registerFont(TTFont('DejaVu', 'DejaVuSans.ttf'))
        font_name = 'DejaVu'
    except Exception:
        font_name = 'Helvetica'

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4
    left = 18*mm
    right = 18*mm
    cur_y = h - 18*mm

    # Header
    c.setFont(font_name, 10)
    c.drawString(left, cur_y, "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM")
    c.drawRightString(w - right, cur_y, "Mẫu số: 01/TNDN")
    cur_y -= 12
    c.setFont(font_name, 9)
    c.drawString(left, cur_y, "Độc lập - Tự do - Hạnh phúc")
    c.drawRightString(w - right, cur_y, "(Ban hành kèm theo Thông tư 78/2014/TT-BTC)")
    cur_y -= 18

    c.setFont(font_name, 13)
    c.drawCentredString(w/2, cur_y, "BẢNG KÊ THU MUA HÀNG HÓA, DỊCH VỤ MUA VÀO KHÔNG CÓ HÓA ĐƠN")
    cur_y -= 20

    # Thông tin đơn vị (nếu có)
    c.setFont(font_name, 10)
    if row.get('don_vi'):
        c.drawString(left, cur_y, f"Đơn vị: {row.get('don_vi')}")
        cur_y -= 12
    if row.get('mst'):
        c.drawString(left, cur_y, f"MST: {row.get('mst')}")
        cur_y -= 12
    if row.get('dia_chi'):
        c.drawString(left, cur_y, f"Địa chỉ: {row.get('dia_chi')}")
        cur_y -= 12
    cur_y -= 6

    # Thông tin thu mua
    c.drawString(left, cur_y, f"Địa điểm thu mua: {row.get('dia_diem','')}")
    cur_y -= 12
    c.drawString(left, cur_y, f"Người phụ trách: {row.get('phu_trach','')}")
    cur_y -= 12
    c.drawString(left, cur_y, f"Ngày lập bảng kê: {row.get('ngay_lap','')}")
    cur_y -= 18

    # Thông tin người bán
    c.setFont(font_name, 11)
    c.drawString(left, cur_y, "Thông tin người bán:")
    cur_y -= 14
    c.setFont(font_name, 10)
    c.drawString(left+6*mm, cur_y, f"Họ và tên: {row.get('ho_va_ten','')}")
    cur_y -= 12
    c.drawString(left+6*mm, cur_y, f"Số CCCD/CMND: {row.get('so_cccd','')}")
    cur_y -= 12
    c.drawString(left+6*mm, cur_y, f"Quê quán: {row.get('que_quan','')}")
    cur_y -= 18

    # Bảng chi tiết (1 dòng)
    headers = ["STT","Tên hàng/dịch vụ","ĐVT","Số lượng","Đơn giá (VNĐ)","Thành tiền (VNĐ)"]
    col_w = [18*mm, 70*mm, 20*mm, 26*mm, 35*mm, 40*mm]
    x = left
    c.setFont(font_name, 9)
    for i, h in enumerate(headers):
        c.rect(x, cur_y-14, col_w[i], 16, stroke=1, fill=0)
        c.drawCentredString(x + col_w[i]/2, cur_y-10, h)
        x += col_w[i]
    cur_y -= 18

    # Row
    x = left
    c.rect(x, cur_y-12, col_w[0], 14, stroke=1)
    c.drawCentredString(x + col_w[0]/2, cur_y-8, "1"); x += col_w[0]

    c.rect(x, cur_y-12, col_w[1], 14, stroke=1)
    c.drawString(x+4, cur_y-10, row.get('mieu_ta','Hàng hóa')); x += col_w[1]

    c.rect(x, cur_y-12, col_w[2], 14, stroke=1)
    c.drawCentredString(x+col_w[2]/2, cur_y-8, row.get('don_vi_unit','')); x += col_w[2]

    c.rect(x, cur_y-12, col_w[3], 14, stroke=1)
    c.drawCentredString(x+col_w[3]/2, cur_y-8, str(row.get('so_luong',''))); x += col_w[3]

    c.rect(x, cur_y-12, col_w[4], 14, stroke=1)
    c.drawRightString(x+col_w[4]-4, cur_y-8, fmt_money(row.get('don_gia',0))); x += col_w[4]

    c.rect(x, cur_y-12, col_w[5], 14, stroke=1)
    c.drawRightString(x+col_w[5]-4, cur_y-8, fmt_money(row.get('thanh_tien',0)))
    cur_y -= 28

    # Tổng và bằng chữ
    c.drawRightString(w - right, cur_y, "Tổng cộng: " + fmt_money(row.get('thanh_tien',0)) + " VNĐ")
    cur_y -= 14
    c.drawString(left, cur_y, "Số tiền bằng chữ: " + to_words_vnd(row.get('thanh_tien',0)))
    cur_y -= 28

    # Chữ ký
    c.drawString(left, cur_y, f"{row.get('dia_diem','')}, ngày {row.get('ngay_lap','')}")
    c.drawString(left+6*mm, cur_y-18, "Người lập bảng kê")
    c.drawString(w/2, cur_y-18, "Người bán")
    c.drawString(w - right - 80*mm, cur_y-18, "Thủ trưởng đơn vị")
    # lines for signature
    c.line(left, cur_y-60, left+60*mm, cur_y-60)
    c.line(w/2, cur_y-60, w/2+60*mm, cur_y-60)
    c.line(w - right - 80*mm, cur_y-60, w - right + 10*mm, cur_y-60)

    c.showPage()
    c.save()
    buffer = buffer = io.BytesIO()
    buffer.seek(0)
    return buffer.getvalue()

# --- Tạo HTML fallback ---
def create_html(row):
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Bảng kê 01/TNDN</title>
<style>
body{{font-family:Arial, Helvetica, sans-serif; color:#111; padding:16px}}
h2{{text-align:center}}
.table{{width:100%;border-collapse:collapse;margin-top:8px}}
.table, .table th, .table td{{border:1px solid #ddd}}
.table th, .table td{{padding:6px}}
.right{{text-align:right}}
.mono{{font-family:monospace}}
</style>
</head><body>
<p><strong>CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM</strong> <span style="float:right">Mẫu số: 01/TNDN</span></p>
<p><em>Độc lập - Tự do - Hạnh phúc</em></p>
<h2>BẢNG KÊ THU MUA HÀNG HÓA, DỊCH VỤ MUA VÀO KHÔNG CÓ HÓA ĐƠN</h2>
{('<p><strong>Đơn vị:</strong> '+row.get('don_vi','')+' &nbsp;&nbsp; <strong>MST:</strong> '+row.get('mst','')+'</p>') if row.get('don_vi') else ''}
<p><strong>Địa điểm thu mua:</strong> {row.get('dia_diem','')} &nbsp;&nbsp; <strong>Người phụ trách:</strong> {row.get('phu_trach','')}</p>
<p><strong>Ngày lập:</strong> {row.get('ngay_lap','')}</p>
<table class="table">
<thead><tr><th>STT</th><th>Tên hàng/dịch vụ</th><th>ĐVT</th><th>Số lượng</th><th>Đơn giá (VNĐ)</th><th>Thành tiền (VNĐ)</th></tr></thead>
<tbody>
<tr><td>1</td><td>{row.get('mieu_ta','Hàng hóa')}</td><td>{row.get('don_vi_unit','')}</td><td class="right">{row.get('so_luong','')}</td><td class="right">{fmt_money(row.get('don_gia',0))}</td><td class="right">{fmt_money(row.get('thanh_tien',0))}</td></tr>
</tbody>
</table>
<p class="right"><strong>Tổng cộng: {fmt_money(row.get('thanh_tien',0))} VNĐ</strong></p>
<p><strong>Số tiền bằng chữ:</strong> {to_words_vnd(row.get('thanh_tien',0))}</p>
<p>{row.get('dia_diem','')}, ngày {row.get('ngay_lap','')}</p>
<table style="width:100%;border:none;margin-top:40px"><tr><td style="text-align:center">Người lập bảng kê<br>(Ký, ghi rõ họ tên)</td><td style="text-align:center">Người bán<br>(Ký, ghi rõ họ tên)</td><td style="text-align:center">Thủ trưởng đơn vị<br>(Ký, đóng dấu)</td></tr></table>
</body></html>
"""
    return html.encode('utf-8')

# --- Xử lý và lưu giao dịch ---
def process_and_save(ho_va_ten, so_cccd, que_quan, so_luong_str, don_gia_str,
                     don_vi_unit='chỉ', mieu_ta='Hàng hóa',
                     don_vi_name='', mst='', dia_chi='', dia_diem='Bến Lức', phu_trach=''):
    try:
        so_luong = float(str(so_luong_str).replace(',', '.'))
    except:
        raise ValueError("Khối lượng không hợp lệ")
    # xử lý don_gia (loại dấu chấm/phẩy phân tách)
    s = str(don_gia_str).replace(' ', '')
    s = s.replace('.', '').replace(',', '') if ('.' in s and ',' in s and s.find('.') < s.find(',')) else s
    s = s.replace('.', '').replace(',', '') if (',' in s and s.find(',') > s.find('.')) else s
    try:
        don_gia = float(s)
    except:
        try:
            don_gia = float(str(don_gia_str).replace(',', ''))
        except:
            raise ValueError("Đơn giá không hợp lệ")
    thanh_tien = so_luong * don_gia

    vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
    now = datetime.now(vn_tz)
    ngay_display = now.strftime("%d/%m/%Y")
    thoi_gian_iso = now.strftime("%Y-%m-%d %H:%M:%S")

    # lưu lịch sử
    df_row = pd.DataFrame([{
        "Thời gian": thoi_gian_iso,
        "Đơn vị bán hàng": don_vi_name,
        "MST": mst,
        "Địa chỉ đơn vị": dia_chi,
        "Địa điểm thu mua": dia_diem,
        "Người phụ trách": phu_trach,
        "Họ và Tên": ho_va_ten,
        "Số CCCD": so_cccd,
        "Quê quán": que_quan,
        "Khối lượng": so_luong,
        "Đơn vị tính": don_vi_unit,
        "Đơn giá": don_gia,
        "Thành tiền": thanh_tien
    }])
    df_row.to_csv(LICH_SU_FILE, mode='a', header=False, index=False)

    row = {
        "don_vi": don_vi_name,
        "mst": mst,
        "dia_chi": dia_chi,
        "dia_diem": dia_diem,
        "phu_trach": phu_trach,
        "ngay_lap": ngay_display,
        "ho_va_ten": ho_va_ten,
        "so_cccd": so_cccd,
        "que_quan": que_quan,
        "so_luong": so_luong,
        "don_gia": don_gia,
        "thanh_tien": thanh_tien,
        "don_vi_unit": don_vi_unit,
        "mieu_ta": mieu_ta
    }
    return row

# --- Giao diện Streamlit ---
st.title("📄 BẢNG KÊ 01/TNDN — Tự động (OCR CCCD + Cân)")
st.markdown("Nhập/chụp ảnh CCCD và ảnh cân → Tạo bản kê theo mẫu 01/TNDN. Ứng dụng sẽ hiển thị PDF để in nếu có thể; nếu không có PDF, tải HTML để in từ trình duyệt.")

# Thông tin đơn vị (tùy chọn)
with st.expander("Thông tin đơn vị (tùy chọn)"):
    don_vi_name = st.text_input("Tên đơn vị (để trống nếu không có)")
    mst = st.text_input("Mã số thuế (MST)")
    dia_chi = st.text_input("Địa chỉ đơn vị")
    dia_diem = st.text_input("Địa điểm thu mua", value="Bến Lức")
    phu_trach = st.text_input("Người phụ trách thu mua")

st.markdown("---")
# session state khởi tạo
if 'ho_ten' not in st.session_state: st.session_state.ho_ten = ""
if 'so_cccd' not in st.session_state: st.session_state.so_cccd = ""
if 'que_quan' not in st.session_state: st.session_state.que_quan = ""
if 'so_luong' not in st.session_state: st.session_state.so_luong = ""

st.header("1) Thông tin người bán (khách hàng)")
col1, col2 = st.columns(2)
with col1:
    st.subheader("OCR từ CCCD")
    up_cccd = st.file_uploader("Tải ảnh CCCD (JPG/PNG) hoặc chụp", type=['jpg','jpeg','png'])
    if up_cccd:
        img = img_from_upload(up_cccd)
        ho, so, que = trich_xuat_cccd_from_image(img)
        st.session_state.ho_ten = ho or st.session_state.ho_ten
        st.session_state.so_cccd = so or st.session_state.so_cccd
        st.session_state.que_quan = que or st.session_state.que_quan

with col2:
    st.subheader("Nhập/Chỉnh thủ công")
    ho_va_ten = st.text_input("Họ và tên", value=st.session_state.ho_ten)
    so_cccd = st.text_input("Số CCCD/CMND", value=st.session_state.so_cccd)
    que_quan = st.text_input("Quê quán", value=st.session_state.que_quan)

st.markdown("---")
st.header("2) Thông tin giao dịch")
c1, c2 = st.columns(2)
with c1:
    st.subheader("OCR từ cân (ảnh)")
    up_can = st.file_uploader("Tải ảnh cân (JPG/PNG) hoặc chụp", type=['jpg','jpeg','png'], key="can")
    if up_can:
        img2 = img_from_upload(up_can)
        so_luong_ex = trich_xuat_can_from_image(img2)
        st.session_state.so_luong = so_luong_ex or st.session_state.so_luong

with c2:
    st.subheader("Nhập thủ công")
    so_luong_input = st.text_input("Khối lượng", value=str(st.session_state.so_luong))
    don_gia_input = st.text_input("Đơn giá (VNĐ)", value="1000000")
    don_vi_unit = st.text_input("Đơn vị tính (ví dụ: chỉ, kg)", value="chỉ")
    mieu_ta = st.text_input("Mô tả hàng (ví dụ: Vàng miếng...)", value="Hàng hóa")

st.markdown("---")
st.header("3) Tạo bản kê & Xuất")
if st.button("Tính tiền & Tạo bản kê"):
    # kiểm tra
    if not ho_va_ten.strip():
        st.error("Nhập họ và tên.")
    elif not so_luong_input.strip() or not don_gia_input.strip():
        st.error("Nhập đủ khối lượng và đơn giá.")
    else:
        try:
            row = process_and_save(
                ho_va_ten.strip(), so_cccd.strip(), que_quan.strip(),
                so_luong_input.strip(), don_gia_input.strip(),
                don_vi_unit=don_vi_unit.strip(), mieu_ta=mieu_ta.strip(),
                don_vi_name=don_vi_name.strip(), mst=mst.strip(),
                dia_chi=dia_chi.strip(), dia_diem=dia_diem.strip(), phu_trach=phu_trach.strip()
            )
        except Exception as e:
            st.error("Lỗi dữ liệu: " + str(e))
            row = None

        if row:
            st.success("Đã ghi lịch sử giao dịch.")
            # Tạo PDF nếu có reportlab
            pdf_bytes = None
            if REPORTLAB_OK:
                try:
                    pdf_bytes = create_pdf_bytes(row)
                except Exception as e:
                    st.warning("Tạo PDF thất bại: " + str(e))
                    pdf_bytes = None

            if pdf_bytes:
                # Hiển thị PDF trong app (embed)
                b64 = base64.b64encode(pdf_bytes).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="800px" type="application/pdf"></iframe>'
                st.markdown("### Xem trước PDF (in trực tiếp từ đây hoặc tải xuống):", unsafe_allow_html=True)
                st.components.v1.html(pdf_display, height=820)
                st.download_button("Tải PDF bản kê", data=pdf_bytes, file_name=f"bangke_01_TNDN_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")
            else:
                # fallback HTML
                html_bytes = create_html(row)
                st.info("Không tạo được PDF tự động — tải HTML và in từ trình duyệt (File → Print → Save as PDF).")
                st.download_button("Tải HTML bản kê (In từ trình duyệt)", data=html_bytes, file_name=f"bangke_01_TNDN_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html", mime="text/html")
                # hiển thị preview HTML
                st.components.v1.html(html_bytes.decode('utf-8'), height=700)

st.markdown("---")
st.header("4) Lịch sử giao dịch")
try:
    df_hist = pd.read_csv(LICH_SU_FILE)
    st.dataframe(df_hist.sort_values(by="Thời gian", ascending=False).reset_index(drop=True))
except Exception as e:
    st.error("Không đọc được lịch sử: " + str(e))
