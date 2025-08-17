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

# Optional PDF libs
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# ------------------ Config ------------------
st.set_page_config(page_title="Bảng kê 01/TNDN", layout="wide", page_icon="📄")
LICH_SU_FILE = "lich_su_giao_dich.csv"

# Create CSV history if missing
if not os.path.exists(LICH_SU_FILE):
    df_empty = pd.DataFrame(columns=[
        "Thời gian", "Đơn vị bán hàng", "MST", "Địa chỉ đơn vị",
        "Địa điểm thu mua", "Người phụ trách",
        "Họ và Tên", "Số CCCD", "Quê quán",
        "Khối lượng", "Đơn vị tính", "Đơn giá", "Thành tiền"
    ])
    df_empty.to_csv(LICH_SU_FILE, index=False)

# ------------------ OCR init ------------------
@st.cache_resource
def get_ocr():
    return PaddleOCR(lang="vi", use_angle_cls=False)
ocr = get_ocr()

# ------------------ Helpers ------------------
def img_from_upload(uploaded_file):
    """Convert UploadedFile to OpenCV image (BGR) directly in memory."""
    if uploaded_file is None:
        return None
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img

# Keep CCCD as string; ensure leading zeros preserved
def normalize_cccd_candidate(candidate):
    cand = ''.join(ch for ch in candidate if ch.isdigit())
    if len(cand) >= 12:
        return cand[:12]
    if 9 <= len(cand) < 12:
        return cand.zfill(12)
    return cand

# number -> formatted string with dot as thousands separator
def fmt_money(v):
    try:
        return f"{int(round(v)):,}".replace(',', '.')
    except:
        return "0"

# Number to Vietnamese words (simple)
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

# ------------------ OCR extractors ------------------
def trich_xuat_cccd_from_img(img):
    """OCR image to (ho_ten, so_cccd (string), que_quan)."""
    try:
        res = ocr.ocr(img, cls=False)
    except Exception:
        return "", "", ""
    if not res or not res[0]:
        return "", "", ""
    lines = res[0]
    ho_ten = so_cccd = que_quan = ""
    for i, ln in enumerate(lines):
        txt = ln[1][0].strip()
        up = txt.upper()
        if "HỌ VÀ TÊN" in up or "HỌ TÊN" in up:
            if i+1 < len(lines):
                ho_ten = lines[i+1][1][0].strip()
        if "SỐ" in up:
            # take last token as candidate
            candidate = txt.split()[-1]
            cand = normalize_cccd_candidate(candidate)
            if len(cand) >= 9:
                so_cccd = cand
        if "QUÊ QUÁN" in up:
            if i+1 < len(lines):
                que_quan = lines[i+1][1][0].strip()
    # fallback: search any 12-digit token
    if not so_cccd:
        for ln in lines:
            txt = ln[1][0]
            digits = ''.join(ch for ch in txt if ch.isdigit())
            if len(digits) >= 12:
                so_cccd = digits[:12]
                break
    return ho_ten, so_cccd, que_quan

def trich_xuat_can_from_img(img):
    """OCR image to first numeric token (string)."""
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

# ------------------ PDF builder ------------------
def build_pdf_bytes(row):
    """
    Build a visually pleasing A4 PDF (bytes) for a single-row bảng kê.
    Expects row dict with keys: don_vi, mst, dia_chi, dia_diem, phu_trach, ngay_lap,
    ho_va_ten, so_cccd, que_quan, so_luong, don_gia, thanh_tien, don_vi_unit, mieu_ta
    """
    if not REPORTLAB_OK:
        return None
    # register DejaVu if available in working dir
    font_registered = False
    try:
        if os.path.exists("DejaVuSans.ttf"):
            pdfmetrics.registerFont(TTFont('DejaVu', 'DejaVuSans.ttf'))
            font_name = 'DejaVu'
            font_registered = True
        else:
            font_name = 'Helvetica'
    except Exception:
        font_name = 'Helvetica'

    buffer = io.BytesIO()
    w, h = A4
    c = canvas.Canvas(buffer, pagesize=A4)
    left = 18*mm
    right = 18*mm
    cur_y = h - 20*mm

    # Header: country and form id
    c.setFont(font_name, 10)
    c.drawString(left, cur_y, "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM")
    c.drawRightString(w - right, cur_y, "Mẫu số: 01/TNDN")
    cur_y -= 12
    c.setFont(font_name, 9)
    c.drawString(left, cur_y, "Độc lập - Tự do - Hạnh phúc")
    c.drawRightString(w - right, cur_y, "(Ban hành kèm theo Thông tư 78/2014/TT-BTC)")
    cur_y -= 18

    # Title
    c.setFont(font_name, 13)
    c.drawCentredString(w / 2, cur_y, "BẢNG KÊ THU MUA HÀNG HÓA, DỊCH VỤ MUA VÀO KHÔNG CÓ HÓA ĐƠN")
    cur_y -= 18

    # Optional unit info (small)
    c.setFont(font_name, 10)
    if row.get('don_vi'):
        c.drawString(left, cur_y, f"Đơn vị: {row.get('don_vi')}")
        cur_y -= 12
    if row.get('mst'):
        c.drawString(left, cur_y, f"Mã số thuế: {row.get('mst')}")
        cur_y -= 12
    if row.get('dia_chi'):
        c.drawString(left, cur_y, f"Địa chỉ: {row.get('dia_chi')}")
        cur_y -= 12
    cur_y -= 6

    # thu mua info
    c.drawString(left, cur_y, f"Địa điểm thu mua: {row.get('dia_diem','')}")
    cur_y -= 12
    c.drawString(left, cur_y, f"Người phụ trách: {row.get('phu_trach','')}")
    cur_y -= 12
    c.drawString(left, cur_y, f"Ngày lập bảng kê: {row.get('ngay_lap','')}")
    cur_y -= 16

    # Seller info box
    c.setFont(font_name, 11)
    c.drawString(left, cur_y, "Thông tin người bán:")
    cur_y -= 12
    c.setFont(font_name, 10)
    c.drawString(left + 6*mm, cur_y, f"Họ và tên: {row.get('ho_va_ten','')}")
    cur_y -= 10
    c.drawString(left + 6*mm, cur_y, f"Số CCCD/CMND: {row.get('so_cccd','')}")
    cur_y -= 10
    c.drawString(left + 6*mm, cur_y, f"Quê quán: {row.get('que_quan','')}")
    cur_y -= 16

    # Table header
    col_w = [18*mm, 80*mm, 22*mm, 30*mm, 38*mm, 40*mm]
    x = left
    c.setFont(font_name, 9)
    headers = ["STT", "Tên hàng/dịch vụ", "ĐVT", "Số lượng", "Đơn giá (VNĐ)", "Thành tiền (VNĐ)"]
    for i, h in enumerate(headers):
        c.rect(x, cur_y-14, col_w[i], 16, stroke=1, fill=0)
        c.drawCentredString(x + col_w[i]/2, cur_y-10, h)
        x += col_w[i]
    cur_y -= 18

    # One row
    x = left
    c.rect(x, cur_y-12, col_w[0], 14, stroke=1); c.drawCentredString(x+col_w[0]/2, cur_y-8, "1"); x+=col_w[0]
    c.rect(x, cur_y-12, col_w[1], 14, stroke=1); c.drawString(x+4, cur_y-10, row.get('mieu_ta','Hàng hóa')); x+=col_w[1]
    c.rect(x, cur_y-12, col_w[2], 14, stroke=1); c.drawCentredString(x+col_w[2]/2, cur_y-8, row.get('don_vi_unit','')); x+=col_w[2]
    c.rect(x, cur_y-12, col_w[3], 14, stroke=1); c.drawCentredString(x+col_w[3]/2, cur_y-8, str(row.get('so_luong',''))); x+=col_w[3]
    c.rect(x, cur_y-12, col_w[4], 14, stroke=1); c.drawRightString(x+col_w[4]-4, cur_y-8, fmt_money(row.get('don_gia',0))); x+=col_w[4]
    c.rect(x, cur_y-12, col_w[5], 14, stroke=1); c.drawRightString(x+col_w[5]-4, cur_y-8, fmt_money(row.get('thanh_tien',0)))
    cur_y -= 28

    # Total
    c.setFont(font_name, 10)
    c.drawRightString(w - right, cur_y, "Tổng cộng: " + fmt_money(row.get('thanh_tien',0)) + " VNĐ")
    cur_y -= 14
    c.drawString(left, cur_y, "Số tiền bằng chữ: " + to_words_vnd(row.get('thanh_tien',0)))
    cur_y -= 28

    # Sign area
    c.drawString(left, cur_y, f"{row.get('dia_diem','')}, ngày {row.get('ngay_lap','')}")
    c.drawString(left+6*mm, cur_y-18, "Người lập bảng kê")
    c.drawString(w/2, cur_y-18, "Người bán")
    c.drawString(w - right - 80*mm, cur_y-18, "Thủ trưởng đơn vị")
    c.line(left, cur_y-60, left+60*mm, cur_y-60)
    c.line(w/2, cur_y-60, w/2+60*mm, cur_y-60)
    c.line(w - right - 80*mm, cur_y-60, w - right + 10*mm, cur_y-60)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

# ------------------ HTML builder (beautiful A4 print CSS) ------------------
def build_html_bytes(row):
    # A4 style and print-friendly CSS
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Bảng kê 01/TNDN</title>
<style>
@page {{ size: A4; margin: 20mm; }}
body{{font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; color:#111; margin:0; padding:0;}}
.container{{width:210mm; padding:12mm; box-sizing:border-box;}}
.header{{display:flex; justify-content:space-between; align-items:flex-start;}}
.h-title{{text-align:center; margin-top:6px}}
.table{{width:100%; border-collapse:collapse; margin-top:12px;}}
.table th, .table td{{border:1px solid #ccc; padding:6px; font-size:13px;}}
.table thead th{{background:#f3f6fb; font-weight:600; text-align:center}}
.small{{color:#555; font-size:12px}}
.right{{text-align:right}}
.sig-row{{margin-top:30px; display:flex; justify-content:space-between}}
.signbox{{width:30%; text-align:center}}
@media print {{
  .no-print {{ display:none; }}
}}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div>
      <div><strong>CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM</strong></div>
      <div class="small">Độc lập - Tự do - Hạnh phúc</div>
    </div>
    <div class="small">Mẫu số: 01/TNDN<br/>(Ban hành kèm theo Thông tư 78/2014/TT-BTC)</div>
  </div>

  <h2 class="h-title">BẢNG KÊ THU MUA HÀNG HÓA, DỊCH VỤ MUA VÀO KHÔNG CÓ HÓA ĐƠN</h2>

  {"<p><strong>Đơn vị:</strong> " + row.get('don_vi','') + " &nbsp;&nbsp; <strong>MST:</strong> " + row.get('mst','') + "</p>" if row.get('don_vi') else ""}
  <p class="small"><strong>Địa điểm thu mua:</strong> {row.get('dia_diem','')} &nbsp;&nbsp; <strong>Người phụ trách:</strong> {row.get('phu_trach','')}</p>
  <p class="small"><strong>Ngày lập:</strong> {row.get('ngay_lap','')}</p>

  <h4>Thông tin người bán</h4>
  <p><strong>Họ và tên:</strong> {row.get('ho_va_ten','')}<br/>
  <strong>Số CCCD:</strong> {row.get('so_cccd','')}<br/>
  <strong>Quê quán:</strong> {row.get('que_quan','')}</p>

  <table class="table">
    <thead>
      <tr><th>STT</th><th>Tên hàng/dịch vụ</th><th>ĐVT</th><th>Số lượng</th><th>Đơn giá (VNĐ)</th><th>Thành tiền (VNĐ)</th></tr>
    </thead>
    <tbody>
      <tr>
        <td class="right">1</td>
        <td>{row.get('mieu_ta','Hàng hóa')}</td>
        <td class="right">{row.get('don_vi_unit','')}</td>
        <td class="right">{row.get('so_luong','')}</td>
        <td class="right">{fmt_money(row.get('don_gia',0))}</td>
        <td class="right">{fmt_money(row.get('thanh_tien',0))}</td>
      </tr>
    </tbody>
  </table>

  <p class="right"><strong>Tổng cộng: {fmt_money(row.get('thanh_tien',0))} VNĐ</strong></p>
  <p><strong>Số tiền bằng chữ:</strong> {to_words_vnd(row.get('thanh_tien',0))}</p>

  <div class="sig-row">
    <div class="signbox">Người lập bảng kê<br/>(Ký, ghi rõ họ tên)</div>
    <div class="signbox">Người bán<br/>(Ký, ghi rõ họ tên)</div>
    <div class="signbox">Thủ trưởng đơn vị<br/>(Ký, đóng dấu)</div>
  </div>
</div>
</body>
</html>
"""
    return html.encode('utf-8')

# ------------------ Save & process ------------------
def process_and_record(ho_va_ten, so_cccd, que_quan, so_luong_str, don_gia_str,
                       don_vi_unit='chỉ', mieu_ta='Hàng hóa',
                       don_vi_name='', mst='', dia_chi='', dia_diem='Bến Lức', phu_trach=''):
    # parse so_luong
    try:
        so_luong = float(str(so_luong_str).replace(',', '.'))
    except:
        raise ValueError("Khối lượng không hợp lệ.")
    # parse don_gia robust
    s = str(don_gia_str).replace(' ', '')
    # if both dot and comma exist, try heuristics
    if ',' in s and '.' in s:
        # assume dot is thousand, remove dots, replace comma by dot for decimal
        if s.find('.') < s.find(','):
            s = s.replace('.', '').replace(',', '.')
        else:
            s = s.replace(',', '')
    else:
        s = s.replace(',', '')
    try:
        don_gia = float(s)
    except:
        raise ValueError("Đơn giá không hợp lệ.")
    thanh_tien = so_luong * don_gia

    now = datetime.now(pytz.timezone("Asia/Ho_Chi_Minh"))
    ngay_display = now.strftime("%d/%m/%Y")
    thoi_gian_iso = now.strftime("%Y-%m-%d %H:%M:%S")

    # record CSV (keep so_cccd as string)
    df_row = pd.DataFrame([{
        "Thời gian": thoi_gian_iso,
        "Đơn vị bán hàng": don_vi_name,
        "MST": mst,
        "Địa chỉ đơn vị": dia_chi,
        "Địa điểm thu mua": dia_diem,
        "Người phụ trách": phu_trach,
        "Họ và Tên": ho_va_ten,
        "Số CCCD": str(so_cccd),
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
        "so_cccd": str(so_cccd),
        "que_quan": que_quan,
        "so_luong": so_luong,
        "don_gia": don_gia,
        "thanh_tien": thanh_tien,
        "don_vi_unit": don_vi_unit,
        "mieu_ta": mieu_ta
    }
    return row

# ------------------ UI ------------------
st.title("📄 BẢNG KÊ 01/TNDN — OCR CCCD & Cân — PDF/HTML đẹp")
st.markdown("Ứng dụng tạo bản kê theo mẫu 01/TNDN. Chụp/tải ảnh CCCD và cân → xem trước PDF/HTML → tải/ in.")

# Optional unit info
with st.expander("Thông tin đơn vị (tùy chọn) — chỉ hiển thị trên bản kê"):
    don_vi_name = st.text_input("Tên đơn vị")
    mst = st.text_input("Mã số thuế (MST)")
    dia_chi = st.text_input("Địa chỉ đơn vị")
    dia_diem = st.text_input("Địa điểm thu mua", value="Bến Lức")
    phu_trach = st.text_input("Người phụ trách thu mua")

st.markdown("---")

# session init
if 'ho_ten' not in st.session_state: st.session_state.ho_ten = ""
if 'so_cccd' not in st.session_state: st.session_state.so_cccd = ""
if 'que_quan' not in st.session_state: st.session_state.que_quan = ""
if 'so_luong' not in st.session_state: st.session_state.so_luong = ""

# 1. seller info
st.header("1) Thông tin người bán (khách hàng)")
c1, c2 = st.columns(2)
with c1:
    st.subheader("OCR từ ảnh CCCD")
    up_cccd = st.file_uploader("Tải ảnh CCCD (JPG/PNG) hoặc chụp", type=['jpg','jpeg','png'])
    if up_cccd:
        img = img_from_upload(up_cccd)
        ho, so, que = trich_xuat_cccd_from_img(img)
        st.session_state.ho_ten = ho or st.session_state.ho_ten
        st.session_state.so_cccd = so or st.session_state.so_cccd
        st.session_state.que_quan = que or st.session_state.que_quan
        st.success("Đã trích xuất từ ảnh (có thể cần chỉnh sửa).")
with c2:
    st.subheader("Nhập/chỉnh thủ công")
    ho_va_ten = st.text_input("Họ và tên", value=st.session_state.ho_ten)
    so_cccd = st.text_input("Số CCCD/CMND (giữ dạng chuỗi)", value=st.session_state.so_cccd)
    que_quan = st.text_input("Quê quán", value=st.session_state.que_quan)

st.markdown("---")
# 2. transaction info
st.header("2) Thông tin giao dịch")
d1, d2 = st.columns(2)
with d1:
    st.subheader("OCR từ ảnh cân")
    up_can = st.file_uploader("Tải ảnh cân (JPG/PNG) hoặc chụp", type=['jpg','jpeg','png'], key="can_upload")
    if up_can:
        img2 = img_from_upload(up_can)
        so_luong_ex = trich_xuat_can_from_img(img2)
        st.session_state.so_luong = so_luong_ex or st.session_state.so_luong
        st.success("Đã trích xuất khối lượng (kiểm tra và chỉnh nếu cần).")
with d2:
    st.subheader("Nhập thủ công")
    so_luong_input = st.text_input("Khối lượng", value=str(st.session_state.so_luong))
    don_gia_input = st.text_input("Đơn giá (VNĐ)", value="1000000")
    don_vi_unit = st.text_input("Đơn vị tính (ví dụ: chỉ, kg)", value="chỉ")
    mieu_ta = st.text_input("Mô tả hàng (VD: Vàng miếng...)", value="Hàng hóa")

st.markdown("---")
# 3. create and preview
st.header("3) Tạo bản kê & Xem trước / Tải")
cola, colb = st.columns([1,1])
with cola:
    if st.button("Tạo bản kê (tính & lưu)"):
        # validation
        if not ho_va_ten.strip():
            st.error("Nhập Họ và tên.")
        elif not so_luong_input.strip() or not don_gia_input.strip():
            st.error("Nhập Khối lượng và Đơn giá.")
        else:
            try:
                row = process_and_record(
                    ho_va_ten.strip(), so_cccd.strip(), que_quan.strip(),
                    so_luong_input.strip(), don_gia_input.strip(),
                    don_vi_unit=don_vi_unit.strip(), mieu_ta=mieu_ta.strip(),
                    don_vi_name=don_vi_name.strip(), mst=mst.strip(), dia_chi=dia_chi.strip(),
                    dia_diem=dia_diem.strip(), phu_trach=phu_trach.strip()
                )
            except Exception as e:
                st.error("Lỗi khi xử lý dữ liệu: " + str(e))
                row = None

            if row:
                st.success("Đã lưu giao dịch vào lịch sử.")
                # Build PDF bytes if possible
                pdf_bytes = None
                if REPORTLAB_OK:
                    try:
                        pdf_bytes = build_pdf_bytes(row)
                    except Exception as e:
                        st.warning("Tạo PDF gặp lỗi, chuyển sang HTML. Lỗi: " + str(e))
                        pdf_bytes = None

                html_bytes = build_html_bytes(row)

                # Show preview and download options
                if pdf_bytes:
                    b64 = base64.b64encode(pdf_bytes).decode('utf-8')
                    pdf_display = f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="700px"></iframe>'
                    st.markdown("**Xem trước PDF (in trực tiếp từ preview hoặc tải xuống):**", unsafe_allow_html=True)
                    st.components.v1.html(pdf_display, height=720)
                    st.download_button("📥 Tải PDF (in A4)", data=pdf_bytes,
                                       file_name=f"bangke_01_TNDN_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                       mime="application/pdf")
                else:
                    # HTML preview and download
                    st.markdown("**PDF không khả dụng — Xem trước HTML và in từ trình duyệt (File → Print → Save as PDF):**")
                    st.components.v1.html(html_bytes.decode('utf-8'), height=700)
                    st.download_button("📥 Tải HTML (In từ trình duyệt)", data=html_bytes,
                                       file_name=f"bangke_01_TNDN_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                       mime="text/html")
                # show quick metrics
                st.metric("Thành tiền (VNĐ)", fmt_money(row.get('thanh_tien',0)))
                st.write("Số tiền bằng chữ:", to_words_vnd(row.get('thanh_tien',0)))

with colb:
    st.info("Ghi chú:\n• OCR không hoàn hảo — kiểm tra kết quả trước khi in.\n• Nếu PDF không hiển thị, hãy tải HTML rồi in từ trình duyệt.\n• Để PDF có tiếng Việt chuẩn, đặt file 'DejaVuSans.ttf' cùng thư mục nếu cần.")
    st.markdown("**Các hành động:**")
    st.button("Xóa form (làm mới)", on_click=lambda: st.experimental_rerun())

st.markdown("---")
# 4. history
st.header("4) Lịch sử giao dịch")
try:
    df_hist = pd.read_csv(LICH_SU_FILE)
    st.dataframe(df_hist.sort_values(by="Thời gian", ascending=False).reset_index(drop=True))
except Exception as e:
    st.error("Không đọc được lịch sử: " + str(e))
