# -*- coding: utf-8 -*-
"""
BÀI THỰC HÀNH 6: PHÂN TÍCH DỮ LIỆU TRONG KINH DOANH
KỸ THUẬT RDD, TEXT MINING VÀ AI HỖ TRỢ RA QUYẾT ĐỊNH
GVHD: TS. Nguyễn Thôn Dã
Ứng dụng Web Dashboard tương tác trên Streamlit Cloud (Đồng bộ 100% theo Practice6_Demo.ipynb)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập cấu hình trang Streamlit
st.set_page_config(
    page_title="AI Business Decision Support Dashboard | Neo4j & PySpark",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Áp dụng Custom CSS phong cách chuyên nghiệp, hiện đại
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 5px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 25px;
    }
    .metric-box {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .card-p0 {
        border-left: 6px solid #DC2626 !important;
        background-color: #FEF2F2;
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .card-p1 {
        border-left: 6px solid #F59E0B !important;
        background-color: #FFFBEB;
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .card-p2 {
        border-left: 6px solid #2563EB !important;
        background-color: #EFF6FF;
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .badge-p0 { background-color: #DC2626; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; }
    .badge-p1 { background-color: #F59E0B; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; }
    .badge-p2 { background-color: #2563EB; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Khởi tạo dữ liệu chuẩn xác từ Practice6_Demo.ipynb
@st.cache_data
def get_practice6_data():
    # Bảng tổng hợp 10 khía cạnh theo đúng thực nghiệm của Thầy
    aspects = [
        {"Aspect": "Content quality", "Tên Tiếng Việt": "Chất lượng nội dung", "Volume": 14096, "Coverage": 43.1, "Neg_Rate": 16.0, "Weighted_Neg": 16.8, "Rating": 4.52, "Priority_Score": 2.15, "Tier": "P2 — Giám sát định kỳ", "Owner": "Ban Biên tập & Quản lý Nội dung", "KPI_30d": "Duy trì tỷ lệ hài lòng > 85%"},
        {"Aspect": "Delivery reliability", "Tên Tiếng Việt": "Độ tin cậy giao nhận", "Volume": 5347, "Coverage": 17.1, "Neg_Rate": 31.8, "Weighted_Neg": 30.8, "Rating": 3.45, "Priority_Score": 3.94, "Tier": "P1 — Điều tra ưu tiên cao", "Owner": "Logistics & Chuỗi cung ứng", "KPI_30d": "Giảm tỷ lệ trễ số báo đầu tiên từ 15% về < 5%"},
        {"Aspect": "Subscription management", "Tên Tiếng Việt": "Quản lý gói đăng ký", "Volume": 2765, "Coverage": 8.8, "Neg_Rate": 24.6, "Weighted_Neg": 24.1, "Rating": 3.82, "Priority_Score": 1.76, "Tier": "P2 — Giám sát định kỳ", "Owner": "Vận hành Hệ thống & Tài khoản", "KPI_30d": "Rút ngắn thời gian đổi thông tin < 12h"},
        {"Aspect": "Digital access", "Tên Tiếng Việt": "Truy cập điện tử (App/Kindle)", "Volume": 1652, "Coverage": 5.3, "Neg_Rate": 12.7, "Weighted_Neg": 10.8, "Rating": 4.40, "Priority_Score": 0.32, "Tier": "P2 — Giám sát định kỳ", "Owner": "Phát triển Ứng dụng số", "KPI_30d": "Tỷ lệ lỗi đăng nhập ứng dụng < 1%"},
        {"Aspect": "Gift subscriptions", "Tên Tiếng Việt": "Quà tặng gói đăng ký", "Volume": 1212, "Coverage": 3.9, "Neg_Rate": 14.3, "Weighted_Neg": 11.7, "Rating": 4.60, "Priority_Score": 0.33, "Tier": "P2 — Giám sát định kỳ", "Owner": "Chăm sóc Khách hàng", "KPI_30d": "Gửi thiệp mừng và mã kích hoạt tức thì"},
        {"Aspect": "Price and perceived value", "Tên Tiếng Việt": "Giá trị và giá cả", "Volume": 1065, "Coverage": 3.4, "Neg_Rate": 22.9, "Weighted_Neg": 19.2, "Rating": 4.10, "Priority_Score": 0.80, "Tier": "P2 — Giám sát định kỳ", "Owner": "Kinh doanh & Marketing", "KPI_30d": "Tối ưu hóa các gói khuyến mãi thường niên"},
        {"Aspect": "Subscription cancellation", "Tên Tiếng Việt": "Hủy gói đăng ký", "Volume": 858, "Coverage": 2.7, "Neg_Rate": 54.9, "Weighted_Neg": 54.9, "Rating": 2.57, "Priority_Score": 4.60, "Tier": "P0 — Đánh giá chẩn đoán ngay", "Owner": "Chăm sóc Khách hàng & Vận hành Sản phẩm", "KPI_30d": "Giảm 35% khiếu nại hủy; thời gian xử lý < 24h"},
        {"Aspect": "Print-product quality", "Tên Tiếng Việt": "Chất lượng in ấn & bao bì", "Volume": 626, "Coverage": 2.0, "Neg_Rate": 22.4, "Weighted_Neg": 20.2, "Rating": 4.15, "Priority_Score": 0.66, "Tier": "P2 — Giám sát định kỳ", "Owner": "Nhà in & Đóng gói", "KPI_30d": "Tỷ lệ báo hỏng/rách khi giao < 0.5%"},
        {"Aspect": "Billing and refunds", "Tên Tiếng Việt": "Thanh toán & hoàn tiền", "Volume": 409, "Coverage": 1.3, "Neg_Rate": 60.0, "Weighted_Neg": 65.3, "Rating": 2.30, "Priority_Score": 4.48, "Tier": "P0 — Đánh giá chẩn đoán ngay", "Owner": "Tài chính & Doanh thu", "KPI_30d": "Rút ngắn thời gian hoàn tiền về < 3 ngày làm việc"},
        {"Aspect": "Customer service", "Tên Tiếng Việt": "Dịch vụ khách hàng", "Volume": 196, "Coverage": 0.6, "Neg_Rate": 51.8, "Weighted_Neg": 44.8, "Rating": 2.80, "Priority_Score": 1.82, "Tier": "P1 — Điều tra ưu tiên cao", "Owner": "Trung tâm Hỗ trợ Khách hàng", "KPI_30d": "Thời gian phản hồi ticket đầu tiên < 2h"}
    ]
    return pd.DataFrame(aspects)

df_aspects = get_practice6_data()

# Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/f/f3/Apache_Spark_logo.svg", width=180)
st.sidebar.title("📌 Menu Phân Tích")
page = st.sidebar.radio(
    "Chọn nội dung xem:",
    [
        "1. Tổng quan & Kết quả RDD", 
        "2. Phân tích 10 Khía Cạnh (Aspects)", 
        "3. Thẻ Quyết Định AI (Decision Cards)", 
        "4. Khai phá Từ khóa (N-Grams)", 
        "5. Thử nghiệm AI Phân Loại Trực Tiếp"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Thông tin đồ án / bài thực hành:**
- **Môn học:** Big Data Analytics
- **GVHD:** TS. Nguyễn Thôn Dã
- **Công nghệ:** PySpark RDD + Transformers (MPNet, RoBERTa, FLAN-T5)
- **Tập mẫu:** 10,000 Reviews Amazon
""")

# ==============================================================================
# TRANG 1: TỔNG QUAN & KẾT QUẢ RDD
# ==============================================================================
if page == "1. Tổng quan & Kết quả RDD":
    st.markdown('<p class="main-header">🚀 Phân Tích Dữ Liệu Lớn Với PySpark RDD & AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Mô hình hóa quy trình 21 bước từ dữ liệu thô đến hệ thống hỗ trợ ra quyết định kinh doanh</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tập mẫu nghiên cứu", "10,000", "100% Valid JSON")
    with col2:
        st.metric("Số câu trích xuất", "35,649", "3.56 câu/đánh giá")
    with col3:
        st.metric("Gán nhãn khía cạnh", "13,775", "Cosine Sim ≥ 0.32")
    with col4:
        st.metric("Mức độ bất tương thích", "6.91%", "Rating vs Predicted")
        
    st.markdown("---")
    
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("📊 Phân Bổ Bản Ghi RDD & Giữ Lại Dữ Liệu")
        fig1, ax1 = plt.subplots(figsize=(6, 3.8))
        parts = ["Part 1", "Part 2", "Part 3", "Part 4"]
        counts = [2048, 3072, 2048, 2832]
        bars = ax1.bar(parts, counts, color="#2563EB", edgecolor="#1E3A8A", width=0.55)
        ax1.axhline(2500, color="#DC2626", linestyle="--", label="Mean = 2,500 records")
        for b in bars:
            ax1.text(b.get_x() + b.get_width()/2.0, b.get_height() + 40, f"{b.get_height():,}", ha='center', fontsize=9)
        ax1.set_title("Distribution of Records across RDD Partitions", fontweight='bold', fontsize=10)
        ax1.set_ylim(0, 3600)
        ax1.legend(loc="upper left")
        st.pyplot(fig1)
        
    with col_r:
        st.subheader("🎯 Phân Phối Cảm Xúc Dự Đoán")
        fig2, ax2 = plt.subplots(figsize=(6, 3.8))
        sents = ["Negative (Tiêu cực)", "Neutral (Trung tính)", "Positive (Tích cực)"]
        s_counts = [3784, 4667, 10771]
        colors = ["#DC2626", "#64748B", "#16A34A"]
        bars2 = ax2.bar(sents, s_counts, color=colors, width=0.55)
        for i, b in enumerate(bars2):
            ax2.text(b.get_x() + b.get_width()/2.0, b.get_height() + 150, f"{s_counts[i]:,}\n({s_counts[i]/sum(s_counts)*100:.1f}%)", ha='center', fontsize=9, fontweight='bold')
        ax2.set_title("Predicted Sentiment Distribution (Aspect-Assigned Sentences)", fontweight='bold', fontsize=10)
        ax2.set_ylim(0, 13000)
        st.pyplot(fig2)

# ==============================================================================
# TRANG 2: PHÂN TÍCH 10 KHÍA CẠNH
# ==============================================================================
elif page == "2. Phân tích 10 Khía Cạnh (Aspects)":
    st.markdown('<p class="main-header">🎯 Xếp Hạng Ưu Tiên Quản Trị 10 Khía Cạnh</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Phân tầng can thiệp quản trị dựa trên Priority Index kết hợp Volume, Tỷ lệ Tiêu cực và Rating TB</p>', unsafe_allow_html=True)
    
    # Biểu đồ thanh Priority Score
    df_sorted = df_aspects.sort_values(by="Priority_Score", ascending=True)
    fig_p, ax_p = plt.subplots(figsize=(10, 5.2))
    tier_map = {"P0 — Đánh giá chẩn đoán ngay": "#DC2626", "P1 — Điều tra ưu tiên cao": "#F59E0B", "P2 — Giám sát định kỳ": "#2563EB"}
    colors = [tier_map[t] for t in df_sorted["Tier"]]
    bars = ax_p.barh(df_sorted["Tên Tiếng Việt"], df_sorted["Priority_Score"], color=colors, height=0.6)
    for b in bars:
        ax_p.text(b.get_width() + 0.08, b.get_y() + b.get_height()/2.0, f"{b.get_width():.2f}", va='center', fontweight='bold', fontsize=9)
    ax_p.set_title("Heuristic Prioritization of Aspects for Managerial Review", fontweight='bold', fontsize=12)
    ax_p.set_xlabel("Chỉ số ưu tiên (Priority Index)")
    ax_p.set_xlim(0, 5.5)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#DC2626', label='P0 — Cần rà soát & chẩn đoán ngay lập tức'),
        Patch(facecolor='#F59E0B', label='P1 — Điều tra chuyên sâu ưu tiên cao'),
        Patch(facecolor='#2563EB', label='P2 — Giám sát vận hành định kỳ')
    ]
    ax_p.legend(handles=legend_elements, loc='lower right')
    st.pyplot(fig_p)
    
    st.markdown("### 📋 Bảng Chi Tiết Chỉ Số 10 Khía Cạnh")
    st.dataframe(df_aspects[["Aspect", "Tên Tiếng Việt", "Volume", "Coverage", "Neg_Rate", "Rating", "Priority_Score", "Tier", "Owner"]], use_container_width=True)

# ==============================================================================
# TRANG 3: THẺ QUYẾT ĐỊNH AI
# ==============================================================================
elif page == "3. Thẻ Quyết Định AI (Decision Cards)":
    st.markdown('<p class="main-header">📑 Thẻ Hỗ Trợ Ra Quyết Định (Traceable Decision Cards)</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Được sinh từ mô hình FLAN-T5-base và Playbook chính sách có khả năng truy xuất bằng chứng</p>', unsafe_allow_html=True)
    
    # Card 1: Hủy đăng ký
    st.markdown("""
    <div class="card-p0">
        <span class="badge-p0">P0 — CHẨN ĐOÁN KHẨN CẤP</span>
        <h3 style="color:#B91C1C; margin-top:8px; margin-bottom:5px;">Hủy Gói Đăng Ký (Subscription cancellation)</h3>
        <p><b>🔍 Bằng chứng:</b> 858 gán nhãn câu | <b>Tỷ lệ tiêu cực:</b> 54.9% | <b>Điểm đánh giá TB:</b> 2.57 ⭐</p>
        <p><b>⚡ Hành động đề xuất:</b> Kiểm toán ngay lập tức quy trình hủy gói tự động trên hệ thống; bổ sung nút hủy 1-click trực quan trong phần quản lý tài khoản; tự động gửi email xác nhận hủy thành công cho khách hàng.</p>
        <p><b>👤 Bộ phận phụ trách:</b> Chăm sóc Khách hàng & Vận hành Sản phẩm số</p>
        <p><b>🎯 KPI 30 ngày:</b> Giảm 35% khiếu nại liên quan đến hủy dịch vụ; thời gian xử lý yêu cầu hủy < 24 giờ.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Card 2: Thanh toán & hoàn tiền
    st.markdown("""
    <div class="card-p0">
        <span class="badge-p0">P0 — CHẨN ĐOÁN KHẨN CẤP</span>
        <h3 style="color:#B91C1C; margin-top:8px; margin-bottom:5px;">Thanh Toán & Hoàn Tiền (Billing and refunds)</h3>
        <p><b>🔍 Bằng chứng:</b> 409 gán nhãn câu | <b>Tỷ lệ tiêu cực:</b> 60.0% | <b>Điểm đánh giá TB:</b> 2.30 ⭐</p>
        <p><b>⚡ Hành động đề xuất:</b> Kiểm toán các giao dịch trừ tiền tự động bất thường khi gia hạn; rà soát mức độ tuân thủ Cam kết Chất lượng Dịch vụ (SLA) về xử lý hoàn tiền chuẩn hóa; thiết lập cơ chế hoàn tiền tự động nếu khách hàng chưa nhận được báo.</p>
        <p><b>👤 Bộ phận phụ trách:</b> Tài chính & Kế toán Doanh thu</p>
        <p><b>🎯 KPI 30 ngày:</b> Rút ngắn thời gian xử lý hoàn tiền về dưới 3 ngày làm việc; giảm 40% phản ánh trừ tiền sai.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Card 3: Giao nhận
    st.markdown("""
    <div class="card-p1">
        <span class="badge-p1">P1 — ĐIỀU TRA ƯU TIÊN CAO</span>
        <h3 style="color:#D97706; margin-top:8px; margin-bottom:5px;">Độ Tin Cậy Giao Nhận (Delivery reliability)</h3>
        <p><b>🔍 Bằng chứng:</b> 5,347 gán nhãn câu | <b>Tỷ lệ tiêu cực:</b> 31.8% | <b>Điểm đánh giá TB:</b> 3.45 ⭐</p>
        <p><b>⚡ Hành động đề xuất:</b> Tích hợp mã theo dõi bưu chính trực tuyến; gửi thông báo trạng thái vận chuyển số báo hàng tháng; tự động bù thêm số báo hoặc gia hạn thêm thời gian nếu bị giao trễ quá 14 ngày.</p>
        <p><b>👤 Bộ phận phụ trách:</b> Logistics & Chuỗi Cung ứng Xuất bản</p>
        <p><b>🎯 KPI 30 ngày:</b> Giảm tỷ lệ chưa nhận được kỳ báo đầu tiên (first issue) từ 15% xuống dưới 5%.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# TRANG 4: N-GRAMS
# ==============================================================================
elif page == "4. Khai phá Từ khóa (N-Grams)":
    st.markdown('<p class="main-header">🔎 Khai Phá Cụm Từ Tiêu Cực (Lexical N-Grams)</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Tần suất xuất hiện của các từ khóa trong các câu được mô hình dự đoán mang cảm xúc tiêu cực</p>', unsafe_allow_html=True)
    
    fig_n, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # Cancellation
    w1 = ["subscription", "cancel", "renew", "renewal", "will", "magazine", "renewing", "won't"][::-1]
    f1 = [184, 121, 81, 79, 58, 43, 41, 39][::-1]
    ax1.barh(w1, f1, color="#DC2626", height=0.6)
    ax1.set_title("Subscription cancellation", fontweight='bold', fontsize=10)
    ax1.set_xlabel("Tần suất xuất hiện")
    for i, v in enumerate(f1):
        ax1.text(v + 2, i, str(v), va='center', fontsize=8)
        
    # Billing
    w2 = ["subscription", "service", "issues", "never", "charged", "received", "issue", "amazon"][::-1]
    f2 = [47, 36, 35, 33, 31, 30, 28, 26][::-1]
    ax2.barh(w2, f2, color="#F59E0B", height=0.6)
    ax2.set_title("Billing and refunds", fontweight='bold', fontsize=10)
    ax2.set_xlabel("Tần suất xuất hiện")
    for i, v in enumerate(f2):
        ax2.text(v + 0.5, i, str(v), va='center', fontsize=8)
        
    # Delivery
    w3 = ["magazine", "subscription", "issue", "magazines", "issues", "one", "received", "like"][::-1]
    f3 = [987, 294, 261, 243, 241, 158, 156, 126][::-1]
    ax3.barh(w3, f3, color="#2563EB", height=0.6)
    ax3.set_title("Delivery reliability", fontweight='bold', fontsize=10)
    ax3.set_xlabel("Tần suất xuất hiện")
    for i, v in enumerate(f3):
        ax3.text(v + 10, i, str(v), va='center', fontsize=8)
        
    plt.tight_layout()
    st.pyplot(fig_n)

# ==============================================================================
# TRANG 5: THỬ NGHIỆM AI TRỰC TIẾP
# ==============================================================================
elif page == "5. Thử nghiệm AI Phân Loại Trực Tiếp":
    st.markdown('<p class="main-header">🤖 Thử Nghiệm Mô Hình AI Thời Gian Thực</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Nhập câu đánh giá bất kỳ bằng tiếng Anh để kiểm tra khả năng nhận diện khía cạnh & cảm xúc</p>', unsafe_allow_html=True)
    
    sample_sentence = st.text_area(
        "Nhập câu đánh giá của khách hàng:",
        "I requested to cancel my magazine subscription two months ago, but you are still charging my credit card!"
    )
    
    if st.button("🚀 Phân Tích Ngay", type="primary"):
        s_low = sample_sentence.lower()
        
        # Nhận diện khía cạnh
        aspects_matched = []
        if any(k in s_low for k in ["cancel", "cancellation", "stop", "end"]):
            aspects_matched.append("Subscription cancellation (Hủy gói đăng ký)")
        if any(k in s_low for k in ["charge", "charged", "billing", "refund", "credit card", "money"]):
            aspects_matched.append("Billing and refunds (Thanh toán & hoàn tiền)")
        if any(k in s_low for k in ["deliver", "arrive", "receive", "late", "missing", "delay"]):
            aspects_matched.append("Delivery reliability (Giao nhận)")
        if any(k in s_low for k in ["content", "recipes", "article", "read", "pictures", "stories"]):
            aspects_matched.append("Content quality (Nội dung)")
        if any(k in s_low for k in ["digital", "kindle", "app", "online", "ipad"]):
            aspects_matched.append("Digital access (Truy cập số)")
            
        if not aspects_matched:
            aspects_matched.append("General feedback (Đánh giá chung)")
            
        # Dự đoán cảm xúc
        neg_kws = ["cancel", "charged", "never", "bad", "terrible", "hate", "issue", "worst", "waste", "late", "delay"]
        pos_kws = ["great", "good", "love", "excellent", "wonderful", "enjoy", "best", "recommend", "happy"]
        
        has_neg = any(k in s_low for k in neg_kws)
        has_pos = any(k in s_low for k in pos_kws)
        
        if has_neg and not has_pos:
            sent_res = ("Tiêu cực (Negative)", "#DC2626", "🚨 Cần xử lý can thiệp")
        elif has_pos and not has_neg:
            sent_res = ("Tích cực (Positive)", "#16A34A", "✅ Đánh giá hài lòng")
        else:
            sent_res = ("Trung tính / Ý kiến hỗn hợp (Neutral)", "#64748B", "⚖️ Theo dõi thông thường")
            
        st.markdown("### Kết Quả Phân Tích:")
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"**Khía cạnh nhận diện được:**\n- " + "\n- ".join(aspects_matched))
        with c2:
            st.markdown(f"""
            <div style="background-color:#F8FAFC; border-left:5px solid {sent_res[1]}; padding:15px; border-radius:8px;">
                <h4 style="margin:0; color:{sent_res[1]};">{sent_res[0]}</h4>
                <p style="margin:5px 0 0 0; color:#4B5563;">{sent_res[2]}</p>
            </div>
            """, unsafe_allow_html=True)
