import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
from IPython.display import display
from wordcloud import WordCloud
import joblib


#------- Tạo công thức cho xuất đầu ra -----

# Đọc dữ liệu
df_cluster = pd.read_csv('top2_clusters_per_company.csv')
df_sentiment = pd.read_csv('sentiment_by_company.csv')

# Tạo đầu ra cluster khi nhập tên công ty
cluster_desc = {
    0: (
        '''
        • Văn hoá vui vẻ & benefit tốt.\n
        • Đồng nghiệp thân thiện, không khí năng động.\n
        • Giờ giấc linh hoạt, OT minh bạch.\n
        • Cơ sở vật chất ổn – campus đẹp, macbook, benefit hấp dẫn.\n'''
    ),

    1: (
        '''
        • Văn phòng đẹp nhưng lương/benefit còn lấn cấn.\n
        • Không gian rộng, thiết bị xịn, môi trường chuyên nghiệp.\n
        • Nhân sự trẻ, thích giờ giấc thoải mái.\n
        • Phản ánh lương chủ yếu ở mức 'cơ bản', cần cạnh tranh hơn.\n'''
    ),

    2: (
        '''
        • Đồng nghiệp năng động, trẻ, chuyên nghiệp.\n
        • Văn phòng đẹp + macbook chuyên nghiệp.\n
        • Nhiều intern/trainee, đồng nghiệp giỏi.\n
        • Chính sách OT rõ ràng, benefit tốt.\n'''
    ),

    3: (
        '''
        • Ổn định truyền thống, tech-stack bắt đầu cũ.\n
        • Công ty tốt, thời gian thoải mái.\n
        • Nhắc nhiều tới kỹ năng mềm, dự án 'tốt vừa'.\n
        • Công nghệ cũ được nêu như điểm trừ.\n'''
    ),

    4: (
        '''
        • Phúc lợi tốt nhưng quy trình lương gặp vấn đề.\n
        • Môi trường thoải mái, quy trình rõ ràng.\n
        • Đồng nghiệp trẻ, phúc lợi ổn.\n
        • Than phiền lương chậm / trung bình.\n'''
    ),

    5: (
        '''
        • Chính sách rõ, môi trường trẻ, lương đầy đủ.\n
        • OT minh bạch, văn phòng hiện đại.\n
        • Đồng nghiệp giỏi, culture 'gen-Z friendly'.\n
        • Được khen lương xứng đáng, chính sách rõ ràng.\n'''
    ),
}


cluster_suggest = {
    0: (
        '''
        - Duy trì hoạt động gắn kết (teambuilding, CLB sở thích).\n
        - Giữ minh bạch OT, khen thưởng kịp thời.\n
        - Triển khai chương trình chia sẻ kiến thức nội bộ.'''
    ),

    1: (
        '''
        - Rà soát thang lương, đảm bảo cạnh tranh thị trường.\n
        - Truyền thông rõ ràng benefit, giảm kỳ vọng lệch.\n
        - Tổ chức workshop phát triển kỹ năng mềm cho nhân viên trẻ.'''
    ),

    2: (
        '''
        - Thiết lập lộ trình career rõ cho intern/trainee.\n
        - Duy trì hackathon, tech-talk để giữ nhịp năng động.\n
        - Triển khai mentorship & coaching cho middle-level.'''
    ),

    3: (
        '''
        - Lên roadmap nâng cấp cơ sở vật chất, công nghệ mới.\n
        - Cấp ngân sách chứng chỉ & khóa học kỹ thuật.\n
        - Tổ chức hackday nội bộ để thử nghiệm sáng tạo.'''
    ),

    4: (
        '''
        - Thiết lập deadline cố định cho pay-run, phòng ngừa lương chậm.\n
        - Triển khai lương thưởng linh hoạt.\n
        - Truyền thông minh bạch KPI trả lương, giảm hoài nghi.'''
    ),

    5: (
        '''
        - Công bố OT/lương công khai nội bộ.\n
        - Phát triển chương trình leadership cho.\n
        - Bổ sung phúc lợi well-being (mental day, gym, health-check).'''
    ),
}


def show_company_cluster(company_name: str):
    if company_name.strip() == "":
        st.info("Vui lòng nhập tên công ty rồi nhấn Enter.")
        return
     # Lọc cluster
    matched_clusters = df_cluster[df_cluster['Company Name']
                                  .str.lower()
                                  .str.contains(company_name.lower())]
    if not matched_clusters.empty:
        # Dominant cluster
        dom_row = matched_clusters.loc[matched_clusters['percent'].idxmax()]
        c_id = int(dom_row['cluster'])
        c_share = dom_row['percent']

        st.markdown(f"##### CỤM CHÍNH: {c_id} – chiếm {c_share:.1f}%")
        st.markdown(f"**Đặc trưng:**  \n{cluster_desc.get(c_id, 'Chưa có mô tả.')}")
        st.markdown(f"**Gợi ý cải tiến:**  \n{cluster_suggest.get(c_id, 'Chưa có gợi ý.')}")
        # Biểu đồ tròn
        # Lấy dữ liệu percent & cluster gốc
        percent_values = matched_clusters['percent'].tolist()
        cluster_labels = matched_clusters['cluster'].tolist()

        # Tính phần còn thiếu
        percent_sum = sum(percent_values)
        percent_other = 100 - percent_sum

        # Nếu còn thiếu, thêm Other
        if percent_other > 0:
            percent_values.append(percent_other)
            cluster_labels.append('Other')

        # Tạo màu: màu rocket + xám cho Other
        colors = sns.color_palette('rocket', len(matched_clusters))
        if percent_other > 0:
            colors.append('lightgray')

        # Vẽ biểu đồ
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        ax1.pie(
        percent_values,
        labels=cluster_labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors
        )
        ax1.axis('equal')
        ax1.set_title('Tỷ trọng theo cụm')
        st.pyplot(fig1)

        # 6️⃣ WordCloud
        keywords_text = ' '.join(matched_clusters['keyword'].astype(str))
        wc = WordCloud(width=800, height=350, background_color='white').generate(keywords_text)

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.imshow(wc.to_array(), interpolation='bilinear')
        ax2.axis("off")
        ax2.set_title(f"Từ khóa đặc trưng – {company_name}")
        st.pyplot(fig2)
    else:
        st.info("⚠️ Chưa có đánh giá công ty trên ITviec.")

# Tạo đầu ra sentiment khi nhập từ khóa hay tên công ty

# Load model
svm_loaded = joblib.load("svm_tfidf_pipeline.pkl")

def show_company_sentiment(company_name: str, df_sentiment):
    # Lọc dữ liệu công ty
    sentiment_data = df_sentiment[df_sentiment['Company Name']
                                  .str.lower()
                                  .str.contains(company_name.lower())]

    if sentiment_data.empty:
        st.warning(f"❌ Không có đánh giá trên ITviec cho công ty: “{company_name}”")
        return

    # Hiển thị bảng
    st.markdown(f"### 🔎 KẾT QUẢ ĐÁNH GIÁ – {company_name}")
    st.dataframe(sentiment_data[['positive', 'neutral', 'negative', 'sentiment_group']])

    # Tính trung bình sentiment (nếu có nhiều dòng)
    pos = sentiment_data['positive'].mean()
    neu = sentiment_data['neutral'].mean()
    neg = sentiment_data['negative'].mean()

    # Biểu đồ tròn
    labels = ['Tích cực', 'Trung lập', 'Tiêu cực']
    sizes = [pos, neu, neg]
    colors = ['#4CAF50', '#FFC107', '#F44336']

    fig, ax = plt.subplots(figsize=(5,5))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal')
    ax.set_title('Tỷ lệ sentiment (%)')

    st.pyplot(fig)

# Hàm gợi ý nhập tên trong list tên công ty
def suggest_company_name(df_sentiment):
    # Lấy danh sách tên công ty duy nhất
    company_list = df_sentiment['Company Name'].dropna().unique().tolist()
    company_list.sort()

    # Tạo selectbox để gợi ý
    selected_name = st.selectbox(
        "Chọn hoặc nhập tên công ty:",
        options=company_list
    )

    return selected_name

#------- Nội dung hiển thị trên tab -----
st.set_page_config(
  page_title="PROJECT_01",
  page_icon="  ",
  layout="wide",
  initial_sidebar_state="expanded",
) 

#------- Giao diện Streamlit -----
#Hình ảnh đầu tiên
st.image('images/channels4_banner.jpg')

# 3 tab nằm ngang
tab1, tab2, tab3 = st.tabs(["BUSINESS OVERVIEWS", "BUIL PROJECT", "NEW PREDICT"])

# Sidebar chứa dự án
with st.sidebar:
    st.sidebar.header("PROJECT_01")
    page = st.radio("Chọn nội dung:", ["Cluster", "Sentiment"])

    st.markdown("<br><br><br>", unsafe_allow_html=True)

    st.sidebar.header('INFORMATION')
    st.sidebar.write('Vo Minh Tri')
    st.sidebar.write('Email: trivm203@gmail.com')
    st.sidebar.write('Pham Thi Thu Thao')
    st.sidebar.write('Email: thaofpham@gmail.com')

#Nội dung cho từng tab
with tab1:
    if page == "Cluster":
        st.header("CLUSTER")
        st.write('''
                 Dựa trên những thông tin từ review của ứng viên/ nhân viên đăng trên ITViec để phân cụm thông tin đánh giá. 
                \nVới kết quả phân cụm, mỗi công ty có thể biết rằng công ty mình thuộc nhóm đánh giá nào. 
                \nTừ đó giúp cải thiện, phát triển công ty tốt hơn.
                ''')
    elif page == "Sentiment":
        st.header("SENTIMENT")
        st.write('''
                Phân tích cảm xúc từ các review của ứng viên/ nhân viên để xác định đánh giá tích cực, tiêu cực hoặc trung lập. 
                \nGiúp công ty hiểu rõ hơn cảm nhận chung từ phía nhân viên.
                ''')

with tab2:
    if page == "Cluster":
        st.header("CLUSTER")
        st.write('''Đánh giá của các công ty trên ITViec được thư thập từ 2016 - 2025.
                \nDữ liệu được làm sạch và thể hiện cụm từ có nghĩa bằng wordcloud''')
        st.image('images/1.1_wordcloud_all.png')
        st.write('Dùng LDA để tìm chủ đề ẩn sau khi đã vector hóa bằng tf-idf')
        st.image('images/1.1_keyword.png')
        st.write('Dùng silhouette_score để tìm số cụm tốt nhất k-best')
        st.image('images/1.1_k_best.png')
        st.write('Phân cụm bằng Kmeans và hiển thị từ khóa đặc trưng')
        st.image('images/1.1_cpn cluster and keyword.png')
        st.write('PCA để giảm chiều và trực quan hóa bằng biểu đồ cluster')
        st.image('images/1.1_clusterchart.png')
        

    elif page == "Sentiment":
        st.header("SENTIMENT")
        st.write('Xác định sentiment mỗi review dựa trên điểm đánh giá cụ thể 1-5')
        st.image('images/1.2_sentiment_mark.png')
        st.write('Dùng sentimet mỗi review và cột review dạng test làm dữ liệu huấn luyện cho mô hình Pycaret')
        st.image('images/1.2_compare_model.png')
        st.write('Chọn mô hình tốt nhất svm')
        st.image('images/1.2_svmmodel.png')
        st.write('Kết quả dự đoán sentiment review và các chỉ số đánh giá dựa trên mô hình svm')
        st.image('images/1.2_predict result.png')
        st.write('Confusion matrix cho mô hình')
        st.image('images/1.2_confusion_matrixpng.png')
        st.write('Kết quả dự đoán sentiment cho công ty dựa trên tất cả đánh giá')
        st.image('images/1.2_sentiment cpn.png')
    
with tab3:
    if page == "Cluster":
        st.header("CLUSTER")
        name = suggest_company_name(df_cluster)
        show_company_cluster(name)
        
        

    elif page == "Sentiment":
        st.header("PREDICT SENTIMENT")
        search = st.text_input('Nhập bình luận của bạn: ')
        # Giả sử bạn dự đoán với 1 text
        if st.button("Dự đoán"):
            if search.strip() != "":
                preds = svm_loaded.predict([search])  # <-- chỉ cần truyền list hoặc Series text
                st.write(f"Dự đoán: {preds[0]}")
            else:
                st.warning("Vui lòng nhập bình luận trước khi dự đoán.")
        
        name_2 = suggest_company_name(df_sentiment)
        if name_2:
            show_company_sentiment(name_2, df_sentiment)
                                 
# Adding a footer
footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
.footer p {
    font-size: 20px;  
    color: blue;
    margin: 10px 0; 
}

</style>
<div class="footer">
<p> Trung tâm Tin Học - Trường Đại Học Khoa Học Tự Nhiên <br> Đồ án tốt nghiệp Data Science </p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)