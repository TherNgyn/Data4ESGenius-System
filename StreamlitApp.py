import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")


# Cài đặt trang
st.set_page_config(
    page_title="Công cụ đánh giá ESG",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tiêu đề ứng dụng
st.title("Công cụ đánh giá ESG (Environmental, Social, and Governance)")
st.markdown("Nhập các chỉ số để đánh giá hiệu suất ESG của dự án")

# Function để tính điểm ESG
def calculate_esg_score(df):
    """
    Tính điểm ESG từ các chỉ số và phân loại thành các score range
    """
    # Tạo một bản sao để không ảnh hưởng đến dữ liệu gốc
    result_df = df.copy()
    
    # 1. Tính điểm percentile cho mỗi chỉ số
    for feature in result_df.columns:
        if pd.api.types.is_numeric_dtype(result_df[feature]):
            # Xác định cực tính của chỉ số (càng cao càng tốt hay càng thấp càng tốt)
            # Mặc định: càng cao càng tốt (ví dụ: Access to electricity)
            # Đảo ngược: càng thấp càng tốt (ví dụ: CO2 emissions, Gini index)
            if feature in ['CO2 emissions', 'Methane emissions', 'Nitrous oxide emissions',
                         'Adjusted savings: natural resources depletion', 'Adjusted savings: net forest depletion',
                         'Mortality rate', 'Unemployment', 'Heat Index 35',
                         'Level of water stress: freshwater withdrawal as a proportion of available freshwater resources',
                         'Population density']:
                # Đảo ngược percentile cho các chỉ số mà giá trị thấp = hiệu suất tốt
                result_df[f"{feature}_percentile"] = 100 - result_df[feature]
            else:
                # Percentile bình thường cho các chỉ số mà giá trị cao = hiệu suất tốt
                result_df[f"{feature}_percentile"] = result_df[feature]
    
    # 2. Phân loại chỉ số thành 3 nhóm E, S, G
    # Danh sách chỉ số môi trường (Environmental)
    e_features = ['CO2 emissions', 'Methane emissions', 'Nitrous oxide emissions',
                 'Adjusted savings: natural resources depletion', 'Adjusted savings: net forest depletion',
                 'Agricultural land', 'Annual freshwater withdrawals', 
                 'Cooling Degree Days', 'Forest area', 'Heat Index 35',
                 'Heating Degree Days', 'Land Surface Temperature',
                 'Level of water stress: freshwater withdrawal as a proportion of available freshwater resources',
                 'Renewable energy consumption', 'Standardised Precipitation-Evapotranspiration Index']
    
    # Danh sách chỉ số xã hội (Social)
    s_features = ['Access to clean fuels and technologies for cooking', 
                 'Access to electricity',
                 'People using safely managed drinking water services',
                 'People using safely managed sanitation services',
                 'School enrollment', 'Food production index',
                 'Fertility rate', 'Labor force participation rate', 
                 'Life expectancy at birth', 'Mortality rate', 
                 'Population ages 65 and above', 'Population density',
                 'Ratio of female to male labor force participation rate', 'Unemployment']
    
    # Danh sách chỉ số quản trị (Governance)
    g_features = ['GDP growth', 'Individuals using the Internet',
                 'Agriculture', 'Control of Corruption: Estimate', 
                 'Government Effectiveness: Estimate', 'Net migration', 
                 'Patent applications', 'Political Stability and Absence of Violence/Terrorism: Estimate',
                 'Proportion of seats held by women in national parliaments',
                 'Regulatory Quality: Estimate', 'Rule of Law: Estimate', 
                 'Scientific and technical journal articles',
                 'Voice and Accountability: Estimate']
    
    # Tạo danh sách các cột percentile cho mỗi nhóm
    e_percentile_features = [f"{feature}_percentile" for feature in e_features 
                           if f"{feature}_percentile" in result_df.columns]
    s_percentile_features = [f"{feature}_percentile" for feature in s_features 
                           if f"{feature}_percentile" in result_df.columns]
    g_percentile_features = [f"{feature}_percentile" for feature in g_features 
                           if f"{feature}_percentile" in result_df.columns]
    
    # 3. Tính điểm ESG (trung bình cộng có trọng số)
    # Điểm E - Environmental
    if e_percentile_features:
        result_df['E_score'] = result_df[e_percentile_features].mean(axis=1)/100
    else:
        result_df['E_score'] = np.nan
        
    # Điểm S - Social
    if s_percentile_features:
        result_df['S_score'] = result_df[s_percentile_features].mean(axis=1)/100
    else:
        result_df['S_score'] = np.nan
        
    # Điểm G - Governance
    if g_percentile_features:
        result_df['G_score'] = result_df[g_percentile_features].mean(axis=1)/100
    else:
        result_df['G_score'] = np.nan
    
    # Tính điểm ESG tổng thể (thang điểm 0-100)
    valid_scores = [score for score in [result_df['E_score'], result_df['S_score'], result_df['G_score']] 
                   if not score.isna().all()]
    
    if valid_scores:
        result_df['ESG_score'] = pd.concat(valid_scores, axis=1).mean(axis=1) * 100
    else:
        result_df['ESG_score'] = np.nan
    
    # 4. Gán nhãn score range
    result_df['score_range'] = pd.cut(result_df['ESG_score'], 
                              bins=[0, 25, 50, 75, 100], 
                              labels=['First Quartile', 'Second Quartile', 'Third Quartile', 'Fourth Quartile'])
    
    return result_df

# Function để load hoặc train mô hình
@st.cache_resource
def get_model():
    try:
        # Thử tải mô hình đã lưu
        model = joblib.load('esg_model.pkl')
        scaler = joblib.load('esg_scaler.pkl')
        feature_order = joblib.load('esg_features.pkl') 
        st.success("Đã tải mô hình thành công!")
    except:
        st.warning("Không tìm thấy mô hình đã lưu. Bạn cần upload dữ liệu để huấn luyện mô hình mới.")
        model = None
        scaler = None
        feature_order = None
    
    return model, scaler, feature_order

# Tải mô hình
model, scaler, feature_order = get_model()

# Sidebar để chọn chế độ đánh giá
st.sidebar.title("Tùy chọn")
evaluation_mode = st.sidebar.radio(
    "Chọn chế độ đánh giá:",
    ["Nhập dữ liệu thủ công", "Upload file dữ liệu", "Train mô hình mới"]
)

# Danh sách các chỉ số phân theo trụ cột ESG
environmental_indicators = ['CO2 emissions', 'Methane emissions', 'Nitrous oxide emissions',
                 'Adjusted savings: natural resources depletion', 'Adjusted savings: net forest depletion',
                 'Agricultural land', 'Annual freshwater withdrawals', 
                 'Cooling Degree Days', 'Forest area', 'Heat Index 35',
                 'Heating Degree Days', 'Land Surface Temperature',
                 'Level of water stress: freshwater withdrawal as a proportion of available freshwater resources',
                 'Renewable energy consumption', 'Standardised Precipitation-Evapotranspiration Index']

social_indicators = ['Access to clean fuels and technologies for cooking', 
                 'Access to electricity',
                 'People using safely managed drinking water services',
                 'People using safely managed sanitation services',
                 'School enrollment', 'Food production index',
                 'Fertility rate', 'Labor force participation rate', 
                 'Life expectancy at birth', 'Mortality rate', 
                 'Population ages 65 and above', 'Population density',
                 'Ratio of female to male labor force participation rate', 'Unemployment']

governance_indicators = ['GDP growth', 'Individuals using the Internet',
                 'Agriculture', 'Control of Corruption: Estimate', 
                 'Government Effectiveness: Estimate', 'Net migration', 
                 'Patent applications', 'Political Stability and Absence of Violence/Terrorism: Estimate',
                 'Proportion of seats held by women in national parliaments',
                 'Regulatory Quality: Estimate', 'Rule of Law: Estimate', 
                 'Scientific and technical journal articles',
                 'Voice and Accountability: Estimate']

# Function để hiển thị kết quả
def display_results(results):
    # Hiển thị điểm số ESG
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Điểm Environmental", f"{results['E_score'].values[0]*10:.1f}/10")
    col2.metric("Điểm Social", f"{results['S_score'].values[0]*10:.1f}/10")
    col3.metric("Điểm Governance", f"{results['G_score'].values[0]*10:.1f}/10")
    col4.metric("Điểm ESG Tổng thể", f"{results['ESG_score'].values[0]:.1f}/100")
    
    # Hiển thị phân loại
    score_range = results['score_range'].values[0]
    
    # Xác định màu sắc cho phân loại
    if score_range == 'First Quartile':
        color = 'red'
        interpretation = "Hiệu suất ESG kém. Cần cải thiện đáng kể trong hầu hết các lĩnh vực."
    elif score_range == 'Second Quartile':
        color = 'orange'
        interpretation = "Hiệu suất ESG khá. Đã đạt được một số tiến bộ nhưng vẫn cần cải thiện nhiều lĩnh vực."
    elif score_range == 'Third Quartile':
        color = 'lightgreen'
        interpretation = "Hiệu suất ESG tốt. Đang đi đúng hướng với nhiều lĩnh vực đạt hiệu quả cao."
    else:
        color = 'green'
        interpretation = "Hiệu suất ESG xuất sắc. Đang dẫn đầu trong phần lớn các lĩnh vực ESG."
    
    st.markdown(f"### Phân loại ESG: <span style='color:{color}'>{score_range}</span>", unsafe_allow_html=True)
    st.markdown(f"**Diễn giải:** {interpretation}")
    
    # Hiển thị biểu đồ
    st.subheader("Phân tích chi tiết")
    
    # Tạo dữ liệu cho biểu đồ radar
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    # Biểu đồ cột cho điểm số 3 trụ cột
    scores = [results['E_score'].values[0]*10, results['S_score'].values[0]*10, results['G_score'].values[0]*10]
    colors = ['green', 'blue', 'red']
    ax[0].bar(['Environmental', 'Social', 'Governance'], scores, color=colors)
    ax[0].set_ylim(0, 10)
    ax[0].set_title('Điểm số 3 trụ cột ESG')
    ax[0].set_ylabel('Điểm (0-10)')
    
    # Biểu đồ tròn cho tỷ trọng của mỗi trụ cột
    ax[1].pie(scores, labels=['Environmental', 'Social', 'Governance'], 
              autopct='%1.1f%%', colors=colors, startangle=90)
    ax[1].set_title('Tỷ trọng các trụ cột trong điểm ESG')
    
    st.pyplot(fig)
    
    # Đề xuất cải thiện
    st.subheader("Đề xuất cải thiện")
    
    # Xác định lĩnh vực yếu nhất
    weakest_pillar = min([('Environmental', results['E_score'].values[0]), 
                          ('Social', results['S_score'].values[0]), 
                          ('Governance', results['G_score'].values[0])],
                        key=lambda x: x[1])[0]
    
    if weakest_pillar == 'Environmental':
        st.markdown("""
        **Ưu tiên cải thiện lĩnh vực Môi trường:**
        - Giảm phát thải khí nhà kính và chuyển đổi sang năng lượng sạch
        - Tăng cường bảo vệ tài nguyên thiên nhiên và đa dạng sinh học
        - Phát triển chiến lược thích ứng với biến đổi khí hậu
        - Áp dụng các biện pháp sử dụng tài nguyên hiệu quả hơn
        """)
    elif weakest_pillar == 'Social':
        st.markdown("""
        **Ưu tiên cải thiện lĩnh vực Xã hội:**
        - Nâng cao chất lượng giáo dục và y tế
        - Cải thiện điều kiện sống và giảm bất bình đẳng xã hội
        - Thúc đẩy bình đẳng giới và hòa nhập xã hội
        - Đảm bảo quyền lao động và các tiêu chuẩn lao động
        """)
    else:
        st.markdown("""
        **Ưu tiên cải thiện lĩnh vực Quản trị:**
        - Tăng cường minh bạch và trách nhiệm giải trình
        - Cải thiện hiệu quả của bộ máy chính phủ và chống tham nhũng
        - Xây dựng các chính sách thúc đẩy phát triển bền vững
        - Đảm bảo sự tham gia của các bên liên quan trong quá trình ra quyết định
        """)

# Chế độ nhập dữ liệu thủ công
# Chế độ nhập dữ liệu thủ công
if evaluation_mode == "Nhập dữ liệu thủ công":
    st.header("Nhập các chỉ số ESG")
    
    # Khởi tạo session state nếu chưa có
    if 'all_values' not in st.session_state:
        st.session_state.all_values = {indicator: 50.0 for indicator in 
                                       environmental_indicators + social_indicators + governance_indicators}
    
    # Phần môi trường
    st.subheader("1. Chỉ số Môi trường (Environmental)")
    env_cols = st.columns(2)
    
    for i, indicator in enumerate(environmental_indicators):
        # Sử dụng callback để cập nhật session_state
        def update_value(indicator_name):
            def callback():
                st.session_state.all_values[indicator_name] = st.session_state[f"env_{indicator_name}"]
            return callback
        
        # Sử dụng giá trị từ session_state làm giá trị mặc định
        env_cols[i % 2].slider(
            f"{indicator}",
            min_value=0.0,
            max_value=100.0,
            value=st.session_state.all_values[indicator],
            help="Kéo thanh trượt để điều chỉnh giá trị",
            key=f"env_{indicator}",
            on_change=update_value(indicator)
        )
    
    # Phần xã hội
    st.subheader("2. Chỉ số Xã hội (Social)")
    soc_cols = st.columns(2)
    
    for i, indicator in enumerate(social_indicators):
        def update_value(indicator_name):
            def callback():
                st.session_state.all_values[indicator_name] = st.session_state[f"soc_{indicator_name}"]
            return callback
            
        soc_cols[i % 2].slider(
            f"{indicator}",
            min_value=0.0,
            max_value=100.0,
            value=st.session_state.all_values[indicator],
            help="Kéo thanh trượt để điều chỉnh giá trị",
            key=f"soc_{indicator}",
            on_change=update_value(indicator)
        )
    
    # Phần quản trị
    st.subheader("3. Chỉ số Quản trị (Governance)")
    gov_cols = st.columns(2)
    
    for i, indicator in enumerate(governance_indicators):
        def update_value(indicator_name):
            def callback():
                st.session_state.all_values[indicator_name] = st.session_state[f"gov_{indicator_name}"]
            return callback
            
        gov_cols[i % 2].slider(
            f"{indicator}",
            min_value=0.0,
            max_value=100.0,
            value=st.session_state.all_values[indicator],
            help="Kéo thanh trượt để điều chỉnh giá trị",
            key=f"gov_{indicator}",
            on_change=update_value(indicator)
        )
    
    # Tính điểm ESG khi nhấn nút
    if st.button("Tính điểm ESG", key="manual_calc_button"):
        # Sử dụng giá trị từ session_state
        input_df = pd.DataFrame([st.session_state.all_values])
        
        # Hiển thị dữ liệu để kiểm tra
        st.write("Dữ liệu từ các thanh trượt:")
        st.dataframe(input_df)
        
        # Tính điểm ESG
        results = calculate_esg_score(input_df)
        
        # Hiển thị kết quả
        st.header("Kết quả Đánh giá ESG")
        display_results(results)
        
        # Kiểm tra dự đoán bằng mô hình nếu có
        if model is not None and scaler and feature_order is not None:
            try:
                st.subheader("Dự đoán bằng mô hình học máy")
                
                # Đảm bảo dữ liệu có đúng các cột mà mô hình yêu cầu
                required_features = feature_order
                st.write(f"Mô hình yêu cầu {len(required_features)} đặc trưng")
                    
                # Kiểm tra và hiển thị các cột thiếu
                missing_features = [f for f in required_features if f not in input_df.columns]
                if missing_features:
                    st.warning(f"Thiếu {len(missing_features)} cột: {', '.join(missing_features[:5])}...")
                    
                # Tạo dataframe mới với đúng các cột
                aligned_data = pd.DataFrame(index=input_df.index)
                for feature in required_features:
                    if feature in input_df.columns:
                        aligned_data[feature] = input_df[feature]
                    else:
                        aligned_data[feature] = 50.0  # Giá trị mặc định
                    
                # Chuẩn hóa dữ liệu
                X_scaled = scaler.transform(aligned_data)
                    
                # Dự đoán
                predicted_score_range = model.predict(X_scaled)
                predicted_proba = model.predict_proba(X_scaled)
                    
                # Hiển thị kết quả dự đoán
                st.success(f"Phân loại dự đoán: {predicted_score_range[0]}")
                    
                # Hiển thị xác suất
                st.write("Xác suất cho từng phân loại:")
                proba_df = pd.DataFrame(
                    [predicted_proba[0]],
                    columns=model.classes_
                )
                st.dataframe(proba_df)

            except Exception as e:
                st.error(f"Lỗi khi dự đoán: {e}")
                st.error(f"Chi tiết lỗi: {str(e)}")

# Chế độ upload file dữ liệu
elif evaluation_mode == "Upload file dữ liệu":
    st.header("Upload dữ liệu ESG")
    
    uploaded_file = st.file_uploader("Chọn file CSV hoặc Excel", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            # Xác định loại file và đọc dữ liệu
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            st.success("Đã tải dữ liệu thành công!")
            
            # Hiển thị dữ liệu
            st.subheader("Dữ liệu đã tải")
            st.dataframe(data.head())
            
            # Người dùng xác nhận cột năm/thời gian nếu có
            has_time_col = st.checkbox("Dữ liệu có cột năm/thời gian?")
            
            time_col = None
            if has_time_col:
                time_cols = data.columns.tolist()
                time_col = st.selectbox("Chọn cột năm/thời gian", time_cols)
                if time_col:
                    data.set_index(time_col, inplace=True)
            
            # Tính điểm ESG và hiển thị
            if st.button("Phân tích ESG"):
                # Tính điểm ESG
                results = calculate_esg_score(data)
                
                # Hiển thị kết quả tổng quan
                st.header("Kết quả Đánh giá ESG")
                
                # Nếu có nhiều dòng dữ liệu
                if len(results) > 1:
                    # Hiển thị bảng kết quả
                    result_table = results[['E_score', 'S_score', 'G_score', 'ESG_score', 'score_range']]
                    result_table['E_score'] = result_table['E_score'] * 10
                    result_table['S_score'] = result_table['S_score'] * 10
                    result_table['G_score'] = result_table['G_score'] * 10
                    
                    st.dataframe(result_table)
                    
                    # Hiển thị biểu đồ xu hướng theo thời gian nếu có
                    if has_time_col:
                        st.subheader("Xu hướng điểm ESG theo thời gian")
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(results.index, results['E_score']*10, 'g-', label='Environmental')
                        ax.plot(results.index, results['S_score']*10, 'b-', label='Social')
                        ax.plot(results.index, results['G_score']*10, 'r-', label='Governance')
                        ax.plot(results.index, results['ESG_score']/10, 'k--', label='ESG Overall (÷10)')
                        ax.legend()
                        ax.set_title('Điểm ESG qua các năm')
                        ax.set_ylabel('Điểm')
                        ax.grid(True)
                        
                        st.pyplot(fig)
                        
                        # Phân loại hiệu suất theo năm
                        st.subheader("Phân loại hiệu suất ESG theo năm")
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        colors = {'First Quartile': 'red', 'Second Quartile': 'orange', 
                                'Third Quartile': 'lightgreen', 'Fourth Quartile': 'darkgreen'}
                        
                        for score_range in results['score_range'].unique():
                            if pd.notna(score_range):
                                mask = results['score_range'] == score_range
                                ax.bar(results.index[mask], results['ESG_score'][mask], 
                                        color=colors.get(score_range, 'gray'), label=score_range)
                        
                        ax.axhline(y=25, color='r', linestyle='-', alpha=0.3)
                        ax.axhline(y=50, color='orange', linestyle='-', alpha=0.3)
                        ax.axhline(y=75, color='g', linestyle='-', alpha=0.3)
                        ax.set_title('Phân loại hiệu suất ESG')
                        ax.set_ylabel('Điểm ESG')
                        ax.legend()
                        
                        st.pyplot(fig)
                else:
                    # Hiển thị kết quả cho một dòng dữ liệu
                    display_results(results)
                
                # Nếu model đã được load, thực hiện dự đoán
                if model is not None and scaler is not None:
                    st.subheader("Dự đoán score range bằng mô hình học máy")
                    
                    # Lấy các cột đặc trưng gốc
                    feature_cols = [col for col in data.columns if not col.endswith('_percentile') 
                                and col not in ['E_score', 'S_score', 'G_score', 'ESG_score', 'score_range']]
                    
                    # # Xử lý NaN
                    # X = data[feature_cols].fillna(data[feature_cols].mean())
                    
                    # # Chuẩn hóa dữ liệu
                    # X_scaled = scaler.transform(X)
                    if feature_order is not None:
                        # Chỉ giữ đúng các cột mô hình yêu cầu, theo đúng thứ tự
                        aligned_data = data.reindex(columns=feature_order, fill_value=50.0)
                    else:
                        # Fallback nếu không có file đặc trưng
                        aligned_data = data[feature_cols].fillna(data[feature_cols].mean())

                    X_scaled = scaler.transform(aligned_data)

                    # Dự đoán
                    predicted_score_range = model.predict(X_scaled)
                    predicted_proba = model.predict_proba(X_scaled)
                    
                    # Hiển thị kết quả dự đoán
                    pred_results = pd.DataFrame({
                        'Actual_score_range': results['score_range'],
                        'Predicted_score_range': predicted_score_range
                    })
                    
                    st.dataframe(pred_results)
                    
                    # Hiển thị xác suất dự đoán
                    st.subheader("Xác suất thuộc về mỗi score range")
                    
                    proba_df = pd.DataFrame(
                        predicted_proba, 
                        columns=model.classes_,
                        index=data.index if hasattr(data, 'index') else range(len(data))
                    )
                    
                    st.dataframe(proba_df)

        except Exception as e:
            st.error(f"Lỗi khi xử lý file: {e}")

# Chế độ train mô hình mới
else:
    st.header("Train mô hình ESG mới")
    
    st.markdown("""
    Để huấn luyện mô hình mới, vui lòng tải lên file dữ liệu đầy đủ chứa các chỉ số ESG. 
    Mô hình sẽ được huấn luyện để dự đoán score range từ các chỉ số ESG.
    """)
    
    uploaded_file = st.file_uploader("Chọn file CSV hoặc Excel", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            # Xác định loại file và đọc dữ liệu
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            st.success("Đã tải dữ liệu thành công!")
            
            # Hiển thị dữ liệu
            st.subheader("Dữ liệu đã tải")
            st.dataframe(data.head())
            
            # Train mô hình
            if st.button("Train mô hình"):
                # Tính điểm ESG
                esg_scored_data = calculate_esg_score(data)
                
                # Chuẩn bị dữ liệu
                original_columns = [col for col in data.columns if not col.endswith('_percentile') 
                                and col not in ['E_score', 'S_score', 'G_score', 'ESG_score', 'score_range']]
                
                X = esg_scored_data[original_columns].fillna(esg_scored_data[original_columns].mean())
                y = esg_scored_data['score_range']
                
                # Loại bỏ các dòng có nhãn NaN
                valid_indices = ~y.isna()
                X = X[valid_indices]
                y = y[valid_indices]
                
                if len(y) == 0:
                    st.error("Không có đủ dữ liệu hợp lệ để huấn luyện mô hình!")
                else:
                    # Chia dữ liệu
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Chuẩn hóa dữ liệu
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Huấn luyện mô hình RandomForest
                    with st.spinner("Đang huấn luyện mô hình..."):
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                        model.fit(X_train_scaled, y_train)
                    
                    # Đánh giá mô hình
                    y_pred = model.predict(X_test_scaled)
                    
                    st.subheader("Kết quả đánh giá mô hình:")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df)
                    
                    # Tính độ quan trọng của các đặc trưng
                    feature_importance = pd.DataFrame({
                        'Feature': original_columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    st.subheader("Top 10 chỉ số quan trọng nhất:")
                    st.dataframe(feature_importance.head(10))
                    
                    # Hiển thị biểu đồ độ quan trọng của đặc trưng
                    fig, ax = plt.subplots(figsize=(10, 6))
                    top_features = feature_importance.head(10)
                    ax.barh(top_features['Feature'], top_features['Importance'])
                    ax.set_title('Top 10 chỉ số ESG quan trọng nhất')
                    ax.set_xlabel('Importance')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Lưu mô hình
                    joblib.dump(model, 'esg_model.pkl')
                    joblib.dump(scaler, 'esg_scaler.pkl')
                    
                    st.success("Đã huấn luyện và lưu mô hình thành công!")
                    
                    # Tạo nút download mô hình
                    with open('esg_model.pkl', 'rb') as f:
                        st.download_button('Tải xuống mô hình', f, file_name='esg_model.pkl')
                    
                    with open('esg_scaler.pkl', 'rb') as f:
                        st.download_button('Tải xuống scaler', f, file_name='esg_scaler.pkl')

        except Exception as e:
            st.error(f"Lỗi khi xử lý file: {e}")

# Hiển thị thông tin về ESG
with st.expander("Thông tin về ESG"):
    st.markdown("""
    ### Environmental, Social, and Governance (ESG)
    
    **ESG** là một khung đánh giá tính bền vững và tác động xã hội của một dự án, doanh nghiệp hoặc quốc gia, bao gồm ba trụ cột:
    
    1. **Environmental (Môi trường):** Đánh giá tác động đến môi trường tự nhiên như phát thải carbon, sử dụng năng lượng, quản lý chất thải, và bảo tồn tài nguyên thiên nhiên.
    
    2. **Social (Xã hội):** Xem xét các vấn đề xã hội như điều kiện lao động, quyền con người, đa dạng và hòa nhập, tác động đến cộng đồng địa phương.
    
    3. **Governance (Quản trị):** Đánh giá các khía cạnh về quản trị như minh bạch, trách nhiệm giải trình, chống tham nhũng, đạo đức kinh doanh.
    
    ### Phân loại score range:
    
    - **First Quartile (0-25)**: Hiệu suất ESG kém
    - **Second Quartile (26-50)**: Hiệu suất ESG khá
    - **Third Quartile (51-75)**: Hiệu suất ESG tốt
    - **Fourth Quartile (76-100)**: Hiệu suất ESG xuất sắc
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Developed by Data4ESGenius Team | ESG Evaluation Tool © 2023</p>
    </div>
    """, 
    unsafe_allow_html=True
)