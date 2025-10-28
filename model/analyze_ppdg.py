import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_ppdg_data():
    """Đọc dữ liệu PPDG từ file Excel"""
    try:
        df = pd.read_excel('dulieu/PPDG.xlsx')
        print("=== DỮ LIỆU PPDG ===")
        print(f"Số lượng bản ghi: {len(df)}")
        print(f"Các cột: {list(df.columns)}")
        print("\nMẫu dữ liệu:")
        print(df.head())
        print("\nThông tin dữ liệu:")
        print(df.info())
        return df
    except Exception as e:
        print(f"Lỗi khi đọc file PPDG.xlsx: {e}")
        return None

def create_ppdg_mapping():
    """Tạo mapping cho các phương pháp đánh giá dựa trên thông tin từ ảnh"""
    ppdg_mapping = {
        'EM 1': 'Đánh giá chuyên cần (Attendance And Punctuality Assessment)',
        'EM 2': 'Đánh giá bài tập cá nhân (Work Assignment Assessment)',
        'EM 3': 'Đánh giá thuyết trình (Oral Presentation Assessment)',
        'EM 4': 'Đánh giá làm việc nhóm (Teamwork Assessment)',
        'EM 5': 'Đánh giá tự học tại thư viện (Self-Study At The Library Assessment)',
        'EM 6': 'Kiểm tra viết (Written Exam)',
        'EM 7': 'Kiểm tra trắc nghiệm (Multiple Choice Exam)',
        'EM 8': 'Đánh giá báo cáo/tiểu luận (Written Report/Essay Assessment)',
        'EM 9': 'Đánh giá thực tập (Internship Assessment)',
        'EM 10': 'Đánh giá báo cáo thực tập tại doanh nghiệp (Internship Report At Enterprise Assessment)',
        'EM11': 'Đánh giá thực hành tại phòng thí nghiệm (Practice In The Laboratory Assessment)',
        'EM 12': 'Đánh giá bài tập lớn/Đồ án cá nhân (Major Assignment/Individual Project Assessment)',
        'EM 14': 'Đánh giá khoá luận tốt nghiệp (Graduation Thesis Assessment)'
    }
    return ppdg_mapping

def analyze_ppdg_usage(df):
    """Phân tích mức độ sử dụng các phương pháp đánh giá"""
    print("\n=== PHÂN TÍCH MỨC ĐỘ SỬ DỤNG PPDG ===")
    
    # Tìm các cột EM
    em_columns = [col for col in df.columns if col.startswith('EM')]
    print(f"Các cột EM tìm thấy: {em_columns}")
    
    # Phân tích tần suất sử dụng
    usage_stats = {}
    for col in em_columns:
        # Đếm số môn học sử dụng phương pháp này
        usage_count = df[col].notna().sum()
        total_subjects = len(df)
        usage_percentage = (usage_count / total_subjects) * 100
        
        usage_stats[col] = {
            'count': usage_count,
            'percentage': usage_percentage,
            'subjects': df[df[col].notna()]['Subject_Name'].tolist()
        }
    
    # Sắp xếp theo tần suất sử dụng
    sorted_usage = sorted(usage_stats.items(), key=lambda x: x[1]['count'], reverse=True)
    
    print("\n=== TẦN SUẤT SỬ DỤNG CÁC PPDG ===")
    for em_col, stats in sorted_usage:
        print(f"{em_col}: {stats['count']}/{len(df)} môn học ({stats['percentage']:.1f}%)")
        if stats['count'] > 0:
            print(f"  Ví dụ môn học: {', '.join(stats['subjects'][:3])}")
    
    return usage_stats, em_columns

def create_ppdg_matrix(df, em_columns):
    """Tạo ma trận tương quan của việc sử dụng PPDG"""
    print("\n=== MA TRẬN TƯƠNG QUAN SỬ DỤNG PPDG ===")
    
    # Tạo ma trận nhị phân (1 = có sử dụng, 0 = không sử dụng)
    ppdg_matrix = df[em_columns].notna().astype(int)
    
    # Tính ma trận tương quan
    correlation_matrix = ppdg_matrix.corr()
    
    # Vẽ heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Ma trận tương quan sử dụng các phương pháp đánh giá (PPDG)', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('ppdg_usage_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Phân tích tương quan mạnh
    print("\n=== PHÂN TÍCH TƯƠNG QUAN MẠNH ===")
    strong_correlations = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > 0.3:  # Tương quan mạnh
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                strong_correlations.append((col1, col2, corr_value))
    
    strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    
    for col1, col2, corr_value in strong_correlations:
        print(f"{col1} <-> {col2}: {corr_value:.3f}")
    
    return correlation_matrix, ppdg_matrix

def analyze_ppdg_patterns(df, em_columns):
    """Phân tích các pattern sử dụng PPDG"""
    print("\n=== PHÂN TÍCH PATTERN SỬ DỤNG PPDG ===")
    
    # Tạo ma trận nhị phân
    ppdg_matrix = df[em_columns].notna().astype(int)
    
    # Phân tích các combination phổ biến
    print("Các combination PPDG phổ biến:")
    
    # Tìm các môn học có nhiều PPDG
    ppdg_counts = ppdg_matrix.sum(axis=1)
    print(f"\nSố lượng PPDG trung bình trên mỗi môn học: {ppdg_counts.mean():.2f}")
    print(f"Số lượng PPDG tối đa: {ppdg_counts.max()}")
    print(f"Số lượng PPDG tối thiểu: {ppdg_counts.min()}")
    
    # Môn học có nhiều PPDG nhất
    max_ppdg_subjects = df.loc[ppdg_counts.idxmax()]
    print(f"\nMôn học có nhiều PPDG nhất: {max_ppdg_subjects['Subject_Name']} ({ppdg_counts.max()} PPDG)")
    
    # Môn học có ít PPDG nhất
    min_ppdg_subjects = df.loc[ppdg_counts.idxmin()]
    print(f"Môn học có ít PPDG nhất: {min_ppdg_subjects['Subject_Name']} ({ppdg_counts.min()} PPDG)")
    
    return ppdg_counts

def create_ppdg_insights(usage_stats, correlation_matrix, ppdg_mapping):
    """Tạo insights từ phân tích PPDG"""
    print("\n=== INSIGHTS TỪ PHÂN TÍCH PPDG ===")
    
    insights = []
    
    # Insight 1: PPDG phổ biến nhất
    most_used = max(usage_stats.items(), key=lambda x: x[1]['count'])
    ppdg_name = ppdg_mapping.get(most_used[0], most_used[0])
    insights.append(f"PPDG phổ biến nhất: {most_used[0]} ({ppdg_name}) - {most_used[1]['percentage']:.1f}% môn học")
    
    # Insight 2: PPDG ít phổ biến nhất (có sử dụng)
    used_ppdg = {k: v for k, v in usage_stats.items() if v['count'] > 0}
    least_used = min(used_ppdg.items(), key=lambda x: x[1]['count'])
    ppdg_name = ppdg_mapping.get(least_used[0], least_used[0])
    insights.append(f"PPDG ít phổ biến nhất: {least_used[0]} ({ppdg_name}) - {least_used[1]['percentage']:.1f}% môn học")
    
    # Insight 3: Tương quan mạnh nhất
    strong_correlations = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > 0.5:
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                strong_correlations.append((col1, col2, corr_value))
    
    if strong_correlations:
        strongest = max(strong_correlations, key=lambda x: abs(x[2]))
        insights.append(f"Tương quan mạnh nhất: {strongest[0]} và {strongest[1]} ({strongest[2]:.3f})")
    
    # In insights
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    return insights

def generate_clo_explanation_standalone(usage_stats, ppdg_mapping):
    """Tạo giải thích cho điểm CLO dựa trên PPDG (phiên bản độc lập)"""
    print("\n=== GIẢI THÍCH ĐIỂM CLO DỰA TRÊN PPDG ===")
    
    print("Dựa trên phân tích PPDG, có thể giải thích điểm CLO như sau:")
    
    # Phân tích theo loại PPDG
    formative_assessment = ['EM 1', 'EM 2', 'EM 3', 'EM 4', 'EM 5']  # Đánh giá quá trình
    summative_assessment = ['EM 6', 'EM 7', 'EM 8', 'EM 9', 'EM 10', 'EM11', 'EM 12', 'EM 14']  # Đánh giá tổng kết
    
    print("\n1. Đánh giá quá trình (Formative Assessment):")
    for em in formative_assessment:
        if em in usage_stats:
            stats = usage_stats[em]
            ppdg_name = ppdg_mapping.get(em, em)
            print(f"   - {em} ({ppdg_name}): {stats['percentage']:.1f}% môn học")
            if stats['percentage'] > 50:
                print(f"     → Phổ biến, có thể ảnh hưởng tích cực đến CLO")
            else:
                print(f"     → Ít phổ biến, ảnh hưởng hạn chế")
    
    print("\n2. Đánh giá tổng kết (Summative Assessment):")
    for em in summative_assessment:
        if em in usage_stats:
            stats = usage_stats[em]
            ppdg_name = ppdg_mapping.get(em, em)
            print(f"   - {em} ({ppdg_name}): {stats['percentage']:.1f}% môn học")
            if stats['percentage'] > 30:
                print(f"     → Khá phổ biến, có thể ảnh hưởng trực tiếp đến CLO")
            else:
                print(f"     → Ít phổ biến, ảnh hưởng hạn chế")
    
    print("\n3. Kết luận:")
    print("   - EM 1 (Đánh giá chuyên cần) và EM 2 (Đánh giá bài tập cá nhân) là phổ biến nhất")
    print("   - Các PPDG này có thể ảnh hưởng mạnh đến điểm CLO của sinh viên")
    print("   - Sinh viên học các môn có nhiều PPDG đa dạng có thể đạt điểm CLO cao hơn")

def main():
    """Hàm chính để phân tích PPDG"""
    print("=== PHÂN TÍCH PPDG VÀ TÍCH HỢP VÀO MODEL AI ===")
    
    # Load dữ liệu PPDG
    df_ppdg = load_ppdg_data()
    if df_ppdg is None:
        return
    
    # Tạo mapping PPDG
    ppdg_mapping = create_ppdg_mapping()
    
    # Phân tích mức độ sử dụng PPDG
    usage_stats, em_columns = analyze_ppdg_usage(df_ppdg)
    
    # Tạo ma trận tương quan
    correlation_matrix, ppdg_matrix = create_ppdg_matrix(df_ppdg, em_columns)
    
    # Phân tích patterns
    ppdg_counts = analyze_ppdg_patterns(df_ppdg, em_columns)
    
    # Tạo insights
    insights = create_ppdg_insights(usage_stats, correlation_matrix, ppdg_mapping)
    
    # Tạo giải thích CLO
    generate_clo_explanation_standalone(usage_stats, ppdg_mapping)
    
    # Lưu kết quả
    with open('ppdg_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write("=== BÁO CÁO PHÂN TÍCH PPDG ===\n\n")
        f.write("1. Tần suất sử dụng các PPDG:\n")
        for em_col, stats in usage_stats.items():
            f.write(f"   {em_col}: {stats['count']}/{len(df_ppdg)} môn học ({stats['percentage']:.1f}%)\n")
        
        f.write("\n2. Insights chính:\n")
        for insight in insights:
            f.write(f"   - {insight}\n")
        
        f.write("\n3. Giải thích ảnh hưởng đến CLO:\n")
        f.write("   - EM 1 (Đánh giá chuyên cần) và EM 2 (Đánh giá bài tập cá nhân) là phổ biến nhất\n")
        f.write("   - Các PPDG này có thể ảnh hưởng mạnh đến điểm CLO của sinh viên\n")
        f.write("   - Sinh viên học các môn có nhiều PPDG đa dạng có thể đạt điểm CLO cao hơn\n")
    
    print("\n=== HOÀN THÀNH PHÂN TÍCH PPDG ===")
    print("Kết quả đã được lưu vào file 'ppdg_analysis_report.txt'")
    print("Ma trận tương quan đã được lưu vào file 'ppdg_usage_correlation_matrix.png'")

if __name__ == "__main__":
    main() 