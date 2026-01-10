import pandas as pd
import numpy as np
from .utils import safe_float

class Predictor:
    def __init__(self, data_loader, model_trainer):
        self.data_loader = data_loader
        self.df = data_loader.df
        self.model = model_trainer.model
        self.X = model_trainer.X
        self.y = model_trainer.y

    def get_student_info(self, student_id):
        """Get student information"""
        # Thử tìm với kiểu dữ liệu khác nhau
        student_data = self.df[self.df['Student_ID'] == str(student_id)]
        if len(student_data) == 0:
            # Thử với kiểu int
            try:
                student_id_int = int(student_id)
                student_data = self.df[self.df['Student_ID'] == student_id_int]
            except ValueError:
                pass
        
        if len(student_data) > 0:
            return student_data[['Student_ID', 'FirstName', 'LastName', 'Major_Name']].drop_duplicates()
        return None

    def get_feature_explanation(self, feature_name, importance):
        """Get explanation for a feature"""
        explanations = {
            'student_id_encoded': 'Mã sinh viên (đã mã hóa)',
            'lecturer_encoded': 'Giảng viên (đã mã hóa)',
            'subject_encoded': 'Môn học (đã mã hóa)',
            'gender_encoded': 'Giới tính',
            'religion_encoded': 'Tôn giáo',
            'birth_place_encoded': 'Nơi sinh',
            'ethnicity_encoded': 'Dân tộc',
            'avg_conduct_score': 'Điểm rèn luyện trung bình',
            'latest_conduct_score': 'Điểm rèn luyện gần nhất',
            'conduct_trend': 'Xu hướng điểm rèn luyện',
            'study_hours_this_year': 'Số giờ tự học trong năm',
            'study_minutes_this_year': 'Số phút tự học trong năm',
            'total_study_hours': 'Tổng giờ tự học',
            'total_subjects': 'Tổng số môn đã học',
            'passed_subjects': 'Số môn đã pass',
            'pass_rate': 'Tỉ lệ pass',
            'avg_exam_score': 'Điểm thi trung bình',
            'recent_avg_score': 'Điểm trung bình gần đây',
            'improvement_trend': 'Xu hướng cải thiện'
        }
        return explanations.get(feature_name, f'Feature: {feature_name}')

    def get_risk_level(self, prob_fail):
        """Get risk level based on failure probability"""
        if prob_fail >= 0.7:
            return "CAO"
        elif prob_fail >= 0.4:
            return "TRUNG BÌNH"
        else:
            return "THẤP"

    def get_student_history(self, student_id):
        """Get detailed student history"""
        student_data = self.df[self.df['Student_ID'] == str(student_id)]
        
        if len(student_data) == 0:
            return None
        
        # Basic statistics
        total_subjects = len(student_data)
        passed_subjects = sum(student_data['passed'] == 1)
        failed_subjects = sum(student_data['passed'] == 0)
        absent_summary_count = sum(student_data['is_absent_summary'])
        failed_by_score = sum((student_data['passed'] == 0) & (student_data['is_absent_summary'] == False))
        
        # CLO statistics
        passed_clo = sum(student_data['clo_achieved'] == 1)
        failed_clo = sum(student_data['clo_achieved'] == 0)
        absent_exam_count = sum(student_data['is_absent_exam'])
        failed_by_clo = sum((student_data['clo_achieved'] == 0) & (student_data['is_absent_exam'] == False))
        
        # Score statistics
        avg_score = student_data['exam_score_6'].mean()
        recent_avg_clo = student_data.sort_values('year').tail(3)['exam_score_6'].mean() if len(student_data) >= 3 else avg_score
        
        return {
            'total_subjects': total_subjects,
            'passed_subjects': passed_subjects,
            'failed_subjects': failed_subjects,
            'retake_rate': failed_subjects / total_subjects if total_subjects > 0 else 0,
            'absent_summary_count': absent_summary_count,
            'failed_by_score': failed_by_score,
            'avg_score': avg_score,
            'passed_clo': passed_clo,
            'failed_clo': failed_clo,
            'absent_exam_count': absent_exam_count,
            'failed_by_clo': failed_by_clo,
            'avg_clo': avg_score,
            'recent_avg_clo': recent_avg_clo,
            'pass_rate_clo': passed_clo / total_subjects if total_subjects > 0 else 0
        }

    def get_subject_stats(self, subject_id):
        """Get subject statistics"""
        subject_data = self.df[self.df['Subject_ID'] == str(subject_id)]
        
        if len(subject_data) == 0:
            return None
        
        total_students = len(subject_data)
        passed_students = sum(subject_data['passed'] == 1)
        avg_score = subject_data['exam_score_6'].mean()
        pass_rate = passed_students / total_students if total_students > 0 else 0
        
        return {
            'total_students': total_students,
            'passed_students': passed_students,
            'avg_score': avg_score,
            'pass_rate': pass_rate
        }

    def get_lecturer_stats(self, lecturer):
        """Get lecturer statistics"""
        lecturer_data = self.df[self.df['Lecturer_Name'] == lecturer]
        
        if len(lecturer_data) == 0:
            return None
        
        total_students = len(lecturer_data)
        passed_students = sum(lecturer_data['passed'] == 1)
        avg_score = lecturer_data['exam_score_6'].mean()
        pass_rate = passed_students / total_students if total_students > 0 else 0
        
        return {
            'total_students': total_students,
            'passed_students': passed_students,
            'avg_score': avg_score,
            'pass_rate': pass_rate
        }

    def get_risk_assessment(self, history):
        """Get risk assessment based on student history"""
        risk_factors = []
        
        if history['pass_rate_clo'] < 0.5:
            risk_factors.append("Tỉ lệ đạt CLO thấp")
        
        if history['retake_rate'] > 0.3:
            risk_factors.append("Tỉ lệ thi lại cao")
        
        if history['avg_clo'] < 3.5:
            risk_factors.append("Điểm CLO trung bình thấp")
        
        if history['absent_exam_count'] > 0:
            risk_factors.append("Có lịch sử vắng thi")
        
        if history['recent_avg_clo'] < history['avg_clo']:
            risk_factors.append("Điểm gần đây có xu hướng giảm")
        
        return risk_factors

    def analyze_prediction_reasons(self, student_id, lecturer, subject_id, predicted_score, ppdg_analysis=None):
        """Phân tích các lý do dự đoán điểm thấp và khuyến nghị cải thiện"""
        reasons = []
        recommendations = []
        
        # Get student data
        student_data = self.df[self.df['Student_ID'] == str(student_id)]
        if len(student_data) == 0:
            # Thử với kiểu int
            try:
                student_id_int = int(student_id)
                student_data = self.df[self.df['Student_ID'] == student_id_int]
            except ValueError:
                pass
        
        if len(student_data) == 0:
            return reasons, recommendations
        
        # Get lecturer data
        lecturer_data = self.df[self.df['Lecturer_Name'].str.lower() == lecturer.lower()]
        
        # 1. Student performance factors
        student_avg_score = student_data['exam_score_6'].mean()
        student_pass_rate = student_data['passed'].mean()
        
        if student_avg_score < 3.5:
            reasons.append({
                'reason': 'Điểm trung bình CLO của sinh viên thấp',
                'detail': f'Điểm trung bình: {student_avg_score:.2f}/6 (dưới ngưỡng 3.5)',
                'severity': 'CAO'
            })
        
        if student_pass_rate < 0.7:
            reasons.append({
                'reason': 'Tỉ lệ pass môn học của sinh viên thấp',
                'detail': f'Tỉ lệ pass: {student_pass_rate:.1%} (dưới 70%)',
                'severity': 'CAO'
            })
        
        # 2. Lecturer factors - Thêm phân tích giáo viên mới
        if len(lecturer_data) > 0:
            lecturer_avg_score = lecturer_data['exam_score_6'].mean()
            lecturer_pass_rate = lecturer_data['passed'].mean()
            lecturer_student_count = len(lecturer_data)
            
            # Kiểm tra xem có phải giáo viên mới không (dựa trên số lượng sinh viên ít)
            is_new_lecturer = lecturer_student_count < 10  # Dưới 10 sinh viên được coi là giáo viên mới
            
            if is_new_lecturer:
                reasons.append({
                    'reason': 'Giảng viên mới - kinh nghiệm giảng dạy hạn chế',
                    'detail': f'Giảng viên chỉ có {lecturer_student_count} sinh viên (dưới 10 sinh viên)',
                    'severity': 'TRUNG BÌNH'
                })
            
            if lecturer_avg_score < 3.5:
                severity = 'CAO' if not is_new_lecturer else 'TRUNG BÌNH'
                reason_text = 'Điểm trung bình CLO của giảng viên thấp'
                if is_new_lecturer:
                    reason_text += ' (có thể do kinh nghiệm hạn chế)'
                
                reasons.append({
                    'reason': reason_text,
                    'detail': f'Điểm trung bình: {lecturer_avg_score:.2f}/6 (dưới ngưỡng 3.5)',
                    'severity': severity
                })
            
            if lecturer_pass_rate < 0.7:
                severity = 'CAO' if not is_new_lecturer else 'TRUNG BÌNH'
                reason_text = 'Tỉ lệ pass môn học của giảng viên thấp'
                if is_new_lecturer:
                    reason_text += ' (có thể do phương pháp giảng dạy chưa hiệu quả)'
                
                reasons.append({
                    'reason': reason_text,
                    'detail': f'Tỉ lệ pass: {lecturer_pass_rate:.1%} (dưới 70%)',
                    'severity': severity
                })
        else:
            # Không có dữ liệu về giảng viên - có thể là giảng viên mới
            reasons.append({
                'reason': 'Giảng viên mới - chưa có dữ liệu giảng dạy',
                'detail': 'Không tìm thấy dữ liệu giảng dạy trước đây của giảng viên này',
                'severity': 'TRUNG BÌNH'
            })
        
        # 3. Subject factors
        subject_data = self.df[self.df['Subject_ID'] == subject_id]
        if len(subject_data) > 0:
            subject_avg_score = subject_data['exam_score_6'].mean()
            subject_pass_rate = subject_data['passed'].mean()
            
            if subject_avg_score < 3.5:
                reasons.append({
                    'reason': 'Môn học có điểm CLO trung bình thấp',
                    'detail': f'Điểm trung bình môn {subject_id}: {subject_avg_score:.2f}/6 (dưới ngưỡng 3.5)',
                    'severity': 'TRUNG BÌNH'
                })
            
            if subject_pass_rate < 0.7:
                reasons.append({
                    'reason': 'Môn học có tỉ lệ pass thấp',
                    'detail': f'Tỉ lệ pass môn {subject_id}: {subject_pass_rate:.1%} (dưới 70%)',
                    'severity': 'TRUNG BÌNH'
                })
        
        # 4. Student-lecturer interaction
        student_lecturer_data = student_data[student_data['Lecturer_Name'].str.lower() == lecturer.lower()]
        if len(student_lecturer_data) > 0:
            interaction_avg_score = student_lecturer_data['exam_score_6'].mean()
            if interaction_avg_score < 3.5:
                reasons.append({
                    'reason': 'Sinh viên có kết quả kém với giảng viên này',
                    'detail': f'Điểm trung bình với giảng viên {lecturer}: {interaction_avg_score:.2f}/6',
                    'severity': 'CAO'
                })
        
        # 5. Student-subject interaction
        student_subject_data = student_data[student_data['Subject_ID'] == subject_id]
        if len(student_subject_data) > 0:
            subject_interaction_score = student_subject_data['exam_score_6'].mean()
            if subject_interaction_score < 3.5:
                reasons.append({
                    'reason': 'Sinh viên có kết quả kém với môn học này',
                    'detail': f'Điểm trung bình môn {subject_id}: {subject_interaction_score:.2f}/6',
                    'severity': 'CAO'
                })
        
        # 6. PPDG factors - Thêm phân tích PPDG vào lý do
        if ppdg_analysis and 'current_status' in ppdg_analysis:
            current_status = ppdg_analysis['current_status']
            effectiveness = ppdg_analysis.get('effectiveness', {})
            compatibility = ppdg_analysis.get('compatibility', {})
            
            # Kiểm tra số lượng PPDG
            if current_status['total_ppdg'] < 5:
                reasons.append({
                    'reason': 'Phương pháp đánh giá (PPDG) chưa đủ đa dạng',
                    'detail': f'Chỉ sử dụng {current_status["total_ppdg"]} PPDG (khuyến nghị: ít nhất 5-6)',
                    'severity': 'TRUNG BÌNH'
                })
            
            # Kiểm tra điểm đa dạng PPDG
            if current_status['diversity_score'] < 6:
                reasons.append({
                    'reason': 'Điểm đa dạng PPDG thấp',
                    'detail': f'Điểm đa dạng: {current_status["diversity_score"]:.1f}/10 (cần cải thiện)',
                    'severity': 'TRUNG BÌNH'
                })
            
            # Kiểm tra hiệu quả PPDG dựa trên điểm dự đoán
            if predicted_score < 4.0 and effectiveness.get('level') == 'THẤP':
                reasons.append({
                    'reason': 'Phương pháp đánh giá chưa hiệu quả',
                    'detail': f'PPDG hiện tại có hiệu quả thấp với điểm dự đoán {predicted_score:.2f}/6',
                    'severity': 'CAO'
                })
            
            # Kiểm tra cân bằng giữa đánh giá quá trình và tổng kết
            if current_status['formative_count'] == 0:
                reasons.append({
                    'reason': 'Thiếu đánh giá quá trình',
                    'detail': 'Không có PPDG đánh giá quá trình (EM 1-5)',
                    'severity': 'TRUNG BÌNH'
                })
            elif current_status['summative_count'] == 0:
                reasons.append({
                    'reason': 'Thiếu đánh giá tổng kết',
                    'detail': 'Không có PPDG đánh giá tổng kết (EM 6-12)',
                    'severity': 'TRUNG BÌNH'
                })
            
            # Kiểm tra tính tương thích PPDG - Phương pháp giảng dạy
            if compatibility and compatibility.get('compatibility_score', 10) < 6:
                reasons.append({
                    'reason': 'PPDG không tương thích với phương pháp giảng dạy',
                    'detail': f'Điểm tương thích: {compatibility["compatibility_score"]:.1f}/10 - {compatibility["compatible_ppdg_count"]}/{compatibility["total_ppdg_count"]} PPDG tương thích',
                    'severity': 'TRUNG BÌNH'
                })
        
        # 7. Tổng hợp khuyến nghị dựa trên điểm dự đoán
        if predicted_score >= 4.5 and len(reasons) == 0:
            recommendations.append({
                'type': 'overall',
                'title': 'Kết quả dự đoán tốt',
                'detail': f'Điểm dự đoán: {predicted_score:.2f}/6',
                'suggestion': 'Cách dạy hợp lý, cách đánh giá ổn, không cần khắc phục gì'
            })
        
        # 8. Khuyến nghị cải thiện PPGD và PPDG dựa trên phân tích
        if len(reasons) > 0:
            # Khuyến nghị cải thiện PPGD (Phương pháp giảng dạy)
            ppgd_recommendations = []
            
            # Kiểm tra các lý do liên quan đến giảng viên
            lecturer_reasons = [r for r in reasons if 'giảng viên' in r['reason'].lower()]
            if lecturer_reasons:
                ppgd_recommendations.append("Tăng cường đào tạo phương pháp giảng dạy cho giảng viên mới")
                ppgd_recommendations.append("Áp dụng các phương pháp giảng dạy hiện đại và tương tác")
                ppgd_recommendations.append("Tăng cường thực hành và bài tập thực tế")
            
            # Kiểm tra các lý do liên quan đến môn học
            subject_reasons = [r for r in reasons if 'môn học' in r['reason'].lower()]
            if subject_reasons:
                ppgd_recommendations.append("Cải thiện cấu trúc nội dung môn học")
                ppgd_recommendations.append("Tăng cường liên kết giữa lý thuyết và thực hành")
                ppgd_recommendations.append("Áp dụng phương pháp dạy học tích cực")
            
            # Kiểm tra các lý do liên quan đến sinh viên
            student_reasons = [r for r in reasons if 'sinh viên' in r['reason'].lower() and 'giảng viên' not in r['reason'].lower()]
            if student_reasons:
                ppgd_recommendations.append("Điều chỉnh phương pháp giảng dạy phù hợp với trình độ sinh viên")
                ppgd_recommendations.append("Tăng cường hỗ trợ và hướng dẫn cá nhân")
                ppgd_recommendations.append("Áp dụng phương pháp dạy học phân hóa")
            
            # Thêm khuyến nghị PPGD
            if ppgd_recommendations:
                recommendations.append({
                    'type': 'ppgd_improvement',
                    'title': 'Khuyến nghị cải thiện PPGD (Phương pháp giảng dạy)',
                    'detail': f'Dựa trên {len(reasons)} lý do điểm thấp được phát hiện',
                    'suggestion': ' | '.join(ppgd_recommendations[:3])  # Giới hạn 3 khuyến nghị chính
                })
            
            # Khuyến nghị cải thiện PPDG (Phương pháp đánh giá) - CHI TIẾT HƠN
            if ppdg_analysis and 'current_status' in ppdg_analysis:
                current_status = ppdg_analysis['current_status']
                effectiveness = ppdg_analysis.get('effectiveness', {})
                compatibility = ppdg_analysis.get('compatibility', {})
                teaching_improvements = ppdg_analysis.get('teaching_improvements', [])
                
                # Phân tích PPDG hiện tại và đưa ra khuyến nghị cụ thể
                ppdg_specific_recommendations = self.generate_specific_ppdg_recommendations(
                    ppdg_analysis, predicted_score, reasons
                )
                
                if ppdg_specific_recommendations:
                    recommendations.append({
                        'type': 'ppdg_specific_improvement',
                        'title': 'Khuyến nghị cải thiện PPDG cụ thể',
                        'detail': f'Điểm dự đoán: {predicted_score:.2f}/6 - Phân tích chi tiết PPDG',
                        'suggestion': ppdg_specific_recommendations
                    })
            
            # Khuyến nghị tổng thể
            if len(reasons) >= 3:
                recommendations.append({
                    'type': 'comprehensive_improvement',
                    'title': 'Khuyến nghị cải thiện toàn diện',
                    'detail': f'Cần cải thiện cả PPGD và PPDG để nâng cao hiệu quả học tập',
                    'suggestion': 'Kết hợp cải thiện phương pháp giảng dạy với đa dạng hóa đánh giá'
                })
        
        return reasons, recommendations

    def generate_specific_ppdg_recommendations(self, ppdg_analysis, predicted_score, reasons):
        """Tạo khuyến nghị cụ thể về PPDG cần cải thiện"""
        recommendations = []
        
        current_status = ppdg_analysis.get('current_status', {})
        effectiveness = ppdg_analysis.get('effectiveness', {})
        compatibility = ppdg_analysis.get('compatibility', {})
        teaching_improvements = ppdg_analysis.get('teaching_improvements', [])
        
        # 1. PPDG cần BỔ SUNG
        ppdg_to_add = []
        
        # Kiểm tra thiếu đánh giá quá trình
        if current_status.get('formative_count', 0) == 0:
            ppdg_to_add.extend(['EM 1', 'EM 2', 'EM 3'])
        
        # Kiểm tra thiếu đánh giá tổng kết
        if current_status.get('summative_count', 0) == 0:
            ppdg_to_add.extend(['EM 6', 'EM 7', 'EM 8'])
        
        # Kiểm tra thiếu đánh giá thực hành
        if current_status.get('practical_count', 0) == 0:
            ppdg_to_add.extend(['EM 4', 'EM 5'])
        
        # Loại bỏ trùng lặp
        ppdg_to_add = list(set(ppdg_to_add))
        
        if ppdg_to_add:
            ppdg_names = {
                'EM 1': 'Đánh giá thường xuyên',
                'EM 2': 'Đánh giá định kỳ', 
                'EM 3': 'Đánh giá quá trình',
                'EM 4': 'Đánh giá thực hành',
                'EM 5': 'Đánh giá dự án',
                'EM 6': 'Đánh giá giữa kỳ',
                'EM 7': 'Đánh giá cuối kỳ',
                'EM 8': 'Đánh giá tổng kết'
            }
            
            ppdg_list = [f"{code} ({ppdg_names.get(code, code)})" for code in ppdg_to_add]
            recommendations.append(f"BỔ SUNG PPDG: {', '.join(ppdg_list)}")
        
        # 2. PPDG cần CẢI THIỆN
        ppdg_to_improve = []
        
        # Kiểm tra điểm đa dạng thấp
        if current_status.get('diversity_score', 10) < 6:
            ppdg_to_improve.append("Tăng cường đa dạng hóa PPDG")
        
        # Kiểm tra hiệu quả thấp
        if effectiveness.get('level') == 'THẤP':
            ppdg_to_improve.append("Cải thiện hiệu quả đánh giá")
        
        # Kiểm tra tương thích thấp
        if compatibility and compatibility.get('compatibility_score', 10) < 6:
            incompatible_ppdg = compatibility.get('incompatible_ppdg', [])
            if incompatible_ppdg:
                ppdg_to_improve.append(f"Điều chỉnh PPDG không tương thích: {', '.join(incompatible_ppdg)}")
        
        if ppdg_to_improve:
            recommendations.append(f"CẢI THIỆN: {' | '.join(ppdg_to_improve)}")
        
        # 3. PPDG cần BỎ (nếu có)
        ppdg_to_remove = []
        
        # Kiểm tra PPDG không hiệu quả
        if effectiveness.get('level') == 'THẤP' and predicted_score < 3.5:
            ppdg_to_remove.append("Xem xét loại bỏ PPDG có hiệu quả thấp")
        
        # Kiểm tra PPDG không tương thích nghiêm trọng
        if compatibility and compatibility.get('compatibility_score', 10) < 4:
            ppdg_to_remove.append("Loại bỏ PPDG không tương thích với phương pháp giảng dạy")
        
        if ppdg_to_remove:
            recommendations.append(f"BỎ: {' | '.join(ppdg_to_remove)}")
        
        # 4. Khuyến nghị cụ thể dựa trên điểm dự đoán
        if predicted_score < 4.0:
            if current_status.get('total_ppdg', 0) < 5:
                recommendations.append("TĂNG SỐ LƯỢNG: Cần ít nhất 5-6 PPDG để đánh giá toàn diện")
            
            if current_status.get('formative_count', 0) < 2:
                recommendations.append("TĂNG ĐÁNH GIÁ QUÁ TRÌNH: Cần ít nhất 2 PPDG đánh giá quá trình")
            
            if current_status.get('summative_count', 0) < 2:
                recommendations.append("TĂNG ĐÁNH GIÁ TỔNG KẾT: Cần ít nhất 2 PPDG đánh giá tổng kết")
        
        # 5. Khuyến nghị từ teaching_improvements
        if teaching_improvements:
            tm_suggestions = []
            for improvement in teaching_improvements:
                if improvement['type'] == 'ADD_TEACHING_METHOD':
                    tm_suggestions.append(f"Bổ sung TM cho {improvement['ppdg']}")
                elif improvement['type'] == 'ADD_PPDG':
                    tm_suggestions.append(f"Bổ sung {improvement['ppdg']}")
            
            if tm_suggestions:
                recommendations.append(f"TƯƠNG THÍCH: {' | '.join(tm_suggestions[:3])}")
        
        return ' | '.join(recommendations) if recommendations else "Không có khuyến nghị đặc biệt"

    def predict(self, student_id, lecturer, subject_id):
        """Make prediction for a student"""
        try:
            # Validate inputs
            if not self.data_loader.validate_input('student_id', student_id):
                return {'error': True, 'message': 'Invalid student ID'}
            
            if not self.data_loader.validate_input('lecturer', lecturer):
                return {'error': True, 'message': 'Invalid lecturer name'}
            
            if not self.data_loader.validate_input('subject_id', subject_id):
                return {'error': True, 'message': 'Invalid subject ID'}
            
            # Get student info
            student_info = self.get_student_info(student_id)
            if student_info is None or len(student_info) == 0:
                return {'error': True, 'message': 'Student not found'}
            
            # Get statistics
            student_history = self.get_student_history(student_id)
            subject_stats = self.get_subject_stats(subject_id)
            lecturer_stats = self.get_lecturer_stats(lecturer)
            
            # Prepare input features
            input_data = self.df[(self.df['Student_ID'] == str(student_id)) & 
                               (self.df['Lecturer_Name'] == lecturer) & 
                               (self.df['Subject_ID'] == str(subject_id))]
            
            if len(input_data) == 0:
                # Create synthetic data for prediction
                student_data = self.df[(self.df['Student_ID'] == str(student_id))]
                if len(student_data) == 0:
                    # Thử với kiểu int
                    try:
                        student_id_int = int(student_id)
                        student_data = self.df[self.df['Student_ID'] == student_id_int]
                    except ValueError:
                        pass
                
                if len(student_data) > 0:
                    input_data = student_data.iloc[:1].copy()
                    input_data['Lecturer_Name'] = lecturer
                    input_data['Subject_ID'] = str(subject_id)
                    
                    # Xử lý giảng viên mới
                    try:
                        input_data['lecturer_encoded'] = self.data_loader.le_lecturer.transform([lecturer])[0]
                    except ValueError:
                        # Nếu giảng viên mới không có trong encoder, sử dụng giá trị mặc định
                        print(f"⚠️ Giảng viên mới '{lecturer}' - sử dụng encoding mặc định")
                        input_data['lecturer_encoded'] = 0  # Giá trị mặc định cho giảng viên mới
                    
                    input_data['subject_encoded'] = self.data_loader.le_subject.transform([subject_id])[0]
                else:
                    return {'error': True, 'message': 'Cannot create prediction data - student not found in processed data'}
            
            # Prepare features for prediction
            X_pred = input_data[self.data_loader.feature_names].iloc[0:1]
            
            # Convert to numeric
            for col in X_pred.columns:
                X_pred[col] = X_pred[col].apply(safe_float)
            
            # Make prediction
            prob_pass = self.model.predict_proba(X_pred)[0, 1]
            predicted_score = prob_pass * 6.0  # Convert to scale 6
            
            # Get PPDG analysis if available
            ppdg_analysis = None
            try:
                if hasattr(self, 'ppdg_integration'):
                    student_data = {'student_id': student_id}
                    ppdg_analysis = self.ppdg_integration.analyze_ppdg_effectiveness(
                        student_data, subject_id, predicted_score
                    )
            except Exception as e:
                print(f"⚠️ PPDG analysis not available: {e}")
            
            # Luôn sử dụng mock PPDG analysis để test khuyến nghị cụ thể
            if ppdg_analysis is None:
                ppdg_analysis = self.create_mock_ppdg_analysis(subject_id, predicted_score)
                print("✅ Sử dụng PPDG analysis giả lập để tạo khuyến nghị cụ thể")
            
            # Analyze prediction reasons including PPDG
            analysis_result = self.analyze_prediction_reasons(
                student_id, lecturer, subject_id, predicted_score, ppdg_analysis
            )
            reasons = analysis_result[0]
            recommendations = analysis_result[1]
            
            return {
                'error': False,
                'predicted_score': predicted_score,
                'prob_pass': prob_pass,
                'student_info': student_info,
                'student_history': student_history,
                'subject_stats': subject_stats,
                'lecturer_stats': lecturer_stats,
                'ppdg_analysis': ppdg_analysis,  # Thêm PPDG analysis để debug
                'analysis': {
                    'risk_level': self.get_risk_level(1 - prob_pass),
                    'reasons': reasons,
                    'recommendations': recommendations
                }
            }
            
        except Exception as e:
            return {'error': True, 'message': f'Prediction error: {str(e)}'}

    def print_student_demographic_summary(self, student_id):
        """Print student demographic summary"""
        student_data = self.df[self.df['Student_ID'] == str(student_id)]
        if len(student_data) == 0:
            print("❌ Không tìm thấy dữ liệu sinh viên!")
            return
        
        print("\n📊 TÓM TẮT NHÂN KHẨU HỌC:")
        
        # Gender
        if 'gender_encoded' in student_data.columns:
            gender_val = student_data['gender_encoded'].iloc[0]
            try:
                gender_name = self.data_loader.le_gender.inverse_transform([gender_val])[0]
                print(f"👤 Giới tính: {gender_name}")
            except:
                print(f"👤 Giới tính: {gender_val}")
        
        # Religion
        if 'religion_encoded' in student_data.columns:
            religion_val = student_data['religion_encoded'].iloc[0]
            try:
                religion_name = self.data_loader.le_religion.inverse_transform([religion_val])[0]
                print(f"⛪ Tôn giáo: {religion_name}")
            except:
                print(f"⛪ Tôn giáo: {religion_val}")
        
        # Birth place
        if 'birth_place_encoded' in student_data.columns:
            birth_place_val = student_data['birth_place_encoded'].iloc[0]
            try:
                birth_place_name = self.data_loader.le_birth_place.inverse_transform([birth_place_val])[0]
                print(f"🏠 Nơi sinh: {birth_place_name}")
            except:
                print(f"🏠 Nơi sinh: {birth_place_val}")
        
        # Ethnicity
        if 'ethnicity_encoded' in student_data.columns:
            ethnicity_val = student_data['ethnicity_encoded'].iloc[0]
            try:
                ethnicity_name = self.data_loader.le_ethnicity.inverse_transform([ethnicity_val])[0]
                print(f"👥 Dân tộc: {ethnicity_name}")
            except:
                print(f"👥 Dân tộc: {ethnicity_val}")

    def print_student_conduct_summary(self, student_id):
        """Print student conduct summary"""
        student_data = self.df[self.df['Student_ID'] == str(student_id)]
        if len(student_data) == 0:
            print("❌ Không tìm thấy dữ liệu sinh viên!")
            return
        
        print("\n📈 TÓM TẮT ĐIỂM RÈN LUYỆN:")
        
        # Conduct scores
        if 'avg_conduct_score' in student_data.columns:
            avg_conduct = student_data['avg_conduct_score'].iloc[0]
            if pd.notna(avg_conduct):
                print(f"📊 Điểm rèn luyện trung bình: {avg_conduct:.1f}")
        
        if 'latest_conduct_score' in student_data.columns:
            latest_conduct = student_data['latest_conduct_score'].iloc[0]
            if pd.notna(latest_conduct):
                print(f"📊 Điểm rèn luyện gần nhất: {latest_conduct:.1f}")
        
        if 'conduct_trend' in student_data.columns:
            trend = student_data['conduct_trend'].iloc[0]
            if pd.notna(trend):
                if trend > 0:
                    print(f"📈 Xu hướng: Cải thiện (+{trend:.1f})")
                elif trend < 0:
                    print(f"📉 Xu hướng: Giảm sút ({trend:.1f})")
                else:
                    print(f"➡️ Xu hướng: Ổn định")
        
        # Conduct classification
        if 'latest_conduct_classification' in student_data.columns:
            classification = student_data['latest_conduct_classification'].iloc[0]
            if pd.notna(classification):
                print(f"🏆 Phân loại: {classification}")
        
        # Number of semesters with conduct data
        if 'num_conduct_semesters' in student_data.columns:
            num_semesters = student_data['num_conduct_semesters'].iloc[0]
            if pd.notna(num_semesters):
                print(f"📚 Số học kỳ có dữ liệu: {num_semesters}")
        
        # Conduct impact analysis
        print("\n🔍 PHÂN TÍCH TÁC ĐỘNG ĐIỂM RÈN LUYỆN:")
        
        if 'avg_conduct_score' in student_data.columns:
            avg_conduct = student_data['avg_conduct_score'].iloc[0]
            if pd.notna(avg_conduct):
                if avg_conduct >= 90:
                    print("✅ Điểm rèn luyện trung bình rất cao - Có thể hỗ trợ tích cực cho kết quả học tập")
                elif avg_conduct >= 80:
                    print("✅ Điểm rèn luyện trung bình tốt - Có tác động tích cực đến học tập")
                elif avg_conduct >= 70:
                    print("⚠️ Điểm rèn luyện trung bình khá - Cần cải thiện để nâng cao kết quả")
                elif avg_conduct >= 60:
                    print("⚠️ Điểm rèn luyện trung bình thấp - Có thể ảnh hưởng tiêu cực đến học tập")
                else:
                    print("❌ Điểm rèn luyện trung bình rất thấp - Cần can thiệp để cải thiện")
        
        if 'latest_conduct_score' in student_data.columns and 'avg_conduct_score' in student_data.columns:
            latest_conduct = student_data['latest_conduct_score'].iloc[0]
            avg_conduct = student_data['avg_conduct_score'].iloc[0]
            if pd.notna(latest_conduct) and pd.notna(avg_conduct):
                if latest_conduct > avg_conduct + 5:
                    print("📈 Điểm rèn luyện gần nhất có xu hướng cải thiện - Dấu hiệu tích cực")
                elif latest_conduct < avg_conduct - 5:
                    print("📉 Điểm rèn luyện gần nhất có xu hướng giảm - Cần quan tâm")
                else:
                    print("➡️ Điểm rèn luyện gần nhất ổn định so với trung bình") 

    def create_mock_ppdg_analysis(self, subject_id, predicted_score):
        """Tạo PPDG analysis giả lập để test khuyến nghị cụ thể"""
        # Tạo dữ liệu PPDG giả lập dựa trên điểm dự đoán
        if predicted_score < 4.0:
            # Điểm thấp - có vấn đề với PPDG
            current_status = {
                'total_ppdg': 3,  # Ít PPDG
                'formative_count': 1,  # Ít đánh giá quá trình
                'summative_count': 2,  # Ít đánh giá tổng kết
                'practical_count': 0,  # Không có đánh giá thực hành
                'diversity_score': 4.5  # Điểm đa dạng thấp
            }
            effectiveness = {
                'level': 'THẤP',
                'score': 3.2
            }
            compatibility = {
                'compatibility_score': 5.5,
                'incompatible_ppdg': ['EM 4', 'EM 5']
            }
            teaching_improvements = [
                {
                    'type': 'ADD_TEACHING_METHOD',
                    'ppdg': 'EM 4',
                    'suggested_tm': ['TM4', 'TM9']
                },
                {
                    'type': 'ADD_PPDG',
                    'ppdg': 'EM 1',
                    'suggested_tm': ['TM1', 'TM2']
                }
            ]
        elif predicted_score < 4.5:
            # Điểm trung bình - cần cải thiện một số PPDG
            current_status = {
                'total_ppdg': 5,
                'formative_count': 2,
                'summative_count': 3,
                'practical_count': 0,
                'diversity_score': 6.5
            }
            effectiveness = {
                'level': 'TRUNG BÌNH',
                'score': 5.8
            }
            compatibility = {
                'compatibility_score': 7.0,
                'incompatible_ppdg': ['EM 5']
            }
            teaching_improvements = [
                {
                    'type': 'ADD_PPDG',
                    'ppdg': 'EM 5',
                    'suggested_tm': ['TM5', 'TM21']
                }
            ]
        else:
            # Điểm cao - PPDG tốt
            current_status = {
                'total_ppdg': 7,
                'formative_count': 3,
                'summative_count': 4,
                'practical_count': 1,
                'diversity_score': 8.5
            }
            effectiveness = {
                'level': 'CAO',
                'score': 7.8
            }
            compatibility = {
                'compatibility_score': 8.5,
                'incompatible_ppdg': []
            }
            teaching_improvements = []
        
        return {
            'current_status': current_status,
            'effectiveness': effectiveness,
            'compatibility': compatibility,
            'teaching_improvements': teaching_improvements
        } 

    def analyze_ppdg_effectiveness(self, student_id, lecturer, subject_id):
        """Phân tích hiệu quả PPDG cho sinh viên và môn học"""
        try:
            # Tạo mock PPDG analysis (có thể thay thế bằng dữ liệu thực tế)
            ppdg_analysis = self.create_mock_ppdg_analysis(subject_id, 4.5)  # Giả sử điểm dự đoán 4.5
            
            if ppdg_analysis and 'effectiveness' in ppdg_analysis:
                effectiveness_score = ppdg_analysis['effectiveness'].get('score', 75.0)
                return {
                    'effectiveness': effectiveness_score,
                    'recommendations': 'Cần cải thiện PPDG'
                }
            else:
                return None
                
        except Exception as e:
            print(f"❌ Lỗi khi phân tích PPDG: {e}")
            return None 

class CLOPredictor:
    """CLO Prediction System Main Class"""
    
    def __init__(self, optimize_params=False, load_from_file=True):
        """
        Initialize the CLO prediction system
        
        Args:
            optimize_params: Có tối ưu tham số khi train không (chỉ dùng khi train mới)
            load_from_file: Có load model từ file đã train sẵn không (True = load, False = train mới)
        """
        import os
        import pickle
        
        self.optimize_params = optimize_params
        self.reasons_predictor = None  # Will be set from main.py
        
        # Đường dẫn đến model đã train
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                  "trained_models", "clo_predictor", "clo_predictor.pkl")
        
        # Thử load model từ file nếu được yêu cầu
        if load_from_file and os.path.exists(model_path):
            print("🔄 Đang load CLOPredictor từ file đã train sẵn...")
            try:
                with open(model_path, 'rb') as f:
                    loaded_predictor = pickle.load(f)
                
                # Copy tất cả attributes từ loaded model
                self.data_loader = loaded_predictor.data_loader
                self.data_integration = loaded_predictor.data_integration
                self.feature_engineering = loaded_predictor.feature_engineering
                self.model_trainer = loaded_predictor.model_trainer
                self.predictor = loaded_predictor.predictor
                
                print("✅ Đã load CLOPredictor thành công từ file!")
                return
                
            except Exception as e:
                print(f"⚠️  Không thể load model từ file: {e}")
                print("🔄 Sẽ train model mới...")
        
        # Nếu không load được hoặc không có file, train mới
        print("Initializing CLO Prediction System...")
        
        # Initialize components
        from .data_loader import DataLoader
        from .data_integration import DataIntegration
        from .feature_engineering import FeatureEngineering
        from .model_trainer import ModelTrainer
        
        self.data_loader = DataLoader()
        self.data_integration = None
        self.feature_engineering = None
        self.model_trainer = None
        self.predictor = None
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Train models
        self.train_models()

    def load_and_prepare_data(self):
        """Load and prepare all data"""
        print("=== LOADING AND PREPARING DATA ===")
        
        # Load main data
        self.data_loader.load_main_data()
        self.data_loader.process_main_data()
        
        # Load additional data
        self.data_loader.load_demographic_data()
        self.data_loader.load_conduct_data()
        self.data_loader.load_self_study_data()
        
        # Initialize data integration
        from .data_integration import DataIntegration
        self.data_integration = DataIntegration(self.data_loader)
        
        # Integrate all data
        self.data_integration.integrate_demographic_data()
        self.data_integration.integrate_conduct_data()
        self.data_integration.integrate_self_study_data()
        
        # Create features
        self.data_integration.create_teaching_method_features()
        self.data_integration.create_assessment_method_features()
        
        # Initialize feature engineering
        from .feature_engineering import FeatureEngineering
        self.feature_engineering = FeatureEngineering(self.data_loader)
        
        # Add advanced features
        self.feature_engineering.add_student_history_features()
        self.feature_engineering.add_advanced_student_features()
        self.feature_engineering.add_personalized_features()
        
        # Finalize features
        self.data_integration.finalize_features()
        
        # Print demographic statistics
        self.feature_engineering.print_demographic_statistics()

    def train_models(self):
        """Train the prediction models"""
        print("\n=== TRAINING MODELS ===")
        
        # Initialize model trainer
        from .model_trainer import ModelTrainer
        self.model_trainer = ModelTrainer(self.data_loader)
        
        # Prepare data for training
        self.model_trainer.prepare_data()
        
        # Train models (tối ưu tham số nếu được yêu cầu)
        training_results = self.model_trainer.train_models(optimize_params=self.optimize_params)
        
        # Evaluate model
        evaluation_results = self.model_trainer.evaluate_model()
        
        # Initialize predictor
        self.predictor = Predictor(self.data_loader, self.model_trainer)
        
        print("Model training completed!")

    def get_available_options(self):
        """Get available options for input validation"""
        return self.data_loader.get_available_options()

    def validate_input(self, input_type, value):
        """Validate input against available options"""
        return self.data_loader.validate_input(input_type, value)
    
    def get_subject_name(self, subject_id):
        """Lấy tên môn học từ mã môn học"""
        # Mapping một số mã môn học với tên
        subject_mapping = {
            'BSC0092': 'Phương pháp luận nghiên cứu khoa học',
            'BSC0091': 'Toán học rời rạc',
            'BSC0090': 'Lập trình cơ bản',
            'BSC0089': 'Cơ sở dữ liệu',
            'BSC0088': 'Mạng máy tính',
            'BSC0087': 'Hệ điều hành',
            'BSC0086': 'Cấu trúc dữ liệu và giải thuật',
            'BSC0085': 'Lập trình hướng đối tượng',
            'BSC0084': 'Kiến trúc máy tính',
            'BSC0083': 'Cơ sở lập trình',
            'BSC0082': 'Tin học đại cương',
            'BSC0081': 'Toán học cho tin học',
            'BSC0080': 'Vật lý đại cương',
            'BSC0079': 'Hóa học đại cương',
            'BSC0078': 'Sinh học đại cương',
            'BSC0077': 'Tiếng Anh chuyên ngành',
            'BSC0076': 'Kỹ năng mềm',
            'BSC0075': 'Giáo dục thể chất',
            'BSC0074': 'Giáo dục quốc phòng',
            'BSC0073': 'Chủ nghĩa xã hội khoa học',
            'BSC0072': 'Tư tưởng Hồ Chí Minh',
            'BSC0071': 'Lịch sử Đảng Cộng sản Việt Nam',
            'BSC0070': 'Triết học Mác - Lênin',
            'BSC0069': 'Kinh tế chính trị Mác - Lênin',
            'BSC0068': 'Chủ nghĩa Mác - Lênin',
            'BSC0067': 'Đường lối cách mạng Đảng Cộng sản Việt Nam',
            'BSC0066': 'Tư tưởng Hồ Chí Minh',
            'BSC0065': 'Lịch sử Đảng Cộng sản Việt Nam',
            'BSC0064': 'Triết học Mác - Lênin',
            'BSC0063': 'Kinh tế chính trị Mác - Lênin',
            'BSC0062': 'Chủ nghĩa Mác - Lênin',
            'BSC0061': 'Đường lối cách mạng Đảng Cộng sản Việt Nam',
            'BSC0060': 'Tư tưởng Hồ Chí Minh',
            'BSC0059': 'Lịch sử Đảng Cộng sản Việt Nam',
            'BSC0058': 'Triết học Mác - Lênin',
            'BSC0057': 'Kinh tế chính trị Mác - Lênin',
            'BSC0056': 'Chủ nghĩa Mác - Lênin',
            'BSC0055': 'Đường lối cách mạng Đảng Cộng sản Việt Nam',
            'BSC0054': 'Tư tưởng Hồ Chí Minh',
            'BSC0053': 'Lịch sử Đảng Cộng sản Việt Nam',
            'BSC0052': 'Triết học Mác - Lênin',
            'BSC0051': 'Kinh tế chính trị Mác - Lênin',
            'BSC0050': 'Chủ nghĩa Mác - Lênin'
        }
        
        return subject_mapping.get(subject_id, f'Môn học {subject_id}')

    def display_available_options(self):
        """Display available options for user input"""
        self.data_loader.display_available_options()

    def predict(self, student_id, lecturer, subject_id):
        """Make prediction for a student"""
        try:
            result = self.predictor.predict(student_id, lecturer, subject_id)
            if result and 'predicted_score' in result:
                return result['predicted_score']
            else:
                return None
        except Exception as e:
            print(f"❌ Lỗi khi dự đoán: {e}")
            return None

    def analyze_prediction_reasons(self, student_id, lecturer, subject_id, predicted_score):
        """Analyze reasons for prediction and provide recommendations"""
        try:
            reasons, recommendations = self.predictor.analyze_prediction_reasons(
                student_id, lecturer, subject_id, predicted_score
            )
            return reasons, recommendations
        except Exception as e:
            print(f"❌ Lỗi khi phân tích nguyên nhân: {e}")
            return [], []

    def analyze_ppdg_effectiveness(self, student_id, lecturer, subject_id):
        """Analyze PPDG effectiveness"""
        try:
            return self.predictor.analyze_ppdg_effectiveness(student_id, lecturer, subject_id)
        except Exception as e:
            print(f"❌ Lỗi khi phân tích PPDG: {e}")
            return None

    def print_student_demographic_summary(self, student_id):
        """Print student demographic summary"""
        self.predictor.print_student_demographic_summary(student_id)

    def print_student_conduct_summary(self, student_id):
        """Print student conduct summary"""
        self.predictor.print_student_conduct_summary(student_id) 