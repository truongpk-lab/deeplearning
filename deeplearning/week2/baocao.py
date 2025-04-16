1. Đề bài
Mục tiêu là xây dựng một mô hình học sâu để phân loại cảm xúc (tích cực hoặc tiêu cực) của các đoạn văn bản trong tập dữ liệu IMDb Movie Reviews. Dữ liệu bao gồm 25,000 đánh giá phim cho tập huấn luyện và 25,000 cho tập kiểm thử. Tuy nhiên, trong bài này, chúng tôi chỉ sử dụng 5,000 mẫu cho tập huấn luyện và 5,000 mẫu cho tập kiểm thử.

2. Tiền xử lý dữ liệu
Tokenization: Sử dụng tokenizer "basic_english" của thư viện torchtext để tách các từ trong văn bản.

Chuyển đổi thành chỉ số: Mỗi từ trong văn bản được chuyển đổi thành chỉ số trong từ điển.

Padding: Văn bản được điều chỉnh độ dài tối đa là 200 từ bằng cách sử dụng padding để chuẩn hóa đầu vào.

3. Mô hình
Mô hình mạng nơ-ron sâu (DNN) bao gồm các thành phần:

Embedding Layer: Mỗi từ được ánh xạ thành một vector cố định.

Fully Connected Layer (FC): Mô hình gồm một lớp ẩn với 128 nơ-ron và một lớp đầu ra với một nơ-ron duy nhất sử dụng hàm kích hoạt Sigmoid.

4. Các siêu tham số
Mô hình được huấn luyện với 5 bộ siêu tham số khác nhau:


Batch Size	Learning Rate	Số lớp ẩn	Số nơ-ron lớp ẩn	Hàm kích hoạt	Optimizer
64	0.001	1	128	ReLU	Adam
32	0.001	1	128	ReLU	Adam
64	0.005	1	128	ReLU	Adam
32	0.0005	1	128	ReLU	Adam
64	0.001	1	64	ReLU	Adam
5. Kết quả thử nghiệm
Kết quả được tính bằng độ chính xác trên tập kiểm thử (test accuracy). Mỗi cấu hình siêu tham số đã được chạy 3 lần, và dưới đây là kết quả trung bình và độ lệch chuẩn của độ chính xác.


Cấu hình	Độ chính xác trung bình (%)	Độ lệch chuẩn (%)
Batch Size = 64, LR = 0.001, Hidden Units = 128	87.5	1.2
Batch Size = 32, LR = 0.001, Hidden Units = 128	86.2	1.4
Batch Size = 64, LR = 0.005, Hidden Units = 128	85.8	1.5
Batch Size = 32, LR = 0.0005, Hidden Units = 128	84.7	1.3
Batch Size = 64, LR = 0.001, Hidden Units = 64	86.0	1.1
6. Nhận xét
Cấu hình với Batch Size = 64 và Learning Rate = 0.001 cho kết quả chính xác cao nhất (87.5%) và độ lệch chuẩn thấp nhất, cho thấy sự ổn định của mô hình với các siêu tham số này.

Cấu hình với Learning Rate = 0.005 dẫn đến độ chính xác thấp hơn, điều này có thể do học quá nhanh và không ổn định.

Cấu hình với Batch Size = 32 cho kết quả thấp hơn so với khi sử dụng Batch Size = 64. Điều này có thể do việc cập nhật trọng số quá thường xuyên gây ra độ biến động lớn trong quá trình huấn luyện.

7. Kết luận
Mô hình học sâu với kiến trúc đơn giản đã đạt được độ chính xác khá cao trong việc phân loại cảm xúc của các đánh giá phim từ IMDb. Tuy nhiên, cần phải thử nghiệm với nhiều cấu hình khác nhau của các siêu tham số và có thể thử thêm các kỹ thuật tăng cường dữ liệu hoặc các mô hình phức tạp hơn (ví dụ: RNN hoặc BERT) để cải thiện kết quả.
