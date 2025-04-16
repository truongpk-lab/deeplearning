Báo cáo Tóm tắt Kết quả
1. Đề bài:
Mục tiêu của bài toán là phân loại cảm xúc của các đánh giá phim trong bộ dữ liệu IMDB (IMDB Movie Reviews) thành hai lớp: tích cực (positive) và tiêu cực (negative). Dữ liệu bao gồm các đoạn văn bản và nhãn tương ứng, yêu cầu xây dựng một mô hình học sâu để phân loại chính xác cảm xúc của mỗi đoạn văn bản.

2. Tiền xử lý dữ liệu:
Dữ liệu: Bộ dữ liệu IMDB được lấy từ IMDB Dataset.csv, chứa 25.000 đánh giá phim với nhãn 'positive' hoặc 'negative'. Dữ liệu được chia thành 20.000 mẫu cho tập huấn luyện và 5.000 mẫu cho tập kiểm tra.

Tokenization: Sử dụng tokenizer cơ bản basic_english từ thư viện torchtext để phân tách các từ trong mỗi đoạn văn bản.

Vocab: Tạo từ điển từ các từ trong dữ liệu với tối đa 10.000 từ, sử dụng GloVe 100D để xây dựng ma trận embedding cho từ vựng.

Chuyển đổi văn bản thành tensor: Mỗi đánh giá được chuyển thành một chuỗi chỉ số tương ứng với các từ trong từ điển và được padding đến độ dài tối đa là 300.

3. Mô hình:
Mô hình Transformer-lite: Mô hình học sâu sử dụng kiến trúc Transformer đơn giản với các lớp Transformer Encoder để xử lý chuỗi đầu vào. Mô hình này bao gồm:

Embedding layer: Sử dụng ma trận embedding từ GloVe với kích thước 100.

Positional Encoding: Thêm thông tin vị trí vào đầu vào của mô hình để mô hình hiểu được thứ tự các từ trong câu.

Transformer Encoder: Sử dụng lớp TransformerEncoderLayer với 2 lớp encoder và 4 đầu attention.

Fully connected layer: Sau khi qua lớp Transformer, dữ liệu được đưa qua một lớp fully connected với kích thước đầu ra là 1, đại diện cho xác suất tích cực.

4. Siêu tham số:
Các cấu hình siêu tham số được thử nghiệm bao gồm:

Cấu hình 1: batch_size = 32, learning_rate = 1e-3, hidden_dim = 128, activation_fn = ReLU, optimizer = Adam

Cấu hình 2: batch_size = 64, learning_rate = 5e-4, hidden_dim = 256, activation_fn = ReLU, optimizer = AdamW

Cấu hình 3: batch_size = 128, learning_rate = 1e-3, hidden_dim = 512, activation_fn = Tanh, optimizer = SGD

Cấu hình 4: batch_size = 64, learning_rate = 1e-4, hidden_dim = 256, activation_fn = GELU, optimizer = Adam

Cấu hình 5: batch_size = 32, learning_rate = 5e-4, hidden_dim = 128, activation_fn = ReLU, optimizer = RMSprop

5. Kết quả:
Cấu hình 1: Mean Accuracy = 87.13%, Std = 0.40%
Mô hình với cấu hình này có độ chính xác trung bình cao, ổn định và ít dao động giữa các lần chạy.

Cấu hình 2: Mean Accuracy = 87.49%, Std = 0.30%
Cấu hình này cho kết quả tương đương với cấu hình 1, nhưng với optimizer AdamW, có độ chính xác trung bình nhỉnh hơn một chút và độ lệch chuẩn thấp hơn.

Cấu hình 3: Mean Accuracy = 50.97%, Std = 0.82%
Cấu hình này với kích thước batch lớn và sử dụng optimizer SGD cho kết quả rất thấp (khoảng 50%), cho thấy rằng việc sử dụng Tanh activation function cùng với optimizer SGD không phù hợp cho bài toán này.

Cấu hình 4: Mean Accuracy = 84.89%, Std = 0.21%
Cấu hình này có độ chính xác trung bình thấp hơn so với các cấu hình 1 và 2. Sử dụng activation GELU có vẻ không hiệu quả trong trường hợp này.

Cấu hình 5: Mean Accuracy = 87.87%, Std = 0.19%
Cấu hình này cho kết quả chính xác trung bình cao nhất, đặc biệt khi sử dụng optimizer RMSprop và learning rate thấp hơn.

6. Nhận xét:
Cấu hình tối ưu: Cấu hình 5 cho kết quả tốt nhất với độ chính xác trung bình cao nhất (87.87%) và độ lệch chuẩn thấp (0.19%). Đây là cấu hình tốt nhất trong 5 cấu hình đã thử.

Cấu hình SGD: Cấu hình 3 sử dụng optimizer SGD và activation Tanh không đem lại kết quả tốt, cho thấy sự kết hợp này không phù hợp với bài toán phân loại văn bản cảm xúc.

Kết quả chung: Các cấu hình sử dụng Adam (hoặc AdamW) và activation ReLU cho kết quả cao hơn hẳn so với các cấu hình khác, chứng tỏ đây là lựa chọn tối ưu cho bài toán này.

Đánh giá độ ổn định: Các cấu hình có độ lệch chuẩn thấp (Config 2 và Config 5) cho thấy độ ổn định cao hơn, cho phép mô hình đạt kết quả tương tự qua nhiều lần huấn luyện.

7. Kết luận:
Cấu hình 5 là lựa chọn tốt nhất để tiếp tục phát triển mô hình, với độ chính xác cao và độ ổn định tốt.

Các thử nghiệm cho thấy việc chọn đúng optimizer và activation function rất quan trọng đối với kết quả huấn luyện.
