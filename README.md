# chest_classification
Đồ án tốt nghiệp thạc sĩ

## Commit Message
Cú pháp: `loại(phạm vi): Mô tả`, ví dụ `model(cnn_symmamba): Add symmetric scan order`.

Nhóm làm thay đổi kết quả thí nghiệm:
- model: thêm / sửa kiến trúc mô hình.
- exp: siêu tham số, giao thức huấn luyện / đánh giá (loss, optimizer, early stopping, EXPERIMENT_PROTOCOL), tái hiện bài báo.
- data: tiền xử lý, chia fold, làm giàu ảnh.

Nhóm không làm thay đổi kết quả:
- feature: thêm tính năng cho pipeline / công cụ (hàng đợi, xuất bảng, so sánh mô hình).
- perf: tăng tốc, giảm bộ nhớ.
- log: ghi log, thống kê theo dõi.
- fix: sửa lỗi.
- refactor: chỉnh cấu trúc code nhưng không thêm tính năng / sửa lỗi.
- docs: sửa tài liệu.
- test: thêm / sửa kiểm thử code (không dùng cho thí nghiệm).
- chore: công việc phụ như dependency, .gitignore.
- style: format code, không thay đổi logic.

Thêm `!` sau phạm vi khi commit làm kết quả số khác đi so với trước (kể cả perf / fix), ví dụ `perf(train)!: Use fused AdamW`.
Commit chốt giao thức trước lượt chạy chính thức được gắn tag, ví dụ `git tag protocol-v1`, và trích dẫn tag này trong luận văn.