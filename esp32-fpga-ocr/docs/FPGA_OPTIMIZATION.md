# Báo cáo thử nghiệm tối ưu FPGA nhận dạng ký tự biển số

Ngày lập: 16/11/2026. Phạm vi: biển ô tô một hàng, 8 ký tự; ảnh mỗi ký tự là ảnh xám 28 × 28, 8 bit. Để đánh giá theo yêu cầu khó hơn, chỉ tiêu 100 fps được quy đổi thành **100 biển hoàn chỉnh mỗi giây**, tức 800 ký tự/giây, thay vì 100 ký tự/giây. Đây là thông lượng duy trì, khác với độ trễ từ lúc chụp đến khi nhận kết quả.

## 1. Thay đổi đã thực hiện

- Thiết kế lõi RTL `OcrBenchCoreK3_parallel8.sv` với tối đa 8 phép nhân–cộng song song theo kênh đầu ra hoặc neuron. Bộ nhớ trọng số được tổ chức lại thành từ 64 bit để cấp dữ liệu cho các phần tử tính toán.
- Duy trì cùng mô hình v4: đầu vào 28 × 28 → Conv1 6 kênh, Tanh, average pooling → Conv2 16 kênh, Tanh, average pooling → FC 120 → FC 84 → FC 30 lớp. Trọng số INT8, hệ số dịch, bias và LUT Tanh của lõi thí nghiệm được kiểm tra khớp checkpoint v4. Không huấn luyện lại mô hình trong lần tối ưu này.
- Dùng xung 50 MHz cho lõi mới; bản cũ chạy với `CORE_ENABLE_DIV2=1`. Mức tăng tốc bên dưới bao gồm cả thay đổi song song hóa và cách cho phép xung của lõi.
- Giữ bộ xử lý sau suy luận ở FPGA theo bản v4: softmax số nguyên, bảng hiệu chỉnh theo dữ liệu nhầm lẫn, ba ứng viên, quy tắc vị trí và quyết định nhận/từ chối. Nhãn chuẩn không được gửi cho mô hình trong lúc suy luận.
- Tạo bản build, ROM, testbench và dữ liệu đối chiếu trong thư mục thí nghiệm riêng. Không thay đổi dự án hoặc firmware đang dùng trên hai kit.

## 2. Mô phỏng RTL và độ chính xác

1. Tập 27 ảnh ký tự gồm 16 ký tự từ 2 biển tổng hợp validation, 8 ký tự từ một ảnh camera nhìn màn hình đã lưu, và 3 ảnh biên đen, trắng, nhiễu. So sánh 30 logit INT32 cho mỗi ảnh: **810/810 logit khớp bit** mô hình INT8 Python độc lập.
2. Tập kiểm thử khóa riêng: 100 biển một hàng tổng hợp trong phân vùng test v4, không thuộc tập train; tách được 800 ký tự. Mô phỏng lõi RTL và đối chiếu **24.000/24.000 logit** với mô hình số nguyên tham chiếu. Đầu ra thô của RTL đúng **800/800 ký tự** so với nhãn chấm sau suy luận.
3. Đưa các logit khớp bit qua thuật toán quyết định v4 bằng Python: **800/800 ký tự**, **100/100 biển đầy đủ**, **100/100 biển được nhận và đúng**, 0 biển được nhận nhưng sai. Phần quyết định trên RTL hoàn chỉnh được đối chiếu riêng trong test UART của 16 ký tự, không phải cả 800 ký tự. Bộ ảnh này là ảnh tổng hợp, có cùng nguồn sinh và các họ phông của bộ dữ liệu; kết quả không chứng minh độ chính xác đối với các biển thực tế độc lập.
4. Mô phỏng RTL toàn mạch UART với 2 biển/16 ký tự: kiểm tra khung truyền, mã CRC, frame ID, thứ tự ký tự, giá trị dự đoán, điểm top-1, cờ nhận/từ chối, input digest, lớp top-3 và báo nhận cả biển. Một gói cố ý sai CRC bị từ chối và mạch tiếp tục xử lý khung sau: `UART_SIM_PASS valid_glyphs=16 bad_crc=1 replies=17`.

Bộ 1.000 biển tổng hợp v4 đã đánh giá trước lần tối ưu có kết quả 1.000/1.000 biển và 8.000/8.000 ký tự. Đây là kết quả phần mềm trên ảnh tổng hợp của mô hình cũ; không thể suy ra tỉ lệ đạt 90% trên camera thực hoặc trên tập biển mới chỉ từ kết quả này.

## 3. Thời gian và thông lượng

| Phạm vi | Bản v4 đã đo trước | Lõi tối ưu trong mô phỏng |
|---|---:|---:|
| Chu kỳ suy luận mỗi ký tự | 872.541–872.542 | 50.910; bộ đếm ở mạch UART ghi 50.914 |
| Thời gian tương ứng tại 50 MHz | Khoảng 17,45 ms | 1,0182 ms/lõi; 1,01828 ms theo bộ đếm mạch UART |
| Tốc độ lõi tính từ số chu kỳ, bỏ qua nạp dữ liệu và điều khiển | Khoảng 57 ký tự/s | Khoảng 982 ký tự/s, hoặc 122,8 biển/s với 8 ký tự/biển |
| So sánh độ trễ lõi | 1× | Khoảng 17,1× nhanh hơn |

Giá trị 122,8 biển/s là phép tính từ thời gian lõi, không tính nạp ảnh, UART, hậu xử lý, chụp, tách ký tự hoặc giao diện; chưa phải tốc độ của hệ thống chạy thật. Lõi cần cấp dữ liệu đủ nhanh và xử lý các tầng còn lại trong ngân sách thời gian trước khi có thể kết luận đạt 100 biển/s trên FPGA ở mức hệ thống.

Với giao thức UART đang dùng, mỗi ký tự cần 790 byte yêu cầu và 25 byte phản hồi, truyền tuần tự 8N1. Với tốc độ danh định 1.843.200 baud, chỉ riêng thời gian truyền lý tưởng cho 8 ký tự là:

`8 × (790 + 25) × 10 / 1.843.200 ≈ 35,37 ms/biển`, tương đương giới hạn lý tưởng khoảng 28,3 biển/s trước mọi thời gian tính toán.

Trong mô phỏng UART ở 50 MHz, bộ chia 27 tương đương khoảng 1,85185 Mbaud; hai lượt truyền 8 ký tự cùng thời gian **43,69776 ms/biển**, tương đương khoảng **22,9 biển/s** từ đầu gói thứ nhất tới cuối gói thứ tám và khoảng nghỉ của testbench. Kết quả này chưa tính camera, cắt ảnh, truy cập mạng hoặc hiển thị. Thử nghiệm không mô phỏng hết sai số giữa xung hai kit hay nhiễu đường dây.

### Thí nghiệm UART có bộ đệm

Đã tạo thêm một thiết kế riêng trong `uart_buffered_experiment/`: UART có bộ chia 6, tốc độ danh nghĩa **8,333 Mbaud** ở xung FPGA 50 MHz, và FIFO nhận 2.048 byte kèm bit báo lỗi. Khi lõi CNN đang suy luận, FIFO vẫn tiếp nhận các gói kế tiếp. Nếu FIFO tràn, thiết kế giữ trạng thái lỗi, bật LEDR2 và dừng nhận dữ liệu cho đến khi reset; chưa có cơ chế điều khiển luồng trở lại ESP32.

Testbench phát **32 lượt biển**, mỗi lượt 8 ảnh ký tự, theo chu kỳ 10 ms và không đợi phản hồi từng ký tự. Ba biển mẫu được lặp lại để kiểm tra thông lượng; đây không phải bài đo độ chính xác của 32 biển độc lập. XSim xác nhận đủ **256 phản hồi** có ID, CRC, lớp, điểm top-1 và cờ nhận tương ứng với tham chiếu; từ chối một gói cố ý sai CRC và xử lý được gói tiếp theo. Khoảng cách giữa các kết quả biển là **10,000 ms**, tương ứng **100 biển/s** trong test này; thời gian từ bắt đầu gửi ký tự đầu đến nhận xong kết quả ký tự cuối là **9,700 ms/biển**, không gồm camera và tách ảnh. Mức FIFO lớn nhất là 1.390/2.048 byte.

Kết quả trên được tạo bởi testbench phát UART lý tưởng, không có nhiễu và không đo sai lệch xung vật lý. Firmware ESP32 đang dùng vẫn hoạt động ở 1.843.200 baud và đợi từng phản hồi; **không tương thích** với bản thử UART nhanh nếu chỉ nạp FPGA. Phải bổ sung bộ phát/nhận bất đồng bộ, điều khiển tốc độ và kiểm tra tràn trên ESP32, rồi xác minh tốc độ 8,333 Mbaud và CRC qua dây thật. Nếu không ổn định, cần thiết kế giao tiếp khác. Chưa thể khẳng định hệ thống camera → ESP32 → FPGA đạt 100 biển/s.

Nhật ký camera trước đây có một lượt ESP32 nhìn biển hiển thị trên màn hình: 582,370 ms từ lấy framebuffer tới khi có kết quả trên ESP32, gồm giải mã JPEG 268,972 ms, tách và tiền xử lý 131,159 ms, UART cộng FPGA 181,623 ms. Đây là một mẫu đo của bản cũ và không chứng minh thông lượng duy trì; nó cho thấy cần tối ưu cả các bước ngoài FPGA.

Để đạt 100 biển/s, toàn hệ thống cần xử lý trung bình một biển mỗi 10 ms. Dữ liệu ảnh thô là 8 × 28 × 28 × 8 = 50.176 bit/biển, tương đương tối thiểu 5,018 Mbit/s một chiều ở 100 biển/s, chưa kể khung truyền, kiểm lỗi và phản hồi. Cần thiết kế lại đường truyền và đệm, có khả năng truyền ảnh kế tiếp đồng thời với suy luận; sau đó đo tốc độ camera và tiền xử lý thực tế. Giữ giao thức UART hiện tại không thể đạt chỉ tiêu toàn tuyến này.

## 4. Tổng hợp Quartus cho DE10-Lite

Tổng hợp đầy đủ trong Quartus Prime Lite 25.1 cho MAX 10 `10M50DAF484C7G`: thành công, 0 lỗi, 52 cảnh báo.

| Chỉ số | Bản v4 trước | Bản thí nghiệm 8 MAC |
|---|---:|---:|
| Logic elements | 6.958 | 6.562 / 49.760, khoảng 13% |
| Registers | 3.644 | 3.398 |
| Bit bộ nhớ | 635.248 | 649.024 / 1.677.312, khoảng 39% |
| Bộ nhân nhúng 9 bit | 5 | 12 / 288, khoảng 4% |
| Fmax nội bộ, góc slow 85 °C | 59,66 MHz | 59,32 MHz |
| Setup slack tại 50 MHz | +3,238 ns | +3,141 ns |
| Hold slack tại góc slow 85 °C | Xem báo cáo cũ | +0,340 ns |

Bản UART có bộ đệm được tổng hợp riêng: **6.679 LE**, **3.475 thanh ghi**, **667.456 bit bộ nhớ**, **12 bộ nhân 9 bit**; Fmax nội bộ **60,8 MHz**, setup slack **+3,553 ns** và hold slack **+0,223 ns** tại góc slow 85 °C với xung 50 MHz. Tổng hợp thành công với 0 lỗi và 53 cảnh báo. Phân tích I/O vẫn chưa đầy đủ như lưu ý bên dưới.

Fmax và slack là kết quả phân tích timing sau tổng hợp/đặt tuyến, không phải phép đo đồng hồ trên bo. File constraints hiện chỉ ràng buộc clock 50 MHz; đường UART_RX và các đầu ra chưa được ràng buộc timing I/O. Cần hoàn thiện phân tích I/O, xem xét các cảnh báo tổng hợp và kiểm tra thực tế trên dây trước khi coi timing và độ ổn định của giao tiếp là đã xác nhận.

## 5. Trạng thái và công việc còn lại

- Đã tạo file `.sof` trong `output_files/` của thư mục thí nghiệm; chưa nạp bản này. SHA-256: `579a596fa10430278c526a6f53e6ca7eb85c4c1f5406fd4c19301ff552700fb9`.
- Bản UART có FIFO là một `.sof` thí nghiệm khác, SHA-256 `a62d90d1aea0c21aff12df86fb2eb736cf388338598f1fc38ef929928a0c4338`; chưa nạp và không dùng với firmware UART hiện tại do khác tốc độ và trình tự truyền.
- Lần kiểm tra ngày 16/11, các cổng COM tìm thấy đều là Bluetooth; `jtagconfig` không thấy USB-Blaster cục bộ. Vì vậy chưa thể kiểm chứng phiên bản mới trên kit hoặc khẳng định dây UART đang thông.
- Khi hai kit xuất hiện trở lại: xác nhận đúng thiết bị và chân, kiểm tra giao tiếp UART/CRC, nạp thử `.sof` qua JTAG, so sánh từng logit và thời gian thực với testbench trước khi thử camera. Kết quả của nhãn chuẩn chỉ được dùng để chấm sau suy luận.
- Chuẩn bị nhiều biển một hàng độc lập được camera ghi trong nhiều phiên, điều kiện ánh sáng và chất lượng khác nhau. Ghi riêng tỉ lệ phát hiện/tách ký tự, đúng từng ký tự, đúng toàn biển, số lần từ chối, độ trễ và thông lượng thực. Không huấn luyện lại bằng tập test camera.
- Thiết kế và xác minh một đường truyền nhanh hơn cùng cơ chế đệm/truyền song song, đồng thời tối ưu chụp, giải mã và tách ảnh trên ESP32. Chỉ kết luận đạt 100 biển/s và xấp xỉ 90% khi có phép đo lặp lại trên tập camera độc lập.

## 6. Tệp minh chứng

- `OcrBenchCoreK3_parallel8.sv`, `LiveDe10UartBoard.sv`, `LiveDe10UART.qsf`: RTL và cấu hình tổng hợp thí nghiệm.
- `sim_vectors.json`, `parallel8_sim_results.csv`: dữ liệu và kết quả 27 ký tự ban đầu.
- `locked_synthetic_test_100/manifest_test.json`, `locked_synthetic_test_100/accuracy_summary.json`, `locked_synthetic_test_100/xsim.log`: phân vùng test tổng hợp, số liệu và log mô phỏng 800 ký tự.
- `tb_uart_parallel8.sv`, `xsim.log`: test UART, CRC và thời gian mô phỏng của hai biển.
- `uart_buffered_experiment/tb_uart_buffered.sv`, `uart_buffered_experiment/xsim.log`: kiểm tra thông lượng 32 lượt biển, CRC và độ chiếm FIFO của giao tiếp UART có bộ đệm; các mẫu biển có lặp lại.
- `uart_buffered_experiment/output_files/LiveDe10UART.fit.summary`, `uart_buffered_experiment/output_files/LiveDe10UART.sta.rpt`: tài nguyên và timing bản UART có FIFO.
- `output_files/LiveDe10UART.fit.summary`, `output_files/LiveDe10UART.sta.rpt`, `output_files/LiveDe10UART.flow.rpt`: tài nguyên, timing và tổng hợp Quartus.
