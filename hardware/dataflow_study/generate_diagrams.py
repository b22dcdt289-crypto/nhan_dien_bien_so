"""Generate five labeled SVG architecture figures for the LeNet-5 study."""

from __future__ import annotations

from html import escape
from pathlib import Path


HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"


def node(x: int, y: int, width: int, title: str, subtitle: str, color: str = "#e8efff") -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="82" rx="12" fill="{color}" stroke="#324867" stroke-width="1.5"/>'
        f'<text x="{x + width // 2}" y="{y + 31}" text-anchor="middle" class="title">{escape(title)}</text>'
        f'<text x="{x + width // 2}" y="{y + 56}" text-anchor="middle" class="sub">{escape(subtitle)}</text>'
    )


def arrow(x1: int, y1: int, x2: int, y2: int, caption: str = "") -> str:
    result = f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="#314968" stroke-width="2.2" marker-end="url(#arrow)"/>'
    if caption:
        result += f'<text x="{(x1 + x2) // 2}" y="{min(y1, y2) - 10}" text-anchor="middle" class="label">{escape(caption)}</text>'
    return result


def figure(filename: str, title: str, description: str, body: str, footer: str) -> None:
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="300" viewBox="0 0 1000 300" role="img" aria-labelledby="title desc">
<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>
<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0 0 L9 4.5 L0 9 Z" fill="#314968"/></marker></defs>
<style>.heading{{font:600 23px Arial,sans-serif;fill:#172b4d}}.title{{font:600 17px Arial,sans-serif;fill:#172b4d}}.sub{{font:14px Arial,sans-serif;fill:#334155}}.label{{font:14px Arial,sans-serif;fill:#334155}}.foot{{font:14px Arial,sans-serif;fill:#334155}}</style>
<rect width="1000" height="300" rx="18" fill="#f7f9fc"/>
<text x="30" y="42" class="heading">{escape(title)}</text>
{body}
<text x="30" y="285" class="foot">{escape(footer)}</text>
</svg>'''
    (FIGURES / filename).write_text(svg, encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    body = (
        node(28, 90, 175, "BRAM trọng số", "đọc W theo tile", "#e4f1ff")
        + node(28, 180, 175, "Kích hoạt vào", "broadcast X", "#e7f8ec")
        + node(285, 125, 172, "PE₀: W cố định", "MAC nhiều vị trí", "#fff0db")
        + node(485, 125, 172, "PE₁: W cố định", "MAC nhiều vị trí", "#fff0db")
        + node(735, 125, 220, "Tích lũy / output", "psum di chuyển", "#f0e9ff")
        + arrow(203, 131, 285, 153, "W") + arrow(203, 220, 285, 184, "X")
        + arrow(457, 166, 485, 166) + arrow(657, 166, 735, 166, "psum")
    )
    figure("weight_stationary.svg", "Weight-stationary (WS)",
           "Mỗi PE giữ trọng số trong thanh ghi và dùng nhiều lần; kích hoạt và tổng từng phần di chuyển.", body,
           "Tối ưu đọc lại W; phải bố trí đường truyền X và psum giữa PE / bộ đệm.")

    body = (
        node(28, 90, 190, "BRAM trọng số", "phát W theo tap", "#e4f1ff")
        + node(28, 180, 190, "Kích hoạt vào", "phát X theo tap", "#e7f8ec")
        + node(330, 125, 240, "PE: psum cố định", "tích lũy 32-bit cục bộ", "#fff0db")
        + node(735, 125, 220, "Feature map ra", "ghi một lần / output", "#f0e9ff")
        + arrow(218, 132, 330, 151, "W") + arrow(218, 221, 330, 181, "X")
        + arrow(570, 166, 735, 166, "Y hoàn tất")
    )
    figure("output_stationary.svg", "Output-stationary (OS)",
           "Một PE giữ tổng từng phần của một phần tử đầu ra, nhận các tap trọng số và kích hoạt đến khi hoàn tất.", body,
           "Giảm ghi/đọc psum; lõi MAC nối tiếp hiện có gần cách giữ psum này.")

    body = (
        node(25, 125, 170, "Luồng pixel", "1 phần tử / nhịp", "#e7f8ec")
        + node(235, 125, 180, "Line buffer", "4 hàng + cửa sổ 5×5", "#e4f1ff")
        + node(455, 125, 190, "PE theo hàng", "giữ 5 W, dùng lại X", "#fff0db")
        + node(690, 125, 275, "Cộng psum lân cận", "ghép 5 hàng và các kênh", "#f0e9ff")
        + arrow(195, 166, 235, 166) + arrow(415, 166, 455, 166)
        + arrow(645, 166, 690, 166)
    )
    figure("row_stationary.svg", "Row-stationary (RS)",
           "Dữ liệu hàng của cửa sổ tích chập, trọng số hàng và tổng từng phần được tái sử dụng cục bộ qua PE lân cận.", body,
           "Giảm chuyển dữ liệu ở nhiều cấp; cần line buffer, mạng PE và lịch ánh xạ tile.")

    body = (
        node(20, 125, 172, "Nguồn 32×32", "JTAG hoặc pixel stream", "#e7f8ec")
        + node(225, 125, 170, "Buffer vào", "1 KiB / ký tự", "#e4f1ff")
        + node(430, 125, 170, "Conv1", "4×32 byte line buffer", "#fff0db")
        + node(635, 125, 170, "Conv2", "4×14×6 byte", "#fff0db")
        + node(840, 125, 140, "FC / OCR", "30 lớp", "#f0e9ff")
        + arrow(192, 166, 225, 166) + arrow(395, 166, 430, 166)
        + arrow(600, 166, 635, 166) + arrow(805, 166, 840, 166)
    )
    figure("streaming_line_buffers.svg", "Streaming và line buffer",
           "Một ảnh ký tự 32 nhân 32 đi qua bộ đệm và hai lớp tích chập; đường JTAG hiện tại nạp trọn frame trước khi tính.", body,
           "JTAG frame-loading đã có; pixel-stream Conv1/Conv2 là kiến trúc đề xuất, chưa post-fit.")

    body = (
        node(25, 125, 180, "BRAM W INT8", "63.150 / 37.950 B", "#e4f1ff")
        + node(250, 125, 190, "Cửa sổ 5×5", "X dùng cho nhiều filter", "#e7f8ec")
        + node(485, 125, 210, "8 PE MAC", "multicast W / X", "#fff0db")
        + node(740, 125, 230, "RF psum + BRAM", "chỉ ghi khi hoàn tất", "#f0e9ff")
        + arrow(205, 166, 485, 145, "W theo tile")
        + arrow(440, 176, 485, 177, "X")
        + arrow(695, 166, 740, 166)
    )
    figure("reuse_network.svg", "Mạng tái sử dụng dữ liệu LeNet-5",
           "Trọng số và kích hoạt được phân phối từ BRAM đến cụm PE; tổng từng phần giữ trong thanh ghi rồi ghi lại.", body,
           "Chọn tile theo dung lượng M9K và số cổng đọc; không đồng nhất số MAC với số lần truy cập bộ nhớ.")


if __name__ == "__main__":
    main()
