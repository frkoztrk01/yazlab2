"""Generate PDF report: title, intro, table descriptions + 5 EK-format tables."""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpdf import FPDF

from scripts.generate_tables import (
    load_all_results,
    table1_performance,
    table2_robustness,
    table3_cross_dataset,
    table4_param_sensitivity,
    table5_runtime,
)
from src.config_loader import load_config, resolve_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = Path("/System/Library/Fonts/Supplemental")
FONT = "TimesNewRoman"
OUTPUT = PROJECT_ROOT / "48_yazlab2_rapor.pdf"
DOWNLOADS_COPY = Path("/Users/faruk/Downloads/48_yazlab2_rapor.pdf")

GROUP_NO = 48
GROUP_MEMBERS = "Faruk Öztürk & Nilay Süzer"
GITHUB_URL = "https://github.com/frkoztrk01/yazlab2"

TABLE_DESCRIPTIONS = {
    "Tablo 1": (
        "Her modelin SKAB ve BATADAL veri setlerindeki ortalama F1 skorunu ve "
        "5 tekrarlı deneydeki standart sapmasını gösterir; model stabilitesini karşılaştırır."
    ),
    "Tablo 2": (
        "SKAB üzerinde orijinal ve gürültülü senaryolardaki F1 değerlerini raporlar. "
        "Otomata için unseen pattern senaryosunda detection rate ve mapping accuracy sunulur."
    ),
    "Tablo 3": (
        "Bir veri setinde eğitilen LSTM modelinin diğer veri setinde test edildiğinde "
        "elde ettiği F1 skorlarını gösterir; veri setleri arası genellenebilirliği ölçer."
    ),
    "Tablo 4": (
        "Otomata modelinde window size ve alphabet size parametrelerinin SKAB performansına "
        "etkisini analiz eder; sabit karşılaştırma değerleri window=4, alphabet=3'tür."
    ),
    "Tablo 5": (
        "Modellerin ortalama eğitim ve çıkarım sürelerini (saniye) karşılaştırır; "
        "hesaplama maliyeti açısından DL ile otomata arasındaki farkı gösterir."
    ),
}


class ReportPDF(FPDF):
    def __init__(self) -> None:
        super().__init__()
        self.add_font(FONT, "", str(FONT_DIR / "Times New Roman.ttf"))
        self.add_font(FONT, "B", str(FONT_DIR / "Times New Roman Bold.ttf"))
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(15, 15, 15)

    def section_title(self, title: str) -> None:
        if self.get_y() > 250:
            self.add_page()
        self.ln(4)
        self.set_font("TimesNewRoman", "B", 12)
        self.set_text_color(20, 40, 80)
        self.multi_cell(0, 7, title)
        self.ln(2)
        self.set_text_color(0, 0, 0)
        self.set_font("TimesNewRoman", "", 10)

    def body(self, text: str) -> None:
        self.multi_cell(0, 5.5, text)
        self.ln(2)


def _parse_markdown_table(md_block: str) -> tuple[str, list[str], list[list[str]]]:
    lines = [ln.rstrip() for ln in md_block.strip().splitlines() if ln.strip()]
    title = lines[0].removeprefix("### ").strip()
    table_lines = [ln for ln in lines if ln.startswith("|")]
    if len(table_lines) < 2:
        return title, [], []

    headers = [c.strip() for c in table_lines[0].split("|")[1:-1]]
    rows = [[c.strip() for c in ln.split("|")[1:-1]] for ln in table_lines[2:]]
    return title, headers, rows


def _table_description(title: str) -> str | None:
    for key, desc in TABLE_DESCRIPTIONS.items():
        if title.startswith(key):
            return desc
    return None


def _column_widths(headers: list[str], rows: list[list[str]], page_width: float) -> list[float]:
    max_lens = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(max_lens):
                max_lens[i] = max(max_lens[i], len(cell))

    weights = [max(l, 8) for l in max_lens]
    total = sum(weights) or 1
    return [page_width * w / total for w in weights]


def _add_table(pdf: ReportPDF, headers: list[str], rows: list[list[str]]) -> None:
    if not headers:
        return

    page_width = pdf.w - pdf.l_margin - pdf.r_margin
    col_widths = _column_widths(headers, rows, page_width)
    row_h = 7

    if pdf.get_y() + row_h * (len(rows) + 2) > pdf.h - pdf.b_margin:
        pdf.add_page()

    pdf.set_font("TimesNewRoman", "B", 9)
    pdf.set_fill_color(240, 240, 240)
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], row_h, header, border=1, fill=True)
    pdf.ln()

    pdf.set_font("TimesNewRoman", "", 9)
    for row in rows:
        if pdf.get_y() + row_h > pdf.h - pdf.b_margin:
            pdf.add_page()
            pdf.set_font("TimesNewRoman", "B", 9)
            pdf.set_fill_color(240, 240, 240)
            for i, header in enumerate(headers):
                pdf.cell(col_widths[i], row_h, header, border=1, fill=True)
            pdf.ln()
            pdf.set_font("TimesNewRoman", "", 9)

        for i, cell in enumerate(row):
            width = col_widths[i] if i < len(col_widths) else col_widths[-1]
            pdf.cell(width, row_h, cell, border=1)
        pdf.ln()


def _add_table_section(pdf: ReportPDF, md_block: str) -> None:
    title, headers, rows = _parse_markdown_table(md_block)
    if pdf.get_y() > 20:
        pdf.ln(4)
    if pdf.get_y() + 30 > pdf.h - pdf.b_margin:
        pdf.add_page()

    pdf.set_font("TimesNewRoman", "B", 11)
    pdf.multi_cell(0, 6, title)

    desc = _table_description(title)
    if desc:
        pdf.ln(1)
        pdf.set_font("TimesNewRoman", "", 9)
        pdf.set_text_color(60, 60, 60)
        pdf.multi_cell(0, 5, desc)
        pdf.set_text_color(0, 0, 0)

    pdf.ln(2)
    _add_table(pdf, headers, rows)
    pdf.ln(4)


def _add_cover(pdf: ReportPDF) -> None:
    pdf.ln(18)
    pdf.set_font("TimesNewRoman", "B", 11)
    pdf.cell(0, 8, "Yazılım Geliştirme Dersi — 2. Proje", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_font("TimesNewRoman", "B", 15)
    pdf.multi_cell(
        0,
        9,
        "From Black-Box to Explainability:\nProbabilistic Automata for Time Series Analysis",
        align="C",
    )
    pdf.ln(10)
    pdf.set_font("TimesNewRoman", "", 12)
    pdf.cell(0, 8, f"Grup {GROUP_NO}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, GROUP_MEMBERS, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_font("TimesNewRoman", "", 10)
    pdf.cell(0, 7, "Teslim: 7 Haziran 2026, 23:59", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"GitHub: {GITHUB_URL}", align="C", new_x="LMARGIN", new_y="NEXT")


def _add_introduction(pdf: ReportPDF) -> None:
    pdf.add_page()
    pdf.section_title("I. Giriş")
    pdf.body(
        "Zaman serisi verileri finans, biyomedikal, IoT ve siber güvenlik gibi alanlarda "
        "yaygın kullanılmaktadır. Bu projede anomali tespiti, iki modelleme paradigması "
        "üzerinden karşılaştırılmıştır: (1) black-box derin öğrenme modelleri (LSTM, GRU, 1D-CNN) "
        "ve (2) sembolik temsil ve durum geçiş olasılıklarına dayalı olasılıksal otomata modeli "
        "(PAA → SAX → sliding window)."
    )
    pdf.body(
        "Deneyler SKAB (valve1/valve2) ve BATADAL (Training Dataset 2, ATT_FLAG hedef değişkeni) "
        "veri setleri üzerinde yürütülmüştür. Üç senaryo test edilmiştir: orijinal veri, "
        "Gaussian gürültü eklenmiş veri ve unseen pattern senaryosu. Her koşul 5 farklı random "
        "seed ile tekrarlanmış; sonuçlar ortalama ± standart sapma olarak raporlanmıştır."
    )
    pdf.body(
        "Araştırma sorusu: Farklı modelleme yaklaşımları, farklı veri koşulları altında "
        "nasıl davranmaktadır? Proje yalnızca en iyi modeli seçmekten ziyade; performans, "
        "genellenebilirlik, gürültüye dayanıklılık, açıklanabilirlik ve hesaplama maliyeti "
        "kriterlerini sistematik biçimde karşılaştırmayı hedefler."
    )

    pdf.section_title("II. Deney Sonuçları Tabloları")
    pdf.body(
        "Aşağıdaki beş tablo, EK doküman formatında deney sonuçlarını özetler. "
        "Tablolar sırasıyla model performansı, gürültü/unseen dayanıklılığı, "
        "cross-dataset transfer, otomata parametre duyarlılığı ve çalışma süresi "
        "karşılaştırmasını içermektedir."
    )


def build_report() -> Path:
    config = load_config()
    results_dir = resolve_path(config, "results")
    records = load_all_results(results_dir)

    table_blocks = [
        table1_performance(records),
        table2_robustness(records),
        table3_cross_dataset(results_dir),
        table4_param_sensitivity(records),
        table5_runtime(records),
    ]

    pdf = ReportPDF()
    pdf.add_page()
    _add_cover(pdf)
    _add_introduction(pdf)

    pdf.set_font("TimesNewRoman", "", 10)
    for block in table_blocks:
        block = re.sub(r"(?m)^>.*\n?", "", block).strip()
        _add_table_section(pdf, block)

    pdf.output(str(OUTPUT))

    if DOWNLOADS_COPY.parent.exists():
        shutil.copy2(OUTPUT, DOWNLOADS_COPY)

    return OUTPUT


if __name__ == "__main__":
    out = build_report()
    print(f"Rapor oluşturuldu: {out}")
