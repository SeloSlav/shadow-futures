"""Static scientific figure using bundled ReportLab charts and PDFium.

Coordinates are logarithmically transformed where the axis labels say log.
The scratch PDF supplies rasterization; the deliverables are PNG and SVG.
"""
from pathlib import Path
import json
import math
from reportlab.graphics.shapes import Drawing, String, Line
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.widgets.markers import makeMarker
from reportlab.graphics import renderPDF, renderSVG
from reportlab.lib.colors import HexColor
import pypdfium2 as pdfium

OUT = Path(__file__).resolve().parent
d = json.loads((OUT / "paper2_results.json").read_text(encoding="utf-8"))
figure = Drawing(920, 690)
BLACK = HexColor("#18232b")
SINGLE = HexColor("#8b3f2f")
REPL = HexColor("#215e8e")
CONTROL = HexColor("#677448")
ts = [16, 256, 2048, 8192]
xs = [math.log10(t) for t in ts]
figure.add(String(45, 660, "Synthetic reinforced allocation: time versus independent replication", fontName="Helvetica-Bold", fontSize=17, fillColor=BLACK))
figure.add(String(45, 638, "Exact-kernel classifier; beta = -0.5 or +0.5; 5,000 independent datasets per law", fontSize=11, fillColor=BLACK))
for x, color, label in [(45, SINGLE, "One inherited history"), (255, REPL, "Independent length-16 markets"), (555, CONTROL, "Uniform no-response control")]:
    figure.add(Line(x, 615, x + 22, 615, strokeColor=color, strokeWidth=2))
    figure.add(String(x + 30, 611, label, fontSize=10, fillColor=BLACK))

def plot(x, y, title, metric, ylimits, yticks, ylabels, log_y=True, show_control=False):
    chart = LinePlot()
    chart.x, chart.y, chart.width, chart.height = x, y, 350, 170
    rows = [d["single_history"], d["independent_length16_markets"]]
    tr = math.log10 if log_y else (lambda z: z)
    series = []
    for rs in rows:
        series.append([(math.log10(r["observations"]), tr(100 * r["bayes_error_mc"] if metric == "risk" else r[metric]["mean"])) for r in rs])
    if show_control:
        series.append([(math.log10(t), tr(50.0 if metric == "risk" else t / 2)) for t in ts])
    chart.data = series
    chart.xValueAxis.valueMin, chart.xValueAxis.valueMax = xs[0], xs[-1]
    chart.xValueAxis.valueSteps = xs
    chart.xValueAxis.labelTextFormat = lambda z: f"{min(ts, key=lambda t: abs(math.log10(t)-z)):,}"
    chart.yValueAxis.valueMin, chart.yValueAxis.valueMax = ylimits
    chart.yValueAxis.valueSteps = yticks
    chart.yValueAxis.labelTextFormat = lambda z: ylabels[min(range(len(yticks)), key=lambda i: abs(yticks[i] - z))]
    for axis in (chart.xValueAxis, chart.yValueAxis):
        axis.labels.fontName = "Helvetica"
        axis.labels.fontSize = 9
        axis.strokeColor = HexColor("#59666f")
        axis.strokeWidth = .5
    chart.yValueAxis.visibleGrid = True
    chart.yValueAxis.gridStrokeColor = HexColor("#e7eaed")
    chart.yValueAxis.gridStrokeWidth = .5
    for i, color in enumerate((SINGLE, REPL, CONTROL)):
        chart.lines[i].strokeColor = color
        chart.lines[i].strokeWidth = 1.7
        if i < 2:
            chart.lines[i].symbol = makeMarker("FilledCircle" if i == 0 else "FilledSquare")
            chart.lines[i].symbol.size = 4
            chart.lines[i].symbol.fillColor = color
        else:
            chart.lines[i].strokeDashArray = [4, 3]
    figure.add(chart)
    figure.add(String(x, y + 190, title, fontName="Helvetica-Bold", fontSize=12, fillColor=BLACK))
    figure.add(String(x + 175, y - 35, "Total allocations (log scale)", fontSize=9, textAnchor="middle", fillColor=BLACK))
    for rs, color in zip(rows, (SINGLE, REPL)):
        for r in rs:
            if metric == "risk":
                lo, hi = [100 * q for q in r["bayes_error_mc_95_interval"]]
            else:
                mean, se = r[metric]["mean"], r[metric]["mc_se"]
                lo, hi = tr(max(mean - se, 10 ** ylimits[0])), tr(mean + se)
            px = x + (math.log10(r["observations"]) - xs[0]) / (xs[-1] - xs[0]) * 350
            pylo = y + (lo - ylimits[0]) / (ylimits[1] - ylimits[0]) * 170
            pyhi = y + (hi - ylimits[0]) / (ylimits[1] - ylimits[0]) * 170
            figure.add(Line(px, pylo, px, pyhi, strokeColor=color, strokeWidth=.8))
            figure.add(Line(px - 2.5, pylo, px + 2.5, pylo, strokeColor=color, strokeWidth=.8))
            figure.add(Line(px - 2.5, pyhi, px + 2.5, pyhi, strokeColor=color, strokeWidth=.8))

plot(70, 390, "Classification error (%)", "risk", (-1, 55), [0, 10, 20, 30, 40, 50], ["0", "10", "20", "30", "40", "50"], log_y=False, show_control=True)
plot(540, 390, "Mean comparison budget (log scale)", "comparison_budget", (-.3, 4), [0, 1, 2, 3, 4], ["1", "10", "100", "1,000", "10,000"], show_control=True)
plot(70, 135, "Mean Fisher information (log scale)", "conditional_fisher_information", (-.3, 3), [0, 1, 2, 3], ["1", "10", "100", "1,000"])
plot(540, 135, "One-step forecast gap (log scale)", "one_step_forecast_disagreement", (-5, -1), [-5, -4, -3, -2, -1], ["0.00001", "0.0001", "0.001", "0.01", "0.1"])
figure.add(String(70, 290, "Uniform control: information = 0", fontSize=9, fillColor=CONTROL))
figure.add(String(720, 155, "Uniform control: gap = 0", fontSize=9, fillColor=CONTROL))
for y, txt in [(67, "Classifier bars: 95% Monte Carlo intervals. Mean bars: +/-1 Monte Carlo standard error."),
               (49, "Forecasts extend the current market before reset (market ages T versus 16). No field data."),
               (31, "No observed classification errors at large replication horizons imply an upper bound, not zero true risk.")]:
    figure.add(String(45, y, txt, fontSize=10, fillColor=BLACK))
scratch_pdf = OUT / "paper2_benchmark_figure.pdf"
renderPDF.drawToFile(figure, str(scratch_pdf))
renderSVG.drawToFile(figure, str(OUT / "paper2_benchmark_figure.svg"))
pdf = pdfium.PdfDocument(str(scratch_pdf))
page = pdf[0]
bitmap = page.render(scale=2)
bitmap.to_pil().save(OUT / "paper2_benchmark_figure.png")
bitmap.close()
page.close()
pdf.close()
print("Saved paper2_benchmark_figure.png and .svg; scratch .pdf supplies rasterization")
