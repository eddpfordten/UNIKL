"""Build a standalone HTML report from the completed Step 1 and Step 2 CSVs."""

import argparse
import csv
import html
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EVALUATION = ROOT / "outputs" / "evaluation"
STEP1_CSV = EVALUATION / "paired_patient_dice.csv"
STEP1_SUMMARY = EVALUATION / "paired_dice_summary.json"
STEP2_CSV = EVALUATION / "onnx_step2" / "onnx_patient_dice.csv"
STEP2_SUMMARY = EVALUATION / "onnx_step2" / "onnx_dice_summary.json"
OUTPUT = EVALUATION / "step1_step2_comparison.html"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def score(value):
    return f"{value:.6f}"


def signed(value):
    return f"{value:+.6f}"


def table_cell(value, numeric=False, delta=False):
    classes = []
    if numeric:
        classes.append("number")
    if delta:
        classes.append("positive" if value > 0 else "negative" if value < 0 else "neutral")
    rendered = signed(value) if delta else score(value) if numeric else str(value)
    return f'<td class="{" ".join(classes)}">{html.escape(rendered)}</td>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=EVALUATION)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    evaluation = args.evaluation_dir
    step1 = read_csv(evaluation / STEP1_CSV.name)
    step2 = read_csv(evaluation / "onnx_step2" / STEP2_CSV.name)
    step1_summary = json.loads((evaluation / STEP1_SUMMARY.name).read_text(encoding="utf-8"))
    step2_summary = json.loads((evaluation / "onnx_step2" / STEP2_SUMMARY.name).read_text(encoding="utf-8"))
    by_case_1 = {row["case_id"]: row for row in step1}
    by_case_2 = {row["case_id"]: row for row in step2}
    if len(by_case_1) != len(step1) or len(by_case_2) != len(step2):
        raise ValueError("Duplicate patient ID in source CSV")
    if set(by_case_1) != set(by_case_2) or len(step1) != 38:
        raise ValueError("Step 1 and Step 2 must contain the same 38 patients")

    own = statistics.fmean(float(row["own_dice"]) for row in step1)
    hayyi = statistics.fmean(float(row["hayyi_dice"]) for row in step1)
    fp32 = statistics.fmean(float(row["onnx_fp32_dice"]) for row in step2)
    int8 = statistics.fmean(float(row["onnx_int8_dice"]) for row in step2)
    if any((
        abs(own - step1_summary["own_mean_dice"]) > 1e-7,
        abs(hayyi - step1_summary["hayyi_mean_dice"]) > 1e-7,
        abs(fp32 - step2_summary["onnx_fp32_mean_dice"]) > 1e-7,
        abs(int8 - step2_summary["onnx_int8_mean_dice"]) > 1e-7,
    )):
        raise ValueError("Source CSV means differ from their summaries")

    size_reduction = 100 * (1 - step2_summary["onnx_int8_bytes"] / step2_summary["onnx_fp32_bytes"])
    int8_changes = [float(row["onnx_int8_dice"]) - float(row["pytorch_dice"]) for row in step2]
    int8_higher = sum(value > 0 for value in int8_changes)
    int8_lower = sum(value < 0 for value in int8_changes)
    rows = []
    for case_id in sorted(by_case_1):
        first = by_case_1[case_id]
        second = by_case_2[case_id]
        own_case = float(first["own_dice"])
        baseline_case = float(second["pytorch_dice"])
        if abs(own_case - baseline_case) > 1e-7:
            raise ValueError(f"Original model score differs between files: {case_id}")
        hayyi_case = float(first["hayyi_dice"])
        fp32_case = float(second["onnx_fp32_dice"])
        int8_case = float(second["onnx_int8_dice"])
        values = [case_id, own_case, hayyi_case, own_case - hayyi_case,
                  fp32_case, int8_case, int8_case - own_case]
        cells = [table_cell(values[0])]
        cells.extend(table_cell(value, numeric=True, delta=index in (3, 6))
                     for index, value in enumerate(values[1:], start=1))
        rows.append("<tr>" + "".join(cells) + "</tr>")

    document = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Brain Tumor Segmentation — Model Comparison</title>
<style>
  :root { color-scheme: light; --ink:#18212b; --muted:#536271; --line:#dce3e8; --paper:#fff; --wash:#f4f7f9; --accent:#174c6b; --good:#14643f; --bad:#9d422b; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--wash); color:var(--ink); font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
  main { max-width:1180px; margin:0 auto; padding:42px 28px 72px; background:var(--paper); }
  h1 { margin:0 0 8px; font-size:30px; line-height:1.2; letter-spacing:-.02em; }
  h2 { margin:44px 0 12px; font-size:20px; border-top:1px solid var(--line); padding-top:26px; }
  h3 { margin:22px 0 7px; font-size:16px; }
  p { margin:8px 0; }
  .eyebrow { color:var(--accent); font-weight:700; font-size:12px; letter-spacing:.09em; text-transform:uppercase; }
  .lede { max-width:860px; color:var(--muted); font-size:17px; }
  .facts { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:0; margin:28px 0; border-block:1px solid var(--line); }
  .fact { padding:17px 20px 17px 0; }
  .fact + .fact { border-left:1px solid var(--line); padding-left:20px; }
  .fact span { display:block; color:var(--muted); font-size:13px; }
  .fact strong { display:block; font-size:23px; font-variant-numeric:tabular-nums; }
  .table-wrap { overflow-x:auto; border:1px solid var(--line); border-radius:5px; }
  table { border-collapse:collapse; width:100%; min-width:860px; }
  th,td { padding:9px 11px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }
  th { background:#eaf0f4; color:#263746; font-size:12px; position:sticky; top:0; }
  tbody tr:nth-child(even) { background:#fafcfd; }
  tbody tr:last-child td { border-bottom:0; }
  .number { text-align:right; font-variant-numeric:tabular-nums; }
  .positive { color:var(--good); }
  .negative { color:var(--bad); }
  .neutral { color:var(--muted); }
  .callout { border-left:4px solid var(--accent); background:#edf3f6; padding:12px 17px; margin:18px 0; }
  .note { color:var(--muted); font-size:13px; }
  .controls { display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin:15px 0; }
  input { width:280px; max-width:100%; padding:9px 11px; border:1px solid #a9b8c3; border-radius:4px; font:inherit; }
  .legend { display:flex; gap:18px; flex-wrap:wrap; color:var(--muted); font-size:12px; }
  ul { padding-left:21px; }
  li { margin:8px 0; }
  footer { margin-top:32px; padding-top:16px; border-top:1px solid var(--line); color:var(--muted); font-size:12px; }
  @media (max-width:700px) { main { padding:24px 16px; } .facts { grid-template-columns:1fr; } .fact + .fact { border-left:0; border-top:1px solid var(--line); padding-left:0; } }
  @media print { body { background:white; } main { max-width:none; padding:0; } input,.controls { display:none; } th { position:static; } tr { break-inside:avoid; } }
</style>
</head>
<body><main>
<div class="eyebrow">BraTS-PEDs · Lecturer comparison</div>
<h1>Brain tumor segmentation: model and ONNX comparison</h1>
<p class="lede">Whole-tumor Dice scores for the same 38 labeled patients held out from the Training folder. Higher Dice means more overlap with the reference tumor mask.</p>
<div class="facts">
  <div class="fact"><span>Your original model · mean Dice</span><strong>__OWN__</strong></div>
  <div class="fact"><span>Hayyi model · mean Dice</span><strong>__HAYYI__</strong></div>
  <div class="fact"><span>Your quantized ONNX model · mean Dice</span><strong>__INT8__</strong></div>
</div>

<h2>Conclusions</h2>
<ul>
  <li>Your original model averaged <strong>__OWN__</strong> Dice and Hayyi's averaged <strong>__HAYYI__</strong>. The mean difference is <strong>__STEP1_DELTA__</strong> in your model's favor; yours scored higher for 24 patients and Hayyi's for 14.</li>
  <li>After ONNX INT8 quantization, your model averaged <strong>__INT8__</strong> Dice, a mean change of <strong>__INT8_DELTA__</strong> from your original model. Quantization raised the score for __INT8_HIGHER__ patients and lowered it for __INT8_LOWER__; the small positive average does not mean every patient improved.</li>
  <li>The regular ONNX export averaged <strong>__FP32__</strong> Dice and reproduced the original PyTorch results (maximum per-patient Dice difference below 0.00000001). The ONNX file decreased from __FP32_MB__ MB to __INT8_MB__ MB, about <strong>__SIZE_REDUCTION__%</strong> smaller.</li>
</ul>
<div class="callout"><strong>Important interpretation:</strong> Hayyi's supplied radiomics values for these patients were calculated using their ground-truth tumor masks. Those labels enter Hayyi's model as input, so the Step 1 comparison is not a blind or fully fair test of generalization. Report both measured means, but do not claim one model is conclusively better based on this comparison.</div>

<h2>Average Dice comparison</h2>
<div class="table-wrap"><table>
<thead><tr><th>Evaluation</th><th>Model</th><th class="number">Patients</th><th class="number">Mean Dice</th><th>Interpretation</th></tr></thead>
<tbody>
<tr><td>Step 1</td><td>Your PyTorch checkpoint</td><td class="number">38</td><td class="number">__OWN__</td><td>Reference for both comparisons</td></tr>
<tr><td>Step 1</td><td>Hayyi + radiomics</td><td class="number">38</td><td class="number">__HAYYI__</td><td>Radiomics derived from reference masks</td></tr>
<tr><td>Step 2</td><td>Your regular ONNX export</td><td class="number">38</td><td class="number">__FP32__</td><td>Checks export agreement</td></tr>
<tr><td>Step 2</td><td>Your quantized ONNX model</td><td class="number">38</td><td class="number">__INT8__</td><td>Static INT8 quantization of convolution layers</td></tr>
</tbody></table></div>

<h2>Patient-by-patient results</h2>
<p class="note">All columns use Dice scores from 0 to 1. Difference columns are signed: positive means your model scored higher than Hayyi's, or INT8 scored higher than your original model.</p>
<div class="controls"><label for="patient-search">Find patient</label><input id="patient-search" type="search" placeholder="e.g. BraTS-PED-00008-000"><span class="legend">Green = positive difference · Brown = negative difference</span></div>
<div class="table-wrap"><table id="patient-table">
<thead><tr><th>Patient ID</th><th class="number">Your original</th><th class="number">Hayyi</th><th class="number">Yours − Hayyi</th><th class="number">ONNX FP32</th><th class="number">ONNX INT8</th><th class="number">INT8 − original</th></tr></thead>
<tbody>__PATIENT_ROWS__</tbody>
</table></div>

<h2>Methods and notes for presentation</h2>
<ul>
  <li><strong>Patient set:</strong> 38 labeled cases held out from the dataset's Training folder using seed 42 and a 15% split. The official Validation folder has MRI scans but no segmentation masks, so Dice cannot be calculated there from the available files.</li>
  <li><strong>Dice definition:</strong> Binary whole tumor (segmentation label &gt; 0), four MRI modalities, resized to 96 × 96 × 96 voxels, with a 0.5 prediction threshold.</li>
  <li><strong>Quantization:</strong> The saved checkpoint was exported to ONNX, then calibrated with 16 patients from the training subset only. Convolution layers use signed INT8 QDQ quantization; other operations may remain in floating point. Inference used ONNX Runtime's CPU provider. No retraining was performed.</li>
  <li><strong>Scope:</strong> The smaller ONNX file and Dice result are measured here. Inference speed and performance on a separate labeled test set were not measured.</li>
</ul>
<footer>Sources: paired_patient_dice.csv, paired_dice_summary.json, onnx_patient_dice.csv, onnx_dice_summary.json. This file is self-contained and can be opened in a browser or printed to PDF.</footer>
</main>
<script>
const search = document.getElementById('patient-search');
search.addEventListener('input', () => {
  const needle = search.value.trim().toLowerCase();
  for (const row of document.querySelectorAll('#patient-table tbody tr')) {
    row.hidden = !row.cells[0].textContent.toLowerCase().includes(needle);
  }
});
</script>
</body></html>
"""
    replacements = {
        "__OWN__": score(own),
        "__HAYYI__": score(hayyi),
        "__FP32__": score(fp32),
        "__INT8__": score(int8),
        "__STEP1_DELTA__": signed(own - hayyi),
        "__INT8_DELTA__": signed(int8 - own),
        "__INT8_HIGHER__": str(int8_higher),
        "__INT8_LOWER__": str(int8_lower),
        "__FP32_MB__": f'{step2_summary["onnx_fp32_bytes"] / 1_000_000:.2f}',
        "__INT8_MB__": f'{step2_summary["onnx_int8_bytes"] / 1_000_000:.2f}',
        "__SIZE_REDUCTION__": f"{size_reduction:.1f}",
        "__PATIENT_ROWS__": "\n".join(rows),
    }
    for marker, replacement in replacements.items():
        document = document.replace(marker, replacement)
    if "__" in document:
        raise ValueError("Unreplaced template marker")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as destination:
        destination.write(document)
    print(f"Created {args.output}")
    print(f"Patients: {len(rows)}; INT8 higher: {int8_higher}; lower: {int8_lower}")


if __name__ == "__main__":
    main()
