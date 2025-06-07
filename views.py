from django.shortcuts import render
from .forms import PredictionForm
import subprocess
from datetime import datetime
import pytz
import os
import platform
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
from django.http import HttpResponse

def predict_view(request):
    result = None
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            dataset = form.cleaned_data['dataset']
            epochs = form.cleaned_data['epochs']
            eval_type = form.cleaned_data.get('eval_type', 'filtered')
            use_triton = form.cleaned_data['use_triton']

            command = [
                "python", "/nfs/stak/users/gangsw/HC-MPNN/main.py",
                "--dataset", dataset,
                "--n_epoch", str(epochs),
                "--num_layer", str(form.cleaned_data['num_layer'])
            ]
            if use_triton:
                command.append("--use_triton")
            
            try:
                start_time = datetime.now()
                output = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd="/nfs/stak/users/gangsw/HC-MPNN"
                )
                end_time = datetime.now()
                elapsed = end_time - start_time

                raw_lines = output.stdout.splitlines()
                local_time = end_time.astimezone(pytz.timezone("America/Los_Angeles"))
                timestamp = local_time.strftime('%Y-%m-%d %H:%M:%S')
                sys_info = f"System: {platform.node()} ({platform.system()} {platform.release()})\n"
                sys_info += f"Python: {platform.python_version()}\n"
                sys_info += f"Use Triton: {use_triton}\n"

                def extract_metrics(start_label, lines):
                    metrics = {}
                    for i, line in enumerate(lines):
                        if start_label in line:
                            for j in range(i+1, i+6):
                                if j < len(lines):
                                    parts = lines[j].strip().split("=")
                                    if len(parts) == 2:
                                        key = parts[0].strip()
                                        val = parts[1].strip()
                                        metrics[key] = val
                            break
                    return metrics

                raw_metrics = extract_metrics("Raw setting:", raw_lines)
                fil_metrics = extract_metrics("Fil setting:", raw_lines)

                result = f"""
🧠 Dataset: {dataset}
🕐 Epochs: {epochs}
📅 Run Time: {timestamp}
⏱ Duration: {elapsed.seconds}s
🖥 System Info:
{sys_info}
"""

                notes = form.cleaned_data.get('notes', '')

                full_result = result
                if eval_type in ["raw", "both"]:
                    full_result += f"""
📊 Raw Evaluation (Test Set):
- Hit@1  = {raw_metrics.get('Hit@1')}
- Hit@3  = {raw_metrics.get('Hit@3')}
- Hit@10 = {raw_metrics.get('Hit@10')}
- MRR    = {raw_metrics.get('MRR')}
- MR     = {raw_metrics.get('MR')}
"""
                if eval_type in ["filtered", "both"]:
                    full_result += f"""
📊 Filtered Evaluation (Test Set):
- Hit@1  = {fil_metrics.get('Hit@1')}
- Hit@3  = {fil_metrics.get('Hit@3')}
- Hit@10 = {fil_metrics.get('Hit@10')}
- MRR    = {fil_metrics.get('MRR')}
- MR     = {fil_metrics.get('MR')}
"""
                if notes:
                    full_result += f"\n📝 Notes:\n{notes}\n"

                request.session['latest_result'] = full_result

                result = full_result

            except subprocess.CalledProcessError as e:
                result = f"Error: {e.stderr}"
    else:
        form = PredictionForm()
    return render(request, 'predict.html', {'form': form, 'result': result})

def download_pdf(request):
    from datetime import datetime
    result = request.session.get('latest_result')
    if not result:
        return HttpResponse("No results to export.", status=400)

    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y = height - 50
    for line in result.split('\n'):
        line = line[:100]
        p.drawString(40, y, line)
        y -= 15
        if y < 50:
            p.showPage()
            y = height - 50

    p.save()
    buffer.seek(0)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    return HttpResponse(buffer, content_type='application/pdf', headers={
        'Content-Disposition': f'attachment; filename="hc-mpnn-summary-{timestamp}.pdf"',
    })