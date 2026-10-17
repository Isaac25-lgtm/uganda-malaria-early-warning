"""A1 landscape evidence draft, assembled from verified tables/figures."""
from pathlib import Path
import pandas as pd
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader,simpleSplit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from matplotlib.font_manager import findfont,FontProperties
from PIL import Image
from research_extensions_2026_10.run_study import OUT,RESULTS,HORIZONS


def main():
    directory=OUT/'presentation';directory.mkdir(parents=True,exist_ok=True)
    pdfmetrics.registerFont(TTFont('Review',findfont(FontProperties(family='DejaVu Sans'))))
    pdfmetrics.registerFont(TTFont('ReviewBold',findfont(FontProperties(family='DejaVu Sans',weight='bold'))))
    width,height=841*mm,594*mm;path=directory/'poster_evidence_draft_A1.pdf';canvas=Canvas(str(path),pagesize=(width,height))
    navy='#173c4a';teal='#176b65';gray='#42545e';margin=72;gap=36;column=(width-2*margin-2*gap)/3
    xs=[margin+i*(column+gap) for i in range(3)]
    all_metrics=pd.read_csv(RESULTS/'master_metrics.csv')
    windows=all_metrics[all_metrics.experiment.str.fullmatch(r'weekly_common_rf_h\d+_start\d+_all')]
    weekly=pd.read_csv(RESULTS/'paired_comparisons.csv')
    weekly=weekly[weekly.comparison.eq('weekly_rf_all_minus_none')&weekly.metric.eq('wape')&weekly.cluster.eq('moving_block')]

    def text(x,y,value,size=23,bold=False,color=gray,maxwidth=column):
        font='ReviewBold' if bold else 'Review';canvas.setFont(font,size);canvas.setFillColor(color)
        for line in simpleSplit(value,font,size,maxwidth):canvas.drawString(x,y,line);y-=size*1.35
        return y

    def image(path,x,y,w):
        with Image.open(path) as img:ratio=img.height/img.width
        canvas.drawImage(ImageReader(str(path)),x,y,width=w,height=w*ratio,preserveAspectRatio=True,mask='auto')

    canvas.setFillColor(navy);canvas.rect(0,height-225,width,225,fill=1,stroke=0)
    title='Climate-Driven Machine Learning Prediction for District-Level Malaria Surges in Uganda'
    y=text(margin,height-75,title,size=43,bold=True,color='#ffffff',maxwidth=width-2*margin)
    text(margin,height-177,'Isaac Omoding  |  Retrospective research evidence for student and supervisor review',size=23,color='#ffffff',maxwidth=width-2*margin)
    text(margin,height-211,'DRAFT: corrected follow-up, 15 October 2026; submitted thesis results retained separately',size=18,color='#d9e9e4',maxwidth=width-2*margin)

    headings=['DATA AND EVALUATION','CORRECTED COUNT FORECASTS','CLIMATE AND LATER-TIME EVIDENCE']
    for x,heading in zip(xs,headings):
        text(x,height-277,heading,size=25,bold=True,color=teal);canvas.setStrokeColor(teal);canvas.line(x,height-294,x+column,height-294)
    y=height-339
    for item in ['146 districts and cities; weekly confirmed case counts.',
                 '2020–2025: 45,451 assembled district-weeks; 247 absent source rows and six adjudicated outcome errors.',
                 '28 inputs in the full tree models; direct forecasts at 4, 8 and 12 weeks.',
                 'Original evaluation: twelve target weeks per horizon. Follow-up: common weekly 2024–2025 targets.']:
        y=text(xs[0],y,item);y-=18
    window_ready=len(windows)==12 and windows.complete.all() and (OUT/'figures/07_training_windows.png').exists()
    image(OUT/('figures/07_training_windows.png' if window_ready else 'figures/05_missingness.png'),xs[0],585,column)
    y=540
    for item in ['Surge label: training-only district Q75, target week ±2; 2022 excluded from the reference.',
                 'Missing outcomes stay missing. Predictor interpolation requires its later anchor to be available at the origin.',
                 'Counts describe reported caseload, rather than population-standardized infection risk.']:
        y=text(xs[0],y,item,size=21);y-=14

    frame=all_metrics[all_metrics.experiment.str.fullmatch(r'legacy12_rf_h\d+_start2020_all')]
    y=height-341;text(xs[1],y,'Horizon       RF WAPE       Precision       Recall',size=21,bold=True);y-=42
    for h in HORIZONS:
        row=frame[frame.experiment.str.contains(f'_h{h}_')].iloc[0]
        text(xs[1],y,f'{h:>2} weeks       {row.wape:5.2f}%          {row.precision*100:5.1f}%          {row.sensitivity*100:5.1f}%',size=23);y-=40
    xgb=all_metrics[all_metrics.experiment.eq('legacy12_xgb_h12_start2020_all')].iloc[0]
    text(xs[1],y-4,f'12-week XGBoost WAPE: {xgb.wape:.2f}% (RF: 39.64%). RF has higher warning F2; count and warning selection differ.',size=19)
    image(OUT/'figures/01_legacy_corrected_rf.png',xs[1],730,column)
    image(OUT/'figures/03_warning_workload.png',xs[1],310,column)
    text(xs[1],255,'False alerts and missed study-defined surges both matter. WAPE is an error measure; it is not “percentage accuracy”.',size=21)

    weekly_ready=len(weekly)==3 and (OUT/'figures/09_weekly_climate_uncertainty.png').exists()
    image(OUT/('figures/09_weekly_climate_uncertainty.png' if weekly_ready else 'figures/02_climate_increment_uncertainty.png'),xs[2],870,column)
    if weekly_ready:
        favorable=int((weekly.ci_upper<0).sum())
        climate_text=f'Exploratory 92-week comparison: {favorable}/3 temporal-block intervals favor full climate. Original twelve-origin intervals all included zero. Retrospective development evidence, not independent confirmation.'
    else:climate_text='Fixed-parameter RF ablation shows a modest point improvement from climate. All twelve-origin intervals include zero; evidence from these sparse original weeks remains inconclusive.'
    text(xs[2],818,climate_text,size=22)
    image(OUT/'figures/08_2026_frozen_history.png',xs[2],270,column)
    text(xs[2],215,'2026W13–W26: fourteen shared target weeks; surveillance-only RF fitted on pre-2026 outcomes and held fixed. Retrospective temporal evaluation, not prospective deployment.',size=19)

    canvas.setFillColor('#e7efec');canvas.rect(0,0,width,113,fill=1,stroke=0)
    text(margin,77,'Research proof-of-concept: no intended-user usability validation or confirmed operational benefit.',size=18,bold=True,color=navy,maxwidth=width-2*margin)
    text(margin,44,'Review draft only. Student to confirm poster format and final wording; formal REC/MoH research approvals remain pending.',size=17,color=gray,maxwidth=width-2*margin)
    canvas.save();print('Poster evidence draft generated:',path.name,flush=True)


if __name__=='__main__':main()
