"""Render a reference diagram without overwriting the author-designed README artwork."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
BG, INK, MUTED = '#f4f7fa', '#152c40', '#4b6275'
TEAL, BLUE = '#087f79', '#355a96'


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans'})
    fig, ax = plt.subplots(figsize=(14, 7.1), facecolor=BG)
    ax.set(xlim=(0, 140), ylim=(0, 71))
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.text(6, 64, 'MONTHLY RAINFALL FORECASTING', fontsize=22, weight='bold', color=INK)
    ax.text(6, 59, 'South America  /  Competition hybrid  /  WorCAP 2026', fontsize=12, color=MUTED)

    def box(x,y,w,h,title,body,color=TEAL):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.7,rounding_size=1.1',facecolor='white',edgecolor='#cfdae2',lw=1.1))
        ax.plot([x+1,x+1],[y+2,y+h-2],color=color,lw=3,solid_capstyle='round')
        ax.text(x+3,y+h-3.4,title,fontsize=11.5,weight='bold',va='top',color=INK)
        ax.text(x+3,y+h-8,body,fontsize=10.3,va='top',color=MUTED,linespacing=1.45)
    def arrow(x,y,xx,yy):
        ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=13,color='#8396a5',lw=1.3))

    box(6,23,32,28,'CLIMATE INPUTS','Atmospheric fields\nOcean indices + seasonality\nSEAS5 + CFSv2 forecasts')
    box(6,8,32,10,'TRAINING REFERENCES','Monthly climatologies',BLUE)
    box(47,39,39,13,'LIGHTGBM · SEASONAL','29 features · SEAS5 mean / anomaly')
    box(47,23,39,13,'LIGHTGBM · MULTISYSTEM','31 features · adds CFSv2',BLUE)
    box(47,7,39,13,'LOCAL RIDGE','2 forecast anomalies · one fit per pixel')
    # The input bus indicates common preprocessing, not identical model features.
    ax.plot([41,41],[13.5,45.5],color='#8396a5',lw=1.3)
    arrow(38.8,37,41,37)
    for y in [45.5,29.5,13.5]:arrow(41,y,46,y)
    arrow(38.8,13,41,13)
    for y,weight in [(45.5,'37.5%'),(29.5,'37.5%'),(13.5,'25%')]:
        arrow(87,y,104,y)
        ax.text(95.5,y+1.6,weight,fontsize=11,weight='bold',color=INK,ha='center')
    ax.plot([105,105],[13.5,45.5],color='#8396a5',lw=1.3)
    arrow(105,29.5,109,29.5)
    box(110,18,24,23,'HYBRID FORECAST','Monthly field\n301 × 261 grid\nmm/day → CSV')
    ax.text(6,2.5,'Up to 30 years of training  •  Chronological validation  •  Input and CSV integrity checks',fontsize=10,color=MUTED)
    path=ROOT/'outputs/model_overview_generated.png'
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=160,facecolor=BG)
    plt.close(fig)
    print('Rendered',path.relative_to(ROOT))


if __name__=='__main__':main()
