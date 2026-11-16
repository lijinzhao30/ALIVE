"""Rebuild the README charts: pip install matplotlib pillow numpy; python docs/visuals/render_benchmarks.py."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
ASSETS=ROOT/'docs/assets'
DATA=json.loads((Path(__file__).parent/'benchmark_scores.json').read_text())
for weight in ['Medium','Bold']:
    font_manager.fontManager.addfont(str(ASSETS/'fonts'/f'Manrope-{weight}.ttf'))
plt.rcParams.update({'font.family':'Manrope README','font.size':12,'font.weight':500,'text.color':'#102B46','axes.labelcolor':'#102B46','svg.fonttype':'path'})

def render(group, title):
    benchmarks=[x for x in DATA['benchmarks'] if x['group']==group]
    fig=plt.figure(figsize=(12,5.6),dpi=120,facecolor='#F5F7FA')
    fig.text(.035,.945,title,fontsize=23,fontweight='bold',va='top')
    fig.legend(handles=[Patch(facecolor=m['color'],label=m['name']) for m in DATA['models']],
               loc='upper left',bbox_to_anchor=(.025,.86),ncol=4,frameon=False,fontsize=11.3,handlelength=1,handleheight=1,columnspacing=1.6)
    n=len(benchmarks);gap=.022;card_w=(.93-gap*(n-1))/n
    moving=[]
    for i,b in enumerate(benchmarks):
        left=.035+i*(card_w+gap)
        card=FancyBboxPatch((left,.065),card_w,.69,boxstyle='round,pad=0.008,rounding_size=0.025',transform=fig.transFigure,facecolor='white',edgecolor='#D2DDE9',linewidth=.85,zorder=0)
        fig.add_artist(card)
        fig.text(left+.017,.716,b['name'],fontsize=17,fontweight='bold',va='top')
        fig.text(left+.017,.66,b['metric'],fontsize=10.4,color='#536A80',va='top')
        ax=fig.add_axes([left+.027,.15,card_w-.044,.438],facecolor='none')
        ax.set_ylim(0,115);ax.set_xlim(-.6,3.6)
        ax.set_yticks([0,20,40,60,80]);ax.set_yticklabels(['0','20','40','60','80'],fontsize=8,color='#728396')
        ax.yaxis.grid(True,color='#E7EDF3',linewidth=.65,zorder=0)
        ax.set_xticks(range(4));ax.set_xticklabels([m['short'] for m in DATA['models']],fontsize=9.2)
        ax.tick_params(axis='both',length=0,pad=9)
        for spine in ['top','right','left']:ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color('#B9C9D8')
        ax.get_xticklabels()[0].set_color(DATA['models'][0]['color'])
        ax.get_xticklabels()[0].set_weight('bold')
        for j,(model,score) in enumerate(zip(DATA['models'],b['scores'])):
            bar=ax.bar(j,score,width=.62,color=model['color'],zorder=2)[0]
            text=ax.text(j,score+2,f'{score:.1f}',ha='center',va='bottom',fontsize=16.5,fontweight='bold',color=model['color'] if j==0 else '#102B46',zorder=4)
            icon=Image.open(ASSETS/'models'/f"{model['icon']}.png").convert('RGBA');icon.thumbnail((40,40))
            box=AnnotationBbox(OffsetImage(np.asarray(icon),zoom=.55),(j,score+25),frameon=True,bboxprops={'boxstyle':'circle,pad=.12','fc':'white','ec':'#DCE5ED','lw':.6},pad=0,zorder=4)
            ax.add_artist(box)
            moving.append((bar,text,box,score,j))
    fig.canvas.draw()
    fig.savefig(ASSETS/f'benchmarks-{group}.png',dpi=180,facecolor=fig.get_facecolor())
    fig.savefig(ASSETS/f'benchmarks-{group}.svg',facecolor=fig.get_facecolor())
    frames=[]
    # Static labels preserve exact final values; only the bars and their labels move.
    for k in range(30):
        progress=1-(1-k/29)**3
        for bar,text,box,score,j in moving:
            height=score*progress
            bar.set_height(height);text.set_position((j,height+2));box.xybox=(j,height+25);box.xy=(j,height+25)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert('RGB'))
    palette=frames[-1].quantize(colors=240)
    indexed=[frame.quantize(palette=palette,dither=Image.Dither.NONE) for frame in frames]
    # Play once and hold on the final, fully readable scores (no flashing/repeated reset).
    indexed[0].save(ASSETS/f'benchmarks-{group}.gif',save_all=True,append_images=indexed[1:],duration=[65]*29+[10000],optimize=True,disposal=1)
    plt.close(fig)

render('understanding','General streaming video understanding')
render('interaction','Streaming interaction')
print('Rendered all five benchmarks with matching static and animated figures.')
