"""Publication figures from saved metrics and six cache-backed pipeline previews."""
import csv, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'figures'
FULL=ROOT/'data/experiments/usc12k/full/feature_band_lama_full'
PRE=HERE/'supplementary/refine/previews'
plt.rcParams.update({'font.family':'Arial','font.size':11,'axes.labelsize':11,'axes.titlesize':12,'text.color':'#17324d','axes.labelcolor':'#17324d','xtick.color':'#17324d','ytick.color':'#17324d','savefig.dpi':300})
COLORS=['#33658a','#2a9d8f','#e9a23b']
def save(fig,name):
    fig.savefig(OUT/name,dpi=300,bbox_inches='tight',facecolor='white',pad_inches=.08)
    plt.close(fig)

def charts():
    obj=json.loads((FULL/'posthoc_analysis/posthoc_summary.json').read_text())
    cm=np.array(obj['classification']['feature_band_verify']['confusion_matrix_true_rows_predicted_columns'])
    fig,ax=plt.subplots(figsize=(5.5,3.65),layout='constrained')
    image=ax.imshow(cm,cmap='Blues',vmin=0,vmax=cm.max())
    ax.set_xticks(range(3),['BG','CO','NOCOD']); ax.set_yticks(range(3),['BG','CO','NOCOD'])
    ax.set_xlabel('Predicted label'); ax.set_ylabel('True label')
    for (i,j),v in np.ndenumerate(cm):
        ax.text(j,i,str(v),ha='center',va='center',fontweight='bold',fontsize=14,color='white' if v>cm.max()*.45 else '#17324d')
    for spine in ax.spines.values():spine.set_visible(False)
    fig.colorbar(image,ax=ax,pad=.04,shrink=.9,label='Images')
    save(fig,'confusion_matrix.png')
    rows=list(csv.DictReader((FULL/'posthoc_analysis/pysodmetrics_three_condition_table.csv').open()))
    metrics=[('S_alpha',r'$S_\alpha\;\uparrow$'),('weighted_F_beta',r'$F_\beta^w\;\uparrow$'),('E_m_adaptive',r'$E_{adp}\;\uparrow$'),('MAE',r'MAE $\downarrow$')]
    fig,axs=plt.subplots(1,4,figsize=(10,2.65))
    fig.subplots_adjust(left=.045,right=.99,bottom=.15,top=.72,wspace=.42)
    for ax,(key,label) in zip(axs,metrics):
        vals=[float(row[key]) for row in rows]
        bars=ax.bar(range(3),vals,color=COLORS,width=.7)
        ax.set_title(label,fontweight='bold'); ax.set_ylim(0,max(vals)*1.24)
        ax.set_xticks([]); ax.tick_params(axis='y',labelsize=9)
        ax.spines[['top','right','bottom']].set_visible(False)
        ax.set_axisbelow(True); ax.grid(axis='y',alpha=.15)
        ax.bar_label(bars,labels=[f'{x:.4f}' for x in vals],padding=3,fontsize=9)
    fig.legend([Patch(color=c) for c in COLORS],['Raw','Verify','Refine'],loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,1.02),fontsize=12)
    save(fig,'pysodmetrics_grouped_bars.png')
    fig,ax=plt.subplots(figsize=(10,1.12)); ax.set(xlim=(0,10),ylim=(0,1)); ax.axis('off')
    stages=[('$I$','Original'),('$M_0$','SINet-V2 candidate'),(r'$\hat I_{bg}$','LaMa reconstruction'),('$R$','Frozen feature residual'),('$M_{final}$','Band + optional refine')]
    for i,(a,b) in enumerate(stages):
        x=i*2
        ax.add_patch(FancyBboxPatch((x+.02,.18),1.7,.7,boxstyle='round,pad=0.03',facecolor='#edf4f8',edgecolor=COLORS[0],linewidth=1))
        ax.text(x+.87,.66,a,ha='center',fontsize=15); ax.text(x+.87,.37,b,ha='center',fontsize=9)
        if i<4:ax.annotate('',xy=(x+2,.53),xytext=(x+1.77,.53),arrowprops=dict(arrowstyle='->',color=COLORS[0],lw=1.5))
    save(fig,'pipeline.png')

def qualitative():
    cases=list(csv.DictReader((HERE/'tables/qualitative_cases.csv').open()))
    preds={r['image_id']:r for r in csv.DictReader((HERE/'supplementary/refine/per_image.csv').open())}
    original={r['image_id']:r for r in csv.DictReader((FULL/'feature_band_refine/per_image.csv').open())}
    audit=[]
    fig,axs=plt.subplots(6,5,figsize=(11,10.5))
    fig.subplots_adjust(left=.15,right=.995,top=.965,bottom=.09,wspace=.025,hspace=.06)
    for j,h in enumerate(['Original','Candidate','Reconstructed','Residual','Final']):axs[0,j].set_title(h,fontweight='bold',fontsize=12,pad=9)
    size=(320,240)
    def read(stem,suffix,mask=False):
        pic=Image.open(PRE/f'{stem}_{suffix}.png').convert('L' if mask else 'RGB')
        return np.array(ImageOps.pad(pic,size,method=Image.Resampling.NEAREST if mask else Image.Resampling.LANCZOS,color=0 if mask else '#f2f5f7'))
    for i,case in enumerate(cases):
        stem=case['image_id']; row=preds[stem]; old=original[stem]
        assert row['predicted_label']==old['predicted_label']==case['predicted_class']
        for key in ['verification_score','predicted_mask_area_fraction']:
            assert abs(float(row[key])-float(old[key]))<1e-10,(stem,key)
        audit.append({'image_id':stem,'class_matches_full_run':True,'score':float(row['verification_score']),'area_fraction':float(row['predicted_mask_area_fraction']),'cache_hit':row['cache_hit']})
        im=read(stem,'original'); mask=read(stem,'candidate',True)/255.; candidate=im.copy().astype(float)
        a=mask[...,None]*.45; candidate=candidate*(1-a)+np.array([42,157,143])*a
        bg=read(stem,'background'); residual=read(stem,'residual',True)/255.; final=read(stem,'final',True)/255.
        axs[i,0].imshow(im);axs[i,1].imshow(candidate.astype('uint8'));axs[i,2].imshow(bg)
        heat=axs[i,3].imshow(residual,cmap='Blues',vmin=0,vmax=1)
        axs[i,4].imshow(final,cmap='gray',vmin=0,vmax=1)
        if case['true_class']=='CO':
            gt=read(stem,'ground_truth',True)>127
            for j in [1,4]:axs[i,j].contour(gt,levels=[.5],colors=[COLORS[2]],linewidths=.85)
        for ax in axs[i]:ax.set_xticks([]);ax.set_yticks([]);ax.spines[:].set_visible(False)
        axs[i,0].set_ylabel(f"{case['true_class']} → {row['predicted_label']}\n{case['panel_status'].title()}\nscore {float(row['verification_score']):.3f}",rotation=0,ha='right',va='center',labelpad=10,fontsize=10)
    cax=fig.add_axes([.46,.04,.26,.011]);fig.colorbar(heat,cax=cax,orientation='horizontal',ticks=[0,.5,1]);cax.tick_params(labelsize=8)
    cax.set_title('Feature residual (fixed scale)',fontsize=9,pad=3)
    fig.text(.15,.001,'Teal: candidate   |   Orange: CO ground truth   |   Final: refined grayscale mask',fontsize=9)
    save(fig,'qualitative_cases.png')
    (HERE/'supplementary/consistency_check.json').write_text(json.dumps(audit,indent=2))

if __name__=='__main__':
    charts();qualitative()
    print('Wrote four 300-DPI figures; six supplementary decisions, scores and areas match the full run.')

