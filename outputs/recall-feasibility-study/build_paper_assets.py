"""Render only verified pilot/confirmation summaries; no dataset access."""
import os
from pathlib import Path
import csv,json,shutil
import numpy as np
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'work/mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study import write
def read(p):
    with open(p,newline='') as f:return list(csv.DictReader(f))
def main():
    verified=json.loads((ROOT/'VALIDATION_CONFIRMATION.json').read_text());assert verified['passed']
    result=json.loads((ROOT/'CONFIRMATION_RESULT.json').read_text());assert result['replicated_H1']
    paper=ROOT.parent/'esann-feasibility-confirmation-paper';out=paper/'generated';out.mkdir(exist_ok=True)
    rr=read(ROOT/'confirmation_aggregate.csv');seed=read(ROOT/'confirmation_seed_summary.csv');dec=read(ROOT/'confirmation_decisions.csv');pol=read(ROOT/'confirmation_policies.csv')
    table=[r'\begin{tabular}{lrrrr}',r'\hline',r'Data & Occupancy & Geometry & Gain (\%) & Gate \\',r'\hline']
    for r in rr:
        d='GloVe' if r['dataset']=='glove' else 'COCO CLIP'
        table.append(d+' & $'+f"{float(r['occupancy_brier_mean']):.4f}\\pm{float(r['occupancy_brier_sd']):.4f}"+'$ & $'+f"{float(r['geometry_brier_mean']):.4f}\\pm{float(r['geometry_brier_sd']):.4f}"+'$ & $'+f"{100*float(r['relative_brier_improvement_mean']):.1f}\\pm{100*float(r['relative_brier_improvement_sd']):.1f}"+'$ & '+r['H1_passes']+r'/5 \\')
    table+=[r'\hline',r'\end{tabular}'];(out/'table_forecasts.tex').write_text('\n'.join(table)+'\n')
    table=[r'\begin{tabular}{llrrrr}',r'\hline',r'Data & Diagnostic & Admit & False admit & False refuse & Correct \\',r'\hline']
    for d in ['glove','coco_clip']:
        for m in ['occupancy','geometry']:
            rs=[r for r in dec if (r['dataset'],r['model'])==(d,m)]
            counts=[sum(int(r[k]) for r in rs) for k in ['admitted','false_admissions','false_refusals','correct']]
            table.append(('GloVe' if d=='glove' else 'COCO CLIP')+' & '+m.capitalize()+' & '+' & '.join(str(v) for v in counts)+r' \\')
    table+=[r'\hline',r'\end{tabular}'];(out/'table_decisions.tex').write_text('\n'.join(table)+'\n')
    plt.rcParams.update({'font.size':7,'axes.labelsize':7,'axes.titlesize':8,'legend.fontsize':6.5,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':600,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(4.8,1.8),layout='constrained')
    for ax,d in zip(axes,['glove','coco_clip']):
        rs=sorted([r for r in seed if r['dataset']==d],key=lambda r:int(r['seed']))
        for field,label,color,marker in [('occupancy_brier','Cap + occupancy','#b65225','s'),('geometry_brier','+ Geometry','#2369a1','o')]:
            ax.plot(range(5),[float(r[field]) for r in rs],marker=marker,ms=3,lw=1,color=color,label=label)
        ax.set_title('GloVe-25' if d=='glove' else 'COCO CLIP text-to-image');ax.set_xlabel('Index seed');ax.set_xticks(range(5));ax.grid(alpha=.15)
    axes[0].set_ylabel('Confirmation severity Brier error');axes[1].legend(frameon=False)
    for ext in ['pdf','png']:fig.savefig(out/f'figure_forecasts.{ext}')
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(4.8,2.6),sharey=True,layout='constrained')
    for i,d in enumerate(['glove','coco_clip']):
        for j,cap in enumerate([.25,.5]):
            ax=axes[i,j];bottom=np.zeros(4)
            for status,label,color in [('issued','Issued','#398060'),('unknown','Unknown','#aab2bc'),('infeasible','Witness','#b65225')]:
                vals=[sum(r['dataset']==d and float(r['cap_fraction'])==cap and int(r['ncal'])==n and r['method']=='uniform' and r['status']==status for r in pol) for n in [100,300,1000,2000]]
                ax.bar(range(4),vals,bottom=bottom,color=color,label=label,width=.7);bottom+=vals
            assert np.all(bottom==5)
            ax.set_title(('GloVe' if d=='glove' else 'COCO CLIP')+f', cap {int(100*cap)}%')
            ax.set_xticks(range(4),['100','300','1000','2000']);ax.set_yticks([0,1,2,3,4,5]);ax.set_ylim(0,5.6)
            if i==1:ax.set_xlabel('Calibration queries')
            if j==0:ax.set_ylabel('Index seeds')
    axes[0,1].legend(frameon=False,ncol=3,loc='lower center',bbox_to_anchor=(.48,1.3),columnspacing=.7)
    for ext in ['pdf','png']:fig.savefig(out/f'figure_certification.{ext}')
    plt.close(fig)
    macros={}
    for r in rr:
        name='Glove' if r['dataset']=='glove' else 'Coco'
        macros[name+'Gain']=f"{100*float(r['relative_brier_improvement_mean']):.1f}"
    (out/'numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n')
    shutil.copyfile(ROOT.parent/'esann-diagnostic-paper/esannV2.cls',paper/'esannV2.cls')
    evidence=paper/'evidence';evidence.mkdir(exist_ok=True)
    for name in ['confirmation_aggregate.csv','confirmation_seed_summary.csv','confirmation_decisions.csv',
                 'confirmation_policies.csv','CONFIRMATION_RESULT.json','CONFIRMATION_LOCK.json','DATA_ACCESS_LEDGER.json',
                 'VALIDATION_CONFIRMATION.json','PROTOCOL.md','FROZEN.json','SOURCE_AUDIT.md','LITERATURE_AUDIT.md']:
        shutil.copyfile(ROOT/name,evidence/name)
    print('Generated confirmation tables and two publication figures from validated outcomes.')
if __name__=='__main__':main()
