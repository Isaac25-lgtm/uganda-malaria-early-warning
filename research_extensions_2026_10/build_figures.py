"""Standalone figures for presentation/review; only complete scenarios plotted."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research_extensions_2026_10.run_study import RESULTS,OUT,HORIZONS

FIGURES=OUT/'figures'
COLORS=['#176b65','#255277','#bb6b36','#7c698d']


def save(fig,name):
    fig.savefig(FIGURES/f'{name}.png',dpi=300,bbox_inches='tight')
    fig.savefig(FIGURES/f'{name}.pdf',bbox_inches='tight');plt.close(fig)


def main():
    FIGURES.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    metrics=pd.read_csv(RESULTS/'master_metrics.csv');metrics=metrics[metrics.complete]
    old=pd.read_csv(RESULTS/'legacy_recomputed_metrics.csv')
    full=metrics[metrics.experiment.str.fullmatch(r'legacy12_rf_h\d+_start2020_all')].copy()
    full['horizon']=full.experiment.str.extract(r'_h(\d+)_')[0].astype(int)
    full=full.sort_values('horizon')
    fig,ax=plt.subplots(figsize=(8,4.5));x=np.arange(len(HORIZONS));w=.34
    values=old[old.model.eq('rf')].sort_values('horizon').wape.to_numpy()
    ax.bar(x-w/2,values,w,label='Submitted RF',color=COLORS[1]);ax.bar(x+w/2,full.wape,w,label='Corrected RF',color=COLORS[0])
    ax.set_xticks(x,[f'{h} weeks' for h in HORIZONS]);ax.set_ylabel('WAPE (%) · lower is better');ax.set_ylim(0,max(values.max(),full.wape.max())*1.2)
    for i,(a,b) in enumerate(zip(values,full.wape)):
        ax.text(i-w/2,a+.6,f'{a:.2f}',ha='center',fontsize=10);ax.text(i+w/2,b+.6,f'{b:.2f}',ha='center',fontsize=10)
    ax.legend(frameon=False,ncols=2);ax.set_title('Original 12 origins: count-error effect of corrections')
    fig.tight_layout();save(fig,'01_legacy_corrected_rf')

    paired=pd.read_csv(RESULTS/'paired_comparisons.csv')
    effect=paired[paired.comparison.eq('rf_all_minus_none')&paired.metric.eq('wape')]
    fig,ax=plt.subplots(figsize=(8.5,4.6))
    for offset,cluster,color in [(-.09,'district',COLORS[0]),(.09,'origin',COLORS[1])]:
        s=effect[effect.cluster.eq(cluster)].sort_values('horizon');ys=np.arange(len(s))+offset
        ax.errorbar(s.difference_first_minus_second,ys,
            xerr=[s.difference_first_minus_second-s.ci_lower,s.ci_upper-s.difference_first_minus_second],
            fmt='o',capsize=4,label=f'{cluster.capitalize()}-cluster 95% interval',color=color)
    ax.axvline(0,color='#888888',lw=1);ax.set_yticks(range(3),[f'{h} weeks' for h in HORIZONS]);ax.invert_yaxis()
    ax.set_xlabel('WAPE difference: all climate − history only (percentage points)');ax.set_title('RF climate increment: paired fixed-parameter ablation')
    ax.legend(frameon=False,fontsize=10,loc='upper right');fig.tight_layout();save(fig,'02_climate_increment_uncertainty')

    fig,axs=plt.subplots(1,2,figsize=(10,4.3));ys=np.arange(3)
    axs[0].barh(ys,full.tp,label='Correct alerts',color=COLORS[0]);axs[0].barh(ys,full.fp,left=full.tp,label='False alerts',color=COLORS[2])
    axs[0].set_yticks(ys,[f'{h} weeks' for h in HORIZONS]);axs[0].set_xlabel('Number of warnings over 12 target weeks');axs[0].legend(frameon=False,fontsize=9)
    axs[1].barh(ys,full.tp,label='Warned surges',color=COLORS[0]);axs[1].barh(ys,full.fn,left=full.tp,label='Missed surges',color=COLORS[1])
    axs[1].set_yticks(ys,[f'{h} weeks' for h in HORIZONS]);axs[1].set_xlabel('Study-defined surge district-weeks');axs[1].legend(frameon=False,fontsize=9)
    fig.suptitle('Corrected RF warning workload under the fixed Q75 rule');fig.tight_layout();save(fig,'03_warning_workload')

    folds=pd.read_csv(RESULTS/'per_origin_metrics.csv');f=folds[folds.experiment.str.fullmatch(r'legacy12_rf_h\d+_start2020_all')].copy()
    f['horizon']=f.experiment.str.extract(r'_h(\d+)_')[0].astype(int)
    fig,ax=plt.subplots(figsize=(10,4.2))
    for h,color in zip(HORIZONS,COLORS):
        s=f[f.horizon.eq(h)].sort_values('target');ax.plot(pd.to_datetime(s.target),s.wape,'o-',label=f'{h} weeks',color=color)
    ax.axvline(pd.Timestamp('2025-12-22'),ls=':',color='#666666');ax.set_ylabel('WAPE (%)');ax.set_title('RF error by evaluated target week; final week retained')
    ax.legend(frameon=False,ncols=3);fig.tight_layout();save(fig,'04_per_origin_error')

    missing=pd.read_csv(RESULTS/'missingness_year.csv');fig,ax=plt.subplots(figsize=(8,4))
    x=np.arange(len(missing));ax.bar(x,missing.source_absent,label='Absent source row',color=COLORS[1])
    ax.bar(x,missing.confirmed_error,bottom=missing.source_absent,label='Adjudicated outcome error',color=COLORS[2])
    ax.set_xticks(x,missing.epi_year);ax.set_ylabel('District-weeks');ax.set_title('Missingness types by epidemiological year')
    for i,r in missing.iterrows():ax.text(i,r.source_absent+r.confirmed_error+3,str(int(r.source_absent+r.confirmed_error)),ha='center')
    ax.legend(frameon=False);fig.tight_layout();save(fig,'05_missingness')

    grid=pd.read_csv(RESULTS/'threshold_sensitivity.csv');fig,axs=plt.subplots(1,3,figsize=(11,3.8))
    for ax,h in zip(axs,HORIZONS):
        s=grid[grid.horizon.eq(h)&~grid.include2022];z=s.pivot(index='quantile',columns='window',values='f2')
        im=ax.imshow(z.to_numpy(),cmap='viridis',vmin=grid.f2.min(),vmax=grid.f2.max())
        ax.set_xticks(range(3),['±1','±2','±3']);ax.set_yticks(range(3),['Q70','Q75','Q80']);ax.set_title(f'{h} weeks')
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{z.iloc[i,j]:.3f}',ha='center',va='center',color='white')
    fig.colorbar(im,ax=axs,label='F2',shrink=.8);fig.suptitle('Threshold sensitivity changes the surge label, not count predictions')
    save(fig,'06_threshold_sensitivity')

    window=metrics[metrics.experiment.str.fullmatch(r'weekly_common_rf_h\d+_start\d+_all')].copy()
    if len(window)==12:
        window['horizon']=window.experiment.str.extract(r'_h(\d+)_')[0].astype(int)
        window['start']=window.experiment.str.extract(r'_start(\d+)_')[0].astype(int)
        fig,axs=plt.subplots(1,2,figsize=(10,4.2))
        for h,color in zip(HORIZONS,COLORS):
            s=window[window.horizon.eq(h)].sort_values('start')
            axs[0].plot(s.start,s.wape,'o-',label=f'{h} weeks',color=color);axs[1].plot(s.start,s.f2,'o-',label=f'{h} weeks',color=color)
        for ax in axs:ax.set_xticks([2020,2021,2022,2023]);ax.set_xlabel('First training-feature year')
        axs[0].set_ylabel('WAPE (%)');axs[1].set_ylabel('F2');fig.suptitle('Training starts on the same 92 weekly targets in 2024–2025')
        fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.93),ncols=3,frameon=False)
        fig.tight_layout(rect=[0,0,1,.87]);save(fig,'07_training_windows')
    hold=pd.read_csv(RESULTS/'2026_locked_metrics.csv');hold=hold[hold.evaluation.eq('frozen_history')]
    fig,axs=plt.subplots(1,2,figsize=(10,4.2))
    for h,color in zip(HORIZONS,COLORS):
        s=hold[hold.horizon.eq(h)].sort_values('start');axs[0].plot(s.start,s.wape,'o-',label=f'{h} weeks',color=color);axs[1].plot(s.start,s.sensitivity,'o-',label=f'{h} weeks',color=color)
    for ax in axs:ax.set_xticks([2020,2021,2022,2023]);ax.set_xlabel('First training-feature year');ax.legend(frameon=False)
    axs[0].set_ylabel('WAPE (%)');axs[1].set_ylabel('Surge sensitivity');axs[1].set_ylim(0,1)
    fig.suptitle('2026W13–W26: frozen surveillance-only RF, no 2026 refitting');fig.tight_layout();save(fig,'08_2026_frozen_history')
    weekly=paired[paired.comparison.eq('weekly_rf_all_minus_none')&paired.metric.eq('wape')&paired.cluster.eq('moving_block')].sort_values('horizon')
    if len(weekly)==3:
        fig,ax=plt.subplots(figsize=(8,4.2));ys=np.arange(3)
        ax.errorbar(weekly.difference_first_minus_second,ys,
            xerr=[weekly.difference_first_minus_second-weekly.ci_lower,weekly.ci_upper-weekly.difference_first_minus_second],fmt='o',capsize=4,color=COLORS[0])
        ax.axvline(0,color='#888888',lw=1);ax.set_yticks(ys,[f'{h} weeks' for h in HORIZONS]);ax.invert_yaxis()
        ax.set_xlabel('Full climate − history-only WAPE (percentage points)')
        ax.set_title('Exploratory 92-week climate comparison\n95% paired 12-week temporal block intervals');fig.tight_layout();save(fig,'09_weekly_climate_uncertainty')
    perm_path=RESULTS/'grouped_permutation_summary.csv'
    if perm_path.exists():
        perm=pd.read_csv(perm_path);fig,axs=plt.subplots(1,2,figsize=(11,4.3))
        cases=perm[perm.group.eq('cases')].sort_values('horizon');axs[0].bar(range(3),cases['mean'],color=COLORS[:3])
        axs[0].set_xticks(range(3),[f'{h} weeks' for h in HORIZONS]);axs[0].set_title('Surveillance-history block')
        groups=['rainfall','temperature','humidity','climate','district'];x=np.arange(len(groups));width=.23
        for index,(h,color) in enumerate(zip(HORIZONS,COLORS)):
            subset=perm[perm.horizon.eq(h)].set_index('group').loc[groups]
            axs[1].bar(x+(index-1)*width,subset['mean'],width,label=f'{h} weeks',color=color)
        axs[1].axhline(0,color='#888888',lw=1);axs[1].set_xticks(x,['Rain','Temp','RH','All climate','District'])
        axs[1].set_title('Environmental and district blocks');axs[1].legend(frameon=False,fontsize=9)
        for ax in axs:ax.set_ylabel('Mean held-out MAE increase (cases)')
        fig.suptitle('Grouped within-origin permutation: 12 origins, 3 repeats\nDiagnostic, not causal; constant time inputs cannot be assessed',fontsize=12)
        fig.tight_layout();save(fig,'10_grouped_permutation')
    print('Standalone presentation figures refreshed.',flush=True)


if __name__=='__main__':main()
