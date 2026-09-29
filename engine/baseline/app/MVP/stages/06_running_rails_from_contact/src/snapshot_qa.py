from rr_common import *
sys.path.insert(0,str(ROOT/'results_rail2d/_packages'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from seed_input import read_input
import argparse

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run');ap.add_argument('frame',type=int);ap.add_argument('--group',default='phase_a');ap.add_argument('--method',default='B1_BISHOP');args=ap.parse_args()
    folder=OUT/args.group/args.method/key(args.run,args.frame);rec=load(folder/'prediction.json');meta,d=read_input(OUT/rec['input_folder'])
    with np.load(folder/'prediction.npz') as z:p={k:z[k] for k in z.files}
    with np.load(folder/'evaluation.npz') as z:e={k:z[k] for k in z.files}
    with np.load(BASE/'future_gt'/(key(args.run,args.frame)+'_reference_v2.npz')) as z:crgt=z['curve']
    B=d['initial_basis'];C=p['C']@B;pred=p['pair']@B;gt=e['gt']@B;cg=crgt@B
    fig,ax=plt.subplots(2,2,figsize=(15,9))
    for k,axis in enumerate((1,2)):
        a=ax[0,k];a.plot(C[:,0],C[:,axis],color='#b336a4',label='C4 CR');a.plot(cg[:,0],cg[:,axis],':',color='#555',label='CR GT proxy')
        for j,(color,name) in enumerate((('#e74b43','near'),('#256abb','far'))):
            a.plot(pred[:,j,0],pred[:,j,axis],color=color,label='Prediction '+name);a.plot(gt[:,j,0],gt[:,j,axis],'.',color=color,alpha=.55,label='Future GT '+name)
        a.set(xlabel='u, м',ylabel=('v' if axis==1 else 'w')+', м',xlim=(0,C[-1,0]+3),title='Вид сверху' if axis==1 else 'Профиль высоты');a.grid(alpha=.2)
    ax[0,0].legend(fontsize=8,ncol=2)
    ax[1,0].plot(p['s'],np.rad2deg(e['alpha_gt']),label='GT α');ax[1,0].plot(p['s'],np.rad2deg(e['alpha_pred_common']),label='Pred α');ax[1,0].fill_between(p['s'],np.rad2deg(e['alpha_pred_common']-1.96*p['alpha_sigma']),np.rad2deg(e['alpha_pred_common']+1.96*p['alpha_sigma']),alpha=.15);ax[1,0].set(xlabel='CR station, м',ylabel='α, градусы');ax[1,0].legend();ax[1,0].grid(alpha=.2)
    for j,name in enumerate(('near','far')):ax[1,1].plot(p['s'],e['error'][:,j]*100,label=name)
    ax[1,1].plot(p['s'],e['center_error']*100,label='center');ax[1,1].set(xlabel='CR station, м',ylabel='Поперечная ошибка, см');ax[1,1].legend();ax[1,1].grid(alpha=.2)
    fig.suptitle(f'{args.run} / {frames(args.run)[args.frame]["file"]} · {args.method}\nGT показан только после prediction; это разметка и оценённая регистрация, не геодезическая точность')
    fig.tight_layout(rect=(0,0,1,.94));path=OUT/'audit/preliminary_qa'/f'{args.method}__{key(args.run,args.frame)}.png';path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=130);plt.close(fig);print(path)

if __name__=='__main__':main()
