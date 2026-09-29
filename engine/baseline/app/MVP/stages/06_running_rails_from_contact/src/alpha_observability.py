"""Exact ambiguity/sensitivity and Monte Carlo checks, independent of data labels."""
from rr_common import *
from track_geometry import *

def main():
    rows=[]
    for deg in (.1,.25,.5,.75,1,1.5,2):
        a=np.deg2rad(deg)
        rows.append(dict(roll_error_deg=deg,near_linear_cm=100*.68*a,far_linear_cm=100*2.20*a,near_exact_cm=100*2*.68*np.sin(a/2),far_exact_cm=100*2*2.20*np.sin(a/2)))
    write_csv(OUT/'alpha_sensitivity.csv',rows)
    B=np.eye(3);C=np.zeros(3);x=np.array([.02,.63,.27,1.56,.28]);q=1
    Px=np.diag(np.array([np.deg2rad(.5),.008,.008,.012,.008])**2);Pc=np.eye(3)*.01**2;ts=np.deg2rad(.2)
    cov,J=rail_covariance(C,B,x,q,Px,Pc,ts);rng=np.random.default_rng(6001);N=100000
    xx=rng.multivariate_normal(x,Px,N);CC=rng.multivariate_normal(C,Pc,N);aa=rng.normal(0,ts,(N,2))
    from scipy.spatial.transform import Rotation
    BB=Rotation.from_rotvec(np.column_stack((np.zeros(N),aa))).as_matrix();pred=rails(CC,BB,xx,q)
    mc=np.array([np.cov(pred[:,j].T) for j in (0,1)])
    check=dict(samples=N,analytic_covariance=cov,monte_carlo_covariance=mc,relative_frobenius_error=[np.linalg.norm(mc[j]-cov[j])/np.linalg.norm(cov[j]) for j in (0,1)])
    assert max(check['relative_frobenius_error'])<.03
    save(OUT/'uncertainty/jacobian_monte_carlo.json',check)
    s=np.arange(0,121,.5);cur=np.column_stack((s,np.zeros_like(s),np.zeros_like(s)));frames_=np.repeat(np.eye(3)[None],len(s),axis=0)
    ambiguous={}
    for name,a in [('constant',np.zeros_like(s)),('smooth_roll',np.deg2rad(2)*np.sin(np.pi*np.maximum(s-8,0)/112)**2)]:
        xx=np.repeat(x[None],len(s),axis=0);xx[:,0]=a;ambiguous[name]=rails(cur,frames_,xx,q)
    np.savez_compressed(OUT/'alpha_observability/identical_cr_different_rails.npz',station=s,contact=cur,**ambiguous)
    (OUT/'alpha_observability/PROOF_RU.md').write_text('''# Наблюдаемость поворота сечения

Известная C(s) определяет положение и касательную, но не угол α(s) вокруг касательной. Для любой гладкой α(s), совпадающей с начальной калибровкой, формулы смещения дают свою пару ходовых рельсов при той же C(s). Следовательно, задача без дополнительных наблюдений или prior неидентифицируема.

Даже точное знание α на первых 8 м не устраняет неоднозначность дальше: можно выбрать гладкую функцию, равную нулю на seed и ненулевую после него. Статистический prior ограничивает вероятные варианты, но не превращает α в измеренную величину.

Для радиуса смещения r ошибка от поворота δα точно равна 2r·|sin(δα/2)|, а при малом угле ≈r·|δα|. Числа 0,68 и 2,20 м здесь служат иллюстрацией рычага detector anchors, не нормативными размерами. Таблица alpha_sensitivity.csv содержит точную и линейную оценки.

Ковариация вычисляется аналитическим Jacobian для α,d,h_near,g,h_far и двух ошибок касательной, с добавлением ковариации положения CR. Проверка на 100000 Monte Carlo реализаций сохранена отдельно. Это проверка вычислений при заданной вероятностной модели; калибровка модели на реальных данных — отдельная задача.
''',encoding='utf-8')
    print('OBSERVABILITY AND JACOBIAN PASS',check['relative_frobenius_error'],flush=True)

if __name__=='__main__':main()
