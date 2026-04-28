# this program prints all parameters associated with the Dual Hybrid attack. It is based on the following file https://github.com/kevin-carrier/CodedDualAttack/blob/main/OptimizeCodedDualAttack/cost_dual_attack.ipynb
from utilitaries import *
def root_Hermite(b):
    b = RR(b)
    return RR(( (b/(2*pi*e))*((pi*b)**(1/b)))**(1/(2*(b-1))))

def B(alpha, x):
    alpha = RR(alpha)
    x = RR(x)
    if x < 1e-100:
        return RR(1)
    return RR( RR(gamma(RR(alpha+RR(1))) * bessel_J(alpha, x)) / RR((x/RR(2)) ** alpha) )

# survival function of D when dlsc is drawn as a normal with mean avg_dlsc and standard deviation sdv_dlsc (approximation)
def sv_D(T, N, q, m, alpha, nenu, nlat, nfft, kfft, beta0, beta1, dlat, avg_dlsc, sdv_dlsc):
    T = RR(T)
    N = RR(N)
    q = RR(q)
    alpha = RR(alpha)
    m = RR(m)
    nlat = RR(nlat)
    nfft = RR(nfft)
    kfft = RR(kfft)
    nenu = RR(nenu)
    n = nlat + nfft + nenu
    beta0=RR(beta0)
    beta1=RR(beta1)
    dlat=RR(dlat)
    avg_dlsc=RR(avg_dlsc)
    sdv_dlsc=RR(sdv_dlsc)


    # begin new code
    # compute the bound of i
    bi_inf = RR(0)
    bi_sup = RR(sqrt(beta1)*q/2)
    bi_cur = RR((bi_inf + bi_sup)/2)
    while bi_sup - bi_inf > 0.000000001:
        if B(RR(beta1/RR(2)),RR(RR(RR(2)*pi*dlat*bi_cur)/RR(q))) >= RR(T/N):
            bi_inf = bi_cur
        else :
            bi_sup = bi_cur
        bi_cur = RR((bi_inf + bi_sup)/RR(2))
    bi = bi_sup

    VolLat_lat = RR(-beta1*m/(m+nlat)) * RR(log(RR(q),2)) + RR(beta1*(m+nlat-beta1)) * RR(log(RR(root_Hermite(beta0)), 2))
    VolLat_lsc = RR(- kfft)*RR(log(RR(q), 2)) 
    VolLat = RR(VolLat_lat + VolLat_lsc)

    def sv_D_Sphere(dlsc):
        def func_j(ii):
            ii = RR(ii)
            Bi = B(RR(beta1/RR(2)),RR(RR(RR(2)*pi*dlat*ii)/RR(q)))
        
            # compute j such that int_0**oo psi(dlsc) B(j) ddlsc = T/N/B(i)
            Thres = RR(T/N/Bi)
            j_inf = RR(0)
            j_sup = RR(sqrt(nfft)*q/2)
            j_cur = RR((j_inf + j_sup)/2)
            while j_sup - j_inf > 0.000000001:
                if B(RR(nfft/RR(2))-RR(1),RR(RR(RR(2)*pi*dlsc*j_cur)/RR(q))) >= Thres:              
                    j_inf = j_cur
                else :
                    j_sup = j_cur
                j_cur = RR((j_inf + j_sup)/RR(2))
            return j_sup
        
        def VolWrong(ii):
            VolSphere_i = RR(1) + RR(beta1/2) * RR(log(pi, 2)) + RR(beta1-1)*RR(log(RR(ii), 2)) - RR(log(RR(gamma(RR(beta1/2))), 2))
            jj = RR(func_j(ii))
            VolBall_j = RR(nfft/2) * RR(log(pi, 2)) + RR(nfft)*RR(log(RR(jj), 2)) - RR(log(RR(gamma(RR(nfft/2 + 1))), 2))
            return RR(RR(2)**RR(VolSphere_i + VolBall_j))
        
        res = 0
        step = RR(bi/48.0)
        ii = RR(step/RR(2))
        while ii < bi:
            res += RR(step * VolWrong(ii))
            ii += RR(step)
        #res = RR(numerical_integral(VolWrong, 0, bi, algorithm='qag', max_points=100, eps_abs=1e-20, eps_rel=1e-20)[0])
        return min(0, float(VolLat + log(res, 2)))

    res = RR(0)
    norm = RR(0)
    step = RR((3.0*sdv_dlsc)/25.0)
    dlsc = RR(avg_dlsc-3.0*sdv_dlsc) + RR(step/RR(2))
    while dlsc < avg_dlsc+3.0*sdv_dlsc:
        prob = RR(exp(RR(-0.5 * ((dlsc-avg_dlsc)/sdv_dlsc)**RR(2))) / RR(sdv_dlsc*sqrt(2*pi)))
        norm += RR(prob * step)
        res += RR(step * prob * (RR(2)**RR(sv_D_Sphere(dlsc))))
        dlsc += RR(step)
    return float(log(res/norm, 2))



if __name__ == "__main__":
    mlwe = False
    file_name = "mlwe_optimized_withExperimentalPolar.pkl"
    with open(file_name, 'rb') as handle:
            results = pickle.load(handle)

    for scheme in results.keys():
        print(scheme)
        for cost_model in results[scheme].keys():
            print(f"\t{cost_model=}:")
            parameters = results[scheme][cost_model]
            print(f"\t\tcomplexity = ", float(log(parameters['complexity'], 2)))
            
            q = RR(3329)
            alpha = (RR(3) if scheme == Kyber512 else RR(2))
            m = RR(parameters['m'])
            beta0 = RR(parameters['beta0'])
            beta1 = RR(parameters['beta1'])
            nlat = RR(parameters['nlat'])
            nfft = RR(parameters['nfft'])
            kfft = RR(parameters['kfft'])
            nenu = RR(parameters['nenu'])
            n = RR(nlat + nfft + nenu)

            N = RR(parameters['N'])
            Threshold = RR(parameters['treshold'])

            dlat = RR(parameters['dlat'])
            avg_dlsc = RR(parameters['avg_dlsc'])
            sdv_dlsc = RR(parameters['sdv_dlsc'])

            eta = RR(parameters['eta'])
            false_pos = RR(parameters['epsilon'])

            print("\t\tq =", int(q))
            print("\t\talpha =", int(alpha))
            print("\t\tn =", int(n))
            print("\t\tm =", int(m))
            print("\t\tbeta0 =", int(beta0))
            print("\t\tbeta1 =", int(beta1))
            print("\t\tnenu =", int(nenu))
            print("\t\tnfft =", int(nfft))
            print("\t\tkfft =", int(kfft))
            print("\t\tnlat =", int(nlat))
            print("\t\tdlat =", dlat.n())
            print("\t\tavg_dlsc =", avg_dlsc.n())
            print("\t\tsdv_dlsc =", sdv_dlsc.n())
            print("\t\tN = ", float(log(N, 2)))
            print("\t\tT =", float(log(Threshold, 2)))

            print("\t\teta = ", float(eta))
            print("\t\tepsilon temp =", float(log(false_pos, 2)))
            print("\t\tsqrt(4/3)^beta1 = ", RR(log(sqrt(4/3) ** beta1, 2)).n())
            R = RR(parameters['R'])
            print("\t\tlogR = ", float(log(R, 2)))
            print("\t\tT_sample = ", float(log(RR(parameters['T_sample']), 2)))
            print("\t\tN * T_decode = ", float(log(RR(parameters['NT_decode']), 2)))
            print("\t\tT_fft = ", float(log(RR(parameters['T_FFT']), 2)))
            
            sv_D_ = sv_D(Threshold, N, q, m, alpha, nenu, nlat, nfft, kfft, beta0, beta1, dlat, avg_dlsc, sdv_dlsc)
            print("\t\tsv_D(T) = ", sv_D_)
            sv_N_ = log(RR(RR(RR(1) - RR(erf(RR(RR(Threshold)/RR(sqrt(N))))))/RR(2)), 2)
            print("\t\tsv_N(T) = ", sv_N_.n())
            print("\t\tPwrong = ", max(float(sv_N_), float(sv_D_)))
            nb_fp = RR((2 ** max(sv_N_, sv_D_))*R*(q ** kfft))
            print("\t\tR q^kfft Pwrong = ", log(nb_fp, 2).n())
            print()
