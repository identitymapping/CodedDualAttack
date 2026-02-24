# this program prints all parameters associated with RotDualHybrid. It is based on the following file https://github.com/kevin-carrier/CodedDualAttack/blob/main/OptimizeCodedDualAttack/cost_dual_attack.ipynb 
from rot_utilitaries import *

file_name = "rotated_optimized_withExperimentalPolar.pkl"

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
