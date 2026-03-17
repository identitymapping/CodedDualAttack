This branch `eprint-2026-279` is derived from a fork of the repository at: https://github.com/TabOg/CodedDualAttack/blob/7b7e8000/.

---

In the public code repository at https://github.com/TabOg/CodedDualAttack/blob/7b7e8000/, the function call `lambda t: lambda_2(t, nfft, alpha, dlat, q)` at Line 124 of `OptimizeCodedDualAttack/rot_utilitaries.py` contains a parameter mismatch. 
According to the original source code in [CMST25, C] (Line 114 of https://github.com/kevin-carrier/CodedDualAttack/blob/main/OptimizeCodedDualAttack/utilitaries.py :
`res = RR(beta1 * numerical_integral(lambda t: RR(RR(t**RR(beta1 - 1)) * RR(exp(RR(-alpha*(pi*t*dlat/q)**2)))), 0, 1)[0])`), 
it should instead be `lambda t: lambda_2(t, beta1, alpha, dlat, q)` (or `lambda t: lambda_1(t, beta1, alpha, dlat, q)`, as the `lambda_1` and `lambda_2` functions are functionally identical). 

The definitions of the `lambda_1` and `lambda_2` functions are as follows:
``` python
# The code is from https://github.com/TabOg/CodedDualAttack/blob/7b7e8000071b5379048af92e5b49a84881f7e8f8/OptimizeCodedDualAttack/rot_utilitaries.py
def lambda_1(t, beta1, alpha, dlat, q):
    return RR(RR(t**RR(beta1 - 1)) * RR(exp(RR(-alpha*(pi*t*dlat/q)**2))))
def lambda_2(t, nfft, alpha, dlsc, q):
    return RR(RR(t**RR(nfft - 1)) * RR(exp(RR(-alpha*(pi*t*dlsc/q)**2))))
```

---

### Code Modification
We modified the code at **OptimizeCodedDualAttack/rot_utilitaries.py Line 124**:

| Type       | Code Snippet                                                                 |
|------------|------------------------------------------------------------------------------|
| Original   | `lambda_2_ = lambda t: lambda_2(t, nfft, alpha, dlat, q)`                    |
| Modified   | `lambda_2_ = lambda t: lambda_2(t, beta1, alpha, dlat, q)`                   |

---

### Execution Results
We re-ran `OptimizeCodedDualAttack/optimizer_naive.py` and `OptimizeCodedDualAttack/rot_optimizer_naive.py`; the execution results are saved to `OptimizeCodedDualAttack/out_optimizer_naive.txt` and `OptimizeCodedDualAttack/out_rot_optimizer_naive.txt` respectively. 

Table 1.1 presented in our paper is generated based on the experimental data recorded in `OptimizeCodedDualAttack/out_optimizer_naive.txt` and `OptimizeCodedDualAttack/out_rot_optimizer_naive.txt`.