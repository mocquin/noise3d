


def var_t_err_tvh(T, V, H, var): return var/(H*V)*(1 - 1/T)
def var_v_err_tvh(T, V, H, var): return var/(H*T)*(1 - 1/V)
def var_h_err_tvh(T, V, H, var): return var/(T*V)*(1 - 1/H)
def var_tv_err_tvh(T, V, H, var): return var/H*(1 - 1/T - 1/V + 1/(T*V))
def var_th_err_tvh(T, V, H, var): return var/V*(1 - 1/T - 1/H + 1/(T*H))
def var_vh_err_tvh(T, V, H, var): return var/T*(1 - 1/V - 1/H + 1/(V*H))
def var_tvh_err_tvh(T, V, H, var): return var*(1 - 1/T - 1/V - 1/H + 1/(T*V) + 1/(T*H) + 1/(V*H) - 1/(T*V*H))
 