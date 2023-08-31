from ctypes.wintypes import HACCEL
import numpy as np
import pandas as pd

circuit_name = 'test_systems'

# Equações Básicas


def admitance(resistance, reactance):
    return complex((condutance(resistance, reactance)), susceptance(resistance, reactance))


def condutance(resistance, reactance):
    return resistance/(resistance**2 + reactance**2)


def susceptance(resistance, reactance):
    return -reactance/(resistance**2 + reactance**2)

# Modelagem de Barras


def bus_data(name, model, voltage_amplitude, voltage_angle, active_power, reactive_power):
    return {
        "name": name,
        "type": model,
        "$V$":  voltage_amplitude,
        "$\\theta$": voltage_angle,
        "$P$": active_power,
        "$Q$": reactive_power, }


def bus_v_theta(name, voltage_amplitude, voltage_angle):
    return bus_data(name=name,
                    model="$(V,\\theta)$",
                    voltage_amplitude=voltage_amplitude,
                    voltage_angle=voltage_angle,
                    active_power=np.nan,
                    reactive_power=np.nan,
                    )


def bus_p_q(name, active_power, reactive_power):
    return bus_data(name=name,
                    model="$(P,Q)$",
                    voltage_amplitude=np.nan,
                    voltage_angle=np.nan,
                    active_power=active_power,
                    reactive_power=reactive_power,
                    )


def bus_p_v(name, active_power, voltage_amplitude):
    return bus_data(name=name,
                    model="$(P,V)$",
                    voltage_amplitude=voltage_amplitude,
                    voltage_angle=np.nan,
                    active_power=active_power,
                    reactive_power=np.nan,
                    )

# Modelagem de Ramos


def branch_data(from_bus, to_bus, model_data):
    return {"from_bus": from_bus,
            "to_bus": to_bus,
            **model_data
            }


def generalized_model(transform_ratio, phase_shift_angle, series_resistance, series_reactance, shunt_susceptance):
    '''Modelo generalizado
       |-------------OO----------[   ]------|
    Barra k     1:ae^{-jphi}       y     Barra m
    '''
    return {
        "$a$": transform_ratio,
        "$\\varphi$": phase_shift_angle,
        "$r$": series_resistance,
        "$x$": series_reactance,
        "$b^{sh}$": shunt_susceptance,
        "$g$": condutance(series_resistance, series_reactance),
        "$b$": susceptance(series_resistance, series_reactance),
        "$y$": admitance(series_resistance, series_reactance),
    }


def transmission_line(name, series_resistance, series_reactance, shunt_susceptance):
    return {"name": name,
            "type": 'transmission_line',
            **generalized_model(transform_ratio=1,
                                phase_shift_angle=0,
                                series_resistance=series_resistance,
                                series_reactance=series_reactance,
                                shunt_susceptance=shunt_susceptance)}


def get_bus_list(df_branch):
    return sorted(list(set(df_branch["from_bus"].to_list() + df_branch["to_bus"].to_list())))


def get_bus_dict(df_branch):
    return {bus_name: index for index,
            bus_name in enumerate(get_bus_list(df_branch))}

# Parte 1: Determinar a matriz de adimitância nodal


def admitance_matrix(df_branch):

    bus_list = get_bus_list(df_branch)
    bus_dict = get_bus_dict(df_branch)
    number_of_bus = len(bus_dict)
    Y = np.zeros((number_of_bus, number_of_bus), dtype=complex)

    for index, data in df_branch.iterrows():
        k = bus_dict[data['from_bus']]
        m = bus_dict[data['to_bus']]

        Y[k, m] = -data['$a$']*np.exp(-1j*data['$\\varphi$'])*data['$y$']
        Y[m, k] = Y[k, m]

    # Adicionalmente ao shunt do ramo, falta adicionar o elemento shunt jb_{k}^{sh} da barra.
    for bus in bus_list:
        k = bus_dict[bus]
        df = df_branch.query(f"from_bus == '{bus}' or to_bus == '{bus}'")
        for index, data in df.iterrows():
            Y[k, k] += 1j*data['$b^{sh}$'] + (data['$a$']**2)*data['$y$']
    return Y


def condutance_matrix(Y):
    return np.real(Y)


def susceptance_matrix(Y):
    return np.imag(Y)


def matrix_to_markdown(M, name_list=None):
    if name_list is not None:
        return pd.DataFrame(M, columns=name_list, index=name_list).to_markdown()
    else:
        return pd.DataFrame(M).to_markdown()
    pass


def flat_start(df_bus):
    for index, data in df_bus.iterrows():
        if pd.isna(data['$V$']):
            df_bus['$V$'][index] = 1.0
        if pd.isna(data['$\\theta$']):
            df_bus['$\\theta$'][index] = 0
    return df_bus

# Algotimo para atualizar as potências


def get_branch_voltage(series_branch, df_bus):
    k, m = get_branch_node_names(series_branch)
    Vk = df_bus.loc[k, '$V$']
    Vm = df_bus.loc[m, '$V$']
    return Vk, Vm


def get_branch_angle(series_branch, df_bus):
    k, m = get_branch_node_names(series_branch)
    theta_km = df_bus.loc[k, '$\\theta$'] - df_bus.loc[m, '$\\theta$']
    return theta_km


def get_branch_node_names(series_branch):
    k = series_branch['from_bus']
    m = series_branch['to_bus']
    return k, m


def get_branch_node_index(series_branch):
    k = bus_dict[series_branch['from_bus']]
    m = bus_dict[series_branch['to_bus']]
    return k, m


def get_bus_conection_dictionary(bus, df_branch):
    filter_branch = df_branch.query(
        f"from_bus == '{bus}' or to_bus == '{bus}'")
    branch_dictionary = []
    for _, x in filter_branch.iterrows():
        if x['to_bus'] == bus:
            k = bus
            m = x['from_bus']
        else:
            k = x['from_bus']
            m = x['to_bus']
        branch_dictionary.append({'from_bus': k, 'to_bus': m})

    branch_dictionary.append({'from_bus': bus, 'to_bus': bus})
    return branch_dictionary


def get_branch_condutance(series_branch, G):
    k, m = get_branch_node_index(series_branch)
    Gkm = G[k, m]
    return Gkm


def get_branch_susceptance(series_branch, B):
    k, m = get_branch_node_index(series_branch)
    Bkm = B[k, m]
    return Bkm


def get_branch_data_values(series_branch, df_bus):
    Vk, Vm, = get_branch_voltage(series_branch, df_bus)
    theta_km = get_branch_angle(series_branch, df_bus)
    Gkm = get_branch_condutance(series_branch, G)
    Bkm = get_branch_condutance(series_branch, B)
    return Vk, Vm, theta_km, Gkm, Bkm


def is_bus_slack(bus, df_bus):
    if df_bus.loc[bus, 'type'] == '$(V,\\theta)$':
        return True
    else:
        return False


def is_bus_pq(bus, df_bus):
    if df_bus.loc[bus, 'type'] == '$(P,Q)$':
        return True
    else:
        return False


def is_bus_pv(bus, df_bus):
    if df_bus.loc[bus, 'type'] == '$(P,V)$':
        return True
    else:
        return False


def update_active_power(Pk, Vk, Vm, theta_km, Gkm, Bkm):
    Pk += Vk*Vm*(Gkm*np.cos(theta_km) + Bkm*np.sin(theta_km))
    return Pk


def update_reactive_power(Qk, Vk, Vm, theta_km, Gkm, Bkm):
    Qk += Vk*Vm*(Gkm*np.sin(theta_km) - Bkm*np.cos(theta_km))
    return Qk


def bus_active_power(bus, df_bus, df_branch):
    # if is_bus_slack(bus, df_bus):
    Pk = 0
    # branch = df_branch.query(f"from_bus == '{bus}' or to_bus == '{bus}'")
    branch_dictionary = get_bus_conection_dictionary(bus, df_branch)
    for branch in branch_dictionary:
        Vk, Vm, theta_km, Gkm, Bkm = get_branch_data_values(branch, df_bus)
        Pk = update_active_power(Pk, Vk, Vm, theta_km, Gkm, Bkm)
    # else:
    #    Pk = df_bus.loc[bus, '$P$']
    return Pk


def bus_reactive_power(bus, df_bus, df_branch):
    # if is_bus_slack(bus, df_bus) or is_bus_pv(bus, df_bus):
    Qk = 0
    # branch = df_branch.query(f"from_bus == '{bus}' or to_bus == '{bus}'")
    # for index, series_branch in branch.iterrows():
    branch_dictionary = get_bus_conection_dictionary(bus, df_branch)
    for branch in branch_dictionary:
        Vk, Vm, theta_km, Gkm, Bkm = get_branch_data_values(branch, df_bus)
        Qk = update_reactive_power(Qk, Vk, Vm, theta_km, Gkm, Bkm)
    # else:
    #    Qk = df_bus.loc[bus, '$Q$']
    return Qk


def residuals_active_power(bus_list, df_bus, df_branch):
    dP = list()
    for bus in bus_list:
        Pk = bus_active_power(bus, df_bus, df_branch)
        if is_bus_pq(bus, df_bus) or is_bus_pv(bus, df_bus):
            dPk = df_bus.loc[bus, '$P$'] - Pk
        else:
            dPk = 0
        dP.append(dPk)
    return dP


def residuals_reactive_power(bus_list, df_bus, df_branch):
    dQ = list()
    for bus in bus_list:
        Qk = bus_reactive_power(bus, df_bus, df_branch)
        if is_bus_pq(bus, df_bus):
            dQk = df_bus.loc[bus, '$Q$'] - Qk
        else:
            dQk = 0
        dQ.append(dQk)
    return dQ


def jacobian_matrix(bus_dict, bus_list, df_bus, df_branch):
    number_of_bus = len(bus_list)
    H = np.zeros((number_of_bus, number_of_bus))
    N = np.zeros((number_of_bus, number_of_bus))
    M = np.zeros((number_of_bus, number_of_bus))
    L = np.zeros((number_of_bus, number_of_bus))
    for bus_k in bus_list:
        for bus_m in bus_list:
            branch = {'from_bus': bus_k, 'to_bus': bus_m}
            k, m = get_branch_node_index(branch)
            Vk, Vm, theta_km, Gkm, Bkm = get_branch_data_values(
                branch, df_bus)

            if k == m:
                Pk = bus_active_power(bus_k, df_bus, df_branch)
                Qk = bus_reactive_power(bus_k, df_bus, df_branch)

                H[k, k] = (-Bkm*(Vk**2) - Qk)
                N[k, k] = (Gkm*(Vk**2) + Pk)/Vk
                M[k, k] = (-Gkm*(Vk**2) + Pk)
                L[k, k] = (-Bkm*(Vk**2) + Qk)/Vk

                if is_bus_slack(bus_k, df_bus):
                    H[k, k] = np.inf
                    L[k, k] = np.inf
                elif is_bus_pv(bus_k, df_bus):
                    L[k, k] = np.inf

            else:
                H[k, m] = +Vk*Vm*(Gkm*np.sin(theta_km)-Bkm*np.cos(theta_km))
                N[k, m] = +Vk*(Gkm*np.cos(theta_km)+Bkm*np.sin(theta_km))
                M[k, m] = -Vk*Vm*(Gkm*np.cos(theta_km) + Bkm*np.sin(theta_km))
                L[k, m] = +Vk*(Gkm*np.sin(theta_km)-Bkm*np.cos(theta_km))

    return np.vstack((np.hstack((H, N)), np.hstack((M, L))))


def update_voltage_and_angle(solution, bus_list, df_bus):
    index = 0
    for bus in bus_list:
        df_bus.loc[bus, '$\\theta$'] += solution[index]
        index += 1

    for bus in bus_list:
        df_bus.loc[bus, '$V$'] += solution[index]
        index += 1
    return df_bus


def report_input_data():
    print(f'''# Equipments' data   
             
**Table 1.1**. Transmission Line Data
      
{df_transmission_lines.to_markdown()}

# Power Flow Inputs 

**Table 2.1**. Bus data

{df_bus.to_markdown()}

**Table 2.2**. Branch data

{df_branch.to_markdown()}''')


def report_input_matrices():
    print(f'''# System's Admitance Matrix
          
The procedure to calculate the admitance matrix is the following:
          
$Y_{{km}} = -a_{{km}}e^{{-j\\varphi}}$
          
$Y_{{kk}} = jb_{{k}}^{{sh}} + \sum_{{m \in \Omega_{{k}}}}(jb_{{km}}^{{sh}} + a^{{2}}_{{km}}y_{{km}})$
          
$Y = G +jB$
          
Here we report the results in Table 2.1 and Table 2.2
          
**Matrix 3.1**. Condutance matrix (G)

{matrix_to_markdown(G,bus_list)}


**Matrix 3.2**. Susceptance Matrix (B)

{matrix_to_markdown(B,bus_list)}

          ''')


def report_newton_iters():
    rep_table = df_bus.loc[:, ['$V$', '$\\theta$']].to_markdown()
    print(f''' ## Iter {iter}

**Residuals**         
- $|\\Delta P|^{{max}} = {max(np.abs(delta_P))}$
- $|\\Delta Q|^{{max}} = {max(np.abs(delta_Q))}$


**Matrix 4.{iter}.1**. Jacobian matrix (J)
          
{matrix_to_markdown(J)}

**Table 4.{iter}.1**. Iter's Voltages (p.u) and Angles (rad)

{rep_table}

          ''')


Sbase = 1000  # kVA

# Modelagem de sistema elétrico de 3 Barras
# Equipamentos
# Linhas de Transmissão
linha_a = transmission_line(name='L0A',
                            series_resistance=0.322,
                            series_reactance=0.270,
                            shunt_susceptance=0.0)
linha_b = transmission_line(name='L0B',
                            series_resistance=0.493,
                            series_reactance=0.2511,
                            shunt_susceptance=0.0)
linha_c = transmission_line(name='L0C',
                            series_resistance=0.3660,
                            series_reactance=0.1864,
                            shunt_susceptance=0.0)
linha_d = transmission_line(name='L0C',
                            series_resistance=0.3811,
                            series_reactance=0.1941,
                            shunt_susceptance=0.0)

Line = [linha_a, linha_b, linha_c, linha_d]
# Barramentos
Bus = [
    bus_v_theta(name="B01",
                voltage_amplitude=1.0,
                voltage_angle=0.0),
    bus_p_q(name="B02",
            active_power=1000/Sbase,
            reactive_power=600/Sbase),
    bus_p_q(name="B03",
            active_power=900/Sbase,
            reactive_power=400/Sbase),
    bus_p_q(name="B04",
            active_power=1200/Sbase,
            reactive_power=800/Sbase),
    bus_p_q(name="B05",
            active_power=800/Sbase,
            reactive_power=600/Sbase)
]

# Ramos
Branch = [
    branch_data(from_bus="B01",
                to_bus="B02",
                model_data=linha_a,),
    branch_data(from_bus="B02",
                to_bus="B03",
                model_data=linha_b,),
    branch_data(from_bus="B03",
                to_bus="B04",
                model_data=linha_c,),
    branch_data(from_bus="B04",
                to_bus="B05",
                model_data=linha_d,)
]


# Sistema Monticelli 2 barras para validar o newton-raphson
'''
[V,theta] ----------- [P,Q]
linha_1 = transmission_line(name='L01',
                            series_resistance=0.20,
                            series_reactance=1.00,
                            shunt_susceptance=0.02)
Line = [linha_1]

Bus = [
    bus_v_theta(name="B01",
                voltage_amplitude=1.0,
                voltage_angle=0.0),
    bus_p_q(name="B02",
            active_power=-0.3,
            reactive_power=0.07)
]

Branch = [
    branch_data(from_bus="B01",
                to_bus="B02",
                model_data=linha_1,)
]
'''

'''
# [V,theta] ----------- [P,V]
linha_1 = transmission_line(name='L01',
                            series_resistance=0.20,
                            series_reactance=1.00,
                            shunt_susceptance=0.02)
Line = [linha_1]
Bus = [
    bus_v_theta(name="B01",
                voltage_amplitude=1.0,
                voltage_angle=0.0),
    bus_p_v(name="B02",
            active_power=-0.4,
            voltage_amplitude=1.0)
]

Branch = [
    branch_data(from_bus="B01",
                to_bus="B02",
                model_data=linha_1,)
]
'''
# Dados do Sistema elétrico
df_bus = pd.DataFrame(Bus)
df_branch = pd.DataFrame(Branch)
df_transmission_lines = pd.DataFrame(Line)
bus_list = get_bus_list(df_branch)
bus_dict = get_bus_dict(df_branch)

# Matrizes
Y = admitance_matrix(df_branch)
G = condutance_matrix(Y)
B = susceptance_matrix(Y)


report_input_data()
report_input_matrices()

# newton raphson
print("# Newton's method for load flow")
tol = 1e-3
df_bus = flat_start(df_bus)
df_bus = df_bus.set_index('name')
iter = 0
while (1):
    delta_P = residuals_active_power(bus_list, df_bus, df_branch)
    delta_Q = residuals_reactive_power(bus_list, df_bus, df_branch)
    g = np.concatenate((delta_P, delta_Q))

    if np.max(np.abs(g)) <= tol:
        break

    J = jacobian_matrix(bus_dict, bus_list, df_bus, df_branch)
    solution = np.linalg.solve(J, g)

    df_bus = update_voltage_and_angle(solution, bus_list, df_bus)
    iter += 1
    report_newton_iters()


# Report solution
for bus in bus_list:
    df_bus.loc[bus, '$P$'] = bus_active_power(bus, df_bus, df_branch)
    df_bus.loc[bus, '$Q$'] = bus_reactive_power(bus, df_bus, df_branch)

print(f'''# Power Flow Report
      
## Stop Criterion      
if $\Delta P$ and $\Delta Q$ $\leq$ {tol}
**Residuals**         
- $|\\Delta P|^{{max}} = {max(np.abs(delta_P))}$
- $|\\Delta Q|^{{max}} = {max(np.abs(delta_Q))}$

**Table 5.1**. Bus data
      
{df_bus.to_markdown()}
            
''')
