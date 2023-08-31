import pandapower as pp

net = pp.create_empty_network()  # create an empty network
# pp.create_bus(net,
#                     name="B01_FOZ_AREIA",
#                     vn_kv=500,
#                     index=None,
#                     geo_data=None,  # (x,y)
#                     type="b",       # bus type
#                     zone=1,
#                     max_vm_pu=10,
#                     min_vm_pu=0,
#                     )
bus1 = pp.create_bus(net,
                     name="B01_FOZ_AREIA",
                     vn_kv=500,
                     type="b",       # bus type
                     zone=1,
                     max_vm_pu=10,
                     min_vm_pu=0,
                     )
bus2 = pp.create_bus(net,
                     name="B02_S.SANTIAGO",
                     vn_kv=500,
                     type="b",       # bus type
                     zone=1,
                     max_vm_pu=10,
                     min_vm_pu=0,
                     )
bus3 = pp.create_bus(net,
                     name="B03_S.SEGREDO",
                     vn_kv=500,
                     type="b",       # bus type
                     zone=1,
                     max_vm_pu=10,
                     min_vm_pu=0,
                     )
bus4 = pp.create_bus(net,
                     name="B04_ITAIPU",
                     vn_kv=765,
                     type="b",       # bus type
                     zone=1,
                     max_vm_pu=10,
                     min_vm_pu=0,
                     )
bus5 = pp.create_bus(net,
                     name="B05_IVAIPORA",
                     vn_kv=500,
                     type="b",       # bus type
                     zone=1,
                     max_vm_pu=10,
                     min_vm_pu=0,
                     )
bus6 = pp.create_bus(net,
                     name="B06_IVAIPORA",
                     vn_kv=765,
                     type="b",       # bus type
                     zone=1,
                     max_vm_pu=10,
                     min_vm_pu=0,
                     )
bus7 = pp.create_bus(net,
                     name="EQUIVALENTE",
                     vn_kv=765,
                     type="b",       # bus type
                     zone=1,
                     max_vm_pu=10,
                     min_vm_pu=0,
                     )
# Create an external grid connection
pp.create_ext_grid(net, bus7, vm_pu=1.03, va_degree=0)

# generator data
pp.create_gen(net, bus1, p_mw=1658, vm_pu=1.03)
pp.create_gen(net, bus2, p_mw=1332, vm_pu=1.03)
pp.create_gen(net, bus3, p_mw=1540, vm_pu=1.03)
pp.create_gen(net, bus4, p_mw=6500, vm_pu=1.03)
# pp.create_gen(net, bus7, p_mw=1658, vm_pu=1.03)

# load data
pp.create_load(net, bus1, p_mw=2405, q_mvar=-467.0)
pp.create_load(net, bus2, p_mw=692.3, q_mvar=-184.0)
pp.create_load(net, bus3, p_mw=688.2, q_mvar=-235.0)
pp.create_load(net, bus4, p_mw=62.6, q_mvar=24.3)
pp.create_load(net, bus5, p_mw=845.8, q_mvar=-9.2)
pp.create_load(net, bus6, p_mw=-4.9, q_mvar=79.8)
pp.create_load(net, bus7, p_mw=2884, q_mvar=-196.0)

# shunt data
pp.create_shunt(net, bus1, name='FOZ AREIA',
                q_mvar=179.2, p_mw=0.0, vn_kv=500,
                step=1, max_step=1,
                in_service=True, index=None)
pp.create_shunt(net, bus2, name='S.SANTIAGO',
                q_mvar=149.1, p_mw=0.0, vn_kv=500,
                step=1, max_step=1,
                in_service=True, index=None)
pp.create_shunt(net, bus3, name='S. SEGREDO',
                q_mvar=114.2, p_mw=0.0, vn_kv=500,
                step=1, max_step=1,
                in_service=True, index=None)
pp.create_shunt(net, bus4, name='ITAIPU',
                q_mvar=36.8, p_mw=0.0, vn_kv=765,
                step=1, max_step=1,
                in_service=True, index=None)
pp.create_shunt(net, bus5, name='IVAIPORA',
                q_mvar=33.0, p_mw=0.0, vn_kv=500,
                step=1, max_step=1,
                in_service=True, index=None)
pp.create_shunt(net, bus6, name='IVAIPORA',
                q_mvar=2142, p_mw=0.0, vn_kv=765,
                step=1, max_step=1,
                in_service=True, index=None)
pp.create_shunt(net, bus7, name='EQUIVALENTE',
                q_mvar=2142, p_mw=0.0, vn_kv=765,
                step=1, max_step=1,
                in_service=True, index=None)
# line data
pp.create_line_from_parameters(net,
                               name='L01',
                               from_bus=bus1, to_bus=bus3,
                               length_km=250,
                               r_ohm_per_km=0.0030,
                               x_ohm_per_km=0.0380,
                               c_nf_per_km=0.0000,
                               max_i_ka=1e20)
pp.create_line_from_parameters(net,
                               name='L02',
                               from_bus=bus1, to_bus=bus5,
                               length_km=250,
                               r_ohm_per_km=0.0190,
                               x_ohm_per_km=0.2450,
                               c_nf_per_km=0.0000,
                               max_i_ka=1e20)
pp.create_line_from_parameters(net,
                               name='L03',
                               from_bus=bus2, to_bus=bus3,
                               length_km=250,
                               r_ohm_per_km=0.0050,
                               x_ohm_per_km=0.0760,
                               c_nf_per_km=0.0000,
                               max_i_ka=1e20)
pp.create_line_from_parameters(net,
                               name='L04',
                               from_bus=bus2, to_bus=bus5,
                               length_km=250,
                               r_ohm_per_km=0.0150,
                               x_ohm_per_km=0.2250,
                               c_nf_per_km=0.0000,
                               max_i_ka=1e20)
pp.create_line_from_parameters(net,
                               name='L05',
                               from_bus=bus4, to_bus=bus6,
                               length_km=585.225,
                               r_ohm_per_km=0.0029,
                               x_ohm_per_km=0.0734,
                               c_nf_per_km=0.0000,
                               max_i_ka=1e20)
pp.create_line_from_parameters(net,
                               name='L06',
                               from_bus=bus6, to_bus=bus7,
                               length_km=585.225,
                               r_ohm_per_km=0.0040,
                               x_ohm_per_km=0.0570,
                               c_nf_per_km=0.0000,
                               max_i_ka=1e20)


# Create an transformer
trafo1 = pp.create_transformer_from_parameters(net,
                                               hv_bus=bus6,
                                               lv_bus=bus5,
                                               std_type="765kV/500kV",
                                               name="765kV/500kV IVAIPORA",
                                               sn_mva=10000,
                                               vn_hv_kv=765,
                                               vn_lv_kv=500,
                                               vk_percent=0,
                                               vkr_percent=0,
                                               pfe_kw=0,
                                               i0_percent=0.039,
                                               shift_degree=0,
                                               tap_pos=1
                                               )

# pp.runpp(net, calculate_voltage_angles=True, init="dc")
pp.runpp(net, calculate_voltage_angles=True)


print('Done!')

# Bus = [bus_p_v(name="B01_FOZ_AREIA",
#               voltage_amplitude=1.03,
#               active_power=-1658 + 2405),
#       bus_p_v(name="B02_S.SANTIAGO",
#               voltage_amplitude=1.03,
#               active_power=-1332 + 692.3),
#       bus_p_v(name="B03_S.SEGREDO",
#               voltage_amplitude=1.029,
#               active_power=-1540 + 688.2),
#       bus_p_v(name="B04_ITAIPU",
#               voltage_amplitude=1.039,
#               active_power=-6500 + 62.6),
#       bus_p_q(name="B05_IVAIPORA",
#               active_power=845.8,
#               reactive_power=-9.2),
#       bus_p_q(name="B06_IVAIPORA",
#               active_power=-4.9,
#               reactive_power=79.8),
#       bus_v_theta(name="B07_EQUIVALENTE",
#                   voltage_amplitude=0.9660,
#                   voltage_angle=0)
#       ]
#
