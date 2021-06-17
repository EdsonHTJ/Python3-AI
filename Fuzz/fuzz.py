import numpy as np 
import skfuzzy as fuzz
from skfuzzy import control

temperatura = control.Antecedent(np.arange(0,  51, 1 ), 'temperatura')
umidade     = control.Antecedent(np.arange(0, 101, 1 ), 'umidade')

variacao    = control.Consequent(np.arange(-15,16,1), 'variacao')

temperatura['muito_frio']   = fuzz.trimf(temperatura.universe,  [0,  0,  8])
temperatura['frio']         = fuzz.trapmf(temperatura.universe, [5, 7,  13, 17])
temperatura['ok']           = fuzz.trapmf(temperatura.universe, [12, 17,23, 27])
temperatura['quente']       = fuzz.trapmf(temperatura.universe, [23, 27 ,38, 42])
temperatura['muito_quente'] = fuzz.trimf(temperatura.universe,  [30,50, 50])

umidade['seco']  = fuzz.trimf(umidade.universe, [0,   0,  50])
umidade['bom']   = fuzz.trimf(umidade.universe, [0,  50, 100])
umidade['umido'] = fuzz.trimf(umidade.universe, [50,100, 100])

variacao['esfriar']   = fuzz.trapmf(variacao.universe, [-15, -15, -10 ,  0])
variacao['manter']    = fuzz.trimf(variacao.universe,  [-5,   0,   5])
variacao['esquentar'] = fuzz.trapmf(variacao.universe, [  0, 10, 15, 15])

temperatura.view()
umidade.view()
variacao.view()

rule1 = control.Rule((temperatura['muito_frio']),  variacao['esquentar'])
rule2 = control.Rule((temperatura['frio']   & ~(umidade['umido'])),  variacao['esquentar'])
rule3 = control.Rule((temperatura['ok'])    | umidade['bom']   , variacao['manter'])
rule4 = control.Rule((temperatura['quente'] & ~(umidade['seco'])), variacao['esfriar'])
rule5 = control.Rule((temperatura['muito_quente']),  variacao['esfriar'])

var_control = control.ControlSystem([rule1, rule2, rule3, rule4, rule5])
var         = control.ControlSystemSimulation(var_control)

var.input['temperatura'] = 20
var.input['umidade'] = 20

var.compute()

print(f"\n A Variação de temperatura sera de  {var.output['variacao']} \n")

variacao.view(sim=var)

a = input()

