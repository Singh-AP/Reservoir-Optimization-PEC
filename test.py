
from platypus import *
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_lists():
    global reservoir_areas
    df = pd.read_excel('bargi.ods', engine='odf')
    # print(df.head())
    # print(df.columns)
    inflow = np.asarray(df["BARGI RESERVOIR INFLOW (VIRGIN FLOW)"])
    evap = np.asarray(df["BARGI EVAP. (mm)"])
    evap = evap*(reservoir_areas[0]/1000.0) # For uniform dimensions. We want all to be in MCM. evap is in mm, area in km^2, hence divide by 1000
    # overflow = np.zeros(evap.shape)
    # print(inflow.shape)
    # print(evap.shape)
    return inflow, evap


def expressions(vars):
    global N
    global Obj_num
    global Constraints_num
    global S_max
    Inflow, Evap = get_lists()
    objs=[]
    obj1=(-1.0*np.sum(np.asarray([(vars[i]-vars[(N//2+i)]) for i in range(N//2)]))) # Obj function: Max z = ∑ S_t − Q_t
    objs.append(obj1)
    constraints = [(vars[i+1]-vars[i]-Inflow[i]+ vars[(N//2+i)] + Evap[i] + max(0,vars[i+1]-S_max)) for i in range(N//2-1) ]
    # print("DEBUG: objs.shape: ", len(objs), " constraints.len: ", len(constraints))
    return objs, constraints

# # get_lists()
# class LoggingArchive(Archive):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, *kwargs)
#         self.log = []
    
#     def add(self, solution):
#         super().add(solution)
#         self.log.append(solution)
        
# log_archive = LoggingArchive()

N=int(324*2) # Decision Vars 324 S_t and 324 Q_t
Obj_num =1
S_min=409.0
S_max=425.7
Q_max =302.0
Constraints_num =int(N/2)-1  #S_t+1 = S_t + I_t − Q_t − E_t − O_t
reservoir_areas=[267.97]  # list of areas of reservoirs [BARGI: km^2]
constraint=[]

for i in range(N//2):
    constraint.append(Real(S_min, S_max)) # S_min <= S_t <=S_max
for i in range(N//2):
    constraint.append(Real(0.01, Q_max))

# algorithms = [NSGAII, (NSGAIII, {"divisions_outer":12})]

problem = Problem(N, Obj_num, Constraints_num)
problem.types[:] = constraint
problem.constraints[:] = "==0"
problem.function = expressions  
# problem.directions[:] = Problem.MAXIMIZE

# with ProcessPoolEvaluator(4) as evaluator:
#         results = experiment(algorithms, problems, nfe=10, evaluator=evaluator)

#         hyp = Hypervolume(minimum=[0, 0, 0], maximum=[1, 1, 1])
#         hyp_result = calculate(results, hyp, evaluator=evaluator)
#         display(hyp_result, ndigits=3)


algorithm = NSGAIII(problem,12)
algorithm.run(10)
feasible_solutions = [s for s in algorithm.result if s.feasible]
nondominated_solutions = nondominated(algorithm.result)

print("# Feasible solutions: ", len(feasible_solutions))
# print(feasible_solutions)
# print(nondominated_solutions)

result = algorithm.result

result_dict ={}
result_dict["obj"] = []
result_dict["storage"] =[]
result_dict["outflow"]=[]

counterr=0
for solution in result:
    print("Count: ", counterr)
    counterr+=1
    # print("Obj: ",solution.objectives)
    # print(solution.variables)
    result_dict["obj"].append(solution.objectives[0])
    res = np.asarray(solution.variables)
    result_dict["storage"].append(res[:N//2])
    result_dict["outflow"].append(res[N//2:])


with open("results_bargi_NSGAIII.dict", "wb") as f:
    pickle.dump(result_dict,f)

    # print(len(solution.variables))
    # res = np.asarray(solution.variables)
    # plt.plot(range(324),res[:N//2])
    # plt.xlabel("July 2008 Onwards")
    # plt.ylabel("Storage milliom m^3")
    # plt.title(str(counterr)+ ": Storage vs Month-period")
    # plt.savefig(str(counterr)+ ": Storage vs Month-period")
    # # plt.show()
    # plt.plot(range(324),res[N//2:])
    # plt.xlabel("July 2008 Onwards")
    # plt.ylabel("Outflow milliom m^3")
    # plt.title(str(counterr)+ ": Outflow vs Month-period")
    # plt.savefig(str(counterr)+ ": Outflow vs Month-period")
    # # plt.show()
    # counterr+=1


# algorithms = [NSGAII,
#                   (NSGAIII, {"divisions_outer":12}),
#                   (CMAES, {"epsilons":[0.05]}),
#                   GDE3,
#                   IBEA,
#                   (MOEAD, {"weight_generator":normal_boundary_weights, "divisions_outer":12}),
#                   (OMOPSO, {"epsilons":[0.05]}),
#                   SMPSO,
#                   SPEA2,
#                   (EpsMOEA, {"epsilons":[0.05]})]
    
#     # run the experiment using Python 3's concurrent futures for parallel evaluation
# with ProcessPoolEvaluator() as evaluator:
#     results = experiment(algorithms, problem, seeds=1, nfe=5, evaluator=evaluator)

# # display the results
# fig = plt.figure()

# for i, algorithm in enumerate(six.iterkeys(results)):
#     result = results[algorithm]["DTLZ2"][0]
    
#     ax = fig.add_subplot(2, 5, i+1, projection='3d')
#     ax.scatter([s.objectives[0] for s in result],
#                 [s.objectives[1] for s in result],
#                 [s.objectives[2] for s in result])
#     ax.set_title(algorithm)
#     ax.set_xlim([0, 1.1])
#     ax.set_ylim([0, 1.1])
#     ax.set_zlim([0, 1.1])
#     ax.view_init(elev=30.0, azim=15.0)
#     ax.locator_params(nbins=4)

# plt.show()
