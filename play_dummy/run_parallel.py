from utils import run_proteus_parallel
import numpy as np

n = 6

run_names = [f"BO/workers/worker{i}" for i in range(n)]

corefracs = np.random.uniform(0.3, 0.9, n)
params =[{"struct.corefrac": i.item()} for i in corefracs]

inputs = list(zip(params, run_names))

observables = None #["M_planet"] # could also set to None

# check whether this script is run or imported
# should only execute when run o/w recursive processes get spawned/ terminate early
if __name__ == '__main__':

    out = run_proteus_parallel(inputs, n, obs = observables)

    print(out)
