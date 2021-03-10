#Load modules
import pandas as pd
from typing import Callable, List, Tuple
import numpy as np
from pandera.typing import DataFrame, Series
try:
    from scipy.interpolate import interp2d
except ImportError:
    interp2d = None
import xTHockey as xT

#Special thanks to socceraction as we adapted their open-source code as a base to work on hockey and as a blueprint
#The socceraction team is now cited and listed as co-author of this code
#Visit their module here:  https://github.com/ML-KULeuven/socceraction)

M: int = 8
N: int = 16
P: int = 1

def move_turnover_matrix(actions, l: int = N, w: int = M) -> np.ndarray:
    move_actions = xT.get_move_actions(actions)
    X = pd.DataFrame()
    X['start_cell'] = xT._get_flat_indexes(move_actions.start_x, move_actions.start_y, l, w)
    X['end_cell'] = xT._get_flat_indexes(move_actions.end_x, move_actions.end_y, l, w)
    X['success'] = move_actions.success

    vc = X.start_cell.value_counts(sort=False)
    start_counts = np.zeros(w * l)
    start_counts[vc.index] = vc

    turnover_matrix = np.zeros((w * l, w * l))

    for i in range(0, w * l):
        vc2 = X[((X.start_cell == i) & (X.success == 0.0))].end_cell.value_counts(
            sort=False)
        turnover_matrix[i, vc2.index] = vc2 / start_counts[i]

    return turnover_matrix

def estimate_nxT_matrix(actions, l: int = N, w: int = M, t: int = P,verbose = True) -> np.array:
    xTModel = xT.ExpectedThreat(l, w)
    xTModel.fit(actions,t)
    xt_matrix = xTModel.xT
    xt = xt_matrix.reshape(w*l)
    vec = np.zeros(w*l)
    vec2 = np.zeros(w*l)
    for i in range(0,w*l):
        if verbose:
            print('Transforming Matrix - Iteration ',str(i+1),'of ',str(w*l))
        uniform_data = np.flip(move_turnover_matrix(actions, l, w)[i])
        oxt = (xt * uniform_data).sum()
        nxt = xt[i] - oxt
        vec[i]=nxt
        vec2[i]=oxt
    oxt_matrix = vec2.reshape(w,l)
    nxt_matrix = vec.reshape(w,l)
    return nxt_matrix, oxt_matrix,xt_matrix

def get_grids(actions, l: int = N, w: int = M, t: int = P, verbose=False) -> np.array:
    nxt_matrix, oxt_matrix,xt_matrix = estimate_nxT_matrix(actions,l,w,t,verbose)
    actions = actions[actions['time']==t]
    startxc, startyc = xT._get_cell_indexes(actions.start_x, actions.start_y, l, w)
    nxT_start = nxt_matrix[w - 1 - startyc, startxc]
    oxT_start = oxt_matrix[w - 1 - startyc, startxc]
    xT_start = xt_matrix[w - 1 - startyc, startxc]
    return nxT_start,oxT_start, xT_start

def return_nxT_values(actions, l: int = N, w: int = M,verbose = False) -> np.array:
    lst = []
    for t in range(1,5):
        indx = actions[actions['time'] == t].index
        nxT, oxT, xT=get_grids(actions, l, w, t,verbose)
        frame = pd.DataFrame(
        data = {'indx':indx,'nxT':nxT,'oxT':oxT,'xT':xT}
        )
        lst.append(frame)
    df = pd.concat(lst)
    df.set_index('indx',drop=True,inplace=True)
    return df
