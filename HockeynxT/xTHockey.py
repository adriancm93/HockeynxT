#Load modules
import pandas as pd
from typing import Callable, List, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from pandera.typing import DataFrame, Series
try:
    from scipy.interpolate import interp2d
except ImportError:
    interp2d = None
import warnings
warnings.filterwarnings("ignore")

M: int = 8
N: int = 16
P: int = 1

def _get_cell_indexes(x: Series, y: Series, l: int = N, w: int = M) -> Tuple[Series, Series]:
    xmin = 0
    ymin = 0
    xi = (x - xmin) / 200 * l
    yj = (y - ymin) / 85 * w
    xi = xi.astype(int).clip(0, l - 1)
    yj = yj.astype(int).clip(0, w - 1)
    return xi, yj

def _get_flat_indexes(x: Series, y: Series, l: int = N, w: int = M) -> Series:
    xi, yj = _get_cell_indexes(x, y, l, w)
    return l * (w - 1 - yj) + xi

def _count(x: Series, y: Series, l: int = N, w: int = M) -> np.ndarray:
    x = x[~np.isnan(x) & ~np.isnan(y)]
    y = y[~np.isnan(x) & ~np.isnan(y)]
    flat_indexes = _get_flat_indexes(x, y, l, w)
    vc = flat_indexes.value_counts(sort=False)
    vector = np.zeros(w * l)
    vector[vc.index] = vc
    return vector.reshape((w, l))

def _safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)

def scoring_prob(actions, l: int = N, w: int = M) -> np.ndarray:
    actions=actions[actions['success'].isna() == False]
    actions=actions[actions['end_x'].isna() == False]
    shot_actions = actions[(actions.type_name == 'Shot')]
    goals = shot_actions[(shot_actions.success == 1.0)]

    shotmatrix = _count(shot_actions.start_x, shot_actions.start_y, l, w)
    goalmatrix = _count(goals.start_x, goals.start_y, l, w)
    return _safe_divide(goalmatrix, shotmatrix)

def scoring_prob_augmented(actions, l: int = N, w: int = M, t: int = P) -> np.ndarray:
    x_cell,y_cell = _get_cell_indexes(actions.start_x, actions.start_y, l=16, w=8)
    actions['x_start_cell'] = x_cell
    actions['y_start_cell'] = y_cell

    actions=actions[['x_start_cell','y_start_cell','time','type_name','success','league']]
    actions['d'] = ((15 - actions['x_start_cell'])**2 + (3.5-actions['y_start_cell'])**2)**(1/2)
    td = pd.get_dummies(actions['x_start_cell'],columns='x_cell',prefix='x_cell')
    actions = pd.concat([actions, td], axis=1)
    actions.loc[actions['x_start_cell'] <10,'x_cell_10'] = 1
    td = pd.get_dummies(actions['y_start_cell'],columns='y_cell',prefix='y_cell')
    actions = pd.concat([actions, td], axis=1)

    td = pd.get_dummies(actions['time'],columns='time',prefix='time')
    actions = pd.concat([actions, td], axis=1)
    actions.drop(columns=['time'],inplace=True)

    actions=actions[actions['league'] == 'scouting']
    actions=actions[actions['type_name'].isin(['Shot'])]
    y=actions['success']

    features = ['x_start_cell', 'd', 'x_cell_10', 'x_cell_11', 'x_cell_12', 'x_cell_13',
                'x_cell_14', 'x_cell_15', 'time_1', 'time_2', 'time_3', 'time_4']

    actions=actions[features]

    clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=100,random_state=42)
    clf.verbose=False
    clf.fit(actions,y)

    #create frame
    m_lst = []
    n_lst = []
    for n in np.arange(0,w,1)[::-1].tolist():
        for m in range(0,l):
            m_lst.append(m)
            n_lst.append(n)
    frame=pd.DataFrame(data={'x_start_cell':m_lst,'y_start_cell':n_lst})
    frame['d'] = ((15 - frame['x_start_cell'])**2 + (3.5-frame['y_start_cell'])**2)**(1/2)
    tdx = pd.get_dummies(frame['x_start_cell'],columns='x_cell',prefix='x_cell')
    frame = pd.concat([frame, tdx], axis=1)
    frame.loc[frame['x_start_cell'] <10,'x_cell_10'] = 1
    td = pd.get_dummies(frame['y_start_cell'],columns='y_cell',prefix='y_cell')
    frame = pd.concat([frame, td], axis=1)
    frame.drop(columns='y_start_cell',inplace=True)
    frame.drop(columns=
              ['x_cell_0', 'x_cell_1', 'x_cell_2', 'x_cell_3', 'x_cell_4',
               'x_cell_5', 'x_cell_6', 'x_cell_7', 'x_cell_8', 'x_cell_9',
               'y_cell_0','y_cell_1','y_cell_2','y_cell_3','y_cell_4','y_cell_5',
               'y_cell_6','y_cell_7']
              ,inplace=True)

    if t == 1:
        print('Time remaining in period, bin: ',t)
        frame['time_1'] = np.repeat(1,l*w)
        frame=frame.assign(time_2 = 0, time_3= 0,time_4=0)
    elif t == 2:
        print('Time remaining in period, bin: ',t)
        frame['time_2'] = np.repeat(1,l*w)
        frame=frame.assign(time_3 = 0, time_4= 0,time_1=0)
    elif t ==3:
        print('Time remaining in period, bin: ',t)
        frame['time_3'] = np.repeat(1,l*w)
        frame=frame.assign(time_4 = 0, time_1= 0,time_2=0)
    else:
        print('Time remaining in period, bin: ',t)
        frame['time_4'] = np.repeat(1,l*w)
        frame=frame.assign(time_1 = 0, time_2= 0,time_3=0)
    frame=frame[features]
    xg = clf.predict_proba(frame)[:, 1]
    xg.reshape(w,l)
    return xg.reshape(w,l)
    
def get_move_actions(actions):
    actions=actions[actions['success'].isna() == False]
    actions=actions[actions['end_x'].isna() == False]
    return actions[
        (actions.type_name == 'Zone Entry')
        | (actions.type_name == 'Pass')
        #| (actions.type_name == 'Faceoff')
    ]

def get_successful_move_actions(actions):
    actions=actions[actions['success'].isna() == False]
    actions=actions[actions['end_x'].isna() == False]
    move_actions = get_move_actions(actions)
    return move_actions[move_actions.success == 1.0]

def action_prob(
    actions, l: int = N, w: int = M
) -> Tuple[np.ndarray, np.ndarray]:
    actions=actions[actions['success'].isna() == False]
    actions=actions[actions['end_x'].isna() == False]
    move_actions = get_move_actions(actions)
    shot_actions = actions[(actions.type_name == 'Shot')]

    movematrix = _count(move_actions.start_x, move_actions.start_y, l, w)
    shotmatrix = _count(shot_actions.start_x, shot_actions.start_y, l, w)
    totalmatrix = movematrix + shotmatrix

    return _safe_divide(shotmatrix, totalmatrix), _safe_divide(movematrix, totalmatrix)

def move_transition_matrix(actions, l: int = N, w: int = M) -> np.ndarray:
    actions=actions[actions['success'].isna() == False]
    actions=actions[actions['end_x'].isna() == False]
    move_actions = get_move_actions(actions)

    X = pd.DataFrame()
    X['start_cell'] = _get_flat_indexes(move_actions.start_x, move_actions.start_y, l, w)
    X['end_cell'] = _get_flat_indexes(move_actions.end_x, move_actions.end_y, l, w)
    X['success'] = move_actions.success

    vc = X.start_cell.value_counts(sort=False)
    start_counts = np.zeros(w * l)
    start_counts[vc.index] = vc

    transition_matrix = np.zeros((w * l, w * l))

    for i in range(0, w * l):
        vc2 = X[((X.start_cell == i) & (X.success == 1.0))].end_cell.value_counts(
            sort=False
        )
        transition_matrix[i, vc2.index] = vc2 / start_counts[i]

    return transition_matrix

class ExpectedThreat:
    def __init__(self, l: int = N, w: int = M, eps: float = 1e-5):
        self.l = l
        self.w = w
        self.eps = eps
        self.heatmaps: List[np.ndarray] = []
        self.xT: np.ndarray = np.zeros((w, l))
        self.scoring_prob_matrix: np.ndarray = np.zeros((w, l))
        self.shot_prob_matrix: np.ndarray = np.zeros((w, l))
        self.move_prob_matrix: np.ndarray = np.zeros((w, l))
        self.transition_matrix: np.ndarray = np.zeros((w * l, w * l))
    def __solve(
        self,
        p_scoring: np.ndarray,
        p_shot: np.ndarray,
        p_move: np.ndarray,
        transition_matrix: np.ndarray,
    ) -> None:
        gs = p_scoring * p_shot
        diff = 1
        it = 0
        self.heatmaps.append(self.xT.copy())

        while np.any(diff > self.eps):
            total_payoff = np.zeros((self.w, self.l))

            for y in range(0, self.w):
                for x in range(0, self.l):
                    for q in range(0, self.w):
                        for z in range(0, self.l):
                            total_payoff[y, x] += (
                                transition_matrix[self.l * y + x, self.l * q + z] * self.xT[q, z]
                            )

            newxT = gs + (p_move * total_payoff)
            diff = newxT - self.xT
            self.xT = newxT
            self.heatmaps.append(self.xT.copy())
            it += 1

        print('# iterations: ', it)

    def fit(self, actions,t) -> 'ExpectedThreat':
        self.scoring_prob_matrix = scoring_prob_augmented(actions,self.l,self.w,t)
        self.shot_prob_matrix, self.move_prob_matrix = action_prob(actions, self.l, self.w)
        self.transition_matrix = move_transition_matrix(actions, self.l, self.w)
        self.__solve(
            self.scoring_prob_matrix,
            self.shot_prob_matrix,
            self.move_prob_matrix,
            self.transition_matrix,
        )
        return self
    
    def get_xT(self, actions) -> np.array:
        l = self.l
        w = self.w
        grid = self.xT
        startxc, startyc = _get_cell_indexes(actions.start_x, actions.start_y, l, w)
        endxc, endyc = _get_cell_indexes(actions.end_x, actions.end_y, l, w)
        xT_start = grid[w - 1 - startyc, startxc]
        xT_end = grid[w - 1 - endyc, endxc]
        return xT_start , xT_end
    
def return_xT_values(actions, l: int = N, w: int = M, t: int = P) -> np.ndarray:
    x = actions.start_x
    y = actions.start_y
    lst = []
    og = actions.copy()
    actions=actions[actions['end_x'].notna()]
    xTModel = ExpectedThreat(l, w)
    for t in range(1,5):    
        data_= actions[actions['time'] == t].copy()
        x = data_.start_x
        y = data_.start_y
        indx = data_.index
        xTModel.fit(actions,t)
        start,end = xTModel.get_xT(data_)
        frame = pd.DataFrame(
            data={'xT':start,'index':indx})
        lst.append(frame)               
    df = pd.concat(lst)
    og.reset_index(inplace=True)
    og=og.merge(df,on='index',how='left')
    og.set_index('index',inplace=True)
    return og
