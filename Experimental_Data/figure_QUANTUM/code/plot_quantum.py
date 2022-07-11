import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
plt.ion()
file_path_list = os.path.dirname(os.path.abspath(__file__)).split('\\')
root_path = '\\'.join(file_path_list)
QAML_SAVEPATH = '\\'.join(file_path_list[:-1])+'\\encode_quantum\\'

def name2idx(variables, name, default=np.nan):
        name = name if isinstance(name, list) else [name]
        names = [variable_i['name'] for variable_i in variables]
        for _name in name:
            if _name in names:
                return names.index(_name)
        else:
            return default

class dataLab:
    def __init__(self):
        pass

    def pickle_load(self, save_path):
        with open(save_path, 'rb') as f:
            para = pickle.load(f)
        self.inds = para['inds']
        self.deps = para['deps']
        self.dim = para['dim']
        self.data = para['data']
        self.parameters = para['parameters']
        self.session = para['session']
        self.dataset_num = para['dataset_num']
        self.datasets_num = para['datasets_num']

    def __getDependentColumnIndex(self, dependentIndex):
        columnList = np.arange(0, len(self.deps)) + len(self.inds)
        columnIndex = columnList[dependentIndex]
        return columnIndex

    def __columns2Matrix(self,
                         independentColumns=[0, 1],
                         dependentColumn=-1,
                         default=np.NaN,
                         steps=None):
        """
        Converts data in columns format into a 2D array.   
        No special order of the datapoints is required.
        The spacing between pixels is the median of nonzero changes
        of independent variables in neigboring lines.
        """

        dims = np.size(independentColumns)
        mins = np.min(self.data[:, independentColumns], axis=0)
        maxs = np.max(self.data[:, independentColumns], axis=0)
        if steps is None:
            steps = np.ones(dims, dtype=float)
            for i in np.arange(dims):
                colSteps = np.diff(np.sort(self.data[:,
                                                     independentColumns[i]]))
                colSteps = colSteps[np.argwhere(colSteps > 1e-8 *
                                                (maxs[i] - mins[i]))]
                if len(colSteps):
                    steps[i] = np.min(abs(colSteps))
        sizes = (np.round((maxs - mins) / steps)).astype(int) + 1
        indices = tuple([
            np.round((self.data[:, independentColumns[i]] - mins[i]) /
                     steps[i]).astype(int) for i in np.arange(dims)
        ])
        if np.iterable(dependentColumn):
            mat = np.resize(default, sizes.tolist() + [len(dependentColumn)])
        else:
            mat = np.resize(default, sizes)
        #Create the 2D image array
        mat[indices] = self.data[:, dependentColumn]
        Vmin = np.min(mat[indices], axis=0)
        Vmax = np.max(mat[indices], axis=0)
        info = {
            'Dmin': Vmin,
            'Dmax': Vmax,
            'Imin': mins,
            'Imax': maxs,
            'Isteps': steps,
            'indices': indices
        }
        if dims == 2:
            info['Xmin'] = mins[0]
            info['Xmax'] = maxs[0]
            info['Xstep'] = steps[0]
            info['Ymin'] = mins[1]
            info['Ymax'] = maxs[1]
            info['Ystep'] = steps[1]
            info['Xindices'] = np.unique(indices[0])
            info['Yindices'] = np.unique(indices[1])
        return mat, info

    def toMatrix(self, dependent=-1, default=np.NaN, steps=None):
        dependent = np.asarray(dependent)
        dependentColumnIndex = self.__getDependentColumnIndex(dependent)
        independentColumns = np.arange(len(self.inds))
        mat, info = self.__columns2Matrix(independentColumns,
                                          dependentColumnIndex,
                                          default=default,
                                          steps=steps)
        if len(independentColumns) == 2:
            info['Xname'] = self.inds[independentColumns[0]]['name']
            info['Xunits'] = self.inds[independentColumns[0]]['units']
            info['Yname'] = self.inds[independentColumns[1]]['name']
            info['Yunits'] = self.inds[independentColumns[1]]['units']
        info['Inames'] = [self.inds[i]['name'] for i in independentColumns]
        info['Iunits'] = [self.inds[i]['units'] for i in independentColumns]
        if np.iterable(dependent):
            info['Dname'] = [self.deps[d]['name'] for d in dependent]
            info['Dunits'] = [self.deps[d]['units'] for d in dependent]
        else:
            info['Dname'] = self.deps[dependent]['name']
            info['Dunits'] = self.deps[dependent]['units']
        self.mat = np.array(mat, dtype='float64')
        self.info = info
        self.matIdx = dependent
        self.xs = np.arange(info['Xmin'], info['Xmax'] + info['Xstep'] / 10.0,
                            info['Xstep'])
        self.ys = np.arange(info['Ymin'], info['Ymax'] + info['Ystep'] / 10.0,
                            info['Ystep'])

    def get_data(self,
                 data_name=None,
                 ind_index=None,
                 dep_index=None,
                 default=None,
                 steps=None,
                 returnXY=True,
                 del_null_col_row=True):
        '''
        Description:
        interface for get data
        if match failed, return default
        ---------------------------
        Note:
        get data according to 1.data_name, 2.ind_index, 3.dep_index
        [bool] del_null_col_row
        in some situation (fit function), you may want to delete columns and rows which have no data actually
        (filled with default), then you can set del_null_col_row = True
        '''
        # among data_name, ind_index, dep_index only one variable allow to be not None
        _s1 = data_name is None
        _s2 = ind_index is None
        _s3 = dep_index is None
        assert ((not _s1) and _s2 and _s3) or (_s1 and (not _s2)
                                               and _s3) or (_s1 and _s2 and
                                                            (not _s3))
        ind_index = name2idx(self.inds,
                             data_name) if ind_index is None else ind_index
        dep_index = name2idx(self.deps,
                             data_name) if dep_index is None else dep_index
        if ind_index is not np.nan:
            return np.copy(self.data[:, ind_index])
        elif dep_index is not np.nan:
            if self.dim == 1:
                return np.copy(
                    self.data[:, self.__getDependentColumnIndex(dep_index)])
            else:
                self.toMatrix(dep_index, default=default, steps=steps)
                xs = np.copy(self.xs)
                ys = np.copy(self.ys)
                mat = np.copy(self.mat)
                if del_null_col_row:
                    xs = xs[self.info['Xindices']]
                    ys = ys[self.info['Yindices']]
                    mat = mat[self.info['Xindices']][:, self.info['Yindices']]
                if returnXY:
                    return xs, ys, mat
                else:
                    return mat
        else:
            return None

    def get_measure_info(self):
        config = self.parameters['config']
        para = {
            'measure': 'name',
            'measure_xy': 'qxy_name',
            'measure_coupler': 'c_name',
            'measure_rr': 'qrr_name',
            'measure_read': 'qread_name'
        }
        measure_info = {}
        name_info = {}
        for key, value in para.items():
            try:
                measure = self.parameters[key]
                name = [config[i] for i in measure]
                measure_info[key] = np.array(measure)
                name_info[value] = np.array(name)
            except:
                pass
        if 'measure' in measure_info.keys():
            measure = measure_info['measure']
            name = name_info['name']
            idxs = np.array(
                [i for i, _name in enumerate(name) if _name.startswith('q')])
            measure_info['measure_qubit'] = measure[idxs]
            name_info['q_name'] = name[idxs]

        return measure_info, name_info
class pickle_depository:
    def __init__(self, save_path):
        self.save_path = save_path

    def _load(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, 'rb') as f:
                data = pickle.load(f)
        else:
            data = {}
        return data

    def _save(self, data):
        with open(self.save_path, 'wb') as f:
            pickle.dump(data, f)

    def _clear(self):
        data = {}
        self._save(data)


def readoutCorrection(probs, measureFids):
    '''
    Readout correction without using matrix product.
    ----------------
    Author: Qiujiang Guo
    Date: 2021/08/07
    ----------------

    Definitions
    -----------
    probs: [x, 2**qNum].
    measureFids:[[measureF0, measureF1], .....].

    returns
    -----------
    return the probability after readout correction.
    '''
    qNum = len(measureFids)
    assert 2**qNum == probs.shape[-1]

    for ii in range(qNum):
        f0 = measureFids[ii][0]
        f1 = measureFids[ii][1]
        corrMat = np.matrix([[f0, 1 - f1], [1 - f0, f1]])
        corrMat = corrMat.I
        indexP0, indexP1 = findIndex(qNum, ii + 1)
        probsCorr = np.copy(probs)
        probsCorr[..., indexP0] = probs[..., indexP0] * corrMat[0, 0] + probs[
            ..., indexP1] * corrMat[0, 1]
        probsCorr[..., indexP1] = probs[..., indexP0] * corrMat[1, 0] + probs[
            ..., indexP1] * corrMat[1, 1]
        probs = probsCorr
    return probs

def findIndex(qNum, qidx):
    '''
    Look for all the indexs for qidx in a state vector.
    ----------------
    Author: qguo
    Date: 2021/08/07
    (This code is wrotten by Kemin Li at the beginning and revised by qguo)
    ----------------

    Definitions
    -----------
    qNum: (int), qubitNuber
    qidx: [1, 2 .... qNum], define which qubit's index you want to find.

    returns
    -----------
    index0: index related for P0 for qidx.
    index1: index related for P1 for qidx.
    '''
    order = 2**(qNum - qidx + 1) * np.arange(0, 2**(qidx - 1))
    index0 = np.reshape(order, [len(order), 1]) + np.arange(
        0, 2**(qNum - qidx))
    index0 = np.reshape(index0, 2**(qNum - 1))
    # index0 = order + np.arange(0, 2**(qNum - qidx))
    index1 = index0 + 2**(qNum - qidx)
    return np.int32(index0), np.int32(index1)

class QAML_pkl(pickle_depository):
    def __init__(self, file_path=None, file_name=None):
        self.file_path = QAML_SAVEPATH if file_path is None else file_path
        self.file_name = 'QAML_exp.pkl' if file_name is None else file_name
        super().__init__(self.file_path + self.file_name)

def get_labels(file_encode):
    encode_params = QAML_pkl(file_name=file_encode)._load()
    labels_train = []
    labels_test = []
    for batch_idx, batch_value in encode_params['y_train'].items():
        labels_train.append(batch_value)
    for batch_idx, batch_value in encode_params['y_test'].items():
        labels_test.append(batch_value)

    labels_train = np.array(labels_train).T
    labels_test = np.array(labels_test).T
    return labels_train[1, :], labels_test[1, :]

def analysis_loss(dataset_train, dataset_test, session, save_path, 
                  file_encode='encode_params_quantum_data.pkl',
                  discriminate_line=0.45,
                  do_plot=True,
                  collect=False):
    '''
        load labels from the following two files used in the experiment: 
        1. encode_params_10q_256.pkl
        2. encode_params_adv00_06_10q_256.pkl
    '''
    labels_train, labels_test = get_labels(file_encode)
    batch_info = {}
    batch_info['full'] = {}
    if 'adv' in file_encode:
        batch_info['origin'] = {}
        batch_info['adv'] = {}

    data_info = {
        'train': {
            'labels': labels_train,
            'dataset': dataset_train
        },
        'test': {
            'labels': labels_test,
            'dataset': dataset_test
        }
    }
    ana_info = {}
    for type, dinfo in data_info.items():
        dataset = dinfo['dataset']
        labels = dinfo['labels']
        batch_info['full'][type] = np.arange(len(labels))
        if 'adv' in file_encode:
            batch_info['origin'][type] = np.arange(len(labels) // 2)
            batch_info['adv'][type] = np.arange(len(labels) // 2, len(labels))
        data = dataLab()
        data.pickle_load(save_path + '_'+session+'_'+str(dataset) + '.pkl')
        probs = data.get_data('P1_corr')
        batch_idx = data.get_data('batch_idx')

        for ana_type, binfo in batch_info.items():
            probs_temp = []
            labels_temp = []
            batch_idx_temp = []
            for idx, bi in enumerate(batch_idx):
                if bi in binfo[type]:
                    labels_temp.append(labels[int(bi)])
                    probs_temp.append(probs[idx])
                    batch_idx_temp.append(bi)
            labels_temp = np.array(labels_temp)
            probs_temp = np.array(probs_temp)
            batch_idx_temp = np.array(batch_idx_temp)
            idxs = [np.argwhere(labels_temp == i).flatten() for i in [0, 1]]
            accuracy_list = np.hstack([
                probs_temp[idxs[0]] < discriminate_line,
                probs_temp[idxs[1]] > discriminate_line
            ])
            loss_list = np.hstack([
                -np.log(1 - probs_temp[idxs[0]]), -np.log(probs_temp[idxs[1]])
            ])

            res = {type: {}}
            for digit in [0, 1]:
                res[type][str(digit)] = {}
                res[type][str(digit)]['probs'] = probs_temp[idxs[digit]]
                res[type][str(digit)]['batch_idx'] = batch_idx_temp[
                    idxs[digit]]
            res[type]['accuracy'] = np.mean(accuracy_list)
            res[type]['loss'] = np.mean(loss_list)
            res[type]['loss_std'] = np.std(loss_list)/np.sqrt(len(loss_list)-1)
            if ana_type in ana_info:
                ana_info[ana_type].update(res)
            else:
                ana_info[ana_type] = res
    return ana_info


def fig3c(file_encode='encode_params_quantum_data.pkl', session='Chao_N36R18_20220301_qaml'):

    datasets_train = [1724, 1731, 1737, 1743, 1749, 1755, 1761, 1767, 1773, 1779, 1785, 1791, 1797, 1803, 1809, 1815, 1821, 1827, 1833, 1839, 1845, 1851, 1857, 1863, 1869, 1875, 1881, 1887, 1893, 1899, 1905, 1911, 1917, 1923, 1929, 1935, 1941, 1947, 1953, 1959, 1965, 1971, 1977, 1983, 1989, 1995, 2001, 2007, 2013, 2019, 2025, 2031, 2037, 2043, 2049, 2055, 2061, 2067, 2073, 2079, 2085, 2091, 2097, 2103, 2109, 2115, 2121, 2127, 2133, 2139, 2145, 2151, 2157, 2163, 2169, 2367, 2374, 2380, 2386, 2392, 2398, 2404, 2410, 2416, 2422, 2428, 2434, 2440, 2446, 2452, 2458, 2464, 2470, 2476, 2482, 2488, 2494, 2500, 2506, 2512, 2518, 2524, 2530, 2536, 2542, 2548, 2554, 2560, 2566, 2572, 2578, 2584, 2590, 2596, 2602, 2608, 2614, 2620, 2626, 2632, 2638, 2644, 2650, 2656, 2662, 2668, 2674, 2680, 2686, 2692, 2698, 2704, 2710, 2716, 2722, 2728, 2734, 2740, 2746, 2752, 2758, 2764, 2770, 2776, 2782, 2788, 2794, 2800, 2806, 2812, 2818, 2824, 2830, 2836, 2842, 2848, 2854, 2860, 2866, 2872, 2878, 2884, 2890, 2896, 2902, 2908, 2914, 2920, 2926, 2932, 2938, 2944, 2950, 2956, 2962, 2968, 2974, 2980, 2986, 2992, 2998, 3004, 3010, 3016, 3022, 3028, 3034, 3040, 3046, 3052, 3058, 3064, 3070, 3076, 3082, 3088, 3094, 3100, 3106, 3112, 3118, 3124, 3130, 3136, 3142, 3148, 3154, 3160, 3166, 3172, 3178, 3184, 3190, 3196, 3202, 3208, 3214, 3220, 3226, 3232, 3238, 3244, 3250, 3256, 3262, 3268, 3274, 3280, 3286, 3292, 3298, 3304, 3310, 3316, 3322, 3328, 3334, 3340, 3346, 3352, 3358, 3364, 3370, 3376, 3382, 3388, 3394, 3400, 3406, 3412, 3418, 3424, 3430, 3436, 3442, 3448, 3454, 3460, 3466, 3472, 3478, 3484, 3490, 3496, 3502, 3508, 3514, 3520, 3526, 3532, 3538, 3544, 3550, 3556, 3562, 3568, 3574, 3580, 3586, 3592, 3598, 3604, 3610, 3616, 3622, 3628, 3634, 3640, 3646, 3652, 3658, 3664, 3670, 3676, 3682, 3688, 3694, 3700, 3706, 3712, 3718, 3724, 3730, 3736, 3742, 3748, 3754, 3760, 3766, 3772, 3778, 3784, 3790, 3796, 3802, 3808, 3814, 3820, 3826, 3832, 3838, 3844, 3850, 3856, 3862, 3868, 3874, 3880, 3886, 3892, 3898, 3904, 3910, 3916, 3922, 3928, 3934, 3940, 3946, 3952, 3958, 3964, 3970, 3976, 3982, 3988, 3994, 4000, 4006, 4012, 4018]

    datasets_test = [1725, 1732, 1738, 1744, 1750, 1756, 1762, 1768, 1774, 1780, 1786, 1792, 1798, 1804, 1810, 1816, 1822, 
    1828, 1834, 1840, 1846, 1852, 1858, 1864, 1870, 1876, 1882, 1888, 1894, 1900, 1906, 1912, 1918, 1924, 1930, 1936, 1942, 1948, 1954, 1960, 1966, 1972, 1978, 1984, 1990, 1996, 2002, 2008, 2014, 2020, 2026, 2032, 2038, 2044, 2050, 2056, 2062, 2068, 2074, 2080, 2086, 2092, 2098, 2104, 2110, 2116, 2122, 2128, 2134, 2140, 2146, 2152, 2158, 2164, 2170, 2368, 2375, 2381, 2387, 2393, 2399, 2405, 2411, 2417, 2423, 2429, 2435, 2441, 2447, 2453, 2459, 2465, 2471, 2477, 2483, 2489, 2495, 2501, 2507, 2513, 2519, 2525, 2531, 2537, 2543, 2549, 2555, 2561, 2567, 2573, 2579, 2585, 2591, 2597, 2603, 2609, 2615, 2621, 2627, 2633, 2639, 2645, 2651, 2657, 2663, 2669, 2675, 2681, 2687, 2693, 2699, 2705, 2711, 2717, 2723, 2729, 2735, 2741, 2747, 2753, 2759, 2765, 2771, 2777, 2783, 2789, 2795, 2801, 2807, 2813, 2819, 2825, 2831, 2837, 2843, 2849, 2855, 2861, 2867, 2873, 2879, 2885, 2891, 2897, 2903, 2909, 2915, 2921, 2927, 2933, 2939, 2945, 2951, 2957, 2963, 2969, 2975, 2981, 2987, 2993, 2999, 3005, 3011, 3017, 3023, 3029, 3035, 3041, 3047, 3053, 3059, 3065, 3071, 3077, 3083, 3089, 3095, 3101, 3107, 3113, 3119, 3125, 3131, 3137, 3143, 3149, 3155, 3161, 3167, 3173, 3179, 3185, 3191, 3197, 3203, 3209, 3215, 3221, 3227, 3233, 3239, 3245, 3251, 3257, 3263, 3269, 3275, 3281, 3287, 3293, 3299, 3305, 3311, 3317, 3323, 3329, 3335, 3341, 3347, 3353, 3359, 3365, 3371, 3377, 3383, 3389, 3395, 3401, 3407, 3413, 3419, 3425, 3431, 3437, 3443, 3449, 3455, 3461, 3467, 3473, 3479, 3485, 3491, 3497, 3503, 3509, 3515, 3521, 3527, 3533, 3539, 3545, 3551, 3557, 3563, 3569, 3575, 3581, 3587, 3593, 3599, 3605, 3611, 3617, 3623, 3629, 3635, 3641, 3647, 3653, 3659, 3665, 3671, 3677, 3683, 3689, 3695, 3701, 3707, 3713, 3719, 3725, 3731, 3737, 3743, 3749, 3755, 3761, 3767, 3773, 3779, 3785, 3791, 3797, 3803, 3809, 3815, 3821, 3827, 3833, 3839, 3845, 3851, 3857, 3863, 3869, 3875, 3881, 3887, 3893, 3899, 3905, 3911, 3917, 3923, 3929, 3935, 3941, 3947, 3953, 3959, 3965, 3971, 3977, 3983, 3989, 3995, 4001, 4007, 4013, 4019]

    save_path = '\\'.join(file_path_list[:-1])+'\\dataset_train\\'

    result = {}
    for dataset_train, dataset_test in zip(datasets_train, datasets_test):
        data_train = dataLab()
        data_train.pickle_load(save_path + '_'+session+'_'+str(dataset_train) + '.pkl')
        data_test = dataLab()
        data_test.pickle_load(save_path + '_'+session+'_'+str(dataset_test) + '.pkl')
        iter_idx = data_train.parameters['iter_idx']
        row_idx = data_train.parameters['row_idx']
        para = analysis_loss(dataset_train, dataset_test,session, save_path,
                             file_encode=file_encode,
                             do_plot=False,
                             discriminate_line=0.45,
                             collect=True)
        result.update({
        (iter_idx, row_idx): {
            'loss_train': para['full']['train']['loss'],
            'loss_train_std': para['full']['train']['loss_std'],
            'accuracy_train': para['full']['train']['accuracy'],
            'loss_test': para['full']['test']['loss'],
            'loss_test_std': para['full']['test']['loss_std'],
            'accuracy_test': para['full']['test']['accuracy']
        }
    })

    plot_train_dynamics_quantum(result, ((29.0, 0.0)))

def plot_train_dynamics_quantum(result, iter_idx_pair_cut=(29.0, 0.0)):
    iter_idx_pair = list(result.keys())
    if iter_idx_pair_cut is not None:
        cut_idx = iter_idx_pair.index(iter_idx_pair_cut) + 1
        iter_idx_pair = iter_idx_pair[:cut_idx]
    iter_idx = np.arange(len(iter_idx_pair))
    loss_train = [result[pair]['loss_train'] for pair in iter_idx_pair]
    loss_train_std = [result[pair]['loss_train_std'] for pair in iter_idx_pair]
    accuracy_train = [
        result[pair]['accuracy_train'] for pair in iter_idx_pair
    ]
    loss_test = [result[pair]['loss_test'] for pair in iter_idx_pair]
    loss_test_std = [result[pair]['loss_test_std'] for pair in iter_idx_pair]
    accuracy_test = [
        result[pair]['accuracy_test'] for pair in iter_idx_pair
    ]

    fig = plt.figure(figsize=[3.75, 2.4])
    markersize = 6
    fontsize = 10
    labelsize = 9
    legendsize = 9
    alpha = 1.0
    plt.subplot(211)
    plt.errorbar(iter_idx[::10],
             loss_train[::10],
             yerr=loss_train_std[::10],
             marker='o',
             markersize=markersize,
             ls='-',
             label='Training',
             color='C2',
             markerfacecolor='none',
             linewidth=1.0,
             capsize=4,
             alpha=alpha)
    plt.errorbar(iter_idx[::10],
             loss_test[::10],
             yerr=loss_test_std[::10],
             marker='^',
             markersize=markersize,
             ls='-',
             label='Test',
             color='C3',
             markerfacecolor='none',
             linewidth=1.0,
             capsize=4,
             alpha=alpha)
    plt.tick_params(labelsize=labelsize)
    plt.ylim(0.5, 0.72)

    plt.subplot(212)
    plt.plot(iter_idx[::10],
             accuracy_train[::10],
             marker='o',
             markersize=markersize,
             ls='-',
             color='C2',
             markerfacecolor='none',
             linewidth=1.0,
             alpha=alpha)
    plt.plot(iter_idx[::10],
             accuracy_test[::10],
             marker='^',
             markersize=markersize,
             ls='-',
             color='C3',
             markerfacecolor='none',
             linewidth=1.0,
             alpha=alpha)
    plt.tick_params(labelsize=labelsize)
    plt.ylim(0.35, 1.05)
    ax_loss, ax_acc = fig.axes
    minor_xticks = []
    major_xticks = []
    minor_xticklabels = []
    major_xticklabels = []
    iter_idx = np.arange(len(iter_idx_pair)+10)
    for i in iter_idx:
        if i % 50 == 0:
            major_xticks.append(int(i))
            major_xticklabels.append(int(i / 10))
        else:
            minor_xticks.append(i)
            minor_xticklabels.append(i % 10)
    ax_loss.set_xticks(major_xticks, minor=False)
    ax_loss.set_xticklabels([], minor=False, fontsize=fontsize)
    ax_loss.tick_params(axis='x',
                        which='major',
                        direction='out',
                        bottom=True,
                        labelbottom=True,
                        top=False,
                        labeltop=False)
    ax_loss.tick_params(axis='x',
                        which='minor',
                        direction='in',
                        bottom=False,
                        labelbottom=False,
                        top=True,
                        labeltop=True)
    ax_loss.set_ylabel('Loss', fontsize=fontsize, fontproperties='Times New Roman')
    ax_loss.set_yticks([0.5, 0.6, 0.7])
    font = {'family':'Times New Roman', 'size': 9}
    ax_loss.legend(prop=font, frameon=False)
    # ax_loss.legend(fontsize=legendsize, frameon=False)
    ax_acc.set_xticks(major_xticks, minor=False)
    ax_acc.set_xticklabels(major_xticklabels, minor=False, fontsize=fontsize)
    ax_acc.tick_params(axis='x',
                       which='major',
                       direction='out',
                       bottom=True,
                       labelbottom=True,
                       top=False,
                       labeltop=False)
    ax_acc.tick_params(axis='x',
                       which='minor',
                       direction='in',
                       bottom=False,
                       labelbottom=False,
                       top=True,
                       labeltop=True)
    ax_acc.set_xlabel('Epochs', fontsize=fontsize, fontproperties='Times New Roman')
    ax_acc.set_ylabel('Accuracy', fontsize=fontsize, fontproperties='Times New Roman')
    ax_acc.set_yticks([0.4, 0.7, 1.0])
    ax_acc.legend(fontsize=legendsize, frameon=False)
    plt.tick_params(labelsize=labelsize)
    plt.subplots_adjust(left=0.14, top=0.95, right=0.97, bottom=0.18, hspace=0.12, wspace=0.0)

#---------------fig 3a-------
def plot_seq():

    f10 = np.arange(4.3, 4.1, -0.01)
    # f10s = np.array([4.26, 4.39, 4.545, 4.69, 4.38, 4.28, 4.12, 4.4, 4.25, 3.99])
    f10s = np.linspace(4.69, 3.99, 10)
    f_int = 4.24
    alpha = (np.sqrt(5) - 1)/2
    qidxs = np.arange(1,11)
    amp = 0.08
    plt.figure(figsize=(4.0, 2.5))
    ax = plt.gca()
    def potential(xs):
        return f_int + amp*np.cos(2*np.pi*alpha*xs)
    
    ks = np.arange(0.5, 10.5, 0.01)
    t = np.linspace(0, 120, 1000)

    for ii in qidxs:
        q_z = env.flattop(20, 80, 1, amp=potential(ii)-f10s[ii-1], overshoot=0.0, overshoot_w=0.0)
        ax.plot(t, q_z(t)+f10s[ii-1], alpha=0.5)
    ax.set_axis_off()

def gaussians(mus=range(10), dips=np.cos(2*np.pi*np.arange(1, 11)*(5**0.5-1)/2)+np.ones(10)*3, sigma=0.07):
    xs = np.arange(-2,12,0.01)
    res = 0
    for idx, dip in enumerate(dips):
        res += -dip*np.exp(-(xs-mus[idx])**2/sigma)
    return xs, res

def plot_potential(dips=-np.cos(2*np.pi*np.arange(1, 11)*(5**0.5-1)/2)+np.ones(10)*3, sigma=0.07):
    xs, ys = gaussians(dips=dips, sigma=sigma)
    plt.figure(figsize=(2.3, 0.7))
    plt.plot(xs, ys, 'k-', linewidth=1.0)
    plt.xlim(-0.3,  9.3)
    plt.axis('off')

def plot_VO():

    f10 = np.arange(4.3, 4.1, -0.01)
    f10s = np.array([4.26, 4.39, 4.545, 4.69, 4.38, 4.28, 4.12, 4.4, 4.25, 3.99])
    f_int = 4.2
    alpha = (np.sqrt(5) - 1)/2
    qidxs = np.arange(1,11)
    amp = 0.02
    plt.figure(figsize=(2.48, 0.35))
    ax = plt.gca()
    def potential(xs):
        return f_int + np.cos(2*np.pi*alpha*xs)
    ks = np.arange(0.5, 16, 0.001)
    t = np.linspace(0, 50, 200)

    ax.plot(qidxs, potential(qidxs), color='crimson', marker='o', ls='', alpha=0.8, markersize=4)
    plt.ylim(2.8, 5.5)
    ax.set_axis_off()

#--------fig 3 b--------------
def fig3b():
    dl_phase_transition = dataLab()
    dl_phase_transition.pickle_load('\\'.join(file_path_list[:-1])+'\\dataset_quantum\\'+'_Chao_N36R18_20220301_qaml_10918.pkl')
    dl = dl_phase_transition
    measure_info, name_info = dl.get_measure_info()
    qNames = name_info['qread_name']
    stateQ = dl.parameters['stateQ']
    reps, Vs, probs = dl.get_data(dep_index=list(range(20)))
    results = {}
    for idx, rep in enumerate(reps):
        results[rep] = {}
        ps = probs[idx, :, :]
        p1s_cal = np.zeros([len(Vs), len(qNames)])
        for jdx, qName in enumerate(qNames):
            measureF = [
                dl.parameters[f'{qName}.measureF0'],
                dl.parameters[f'{qName}.measureF1']
            ]
            ps_q = ps[:, 2 * jdx:2 * (jdx + 1)]
            ps_q_cal = readoutCorrection(ps_q, [measureF])
            p1s_cal[:, jdx] = ps_q_cal[:, 1]
        results[rep]['p1s_cal'] = p1s_cal
        results[rep]['Vs'] = Vs

    p1s_cals = np.mean([data['p1s_cal'] for data in results.values()], axis=0)
    h_imshow = plot2D(p1s_cals[:,::-1].T,
              Vs / 5,
              range(11),
              xname='V/g',
              yname='Qubit index',
              size=[3.75, 2.0],
              cmap='RdYlBu_r',
              clim=[0, 1],
              isTight=False,
              cticks=[0, 0.5, 1],
              isCbar=True)
    plt.yticks(np.arange(0.5, 10, 1), ['10', '9', '8', '7', '6', '5', '4', '3', '2', '1'], fontsize=9)
    plt.xlabel('V/g', fontproperties='Times New Roman')
    plt.ylabel('Qubit index', fontproperties='Times New Roman')
    plt.text(6.5, -1.5, '$P_1$', fontsize=10)
    plt.grid(0)
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, top=0.95, right=0.95, bottom=0.25, hspace=0.12, wspace=0.12)
    # fig = plt.gcf()
    # cbaxes = fig.add_subplot()
    # cbaxes.set_position([0.65,0.15,0.26,0.05])
    # cbaxes.set_position([1.0,0.5,0.26,0.05])
    # cbar = plt.colorbar(h_imshow, cax=cbaxes, orientation='horizontal')
    # cbar = plt.colorbar(h_imshow, cax=cbaxes, orientation='vertical')
    # cbar.set_ticks([0,0.5,1])
    # cbaxes.tick_params(labelsize=9)


def fig3d():
    dl_quantum_data_legitimate_thermal = dataLab()
    dl_quantum_data_legitimate_thermal.pickle_load('\\'.join(file_path_list[:-1])+'\\dataset_quantum\\'+ '_Chao_N36R18_20220301_qaml_14727.pkl')
    dl_quantum_data_legitimate_local = dataLab()
    dl_quantum_data_legitimate_local.pickle_load('\\'.join(file_path_list[:-1])+'\\dataset_quantum\\'+ '_Chao_N36R18_20220301_qaml_14728.pkl')

    dl_quantum_data_adversarial_thermal = dataLab()
    dl_quantum_data_adversarial_thermal.pickle_load('\\'.join(file_path_list[:-1])+'\\dataset_quantum\\'+'_Chao_N36R18_20220301_qaml_14733.pkl')
    dl_quantum_data_adversarial_local = dataLab()
    dl_quantum_data_adversarial_local.pickle_load('\\'.join(file_path_list[:-1])+'\\dataset_quantum\\'+'_Chao_N36R18_20220301_qaml_14734.pkl')

    xticks = np.arange(0,21,10)
    #-----------------up pannel------------------
    p1s_legitimate = []
    for data in [dl_quantum_data_legitimate_thermal, dl_quantum_data_legitimate_local]:
        reps, batch_idxs, probs = data.get_data(dep_index=list(range(20,40)))
        qubit_num = 10
        p1 = np.mean(probs[:,:,1::2], axis=0)
        p1s_legitimate.append(p1[:,::-1])
    
    p1s_adversarial = []
    for data in [dl_quantum_data_adversarial_thermal, dl_quantum_data_adversarial_local]:
        reps, batch_idxs, probs = data.get_data(dep_index=list(range(20,40)))
        qubit_num = 10
        p1 = np.mean(probs[:,:,1::2], axis=0)
        p1s_adversarial.append(p1[:,::-1])
    
    p1s = [np.vstack(p1s_legitimate), np.vstack(p1s_adversarial)]
    #-------------------------------------------
    fig = plt.figure(figsize=(3.75, 2.3))
    for idx, p1 in enumerate(p1s):
        ax = fig.add_subplot(2,2,idx+1)
        ax.set_position(get_axes_position(2,2,idx+1))
        plot2D(p1.T,
                xs=range(0, 20+1),
                ys=range(qubit_num+1),
                size=[3.75, 2,3],
                cmap='RdYlBu_r',
                yname='',
                isCbar=False,
                isTight=False,
                ax=ax)
        if idx == 0:
            plt.yticks(np.arange(0.5, 10, 1), [], fontsize=9)
            familydic = dict(fontsize=10, family='Times New Roman')
            plt.text(-6, 1.6, 'Qubit index', rotation=90, fontdict=familydic)
            plt.text(-2.5, 9.0, '1', fontsize=9)
            plt.text(-3.5, 0.0, '10', fontsize=9)
        else:
            plt.yticks(np.arange(0.5, 10, 1), ['' for idx in range(1, 11)])
        plt.xticks(xticks, ['']*len(xticks))
        
        plt.grid(0)
        plt.tight_layout()
    #-----------------down pannel------------------
    dl_output_legitimate = dataLab()
    dl_output_legitimate.pickle_load('\\'.join(file_path_list[:-1])+'\\dataset_quantum\\''_Chao_N36R18_20220301_qaml_10894.pkl')
    probs_all = []
    for data in [dl_output_legitimate]:
        reps, batch_idxs, p1s = data.get_data('P1_corr', steps=[1, 1])
        idx0 = np.mod(batch_idxs, 2) == 0
        idx1 = np.mod(batch_idxs, 2) == 1
        p1s = np.hstack([p1s[:,idx1], p1s[:,idx0]])
        probs_all.append(p1s)

    save_path = '\\'.join(file_path_list[:-1])+'\dataset_quantum\\'
    session = 'Chao_N36R18_20220301_qaml'
    datasets = [13126, 13230, 13334, 13438, 13542, 13646, 13750, 13854, 13958, 14062, 14126, 14190, 14254, 14318, 14382, 14446, 14510, 14574, 14638, 14702]
    p1s = []
    for dataset in datasets:
        data = dataLab()
        data.pickle_load(save_path + '_'+session+'_'+str(dataset) + '.pkl')
        p1_corr = data.get_data('P1_corr')
        p1s.append(np.array(p1_corr))
    p1s = np.array(p1s)
    probs_all.append(p1s.T)
    xs = np.arange(1, len(batch_idxs) + 1)
    idxs = [np.arange(10), np.arange(10,20)]
    state_labels = ['thermalized', 'localized']
    markers = 'sD'
    colors = [['Crimson', 'RoyalBlue'], ['Crimson', 'RoyalBlue']]
    for didx, probs in enumerate(probs_all):
        ax = fig.add_subplot(2,2,didx+3)
        ax.set_position(get_axes_position(2,2,didx+3))
        for ii, idx in enumerate(idxs):
            ps = probs[:, idx]
            ax.plot(xs[idx], np.mean(1-2*ps, axis=0), label=f'{state_labels[ii]}', ls='', marker=markers[didx], markerfacecolor='none', markersize=4, color=colors[didx][ii])
        ax.hlines(0.1, xmin=xs[0] - 1, xmax=xs[-1] + 1, linestyles='dashed', color='k')
        ax.tick_params(labelsize=9)
        ax.set_yticks([-0.5,0.1,0.7])
        ax.set_yticklabels([-0.5,0.1,0.7], fontsize=9)
        if didx == 0:
            familydic = dict(fontsize=10, family='Times New Roman')
            plt.text(-7.0, 0.01, r'$\langle\hat\sigma_z\rangle$', rotation=90, fontdict=familydic)
            plt.text(15, -1.0, 'Sample index', fontproperties='Times New Roman', fontsize=10)
        else:
            ax.set_yticklabels(['']*3)
        ax.set_ylim([-0.5, 0.7])
        ax.set_xlim([xs[0] - 0.5, xs[-1] + 0.5])
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(i) for i in xticks], size=9)
        ax.tick_params(labelsize=9)
        plt.subplots_adjust(left=0.15, top=0.95, right=0.91, bottom=0.18, hspace=0.12, wspace=0.12)

def plot2D(mat,
           xs,
           ys,
           fig=None,
           titleName='',
           xname='',
           yname='',
           size=[10, 6.18],
           colorbarName='',
           isGrid=True,
           isTight=True,
           isCbar=True,
           clim=None,
           cticks=None,
           increase_text_size=0,
           cmap='jet',
           ax=None):
    if clim is None:
        clim=[None, None]
    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=size)
        elif fig == -1:
            pass
        else:
            fig = plt.figure(fig, figsize=size)
        ax = plt.gca()
    
    h_imshow = ax.imshow(mat,
                     extent=[xs[0], xs[-1], ys[0], ys[-1]],
                     interpolation='nearest',
                     origin='lower',
                     aspect='auto',
                     vmin=clim[0],
                     vmax=clim[1],
                     cmap=cmap)
    ax.set_xlabel(xname, size=10 + increase_text_size)
    ax.set_ylabel(yname, size=10 + increase_text_size)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[0], ys[-1])
    ax.tick_params(labelsize=9 + increase_text_size)
    ax.set_title(titleName, size=18 + increase_text_size)
    if isCbar:
        cbar = plt.colorbar(h_imshow)
        cbar.ax.tick_params(labelsize=f'{9 + increase_text_size}')
        font = {'family':'Times New Roman', 'size':10}
        # cbar.ax.set_ylabel(colorbarName, size=10 + increase_text_size)
        cbar.ax.set_ylabel(colorbarName, fontdict=font)
        cbar.set_ticks(cticks)
    if isGrid:
        ax.grid('on')
    if isTight:
        plt.tight_layout()
    return h_imshow

def get_axes_position(n_row, n_column, idx):
    hsep = 0.1 / n_row
    wsep = 0.06 / n_column
    width = 0.7 / n_column
    height = 0.7 / n_row
    left_start = 0.15
    bottom_start = 0.15

    row_idx = n_row - np.floor((idx - 1) / n_column) - 1
    column_idx = np.mod(idx - 1, n_column)

    left = left_start + column_idx * (width + wsep)
    bottom = bottom_start + row_idx * (height + hsep)

    position = [left, bottom, width, height]
    return position

def run():
    fig3b()
    fig3c()
    fig3d()