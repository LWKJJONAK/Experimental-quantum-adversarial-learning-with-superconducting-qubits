import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pickle
plt.ion()
file_path_list = os.path.dirname(os.path.abspath(__file__)).split('\\')
root_path = '\\'.join(file_path_list)
QAML_SAVEPATH = '\\'.join(file_path_list[:-1])+'\\encode_data_medical\\'

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
                  file_encode='encode_params_medical_10q_256.pkl',
                  discriminate_line=0.5,
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


def fig2_c(file_encode='encode_params_medical_10q_256.pkl', session='Chao_N36R18_20220203_qaml'):

    datasets_train = [87,   94,  100,  106,  112,  118,  124,  130,  136,  142,  148, 154,  160,  166,  172,  178,  184,  190,  196,  202,  208,  214,  220,  226,  232,  238,  244,  250,  256,  262,  268,  274,  280, 286,  292,  298,  304,  310,  316,  322,  328,  334,  340,  346,  352,  358,  364,  370,  376,  382,  388,  394,  400,  406,  412, 418,  424,  430,  436,  442,  448,  454,  460,  466,  472,  478, 484,  490,  496,  502,  508,  514,  520,  526,  532,  538,  544,  550,  556,  562,  568,  574,  580,  586,  592,  598,  604,  610,  616,  622,  628,  634,  640,  646,  652,  658,  664,  670,  676, 682,  688,  694,  700,  706,  712,  718,  724,  730,  736,  742, 965,  971,  977,  983,  989,  995, 1001, 1007, 1013, 1019, 1025, 1031,  820,  826,  832,  838, 1100, 1106, 1112, 1118, 1124, 1130, 1136, 1142, 1148, 1154, 1160, 1166, 1172, 1178, 1184, 1190, 1196, 1202, 1208, 1214, 1220, 1226, 1232, 1238, 1244, 1250, 1256, 1262, 1268, 1274, 1280, 1286, 1292, 1298, 1304, 1310, 1316, 1322, 1328, 1334, 1340, 1346, 1352, 1358, 1364, 1370, 1376, 1382, 1388, 1394, 1400, 1406, 1412, 1418, 1424, 1430, 1436, 1442, 1448, 1454, 1460, 1466, 1472, 1478, 1484, 1490, 1496, 1502, 1508, 1514, 1520, 1526, 1749, 1756, 1762]
    #, 1768, 1810, 1816, 1822, 1828, 1834, 1840, 1846, 1774, 1780, 1786, 1792, 1798, 1804, 1852, 1858, 1864, 1870, 1876, 1882]

    datasets_test = [88,   95,  101,  107,  113,  119,  125,  131,  137,  143,  149, 155,  161,  167,  173,  179,  185,  191,  197,  203,  209,  215, 221,  227,  233,  239,  245,  251,  257,  263,  269,  275,  281, 287,  293,  299,  305,  311,  317,  323,  329,  335,  341,  347, 353,  359,  365,  371,  377,  383,  389,  395,  401,  407,  413, 419,  425,  431,  437,  443,  449,  455,  461,  467,  473,  479, 485,  491,  497,  503,  509,  515,  521,  527,  533,  539,  545, 551,  557,  563,  569,  575,  581,  587,  593,  599,  605,  611,  617,  623,  629,  635,  641,  647,  653,  659,  665, 671,  677, 683,  689,  695,  701,  707,  713,  719,  725,  731,  737,  743, 966,  972,  978,  984,  990,  996, 1002, 1008, 1014, 1020, 1026, 1032,  821,  827,  833,  839, 1101, 1107, 1113, 1119, 1125, 1131, 1137, 1143, 1149, 1155, 1161, 1167, 1173, 1179, 1185, 1191, 1197, 1203, 1209, 1215, 1221, 1227, 1233, 1239, 1245, 1251, 1257, 1263, 1269, 1275, 1281, 1287, 1293, 1299, 1305, 1311, 1317, 1323, 1329, 1335, 1341, 1347, 1353, 1359, 1365, 1371, 1377, 1383, 1389, 1395, 1401, 1407, 1413, 1419, 1425, 1431, 1437, 1443, 1449, 1455, 1461, 1467, 1473, 1479, 1485, 1491, 1497, 1503, 1509, 1515, 1521, 1527, 1750, 1757, 1763]
    #, 1769, 1811, 1817, 1823, 1829, 1835, 1841, 1847, 1775, 1781, 1787, 1793, 1799, 1805, 1853, 1859, 1865, 1871, 1877, 1883]

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

    plot_train_dynamics_medical(result, ((20.0, 0.0)))

def plot_train_dynamics_medical(result, iter_idx_pair_cut=(20.0, 0.0)):
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

    fig = plt.figure(figsize=[3.75, 2.5])
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
             color='C2',
             label='Training',
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
             color='C3',
             label='Test',
             markerfacecolor='none',
             linewidth=1.0,
             capsize=4, 
             alpha=alpha)
    plt.tick_params(labelsize=labelsize)

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
    plt.ylim(0.35, 1.05)
    ax_loss, ax_acc = fig.axes
    minor_xticks = []
    major_xticks = []
    minor_xticklabels = []
    major_xticklabels = []
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
    ax_loss.set_yticks([0.4, 0.6, 0.8])
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
    font = {'family':'Times New Roman', 'size': 9}
    ax_acc.legend(prop=font, frameon=False)
    # ax_acc.legend(fontsize=legendsize, frameon=False, prop=font)
    plt.tick_params(labelsize=labelsize)
    plt.subplots_adjust(left=0.14, top=0.95, right=0.97, bottom=0.18, hspace=0.12, wspace=0.0)

def fig4_a(file_encode='encode_params_medical_adv_10q_256.pkl', session='Chao_N36R18_20220203_qaml'):
    datasets_train = [2235, 2242, 2248, 2254, 2260, 2266, 2272, 2278, 2284, 2290, 2296, 2302, 2308, 2314, 2320, 2326, 2332, 2338, 2344, 2350, 2356, 2362, 2368, 2374, 2380, 2386, 2392, 2398, 2404, 2410, 2416, 2422, 2428, 2434, 2440, 2446, 2452, 2458, 2464, 2470, 2476, 2482, 2488, 2494, 2500, 2506, 2512, 2518, 2524, 2530, 2536, 2542, 2548, 2554, 2560, 2566, 2572, 2578, 2584, 2590, 2596, 2602, 2608, 2614, 2620, 2626, 2632, 2638, 2644, 2650, 2656, 2662, 2668, 2674, 2680, 2686, 2693, 2699, 2705, 2711, 2717, 2723, 2729, 2735, 2741, 2747, 2753, 2759, 2765, 2771, 2777, 2783, 2789, 2795, 2801, 2807, 2813, 2819, 2825, 2831, 2837, 2843, 2849, 2855, 2861, 2867, 2873, 2879, 2885, 2891, 2897, 2903, 2909, 2915, 2921, 2927, 2933, 2939, 2945, 2951, 2957, 2963, 2970, 2976, 2982, 2988, 2994, 3000, 3006, 3012, 3018, 3024, 3030, 3036, 3042, 3050, 3057, 3063, 3069, 3075, 3081, 3104, 3111, 3117, 3123, 3129, 3135, 3141, 3147, 3153, 3159, 3165, 3171, 3177, 3183, 3189, 3195, 3201, 3207, 3213, 3219, 3225, 3231, 3237, 3243, 3249, 3255, 3261, 3267, 3273, 3279, 3285, 3291, 3297, 3303, 3309, 3315, 3321, 3327, 3333, 3339, 3345, 3351, 3357, 3363, 3370, 3376, 3382, 3388, 3394, 3400, 3406, 3412, 3418, 3424, 3430, 3436, 3442, 3448, 3454, 3460, 3466, 3472, 3478, 3484, 3490, 3496, 3502, 3508, 3514, 3520, 3526, 3532, 3538, 3544, 3550, 3556, 3562, 3568, 3574, 3580, 3586, 3592, 3598, 3604, 3610, 3616, 3622, 3628, 3634, 3640, 3646, 3652, 3658, 3665, 3671, 3677, 3683, 3689, 3695, 3701, 3707, 3713, 3719, 3725, 3731, 3737, 3743, 3749, 3755, 3761, 3767, 3773, 3779, 3785, 3791, 3797, 3803, 3809, 3815, 3821, 3827, 3833, 3839, 3845, 3851, 3857, 3863, 3869, 3875, 3881, 3887, 3893, 3899, 3905, 3911, 3917, 3923, 3929, 3935, 3941, 3947, 3953, 3959, 3965, 3971, 3977, 3983, 3989, 3995, 4001, 4007, 4013, 4019, 4025, 4031, 4037, 4043, 4049, 4055, 4061]
    #, 4067, 4073, 4079, 4085, 4091, 4097, 4103, 4109, 4115, 4121, 4127, 4133, 4139, 4145, 4151, 4157, 4163, 4169, 4175, 4181, 4187, 4193, 4199, 4205, 4211, 4217, 4223, 4229, 4235, 4241, 4247, 4253, 4259, 4265, 4271, 4277, 4283, 4289, 4295, 4301, 4307, 4313, 4319, 4325, 4331, 4337, 4343, 4349, 4355]

    datasets_test = [2236, 2243, 2249, 2255, 2261, 2267, 2273, 2279, 2285, 2291, 2297, 2303, 2309, 2315, 2321, 2327, 2333, 2339, 2345, 2351, 2357, 2363,  2369, 2375, 2381, 2387, 2393, 2399, 2405, 2411, 2417, 2423, 2429, 2435, 2441, 2447, 2453, 2459, 2465, 2471, 2477, 2483, 2489, 2495, 2501, 2507, 2513, 2519, 2525, 2531, 2537, 2543, 2549, 2555, 2561, 2567, 2573, 2579, 2585, 2591, 2597, 2603, 2609, 2615, 2621, 2627, 2633, 2639, 2645, 2651, 2657, 2663, 2669, 2675, 2681, 2687, 2694, 2700, 2706, 2712, 2718, 2724, 2730, 2736, 2742, 2748, 2754, 2760, 2766, 2772, 2778, 2784, 2790, 2796, 2802, 2808, 2814, 2820, 2826, 2832, 2838, 2844, 2850, 2856, 2862, 2868, 2874, 2880, 2886, 2892, 2898, 2904, 2910, 2916, 2922, 2928, 2934, 2940, 2946, 2952, 2958, 2964, 2971, 2977, 2983, 2989, 2995, 3001, 3007, 3013, 3019, 3025, 3031, 3037, 3043, 3051, 3058, 3064, 3070, 3076, 3082, 3105, 3112, 3118, 3124, 3130, 3136, 3142, 3148, 3154, 3160, 3166, 3172, 3178, 3184, 3190, 3196, 3202, 3208, 3214, 3220, 3226, 3232, 3238, 3244, 3250, 3256, 3262, 3268, 3274, 3280, 3286, 3292, 3298, 3304, 3310, 3316, 3322, 3328, 3334, 3340, 3346, 3352, 3358, 3364, 3371, 3377, 3383, 3389, 3395, 3401, 3407, 3413, 3419, 3425, 3431, 3437, 3443, 3449, 3455, 3461, 3467, 3473, 3479, 3485, 3491, 3497, 3503, 3509, 3515, 3521, 3527, 3533, 3539, 3545, 3551, 3557, 3563, 3569, 3575, 3581, 3587, 3593, 3599, 3605, 3611, 3617, 3623, 3629, 3635, 3641, 3647, 3653, 3659, 3666, 3672, 3678, 3684, 3690, 3696, 3702, 3708, 3714, 3720, 3726, 3732, 3738, 3744, 3750, 3756, 3762, 3768, 3774, 3780, 3786, 3792, 3798, 3804, 3810, 3816, 3822, 3828, 3834, 3840, 3846, 3852, 3858, 3864, 3870, 3876, 3882, 3888, 3894, 3900, 3906, 3912, 3918, 3924, 3930, 3936, 3942, 3948, 3954, 3960, 3966, 3972, 3978, 3984, 3990, 3996, 4002, 4008, 4014, 4020, 4026, 4032, 4038, 4044, 4050, 4056, 4062]#, 4068, 4074, 4080, 4086, 4092, 4098, 4104, 4110, 4116, 4122, 4128, 4134, 4140, 4146, 4152, 4158, 4164, 4170, 4176, 4182, 4188, 4194, 4200, 4206, 4212, 4218, 4224, 4230, 4236, 4242, 4248, 4254, 4260, 4266, 4272, 4278, 4284, 4290, 4296, 4302, 4308, 4314, 4320, 4326, 4332, 4338, 4344, 4350, 4356]

    save_path = '\\'.join(file_path_list[:-1])+'\\dataset_train_adv\\'

    result = {'full':{}, 'adv':{}, 'origin':{}}
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
                             collect=True)
        for des in ['full', 'adv', 'origin']:
            result[des].update({
            (iter_idx, row_idx): {
                'loss_train': para[des]['train']['loss'],
                'loss_train_std': para[des]['train']['loss_std'],
                'accuracy_train': para[des]['train']['accuracy'],
                'loss_test': para[des]['test']['loss'],
                'loss_test_std': para[des]['test']['loss_std'],
                'accuracy_test': para[des]['test']['accuracy']
            }
        })
    plot_train_dynamics_medical_adv(result['origin'], result['adv'], iter_idx_pair_cut=(30.0, 0.0))


def plot_train_dynamics_medical_adv(result_ori, result_adv, iter_idx_pair_cut=(30.0, 0.0)):
    iter_idx_pair = list(result_ori.keys())
    if iter_idx_pair_cut is not None:
        cut_idx = iter_idx_pair.index(iter_idx_pair_cut) + 1
        iter_idx_pair = iter_idx_pair[:cut_idx]
    iter_idx = np.arange(len(iter_idx_pair))
    accuracy_test_ori = [
        result_ori[pair]['accuracy_test'] for pair in iter_idx_pair
    ]
    accuracy_test_adv = [
        result_adv[pair]['accuracy_test'] for pair in iter_idx_pair
    ]

    fig = plt.figure(figsize=[2.5, 2.5])
    alpha = 1.0
    markersize = 4
    ax = fig.add_subplot(1,1,1, projection='polar')
    iter_idxs = np.arange(0, 31, 1)
    iter_idxs = iter_idxs*2*np.pi/(len(iter_idxs)+5)
    accuracy_test_oris = accuracy_test_ori[::10]
    accuracy_test_advs = accuracy_test_adv[::10]
    ax.plot(iter_idxs,
             accuracy_test_oris,
             marker='o',
             markersize=markersize,
             ls='-',
             color='C0',
             markerfacecolor='none',
             linewidth=1.0,
             label='Legitimate',
             alpha=alpha)
    ax.plot(iter_idxs,
             accuracy_test_advs,
             marker='^',
             markersize=markersize,
             ls='-',
             color='C1',
             markerfacecolor='none',
             linewidth=1.0,
             label='Adversarial',
             alpha=alpha)
    font = {'family':'Times New Roman', 'size': 9}
    ax.legend(['Legitimate', 'Adversarial'], bbox_to_anchor=[1.0, -0.05, 0.0, 0.0], frameon=False,  ncol=3, handletextpad=0.1, columnspacing=0.5, borderaxespad=0.0, prop=font)
    angles = np.array([0, 5, 10, 15, 20, 25, 30])*360/(len(iter_idxs)+5)
    ax.set_thetagrids(angles, labels=[], fontsize=9)
    ax.set_rlim(0.0, 1.05)
    plt.tick_params('y', direction='out', labelbottom=True, bottom=True)
    plt.yticks(ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[])
    ax.set_rlabel_position(0)
    # return ax
    ax.spines['polar'].set_color('none')
    plt.text(0.05, 1.1, '0', horizontalalignment='center', verticalalignment='top', fontsize=8)
    plt.text(np.math.radians(52), 1.2, '5', horizontalalignment='center', verticalalignment='top', fontsize=8)
    plt.text(np.math.radians(50*2), 1.22, '10', horizontalalignment='center', verticalalignment='top', fontsize=8)
    plt.text(np.math.radians(49*3), 1.2, '15', horizontalalignment='center', verticalalignment='top', fontsize=8)
    plt.text(np.math.radians(49*4), 1.15, '20', horizontalalignment='center', verticalalignment='top', fontsize=8)
    plt.text(np.math.radians(49.5*5), 1.12, '25', horizontalalignment='center', verticalalignment='top', fontsize=8)
    plt.text(np.math.radians(51*6), 1.15, '30', horizontalalignment='center', verticalalignment='top', fontsize=8)
    plt.tight_layout()

def fig2_d(datasets_train=None, datasets_test=None, session=None, save_path=None, file_encode='encode_params_medical_10q_256.pkl'):
    if datasets_train is None:
        datasets_train = [87, 388, 1762]
    if datasets_test is None:
        datasets_test = [88, 389, 1763]
    if session is None:
        session = 'Chao_N36R18_20220203_qaml'
    if save_path is None:
        save_path = '\\'.join(file_path_list[:-1])+'\\dataset_train\\'
    plt.figure(figsize=[3.75, 1.6])
    fontsize = 10
    for ii, dataset_train in enumerate(datasets_train):
        dataset_test = datasets_test[ii]
        ana_info = analysis_loss(dataset_train, dataset_test,session, save_path,
                             file_encode=file_encode,
                             do_plot=False,
                             collect=True)
        plt.subplot(1, 3, ii+1)
        for ana_type, ai in ana_info.items():
            plot_loss_accuracy(ai, ana_type)   
        if ii == 0:
            familydic = dict(fontsize=10, family='Times New Roman')
            plt.text(-27, -0.1, r'$\langle\hat\sigma_z\rangle$', rotation='vertical', fontdict=familydic)
            plt.yticks([-0.6, 0.0, 0.6], size=9)
            font = {'family':'Times New Roman', 'size': 9}
            plt.legend(['Hand', 'Breast'], frameon=False, loc=3, bbox_to_anchor=(2.0, 1.0), ncol=3,  handletextpad=0.0, columnspacing=0.3, borderaxespad=0.0, prop=font)
        elif ii == 1:
            plt.yticks([-0.6, 0.0, 0.6], labels=[], size=9)
            plt.xlabel('Sample index', fontsize=fontsize, fontproperties='Times New Roman')
        else:
            plt.yticks([-0.6, 0.0, 0.6], labels=[], size=9)
    plt.subplots_adjust(left=0.14, top=0.85, right=0.97, bottom=0.28, hspace=0, wspace=0.15)

def plot_loss_accuracy(ana_info, titleStr=''):
    colors = {'0': '#4169E1', '1': '#DC143C'}
    mks = {'train': 'o', 'test': 's'}
    l = {}
    a = {}
    xs = {}
    xs_temp = {}
    for type, type_info in ana_info.items():
        if type == 'test':
            bi0 = type_info['0']['batch_idx']
            bi1 = type_info['1']['batch_idx']
            bi = list(np.sort(np.hstack([bi0, bi1])))
            xs[type] = np.arange(len(bi))
            xs_temp['0'] = np.array([bi.index(idx) for idx in bi0])
            xs_temp['1'] = np.array([bi.index(idx) for idx in bi1])
            if type == 'test' and 'train' in xs:
                xs[type] += xs['train'][-1] + 1
                xs_temp['0'] += xs['train'][-1] + 1
                xs_temp['1'] += xs['train'][-1] + 1
            l[type] = type_info['loss']
            a[type] = type_info['accuracy']
            for digit, digit_info in type_info.items():
                # sigmaz = 1 - 2*digit_info['probs']
                if digit in ['0', '1']:
                    sigmaz = 1 - 2*digit_info['probs']
                    plt.plot(xs_temp[digit],
                            sigmaz,
                            color=colors[digit],
                            marker = mks[type],
                            ms=4,
                            alpha=1.0, markerfacecolor='none', ls='')
    plt.hlines(y=0.0, xmin=0, xmax=10000, linestyles='--', color='k', linewidth=1.0)
    plt.xticks([0, 25, 50], size=9, rotation=0)
    plt.xlim([-1, 51])
    plt.ylim([-0.6, 0.6])
    # plt.tight_layout()

def fig2_f(dataset_train=None, dataset_test=None, session=None, save_path=None, file_encode='encode_params_medical_adv_10q_256.pkl'):
    if dataset_train is None:
        dataset_train = 1915
    if dataset_test is None:
        dataset_test = 1916
    if session is None:
        session = 'Chao_N36R18_20220203_qaml'
    if save_path is None:
        save_path = '\\'.join(file_path_list[:-1])+'\\dataset_medical\\'

    idx_all = [3,  4,  5,  6,  8, 11, 12, 13, 14, 15, 16, 19, 22, 23, 24, 27, 29, 30, 32, 36, 38, 39, 42, 44, 45, 47, 52, 54, 56,57, 58, 60, 61, 62, 64, 66, 68, 69, 70, 76, 80, 82, 85, 86, 87, 90, 91, 94, 97, 98]

    idx0 = [4,  5,  6,  8, 13, 14, 16, 22, 23, 24, 38, 39, 44, 45, 52, 57, 60, 66, 76, 85, 91, 97]

    idx1 = [3, 11, 12, 15, 19, 27, 29, 30, 32, 36, 42, 47, 54, 56, 58, 61, 62, 64, 68, 69, 70, 80, 82, 86, 87, 90, 94, 98]

    ana_info = analysis_loss(dataset_train, dataset_test,session, save_path,
                             file_encode=file_encode,
                             do_plot=False,
                             collect=True)

    batch_idx0 = ana_info['adv']['test']['0']['batch_idx']
    batch_idx1 = ana_info['adv']['test']['1']['batch_idx']
    probs_0 = ana_info['adv']['test']['0']['probs']
    probs_1 = ana_info['adv']['test']['1']['probs']

    batch_idxs = list(np.sort(np.hstack([batch_idx0, batch_idx1])))
    probs = [0]*len(batch_idxs)
    for idx, batch_idx in enumerate(batch_idxs):
        if batch_idx in batch_idx0:
            ii = list(batch_idx0).index(batch_idx)
            probs[idx] = probs_0[ii]
        else:
            ii = list(batch_idx1).index(batch_idx)
            probs[idx] = probs_1[ii]
    probs = np.array(probs)
    colors = ['RoyalBlue', 'Crimson']
    plt.figure(figsize=(3, 1.6))
    plt.plot(idx0[0], 1-2*probs[idx0[0]], marker='D', color='RoyalBlue', markerfacecolor='none', alpha=1.0, markersize=4, ls='')
    
    for index, idx in enumerate(idx_all):
        if index == idx0[0]:
            pass
        elif idx in idx0:
            color=colors[0]
        else:
            color=colors[1]
        plt.plot(index, 1-2*probs[idx], marker='D', color=color, markerfacecolor='none', alpha=1.0, markersize=4, ls='')
    plt.hlines(y=0.0, xmin=0, xmax=10000, linestyles='--', color='k', linewidth=1.0)
    plt.xlim(-1, 51)
    plt.xticks([0, 25, 50], fontsize=9)
    plt.yticks([-0.6, 0,  0.6], fontsize=9)
    familydic = dict(fontsize=10, family='Times New Roman')
    plt.text(-13, -0.1, r'$\langle\hat\sigma_z\rangle$', rotation='vertical', fontdict=familydic)
    plt.xlabel('Sample index', fontsize=10, fontproperties='Times New Roman')
    font = {'family':'Times New Roman', 'size': 9}
    plt.legend(['Hand', 'Breast'], frameon=False, loc=3, bbox_to_anchor=(0.35, 1.0), ncol=3, handletextpad=0.0, columnspacing=0.3, borderaxespad=0.0, prop=font)
    plt.subplots_adjust(left=0.2, top=0.85, right=0.85, bottom=0.28, hspace=None, wspace=None)

def medical():
    encode_path = '\\'.join(file_path_list[:-1])+'\\adv_matmedical_data_new_20220221.pkl'
    with open(encode_path, 'rb') as f:
        encode_params = pickle.load(f)
    legi_sample_breast = encode_params['legi_sample_breast']
    legi_sample_hand = encode_params['legi_sample_hand']
    adv_sample_breast = encode_params['adv_sample_breast']
    adv_sample_hand = encode_params['adv_sample_hand']
    legi_sample_breast = np.reshape((legi_sample_breast.T.flatten())[:256], [16, 16])-np.pi
    adv_sample_breast = np.reshape((adv_sample_breast.T.flatten())[:256], [16, 16])-np.pi
    legi_sample_hand = np.reshape((legi_sample_hand.T.flatten())[:256], [16, 16])-np.pi
    adv_sample_hand = np.reshape((adv_sample_hand.T.flatten())[:256], [16, 16])-np.pi
    lw=2
    markersize=10
    fig=plt.figure(figsize=[2.3, 2.3])
    ax1 = plt.gca()
    ax1.imshow(legi_sample_hand.T, cmap='Greys', vmin=0.0, vmax=0.3)
    ax1.set_xlim(-0.5, 15.5)
    ax1.set_ylim(15.5, -0.5)
    ax1.tick_params(axis='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    ax1.hlines(np.arange(0, 17)-0.5, -0.5, 15.5, color='w', alpha=1, lw=2)
    ax1.vlines(np.arange(0, 17)-0.5, -0.5, 15.5, color='w', alpha=1, lw=2)
    ax1.axis(True)
    ax1.vlines(np.arange(0, 4)-0.5, -0.5, 0.5, color='w', alpha=1, lw=lw)
    ax1.hlines([-0.5, 0.5], -0.5, 3-0.5, color='w', alpha=1, lw=lw)
    ax1.vlines(np.arange(0, 4)+13-0.5, -0.5+15, 0.5+15, color='w', alpha=1, lw=lw)
    ax1.hlines([-0.5+15, 0.5+15], 13-0.5, 16-0.5, color='w', alpha=1, lw=lw)

def fig2_e():
    encode_path = '\\'.join(file_path_list[:-1])+'\\adv_matmedical_data_new_20220221.pkl'
    with open(encode_path, 'rb') as f:
        encode_params = pickle.load(f)
    legi_sample_breast = encode_params['legi_sample_breast']
    legi_sample_hand = encode_params['legi_sample_hand']
    adv_sample_breast = encode_params['adv_sample_breast']
    adv_sample_hand = encode_params['adv_sample_hand']
    legi_sample_breast = np.reshape((legi_sample_breast.T.flatten())[:256], [16, 16])-np.pi
    adv_sample_breast = np.reshape((adv_sample_breast.T.flatten())[:256], [16, 16])-np.pi
    legi_sample_hand = np.reshape((legi_sample_hand.T.flatten())[:256], [16, 16])-np.pi
    adv_sample_hand = np.reshape((adv_sample_hand.T.flatten())[:256], [16, 16])-np.pi
    lw=2
    markersize=10
    # plt.close('all')
    data = {'legi_sample_breast':legi_sample_breast, 'adv_sample_breast':adv_sample_breast, 'legi_sample_hand':legi_sample_hand, 'adv_sample_hand':adv_sample_hand}
    keys = ['legi_sample_breast', 'adv_sample_breast', 'legi_sample_hand', 'adv_sample_hand']
    for ii in range(4):
        fig=plt.figure(figsize=[2.3, 2.3])
        ax1 = plt.gca()
        ax1.imshow(data[keys[ii]].T, cmap='Greys')
        ax1.set_xlim(-0.5, 15.5)
        ax1.set_ylim(15.5, -0.5)
        ax1.tick_params(axis='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        
        ax1.hlines(np.arange(0, 17)-0.5, -0.5, 15.5, color='w', alpha=1, lw=2)
        ax1.vlines(np.arange(0, 17)-0.5, -0.5, 15.5, color='w', alpha=1, lw=2)
        ax1.axis(True)
        
        ax1.vlines(np.arange(0, 4)-0.5, -0.5, 0.5, color='w', alpha=1, lw=lw)
        ax1.hlines([-0.5, 0.5], -0.5, 3-0.5, color='w', alpha=1, lw=lw)
        
        ax1.vlines(np.arange(0, 4)+13-0.5, -0.5+15, 0.5+15, color='w', alpha=1, lw=lw)
        ax1.hlines([-0.5+15, 0.5+15], 13-0.5, 16-0.5, color='w', alpha=1, lw=lw)

def medical_noise():
    encode_path = '\\'.join(file_path_list[:-1])+'\\adv_matmedical_data_new_20220221.pkl'
    with open(encode_path, 'rb') as f:
        encode_params = pickle.load(f)
    legi_sample_breast = encode_params['legi_sample_breast']
    legi_sample_hand = encode_params['legi_sample_hand']
    adv_sample_breast = encode_params['adv_sample_breast']
    adv_sample_hand = encode_params['adv_sample_hand']
    legi_sample_breast = np.reshape((legi_sample_breast.T.flatten())[:256], [16, 16])-np.pi
    adv_sample_breast = np.reshape((adv_sample_breast.T.flatten())[:256], [16, 16])-np.pi
    legi_sample_hand = np.reshape((legi_sample_hand.T.flatten())[:256], [16, 16])-np.pi
    adv_sample_hand = np.reshape((adv_sample_hand.T.flatten())[:256], [16, 16])-np.pi
    lw=2
    markersize=10
    fig=plt.figure(figsize=[2.3, 2.3])
    ax1 = plt.gca()
    ax1.imshow(np.abs(adv_sample_hand.T-legi_sample_hand.T), cmap='Greys', vmin=0.0, vmax=0.3)
    ax1.set_xlim(-0.5, 15.5)
    ax1.set_ylim(15.5, -0.5)
    ax1.tick_params(axis='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    ax1.hlines(np.arange(0, 17)-0.5, -0.5, 15.5, color='w', alpha=1, lw=2)
    ax1.vlines(np.arange(0, 17)-0.5, -0.5, 15.5, color='w', alpha=1, lw=2)
    ax1.axis(True)
    ax1.vlines(np.arange(0, 4)-0.5, -0.5, 0.5, color='w', alpha=1, lw=lw)
    ax1.hlines([-0.5, 0.5], -0.5, 3-0.5, color='w', alpha=1, lw=lw)
    ax1.vlines(np.arange(0, 4)+13-0.5, -0.5+15, 0.5+15, color='w', alpha=1, lw=lw)
    ax1.hlines([-0.5+15, 0.5+15], 13-0.5, 16-0.5, color='w', alpha=1, lw=lw)

def fig4_b():
    encode_path = '\\'.join(file_path_list[:-1])+'\\adv_matmedical_data_new_20220221.pkl'
    with open(encode_path, 'rb') as f:
        encode_params = pickle.load(f)
    legi_sample_breast = encode_params['legi_sample_breast']
    legi_sample_hand = encode_params['legi_sample_hand']
    adv_sample_breast = encode_params['adv_sample_breast']
    adv_sample_hand = encode_params['adv_sample_hand']
    legi_sample_breast = np.reshape((legi_sample_breast.T.flatten())[:256], [16, 16])-np.pi
    adv_sample_breast = np.reshape((adv_sample_breast.T.flatten())[:256], [16, 16])-np.pi
    legi_sample_hand = np.reshape((legi_sample_hand.T.flatten())[:256], [16, 16])-np.pi
    adv_sample_hand = np.reshape((adv_sample_hand.T.flatten())[:256], [16, 16])-np.pi
    lw=2
    markersize=10
    # plt.close('all')
    fig=plt.figure(figsize=[2.3, 2.3])

    ax1 = plt.gca()
    ax1.imshow(adv_sample_hand.T, cmap='Greys')
    ax1.set_xlim(-0.5, 15.5)
    ax1.set_ylim(15.5, -0.5)
    ax1.tick_params(axis='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    ax1.hlines(np.arange(0, 17)-0.5, -0.5, 15.5, color='w', alpha=1, lw=2)
    ax1.vlines(np.arange(0, 17)-0.5, -0.5, 15.5, color='w', alpha=1, lw=2)
    ax1.axis(True)

    ax1.vlines(np.arange(0, 4)-0.5, -0.5, 0.5, color='w', alpha=1, lw=lw)
    ax1.hlines([-0.5, 0.5], -0.5, 3-0.5, color='w', alpha=1, lw=lw)
    
    ax1.vlines(np.arange(0, 4)+13-0.5, -0.5+15, 0.5+15, color='w', alpha=1, lw=lw)
    ax1.hlines([-0.5+15, 0.5+15], 13-0.5, 16-0.5, color='w', alpha=1, lw=lw)

def get_measureF0F1(session='Chao_N36R18_20220203_qaml', train=True, adv=False):
    datasets_train = [87,   94,  100,  106,  112,  118,  124,  130,  136,  142,  148, 154,  160,  166,  172,  178,  184,  190,  196,  202,  208,  214,  220,  226,  232,  238,  244,  250,  256,  262,  268,  274,  280, 286,  292,  298,  304,  310,  316,  322,  328,  334,  340,  346,  352,  358,  364,  370,  376,  382,  388,  394,  400,  406,  412, 418,  424,  430,  436,  442,  448,  454,  460,  466,  472,  478, 484,  490,  496,  502,  508,  514,  520,  526,  532,  538,  544,  550,  556,  562,  568,  574,  580,  586,  592,  598,  604,  610,  616,  622,  628,  634,  640,  646,  652,  658,  664,  670,  676, 682,  688,  694,  700,  706,  712,  718,  724,  730,  736,  742, 965,  971,  977,  983,  989,  995, 1001, 1007, 1013, 1019, 1025, 1031,  820,  826,  832,  838, 1100, 1106, 1112, 1118, 1124, 1130, 1136, 1142, 1148, 1154, 1160, 1166, 1172, 1178, 1184, 1190, 1196, 1202, 1208, 1214, 1220, 1226, 1232, 1238, 1244, 1250, 1256, 1262, 1268, 1274, 1280, 1286, 1292, 1298, 1304, 1310, 1316, 1322, 1328, 1334, 1340, 1346, 1352, 1358, 1364, 1370, 1376, 1382, 1388, 1394, 1400, 1406, 1412, 1418, 1424, 1430, 1436, 1442, 1448, 1454, 1460, 1466, 1472, 1478, 1484, 1490, 1496, 1502, 1508, 1514, 1520, 1526, 1749, 1756, 1762]
    #, 1768, 1810, 1816, 1822, 1828, 1834, 1840, 1846, 1774, 1780, 1786, 1792, 1798, 1804, 1852, 1858, 1864, 1870, 1876, 1882]

    save_path = '\\'.join(file_path_list[:-1])+'\\dataset_train\\'


    datasets_train_adv = [2235, 2242, 2248, 2254, 2260, 2266, 2272, 2278, 2284, 2290, 2296, 2302, 2308, 2314, 2320, 2326, 2332, 2338, 2344, 2350, 2356, 2362, 2368, 2374, 2380, 2386, 2392, 2398, 2404, 2410, 2416, 2422, 2428, 2434, 2440, 2446, 2452, 2458, 2464, 2470, 2476, 2482, 2488, 2494, 2500, 2506, 2512, 2518, 2524, 2530, 2536, 2542, 2548, 2554, 2560, 2566, 2572, 2578, 2584, 2590, 2596, 2602, 2608, 2614, 2620, 2626, 2632, 2638, 2644, 2650, 2656, 2662, 2668, 2674, 2680, 2686, 2693, 2699, 2705, 2711, 2717, 2723, 2729, 2735, 2741, 2747, 2753, 2759, 2765, 2771, 2777, 2783, 2789, 2795, 2801, 2807, 2813, 2819, 2825, 2831, 2837, 2843, 2849, 2855, 2861, 2867, 2873, 2879, 2885, 2891, 2897, 2903, 2909, 2915, 2921, 2927, 2933, 2939, 2945, 2951, 2957, 2963, 2970, 2976, 2982, 2988, 2994, 3000, 3006, 3012, 3018, 3024, 3030, 3036, 3042, 3050, 3057, 3063, 3069, 3075, 3081, 3104, 3111, 3117, 3123, 3129, 3135, 3141, 3147, 3153, 3159, 3165, 3171, 3177, 3183, 3189, 3195, 3201, 3207, 3213, 3219, 3225, 3231, 3237, 3243, 3249, 3255, 3261, 3267, 3273, 3279, 3285, 3291, 3297, 3303, 3309, 3315, 3321, 3327, 3333, 3339, 3345, 3351, 3357, 3363, 3370, 3376, 3382, 3388, 3394, 3400, 3406, 3412, 3418, 3424, 3430, 3436, 3442, 3448, 3454, 3460, 3466, 3472, 3478, 3484, 3490, 3496, 3502, 3508, 3514, 3520, 3526, 3532, 3538, 3544, 3550, 3556, 3562, 3568, 3574, 3580, 3586, 3592, 3598, 3604, 3610, 3616, 3622, 3628, 3634, 3640, 3646, 3652, 3658, 3665, 3671, 3677, 3683, 3689, 3695, 3701, 3707, 3713, 3719, 3725, 3731, 3737, 3743, 3749, 3755, 3761, 3767, 3773, 3779, 3785, 3791, 3797, 3803, 3809, 3815, 3821, 3827, 3833, 3839, 3845, 3851, 3857, 3863, 3869, 3875, 3881, 3887, 3893, 3899, 3905, 3911, 3917, 3923, 3929, 3935, 3941, 3947, 3953, 3959, 3965, 3971, 3977, 3983, 3989, 3995, 4001, 4007, 4013, 4019, 4025, 4031, 4037, 4043, 4049, 4055, 4061]
    save_path_adv = '\\'.join(file_path_list[:-1])+'\\dataset_train_adv\\'


    qubits = ['q3_5', 'q3_3', 'q5_3', 'q7_3', 'q9_3', 'q9_5', 'q9_7', 'q7_7', 'q7_9', 'q5_9']
    measureF0 = {'q3_5':[], 'q3_3':[], 'q5_3':[], 'q7_3':[], 'q9_3':[], 'q9_5':[], 'q9_7':[], 'q7_7':[], 'q7_9':[], 'q5_9':[]}
    measureF1 = {'q3_5':[], 'q3_3':[], 'q5_3':[], 'q7_3':[], 'q9_3':[], 'q9_5':[], 'q9_7':[], 'q7_7':[], 'q7_9':[], 'q5_9':[]}
    if train:
        for dataset in datasets_train:
            data = dataLab()
            data.pickle_load(save_path + '_'+session+'_'+str(dataset) + '.pkl')
            for qubit in qubits:
                F0 = data.parameters[qubit + '.measureF0']
                measureF0[qubit].append(F0)
                F1 = data.parameters[qubit + '.measureF1']
                measureF1[qubit].append(F1)
    if adv:
        for dataset in datasets_train_adv:
            data = dataLab()
            data.pickle_load(save_path_adv + '_'+session+'_'+str(dataset) + '.pkl')
            for qubit in qubits:
                F0 = data.parameters[qubit + '.measureF0']
                measureF0[qubit].append(F0)
                F1 = data.parameters[qubit + '.measureF1']
                measureF1[qubit].append(F1)
    print(len(measureF0['q9_3']))
    for qubit in qubits:
        print(np.round(np.mean(measureF0[qubit]), 4), np.round(np.mean(measureF1[qubit]), 4))

def run():
    fig2_c()
    fig2_d()
    fig2_e()
    fig2_f()
    fig4_a()
    fig4_b()
