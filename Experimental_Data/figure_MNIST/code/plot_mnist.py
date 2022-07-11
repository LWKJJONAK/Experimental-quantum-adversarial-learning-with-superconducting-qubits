import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
plt.ion()
file_path_list = os.path.dirname(os.path.abspath(__file__)).split('\\')
root_path = '\\'.join(file_path_list)
QAML_SAVEPATH = '\\'.join(file_path_list[:-1])+'\\encode_MNIST\\'

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
                  file_encode='encode_params_10q_256.pkl',
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

#file_encode='encode_params_10q_256.pkl'
#file_encode='encode_params_adv00_06_10q_256.pkl'

def analysis_train_dynamics(file_encode='encode_params_10q_256.pkl', session='Chao_N36R18_20220120_qaml', do_plot=True):
    
    if file_encode == 'encode_params_10q_256.pkl':
        # datasets_qaml = [17, 294, 852, 955, 1123, 1134, 1151, 1167, 1183, 1200, 1271]
        # data_name = ['_Chao_N36R18_20220120_qaml_17.pkl', '_Chao_N36R18_20220120_qaml_294.pkl', '_Chao_N36R18_20220120_qaml_852.pkl', '_Chao_N36R18_20220120_qaml_955.pkl', '_Chao_N36R18_20220120_qaml_1123.pkl', '_Chao_N36R18_20220120_qaml_1134.pkl', '_Chao_N36R18_20220120_qaml_1151.pkl', '_Chao_N36R18_20220120_qaml_1167.pkl', '_Chao_N36R18_20220120_qaml_1183.pkl', '_Chao_N36R18_20220120_qaml_1200.pkl', '_Chao_N36R18_20220120_qaml_1271.pkl']
        datasets_train = [11, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144,  150, 156, 162, 168, 288, 295, 301, 307, 313, 319, 325, 331, 337, 343, 349, 355, 361, 367, 373, 379, 385, 391, 397, 403,  409, 415, 421, 427, 433, 439, 445, 451, 457, 463, 469, 475, 481, 487, 493, 499, 505, 511, 517, 523, 529, 535, 541,  547, 553, 559, 565, 571, 577, 583, 589, 595, 601, 607, 613, 619, 625, 631, 637, 643, 649, 655, 661, 668, 674, 680, 686, 692,  698,  704,  710,  716, 722,  728,  734,  740,  746,  752,  758,  764,  770,  776,  782, 788,  794,  800,  806,  812, 818,  824,  846,  853,  859,  865, 871,  877,  883,  889,  895,  901,  907,  913,  919,  925,  931, 937,  949,  956, 962,  968,  974,  980, 1117, 1128, 1135, 1145, 1152, 1161, 1168, 1177, 1184, 1194, 1201, 1207, 1213, 1219, 1225, 1231, 1237, 1243, 1249, 1255, 1265, 1272, 1278, 1284, 1290, 1296, 1302, 1308, 1314, 1320, 1326, 1332, 1338, 1344, 1350, 1356, 1362, 1368, 1374, 1380, 1386, 1392]
        
        datasets_test = [12, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79, 85, 91, 97, 103, 109, 115, 121, 127, 133, 139, 145, 151, 157, 163, 169, 289, 296, 302, 308, 314, 320, 326, 332, 338, 344, 350, 356, 362, 368, 374, 380, 386, 392,  398,  404,  410,  416,  422,  428,  434,  440,  446,  452, 458,  464,  470,  476,  482,  488,  494,  500,  506,  512,  518, 524,  530,  536,  542,  548,  554,  560,  566,  572,  578,  584, 590,  596,  602,  608,  614,  620,  626,  632,  638,  644,  650, 656,  662,  669,  675,  681,  687,  693,  699,  705,  711,  717, 723,  729,  735,  741,  747,  753,  759,  765,  771,  777,  783, 789,  795,  801,  807,  813,  819,  825,  847,  854,  860,  866, 872,  878,  884,  890,  896,  902,  908,  914,  920,  926,  932, 938,  950,  957,  963,  969,  975,  981, 1118, 1129, 1136, 1146, 1153, 1162, 1169, 1178, 1185, 1195, 1202, 1208, 1214, 1220, 1226, 1232, 1238, 1244, 1250, 1256, 1266, 1273, 1279, 1285, 1291, 1297, 1303, 1309, 1315, 1321, 1327, 1333, 1339, 1345, 1351, 1357, 1363, 1369, 1375, 1381, 1387, 1393]

        save_path = '\\'.join(file_path_list[:-1])+'\\dataset_train\\'


    elif file_encode == 'encode_params_adv00_06_10q_256.pkl':
    #     datasets_qaml = [1513, 2120]
    #     data_name = ['_Chao_N36R18_20220120_qaml_1513.pkl', '_Chao_N36R18_20220120_qaml_2120.pkl']
        datasets_train = [1507, 1514, 1520, 1526, 1532, 1538, 1544, 1550, 1556, 1562, 1568, 1574, 1580, 1586, 1592, 1598, 1604, 1610, 1616, 1622, 1628, 1634, 1640, 1646, 1652, 1658, 1664, 1670, 1676, 1682, 1688, 1694, 1700, 1706, 1712, 1718, 1724, 1730, 1736, 1742, 1748, 1754, 1760, 1766, 1772, 1778, 1784, 1790, 1796, 1802, 1808, 1814, 1820, 1826, 1832, 1838, 1844, 1850, 1856, 1862, 1868, 1874, 1880, 1886, 1892, 1898, 1904, 1910, 1916, 1922, 1928, 1934, 1940, 1946, 1952, 1958, 1964, 1970, 1976, 1982, 1988, 1994, 2000, 2006, 2012, 2018, 2024, 2030, 2036, 2042, 2048, 2054, 2060, 2066, 2072, 2078, 2084,2090, 2096, 2102, 2108, 2114, 2121, 2127, 2133, 2139, 2145, 2151, 2157, 2163, 2169, 2175, 2181, 2187, 2193, 2199, 2205,2211, 2217, 2223, 2229, 2235, 2241, 2247, 2253, 2259, 2265, 2271, 2277, 2283, 2289, 2295, 2301, 2307, 2313]

        datasets_test = [1508, 1515, 1521, 1527, 1533, 1539, 1545, 1551, 1557, 1563, 1569, 1575, 1581, 1587, 1593, 1599, 1605, 1611, 1617, 1623, 1629, 1635, 1641, 1647, 1653, 1659, 1665, 1671, 1677, 1683, 1689, 1695, 1701, 1707, 1713, 1719, 1725, 1731, 1737, 1743, 1749, 1755, 1761, 1767, 1773, 1779, 1785, 1791, 1797, 1803, 1809, 1815, 1821, 1827, 1833, 1839, 1845, 1851, 1857, 1863, 1869, 1875, 1881, 1887, 1893, 1899, 1905, 1911, 1917, 1923, 1929, 1935, 1941, 1947, 1953, 1959, 1965, 1971, 1977, 1983, 1989, 1995, 2001, 2007, 2013, 2019, 2025, 2031, 2037, 2043, 2049, 2055, 2061, 2067, 2073, 2079, 2085, 2091, 2097, 2103, 2109, 2115, 2122, 2128, 2134, 2140, 2146, 2152, 2158, 2164, 2170, 2176, 2182, 2188, 2194, 2200, 2206, 2212, 2218, 2224, 2230, 2236, 2242, 2248, 2254, 2260, 2266, 2272, 2278, 2284, 2290, 2296, 2302, 2308, 2314]

        save_path = '\\'.join(file_path_list[:-1])+'\\dataset_train_adv\\'

    iter_idx = []
    results = {}
    for dataset_train, dataset_test in zip(datasets_train, datasets_test):
        data_train = dataLab()
        data_train.pickle_load(save_path + '_'+session+'_'+str(dataset_train) + '.pkl')
        data_test = dataLab()
        data_test.pickle_load(save_path + '_'+session+'_'+str(dataset_test) + '.pkl')
        para = analysis_loss(dataset_train, dataset_test,session, save_path,
                             file_encode=file_encode,
                             do_plot=False,
                             collect=True)
        iter_idx.append(data_train.parameters['iter_idx'])
        for ana_type, ana_info in para.items():
            if ana_type not in results:
                results[ana_type] = {}
            for type, type_info in ana_info.items():
                if type not in results[ana_type]:
                    results[ana_type][type] = {}
                for d in ['loss', 'accuracy', 'loss_std']:
                    if d not in results[ana_type][type]:
                        results[ana_type][type][d] = [type_info[d]]
                    else:
                        results[ana_type][type][d] += [type_info[d]]
    if do_plot:
        if 'adv' in file_encode:
            plot_train_dynamics_adv(iter_idx, results['origin'], results['adv'])
            # plot_train_dynamics(iter_idx, results['adv'], 'adv')
        else:
            plot_train_dynamics(iter_idx, results['full'], '')

def plot_train_dynamics(iter_idx, result, titleStr):
    loss_train = np.array(result['train']['loss'])
    loss_train_std = np.array(result['train']['loss_std'])
    accuracy_train = result['train']['accuracy']
    loss_test = np.array(result['test']['loss'])
    loss_test_std = np.array(result['test']['loss_std'])
    accuracy_test = result['test']['accuracy']

    plt.figure(figsize=[3.75, 2.4])
    plt.subplot(211)
    plot1D(iter_idx,
            loss_train,
            marker='',
            markersize=5,
            ls='-',
            lw=1.0,
            label='Training',
            color='C2',
            markeredgewidth=2,
            isLegend=True,
            fig=-1)
    plot1D(iter_idx,
            loss_test,
            marker='',
            markersize=5,
            ls='-',
            lw=1.0,
            label='Test',
            color='C3',
            yname='Loss',
            markeredgewidth=2, 
            isLegend=True,
            fig=-1)
    plt.fill_between(iter_idx,
            loss_train+loss_train_std,
           loss_train-loss_train_std,
            ls='-',
            linewidth=1,
            color='C2',
            alpha=0.5)
    plt.fill_between(iter_idx,
            loss_test+loss_test_std,
            loss_test-loss_test_std,
            ls='-',
            linewidth=1,
            color='C3', 
            alpha=0.5)
    plt.xticks([0, 30, 60, 90, 120, 150, 180], labels=[], size=9)
    plt.yticks([0.2, 0.5, 0.8], size=9)
    font = {'family':'Times New Roman', 'size': 9}
    plt.legend(frameon=False, prop=font)
    plt.ylabel('Loss', fontsize=10, fontproperties='Times New Roman')
    plt.subplot(212)
    plot1D(iter_idx,
            accuracy_train,
            marker='',
            markersize=5,
            ls='-',
            lw=2,
            label='Training',
            color='C2',
            markeredgewidth=2, 
            isLegend=False,
            fig=-1)
    plot1D(iter_idx,
            accuracy_test,
            marker='',
            markersize=5,
            ls='-',
            lw=2,
            label='Test',
            color='C3',
            xname='Epochs',
            yname='Accuracy',
            markeredgewidth=2,
            isLegend=False,
            fig=-1)
    plt.xticks([0, 30, 60, 90, 120, 150, 180], size=9)
    plt.yticks([0.2, 0.6, 1.0], size=9)
    plt.tight_layout()
    plt.subplots_adjust(left=0.14, top=0.95, right=0.97, bottom=0.18, hspace=0.12, wspace=0.0)

def plot_train_dynamics_adv(iter_idx, result_org, result_adv):
    train_loss_orig = np.array(result_org['train']['loss'])
    loss_train_std_org = np.array(result_org['train']['loss_std'])
    accuracy_train_org = result_org['train']['accuracy']
    test_loss_orig = np.array(result_org['test']['loss'])[:121]
    test_loss_std_orig = np.array(result_org['test']['loss_std'])[:121]
    test_acc_orig = result_org['test']['accuracy'][:121]

    train_loss_adv = np.array(result_adv['train']['loss'])
    loss_train_std_adv = np.array(result_adv['train']['loss_std'])
    accuracy_train_adv = result_adv['train']['accuracy']
    test_loss_adv = np.array(result_adv['test']['loss'])[:121]
    test_loss_std_adv = np.array(result_adv['test']['loss_std'])[:121]
    test_acc_adv = result_adv['test']['accuracy'][:121]
    iter_idx = iter_idx[:121]
    plt.figure(figsize=[3.75, 2.4])
    plt.subplot(211)
    plot1D(iter_idx,
            test_loss_orig,
            marker='',
            markersize=5,
            ls='-',
            lw=1,
            color='C0',
            markerfacecolor='none',
            xticks=[0, 30, 60, 90, 120],
            isLegend=True,
            label='Legitimate',
            fig=-1)
    plot1D(iter_idx,
            test_loss_adv,
            marker='',
            markersize=5,
            ls='-',
            lw=1,
            label='Adversarial',
            color='C1',
            yname='Loss',
            markerfacecolor='none', 
            xticks=[0, 30, 60, 90, 120],
            yticks=[0, 0.5, 1.0],
            isLegend=True,
            fig=-1)
    plt.fill_between(iter_idx,
            test_loss_orig + test_loss_std_orig,
            test_loss_orig - test_loss_std_orig,
            ls='-',
            linewidth=1,
            color='C0',
            alpha=0.5)
    plt.fill_between(iter_idx,
            test_loss_adv + test_loss_std_adv,
            test_loss_adv - test_loss_std_adv,
            ls='-',
            linewidth=1,
            color='C1', 
            alpha=0.5)
    plt.xticks([0, 30, 60, 90, 120], labels=[], size=9)
    plt.yticks([0.2, 0.5, 0.8], size=9)
    # plt.legend(frameon=False, fontsize=9)
    plt.subplot(212)
    plot1D(iter_idx,
            test_acc_orig,
            marker='',
            markersize=5,
            ls='-',
            lw=2,
            label='Legitimate',
            color='C0',
            markerfacecolor='none',
            isLegend=False,
            fig=-1)
    plot1D(iter_idx,
            test_acc_adv,
            marker='',
            markersize=5,
            ls='-',
            lw=2,
            label='Adversarial',
            color='C1',
            yname='Accuracy',
            xname='Epochs',
            markerfacecolor='none',
            isLegend=False,
            xticks=[0, 30, 60, 90, 120], 
            yticks=[0, 0.5, 1.0],
            fig=-1)
    plt.xticks([0, 30, 60, 90, 120], size=9)
    plt.yticks([0.2, 0.6, 1.0], size=9)
    # plt.tight_layout()
    plt.subplots_adjust(left=0.14, top=0.95, right=0.97, bottom=0.18, hspace=0.12, wspace=0.0)

def plot_test_probs(datasets_train=None, datasets_test=None, session=None, save_path=None, file_encode='encode_params_10q_256.pkl'):
    if datasets_train is None:
        datasets_train = [11, 487, 1392]
    if datasets_test is None:
        datasets_test = [12, 488, 1393]
    if session is None:
        session = 'Chao_N36R18_20220120_qaml'
    if save_path is None:
        save_path = '\\'.join(file_path_list[:-1])+'\\dataset_train\\'
    plt.figure(figsize=[3.75, 2.4])
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
            plt.text(-27, -0.05, r'$\langle\hat\sigma_z\rangle$', fontsize=10, rotation='vertical')
            plt.yticks([-0.6, 0.0, 0.6], fontsize=9)
            plt.text(3, -0.6, 'epoch 0', fontsize=9, fontproperties='Times New Roman')
            plt.legend(['0', '1'], fontsize=9, frameon=False, loc='upper center')
        elif ii == 1:
            plt.xlabel('Sample index', size=10, fontproperties='Times New Roman')
            plt.text(3, -0.6, 'epoch 60', fontsize=9, fontproperties='Times New Roman')
            plt.xticks([0, 25, 50], fontsize=9)
            plt.yticks([-0.6, 0.0, 0.6], fontsize=9, labels=[])
        else:
            plt.text(3, -0.6, 'epoch 180', fontsize=9, fontproperties='Times New Roman')
            plt.yticks([-0.6, 0.0, 0.6], fontsize=9, labels=[])
    plt.subplots_adjust(left=0.14, top=0.95, right=0.97, bottom=0.2, hspace=0.0, wspace=0.15)

def plot_loss_accuracy(ana_info, titleStr=''):
    colors = {'0': 'royalBlue', '1': 'crimson'}
    mks = {'train': 's', 'test': 's'}
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
                if digit in ['0', '1']:
                    plt.plot(xs_temp[digit],
                                1-2*digit_info['probs'],
                            color=colors[digit],
                            marker=mks[type],
                                ls='',
                                ms=6,
                                alpha=1.0, markerfacecolor='none')
    plt.hlines(y=0.0, xmin=0, xmax=51, linestyles='--', color='k')
    plt.xticks([0, 25, 50], fontsize=9)
    plt.xlim([0, 51])
    plt.ylim([-0.65, 0.65])

def cut_data(x, y, xlim=None):
    x = np.copy(x)
    y = np.copy(y)
    if xlim is not None:
        xlim = np.sort(xlim)
        y = y[(x >= xlim[0]) & (x <= xlim[1])]
        x = x[(x >= xlim[0]) & (x <= xlim[1])]
    return x, y

def plot1D(x=None,
           y=None,
           xlim=None,
           fig=None,
           size=[10, 6.18],
           titleName=None,
           xname=None,
           yname=None,
           xticks=True,
           yticks=True,
           xtick_size=8,
           ytick_size=8,
           color='RoyalBlue',
           ls='-',
           lw=2,
           marker='o',
           markersize=6,
           markeredgewidth=2, 
           markerfacecolor='none',
           isLegend=True,
           increase_text_size=0,
           **kwargs):
    if fig == -1:
        pass
    else:
        fig = plt.figure(fig, figsize=size)
    if y is not None:
        x1, y1 = cut_data(x, y, xlim)
        plt.plot(x1,
                 y1,
                 color=color,
                 ls=ls,
                 lw=lw,
                 marker=marker,
                 markersize=markersize,
                 markerfacecolor=markerfacecolor,
                 markeredgewidth=markeredgewidth,
                 **kwargs)
    if titleName is not None:
        plt.title(titleName, size=10 + increase_text_size)
    if xname is not None:
        plt.xlabel(xname, size=10 + increase_text_size, fontproperties='Times New Roman')
    if yname is not None:
        plt.ylabel(yname, size=10 + increase_text_size, fontproperties='Times New Roman')
    if xticks is None:
        plt.xticks([])
    else:
        plt.xticks(size=xtick_size + increase_text_size)
    if yticks is None:
        plt.yticks([])
    else:
        plt.yticks(size=ytick_size + increase_text_size)
    if isLegend == True:
        font = {'family':'Times New Roman', 'size': 9}
        plt.legend(loc=0, prop=font, frameon=False)
    plt.grid(False)
    plt.tight_layout()

def digit_adv():
    encode_path = '\\'.join(file_path_list[:-1])+'\\encode_MNIST\\encode_params_adv00_06_10q_256.pkl'
    with open(encode_path, 'rb') as f:
        encode_params = pickle.load(f)
    # x_adv_train_06 = encode_params['x_adv_train_06']
    x_train = encode_params['x_train']
    digit0_data = np.reshape((x_train[505].T.flatten())[:256], [16, 16])-np.pi
    digit1_data = np.reshape((x_train[513].T.flatten())[:256], [16, 16])-np.pi
    # digit1_data = np.reshape((x_adv_train_06[1].T.flatten())[:256], [16, 16])-np.pi
    lw=2
    markersize=10
    fig = plt.figure(figsize=[2.3, 2.3])
    ax1 = plt.gca()
    ax1.imshow(digit1_data, cmap='Greys')
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
    # digit0_data = np.reshape((x_adv_train_06[0].T.flatten())[:256], [16, 16])-np.pi
    fig = plt.figure(figsize=[2.3, 2.3])
    ax1 = plt.gca()
    ax1.imshow(digit0_data, cmap='Greys')
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
def run():
    # a
    analysis_train_dynamics()

    #b
    analysis_train_dynamics('encode_params_adv00_06_10q_256.pkl')

    #c
    digit_adv()

    #d
    plot_test_probs()
