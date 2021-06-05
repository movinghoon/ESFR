import numpy as np
import pickle as pkl


def numpy_one_hot(x, depth=5):
    y = np.zeros((x.size, depth))
    y[np.arange(x.size), x] = 1
    return y


class BaseEmbeddingFSLLoader(object):
    def __init__(self, file_path, num_qry_samples=None, way=5, shot=5, meta_batch_size=2,
                 query_samples_per_class=15):
        # Task Information
        self.way = way
        self.shot = shot
        self.meta_batch_size = meta_batch_size
        self.num_qry_samples = [query_samples_per_class] * way if num_qry_samples is None else list(num_qry_samples)
        self.query_samples_per_class = query_samples_per_class

        # Load Data
        with open(file_path, 'rb') as f:
            data = pkl.load(f)
        self.data = data['data']
        self.num_classes = len(self.data.keys())    # self.data.shape[0]
        self.feat_dim = self.data[list(self.data.keys())[0]].shape[-1]

        # # Note that labels are always fixed
        self.label = self._get_label()

    def _get_label(self):
        supp_lbs = np.repeat(np.repeat(np.eye(self.way), self.shot, axis=0).reshape((1, self.way, self.shot, self.way)),
                             self.meta_batch_size, axis=0).astype(np.float32)
        qry_lbs = []
        for i in range(self.way):
            qry_lbs += [i] * self.num_qry_samples[i]
        qry_lbs = np.repeat(numpy_one_hot(np.array(qry_lbs), depth=5).reshape((1, self.way, -1, self.way)),
                            self.meta_batch_size, axis=0).astype(np.float32)
        return np.concatenate([supp_lbs, qry_lbs], axis=2)

    def sample(self):
        supp = np.zeros((self.meta_batch_size, self.way, self.shot, self.feat_dim), dtype=np.float32)
        qry = np.zeros((self.meta_batch_size, self.query_samples_per_class * self.way, self.feat_dim), dtype=np.float32)
        for i in range(self.meta_batch_size):
            temp = []
            idx_ways = np.random.choice(self.num_classes, size=self.way, replace=False)
            for j in range(self.way):
                key = list(self.data.keys())[idx_ways[j]]
                idx = np.random.choice(self.data[key].shape[0], size=self.shot + self.num_qry_samples[j], replace=False)
                data = self.data[key][idx]
                supp[i, j] = data[:self.shot]   # self.data[idx_ways[j], idx[:self.shot]]
                temp.append(data[self.shot:])
            qry[i] = np.concatenate(temp, axis=0)
        data = np.concatenate([supp, qry.reshape((self.meta_batch_size, self.way, -1, self.feat_dim))], axis=2)
        return data, self.label

    def __iter__(self):
        return self

    def __next__(self):
        return self.sample()


class ImbalanceEmbeddingFSLLoader(BaseEmbeddingFSLLoader):
    def __init__(self, architecture='WidResNet', dset='mini', data_type='test',
                 num_qry_samples=None, way=5, shot=5, meta_batch_size=2, query_samples_per_class=15):
        dset = dset.upper()
        file_path = './embeddings/' + dset + '_' + architecture + '_' + data_type
        file_path += '.pkl'
        super(ImbalanceEmbeddingFSLLoader, self).__init__(file_path, num_qry_samples, way, shot, meta_batch_size,
                                                          query_samples_per_class)


class EmbeddingFSLLoader(ImbalanceEmbeddingFSLLoader):
    def __init__(self, architecture='WidResNet', dset='mini', data_type='test',
                 way=5, shot=5, meta_batch_size=2, query_samples_per_class=15):
        super(EmbeddingFSLLoader, self).__init__(architecture, dset, data_type,
                                                 None, way, shot, meta_batch_size, query_samples_per_class)

    def _get_label(self):
        label = np.repeat(np.eye(self.way), self.query_samples_per_class + self.shot, axis=0)
        label = label.reshape((1, self.way, self.query_samples_per_class + self.shot, self.way))
        return np.repeat(label, self.meta_batch_size, axis=0).astype(np.float32)

    def sample(self):
        num_samples_per_class = self.query_samples_per_class + self.shot
        data = np.zeros((self.meta_batch_size, self.way, num_samples_per_class, self.feat_dim), dtype=np.float32)
        for i in range(self.meta_batch_size):
            idx_ways = np.random.choice(self.num_classes, size=self.way, replace=False)
            for j in range(self.way):
                key = list(self.data.keys())[idx_ways[j]]
                idx_samples = np.random.choice(self.data[key].shape[0], size=num_samples_per_class, replace=False)
                data[i, j] = self.data[key][idx_samples]
        return data, self.label