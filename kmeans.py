import numpy


class KMeans:
    def __init__(self, n_cluster: int, init_pp: bool = True, max_iter: int = 500, tolerance: float=1e-3):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.init_pp = init_pp
        pass

    def fit(self, data: numpy.ndarray):
        self.centroid = self._init_centroid(data)
        pass

    def _init_centroid(self, data: numpy.ndarray):
        if self.init_pp:
            centroid = [int(numpy.random.uniform()*len(data))]
            for _ in range(1, self.n_cluster):
                dist = numpy.array([min([(x - data[c]) * (x - data[c])] for c in centroid)
                                    for i, x in enumerate(data) if i not in centroid])
                dist = dist / dist.sum()
                cumdist = numpy.cumsum(dist)

                prob = numpy.random.rand()
                for i, c in enumerate(cumdist):
                    if prob > c:
                        centroid.append(i)
                        break
            centroid = numpy.array([data[c] for c in centroid])
        else:
            idx = numpy.random.choice(range(len(data)), size=(self.n_cluster))
            centroid = data[idx]
        return centroid

    def _calc_distance(self, data: numpy.ndarray):
        pass

    def _assign_cluster(self, distance: numpy.ndarray):
        pass

    def _update_centroid(self, data: numpy.ndarray, cluster: numpy.ndarray):
        pass
