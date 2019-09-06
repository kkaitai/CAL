import numpy as np


class Q_Agent():
    def __init__(self, GridSize, N_ex, l_rate):
        self.N_ex = N_ex
        self.StateSize = GridSize ** 2 * 4
        self.Q = np.zeros([self.StateSize, N_ex])
        self.alpha = np.zeros([self.StateSize, N_ex])
        self.epsilon = np.zeros([GridSize ** 2, N_ex])
        self.l_rate = l_rate

    def action(self, s):
        columns = np.arange(self.N_ex)
        a = np.argmax(self.Q[np.array([s * 4, s * 4 + 1, s * 4 + 2, s * 4 + 3]), np.tile(columns, (4, 1))], axis=0)
        self.epsilon[s, columns] += 1
        a = np.vectorize(np.random.choice, otypes=[int], signature='(m),(m)->()')(
            a=np.array([a, np.random.randint(4, size=self.N_ex)]).T,
            p=np.array([1 - 1 / np.sqrt(self.epsilon[s, columns]), 1 / np.sqrt(self.epsilon[s, columns])]).T)
        return a

    def update(self, s, a, snew, r, DF):
        columns = np.arange(self.N_ex)
        self.alpha[s * 4 + a, columns] += 1

        targetQ = np.max(
            self.Q[np.array([snew * 4, snew * 4 + 1, snew * 4 + 2, snew * 4 + 3]), np.tile(columns, (4, 1))], axis=0)

        self.Q[s * 4 + a, columns] += 1 / (self.alpha[s * 4 + a, columns] ** self.l_rate) * (
        r + DF * targetQ - self.Q[s * 4 + a, columns])


class DoubleQ_Agent():
    def __init__(self, GridSize, N_ex, l_rate):
        self.K = 2
        self.N_ex = N_ex
        self.StateSize = GridSize ** 2 * 4
        self.Q = np.zeros([self.StateSize * self.K, N_ex])
        self.alpha = np.zeros([self.StateSize * self.K, N_ex])
        self.epsilon = np.zeros([GridSize ** 2, N_ex])
        self.l_rate = l_rate

    def action(self, s):
        columns = np.arange(self.N_ex)
        a = np.argmax(np.sum([self.Q[np.array(
            [s * 4 + self.StateSize * k, s * 4 + 1 + self.StateSize * k, s * 4 + 2 + self.StateSize * k,
             s * 4 + 3 + self.StateSize * k]), np.tile(columns, (4, 1))] for k
                              in range(self.K)], axis=0), axis=0)
        self.epsilon[s, columns] += 1
        a = np.vectorize(np.random.choice, otypes=[int], signature='(m),(m)->()')(
            a=np.array([a, np.random.randint(4, size=self.N_ex)]).T,
            p=np.array([1 - 1 / np.sqrt(self.epsilon[s, columns]), 1 / np.sqrt(self.epsilon[s, columns])]).T)
        return a

    def update(self, s, a, snew, r, DF):
        columns = np.arange(self.N_ex)
        update = np.random.randint(self.K, size=self.N_ex)

        nonupdate = (update + 1) % 2
        self.alpha[s * 4 + a + update * self.StateSize, columns] += 1

        argQ = self.Q[np.array([snew * 4 + update * self.StateSize, snew * 4 + 1 + update * self.StateSize,
                                snew * 4 + 2 + update * self.StateSize,
                                snew * 4 + 3 + update * self.StateSize]), np.tile(columns, (4, 1))]

        aa = np.argmax(argQ, axis=0)

        targetQ = self.Q[snew * 4 + aa + nonupdate * self.StateSize, columns]

        self.Q[s * 4 + a + update * self.StateSize, columns] += 1 / (
        self.alpha[s * 4 + a + update * self.StateSize, columns] ** self.l_rate) * (r + DF * targetQ - self.Q[
            s * 4 + a + update * self.StateSize, columns])


class WeightedQ_Agent():
    def __init__(self, GridSize, N_ex, l_rate, weights):
        self.K = len(weights)
        self.N_ex = N_ex
        self.StateSize = GridSize ** 2 * 4
        self.Q = np.zeros([self.StateSize * self.K, N_ex])
        self.alpha = np.zeros([self.StateSize * self.K, N_ex])
        self.epsilon = np.zeros([GridSize ** 2, N_ex])
        self.l_rate = l_rate
        self.weights = weights

    def action(self, s):
        columns = np.arange(self.N_ex)
        a = np.argmax(np.sum([self.Q[np.array(
            [s * 4 + self.StateSize * k, s * 4 + 1 + self.StateSize * k, s * 4 + 2 + self.StateSize * k,
             s * 4 + 3 + self.StateSize * k]), np.tile(columns, (4, 1))] for k
                              in range(self.K)], axis=0), axis=0)
        self.epsilon[s, columns] += 1
        a = np.vectorize(np.random.choice, otypes=[int], signature='(m),(m)->()')(
            a=np.array([a, np.random.randint(4, size=self.N_ex)]).T,
            p=np.array([1 - 1 / np.sqrt(self.epsilon[s, columns]), 1 / np.sqrt(self.epsilon[s, columns])]).T)
        return a

    def update(self, s, a, snew, r, DF):
        columns = np.arange(self.N_ex)
        update = np.random.randint(self.K, size=self.N_ex)

        self.alpha[s * 4 + a + update * self.StateSize, columns] += 1

        argQ = np.sum([self.Q[np.array(
            [snew * 4 + k * self.StateSize, snew * 4 + 1 + k * self.StateSize, snew * 4 + 2 + k * self.StateSize,
             snew * 4 + 3 + k * self.StateSize]), np.tile(columns, (4, 1))] * self.weights[update - 1, k] for k in
                       range(self.K)], axis=0)
        aa = np.argmax(argQ, axis=0)
        targetQ = np.sum(
            [self.Q[snew * 4 + aa + k * self.StateSize, columns] * self.weights[update, k] for k in range(self.K)],
            axis=0)

        self.Q[s * 4 + a + update * self.StateSize, columns] += 1 / (
        self.alpha[s * 4 + a + update * self.StateSize, columns] ** self.l_rate) * (r + DF * targetQ - self.Q[
            s * 4 + a + update * self.StateSize, columns])


class WeightedPlusQ_Agent():
    def __init__(self, GridSize, N_ex, l_rate, c):
        self.K = 2
        self.N_ex = N_ex
        self.StateSize = GridSize ** 2 * 4
        self.Q = np.zeros([self.StateSize * self.K, N_ex])
        self.alpha = np.zeros([self.StateSize * self.K, N_ex])
        self.epsilon = np.zeros([GridSize ** 2, N_ex])
        self.l_rate = l_rate
        self.c = c

    def action(self, s):
        columns = np.arange(self.N_ex)
        a = np.argmax(np.sum([self.Q[np.array(
            [s * 4 + self.StateSize * k, s * 4 + 1 + self.StateSize * k, s * 4 + 2 + self.StateSize * k,
             s * 4 + 3 + self.StateSize * k]), np.tile(columns, (4, 1))] for k
                              in range(self.K)], axis=0), axis=0)
        self.epsilon[s, columns] += 1
        a = np.vectorize(np.random.choice, otypes=[int], signature='(m),(m)->()')(
            a=np.array([a, np.random.randint(4, size=self.N_ex)]).T,
            p=np.array([1 - 1 / np.sqrt(self.epsilon[s, columns]), 1 / np.sqrt(self.epsilon[s, columns])]).T)
        return a

    def update(self, s, a, snew, r, DF):
        columns = np.arange(self.N_ex)
        update = np.random.randint(self.K, size=self.N_ex)

        nonupdate = (update + 1) % 2
        self.alpha[s * 4 + a + update * self.StateSize, columns] += 1

        argQ = self.Q[np.array([snew * 4 + update * self.StateSize, snew * 4 + 1 + update * self.StateSize,
                                snew * 4 + 2 + update * self.StateSize,
                                snew * 4 + 3 + update * self.StateSize]), np.tile(columns, (4, 1))]
        aa = np.argmax(argQ, axis=0)
        aL = np.argmin(argQ, axis=0)

        Qaa = self.Q[snew * 4 + aa + nonupdate * self.StateSize, columns]
        QL = self.Q[snew * 4 + aL + nonupdate * self.StateSize, columns]
        weights = np.abs(Qaa - QL)
        weights = weights / (self.c + weights)

        targetQ = weights * self.Q[snew * 4 + aa + update * self.StateSize, columns] + (1 - weights) * Qaa

        self.Q[s * 4 + a + update * self.StateSize, columns] += 1 / (
        self.alpha[s * 4 + a + update * self.StateSize, columns] ** self.l_rate) * (r + DF * targetQ - self.Q[
            s * 4 + a + update * self.StateSize, columns])


class AccurateQ_Agent():
    def __init__(self, GridSize, N_ex, l_rate):
        self.N_ex = N_ex
        self.StateSize = GridSize ** 2 * 4
        self.Q = np.zeros([self.StateSize, N_ex])
        self.M = np.zeros([self.StateSize, N_ex])
        self.alpha = np.zeros([self.StateSize, N_ex])
        self.epsilon = np.zeros([GridSize ** 2, N_ex])
        self.l_rate = l_rate

    def action(self, s):
        columns = np.arange(self.N_ex)
        a = np.argmax(self.Q[np.array([s * 4, s * 4 + 1, s * 4 + 2, s * 4 + 3]), np.tile(columns, (4, 1))], axis=0)
        self.epsilon[s, columns] += 1
        a = np.vectorize(np.random.choice, otypes=[int], signature='(m),(m)->()')(
            a=np.array([a, np.random.randint(4, size=self.N_ex)]).T,
            p=np.array([1 - 1 / np.sqrt(self.epsilon[s, columns]), 1 / np.sqrt(self.epsilon[s, columns])]).T)
        return a

    def update(self, s, a, snew, r, DF):
        columns = np.arange(self.N_ex)

        self.alpha[s * 4 + a, columns] += 1
        targetQ = np.max(
            self.Q[np.array([snew * 4, snew * 4 + 1, snew * 4 + 2, snew * 4 + 3]), np.tile(columns, (4, 1))], axis=0)
        delta = targetQ - self.M[s * 4 + a, columns]

        self.M[s * 4 + a, columns] = targetQ
        self.Q[s * 4 + a, columns] += 1 / (self.alpha[s * 4 + a, columns] ** self.l_rate) * (
        r + DF * targetQ - self.Q[s * 4 + a, columns]) + DF * (
        1 - 1 / (self.alpha[s * 4 + a, columns] ** self.l_rate)) * delta


class SpeedyQ_Agent():
    def __init__(self, GridSize, N_ex, l_rate):
        self.N_ex = N_ex
        self.StateSize = GridSize ** 2 * 4
        self.Q = np.zeros([self.StateSize, N_ex])
        self.Q_minus = np.zeros([self.StateSize, N_ex])
        self.alpha = np.zeros([self.StateSize, N_ex])
        self.epsilon = np.zeros([GridSize ** 2, N_ex])
        self.l_rate = l_rate

    def action(self, s):
        columns = np.arange(self.N_ex)
        a = np.argmax(self.Q[np.array([s * 4, s * 4 + 1, s * 4 + 2, s * 4 + 3]), np.tile(columns, (4, 1))], axis=0)
        self.epsilon[s, columns] += 1
        a = np.vectorize(np.random.choice, otypes=[int], signature='(m),(m)->()')(
            a=np.array([a, np.random.randint(4, size=self.N_ex)]).T,
            p=np.array([1 - 1 / np.sqrt(self.epsilon[s, columns]), 1 / np.sqrt(self.epsilon[s, columns])]).T)
        return a

    def update(self, s, a, snew, r, DF):
        columns = np.arange(self.N_ex)

        self.alpha[s * 4 + a, columns] += 1
        target_minus = np.max(
            self.Q_minus[np.array([snew * 4, snew * 4 + 1, snew * 4 + 2, snew * 4 + 3]), np.tile(columns, (4, 1))],
            axis=0)
        targetQ = np.max(
            self.Q[np.array([snew * 4, snew * 4 + 1, snew * 4 + 2, snew * 4 + 3]), np.tile(columns, (4, 1))], axis=0)
        delta = DF * (1 - 2 / (self.alpha[s * 4 + a, columns] ** self.l_rate)) * (targetQ - target_minus)
        self.Q_minus[:] = self.Q[:]
        self.Q[s * 4 + a, columns] += 1 / (self.alpha[s * 4 + a, columns] ** self.l_rate) * (
        r + DF * targetQ - self.Q[s * 4 + a, columns]) + delta


class NewAccurateQ_Agent():
    def __init__(self, GridSize, N_ex, l_rate):
        self.N_ex = N_ex
        self.StateSize = GridSize ** 2 * 4
        self.Q = np.zeros([self.StateSize, N_ex])
        self.E = np.zeros([self.StateSize, N_ex])
        self.M = np.zeros([self.StateSize, N_ex])
        self.diff = np.zeros([self.StateSize, N_ex])
        self.alpha = np.zeros([self.StateSize, N_ex])
        self.epsilon = np.zeros([GridSize ** 2, N_ex])
        self.l_rate = l_rate

    def action(self, s):
        columns = np.arange(self.N_ex)
        a = np.argmax(self.Q[np.array([s * 4, s * 4 + 1, s * 4 + 2, s * 4 + 3]), np.tile(columns, (4, 1))], axis=0)
        self.epsilon[s, columns] += 1
        a = np.vectorize(np.random.choice, otypes=[int], signature='(m),(m)->()')(
            a=np.array([a, np.random.randint(4, size=self.N_ex)]).T,
            p=np.array([1 - 1 / np.sqrt(self.epsilon[s, columns]), 1 / np.sqrt(self.epsilon[s, columns])]).T)
        return a

    def update(self, s, a, snew, r, DF):
        columns = np.arange(self.N_ex)

        # beta =  np.exp(-0.05*self.alpha[s * 4 + a , columns])/(np.exp(-0.05*self.alpha[s * 4 + a , columns])+np.exp(-10))

        beta = np.exp(0.02 * (self.diff[s * 4 + a, columns] - self.alpha[s * 4 + a, columns])) \
               / (np.exp(0.02 * (self.diff[s * 4 + a, columns] - self.alpha[s * 4 + a, columns])) + np.exp(-10))
        E_term = self.E[s * 4 + a, columns] * (1 - beta) + self.M[s * 4 + a, columns] * beta
        delta = self.Q[s * 4 + a, columns] - DF * E_term

        self.alpha[s * 4 + a, columns] += 1
        om_alpha = (1 - 1 / (self.alpha[s * 4 + a, columns] ** self.l_rate))
        # beta =  np.exp(-0.05*self.alpha[s * 4 + a , columns])/(np.exp(-0.05*self.alpha[s * 4 + a , columns])+np.exp(-10))

        beta = np.exp(0.02 * (self.diff[s * 4 + a, columns] - self.alpha[s * 4 + a, columns])) \
               / (np.exp(0.02 * (self.diff[s * 4 + a, columns] - self.alpha[s * 4 + a, columns])) + np.exp(-10))
        targetQ = np.max(
            self.Q[np.array([snew * 4, snew * 4 + 1, snew * 4 + 2, snew * 4 + 3]), np.tile(columns, (4, 1))], axis=0)

        self.E[s * 4 + a, columns] = om_alpha * self.E[s * 4 + a, columns] + (1 - om_alpha) * targetQ
        self.M[s * 4 + a, columns] = targetQ
        E_term = self.E[s * 4 + a, columns] * (1 - beta) + self.M[s * 4 + a, columns] * beta
        Qn = om_alpha * delta + DF * E_term + (1 - om_alpha) * r
        self.diff[s * 4 + a, columns] = np.abs(Qn - self.Q[s * 4 + a, columns])
        self.Q[s * 4 + a, columns] = Qn

