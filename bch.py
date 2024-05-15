# import sympy
# from sympy import Matrix, symbols, Poly
import numpy as np

'''
仅实现了 p = 2 的 BCH 编解码
'''

# 本原多项式
P = {
    # key: "p.m"
    '2.3': [1,0,1,1],
    '2.4': [1,0,0,1,1],
    '2.5': [1,0,0,1,0,1],
    '2.6': [1,0,0,0,0,1,1]
}
# 生成多项式
G = {
    # key: 'n.k.t'
    '7.4.1': 0o13,
    '7.1.2': 0o177,
    '15.11.1': 0o23,
    '15.7.2': 0o721,
    '15.5.3': 0o2467,
    '31.21.2': 0o3551,
    '63.36.5': 0o1033500423
}

def bin2dec(t):
    n = t.shape[-1]
    w = np.arange(n)[::-1]
    w = 2 ** w
    t = t * w
    return np.sum(t, -1)

class EncoderDecoder:
    '''
    BCH 编码和解码，请参考 
    1. https://www.ece.unb.ca/tervo/ece4253/bch.shtml
    2. https://blog.csdn.net/qq_39731597/article/details/125738795
    获取更多信息。
    '''
    def __init__(self, m, k, t):
        '''
        validilities of (m, k, t) is ensured by the caller
        '''
        self.m = m
        self.k = k
        self.t = t
        self.n = 2 ** m - 1
        key = f'{self.n}.{k}.{t}'
        if key in G:
            self.g_bin = np.array([int(c) for c in bin(G[key])[2:]])
        else:
            self.g_bin = 0
        
        self.locator = Locator(m, k, t)

    def encode(self, i):
        '''
        i: (k, ) message with k bits, if msg length < k, pad 0
        使用系统编码，即编码后码字为：消息码+校验码，相应地解码则直接提取前 k bits
        返回编码后的消息, n bits。
        可通过 Locator 类定位错误。
        c(x) = i(x) * x^{n-k}
        c(x) = c(x) + c(x) mod g(x)
        '''
        i = np.append(i, np.zeros((self.n - self.k), dtype=np.int32))
        m = self.g_bin.shape[0]
        r = i[:m].copy()               # 余式
        for k in range(self.n - m + 1):
            if r[0] == 1:
                r ^= self.g_bin
            if k < self.n - m:
                r = np.append(r[1:], i[k+m:k+m+1])
        i[-m+1:] |= r[1:]
        return i
    
    def encode2(self, i):
        '''
        i: (k, ) message with k bits, if msg length < k, pad 0
        非系统编码: c(x) = i(x) * g(x)  多项式乘法
        解码: i(x) = c(x) / g(x)        多项式除法
        '''
        m = self.g_bin.shape[0]
        r = np.zeros(self.n, dtype=np.int32)
        for k in range(self.n - m + 1):
            r[k:k+m] += i[k] * self.g_bin
        return r % 2
    
    def decode(self, r):
        '''
        系统解码
        r: (n, ) received message with n bits
        '''
        r = self.correct(r)
        if r is None: return None
        return r[:self.k]
    
    def decode2(self, r):
        '''
        非系统解码（非系统编码不能通过 Locator 类定位错误，好像只能定位 t-1 个错误，可能是纠错实现算法有问题）
        r: (n, ) received message with n bits
        '''
        # r = self.correct(r)
        m = self.g_bin.shape[0]
        q = []
        d = r[:m].copy()
        for i in range(self.n - m + 1):
            q.append(d[0])
            if d[0] == 1:
                d ^= self.g_bin
            if i < self.n - m:
                d = np.append(d[1:], r[i+m:i+m+1])
        assert np.all(d == 0), f'余式不为 0, 解码错误'
        return np.array(q)

    def correct(self, r):
        ls = self.locator.locate(r)
        if ls is None: return None
        for l in ls:
            r[l] ^= 1
        return r
    
    def _gen_poly(self):
        '''
        计算得到生成多项式
        选择 a, a^2, ..., a^{2t} 的所有不重复的最小多项式，相乘得生成多项式
        将 x^n + 1 因式分解，得到所有的最小多项式
        a^i 的最小多项式以 a^i 为根
        生成多项式写成：
        G(x) = p(x)p_3(x) ... p_{2t-1}(x)
        其中 p(x) 是本原多项式，而 p_3(x^3),...,p_{2t-1}(x^{2t-1}) 能被 p(x) 整除
        这里 p_i(x) 的根就是 a^i 所以 p_i(x) 也就是 a^i 的最小多项式
        '''
        pass


class Locator:
    '''
    GF 域中元素列表，其中 GF 元素多项式通过单项 a^i mod P(a) 得到

    id      单项a^i     GF元素多项式        bin     dec
    -1      0               0             000     0
    0       a^0             a^0           001     1
    1       a^1             a^1           010     2
    ...
    2^m-2   a^{2^m-2}
    '''
    def __init__(self, m, k, t):
        '''
        validilities of (m, k, t) is ensured by the caller
        '''
        self.m = m
        self.k = k
        self.t = t
        self.n = 2 ** m - 1
        # self.x, self.a = symbols('x a')
        # self.xs = Matrix([[self.x ** i for i in range(m, -1, -1)]]) # 行向量，阶次左高右低
        self.p_bin = np.array(P[f'2.{m}'], dtype=np.uint8)     # 本原多项式二进制表示
        # p2 = Matrix(self.p_bin)                    # 列向量
        # self.p_poly = self.xs * p2                            # 本原多项式

        # 校验矩阵，数值型表示 (2t, n)，每个值表示 a^i 的幂次 i，幂次范围为 [0, n-1]
        self.H_num = np.array([[(j * (i+1) % self.n) for j in range(self.n - 1, -1, -1)] for i in range(2 * t)]) % self.n
        # print('H_num', self.H_num)
        # # 校验矩阵，多项式表示 (2t, n)
        # self.H_poly = Matrix([[self.a ** (j * (i+1) % self.n) for j in range(self.n - 1, -1, -1)] for i in range(2 * t)])

        # 从 GF 域中二进制表示到 id, 注意 id 为 -1 即单项为 0 的二进制表示：全 0，这是 GF 域中的加法单位元
        # 即 多项式的指数所表示的向量 -> a^i 的指数 i
        self.gf2id = np.zeros((self.n + 1, ), dtype=np.int32)
        # GF 域元素
        self.gf_bin = []    # 二进制表示，从 id 到 bin 的映射
        # self.gf_poly = []   # 多项式表示
        for i in range(-1, self.n):
            gf_bin = self.gf(i)
            gf_bin = gf_bin[1:]
            # print(i, gf_bin)
            gf_dec = self.bin2dec(gf_bin)
            self.gf2id[gf_dec] = i
            self.gf_bin.append(gf_bin)
            # self.gf_poly.append((self.xs * Matrix(gf_bin)))
        # gf_bin 的 index 表示 a^i 的指数 i，范围从 0 ~ n，对应的值向量表示多项式的指数
        self.gf_bin = np.stack(self.gf_bin[1:] + self.gf_bin[:1], axis=0)         # (n+1, m)
        # print(bin2dec(self.gf_bin))
    
    def bin2dec(self, b):
        '''
        二进制转十进制
        '''
        r = 0
        for i in b:
            r = (r << 1) + i
        return r
    
    def gf(self, i):
        '''
        将一个数根据本原多项式映射到 GF 域中，使用二进制表示, np.ndarray 表示，长度为 m+1, 
        第一个元素必定为 0, 从左往右对应位数从高到低。
        i: 范围 -1~n-1  一共 n+1 个元素，表示 a^i, 注意当 i=-1 时，表示 0 （加法单位元）
        p: 本原多项式，一个 np.ndarray 数组表示，长度为 m+1 , 表示阶数为 m 的多项式
        '''
        if i == -1:
            return np.zeros((self.m + 1,), dtype=np.uint8)
        # 构造 0...010...0
        d = np.zeros((self.n,), dtype=np.uint8)
        d[-i-1] = 1

        r = d[:self.m+1]
        
        for j in range(self.n - self.m):
            if r[0] == 1:
                r ^= self.p_bin
            if j < self.n - self.m - 1:
                r = np.concatenate((r[1:], d[j+self.m+1:j+self.m+2]))
        return r
    
    def locate(self, r):
        '''
        Berlekamp迭代算法。请参考 https://blog.csdn.net/njw21377128/article/details/133991325
        获取更多信息。
        r: a list of 0/1s, np.ndarray 类型，(n, )
        返回: 错误位置，从右往左，下标从 0 开始计数
        '''
        # 综合值    (2t, n)
        # S = self.H_num * r[np.newaxis, :]   # 这种计算错了，例如 a^i * 0 = 0，但是这里存储的是指数，这种情况指数为 0，表示 a^0=1，不等于 0，故错误
        mask = r == 1
        # print(self.H_num[8])
        S = self.H_num[:, mask]    # (2t, Nr) Nr 表示 r 中 1 元素数量
        S = self.gf_bin[S.reshape(-1), :].reshape(2*self.t, -1, self.m)
        S_b = np.sum(S, axis=1) % 2       # 记住需要 mod 2，  (2t, m)，综合值的二进制表示

        # 二进制转十进制，范围为 1~n，注意与单项式不是顺序对应，需要通过 self.gf2id 得到单项式
        # (2t, ) 综合值的数值表示，例如 a^0，那么这里存储 1，a^1 这里存储 2，0/a^63 这里存储 0
        S = np.sum(S_b * (2 ** np.arange(self.m)[::-1]), axis=-1, dtype=np.int32)     
        # print('综合值', S)
        # 得到综合值的单项式的指数，即单项式 a^i 的指数 i，如果是 0/a^63，那么指数设置为 -1
        # print('综合值2', self.gf2id[S]) 
        # t 阶多项式，上高下低，当前获得的错误多项式 sigma(x)，此多项式的根就是错误位置
        # i 行对应单项式 x^i 的系数 a^j，列表示单项式系数的阶数，若单项式不存在，使用 -1 表示
        sx = np.ones((self.t + 1, ), dtype=np.int32) * -1
        # \sum_i=0^t sx[i]*x^{t-i}，其中 sx[i] 表示 a^j 的指数 j，特殊地，当 j=-1 时，表示 a^j=0
        sx[-1] = 0      # 最低单项式为 1，即 s(x) = a^0·x^0 = 1
        # u: 迭代步数
        # sx: 第 u 次迭代计算出的错误多项式 sigma(x)
        # du: 误差，单项式的 dec 表示，例如 1 为 a^0，0 为 0，2 为 a^1
        # lu: sx 的阶数
        # rho: 
        #       u   sx      du      lu  u-lu    rho
        vs = [[-1,  sx,     1,      0,  -1,     None],
              [0,   sx,     S[0],   0,  0,      -1]]
        # ====== 解释一下以上 vs 初值设置地含义 ======
        # \sigma0 = 1
        # \sigma1 = beta1+beta2+...+betav
        # 牛顿恒等式
        # S1 + \sigma1 = 0 => \sigma1=S1
        # 最低次数多项式 \sigma(x)=\sigma0 + \sigma1*x + \sigma2*x2 + ...
        # \sigma(-1)(x) 表示第 -1 次迭代时的 \sigma(x)，为 1，即 s(x)=a^0*x^0=1

        max_u_lu = 0
        max_rho = 0
        for u in range(2 * self.t): # 迭代 2t 次
            v = vs[-1]
            if v[2] != 0:   # du 是否等于 0
                rho = v[-1]
                du = self.gf2id[v[2]]   # 转为 a^i 单项式的指数 i，注意 0 的指数为 -1
                dp = self.gf2id[vs[rho+1][2]]   # d_rho 的单项式的指数，注意 0 的指数为 -1
                dp_rec = self.n - dp    # dp 值不可能为 -1，因为 -1 表示单项式 0 的指数，而单项式 0 没有倒数
                # du2 表示 du*dp^-1，两个单项式相乘，指数相加即可。
                du2 = (du + dp_rec) % self.n    # 转为 a^i 单项式，那么相乘可以直接使用幂次相加实现
                sux = v[1]  # \sigma(u)(x)，用多项式的系数表示多项式，其中 -1 表示系数为 0，0 表示系数为 a^0，依次类推
                spx = vs[rho+1][1]  # \sigma(rho)(x)
                # spx * x ** (u - rho)，相当于 系数 上移 u-rho 位
                spx = np.concatenate((spx[u-rho:], # 原来最高的 u-rho 位 直接丢弃，因为系数肯定为 0
                                      np.ones((u-rho, ), dtype=np.int32) * -1))
                # spx * du2，即 系数 a^i 的幂次相加
                spx[spx >= 0] = (spx[spx >= 0] +du2) % self.n   # 注意元素值为 -1 的表示系数为 0
                # 下面执行 \sigma(u+1)(x) = \sigma(u)(x) + du*dp^-1*x^{u-p}*\sigma(p)(x)
                # 这是两个多项式相加，显然同类项的系数相加即可，将系数从单项式指数形式转为二进制形式
                # 转为二进制相加 mod 2，然后再转为单项式幂次
                sux_b = self.gf_bin[sux, :] # (t+1, m)
                spx_b = self.gf_bin[spx, :] # (t+1, m)
                sx_b = (sux_b + spx_b) % 2  # (t+1, m)  # 多项式各项系数的二进制形式
                # 二进制转为十进制，(t+1, )
                sx_d = np.sum(sx_b * 2 ** np.arange(self.m)[np.newaxis, ::-1], axis=-1, dtype=np.int32)
                # 转为序号 id，即单项式 a^i 的指数 i
                sx = self.gf2id[sx_d]   # (t+1, )
                # lu = max(v[3], vs[rho+1][3] + u - rho)
                lu = max(v[3], u+1-v[3])
            else:
                sx = v[1]
                lu = v[3]
            if lu > self.t:     # 错误多项式的阶数 > t，那么错误太多，无法定位错误
                return None
            
            if u + 1 < 2 * self.t:  # 注意 u 为 0 对应 S2 也就是 S[1]
                du = [S_b[u + 1]]       # S_{u+1} 二进制形式
                for i in range(lu):
                    s_id = self.gf2id[S[u-i]]   # S_{u-i} 转为单项式 a^i 的指数 i
                    x_id = sx[-i-2]     # 多项式 \sigma(u+1)(x) 的各项系数，即单项式 a^i 的指数 i
                    # 两者相乘，即指数相加，指数如为 -1，表示单项式为 0
                    if s_id >= 0 and x_id >= 0:   # a^i 的幂次 i
                        du_id = (s_id + x_id) % self.n  # 相乘后的单项式指数
                        du.append(self.gf_bin[du_id])
                du = np.stack(du, axis=0)   # (N_du, m) # 得到 du 的多项式表示，du=S_u+1+\sigma1(u)Su+\sigma2(u)S_u-1
                du = np.sum(du, axis=0, dtype=np.uint8) % 2 # du 归约为单项式的二进制表示
                du = self.bin2dec(du)
                vs.append([u+1, sx, du, lu, u+1-lu, max_rho])
                if du == 0:
                    rho = 0     # du == 0 时，对应的 rho 值无所谓
                else:
                    if max_u_lu < u + 1 - lu:
                        max_u_lu = u + 1 - lu
                        max_rho = u + 1
        m = sx > -1   # valid mask，即系数不为 0 的单项
        x0 = np.arange(self.t, -1, -1)  # 基本元 a 代入 sx，每一个单项的幂次
        sx = sx[m]  # 筛选有效（系数不为 0）的单项式系数的指数
        x0 = x0[m]  # 筛选有效（系数不为 0）的单项的幂次，也是指数
        locs = []
        for i in range(self.n): # for a^i, check sx(a^i) == 0 is True or False
            x = x0 * i  # 幂次相乘，例如 x^2，代入 x=a^3，变成 (a^3)^2=a^6
            v = (x + sx) % self.n      # (t+1, )，系数相乘例如 a^i·a^j，等于幂次相加 i+j
            v_bin = self.gf_bin[v]  # (t+1, m)，将单项式从指数形式转为二进制形式
            v = np.sum(v_bin, axis=0) % 2   # (m, )
            if np.all(v == 0):   # i = 0 没有意义，对于 n 长度的消息，从右往左下标为 0~n-1
                locs.append(i - 1)     # np.all(v == 0) 表示 a^i 是 sx 的一个根。
        return locs
