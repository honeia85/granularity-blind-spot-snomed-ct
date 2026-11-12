"""PCG64 random generator with SeedSequence seeding, bounded integers and multinomial sampling, in pure Python.

The streams equal those of numpy.random.default_rng(seed).integers(0, n, size) and
numpy.random.default_rng(seed).multinomial(n, [1/k] * k) (numpy 1.17 or later), so the bootstrap replicates can be
generated without numpy:
  * SeedSequence(seed) expands the integer seed into four 64-bit words (pool of four 32-bit words, hashmix);
  * PCG64 is the 128-bit linear congruential generator with XSL-RR output, seeded from those words;
  * a double is (next 64-bit output >> 11) * 2**-53;
  * a 32-bit draw is the low half of a 64-bit output, and the next 32-bit draw is its high half;
  * integers(0, n) with n <= 2**32 uses Lemire's multiply-and-reject method on 32-bit draws;
  * a multinomial draw takes, category by category, a binomial draw of the remaining count with probability
    p_j / (remaining probability); binomial draws with n * min(p, 1 - p) <= 30 use inversion.
"""
import math

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
MASK128 = (1 << 128) - 1
PCG_MULT = 0x2360ED051FC65DA44385DF649FCCF645


def _seed_words(seed, n_words64=4):
    INIT_A, MULT_A, INIT_B, MULT_B = 0x43B0D7E5, 0x931E8875, 0x8B51F9DD, 0x58F38DED
    MIX_L, MIX_R, XSHIFT, POOL = 0xCA01F9DD, 0x4973F715, 16, 4
    entropy = []
    while True:
        entropy.append(seed & MASK32)
        seed >>= 32
        if not seed:
            break
    h = [INIT_A]

    def hashmix(v):
        v = (v ^ h[0]) & MASK32
        h[0] = (h[0] * MULT_A) & MASK32
        v = (v * h[0]) & MASK32
        return v ^ (v >> XSHIFT)

    def mix(x, y):
        r = (MIX_L * x - MIX_R * y) & MASK32
        return r ^ (r >> XSHIFT)

    pool = [hashmix(entropy[i] if i < len(entropy) else 0) for i in range(POOL)]
    for s in range(POOL):
        for d in range(POOL):
            if s != d:
                pool[d] = mix(pool[d], hashmix(pool[s]))
    for s in range(POOL, len(entropy)):
        for d in range(POOL):
            pool[d] = mix(pool[d], hashmix(entropy[s]))
    hb = INIT_B
    words = []
    for i in range(2 * n_words64):
        v = (pool[i % POOL] ^ hb) & MASK32
        hb = (hb * MULT_B) & MASK32
        v = (v * hb) & MASK32
        words.append(v ^ (v >> XSHIFT))
    return [words[2 * i] | (words[2 * i + 1] << 32) for i in range(n_words64)]


class Generator:
    def __init__(self, seed):
        w = _seed_words(seed)
        init_state = (w[0] << 64) | w[1]
        init_seq = (w[2] << 64) | w[3]
        self.inc = ((init_seq << 1) | 1) & MASK128
        self.state = 0
        self._step()
        self.state = (self.state + init_state) & MASK128
        self._step()
        self._has32 = False
        self._buf32 = 0

    def _step(self):
        self.state = (self.state * PCG_MULT + self.inc) & MASK128

    def next_double(self):
        s = (self.state * PCG_MULT + self.inc) & MASK128
        self.state = s
        x = ((s >> 64) ^ s) & MASK64
        r = s >> 122
        x = ((x >> r) | (x << (64 - r))) & MASK64
        return (x >> 11) * 2.0 ** -53

    def integers(self, n, count):
        """count draws of integers(0, n), 1 <= n <= 2**32, as a list."""
        out = [0] * count
        if n == 1:
            return out
        threshold = (MASK32 - (n - 1)) % n
        state, inc, has32, buf32 = self.state, self.inc, self._has32, self._buf32
        for i in range(count):
            while True:
                if has32:
                    u = buf32
                    has32 = False
                else:
                    state = (state * PCG_MULT + inc) & MASK128
                    x = ((state >> 64) ^ state) & MASK64
                    r = state >> 122
                    x = ((x >> r) | (x << (64 - r))) & MASK64
                    u = x & MASK32
                    buf32 = x >> 32
                    has32 = True
                m = u * n
                if (m & MASK32) >= threshold:
                    break
            out[i] = m >> 32
        self.state, self._has32, self._buf32 = state, has32, buf32
        return out

    def _binomial_inversion(self, n, p):
        q = 1.0 - p
        qn = math.exp(n * math.log(q))
        npq = n * p
        bound = min(n, npq + 10.0 * math.sqrt(npq * q + 1))
        x = 0
        px = qn
        u = self.next_double()
        while u > px:
            x += 1
            if x > bound:
                x = 0
                px = qn
                u = self.next_double()
            else:
                u -= px
                px = ((n - x + 1) * p * px) / (x * q)
        return x

    def binomial(self, n, p):
        if n == 0 or p == 0.0:
            return 0
        if p <= 0.5:
            if p * n <= 30.0:
                return self._binomial_inversion(n, p)
        else:
            q = 1.0 - p
            if q * n <= 30.0:
                return n - self._binomial_inversion(n, q)
        raise NotImplementedError('binomial draws with n * min(p, 1 - p) > 30 are not needed here')

    def multinomial_uniform(self, n, k):
        """One draw of multinomial(n, [1/k] * k) as a list of k counts."""
        p = 1.0 / k
        out = [0] * k
        remaining_p = 1.0
        dn = n
        for j in range(k - 1):
            x = self.binomial(dn, p / remaining_p)
            out[j] = x
            dn -= x
            if dn <= 0:
                break
            remaining_p -= p
        if dn > 0:
            out[k - 1] = dn
        return out


def quantile_linear(values, q):
    """numpy.percentile(values, 100 q) with linear interpolation, for finite values."""
    s = sorted(values)
    k = (len(s) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[int(f)] * (c - k) + s[int(c)] * (k - f)


def quantile_inverted_cdf(values, q):
    """numpy.quantile(values, q, method='inverted_cdf') for finite values."""
    s = sorted(values)
    index = len(s) * q - 1
    lower = math.floor(index)
    pos = lower + 1 if index - lower > 0 else lower
    return s[max(0, min(len(s) - 1, int(pos)))]
