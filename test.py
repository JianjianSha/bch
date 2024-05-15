import bch
import numpy as np

ed = bch.EncoderDecoder(6, 36, 5)
success = 0
fail = 0
num = 10000
for _ in range(num):
    i = np.random.randint(0, 2, (36,), dtype=np.int32)
    r = ed.encode(i)
    ei = np.random.randint(0, 63, (5,))
    r[ei] ^= 1
    i2 = ed.decode(r)
    if np.sum(np.abs(i2 - i)) == 0:
        success += 1
    else:
        fail += 1

print('| 成功 | 失败 | 总计 |')
print('|—————|—————|——————|')
print('|%-6d|%-6d|%-6d|' % (success, fail, num))
