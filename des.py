__author__ = 'jadeszek'

B_SIZE = 64  # block size

# The 64 bits of the input block to be enciphered are first subjected to the following permutation,
# called the initial permutation IP:

IP = [i - 1 for i in [58, 50, 42, 34, 26, 18, 10, 2,
                      60, 52, 44, 36, 28, 20, 12, 4,
                      62, 54, 46, 38, 30, 22, 14, 6,
                      64, 56, 48, 40, 32, 24, 16, 8,
                      57, 49, 41, 33, 25, 17, 9, 1,
                      59, 51, 43, 35, 27, 19, 11, 3,
                      61, 53, 45, 37, 29, 21, 13, 5,
                      63, 55, 47, 39, 31, 23, 15, 7]
      ]

IP_INV = [i - 1 for i in [40, 8, 48, 16, 56, 24, 64, 32,
                          39, 7, 47, 15, 55, 23, 63, 31,
                          38, 6, 46, 14, 54, 22, 62, 30,
                          37, 5, 45, 13, 53, 21, 61, 29,
                          36, 4, 44, 12, 52, 20, 60, 28,
                          35, 3, 43, 11, 51, 19, 59, 27,
                          34, 2, 42, 10, 50, 18, 58, 26,
                          33, 1, 41, 9, 49, 17, 57, 25]
          ]

# Let E denote a function which takes a block of 32 bits as input and yields a block of 48 bits as
# output. Let E be such that the 48 bits of its output, written as 8 blocks of 6 bits each, are
# obtained by selecting the bits in its inputs in order according to the following table:
E = [i - 1 for i in [32, 1, 2, 3, 4, 5,
                     4, 5, 6, 7, 8, 9,
                     8, 9, 10, 11, 12, 13,
                     12, 13, 14, 15, 16, 17,
                     16, 17, 18, 19, 20, 21,
                     20, 21, 22, 23, 24, 25,
                     24, 25, 26, 27, 28, 29,
                     28, 29, 30, 31, 32, 1]
     ]

S = [
    [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7,
     0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8,
     4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0,
     15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13],

    [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10,
     3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5,
     0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15,
     13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9],

    [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8,
     13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1,
     13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7,
     1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12],

    [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15,
     13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9,
     10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4,
     3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14],

    [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9,
     14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6,
     4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14,
     11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3],

    [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11,
     10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8,
     9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6,
     4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13],

    [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1,
     13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6,
     1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2,
     6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12],

    [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7,
     1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2,
     7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8,
     2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]
]

P = [i - 1 for i in [16, 7, 20, 21,
                     29, 12, 28, 17,
                     1, 15, 23, 26,
                     5, 18, 31, 10,
                     2, 8, 24, 14,
                     32, 27, 3, 9,
                     19, 13, 30, 6,
                     22, 11, 4, 25]
     ]

PC = [
    [i - 1 for i in [57, 49, 41, 33, 25, 17, 9,
                     1, 58, 50, 42, 34, 26, 18,
                     10, 2, 59, 51, 43, 35, 27,
                     19, 11, 3, 60, 52, 44, 36,
                     63, 55, 47, 39, 31, 23, 15,
                     7, 62, 54, 46, 38, 30, 22,
                     14, 6, 61, 53, 45, 37, 29,
                     21, 13, 5, 28, 20, 12, 4]
     ],
    [i - 1 for i in [14, 17, 11, 24, 1, 5,
                     3, 28, 15, 6, 21, 10,
                     23, 19, 12, 4, 26, 8,
                     16, 7, 27, 20, 13, 2,
                     41, 52, 31, 37, 47, 55,
                     30, 40, 51, 45, 33, 48,
                     44, 49, 39, 56, 34, 53,
                     46, 42, 50, 36, 29, 32
                     ]
     ]
]

KEY_SHIFT = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]


def log(bin_list, width=8):
    s = ""
    i = 0
    for b in bin_list:
        s += str(b)
        i += 1
        if i % width == 0:
            s += " "
    return s


def apply_permutation(byte_block, perm):
    result = [0] * len(perm)
    i = 0
    for p in perm:
        result[i] = byte_block[p]
        i += 1
    return result


def init(byte_block):
    return apply_permutation(byte_block, IP)


def split_block(byte_block):
    half = len(byte_block) / 2
    return [byte_block[:half], byte_block[half:]]


def left_shift(block, step):
    return block[step:] + block[:step]


def right_shift(block, step):
    return left_shift(block, -step)


def generate_keys(base_key):  # key in byte form
    KP = apply_permutation(base_key, PC[0])
    #print "K_PC1", log(permuted, 6)
    C, D = split_block(KP)
    keys = []
    for i in range(16):
        shift = KEY_SHIFT[i]
        C = left_shift(C, shift)
        D = left_shift(D, shift)
       # print "C", i+1, log(C, 6)
       # print "D", i+1, log(D, 6)
        CD = unite(C, D)
        print "CD[%s]: " % str(i+1), log(CD, 7)
        K = apply_permutation(CD, PC[1])
        print "KS[%s]: " % str(i+1), log(K, 6)
        keys.append(K)
    return keys

def dec2bin(decimal, width=4):
    return [int(d) for d in bin(decimal)[2:].zfill(width)]


def bin2dec(bin_list):
    return int(reduce(lambda x, y: str(x) + str(y), bin_list), 2)


def get_sbox_coordinates(six_byte_chunk):
    row_bin = [six_byte_chunk[0]] + [six_byte_chunk[-1]]
    column_bin = six_byte_chunk[1:-1]

  #  print(row_bin, column_bin)
    return bin2dec(row_bin), bin2dec(column_bin)


def apply_sbox(chunk, sbox):
    row, col = get_sbox_coordinates(chunk)
    number = sbox[row * 16 + col]
   # print "(", row, ",", col, ") ->", number
    return dec2bin(number)


def xor(A, B):
    return [a ^ b for a, b in zip(A, B)]


def unite(left_block, right_block):
    return left_block + right_block


def f(right_block, key, iter):
    print "\n >>> round", iter+1, "\n"
    permuted = apply_permutation(right_block, E)
    print "E", len(permuted), log(permuted, 6)

    print "KS", iter+1, len(key), log(key, 6)
    assert len(permuted) == len(key)
    xored = xor(permuted, key)
    print "E ^ KS", len(xored), log(xored, 6)
    six_byte_chunks = [xored[i:i + 6] for i in range(0, len(xored), 6)]
  #  print "six_byte_chunks ", log(six_byte_chunks)
    sbox_out = [apply_sbox(chunk, sbox) for chunk, sbox in zip(six_byte_chunks, S)]
    sbox_out = reduce(lambda c1, c2: c1 + c2, sbox_out)
    print "Sbox", len(sbox_out), log(sbox_out, 4)

    return apply_permutation(sbox_out, P)


def string2bin(text):
    result = []
    for c in text:
        result += dec2bin(ord(c), 8)
    return result


def bin2hex(bin_list):
    return hex(bin2dec(bin_list))[:-1]  # without L


def final(byte_block):
    return apply_permutation(byte_block, IP_INV)


test = [0, 0, 0, 1, 0, 1, 1, 0,
        1, 0, 1, 1, 1, 1, 0, 1,
        0, 1, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 1, 0, 1, 0, 0,
        1, 0, 1, 1, 0, 1, 0, 0,
        0, 1, 0, 1, 0, 0, 1, 1,
        1, 0, 0, 0, 1, 0, 1, 0,
        0, 1, 1, 1, 1, 0, 1, 0, ]

key = [1, 1, 1, 1, 0, 1, 1, 0,
       1, 0, 1, 1, 1, 1, 0, 1,
       0, 1, 0, 0, 0, 0, 0, 1,
       1, 0, 0, 1, 0, 1, 0, 0,
       1, 0, 1, 1, 0, 1, 0, 0,
       0, 1, 0, 1, 0, 1, 1, 1,
       1, 0, 0, 0, 1, 0, 1, 0,
       0, 1, 1, 1, 1, 0, 1, 1, ]

# t = 0x0123456789ABCDEF  # "01234567"


def encode_des(block, key):
    print "block", len(block), log(block)
    print "key", len(key), log(key)
    keys = generate_keys(key)
    return bin2hex(DES(block, keys))


def decode_des(block, key):
    print "block", len(block), log(block)
    print "key", len(key), log(key)
    keys = generate_keys(key)[::-1]
    return bin2hex(DES(block, keys))


def DES(block, keys):
    permuted = init(block)
    print "permuted", len(permuted), log(permuted)
    L, R = split_block(permuted)
    print "L", 0, len(L), log(L)
    print "R", 0, len(R), log(R)
    for i in range(16):
        Rf = f(R, keys[i], i)
        print "Rf", i+1, log(Rf)
        L, R = R, xor(Rf, L)
        print "L", i+1, len(L), log(L)
        print "R", i+1, len(R), log(R)
    united = unite(R, L)
    print "united", len(united), log(united)
    result = final(united)
    print "result", len(result), log(result)
    return result


# DES(test, key)


t = 0xBABC1AD1AD1A0000 # 0xaa39b9777efc3c14
k = 0x1234567887654321 # 0x3b3898371520f75e

e = encode_des(dec2bin(t, 64), dec2bin(k, 64))
d = decode_des(dec2bin(int(e, 16), 64), dec2bin(k, 64))

print k
print hex(t)
print e
print d
























