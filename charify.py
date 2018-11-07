import sys
from collections import Counter

with open(sys.argv[1], 'r', encoding = 'utf-8') as f:
  CHARSET = set([char for (char, count) in Counter(f.read()).most_common() if count >= 25])

# This specific character is used in main.evaluate (L505) for the loss summing of characters into words
SPACE = '⁀'

for filename in [sys.argv[1], sys.argv[1].replace('train', 'valid'), sys.argv[1].replace('train', 'test')]:
  with open(filename, 'r', encoding = 'utf-8') as f:
    with open(filename+'.charunked', 'w', encoding = 'utf-8') as fw:
      for line in f:
        line = ''.join(c if c in CHARSET or c.isspace() else '◊' for c in line)
        fw.write(line)
        # print(' '.join(SPACE.join(line.split())))
