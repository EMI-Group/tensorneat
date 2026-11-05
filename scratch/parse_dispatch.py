import re, sys, collections

path = sys.argv[1]
lines = open(path).read().splitlines()
# find start of the top-level util.func public @main body
start = None
for i, l in enumerate(lines):
    if l.strip().startswith("util.func public @main("):
        start = i
        break
body = lines[start:]
block = "^entry"
blocks = collections.OrderedDict()
succ = {}
order = ["^entry"]
for l in body:
    s = l.strip()
    m = re.match(r"\^([A-Za-z0-9_.$]+):", s)
    if m:
        block = "^" + m.group(1)
        if block not in blocks:
            blocks[block] = []
            order.append(block)
        continue
    if block not in blocks:
        blocks[block] = []
    m = re.match(r'stream\.cmd\.dispatch @([^\(]+)', s)
    if m:
        blocks[block].append(m.group(1))
    m = re.search(r"(?:cf\.cond_br|cf\.br)\s+(.*)", s)
    if m:
        targets = re.findall(r"\^([A-Za-z0-9_.$]+)", m.group(1))
        succ.setdefault(block, []).extend("^"+t for t in targets)

print("blocks:", len(blocks), "total dispatch sites:", sum(len(v) for v in blocks.values()))
# report per block: index, n dispatch, successors, first few dispatch names
idx = {b: i for i, b in enumerate(order)}
for b in order:
    ds = blocks[b]
    tg = succ.get(b, [])
    tg = [t for t in tg if t in idx]
    back = [t for t in tg if t in idx and idx[t] <= idx[b]]
    print(f"{b} [{list(blocks).index(b)}] dispatch={len(ds)} succ={tg} backedge={back}")
    for d in ds:
        print("    ", d)
