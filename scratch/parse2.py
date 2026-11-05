import re, sys, collections
path = sys.argv[1]
lines = open(path).read().splitlines()
start = next(i for i, l in enumerate(lines) if l.strip().startswith("util.func public @main("))
body = lines[start:]
blocks = collections.OrderedDict(); succ = {}; cur = "^entry"; blocks[cur] = []
for l in body:
    s = l.strip()
    m = re.match(r"\^([A-Za-z0-9_.$]+)(\([^)]*\))?:", s)
    if m:
        cur = "^" + m.group(1); blocks.setdefault(cur, []); continue
    blocks.setdefault(cur, [])
    m = re.match(r'stream\.cmd\.dispatch @([^\(]+)', s)
    if m: blocks[cur].append(m.group(1).split("::")[-1].split("{")[0].strip().rstrip("{"))
    m = re.search(r"(?:cf\.cond_br|cf\.br)\s+(.*)", s)
    if m: succ.setdefault(cur, []).extend("^" + t for t in re.findall(r"\^([A-Za-z0-9_.$]+)", m.group(1)))
idx = {b: i for i, b in enumerate(blocks)}
print("blocks", len(blocks), "total dispatch sites", sum(len(v) for v in blocks.values()))
for b in blocks:
    tb = [t for t in succ.get(b, []) if t in idx]
    back = [t for t in tb if idx[t] <= idx[b]]
    print(f"{b} [{idx[b]:2d}] d={len(blocks[b]):3d} succ={tb} back={back}")
