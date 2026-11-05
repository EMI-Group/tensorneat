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
    if m: blocks[cur].append(m.group(1).split("::")[-1].split("{")[0].strip())
    m = re.search(r"(?:cf\.cond_br|cf\.br)\s+(.*)", s)
    if m: succ.setdefault(cur, []).extend("^" + t for t in re.findall(r"\^([A-Za-z0-9_.$]+)", m.group(1)))
idx = {b: i for i, b in enumerate(blocks)}
print("blocks", len(blocks), "total dispatch sites", sum(len(v) for v in blocks.values()))
print("block sequence:", " ".join(f"{b}[{len(blocks[b])}]" for b in blocks))
def children(h):
    """all blocks reachable from h (including h)"""
    seen = set(); stack = [h]
    while stack:
        b = stack.pop()
        if b in seen or b not in idx: continue
        seen.add(b); stack.extend(t for t in succ.get(b, []) if t in idx)
    return seen
# loop headers: a block with a successor that can reach it (back edge), found via DFS from header succ
loops = []
for b in blocks:
    for t in succ.get(b, []):
        if t in idx and b in children(t):
            loops.append((b, t))
for hdr, tgt in loops:
    region = [x for x in blocks if x in children(tgt)]
    # per-iteration dispatch total excluding nested loops inside region
    seq = []
    for x in blocks:
        if x in region: seq.extend(blocks[x])
    print(f"\nLOOP header={hdr} back-target={tgt} region={' '.join(sorted(region, key=lambda z: idx[z]))}")
    print(f"  per-iteration dispatch sites in region: {len(seq)}")
    for d in seq: print("   ", d)
