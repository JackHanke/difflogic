## compiling (or extracting) network
import jax.numpy as jnp

def ext_gate_name(idx, l, r):
    names = [
        lambda a, b: "0",
        lambda a, b: f"{a} & {b}",
        lambda a, b: f"{a} & ~{b}",
        lambda a, b: a,
        lambda a, b: f"{b} & ~{a}",
        lambda a, b: b,
        lambda a, b: f"{a} ^ {b}",
        lambda a, b: f"{a} | {b}",
        lambda a, b: f"~({a} | {b})",
        lambda a, b: f"~({a} ^ {b})",
        lambda a, b: f"~{b}",
        lambda a, b: f"{a} | ~{b}",
        lambda a, b: f"~{a}",
        lambda a, b: f"{b} | ~{a}",
        lambda a, b: f"~({a} & {b})",
        lambda a, b: "~0",
    ]
    return names[idx](l, r)

def ext_add_deps(req, idx, l, r):
    deps = [
        lambda a, b: [],
        lambda a, b: [a, b],
        lambda a, b: [a, b],
        lambda a, b: [a],
        lambda a, b: [a, b],
        lambda a, b: [b],
        lambda a, b: [a, b],
        lambda a, b: [a, b],
        lambda a, b: [a, b],
        lambda a, b: [a, b],
        lambda a, b: [b],
        lambda a, b: [a, b],
        lambda a, b: [a],
        lambda a, b: [a, b],
        lambda a, b: [a, b],
        lambda a, b: [],
    ]
    for g in deps[idx](l, r):
        req.add(g)
    return req

def ext_layer(param, left, right, layer):
    out = []
    for i, (g, l, r) in enumerate(zip(param.T, left, right)):
        idx_g = jnp.argmax(g, axis=0)
        idx_l = jnp.argmax(l, axis=0)
        idx_r = jnp.argmax(r, axis=0)
        instr = (f"g_{layer+1}_{i}", idx_g, f"g_{layer}_{idx_l}", f"g_{layer}_{idx_r}")
        if instr is not None:
            out.append(instr)
    return out

def ext_logic(params, wires):
    out = []
    for layer, (param, (left, right)) in list(enumerate(zip(params, wires)))[::-1]:
        # print(param.shape, left.shape, right.shape)
        instrs = ext_layer(param, left, right, layer)
        out = instrs + out
    root = f"g_{len(params)}_{0}"
    out = ext_elim(out, root)
    out = ext_copy_prop(out, root)
    out = ext_alpha_rename(out, root)
    return out

def ext_format(instr):
    (o, idx, l, r) = instr
    name = ext_gate_name(idx, l, r)
    return f"    cell {o} = {name};\n"

def ext_elim(instrs, root):
    out = []
    req = set([root])
    for instr in instrs[::-1]:
        (o, idx, l, r) = instr
        if o in req:
            req = ext_add_deps(req, idx, l, r)
            out.append(instr)
    return list(out[::-1])

def ext_copy_prop(instrs, root):
    out = []
    rename = dict()
    for instr in instrs:
        (o, idx, l, r) = instr
        if l in rename: l = rename[l]
        if r in rename: r = rename[r]
        if o == root: out.append((o, idx, l, r))
        elif idx == 3: rename[o] = l
        elif idx == 5: rename[o] = r
        else: out.append((o, idx, l, r))
    return out

def ext_alpha_count():
    # j and q, only letters not used in c keywords
    # no d or i to avoid do, if, in
    letters = "abcefghjklmnopqrstuvwxyz"
    for letter in letters:
        yield letter
    for letter in ext_alpha_count():
        for subletter in letters:
            yield letter + subletter

# for count in ext_alpha_count():
#     print(count)

def ext_regs_unique(instrs):
    seen = set()
    return [(o, seen.add(o))[0] for (o, _, _, _) in instrs if o not in seen]

def ext_alpha_rename(instrs, root):
    imm_regs = ext_regs_unique(instrs)
    rename = dict(zip(imm_regs, ext_alpha_count()))
    if root in rename: rename[root] = "out"
    for i in range(9): rename[f"g_0_{i}"] = f"in_{i}"
    out = []
    for (o, idx, l, r) in instrs:
        o = rename[o] if o in rename else o
        l = rename[l] if l in rename else l
        r = rename[r] if r in rename else r
        out.append((o, idx, l, r))
    return out

def ext_compile_to_c(params, wires):
    with open("gate.c.template", "r") as fin:
        before, after = fin.read().split("    {{ logic }}\n")
    with open("gate.c", "w") as fout:
        fout.write(before)
        for instr in ext_logic(params, wires):
            fout.write(ext_format(instr))
        fout.write(after)
    print("wrote circuit to gate.c")

