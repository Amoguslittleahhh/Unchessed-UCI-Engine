import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    src = f.read()

old = '''        if self.is_repetition(pos.hash) {
            return self.draw(ply);
        }

        // mate distance pruning: we're already `ply` moves deep, so the
        // fastest possible mate FOR us from here is mate-in-(ply+1) from the
        // root's perspective, and the fastest possible mate AGAINST us is
        // being-mated-in-ply. Tightening alpha/beta to these bounds can't
        // change the game-theoretic result (a real mate score can never
        // fall outside them), but prunes subtrees that could only ever
        // find a slower mate than one already known elsewhere in the tree.
        alpha = alpha.max(-MATE + ply as i32);
        let beta = beta.min(MATE - ply as i32 - 1);
        if alpha >= beta {
            return alpha;
        }

        let in_chk = in_check(pos);'''

new = '''        if self.is_repetition(pos.hash) {
            return self.draw(ply);
        }

        let in_chk = in_check(pos);'''

assert old in src, "MDP block not found verbatim"
src = src.replace(old, new)

with open(path, "w", encoding="utf-8") as f:
    f.write(src)
print("reverted OK")
