#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef uint64_t cell;

cell conway(cell in[9]) {
    cell in_0 = in[0];
    cell in_1 = in[1];
    cell in_2 = in[2];
    cell in_3 = in[3];
    cell in_4 = in[4];
    cell in_5 = in[5];
    cell in_6 = in[6];
    cell in_7 = in[7];
    cell in_8 = in[8];
    cell a = ~in_6;
    cell b = in_1 ^ in_7;
    cell c = in_6 & in_8;
    cell e = in_6 & in_1;
    cell f = in_7 & in_1;
    cell g = in_3 ^ in_2;
    cell h = in_2 & in_6;
    cell j = in_1 & in_4;
    cell k = in_1 & in_8;
    cell l = in_4 | in_5;
    cell m = in_4 & in_5;
    cell n = in_7 & in_2;
    cell o = in_6 & in_3;
    cell p = in_6 & in_3;
    cell q = in_6 | in_3;
    cell r = in_2 & in_7;
    cell s = in_4 | in_5;
    cell t = in_1 | in_2;
    cell u = in_3 | in_8;
    cell v = in_3 | ~a;
    cell w = in_2 & p;
    cell x = in_0 ^ r;
    cell y = e | f;
    cell z = in_0 | in_5;
    cell aa = s & ~in_6;
    cell ab = in_2 | in_5;
    cell ac = in_3 | in_4;
    cell ae = in_5 | in_0;
    cell af = c & in_2;
    cell ag = m & in_5;
    cell ah = in_1 | in_5;
    cell aj = in_1 | in_0;
    cell ak = n | in_8;
    cell al = in_1 ^ in_8;
    cell am = f & g;
    cell an = ab & in_5;
    cell ao = ag & b;
    cell ap = in_3 & z;
    cell aq = x | y;
    cell ar = in_3 | af;
    cell as = in_4 | in_0;
    cell at = l & in_7;
    cell au = y | in_2;
    cell av = ac | ae;
    cell aw = in_8 | in_6;
    cell ax = af | ~in_4;
    cell ay = in_7 | in_1;
    cell az = ae | in_8;
    cell ba = in_1 & ab;
    cell bb = in_4 | in_7;
    cell bc = in_0 | r;
    cell be = in_0 | in_5;
    cell bf = ah | in_0;
    cell bg = in_5 & in_7;
    cell bh = aj & bc;
    cell bj = aa & aj;
    cell bk = in_0 & in_1;
    cell bl = in_2 | ay;
    cell bm = u & au;
    cell bn = w & q;
    cell bo = aq & ar;
    cell bp = v & as;
    cell bq = aw & in_3;
    cell br = in_7 | ~ax;
    cell bs = in_0 | t;
    cell bt = in_6 & av;
    cell bu = in_6 | in_7;
    cell bv = r | in_5;
    cell bw = in_5 | in_0;
    cell bx = in_4 | o;
    cell by = ao | in_2;
    cell bz = in_1 | b;
    cell ca = bo & in_7;
    cell cb = in_0 | be;
    cell cc = ay | in_1;
    cell ce = in_2 & in_8;
    cell cf = bw | in_1;
    cell cg = ap | br;
    cell ch = bf & bb;
    cell cj = ak | al;
    cell ck = bp | bq;
    cell cl = bh | an;
    cell cm = in_7 | in_2;
    cell cn = in_7 | ar;
    cell co = by | in_2;
    cell cp = bx | bs;
    cell cq = bm & au;
    cell cr = in_0 & in_5;
    cell cs = bk & al;
    cell ct = cc | bn;
    cell cu = bj | ce;
    cell cv = ba | in_8;
    cell cw = ck | am;
    cell cx = bq | h;
    cell cy = ar | in_7;
    cell cz = cq & au;
    cell ea = cg | bz;
    cell eb = co & cf;
    cell ec = ch | bl;
    cell ee = in_2 & az;
    cell ef = cp | in_7;
    cell eg = bs & in_5;
    cell eh = bv & in_4;
    cell ej = bu & cx;
    cell ek = cx & q;
    cell el = az | ca;
    cell em = cr & in_5;
    cell en = cy | cz;
    cell eo = ec | in_5;
    cell ep = j ^ at;
    cell eq = in_6 | ee;
    cell er = ef | eg;
    cell es = cu | eh;
    cell et = cw & cm;
    cell eu = ek & el;
    cell ev = eo | in_2;
    cell ew = cj & en;
    cell ex = ea | bg;
    cell ey = in_3 | em;
    cell ez = cs | in_0;
    cell fa = ez & ep;
    cell fb = es | et;
    cell fc = bt & av;
    cell fe = ev | in_5;
    cell ff = cz & cl;
    cell fg = cn | em;
    cell fh = in_0 & ct;
    cell fj = cb | in_1;
    cell fk = eb | fa;
    cell fl = fc & fe;
    cell fm = ex & ey;
    cell fn = fk | fl;
    cell fo = ej | et;
    cell fp = in_8 | in_2;
    cell fq = eu & ct;
    cell fr = er | eg;
    cell fs = fb | fh;
    cell ft = in_3 | fo;
    cell fu = fj & cv;
    cell fv = fn | ew;
    cell fw = fv | fm;
    cell fx = ew | k;
    cell fy = fu | fq;
    cell fz = ff & bz;
    cell ga = fs | bg;
    cell gb = fx | fo;
    cell gc = ft & fp;
    cell ge = ga | in_6;
    cell gf = fw | in_8;
    cell gg = fy & eq;
    cell gh = fz | ~gf;
    cell gj = ge | in_6;
    cell gk = ~(fq ^ fr);
    cell gl = gg & fg;
    cell gm = gj | gc;
    cell gn = gh | ~gm;
    cell go = gb | fo;
    cell gp = em & go;
    cell gq = ~gl;
    cell gr = gq & ~gk;
    cell gs = gr & ~gn;
    cell out = gs & ~gp;
    return out;
}

// human attempt, 54 gates
cell conway_clayton(cell in[9]) {
    // pairs
    cell p00 = ~(in[0] | in[1]);
    cell p10 = ~(in[2] | in[3]);
    cell p20 = ~(in[5] | in[6]);
    cell p30 = ~(in[7] | in[8]);
    cell p01 = in[0] ^ in[1];
    cell p11 = in[2] ^ in[3];
    cell p21 = in[5] ^ in[6];
    cell p31 = in[7] ^ in[8];
    cell p02 = in[0] & in[1];
    cell p12 = in[2] & in[3];
    cell p22 = in[5] & in[6];
    cell p32 = in[7] & in[8];
    // halfs
    cell h00 = p00 & p10;
    cell h10 = p20 & p30;
    cell h01 = (p00 & p11) | (p10 & p01);
    cell h11 = (p20 & p31) | (p30 & p21);
    cell h03 = (p02 & p11) | (p12 & p01);
    cell h13 = (p22 & p31) | (p32 & p21);
    cell h02 = (p02 & p10) | (p00 & p12) | (p01 & p11);
    cell h12 = (p22 & p30) | (p20 & p32) | (p21 & p31);
    // neighbors
    cell n2 = (h01 & h11) | (h02 & h10) | (h00 & h12);
    cell n3 = (h03 & h10) | (h13 & h00) | (h02 & h11 ) | (h12 & h01);
    // rule
    cell out = n3 | (in[4] & n2);
    return out;
}

typedef struct {
    cell* cells;
    size_t cells_len;
    size_t width;
    size_t height;
} board_t;

void fatal(const char* message) {
    fprintf(stderr, "fatal: %s\n", message);
    exit(1);
}

board_t board_new(size_t width, size_t height) {
    if (width % 64 != 0) { fatal("board width must be multiple of 64"); }

    size_t cells_len = (size_t)(width / 64) * height;
    cell* cells = malloc(cells_len * sizeof(cell));
    if (cells == NULL) { fatal("not able to allocate memory for board "); }

    board_t board = { cells, cells_len, width, height };
    return board;
}

// https://en.wikipedia.org/wiki/Xorshift
// constant is frac(golden_ratio) * 2^64
// global state bad cry me a river
uint64_t rand_state = 0x9e3779b97f4a7c55;

uint64_t rand_uint64_t() {
    uint64_t x = rand_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rand_state = x;
    return x;
}

cell rand_cell() {
    // return 0x0101010101010101;
    return rand_uint64_t();
}

void rand_board_mut(board_t *board) {
    for (size_t i = 0; i < board->cells_len; ++i) {
        board->cells[i] = rand_cell();
    }
}

void board_debug(board_t *board) {
    size_t cells_per_row = board->width / 64;
    for (size_t i = 0; i < board->height; ++i) {
        for (size_t j = 0; j < cells_per_row; ++j) {
            cell current_cell = board->cells[i * cells_per_row + j];
            for (int k = 0; k < 64; k++) {
                if ((current_cell >> k) & 1) {
                    printf("█ ");
                } else {
                    printf("  ");
                }
            }
        }
        printf("|\n");
    }
    for (size_t i = 0; i < board->width * 2; ++i) {
        printf("-");
    }
    printf("\n");
}

void board_step_scratch_mut(
    board_t *board,
    board_t *scratch_left,
    board_t *scratch_right
) {
    size_t cells_per_row = board->width / 64;
    for (size_t i = 0; i < board->height; ++i) {
        cell msb_prev = (board->cells[i * cells_per_row + cells_per_row - 1] >> 63) & 1;
        for (size_t j = 0; j < cells_per_row; ++j) {
            size_t idx = i * cells_per_row + j;
            cell cell_curr = board->cells[idx];
            cell cell_shift = cell_curr << 1;
            cell msb_curr = (cell_curr >> 63) & 1;
            scratch_left->cells[idx] = cell_shift | msb_prev;
            msb_prev = msb_curr;
        }
    }
    for (size_t i = 0; i < board->height; ++i) {
        cell lsb_next = board->cells[i * cells_per_row] & 1;
        for (size_t j = cells_per_row; j-- > 0; ) {
            size_t idx = i * cells_per_row + j;
            cell cell_curr = board->cells[idx];
            cell cell_shift = cell_curr >> 1;
            cell lsb_curr = cell_curr & 1;
            scratch_right->cells[idx] = cell_shift | (lsb_next << 63);
            lsb_next = lsb_curr;
        }
    }
}

void board_step_mut(
    board_t *board,
    board_t *s_left, // scratch
    board_t *s_right,
    board_t *s_out
) {
    board_step_scratch_mut(board, s_left, s_right);

    size_t step = board->width / 64;
    size_t wrap = board->cells_len;

    for (size_t i = 0; i < board->cells_len; i++) {
        cell in[9];
        size_t i_top = (i + wrap - step) % wrap;
        size_t i_bottom = (i + step) % wrap;
        // top row
        in[0] = s_left->cells[i_top];
        in[1] = board->cells[i_top];
        in[2] = s_right->cells[i_top];
        // middle row
        in[3] = s_left->cells[i];
        in[4] = board->cells[i];
        in[5] = s_right->cells[i];
        // bottom row
        in[6] = s_left->cells[i_bottom];
        in[7] = board->cells[i_bottom];
        in[8] = s_right->cells[i_bottom];
        // update output
        s_out->cells[i] = conway(in);
    }

    // double-buffering
    cell* tmp_cells = board->cells;
    board->cells = s_out->cells;
    s_out->cells = tmp_cells;
}

int main() {
    size_t width = 512;
    size_t height = 512;

    board_t board = board_new(width, height);
    board_t sl = board_new(width, height);
    board_t sr = board_new(width, height);
    board_t so = board_new(width, height);

    rand_board_mut(&board);

    for (size_t count = 0; count < 100000; count++) {
        // vvv comment out for benchmarking
        printf("\033[H");
        board_debug(&board);
        printf("Step: %zu\n", count);
        // ^^^ comment out for benchmarking
        board_step_mut(&board, &sl, &sr, &so);
    }

    printf("done!\n");
}
