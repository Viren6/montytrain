use bullet::default::formats::montyformat::chess::{Attacks, Move, Piece, Position, Side};

pub const MAX_MOVES: usize = 96;
pub const NUM_MOVES: usize = OFFSETS[64] + PROMOS;
pub const PROMOS: usize = 4 * 22;

pub fn map_move_to_index(pos: &Position, mov: Move) -> usize {
    let hm = if pos.king_index() % 8 > 3 { 7 } else { 0 };
    if mov.is_promo() {
        let ffile = (mov.src() ^ hm) % 8;
        let tfile = (mov.to() ^ hm) % 8;
        let promo_id = 2 * ffile + tfile;

        OFFSETS[64] + 22 * (mov.promo_pc() - Piece::KNIGHT) + usize::from(promo_id)
    } else {
        let flip = if pos.stm() == Side::BLACK { 56 } else { 0 };
        let from = usize::from(mov.src() ^ flip ^ hm);
        let dest = usize::from(mov.to() ^ flip ^ hm);

        let below = ALL_DESTINATIONS[from] & ((1 << dest) - 1);

        OFFSETS[from] + below.count_ones() as usize
    }
}

macro_rules! init {
    (|$sq:ident, $size:literal | $($rest:tt)+) => {{
        let mut $sq = 0;
        let mut res = [{$($rest)+}; $size];
        while $sq < $size {
            res[$sq] = {$($rest)+};
            $sq += 1;
        }
        res
    }};
}

const OFFSETS: [usize; 65] = {
    let mut offsets = [0; 65];

    let mut curr = 0;
    let mut sq = 0;

    while sq < 64 {
        offsets[sq] = curr;
        curr += ALL_DESTINATIONS[sq].count_ones() as usize;
        sq += 1;
    }

    offsets[64] = curr;

    offsets
};

const ALL_DESTINATIONS: [u64; 64] = init!(|sq, 64| {
    let rank = sq / 8;
    let file = sq % 8;

    let rooks = (0xFF << (rank * 8)) ^ (A << file);
    let bishops = DIAGS[file + rank].swap_bytes() ^ DIAGS[7 + file - rank];

    rooks | bishops | KNIGHT[sq] | KING[sq]
});

const A: u64 = 0x0101_0101_0101_0101;
const H: u64 = A << 7;

const DIAGS: [u64; 15] = [
    0x0100_0000_0000_0000,
    0x0201_0000_0000_0000,
    0x0402_0100_0000_0000,
    0x0804_0201_0000_0000,
    0x1008_0402_0100_0000,
    0x2010_0804_0201_0000,
    0x4020_1008_0402_0100,
    0x8040_2010_0804_0201,
    0x0080_4020_1008_0402,
    0x0000_8040_2010_0804,
    0x0000_0080_4020_1008,
    0x0000_0000_8040_2010,
    0x0000_0000_0080_4020,
    0x0000_0000_0000_8040,
    0x0000_0000_0000_0080,
];

const KNIGHT: [u64; 64] = init!(|sq, 64| {
    let n = 1 << sq;
    let h1 = ((n >> 1) & 0x7f7f_7f7f_7f7f_7f7f) | ((n << 1) & 0xfefe_fefe_fefe_fefe);
    let h2 = ((n >> 2) & 0x3f3f_3f3f_3f3f_3f3f) | ((n << 2) & 0xfcfc_fcfc_fcfc_fcfc);
    (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8)
});

const KING: [u64; 64] = init!(|sq, 64| {
    let mut k = 1 << sq;
    k |= (k << 8) | (k >> 8);
    k |= ((k & !A) >> 1) | ((k & !H) << 1);
    k ^ (1 << sq)
});

const SEE_VALS: [i32; 8] = [0, 0, 100, 450, 450, 650, 1250, 0];

fn gain(pos: &Position, mov: &Move) -> i32 {
    if mov.is_en_passant() {
        return SEE_VALS[Piece::PAWN];
    }
    let mut score = SEE_VALS[pos.get_pc(1 << mov.to())];
    if mov.is_promo() {
        score += SEE_VALS[mov.promo_pc()] - SEE_VALS[Piece::PAWN];
    }
    score
}

pub fn see(pos: &Position, mov: &Move) -> i32 {
    let sq = usize::from(mov.to());
    assert!(sq < 64, "wha");

    /* ---------------------------------------------------------------
     *  Pre-move legality checks (king safety, en-passant)
     * ------------------------------------------------------------- */
    let side = pos.stm();               // 0 = white, 1 = black

    // King squares for both colours
    let mut king_sq = [
        pos.king_sq(Side::WHITE),
        pos.king_sq(Side::BLACK),
    ];

    // Piece that makes the initial capture
    let moved = if mov.is_promo() {
        mov.promo_pc()
    } else {
        pos.get_pc(1 << mov.src())
    };

    // Occupancy after applying the capture
    let mut occ =
        (pos.piece(Side::WHITE) | pos.piece(Side::BLACK)) ^ (1 << mov.src()) ^ (1 << sq);
    if mov.is_en_passant() {
        occ ^= 1 << (sq ^ 8);           // remove the e.p. victim
    }
    occ |= 1 << sq;                     // our piece now sits on the target square

    // Update king square if our king moved
    if moved == Piece::KING {
        king_sq[side] = sq;
    }

    /* ---------------------------------------------------------------
     *  Static-exchange evaluation - fully legal
     * ------------------------------------------------------------- */
    let mut next  = moved;
    let mut score = gain(pos, mov) - SEE_VALS[next];

    let bishops = pos.piece(Piece::BISHOP) | pos.piece(Piece::QUEEN);
    let rooks   = pos.piece(Piece::ROOK)   | pos.piece(Piece::QUEEN);

    let mut us = side ^ 1;              // side to move after our capture

    let mut attackers = (Attacks::knight(sq) & pos.piece(Piece::KNIGHT))
        | (Attacks::king(sq)   & pos.piece(Piece::KING))
        | (Attacks::pawn(sq, Side::WHITE) & pos.piece(Piece::PAWN) & pos.piece(Side::BLACK))
        | (Attacks::pawn(sq, Side::BLACK) & pos.piece(Piece::PAWN) & pos.piece(Side::WHITE))
        | (Attacks::rook(sq,   occ) & rooks)
        | (Attacks::bishop(sq, occ) & bishops);

    loop {
        let our_attackers = attackers & pos.piece(us);
        if our_attackers == 0 {
            break;                      // no more recaptures
        }

        /* ---- pick the least-valuable *legal* attacker ---- */
        let mut chosen = 0u64;
        for pc in Piece::PAWN..=Piece::KING {
            let mut bb_pc = our_attackers & pos.piece(pc);
            while bb_pc > 0 {
                let from = bb_pc.trailing_zeros() as usize;
                let bit  = 1u64 << from;

                // Occupancy after this recapture
                let occ_after = (occ ^ bit) | (1 << sq);

                // King square for this side if its king moves
                let ksq = if pc == Piece::KING { sq } else { king_sq[us] };

                // Only accept attacker if its king remains safe
                if !pos.is_square_attacked(ksq, us, occ_after) {
                    chosen = bit;
                    next   = pc;
                    occ    = occ_after;
                    if pc == Piece::KING {
                        king_sq[us] = sq;
                    }
                    break;
                }
                bb_pc &= bb_pc - 1;      // try next attacker of same type
            }
            if chosen != 0 {
                break;                  // we found a legal recapture
            }
        }

        if chosen == 0 {
            break;                      // no legal recapture exists
        }

        // Newly revealed x-ray attacks
        attackers |= Attacks::bishop(sq, occ) & bishops;
        attackers |= Attacks::rook(sq,   occ) & rooks;
        attackers &= occ;

        score = -score - 1 - SEE_VALS[next];
        us   ^= 1;                      // change side

        if score >= 0 {
            if next == Piece::KING && attackers & pos.piece(us) > 0 {
                us ^= 1;                // special: side still has the move
            }
            break;
        }
    }

    // SEE is true when the original side comes out ahead
    if pos.stm() == us { -score } else { score }
}