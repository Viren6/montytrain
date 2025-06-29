use bullet::default::formats::montyformat::chess::{Attacks, Move, Piece, Position, Side};

pub const MAX_MOVES: usize = 96;
pub const NUM_MOVES: usize = 2 * (OFFSETS[64] + PROMOS);
pub const PROMOS: usize = 4 * 22;

pub fn map_move_to_index(pos: &Position, mov: Move) -> usize {
    let hm = if pos.king_index() % 8 > 3 { 7 } else { 0 };
    let good_see = (OFFSETS[64] + PROMOS) * usize::from(see(pos, &mov, -108));

    let idx = if mov.is_promo() {
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
    };

    good_see + idx
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

fn see(pos: &Position, mov: &Move, threshold: i32) -> bool {
    let from_sq = usize::from(mov.src());
    let to_sq = usize::from(mov.to());
    let side = pos.stm();

    // The piece that makes the initial move.
    let moved_pc = if mov.is_promo() {
        mov.promo_pc()
    } else {
        pos.get_pc(1 << from_sq)
    };

    // --- Start of rewritten logic ---

    // 1. Calculate initial gain and check against the threshold.
    // This is the best-case scenario for the moving side.
    let mut gain = gain(pos, mov);
    if gain < threshold {
        return false;
    }

    // 2. Setup the board state for the exchange sequence.
    let mut occ = pos.occ() ^ (1 << from_sq) ^ (1 << to_sq);
    if mov.is_en_passant() {
        occ ^= 1 << (to_sq ^ 8);
    }

    let mut king_sq = [pos.king_sq(Side::WHITE), pos.king_sq(Side::BLACK)];
    if moved_pc == Piece::KING {
        king_sq[side] = to_sq;
    }

    // After the initial move, the piece on the square is `moved_pc`.
    // Its value is what the opponent can win next.
    let mut captured_val = SEE_VALS[moved_pc];
    let mut current_side = side ^ 1;

    // 3. Find all potential attackers to the square.
    let bishops = pos.piece(Piece::BISHOP) | pos.piece(Piece::QUEEN);
    let rooks = pos.piece(Piece::ROOK) | pos.piece(Piece::QUEEN);
    let mut attackers = (Attacks::knight(to_sq) & pos.piece(Piece::KNIGHT))
        | (Attacks::king(to_sq) & pos.piece(Piece::KING))
        | (Attacks::pawn(to_sq, Side::WHITE) & pos.piece(Piece::PAWN) & pos.piece(Side::BLACK))
        | (Attacks::pawn(to_sq, Side::BLACK) & pos.piece(Piece::PAWN) & pos.piece(Side::WHITE))
        | (Attacks::rook(to_sq, occ) & rooks)
        | (Attacks::bishop(to_sq, occ) & bishops);

    loop {
        // 4. Check if the current side can recapture.
        let our_attackers = attackers & pos.piece(current_side);
        if our_attackers == 0 {
            break;
        }

        // 5. Find the least valuable, legal attacker.
        let mut recapturer_sq = 64;
        let mut recapturing_pc = Piece::EMPTY;
        for pc in Piece::PAWN..=Piece::KING {
            let mut bb_pc = our_attackers & pos.piece(pc);
            if bb_pc > 0 {
                let from = bb_pc.trailing_zeros() as usize;

                // Legality Check: Does this recapture leave the king in check?
                let ksq = if pc == Piece::KING { to_sq } else { king_sq[current_side] };
                let next_occ = occ ^ (1 << from); // Occupancy if this piece moves
                if !pos.is_square_attacked(ksq, current_side, next_occ) {
                    recapturer_sq = from;
                    recapturing_pc = pc;
                    break;
                }
            }
        }

        if recapturing_pc == Piece::EMPTY {
            break; // No legal recaptures found.
        }

        // 6. Update state for the next iteration.
        occ ^= 1 << recapturer_sq; // The recapturing piece is removed from its square.
        if recapturing_pc == Piece::KING {
            king_sq[current_side] = to_sq;
        }

        // The current side loses `recapturing_pc` but wins `captured_val`.
        // We update `gain` from the perspective of the *initial* mover.
        gain -= captured_val;

        // The new "captured" piece is the one that just moved.
        captured_val = SEE_VALS[recapturing_pc];

        // Pruning: if the exchange is already losing for the initial side, stop.
        if gain < threshold {
            return false;
        }

        // Optimization: if the opponent's recapture value is greater than the
        // current gain, they can't make the exchange profitable for them.
        // So the initial side comes out ahead.
        if captured_val > gain {
            return true;
        }

        // 7. Update attackers for x-rays using the NEW occupancy.
        attackers &= occ; // Remove the piece that just moved.
        if [Piece::PAWN, Piece::BISHOP, Piece::QUEEN].contains(&recapturing_pc) {
            attackers |= Attacks::bishop(to_sq, occ) & bishops;
        }
        if [Piece::ROOK, Piece::QUEEN].contains(&recapturing_pc) {
            attackers |= Attacks::rook(to_sq, occ) & rooks;
        }

        current_side ^= 1;
    }

    // If the loop terminates, the final gain determines the result.
    gain >= threshold
}
