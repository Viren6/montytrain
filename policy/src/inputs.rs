use montyformat::chess::{Attacks, Move, Piece, Position, Side};

pub const MAX_MOVES: usize = 64;
pub const INPUT_SIZE: usize = 768 * 4;
pub const MAX_ACTIVE_BASE: usize = 32;

pub fn map_move_to_index(pos: &Position, mov: Move) -> i32 {
    let dst_idx = map_sq(pos, mov.to()) + 64 * i32::from(see(pos, mov, -108));
    64 * dst_idx + map_sq(pos, mov.src())
}

fn map_sq(pos: &Position, sq: u16) -> i32 {
    let vert = if pos.stm() == Side::BLACK { 56 } else { 0 };
    let hori = if pos.king_index() % 8 > 3 { 7 } else { 0 };
    i32::from(sq ^ vert ^ hori)
}

pub fn map_base_inputs<F: FnMut(usize)>(pos: &Position, mut f: F) {
    let vert = if pos.stm() == Side::BLACK { 56 } else { 0 };
    let hori = if pos.king_index() % 8 > 3 { 7 } else { 0 };
    let flip = vert ^ hori;

    let threats = pos.threats_by(pos.stm() ^ 1);
    let defences = pos.threats_by(pos.stm());

    for piece in Piece::PAWN..=Piece::KING {
        let pc = 64 * (piece - 2);

        let mut our_bb = pos.piece(piece) & pos.piece(pos.stm());
        let mut opp_bb = pos.piece(piece) & pos.piece(pos.stm() ^ 1);

        while our_bb > 0 {
            let sq = our_bb.trailing_zeros() as usize;
            let mut feat = pc + (sq ^ flip);

            let bit = 1 << sq;
            if threats & bit > 0 {
                feat += 768;
            }

            if defences & bit > 0 {
                feat += 768 * 2;
            }

            f(feat);

            our_bb &= our_bb - 1;
        }

        while opp_bb > 0 {
            let sq = opp_bb.trailing_zeros() as usize;
            let mut feat = 384 + pc + (sq ^ flip);

            let bit = 1 << sq;
            if threats & bit > 0 {
                feat += 768;
            }

            if defences & bit > 0 {
                feat += 768 * 2;
            }

            f(feat);

            opp_bb &= opp_bb - 1;
        }
    }
}

const SEE_VALS: [i32; 8] = [0, 0, 100, 450, 450, 650, 1250, 0];

fn gain(pos: &Position, mov: Move) -> i32 {
    if mov.is_en_passant() {
        return SEE_VALS[Piece::PAWN];
    }
    let mut score = SEE_VALS[pos.get_pc(1 << mov.to())];
    if mov.is_promo() {
        score += SEE_VALS[mov.promo_pc()] - SEE_VALS[Piece::PAWN];
    }
    score
}

fn see(pos: &Position, mov: Move, threshold: i32) -> bool {
    let sq = usize::from(mov.to());
    assert!(sq < 64, "wha");
    let mut next = if mov.is_promo() { mov.promo_pc() } else { pos.get_pc(1 << mov.src()) };
    let mut score = gain(pos, mov) - threshold - SEE_VALS[next];

    if score >= 0 {
        return true;
    }

    let mut occ = (pos.piece(Side::WHITE) | pos.piece(Side::BLACK)) ^ (1 << mov.src()) ^ (1 << sq);
    if mov.is_en_passant() {
        occ ^= 1 << (sq ^ 8);
    }

    let bishops = pos.piece(Piece::BISHOP) | pos.piece(Piece::QUEEN);
    let rooks = pos.piece(Piece::ROOK) | pos.piece(Piece::QUEEN);
    let mut us = pos.stm() ^ 1;
    let mut attackers = (Attacks::knight(sq) & pos.piece(Piece::KNIGHT))
        | (Attacks::king(sq) & pos.piece(Piece::KING))
        | (Attacks::pawn(sq, Side::WHITE) & pos.piece(Piece::PAWN) & pos.piece(Side::BLACK))
        | (Attacks::pawn(sq, Side::BLACK) & pos.piece(Piece::PAWN) & pos.piece(Side::WHITE))
        | (Attacks::rook(sq, occ) & rooks)
        | (Attacks::bishop(sq, occ) & bishops);

    loop {
        let our_attackers = attackers & pos.piece(us);
        if our_attackers == 0 {
            break;
        }

        for pc in Piece::PAWN..=Piece::KING {
            let board = our_attackers & pos.piece(pc);
            if board > 0 {
                occ ^= board & board.wrapping_neg();
                next = pc;
                break;
            }
        }

        if [Piece::PAWN, Piece::BISHOP, Piece::QUEEN].contains(&next) {
            attackers |= Attacks::bishop(sq, occ) & bishops;
        }
        if [Piece::ROOK, Piece::QUEEN].contains(&next) {
            attackers |= Attacks::rook(sq, occ) & rooks;
        }

        attackers &= occ;
        score = -score - 1 - SEE_VALS[next];
        us ^= 1;

        if score >= 0 {
            if next == Piece::KING && attackers & pos.piece(us) > 0 {
                us ^= 1;
            }
            break;
        }
    }

    pos.stm() != us
}
