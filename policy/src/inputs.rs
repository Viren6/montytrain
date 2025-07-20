use montyformat::chess::{Castling, Flag, Move, Piece, Position, Side};

pub const MAX_MOVES: usize = 64;
pub const INPUT_SIZE: usize = 768 * 4;
pub const MAX_ACTIVE_BASE: usize = 32;

pub fn map_base_inputs<F: FnMut(usize)>(pos: &Position, threats: u64, defences: u64, mut f: F) {
    let vert = if pos.stm() == Side::BLACK { 56 } else { 0 };
    let hori = if pos.king_index() % 8 > 3 { 7 } else { 0 };
    let flip = vert ^ hori;

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

pub fn get_diff(pos: &Position, castling: &Castling, mov: Move, threats: u64, defences: u64) -> [i32; 4] {
    let vert = if pos.stm() == Side::BLACK { 56 } else { 0 };
    let hori = if pos.king_index() % 8 > 3 { 7 } else { 0 };
    let flip = vert ^ hori;

    let idx = |stm, pc, sq| {
        let mut feat = ([0, 384][stm] + 64 * (pc - 2) + (sq ^ flip)) as i32;

        let bit = 1u64 << sq;
        if threats & bit > 0 {
            feat += 768;
        }

        if defences & bit > 0 {
            feat += 768 * 2;
        }

        feat
    };

    let mut diff = [-1; 4];

    let src = mov.src() as usize;
    let dst = mov.to() as usize;

    let moved = pos.get_pc(1 << src);
    diff[0] = INPUT_SIZE as i32 + idx(0, moved, src);

    if mov.is_en_passant() {
        diff[1] = INPUT_SIZE as i32 + idx(1, Piece::PAWN, dst ^ 8);
    } else if mov.is_capture() {
        diff[1] = INPUT_SIZE as i32 + idx(1, pos.get_pc(1 << dst), dst);
    }

    if mov.is_promo() {
        let promo = usize::from((mov.flag() & 3) + 3);
        diff[2] = idx(0, promo, dst);
    } else {
        diff[2] = idx(0, moved, dst);
    }

    if mov.flag() == Flag::KS || mov.flag() == Flag::QS {
        assert_eq!(diff[1], -1);

        let ks = usize::from(mov.flag() == Flag::KS);
        let sf = 56 * pos.stm();

        diff[1] = INPUT_SIZE as i32 + idx(0, Piece::ROOK, sf + castling.rook_file(pos.stm(), ks) as usize);
        diff[3] = idx(0, Piece::ROOK, sf + [3, 5][ks]);
    }

    for i in diff {
        assert!(i < 2 * INPUT_SIZE as i32);
        assert!(i >= -1);
    }

    diff
}
