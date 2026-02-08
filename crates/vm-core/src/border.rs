#[derive(Debug, Clone, PartialEq)]
pub enum BorderMode<T> {
    Clamp,
    Constant(T),
    Reflect101,
}

pub fn map_index<T>(i: isize, len: usize, mode: &BorderMode<T>) -> Option<usize> {
    match mode {
        BorderMode::Constant(_) => None,
        BorderMode::Clamp => {
            if len == 0 {
                return None;
            }
            if i < 0 {
                Some(0)
            } else {
                let idx = i as usize;
                Some(idx.min(len - 1))
            }
        }
        BorderMode::Reflect101 => {
            if len == 0 {
                return None;
            }
            if len == 1 {
                return Some(0);
            }

            let period = (2 * len - 2) as isize;
            let r = i.rem_euclid(period) as usize;
            if r < len {
                Some(r)
            } else {
                Some((2 * len - 2) - r)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BorderMode, map_index};

    #[test]
    fn clamp_mapping_handles_negative_and_overflow() {
        let mode = BorderMode::<u8>::Clamp;

        assert_eq!(map_index(-3, 5, &mode), Some(0));
        assert_eq!(map_index(-1, 5, &mode), Some(0));
        assert_eq!(map_index(0, 5, &mode), Some(0));
        assert_eq!(map_index(4, 5, &mode), Some(4));
        assert_eq!(map_index(5, 5, &mode), Some(4));
        assert_eq!(map_index(99, 5, &mode), Some(4));
    }

    #[test]
    fn reflect101_len1_len2_len5() {
        let mode = BorderMode::<u8>::Reflect101;

        for i in -8..=8 {
            assert_eq!(map_index(i, 1, &mode), Some(0));
        }

        let expected_len2 = [0, 1, 0, 1, 0, 1, 0, 1, 0];
        for (offset, expected) in (-4..=4).zip(expected_len2) {
            assert_eq!(map_index(offset, 2, &mode), Some(expected));
        }

        let cases_len5 = [
            (-7, 1),
            (-6, 2),
            (-5, 3),
            (-4, 4),
            (-3, 3),
            (-2, 2),
            (-1, 1),
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 3),
            (6, 2),
            (7, 1),
        ];

        for (i, expected) in cases_len5 {
            assert_eq!(map_index(i, 5, &mode), Some(expected));
        }
    }
}
