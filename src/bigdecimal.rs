// SPDX-FileCopyrightText: 2026 Matthew Milner <matterhorn103@proton.me>
// SPDX-License-Identifier: MIT OR Apache-2.0

use bigdecimal::BigDecimal;

pub struct SciBigDecimal {
    number: BigDecimal,
    uncertainty: BigDecimal,
}
