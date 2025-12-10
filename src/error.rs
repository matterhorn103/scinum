// SPDX-FileCopyrightText: 2025 Matthew Milner <matterhorn103@proton.me>
// SPDX-License-Identifier: MIT

use std::{error::Error, fmt};

#[derive(Clone, Debug)]
pub enum SciNumError {
    Parse(String),
    Cast(String),
}

impl fmt::Display for SciNumError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SciNumError::Parse(string) => write!(f, "Failed to parse: {string}"),
            SciNumError::Cast(t) => write!(f, "Failed to cast to {t}"),
        }
    }
}

impl Error for SciNumError {}
