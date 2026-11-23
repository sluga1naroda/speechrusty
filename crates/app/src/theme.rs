//! Light and dark theme palettes. Colors are defined here and only here.

use gpui_kit::{Hsla, hsla};
use sr_core::Speaker;

#[derive(Clone, Copy, Debug)]
pub struct Palette {
    pub dark: bool,
    /// Window background; alpha comes from the dimming slider.
    pub bg: Hsla,
    /// Backdrop for cards and fields (on top of the translucent background).
    pub surface: Hsla,
    pub surface_hover: Hsla,
    pub fg: Hsla,
    pub muted: Hsla,
    pub faint: Hsla,
    pub border: Hsla,
    pub accent: Hsla,
    pub accent_fg: Hsla,
    pub rec: Hsla,
    pub warn: Hsla,
    speakers: [Hsla; 8],
}

fn hex(h: f32, s: f32, l: f32) -> Hsla {
    hsla(h / 360.0, s, l, 1.0)
}

impl Palette {
    pub fn dark() -> Self {
        Self {
            dark: true,
            bg: hex(228., 0.18, 0.08),
            surface: hsla(228. / 360., 0.20, 0.16, 0.85),
            surface_hover: hsla(228. / 360., 0.22, 0.22, 0.95),
            fg: hex(220., 0.15, 0.94),
            muted: hex(222., 0.10, 0.66),
            faint: hex(222., 0.08, 0.46),
            border: hsla(232. / 360., 0.30, 0.60, 0.28),
            accent: hex(239., 0.84, 0.70),
            accent_fg: hex(0., 0., 1.0),
            rec: hex(0., 0.84, 0.60),
            warn: hex(38., 0.92, 0.60),
            speakers: [
                hex(199., 0.89, 0.66), // light blue
                hex(28., 0.92, 0.64),  // orange
                hex(292., 0.70, 0.72), // lilac
                hex(48., 0.90, 0.60),  // yellow
                hex(340., 0.80, 0.70), // pink
                hex(172., 0.66, 0.55), // turquoise
                hex(16., 0.80, 0.66),  // coral
                hex(262., 0.75, 0.75), // violet
            ],
        }
    }

    pub fn light() -> Self {
        Self {
            dark: false,
            bg: hex(225., 0.30, 0.97),
            surface: hsla(0., 0., 1.0, 0.85),
            surface_hover: hsla(228. / 360., 0.40, 0.93, 1.0),
            fg: hex(228., 0.28, 0.12),
            muted: hex(226., 0.12, 0.38),
            faint: hex(226., 0.10, 0.56),
            border: hsla(230. / 360., 0.25, 0.40, 0.25),
            accent: hex(239., 0.70, 0.52),
            accent_fg: hex(0., 0., 1.0),
            rec: hex(0., 0.75, 0.50),
            warn: hex(32., 0.95, 0.42),
            speakers: [
                hex(203., 0.85, 0.38),
                hex(24., 0.90, 0.42),
                hex(290., 0.55, 0.45),
                hex(42., 0.95, 0.32),
                hex(338., 0.70, 0.45),
                hex(174., 0.75, 0.28),
                hex(12., 0.75, 0.46),
                hex(262., 0.55, 0.50),
            ],
        }
    }

    /// Speaker color. "Me" gets the accent color, the rest cycle through 8 colors.
    pub fn speaker(&self, s: Speaker) -> Hsla {
        match s {
            Speaker::Me => self.accent,
            Speaker::Remote(n) => self.speakers[(n.max(1) as usize - 1) % self.speakers.len()],
        }
    }
}
