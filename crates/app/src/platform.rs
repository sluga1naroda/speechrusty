//! Platform-specific window details and the global hotkey.

use gpui_kit::{Window, WindowKind};

/// macOS: NSFloatingWindowLevel panel — gpui does this itself for `Floating`.
/// Windows: a regular window (has a frame for resizing and a taskbar icon) + HWND_TOPMOST.
pub fn window_kind() -> WindowKind {
    if cfg!(target_os = "macos") { WindowKind::Floating } else { WindowKind::Normal }
}

#[cfg(windows)]
pub fn make_topmost(window: &Window) {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use windows::Win32::Foundation::HWND;
    use windows::Win32::UI::WindowsAndMessaging::{HWND_TOPMOST, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOSIZE, SetWindowPos};
    let Ok(h) = HasWindowHandle::window_handle(window) else { return };
    if let RawWindowHandle::Win32(w) = h.as_raw() {
        let hwnd = HWND(w.hwnd.get() as _);
        // SAFETY: hwnd is a live window created by gpui on this same thread.
        if let Err(e) = unsafe { SetWindowPos(hwnd, Some(HWND_TOPMOST), 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE) } {
            log::warn!("SetWindowPos(HWND_TOPMOST) failed: {e}");
        }
    }
}

#[cfg(not(windows))]
pub fn make_topmost(_: &Window) {}

/// Hotkey label for the current OS.
pub fn local_hotkey_label() -> &'static str {
    if cfg!(target_os = "macos") { "⌘R" } else { "Ctrl+R" }
}

pub fn global_hotkey_label() -> &'static str {
    if cfg!(target_os = "macos") { "⌘⇧R" } else { "Ctrl+Shift+R" }
}

/// Global hotkey (works even when the window is not focused).
pub struct GlobalHotkey {
    manager: global_hotkey::GlobalHotKeyManager,
    hotkey: global_hotkey::hotkey::HotKey,
    registered: bool,
}

impl GlobalHotkey {
    pub fn new() -> anyhow::Result<Self> {
        use global_hotkey::hotkey::{Code, HotKey, Modifiers};
        let mods = if cfg!(target_os = "macos") { Modifiers::META | Modifiers::SHIFT } else { Modifiers::CONTROL | Modifiers::SHIFT };
        Ok(Self { manager: global_hotkey::GlobalHotKeyManager::new()?, hotkey: HotKey::new(Some(mods), Code::KeyR), registered: false })
    }

    pub fn set_enabled(&mut self, on: bool) -> anyhow::Result<()> {
        if on && !self.registered {
            self.manager.register(self.hotkey)?;
            self.registered = true;
        } else if !on && self.registered {
            self.manager.unregister(self.hotkey)?;
            self.registered = false;
        }
        Ok(())
    }

    /// Whether it was pressed since the last poll.
    pub fn poll_pressed(&self) -> bool {
        let mut pressed = false;
        while let Ok(ev) = global_hotkey::GlobalHotKeyEvent::receiver().try_recv() {
            if ev.id == self.hotkey.id() && ev.state == global_hotkey::HotKeyState::Pressed {
                pressed = true;
            }
        }
        pressed
    }
}
