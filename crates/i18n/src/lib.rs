//! Interface strings in 5 languages. Each string is defined as an array of 5 elements
//! in the order [en, de, fr, es, ru] — the compiler won't let you skip a translation.

pub use sr_core::UiLang;
use sr_core::{Lang, Speaker};

macro_rules! keys {
    ($($key:ident => [$en:expr, $de:expr, $fr:expr, $es:expr, $ru:expr $(,)?]),* $(,)?) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum Key { $($key),* }

        impl Key {
            pub const ALL: &'static [Key] = &[$(Key::$key),*];
        }

        pub fn t(lang: UiLang, key: Key) -> &'static str {
            let row: [&'static str; 5] = match key { $(Key::$key => [$en, $de, $fr, $es, $ru]),* };
            row[match lang {
                UiLang::En => 0,
                UiLang::De => 1,
                UiLang::Fr => 2,
                UiLang::Es => 3,
                UiLang::Ru => 4,
            }]
        }
    };
}

keys! {
    StartConversation => ["Start conversation", "Gespräch starten", "Démarrer une conversation", "Iniciar conversación", "Начать беседу"],
    History => ["History", "Verlauf", "Historique", "Historial", "История"],
    Settings => ["Settings", "Einstellungen", "Paramètres", "Ajustes", "Настройки"],
    Back => ["Back", "Zurück", "Retour", "Atrás", "Назад"],
    NewConversation => ["New conversation", "Neues Gespräch", "Nouvelle conversation", "Nueva conversación", "Новая беседа"],
    Title => ["Title", "Titel", "Titre", "Título", "Название"],
    TitlePlaceholder => ["e.g. Call with partners", "z. B. Call mit Partnern", "ex. Appel avec les partenaires", "p. ej. Llamada con socios", "например, Созвон с партнёрами"],
    TranslateInto => ["Translate into", "Übersetzen in", "Traduire en", "Traducir a", "Переводить на"],
    SpeakersLanguage => ["Speakers' language", "Sprache der Sprecher", "Langue des interlocuteurs", "Idioma de los hablantes", "Язык собеседников"],
    Auto => ["Auto", "Auto", "Auto", "Auto", "Авто"],
    Microphone => ["Microphone", "Mikrofon", "Microphone", "Micrófono", "Микрофон"],
    MicHint => ["Also transcribe my microphone — my lines are marked “Me” (use headphones to avoid echo)", "Auch mein Mikrofon erfassen — meine Beiträge heißen „Ich“ (Kopfhörer vermeiden Echo)", "Transcrire aussi mon micro — mes répliques sont marquées « Moi » (casque conseillé contre l’écho)", "Transcribir también mi micrófono — mis frases se marcan «Yo» (usa auriculares para evitar eco)", "Слушать и мой микрофон — мои реплики помечаются «Я» (лучше в наушниках, иначе будет эхо)"],
    SpeechModel => ["Speech recognition", "Spracherkennung", "Reconnaissance vocale", "Reconocimiento de voz", "Распознавание речи"],
    TranslationModel => ["Translation", "Übersetzung", "Traduction", "Traducción", "Перевод"],
    Recommended => ["recommended", "empfohlen", "recommandé", "recomendado", "рекомендуется"],
    Start => ["Start", "Starten", "Démarrer", "Iniciar", "Начать"],
    PressToStart => ["to start translating", "zum Starten", "pour démarrer", "para empezar", "— начать перевод"],
    Translating => ["Translating", "Übersetzung läuft", "Traduction en cours", "Traduciendo", "Идёт перевод"],
    Paused => ["Paused", "Pausiert", "En pause", "En pausa", "Пауза"],
    Dimming => ["Dimming", "Abdunkeln", "Assombrir", "Oscurecer", "Затемнение"],
    Compact => ["Compact mode", "Kompaktmodus", "Mode compact", "Modo compacto", "Компактный режим"],
    Expand => ["Full mode", "Vollmodus", "Mode complet", "Modo completo", "Полный режим"],
    Theme => ["Theme", "Design", "Thème", "Tema", "Тема"],
    ThemeSystem => ["System", "System", "Système", "Sistema", "Системная"],
    ThemeLight => ["Light", "Hell", "Clair", "Claro", "Светлая"],
    ThemeDark => ["Dark", "Dunkel", "Sombre", "Oscuro", "Тёмная"],
    InterfaceLanguage => ["Interface language", "Sprache der Oberfläche", "Langue de l’interface", "Idioma de la interfaz", "Язык интерфейса"],
    ShowOriginal => ["Show original text under translation", "Originaltext unter der Übersetzung zeigen", "Afficher le texte original sous la traduction", "Mostrar el texto original bajo la traducción", "Показывать оригинал под переводом"],
    GlobalHotkey => ["Global hotkey (works in any app)", "Globaler Hotkey (in jeder App)", "Raccourci global (dans toute application)", "Atajo global (en cualquier aplicación)", "Глобальная клавиша (работает в любом приложении)"],
    Me => ["Me", "Ich", "Moi", "Yo", "Я"],
    SpeakerN => ["Speaker", "Sprecher", "Intervenant", "Hablante", "Спикер"],
    NotRecognized => ["speech not recognized", "Sprache nicht erkannt", "parole non reconnue", "voz no reconocida", "речь не распознана"],
    TranslationFailed => ["translation failed", "Übersetzung fehlgeschlagen", "échec de la traduction", "error de traducción", "ошибка перевода"],
    NewLines => ["New", "Neu", "Nouveau", "Nuevo", "Новые"],
    HistoryEmpty => ["No conversations yet", "Noch keine Gespräche", "Aucune conversation pour l’instant", "Aún no hay conversaciones", "Бесед пока нет"],
    OpenFolder => ["Open folder", "Ordner öffnen", "Ouvrir le dossier", "Abrir carpeta", "Открыть папку"],
    EndConversation => ["End conversation", "Gespräch beenden", "Terminer la conversation", "Terminar conversación", "Завершить беседу"],
    WaitingForSpeech => ["Speech will appear here once translation starts", "Hier erscheint die Rede, sobald die Übersetzung läuft", "Le texte apparaîtra ici dès le début de la traduction", "El texto aparecerá aquí al iniciar la traducción", "Здесь появится перевод, когда начнётся запись"],
    DemoNote => ["Demo mode: scripted dialogue instead of real audio", "Demo-Modus: vorbereiteter Dialog statt echtem Ton", "Mode démo : dialogue préparé au lieu de l’audio réel", "Modo demo: diálogo preparado en lugar de audio real", "Демо-режим: заготовленный диалог вместо настоящего звука"],
    LoadingSpeechModel => ["Loading speech recognition model…", "Spracherkennungsmodell wird geladen…", "Chargement du modèle de reconnaissance vocale…", "Cargando el modelo de reconocimiento de voz…", "Загружается модель распознавания речи…"],
    LoadingTranslationModel => ["Loading translation model…", "Übersetzungsmodell wird geladen…", "Chargement du modèle de traduction…", "Cargando el modelo de traducción…", "Загружается модель перевода…"],
    TranslationUnavailable => ["Translation unavailable, showing original text", "Übersetzung nicht verfügbar, Originaltext wird angezeigt", "Traduction indisponible, texte original affiché", "Traducción no disponible, se muestra el texto original", "Перевод недоступен, показываю оригинал"],
    TranslatingHint => ["translating…", "wird übersetzt…", "traduction…", "traduciendo…", "перевод…"],
    Diarization => ["Tell speakers apart (Speaker-1, Speaker-2…)", "Sprecher unterscheiden (Sprecher-1, Sprecher-2…)", "Distinguer les intervenants (Intervenant-1, Intervenant-2…)", "Distinguir hablantes (Hablante-1, Hablante-2…)", "Различать собеседников (Спикер-1, Спикер-2…)"],
    DiarizationUnavailable => ["Speaker separation unavailable", "Sprechertrennung nicht verfügbar", "Séparation des intervenants indisponible", "Separación de hablantes no disponible", "Различение собеседников недоступно"],
    RenameHint => ["Click a name to rename", "Zum Umbenennen auf den Namen klicken", "Cliquez sur un nom pour le renommer", "Haz clic en un nombre para cambiarlo", "Нажмите на имя, чтобы переименовать"],
    ModelNotFound => ["Model file not found", "Modelldatei nicht gefunden", "Fichier du modèle introuvable", "No se encontró el archivo del modelo", "Файл модели не найден"],
    Close => ["Close", "Schließen", "Fermer", "Cerrar", "Закрыть"],
    Untitled => ["Untitled conversation", "Unbenanntes Gespräch", "Conversation sans titre", "Conversación sin título", "Беседа без названия"],
    // Model comments
    AsrTurboNote => ["Best balance of speed and quality. Needs a GPU.", "Beste Balance aus Tempo und Qualität. Braucht eine GPU.", "Meilleur équilibre vitesse/qualité. Nécessite un GPU.", "Mejor equilibrio entre velocidad y calidad. Requiere GPU.", "Лучший баланс скорости и качества. Нужна видеокарта."],
    AsrMediumNote => ["More accurate on noisy speech, but slower.", "Genauer bei Störgeräuschen, aber langsamer.", "Plus précis avec du bruit, mais plus lent.", "Más preciso con ruido, pero más lento.", "Точнее на шумной речи, но медленнее."],
    AsrSmallNote => ["For weak computers and CPU-only.", "Für schwache Rechner und nur CPU.", "Pour les ordinateurs modestes et le CPU seul.", "Para equipos modestos y solo CPU.", "Для слабых компьютеров и работы на процессоре."],
    MtFastNote => ["Fastest, also translates text while it's being spoken. Simpler, sometimes inaccurate.", "Am schnellsten, übersetzt schon während des Sprechens. Einfacher, manchmal ungenau.", "Le plus rapide, traduit déjà pendant la parole. Plus simple, parfois imprécis.", "El más rápido, traduce ya mientras se habla. Más simple, a veces impreciso.", "Самый быстрый, переводит ещё по ходу речи. Перевод проще, бывают неточности."],
    MtMidNote => ["Natural translation that uses context of previous lines.", "Natürliche Übersetzung mit Kontext der vorigen Sätze.", "Traduction naturelle qui tient compte du contexte.", "Traducción natural que usa el contexto previo.", "Естественный перевод с учётом предыдущих реплик."],
    MtLargeNote => ["Best quality. Needs 8+ GB VRAM or a Mac with 16+ GB.", "Beste Qualität. Braucht 8+ GB VRAM oder Mac mit 16+ GB.", "Meilleure qualité. 8+ Go de VRAM ou Mac 16+ Go.", "Mejor calidad. Requiere 8+ GB de VRAM o Mac con 16+ GB.", "Лучшее качество. Нужно 8+ ГБ видеопамяти или Mac с 16+ ГБ."],
}

/// Speaker label: "Me" or "Speaker-2".
pub fn speaker_label(lang: UiLang, s: Speaker) -> String {
    match s {
        Speaker::Me => t(lang, Key::Me).to_string(),
        Speaker::Remote(n) => format!("{}-{n}", t(lang, Key::SpeakerN)),
    }
}

/// Translation direction label: "en→ru".
pub fn direction(src: Option<Lang>, dst: Lang) -> String {
    match src {
        Some(s) if s != dst => format!("{}→{}", s.code(), dst.code()),
        Some(s) => s.code().to_string(),
        None => format!("?→{}", dst.code()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_key_translated_and_non_empty() {
        for &k in Key::ALL {
            for l in UiLang::ALL {
                assert!(!t(l, k).trim().is_empty(), "{k:?} пусто для {l:?}");
            }
        }
    }

    #[test]
    fn labels() {
        assert_eq!(speaker_label(UiLang::Ru, Speaker::Remote(2)), "Спикер-2");
        assert_eq!(speaker_label(UiLang::En, Speaker::Me), "Me");
        assert_eq!(direction(Some(Lang::En), Lang::Ru), "en→ru");
        assert_eq!(direction(Some(Lang::Ru), Lang::Ru), "ru");
    }
}
