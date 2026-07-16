"""Minimal i18n for the desktop editor — dictionary-based, no Qt
QTranslator/.ts files (keeps translation + code in one place, easy to
extend). Language changes take effect after restarting the app: widget
text is only read through `tr()` at construction time, so switching
`set_language()` mid-session wouldn't retroactively update anything
already on screen without a full UI rebuild — see `_set_language` in
main_window.py, which saves the preference and prompts for a restart.
"""

from __future__ import annotations

import json
import os

_LANG_FILE = os.path.join(os.path.expanduser("~"), ".comiccolorer_lang.json")

_STRINGS: dict = {
    "zh": {
        "window_title": "Colortina",
        "device_label": "设备：",
        "ready": "就绪",

        # Left panel
        "import_images": "＋ 导入图片",
        "import_pdf": "＋ 导入 PDF",
        "import_folder": "＋ 导入文件夹",
        "folder_no_images": "该文件夹中未找到图片",
        "pages_label": "页面（可多选，Ctrl/Shift 或全选按钮）",
        "select_all": "全选",
        "select_none": "取消全选",
        "move_up": "↑ 上移",
        "move_down": "↓ 下移",
        "delete_pages": "🗑 删除选中页面",

        # Canvas toolbar
        "fit_view": "适应窗口",

        # Style / quality group
        "style_quality_group": "风格",
        "skip_colored_checkbox": "跳过已上色页面",
        "skip_colored_tooltip": "检测到页面已包含彩色内容时直接跳过，不重新上色（借鉴 Manga-Colorization-FJ）",
        "style_label": "风格",
        "quality_label": "质量",
        "extract_style_btn": "从参考图提取风格...（可多选）",
        "style_profile_unset": "未使用提取的风格（使用上方预设）",
        "character_memory_checkbox": "角色一致性（不同角色发色分别记忆，整本一致）",
        "character_memory_tooltip": (
            "开启后，同一本书里不同角色的头发会分别记住各自的颜色，\n"
            "而不是全书所有头发都涂成同一个颜色。基于线稿灰度/色调区分角色，\n"
            "色调非常接近的两个角色可能仍会共用同一个颜色。"),
        "extract_style_dialog_title": "选择彩色参考图（封面/彩页，可多选）",
        "extract_style_fail_title": "提取失败",
        "extract_style_fail_body": "无法读取选中的图片",
        "extract_style_result": (
            "已使用提取的风格：{name}（来自 {n} 张参考图；饱和度 {saturation:.2f}，"
            "对比度 {contrast:.2f}，{temperature}调；角色一致性已播种 {seeded} 个发色）"),
        "extract_style_status": "风格已提取并保存：{name}.ccstyle",
        "style_profile_active": "当前使用自定义风格：{name}（饱和度 {saturation:.2f}，对比度 {contrast:.2f}，{temperature}调）",

        # Custom style library (new / load / delete)
        "custom_style_group": "自定义风格库",
        "custom_style_combo_placeholder": "（无已保存的自定义风格）",
        "btn_new_style": "＋ 新建风格（从参考图提取）...",
        "btn_load_style_file": "📂 加载风格文件（.ccstyle）...",
        "btn_apply_saved_style": "使用选中风格",
        "btn_delete_style": "🗑 删除选中风格",
        "btn_clear_style_profile": "✕ 清除自定义风格（改用上方预设）",
        "new_style_name_title": "新建风格",
        "new_style_name_label": "为这个风格起个名字：",
        "new_style_name_default": "我的风格",
        "style_deleted_msg": "已删除风格：{name}",
        "style_loaded_msg": "已加载风格：{name}",
        "style_cleared_msg": "已清除自定义风格，改用预设：{name}",
        "no_style_selected_msg": "请先在列表中选择一个已保存的风格",
        "confirm_delete_style_title": "确认删除风格",
        "confirm_delete_style_body": "确定要删除风格「{name}」吗？此操作不可撤销。",
        "load_style_dialog_title": "选择风格文件（.ccstyle）",
        "load_style_fail_title": "加载失败",
        "load_style_fail_body": "无法加载该风格文件：{exc}",

        # Language switch button (top toolbar)
        "lang_button": "🌐 中文 / EN",

        # Auto colorize
        "auto_group": "自动上色",
        "auto_btn": "自动上色（选中的页面）",

        # Edit tools
        "edit_group": "编辑（手动颜色提示）",
        "tool_brush": "画笔",
        "tool_eyedropper": "吸管",
        "tool_bucket": "区域上色",
        "bucket_hint": ("区域上色：点一下，按线稿/颜色边界整片重新上色\n"
                       "（直接在当前结果上编辑，秒出效果，不用重新跑模型）"),
        "color_label": "颜色",
        "brush_size_label": "笔刷大小",
        "gap_close_label": "线稿缺口修补",
        "gap_close_tooltip": (
            "数值越大，越能自动'封住'线稿上的小缺口，避免区域上色\n"
            "漏到相邻区域；如果发现颜色漏出了边界，调大这个值（0-100）"),
        "fill_mode_label": "区域上色模式",
        "fill_mode_shift": "自然色相迁移（推荐，保留原有笔触质感）",
        "fill_mode_shading": "统一色相，保留明暗",
        "fill_mode_flat": "完全统一纯色（修复严重错色/发黑）",
        "fill_mode_hint": ("如果发现填色后显得很\"平\"、跟周围格格不入，先用第一种；\n"
                          "如果区域内本身黑一块红一块很乱，第一种可能修不干净，换第三种"),
        "undo_last_hint": "撤销上一笔",
        "clear_manual_hints": "清除手动笔画",
        "regenerate_btn": "重新生成（选中的页面，含手动提示）",
        "undo_edit": "↶ 撤销 (Ctrl+Z)",
        "redo_edit": "↷ 重做 (Ctrl+Shift+Z)",

        # View / restore
        "view_group": "查看 / 恢复",
        "view_original": "原图（黑白）",
        "view_ai": "AI结果（最近一次自动上色）",
        "view_edited": "编辑后（当前，含区域上色）",
        "restore_ai": "恢复到AI结果（丢弃区域上色修改）",
        "restore_bw": "重置这一页（清空上色，回到黑白原图）",

        # Export
        "export_group": "导出",
        "export_page": "导出当前页",
        "export_all": "导出全部页面",

        # Menu / language
        "menu_language": "语言 / Language",
        "lang_zh": "简体中文",
        "lang_en": "English",
        "lang_restart_title": "需要重启",
        "lang_restart_body": "语言设置已保存，重启程序后生效。",

        # Misc status/messages
        "no_result_title": "提示",
        "no_result_body": "请先对这一页运行一次自动上色，区域上色工具是在已上色结果上做局部修正的。",
        "no_fill_area": "这个位置没有找到可填充的区域",
        "region_fill_done": "区域上色完成（Ctrl+Z 可撤销）",
        "pick_color_title": "选择颜色",
        "undone": "已撤销",
        "redone": "已重做",
        "confirm_reset_title": "确认重置",
        "confirm_reset_body": "这会清空这一页的上色结果、手动笔画和撤销历史，回到刚导入时的黑白状态。确定吗？",
        "restored_ai": "已恢复到最近一次AI上色结果",
        "no_colorized_page": "请先对当前页面运行自动上色",
        "no_colorized_pages": "还没有已上色的页面",
        "exported_to": "已导出：{path}",
        "select_export_folder": "选择导出文件夹",
        "exported_n_pages": "已导出 {n} 页到 {dir}",
        "splitting_pdf": "正在拆分 PDF 页面...",
        "error_title": "错误",
        "warning_title": "警告",
        "pdf_split_failed": "PDF 拆分失败：{exc}",
        "imported_n_pages": "已导入 {n} 页",
        "cannot_read_image": "无法读取图片：{path}",
        "confirm_delete_title": "确认删除",
        "confirm_delete_body": "删除选中的 {n} 个页面？（不会删除磁盘上的原文件）",
        "deleted_n_pages": "已删除 {n} 个页面",
        "batch_colorizing": "正在批量上色（共 {n} 页）...",
        "colorizing": "正在上色...",
        "page_colorize_failed": "{name} 上色失败：{message}",
        "colorize_done": "上色完成",
    },
    "en": {
        "window_title": "Colortina",
        "device_label": "Device: ",
        "ready": "Ready",

        "import_images": "＋ Import Images",
        "import_pdf": "＋ Import PDF",
        "import_folder": "＋ Import Folder",
        "folder_no_images": "No images found in that folder",
        "pages_label": "Pages (multi-select: Ctrl/Shift, or use buttons)",
        "select_all": "Select All",
        "select_none": "Deselect All",
        "move_up": "↑ Move Up",
        "move_down": "↓ Move Down",
        "delete_pages": "🗑 Delete Selected Pages",

        "fit_view": "Fit to Window",

        "style_quality_group": "Style",
        "skip_colored_checkbox": "Skip already-colored pages",
        "skip_colored_tooltip": "Skip pages that already contain color instead of re-colorizing them (from Manga-Colorization-FJ)",
        "style_label": "Style",
        "quality_label": "Quality",
        "extract_style_btn": "Extract Style from Reference...(multi-select)",
        "style_profile_unset": "No extracted style in use (using the preset above)",
        "character_memory_checkbox": "Character Consistency (distinct hair color per character, book-wide)",
        "character_memory_tooltip": (
            "When on, different characters in the same book will each keep\n"
            "their own hair color, instead of the whole book's hair being one\n"
            "single color. Characters are told apart by lineart tone/grayscale,\n"
            "so two characters with very similar tone may still share a color."),
        "extract_style_dialog_title": "Select color reference image(s) (cover/color pages, multi-select)",
        "extract_style_fail_title": "Extraction Failed",
        "extract_style_fail_body": "Could not read the selected image(s)",
        "extract_style_result": (
            "Using extracted style: {name} (from {n} reference image(s); saturation "
            "{saturation:.2f}, contrast {contrast:.2f}, {temperature} tone; "
            "character consistency seeded {seeded} hair color(s))"),
        "extract_style_status": "Style extracted and saved: {name}.ccstyle",
        "style_profile_active": "Using custom style: {name} (saturation {saturation:.2f}, contrast {contrast:.2f}, {temperature})",

        # Custom style library (new / load / delete)
        "custom_style_group": "Custom Style Library",
        "custom_style_combo_placeholder": "(no saved custom styles yet)",
        "btn_new_style": "＋ New Style (Extract from Reference)...",
        "btn_load_style_file": "📂 Load Style File (.ccstyle)...",
        "btn_apply_saved_style": "Use Selected Style",
        "btn_delete_style": "🗑 Delete Selected Style",
        "btn_clear_style_profile": "✕ Clear Custom Style (Use Preset Above)",
        "new_style_name_title": "New Style",
        "new_style_name_label": "Name this style:",
        "new_style_name_default": "My Style",
        "style_deleted_msg": "Deleted style: {name}",
        "style_loaded_msg": "Loaded style: {name}",
        "style_cleared_msg": "Custom style cleared, using preset: {name}",
        "no_style_selected_msg": "Select a saved style from the list first",
        "confirm_delete_style_title": "Confirm Delete Style",
        "confirm_delete_style_body": "Delete the style \"{name}\"? This can't be undone.",
        "load_style_dialog_title": "Select a Style File (.ccstyle)",
        "load_style_fail_title": "Load Failed",
        "load_style_fail_body": "Could not load this style file: {exc}",

        # Language switch button (top toolbar)
        "lang_button": "🌐 中文 / EN",

        "auto_group": "Auto Colorize",
        "auto_btn": "Auto Colorize (Selected Pages)",

        "edit_group": "Edit (Manual Color Hints)",
        "tool_brush": "Brush",
        "tool_eyedropper": "Eyedropper",
        "tool_bucket": "Region Fill",
        "bucket_hint": ("Region fill: click once to recolor the whole lineart/color-\n"
                       "bounded area (edits the current result directly — instant, no re-run)"),
        "color_label": "Color",
        "brush_size_label": "Brush Size",
        "gap_close_label": "Lineart Gap Closing",
        "gap_close_tooltip": (
            "Higher values auto-'seal' small gaps in the lineart so region fill\n"
            "doesn't leak into neighboring areas; raise this if color leaks past a boundary (0-100)"),
        "fill_mode_label": "Region Fill Mode",
        "fill_mode_shift": "Natural hue shift (recommended, keeps original texture)",
        "fill_mode_shading": "Uniform hue, keep shading",
        "fill_mode_flat": "Fully flat color (fixes badly miscolored/dark areas)",
        "fill_mode_hint": ("If the fill looks too \"flat\" against its surroundings, try the first mode;\n"
                          "if the area itself is a messy mix of black/red, the first mode may not fully fix it — try the third"),
        "undo_last_hint": "Undo Last Stroke",
        "clear_manual_hints": "Clear Manual Strokes",
        "regenerate_btn": "Regenerate (Selected Pages, with Manual Hints)",
        "undo_edit": "↶ Undo (Ctrl+Z)",
        "redo_edit": "↷ Redo (Ctrl+Shift+Z)",

        "view_group": "View / Restore",
        "view_original": "Original (B&W)",
        "view_ai": "AI Result (Last Auto Colorize)",
        "view_edited": "Edited (Current, with Region Fills)",
        "restore_ai": "Restore AI Result (Discard Region Fill Edits)",
        "restore_bw": "Reset This Page (Clear Colors, Back to B&W)",

        "export_group": "Export",
        "export_page": "Export Current Page",
        "export_all": "Export All Pages",

        "menu_language": "语言 / Language",
        "lang_zh": "简体中文",
        "lang_en": "English",
        "lang_restart_title": "Restart Required",
        "lang_restart_body": "Language preference saved — restart the app to apply it.",

        "no_result_title": "Notice",
        "no_result_body": ("Run Auto Colorize on this page first — region fill only "
                          "touches up an existing colorized result."),
        "no_fill_area": "No fillable region found at that position",
        "region_fill_done": "Region fill done (Ctrl+Z to undo)",
        "pick_color_title": "Pick a Color",
        "undone": "Undone",
        "redone": "Redone",
        "confirm_reset_title": "Confirm Reset",
        "confirm_reset_body": ("This clears this page's colorized result, manual strokes, "
                              "and undo history, back to the black-and-white state it was "
                              "imported in. Continue?"),
        "restored_ai": "Restored to the last AI colorize result",
        "no_colorized_page": "Run Auto Colorize on the current page first",
        "no_colorized_pages": "No colorized pages yet",
        "exported_to": "Exported: {path}",
        "select_export_folder": "Select Export Folder",
        "exported_n_pages": "Exported {n} page(s) to {dir}",
        "splitting_pdf": "Splitting PDF pages...",
        "error_title": "Error",
        "warning_title": "Warning",
        "pdf_split_failed": "PDF split failed: {exc}",
        "imported_n_pages": "Imported {n} page(s)",
        "cannot_read_image": "Could not read image: {path}",
        "confirm_delete_title": "Confirm Delete",
        "confirm_delete_body": "Delete the {n} selected page(s)? (original files on disk are kept)",
        "deleted_n_pages": "Deleted {n} page(s)",
        "batch_colorizing": "Batch colorizing ({n} pages)...",
        "colorizing": "Colorizing...",
        "page_colorize_failed": "{name} failed: {message}",
        "colorize_done": "Colorize done",
    },
}

_current_lang = "zh"


def _load_saved_language() -> str:
    try:
        with open(_LANG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        lang = data.get("language", "zh")
        return lang if lang in _STRINGS else "zh"
    except Exception:
        return "zh"


def set_language(lang: str, persist: bool = True) -> None:
    global _current_lang
    if lang not in _STRINGS:
        lang = "zh"
    _current_lang = lang
    if persist:
        try:
            with open(_LANG_FILE, "w", encoding="utf-8") as f:
                json.dump({"language": lang}, f)
        except Exception:
            pass


def get_language() -> str:
    return _current_lang


def tr(key: str) -> str:
    """Translate `key` into the current language. Falls back to the key
    itself (visibly obvious, easy to spot as an untranslated string) if
    missing from both languages."""
    return _STRINGS.get(_current_lang, {}).get(key, _STRINGS["zh"].get(key, key))


# Load the saved preference once at import time.
set_language(_load_saved_language(), persist=False)
