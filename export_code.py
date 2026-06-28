#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
اسکریپت استخراج تمام کدهای Python پروژه در یک فایل متنی.
فایل‌های .py در تمام زیرپوشه‌ها به جز پوشه‌های غیرضروری اسکن می‌شوند.
نتیجه در فایل project_code_export.txt ذخیره می‌شود.
"""

import os
import sys
from pathlib import Path

# پوشه‌هایی که نباید اسکن شوند
EXCLUDE_DIRS = {
    '__pycache__', '.venv', 'venv', 'env', 'cache', 'results', 'test_cache',
    'node_modules', '.git', '.idea', '.vscode', '__pypackages__', 'dist',
    'build', 'logs', 'temp', 'tmp', 'data', 'tests'  # tests را هم می‌توانید حذف کنید
}

# پسوندهای مجاز
ALLOWED_EXTENSIONS = {'.py'}

def should_exclude_dir(dir_path: Path) -> bool:
    """بررسی می‌کند که آیا یک پوشه باید از اسکن حذف شود."""
    dir_name = dir_path.name
    if dir_name in EXCLUDE_DIRS:
        return True
    # پوشه‌هایی که با '.' شروع می‌شوند (مثل .git) را نیز حذف می‌کنیم
    if dir_name.startswith('.'):
        return True
    return False

def collect_py_files(root_dir: Path):
    """همه فایل‌های .py را در زیرپوشه‌های مجاز جمع‌آوری می‌کند."""
    py_files = []
    for root, dirs, files in os.walk(root_dir):
        # حذف پوشه‌های نامطلوب از لیست dirs تا وارد آن‌ها نشویم
        dirs[:] = [d for d in dirs if not should_exclude_dir(Path(root) / d)]
        for file in files:
            if Path(file).suffix in ALLOWED_EXTENSIONS:
                full_path = Path(root) / file
                py_files.append(full_path)
    return sorted(py_files)  # مرتب‌سازی بر اساس مسیر برای خوانایی بهتر

def export_to_text(py_files: list, output_file: str = "project_code_export.txt"):
    """فایل‌های پایتون را در یک فایل متنی با هدر و جداکننده می‌نویسد."""
    with open(output_file, 'w', encoding='utf-8') as out:
        out.write("=" * 80 + "\n")
        out.write(f"EXPORTED PYTHON CODE FILES\n")
        out.write(f"Total files: {len(py_files)}\n")
        out.write("=" * 80 + "\n\n")

        for py_file in py_files:
            try:
                rel_path = py_file.relative_to(Path.cwd())
            except ValueError:
                rel_path = py_file  # fallback

            out.write(f"File: {rel_path}\n")
            out.write("-" * 80 + "\n")
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                out.write(content)
            except Exception as e:
                out.write(f"!! ERROR reading file: {e}\n")
            out.write("\n" + "=" * 80 + "\n\n")

    print(f"✅ Export completed. Output saved to: {output_file}")

def main():
    root = Path.cwd()
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
        if not root.is_dir():
            print(f"❌ Error: '{root}' is not a valid directory.")
            sys.exit(1)

    print(f"📁 Scanning directory: {root}")
    py_files = collect_py_files(root)
    print(f"🔍 Found {len(py_files)} .py files.")

    if not py_files:
        print("⚠️ No Python files found. Nothing to export.")
        return

    output_file = "project_code_export.txt"
    export_to_text(py_files, output_file)

if __name__ == "__main__":
    main()