"""Parse combined story files into individual text files."""

import re
from pathlib import Path
import sys


def parse_sin_file(sin_name: str):
    """Parse a sin file and split into 10 individual story files."""
    input_file = Path(f"{sin_name}.txt")
    output_dir = Path("sins") / sin_name.lower()
    
    if not input_file.exists():
        print(f"❌ File not found: {input_file}")
        return
    
    # Read the file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split by newline and filter out empty lines
    stories = [s.strip() for s in content.split('\n') if s.strip()]
    
    # Save each story to a numbered file
    for idx, story in enumerate(stories[:10], 1):
        output_file = output_dir / f"{idx}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(story)
        print(f"  ✓ Saved {idx}.txt")
    
    print(f"✅ Parsed '{sin_name}' into {len(stories[:10])} stories in {output_dir}")


def main():
    """Parse all sin files."""
    sin_file = Path(__file__).parent.parent / "seven-deadly-sins.txt"
    
    if not sin_file.exists():
        print(f"❌ Could not find seven-deadly-sins.txt")
        return
    
    with open(sin_file, 'r') as f:
        sins = [line.strip() for line in f if line.strip()]
    
    print("📖 Parsing sin story files...\n")
    
    for sin in sins:
        sin_lower = sin.lower().replace(" ", "_")
        print(f"Parsing {sin}...")
        parse_sin_file(sin_lower)
        print()
    
    print("🎉 All files parsed!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Parse specific sin file if provided
        sin_name = sys.argv[1]
        parse_sin_file(sin_name)
    else:
        main()
