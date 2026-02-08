#!/bin/bash
# Fix Makefile tabs - converts spaces to tabs for recipe lines

# Check if Makefile exists
if [ ! -f Makefile ]; then
    echo "Error: Makefile not found"
    exit 1
fi

# Create backup
cp Makefile Makefile.backup

# Use sed to fix common tab issues
# This is a simple fix - for complex cases, manual editing may be needed
sed -i 's/^    /\t/' Makefile 2>/dev/null || sed -i 's/^    /\t/' Makefile

echo "Makefile fixed (backup saved as Makefile.backup)"
echo "Try: make dom0"
