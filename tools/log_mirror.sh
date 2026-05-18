#!/bin/bash
SRC="$1"
DST="$2"
[ -z "$SRC" ] || [ -z "$DST" ] && { echo "usage: $0 SRC DST"; exit 1; }
while true; do
    if [ -f "$SRC" ]; then
        cp "$SRC" "$DST.tmp" && mv "$DST.tmp" "$DST"
    fi
    sleep 30
done
