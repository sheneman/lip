#!/usr/bin/bash
cat scores.csv | cut -f6 -d"," | awk '{ total += $1; count++ } END { print total/count }'
cat scores.csv | cut -f7 -d"," | awk '{ total += $1; count++ } END { print total/count }'
