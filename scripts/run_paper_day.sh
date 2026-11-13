#!/usr/bin/env bash
# One trading day of the paper forward test. Schedule before the open on an always-on machine, e.g.
#   CRON_TZ=America/New_York
#   20 9 * * 1-5  cd /path/to/repo2 && ./scripts/run_paper_day.sh >> live/cron.log 2>&1
# Needs ALPACA_API_KEY / ALPACA_SECRET_KEY (paper keys). Exits quietly on market holidays.
set -u
cd "$(dirname "$0")/.."
day=$(TZ=America/New_York date +%F)
mkdir -p "live/$day"
python3 -u -m scripts.paper_trade >> "live/$day/runner.out" 2>&1
# belt and braces: whatever happened above, make sure OUR SPY stock position is flat
python3 -m scripts.paper_trade --flatten >> "live/$day/runner.out" 2>&1
# SIP bars for the fill comparison are available 15 minutes after the fact
now=$(TZ=America/New_York date +%H%M)
if [ "$now" -lt 1616 ]; then sleep $(( ( $(TZ=America/New_York date -d 16:16 +%s) - $(date +%s) ) )); fi
[ -f "live/$day/summary.json" ] && python3 -m scripts.daily_report "$day" > /dev/null
