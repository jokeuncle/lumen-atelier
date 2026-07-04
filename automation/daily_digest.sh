#!/usr/bin/env bash
# 每日学习 digest — 双引擎（codex 主力 / claude 容灾）
#
# 流程：
#   1) 组装 prompt：curriculum.md（课程表）+ ledger.md（进度）+ 今天日期
#   2) codex exec 生成 digest；失败则 claude -p 兜底
#   3) 产物落盘 learning/digests/YYYY-MM-DD-{morning|evening}.md（先落盘再推送）
#   4) 经 feishu_push.py 发卡片
#
# 用法：daily_digest.sh morning|evening

set -uo pipefail

MODE="${1:-morning}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LEARN="$ROOT/learning"
OUT_DIR="$LEARN/digests"
TODAY=$(date +%Y-%m-%d)
DOW=$(date +%u)   # 1-7, 7=周日
OUT="$OUT_DIR/$TODAY-$MODE.md"
LOG="$OUT_DIR/$TODAY-$MODE.log"
mkdir -p "$OUT_DIR"

# PATH：launchd 环境干净，需手动补
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

# ── 1. 组装 prompt ────────────────────────────────────────────────
if [ "$MODE" = "morning" ]; then
  if [ "$DOW" = "7" ]; then
    TASK="今天是周日复盘日。生成一张复盘引导卡片：对照 ledger 列出本周完成的课程日与验收状态，引导按 weekly-review.md 模板做复盘，提醒把复盘产物写成博客周记。"
  else
    TASK="生成今天（${TODAY}）的晨间学习卡片。严格按 curriculum.md 末尾的 digest 生成规则：读 ledger 确定进度 → 找到下一个未完成学习日 → 输出五段式（昨天回顾/今天节点/主任务≤45min/数学配菜≤15min/今日验收题）。若 ledger 显示超过 5 天未活动，改为 15 分钟重新入场微任务。"
  fi
else
  TASK="生成今天（${TODAY}）的晚间打卡卡片：温和地问三件事——今天的主任务动了吗？验收题能答上吗？ledger 记了吗（没有证据链接不算数）？如果 ledger 今天没有新条目，提醒明早的 digest 会把今天记为未活动。最后用一句话预告明天的学习节点。保持简短，不说教。"
fi

PROMPT="你是 Lumen Atelier（90 天 AI Infra 学习项目）的学习教练。以下是课程表与台账，据此完成任务。只输出卡片正文 markdown（飞书 lark_md 语法：**粗体**、换行用两个换行），不要输出任何前言或解释。

=== 任务 ===
$TASK

=== curriculum.md ===
$(cat "$LEARN/curriculum.md")

=== ledger.md ===
$(cat "$LEARN/ledger.md")
"

# ── 2. 双引擎生成 ─────────────────────────────────────────────────
ENGINE="codex"
if ! printf '%s' "$PROMPT" | timeout 300 codex exec --skip-git-repo-check - > "$OUT" 2>"$LOG" || [ ! -s "$OUT" ]; then
  ENGINE="claude"
  if ! printf '%s' "$PROMPT" | timeout 300 claude -p --model sonnet > "$OUT" 2>>"$LOG" || [ ! -s "$OUT" ]; then
    # 双引擎全挂：发降级卡片，至少提醒不断链
    ENGINE="fallback"
    {
      echo "**digest 引擎今日不可用**（codex 与 claude 都失败，日志见 $LOG）"
      echo ""
      echo "保底提醒：打开 learning/curriculum.md 找到下一个未完成学习日，手动开始。"
    } > "$OUT"
  fi
fi

# ── 3. 推送 ──────────────────────────────────────────────────────
if [ "$MODE" = "morning" ]; then
  TITLE="🌅 Lumen Atelier · $TODAY 学习任务"
  COLOR="blue"
else
  TITLE="🌙 Lumen Atelier · 今日打卡"
  COLOR="purple"
fi
[ "$DOW" = "7" ] && [ "$MODE" = "morning" ] && TITLE="📋 Lumen Atelier · 周日复盘" && COLOR="green"

python3 "$ROOT/automation/feishu_push.py" --title "$TITLE ($ENGINE)" --color "$COLOR" < "$OUT"
echo "engine=$ENGINE out=$OUT"
