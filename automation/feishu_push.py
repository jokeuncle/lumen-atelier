#!/usr/bin/env python3
"""飞书卡片推送 — 学习保障系统的送信人。

用法：
    python3 feishu_push.py --title "标题" --color blue < body.md
    echo "**内容**" | python3 feishu_push.py --title "早安"

密钥从 ~/.config/lumen-atelier/feishu.env 读取，不入 git。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

ENV_FILE = os.path.expanduser("~/.config/lumen-atelier/feishu.env")
CHAT_ID = "oc_177407904dd5ea55a222dc1cf7d83723"  # 钱磊测试机器人服务
BASE = "https://open.feishu.cn/open-apis"


def load_env() -> dict[str, str]:
    env = {}
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k] = v
    return env


def post(url: str, payload: dict, token: str | None = None) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def get_token(env: dict[str, str]) -> str:
    r = post(
        f"{BASE}/auth/v3/tenant_access_token/internal",
        {"app_id": env["FEISHU_APP_ID"], "app_secret": env["FEISHU_APP_SECRET"]},
    )
    if r.get("code") != 0:
        sys.exit(f"token 获取失败: {r}")
    return r["tenant_access_token"]


def send_card(title: str, body_md: str, color: str = "blue") -> None:
    env = load_env()
    token = get_token(env)
    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": color,
            "title": {"tag": "plain_text", "content": title},
        },
        "elements": [
            {"tag": "div", "text": {"tag": "lark_md", "content": body_md}},
        ],
    }
    r = post(
        f"{BASE}/im/v1/messages?receive_id_type=chat_id",
        {
            "receive_id": CHAT_ID,
            "msg_type": "interactive",
            "content": json.dumps(card, ensure_ascii=False),
        },
        token,
    )
    if r.get("code") != 0:
        sys.exit(f"发送失败: {r}")
    print(f"✓ 已推送: {title}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--title", required=True)
    p.add_argument("--color", default="blue", choices=["blue", "green", "orange", "red", "purple"])
    args = p.parse_args()
    body = sys.stdin.read().strip()
    if not body:
        sys.exit("stdin 为空，没有可发送的内容")
    # 飞书 lark_md 卡片有长度上限，保守截断
    if len(body) > 4000:
        body = body[:4000] + "\n\n…(截断，完整版见 ledger)"
    send_card(args.title, body, args.color)
